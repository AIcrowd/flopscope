"""Tests verifying numpy's __array_ufunc__ and __array_function__ protocols
route numpy calls through flopscope's counted functions when the operands are
FlopscopeArray (or SymmetricTensor).

Includes adversarial coverage for recursion, out= tuples, kwargs passthrough,
mixed operands, unsupported ufunc methods, and identity preservation.

Translated against post-PR-#51 unified SymmetryGroup API.
"""

from __future__ import annotations

import numpy as np
import numpy.ma as ma
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._ndarray import FlopscopeArray
from flopscope.errors import SymmetryLossWarning

# ----- __array_ufunc__: ufunc.__call__ -----

UFUNC_CALL_CASES = [
    ("add", lambda a, b: np.add(a, b), "add"),
    ("multiply", lambda a, b: np.multiply(a, b), "multiply"),
    ("subtract", lambda a, b: np.subtract(a, b), "subtract"),
    ("maximum", lambda a, b: np.maximum(a, b), "maximum"),
]


@pytest.mark.parametrize("name,op,we_name", UFUNC_CALL_CASES)
def test_np_ufunc_call_tracks_flops(name, op, we_name):
    a = fnp.random.randn(8)
    b = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        op(a, b)
    we_func = getattr(fnp, we_name)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        we_func(a, b)
    assert b1.flops_used == b2.flops_used > 0


def test_np_unary_ufunc_call_tracks_flops():
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.sin(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        fnp.sin(a)
    assert b1.flops_used == b2.flops_used > 0


# ----- __array_ufunc__: ufunc.reduce -----


def test_np_add_reduce_tracks_flops():
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.add.reduce(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        fnp.sum(a)
    assert b1.flops_used == b2.flops_used > 0


def test_np_maximum_reduce_tracks_flops():
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.maximum.reduce(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        fnp.max(a)
    assert b1.flops_used == b2.flops_used > 0


def test_np_add_reduce_2d_matches_axis0_semantics():
    """``ufunc.reduce`` defaults to ``axis=0``, whereas ``fnp.sum`` defaults
    to ``axis=None`` (full reduction). The ``__array_ufunc__`` dispatch
    must inject ``axis=0`` so 2D inputs get partial-reduction semantics."""
    a = fnp.random.randn(4, 5)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        r1 = np.add.reduce(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        r2 = fnp.sum(a, axis=0)
    assert r1.shape == r2.shape == (5,)
    assert b1.flops_used == b2.flops_used > 0


# ----- __array_ufunc__: ufunc.accumulate -----


def test_np_add_accumulate_tracks_flops():
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.add.accumulate(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        fnp.cumsum(a)
    assert b1.flops_used == b2.flops_used > 0


def test_np_add_accumulate_2d_matches_axis0_semantics():
    """Same axis-default issue as reduce: ``ufunc.accumulate`` defaults to
    ``axis=0``, ``fnp.cumsum`` defaults to ``axis=None`` (flatten)."""
    a = fnp.random.randn(4, 5)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        r1 = np.add.accumulate(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        r2 = fnp.cumsum(a, axis=0)
    assert r1.shape == r2.shape == (4, 5)
    assert b1.flops_used == b2.flops_used > 0


# ----- Recursion guards (regression-only) -----


def test_dunder_does_not_recurse_after_protocol_enabled():
    """FlopscopeArray.__add__ → me.add → _np.add must NOT re-dispatch through
    __array_ufunc__ → me.add → ∞.

    The strip-before-NumPy invariant in counted wrappers prevents this.
    """
    a = fnp.random.randn(8)
    b = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        c = a + b
    assert bc.flops_used > 0
    assert c.shape == (8,)


def test_np_add_does_not_recurse():
    a = fnp.random.randn(8)
    b = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        np.add(a, b)
    assert bc.flops_used > 0


def test_np_sort_does_not_recurse():
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        np.sort(a)
    assert bc.flops_used > 0


# ----- ufunc kwargs passthrough -----


def test_np_add_out_unwraps_single_output_tuple():
    """NumPy passes out=(out_arr,) to __array_ufunc__; flopscope expects out=arr."""
    a = fnp.random.randn(8)
    b = fnp.random.randn(8)
    out = fnp.empty_like(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        returned = np.add(a, b, out=out)
    assert returned is out
    assert bc.flops_used > 0


def test_np_add_out_refuses_when_out_symmetry_would_be_destroyed():
    """``np.add(A_sym, B_unsymmetric, out=A_sym)`` would write
    unsymmetric bytes into a buffer whose metadata still claims
    symmetry. Same correctness issue as in-place dunders, just via an
    explicit ``out=``. The wrapper's symmetry validation must refuse
    before the NumPy call writes any bytes.

    Post-PR-#51, this is enforced by ``_prepare_symmetric_out`` in
    ``_pointwise.py``: if the out's symmetry doesn't match the result's,
    it raises (via ``SymmetryError`` or ``ValueError``).
    """
    A = flops.symmetrize(
        fnp.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    B = fnp.random.randn(4, 4)  # plain FlopscopeArray, no symmetry
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises((ValueError, flops.errors.SymmetryError)):
            np.add(A, B, out=A)


def test_np_add_out_allows_matching_symmetric_out():
    """Positive case: when the operation's output symmetry matches the
    declared symmetry on ``out=``, the call goes through cleanly.
    ``A + 1.0`` preserves every group exactly (binary-with-scalar), so
    writing into a SymmetricTensor ``out`` with the same axes is safe."""
    A = flops.symmetrize(
        fnp.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    out = flops.symmetrize(
        fnp.zeros((4, 4)),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        ret = np.add(A, 1.0, out=out)
    assert ret is out
    assert isinstance(out, flops.SymmetricTensor)
    assert bc.flops_used > 0


def test_np_transpose_of_whest_returns_whest():
    """Post-Stage-4: np.transpose dispatches via __array_function__ to
    me.transpose, which works on FlopscopeArray (zero-FLOP shape op)."""
    a = fnp.random.randn(2, 3)
    with flops.BudgetContext(flop_budget=int(1e9)):
        r = np.transpose(a)
    assert isinstance(r, fnp.ndarray)
    assert r.shape == (3, 2)


def test_np_add_where_kwarg_tracks():
    a = fnp.random.randn(8)
    b = fnp.random.randn(8)
    mask = np.array([True, False] * 4)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        np.add(a, b, where=mask)
    assert bc.flops_used > 0


def test_np_sin_where_kwarg_tracks():
    """Unary ufunc with ``where=`` mask. Mirrors the binary-``where=``
    test above, exercising ``_counted_unary``'s ``where`` strip path."""
    a = fnp.random.randn(8)
    mask = np.array([True, False] * 4)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        np.sin(a, where=mask)
    assert bc.flops_used > 0


def test_np_add_dtype_kwarg_tracks():
    a = fnp.random.randn(8).astype(np.float32)
    b = fnp.random.randn(8).astype(np.float32)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        r = np.add(a, b, dtype=np.float64)
    assert bc.flops_used > 0
    assert r.dtype == np.float64


# ----- Mixed operands: numpy on left, flopscope on right -----


def test_mixed_numpy_left_operand_dispatches():
    """np.ndarray + FlopscopeArray must still dispatch through flopscope tracking
    (NEP 13: NumPy defers to subclasses' __array_ufunc__)."""
    a = np.ones(8)
    b = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        c = a + b
    assert bc.flops_used > 0
    assert isinstance(c, fnp.ndarray)


def test_mixed_python_scalar_left_dispatches():
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        c = 2.0 + a
    assert bc.flops_used > 0


# ----- Immutability: in-place ops must raise -----


def test_setitem_raises_immutable():
    import flopscope.numpy as fnp

    a = fnp.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="immutable"):
        a[0] = 9.0


def test_iadd_raises_immutable():
    import flopscope.numpy as fnp

    a = fnp.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="immutable"):
        a += fnp.array([1.0, 1.0, 1.0])


def test_inplace_sort_raises_immutable():
    import flopscope.numpy as fnp

    a = fnp.array([3.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="immutable"):
        a.sort()


def test_fill_raises_immutable():
    # fill is a C-level mutator that bypasses __setitem__; it must be overridden
    # explicitly or native arrays stay mutable (and diverge from the client).
    import flopscope.numpy as fnp

    a = fnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="immutable"):
        a.fill(0.0)
    assert a.tolist() == [1.0, 2.0, 3.0]  # unchanged


def test_put_raises_immutable():
    import flopscope.numpy as fnp

    a = fnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="immutable"):
        a.put([0], [9.0])
    assert a.tolist() == [1.0, 2.0, 3.0]


def test_resize_raises_immutable():
    import flopscope.numpy as fnp

    a = fnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="immutable"):
        a.resize((1, 3))


# ----- ufunc.outer / .reduceat / .at / generic .reduce / .accumulate -----


def test_np_add_outer_routes_through_array_ufunc():
    """``np.add.outer(FlopscopeArray, FlopscopeArray)`` produces a tracked
    FlopscopeArray of shape ``a.shape + b.shape`` with FLOPs deducted."""
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        a = fnp.array([1.0, 2.0, 3.0])
        b = fnp.array([10.0, 20.0])
        result = np.add.outer(a, b)
    assert isinstance(result, FlopscopeArray)
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(np.asarray(result), [[11, 21], [12, 22], [13, 23]])
    assert bc.flops_used > 0


def test_np_add_outer_preserves_direct_product_symmetry():
    """``np.add.outer(A, B)`` for ``A``, ``B`` SymmetricTensors produces
    a SymmetricTensor whose symmetry is the direct product of the
    inputs', with ``B``'s axes lifted past ``A``'s ndim."""
    with flops.BudgetContext(flop_budget=int(1e10)):
        sym = flops.SymmetryGroup.symmetric(axes=(0, 1))
        A = flops.symmetrize(fnp.array([[1.0, 2.0], [2.0, 3.0]]), symmetry=sym)
        B = flops.symmetrize(fnp.array([[5.0, 6.0], [6.0, 7.0]]), symmetry=sym)
        result = np.add.outer(A, B)
    assert isinstance(result, flops.SymmetricTensor)
    assert result.shape == (2, 2, 2, 2)
    # Output symmetry has both S2 generators (one on (0,1), one on (2,3)).
    assert result.symmetry is not None
    assert set(result.symmetry.axes) == {0, 1, 2, 3}  # pyright: ignore[reportArgumentType]


def test_np_add_outer_symmetric_cost_lower_than_dense():
    """Symmetric outer charges fewer FLOPs than dense outer (placeholder
    cost = dense × unique_output / dense_output ratio)."""
    n = 10
    sym = flops.SymmetryGroup.symmetric(axes=(0, 1))
    with flops.BudgetContext(flop_budget=int(1e10)) as dense_bc:
        a = fnp.random.randn(n, n)
        b = fnp.random.randn(n, n)
        _ = np.add.outer(a, b)
    with flops.BudgetContext(flop_budget=int(1e10)) as sym_bc:
        a = flops.symmetrize(fnp.random.randn(n, n), symmetry=sym)
        b = flops.symmetrize(fnp.random.randn(n, n), symmetry=sym)
        _ = np.add.outer(a, b)
    # Symmetric is strictly cheaper than dense (input setup cost is the
    # same for both; only the outer-op portion shrinks).
    assert sym_bc.flops_used < dense_bc.flops_used


def test_np_add_outer_warns_and_bails_on_oversized_symmetry():
    """High-degree symmetry groups (e.g. ``S_n`` from ``np.ones((1,)*n)``
    for large ``n``) would require Burnside enumeration on ``n!``
    elements, which is infeasible. The wrapper bails to dense cost and
    emits :class:`CostFallbackWarning` once per ``(op, |G|)``."""
    import warnings as _warnings

    from flopscope.errors import CostFallbackWarning

    deep = fnp.ones((1,) * 33)  # S_33 auto-inferred symmetry
    assert isinstance(deep, flops.SymmetricTensor)
    with flops.BudgetContext(flop_budget=int(1e10)):
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            # The op itself raises ValueError because numpy refuses
            # ndim > 32 — but our wrapper still emits the warning before
            # numpy raises.
            with pytest.raises(ValueError):
                np.add.outer(deep, deep)
    cost_warnings = [w for w in caught if issubclass(w.category, CostFallbackWarning)]
    assert len(cost_warnings) == 1, [str(w.message) for w in caught]
    assert "order " in str(cost_warnings[0].message)
    assert "budget " in str(cost_warnings[0].message)


def test_cost_fallback_warning_suppressed_by_configure():
    """``flops.configure(symmetry_warnings=False)`` silences
    :class:`CostFallbackWarning` (shares the flag with
    :class:`SymmetryLossWarning` since both are symmetry diagnostics)."""
    import warnings as _warnings

    from flopscope.errors import CostFallbackWarning

    # Use a fresh degree (not 33, which the previous test already cached)
    # so we hit a cold cache key and the warning would otherwise fire.
    # rank-20 → output ndim=40 which fits inside numpy 2.x's 64-axis
    # limit, so the op itself succeeds.
    deep = fnp.ones((1,) * 20)
    flops.configure(symmetry_warnings=False)
    try:
        with flops.BudgetContext(flop_budget=int(1e10)):
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                np.add.outer(deep, deep)
    finally:
        flops.configure(symmetry_warnings=True)
    cost_warnings = [w for w in caught if issubclass(w.category, CostFallbackWarning)]
    assert cost_warnings == [], [str(w.message) for w in cost_warnings]


def test_np_subtract_reduce_uses_generic_path():
    """Non-table reduces (``subtract``, ``true_divide``, …) route through
    the generic ``_counted_ufunc_reduce_generic`` fallback."""
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        a = fnp.array([10.0, 3.0, 2.0, 1.0])
        result = np.subtract.reduce(a)
    assert float(result) == 4.0  # 10 - 3 - 2 - 1
    assert bc.flops_used > 0


def test_np_subtract_accumulate_uses_generic_path():
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        a = fnp.array([10.0, 3.0, 2.0, 1.0])
        result = np.subtract.accumulate(a)
    np.testing.assert_array_equal(np.asarray(result), [10.0, 7.0, 5.0, 4.0])
    assert bc.flops_used > 0


def test_np_add_reduceat_routes_through_array_ufunc():
    """``ufunc.reduceat`` segments are tracked; output symmetry is
    dropped (segment boundaries don't respect axis-permutation
    invariance)."""
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        a = fnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = np.add.reduceat(a, [0, 3])
    np.testing.assert_array_equal(np.asarray(result), [6.0, 15.0])
    assert bc.flops_used > 0


def test_np_add_at_on_plain_whest_array_mutates_in_place():
    """``np.add.at(FlopscopeArray, indices, values)`` mutates the underlying
    array — repeated indices accumulate (unlike ``a[indices] +=``)."""
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        a = fnp.array([0.0, 0.0, 0.0])
        result = np.add.at(a, [0, 0, 1], [1.0, 2.0, 3.0])
    assert result is None  # ufunc.at returns None
    np.testing.assert_array_equal(np.asarray(a), [3.0, 3.0, 0.0])
    assert bc.flops_used > 0


def test_np_add_at_on_symmetric_tensor_refuses():
    """``ufunc.at`` on a SymmetricTensor would corrupt the tagged
    symmetry; flopscope refuses with a directive to downgrade first."""
    with flops.BudgetContext(flop_budget=int(1e10)):
        sym = flops.SymmetryGroup.symmetric(axes=(0, 1))
        S = flops.symmetrize(fnp.array([[1.0, 2.0], [2.0, 3.0]]), symmetry=sym)
        with pytest.raises(ValueError, match="symmetry"):
            np.add.at(S, ([0],), 1.0)


# ----- ufunc.outer / .reduce / .accumulate / .reduceat / .at must forward a
# foreign operand's ORIGINAL object to numpy, not a subclass-stripped view.
#
# The billing hardening in these wrappers reads a/b/out=/values off a
# ``_np.asarray(...)`` view so a lying ``.dtype``/``.shape`` property can't
# under-report the bill -- but that view must stay LOCAL to the billing
# math. Forwarding it (instead of the caller's real object) to the actual
# numpy call silently drops a legitimate foreign ndarray subclass's
# semantics -- a mask, a unit system, anything hanging off
# ``__array_ufunc__``/``__array_wrap__`` -- even though only the bill
# needed the honest read.


class _TrackingArray(np.ndarray):
    """ndarray subclass with a genuine ``__array_ufunc__`` override.

    Records every ufunc dispatch it wins, then delegates to the real
    computation so the call still succeeds normally. Its own
    ``__array_ufunc__`` can only fire a second time (from inside
    flopscope's wrapper, which calls the raw ufunc directly) if flopscope
    handed numpy this exact object -- a stripped, subclass-free
    ``np.ndarray`` view could never trigger it.
    """

    calls: list[tuple[str, str]] = []

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _TrackingArray.calls.append((ufunc.__name__, method))

        def _strip(x):
            if isinstance(x, _TrackingArray):
                return x.view(np.ndarray)
            if isinstance(x, tuple):
                return tuple(_strip(e) for e in x)
            return x

        inputs = tuple(_strip(i) for i in inputs)
        if kwargs.get("out") is not None:
            kwargs["out"] = _strip(kwargs["out"])
        return getattr(ufunc, method)(*inputs, **kwargs)


def _tracking(data):
    return np.asarray(data, dtype=np.float64).view(_TrackingArray)


def test_np_multiply_outer_preserves_masked_array_operand():
    """Regression pin: ``outer``'s ``b`` operand used to be reassigned to
    a ``_np.asarray``-stripped view and THAT stripped view (not the
    caller's ``b``) was what reached ``ufunc.outer`` -- so a
    ``np.ma.MaskedArray`` silently lost its mask and came back as a plain
    ``FlopscopeArray`` with the masked element computed as ordinary data.
    """
    with flops.BudgetContext(flop_budget=int(1e10)):
        a = fnp.array([1.0, 2.0, 3.0])
        masked = ma.MaskedArray([10.0, 20.0], mask=[False, True])
        result = np.multiply.outer(a, masked)
    assert isinstance(result, ma.MaskedArray)
    np.testing.assert_array_equal(result.mask, [[False, True]] * 3)
    np.testing.assert_array_equal(
        np.asarray(result), [[10.0, 20.0], [20.0, 40.0], [30.0, 60.0]]
    )


def test_np_multiply_outer_a_masked_b_flopscope_preserves_mask():
    """Same guarantee with the masked operand in the ``a`` slot instead --
    ``outer`` strips both operands, so both directions must be covered."""
    with flops.BudgetContext(flop_budget=int(1e10)):
        masked = ma.MaskedArray([1.0, 2.0], mask=[True, False])
        b = fnp.array([10.0, 20.0, 30.0])
        result = np.multiply.outer(masked, b)
    assert isinstance(result, ma.MaskedArray)
    np.testing.assert_array_equal(result.mask, [[True] * 3, [False] * 3])


class _StatefulArrayLike:
    """A pure ``__array__`` duck type -- NOT an ndarray subclass -- whose
    ``__array__`` returns a DIFFERENT (larger) array on its second call
    than its first.

    Regression pin: ``outer``'s billing view used to be built from one
    ``_np.asarray(x)`` call while the caller's original, unresolved ``x``
    was separately forwarded to the real ``ufunc.outer`` -- which performs
    its OWN ``np.asarray`` conversion internally. For an operand backed by
    a live view (e.g. an already-materialized ndarray) those two
    conversions agree. For a stateful duck type like this one they don't:
    the small first-call array got billed while numpy actually computed
    over the large second-call array, undercharging the real work.
    """

    def __init__(self):
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        self.calls += 1
        size = 2 if self.calls == 1 else 3000
        return np.ones(size, dtype=np.float64)


def test_np_multiply_outer_stateful_array_like_operand_resolved_once():
    """``__array__`` must be invoked EXACTLY ONCE for a non-ndarray operand,
    and the bill must match the honest cost of the shape numpy actually
    computes over -- not a smaller shape returned by an earlier, discarded
    call.
    """
    # Operands are created OUTSIDE the measured budget contexts below so
    # that array-construction cost doesn't leak into the comparison -- only
    # the ``outer`` call itself is being measured.
    a = fnp.array([1.0, 1.0, 1.0, 1.0])
    grow = _StatefulArrayLike()
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        result = np.multiply.outer(a, grow)

    assert grow.calls == 1
    assert result.shape == (4, 2)

    a_honest = fnp.array([1.0, 1.0, 1.0, 1.0])
    b_honest = fnp.array([1.0, 1.0])
    with flops.BudgetContext(flop_budget=int(1e10)) as honest_bc:
        np.multiply.outer(a_honest, b_honest)

    assert bc.flops_used == honest_bc.flops_used


class _ProtocolDuck:
    """A non-ndarray duck type implementing BOTH a stateful ``__array__``
    AND ``__array_ufunc__`` (NEP 13), the latter unconditionally returning
    ``NotImplemented``.

    Regression pin for the three-way refinement to
    :func:`_resolve_ufunc_data_operand` (and its ``_resolve_at_operand`` /
    reduce / accumulate / reduceat siblings): the prior two-way rule
    materialized ANY non-ndarray operand via ``__array__`` before
    forwarding it -- including one whose type implements
    ``__array_ufunc__`` -- which silently bypassed numpy's own dispatch
    protocol. flopscope would then compute from the materialized array
    where plain numpy hands the operation to ``__array_ufunc__`` instead
    (and, for a duck like this one whose override always declines, raises
    ``TypeError``). The fix forwards the ORIGINAL object so that protocol
    runs -- and, critically, does so WITHOUT re-invoking ``__array__`` a
    second time: billing reads shape/dtype via a single ``_np.asarray``
    view, and the real (forwarded) call never touches ``__array__`` at all
    because numpy dispatches it to ``__array_ufunc__`` instead.
    """

    def __init__(self):
        self.array_calls = 0
        self.ufunc_calls = 0

    def __array__(self, dtype=None, copy=None):
        self.array_calls += 1
        return np.ones(3, dtype=np.float64)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.ufunc_calls += 1
        return NotImplemented


@pytest.mark.parametrize(
    "op_name,call",
    [
        ("outer.b", lambda a, duck: np.add.outer(a, duck)),
        ("outer.a", lambda a, duck: np.add.outer(duck, a)),
        ("reduce.a", lambda a, duck: np.subtract.reduce(duck, out=a, axis=None)),
        ("accumulate.a", lambda a, duck: np.subtract.accumulate(duck, out=a)),
        ("reduceat.a", lambda a, duck: np.add.reduceat(duck, [0, 1], out=a, axis=0)),
        ("at.values", lambda a, duck: np.add.at(a, [0, 1, 2], duck)),
    ],
)
def test_duck_array_ufunc_protocol_fires_matching_raw_numpy(op_name, call):
    """A non-ndarray operand whose TYPE implements ``__array_ufunc__`` must
    have that protocol actually invoked by the real numpy call below --
    exactly as it would be with flopscope out of the loop entirely --
    rather than being silently materialized through ``__array__`` and
    computed from that instead. ``_ProtocolDuck.__array_ufunc__`` always
    declines (returns ``NotImplemented``), so a correctly-dispatched call
    raises ``TypeError``, matching what raw numpy does for this duck.
    """
    duck = _ProtocolDuck()
    a = fnp.array([0.0, 0.0, 0.0])
    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.raises(TypeError):
            call(a, duck)
    assert duck.ufunc_calls >= 1, (
        f"{op_name}: Duck's __array_ufunc__ never fired -- the operand was "
        "materialized through __array__ and numpy's dispatch protocol was "
        "bypassed"
    )


@pytest.mark.parametrize(
    "op_name,make_other,call",
    [
        (
            "outer.b",
            lambda: _tracking([10.0, 20.0]),
            lambda a, o: np.multiply.outer(a, o),
        ),
        (
            "reduce.out",
            lambda: _tracking(0.0),
            lambda a, o: np.subtract.reduce(a, out=o),
        ),
        (
            "accumulate.out",
            lambda: _tracking([0.0, 0.0, 0.0, 0.0]),
            lambda a, o: np.subtract.accumulate(a, out=o),
        ),
        (
            "reduceat.out",
            lambda: _tracking([0.0, 0.0]),
            lambda a, o: np.add.reduceat(a, [0, 2], out=o),
        ),
    ],
)
def test_ufunc_method_forwards_original_foreign_operand(op_name, make_other, call):
    """A foreign ndarray subclass passed as the non-flopscope operand
    (``outer``'s ``b``, or ``out=`` for the reduce/accumulate/reduceat
    family) must reach numpy as ITSELF. Proven by its own
    ``__array_ufunc__`` firing a second time when flopscope's wrapper
    calls the raw ufunc -- a stripped, subclass-free view could never
    trigger it, which is exactly the bug this pins against.
    """
    _TrackingArray.calls.clear()
    with flops.BudgetContext(flop_budget=int(1e10)):
        a = fnp.array([4.0, 3.0, 2.0, 1.0])
        other = make_other()
        call(a, other)
    assert _TrackingArray.calls, (
        f"{op_name}: the foreign operand's own __array_ufunc__ never fired -- "
        "it reached numpy as a stripped plain ndarray instead of itself"
    )


def test_np_add_at_forwards_original_foreign_values_operand():
    """Same guarantee as above, for ``ufunc.at``'s ``values`` operand."""
    _TrackingArray.calls.clear()
    with flops.BudgetContext(flop_budget=int(1e10)):
        dst = fnp.array([0.0, 0.0, 0.0])
        np.add.at(dst, [0, 0, 1], _tracking([1.0, 2.0, 3.0]))
    assert _TrackingArray.calls, (
        "ufunc.at: values' own __array_ufunc__ never fired -- it reached "
        "numpy as a stripped plain ndarray instead of itself"
    )
    np.testing.assert_array_equal(np.asarray(dst), [3.0, 3.0, 0.0])


def test_np_add_at_preserves_masked_array_values_operand():
    """``ufunc.at``'s ``values`` operand keeps its mask-array identity
    through the call, mirroring the ``outer`` guarantee above."""
    with flops.BudgetContext(flop_budget=int(1e10)):
        dst = fnp.array([0.0, 0.0, 0.0])
        masked_values = ma.MaskedArray([1.0, 2.0, 3.0], mask=[False, True, False])
        np.add.at(dst, [0, 0, 1], masked_values)
    np.testing.assert_array_equal(np.asarray(dst), [3.0, 3.0, 0.0])


# ----- stale billing snapshot: ``a``'s shape/size/dtype must be read for
# billing AFTER every participant-controlled protocol resolution (axis=,
# dtype=, the index/values operands) has run, not before -- because that
# resolution can run arbitrary code, and an OWNING ndarray subclass's
# ``resize(n, refcheck=False)`` lets it grow ``a`` in place. See
# ``tests/test_reduceat_cost.py`` for the ``reduceat`` pin (including the
# original confirmed reproduction); the cases here cover the other four
# generic ufunc-method paths that share the same ordering discipline.


class _OwningFloat64(np.ndarray):
    """A plain ndarray subclass that OWNS its data (is not a view of
    something else) -- the same shape ``np.empty``/``np.zeros`` builds.
    That ownership is what makes ``resize(n, refcheck=False)`` a legitimate
    in-place grow rather than a ``ValueError``.
    """

    def __new__(cls, n, fill=1.0):
        obj = super().__new__(cls, (n,), dtype=np.float64)
        obj[...] = fill
        return obj


def test_np_subtract_reduce_resizing_axis_bills_the_post_resize_array():
    """``axis=``'s ``__index__`` grows ``a`` in place before returning a
    valid axis. ``out=`` is a flopscope array so dispatch reaches
    flopscope's wrapper even though ``a`` itself is a foreign subclass."""
    n = 1_000_000

    class _ResizingAxis:
        def __index__(self):
            a.resize(n, refcheck=False)
            return 0

    a = _OwningFloat64(4)
    out = fnp.array(0.0)
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        np.subtract.reduce(a, axis=_ResizingAxis(), out=out)
    assert a.size == n, "sanity: the resize actually ran"

    with flops.BudgetContext(flop_budget=int(1e10)) as honest_bc:
        np.subtract.reduce(fnp.asarray(np.full(n, 1.0)), axis=0)
    assert bc.flops_used == honest_bc.flops_used


def test_np_subtract_accumulate_resizing_axis_bills_the_post_resize_array():
    """Same defect as the ``reduce`` case above, for ``accumulate``."""
    n = 1_000_000

    class _ResizingAxis:
        def __index__(self):
            a.resize(n, refcheck=False)
            return 0

    a = _OwningFloat64(4)
    out = fnp.zeros(n)
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        np.subtract.accumulate(a, axis=_ResizingAxis(), out=out)
    assert a.size == n, "sanity: the resize actually ran"

    with flops.BudgetContext(flop_budget=int(1e10)) as honest_bc:
        np.subtract.accumulate(fnp.asarray(np.full(n, 1.0)), axis=0)
    assert bc.flops_used == honest_bc.flops_used


def test_np_multiply_outer_resizing_dtype_bills_the_post_resize_array():
    """A ``dtype=`` object whose ``.dtype`` PROPERTY -- which ``np.dtype()``
    honours -- resizes ``a`` in place as a side effect of reporting a
    valid dtype."""
    n = 1_000_000
    a = _OwningFloat64(4)

    class _ResizingDtype:
        @property
        def dtype(self):
            a.resize(n, refcheck=False)
            return np.dtype(np.float64)

    # ``b`` and the honest comparison's operands are built OUTSIDE the
    # measured contexts below -- see
    # ``test_np_multiply_outer_stateful_array_like_operand_resolved_once``
    # above for why: array construction itself is billable, and folding it
    # into either measured window would leak into the comparison.
    b = fnp.array([1.0, 1.0])
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        np.multiply.outer(a, b, dtype=_ResizingDtype())
    assert a.size == n, "sanity: the resize actually ran"

    honest_a = fnp.asarray(np.full(n, 1.0))
    honest_b = fnp.array([1.0, 1.0])
    with flops.BudgetContext(flop_budget=int(1e10)) as honest_bc:
        np.multiply.outer(honest_a, honest_b, dtype=np.float64)
    assert bc.flops_used == honest_bc.flops_used


# ``ufunc.at`` has no ``out=`` slot, so (unlike outer/reduce/accumulate/
# reduceat) numpy dispatches to flopscope's wrapper only when ``a`` ITSELF
# is flopscope-aware -- there is no other operand to hang dispatch off of.
# flopscope arrays are deliberately immutable (``resize`` raises
# ``ValueError`` on one -- see ``flopscope-immutability-intent``), so a
# foreign, resize-capable ``a`` can never reach ``_counted_ufunc_at``
# through the public ``np.add.at(...)`` surface at all. The two cases below
# call the wrapper directly instead -- the same pattern
# ``test_reduceat_lying_dtype_a_operand_bills_the_real_dtype`` in
# ``tests/test_reduceat_cost.py`` uses for the same underlying reason (a
# foreign, non-flopscope ``a`` with nothing to trigger dispatch) -- so the
# ordering fix is still exercised even though this specific vector is not
# independently reachable from outside the package.


def test_np_add_at_resizing_index_bills_the_post_resize_array():
    """``ufunc.at``'s index entry can itself expose ``__index__`` and, via
    that, resize ``a`` in place before the real write runs."""
    from flopscope._pointwise import _counted_ufunc_at

    n = 1_000_000
    a = _OwningFloat64(4)

    class _ResizingIndex:
        def __index__(self):
            a.resize(n, refcheck=False)
            return 0

    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        _counted_ufunc_at(np.add, a, _ResizingIndex(), 1.0)
    assert a.size == n, "sanity: the resize actually ran"

    honest_a = np.full(n, 1.0)
    with flops.BudgetContext(flop_budget=int(1e10)) as honest_bc:
        _counted_ufunc_at(np.add, honest_a, 0, 1.0)
    assert bc.flops_used == honest_bc.flops_used


def test_np_add_at_resizing_values_bills_the_post_resize_array():
    """Same defect, via the ``values`` operand's ``__array__`` instead of
    the index: resizes ``a`` in place as a side effect of materializing."""
    from flopscope._pointwise import _counted_ufunc_at

    n = 1_000_000
    a = _OwningFloat64(4)

    class _ResizingValues:
        def __array__(self, dtype=None, copy=None):
            a.resize(n, refcheck=False)
            return np.ones(n)

    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        _counted_ufunc_at(np.add, a, slice(None), _ResizingValues())
    assert a.size == n, "sanity: the resize actually ran"

    honest_a = np.full(n, 1.0)
    with flops.BudgetContext(flop_budget=int(1e10)) as honest_bc:
        _counted_ufunc_at(np.add, honest_a, slice(None), np.ones(n))
    assert bc.flops_used == honest_bc.flops_used


# ----- Multi-output ufuncs route through __array_ufunc__ -----


def test_np_divmod_routes_to_we_divmod():
    """``np.divmod(FlopscopeArray, FlopscopeArray)`` dispatches to ``fnp.divmod``
    via ``__array_ufunc__``, returns a tuple of FlopscopeArrays, and
    deducts FLOPs."""
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        a = fnp.array([10.0, 20.0, 30.0])
        b = fnp.array([3.0, 4.0, 5.0])
        q, r = np.divmod(a, b)
    assert isinstance(q, FlopscopeArray)
    assert isinstance(r, FlopscopeArray)
    np.testing.assert_array_equal(np.asarray(q), [3.0, 5.0, 6.0])
    np.testing.assert_array_equal(np.asarray(r), [1.0, 0.0, 0.0])
    assert bc.flops_used > 0


def test_np_modf_routes_to_we_modf():
    """``np.modf`` (nout=2) routes through ``__array_ufunc__`` and
    returns ``(frac, integer)`` as FlopscopeArrays with FLOPs deducted."""
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        a = fnp.array([1.5, 2.7, 3.0])
        frac, integer = np.modf(a)
    assert isinstance(frac, FlopscopeArray)
    assert isinstance(integer, FlopscopeArray)
    np.testing.assert_allclose(np.asarray(integer), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(frac), [0.5, 0.7, 0.0], atol=1e-12)
    assert bc.flops_used > 0


def test_np_frexp_routes_to_we_frexp():
    """``np.frexp`` returns ``(mantissa, exponent)`` with the exponent
    in integer dtype; both reach the caller as FlopscopeArrays."""
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        a = fnp.array([1.5, 2.7, 3.0])
        mantissa, exponent = np.frexp(a)
    assert isinstance(mantissa, FlopscopeArray)
    assert isinstance(exponent, FlopscopeArray)
    assert np.issubdtype(exponent.dtype, np.integer)
    assert bc.flops_used > 0


def test_np_divmod_with_out_tuple_preserves_identity():
    """``np.divmod(a, b, out=(q, r))`` writes through both buffers and
    returns the same Python objects (per-slot identity contract)."""
    with flops.BudgetContext(flop_budget=int(1e10)):
        a = fnp.array([10.0, 20.0])
        b = fnp.array([3.0, 4.0])
        q = fnp.zeros(2)
        r = fnp.zeros(2)
        result = np.divmod(a, b, out=(q, r))
    assert result[0] is q
    assert result[1] is r
    np.testing.assert_array_equal(np.asarray(q), [3.0, 5.0])
    np.testing.assert_array_equal(np.asarray(r), [1.0, 0.0])


def test_np_modf_with_out_tuple_preserves_identity():
    with flops.BudgetContext(flop_budget=int(1e10)):
        a = fnp.array([1.5, 2.7, 3.0])
        frac = fnp.zeros(3)
        integer = fnp.zeros(3)
        result = np.modf(a, out=(frac, integer))
    assert result[0] is frac
    assert result[1] is integer
    np.testing.assert_allclose(np.asarray(integer), [1.0, 2.0, 3.0])


def test_np_divmod_with_partial_out_allocates_remaining():
    """``out=(q, None)`` writes through ``q`` and lets numpy allocate
    the second buffer; the freshly-allocated slot comes back as a
    FlopscopeArray."""
    with flops.BudgetContext(flop_budget=int(1e10)):
        a = fnp.array([10.0, 20.0])
        b = fnp.array([3.0, 4.0])
        q = fnp.zeros(2)
        result = np.divmod(a, b, out=(q, None))  # pyright: ignore[reportArgumentType]
    assert result[0] is q
    assert isinstance(result[1], FlopscopeArray)
    np.testing.assert_array_equal(np.asarray(q), [3.0, 5.0])
    np.testing.assert_array_equal(np.asarray(result[1]), [1.0, 0.0])


def test_np_modf_with_positional_out_args():
    """NumPy normalises positional out args (``np.modf(a, o1, o2)``)
    into ``out=(o1, o2)`` before reaching ``__array_ufunc__``."""
    with flops.BudgetContext(flop_budget=int(1e10)):
        a = fnp.array([1.5, 2.7, 3.0])
        o1 = fnp.zeros(3)
        o2 = fnp.zeros(3)
        result = np.modf(a, o1, o2)
    assert result[0] is o1
    assert result[1] is o2


def test_np_divmod_preserves_shared_symmetry():
    """``np.divmod`` of two SymmetricTensors that share an axis-permutation
    group produces both outputs as SymmetricTensors with the same
    symmetry."""
    with flops.BudgetContext(flop_budget=int(1e10)):
        sym = flops.SymmetryGroup.symmetric(axes=(0, 1))
        a = flops.symmetrize(fnp.array([[10.0, 12.0], [12.0, 14.0]]), symmetry=sym)
        b = flops.symmetrize(fnp.array([[3.0, 4.0], [4.0, 5.0]]), symmetry=sym)
        q, r = np.divmod(a, b)
    assert isinstance(q, flops.SymmetricTensor)
    assert isinstance(r, flops.SymmetricTensor)
    assert q.symmetry == sym
    assert r.symmetry == sym


def test_np_modf_preserves_input_symmetry():
    """``np.modf`` is unary elementwise; both outputs inherit the
    SymmetricTensor symmetry of the input."""
    with flops.BudgetContext(flop_budget=int(1e10)):
        sym = flops.SymmetryGroup.symmetric(axes=(0, 1))
        S = flops.symmetrize(fnp.array([[1.5, 2.5], [2.5, 3.5]]), symmetry=sym)
        frac, integer = np.modf(S)
    assert isinstance(frac, flops.SymmetricTensor)
    assert isinstance(integer, flops.SymmetricTensor)
    assert frac.symmetry == sym
    assert integer.symmetry == sym


def test_np_divmod_loses_unshared_symmetry_with_warning():
    """When inputs don't share symmetry, both outputs degrade to plain
    FlopscopeArray and a SymmetryLossWarning is emitted (parity with
    single-output binary ufuncs)."""
    import warnings as _warnings

    with flops.BudgetContext(flop_budget=int(1e10)):
        a = flops.symmetrize(
            fnp.array([[10.0, 12.0], [12.0, 14.0]]),
            symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
        )
        b = fnp.array([[3.0, 4.0], [5.0, 6.0]])  # not symmetric
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            q, r = np.divmod(a, b)
    messages = [str(item.message).lower() for item in caught]
    assert any("symmetry" in m for m in messages), messages
    assert type(q) is FlopscopeArray
    assert type(r) is FlopscopeArray


def test_np_divmod_scalar_left_preserves_array_symmetry():
    """``np.divmod(scalar, symmetric)`` inherits the array's symmetry on
    both outputs (scalar special-case in ``_counted_binary_multi``)."""
    with flops.BudgetContext(flop_budget=int(1e10)):
        sym = flops.SymmetryGroup.symmetric(axes=(0, 1))
        S = flops.symmetrize(fnp.array([[3.0, 4.0], [4.0, 5.0]]), symmetry=sym)
        q, r = np.divmod(20.0, S)
    assert isinstance(q, flops.SymmetricTensor)
    assert isinstance(r, flops.SymmetricTensor)
    assert q.symmetry == sym


def test_we_modf_invalid_out_tuple_length_raises():
    """``out=`` of wrong length is rejected by the multi-output helper
    rather than silently passed through to numpy (which would error
    later with a less helpful message)."""
    with flops.BudgetContext(flop_budget=int(1e10)):
        a = fnp.array([1.5, 2.5, 3.0])
        single = fnp.zeros(3)
        with pytest.raises(ValueError, match="exactly one entry per ufunc output"):
            fnp.modf(a, out=(single,))  # pyright: ignore[reportArgumentType]


def test_we_modf_invalid_out_type_raises():
    """``out=`` that is not a tuple is rejected by the multi-output
    helper for clarity."""
    with flops.BudgetContext(flop_budget=int(1e10)):
        a = fnp.array([1.5, 2.5, 3.0])
        single = fnp.zeros(3)
        with pytest.raises(TypeError, match="tuple"):
            fnp.modf(a, out=single)  # pyright: ignore[reportArgumentType]


# ----- Recursion guard for raw flopscope functions -----


def test_we_add_does_not_recurse_after_protocol_enabled():
    """``fnp.add(FlopscopeArray, FlopscopeArray)`` must not enter an infinite loop
    via ``_np.add`` → ``__array_ufunc__`` → ``fnp.add`` → ... after Task
    3 enables ``__array_ufunc__``. The strip in ``_counted_binary``
    breaks the cycle by viewing both operands as plain ndarray
    before calling ``_np.add``.
    """
    a = fnp.random.randn(8)
    b = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        c = fnp.add(a, b)
    assert bc.flops_used > 0
    assert c.shape == (8,)
    assert isinstance(c, fnp.ndarray)


# ----- __array_function__: np.<func>(flopscope) -----


def test_np_sort_tracks_flops():
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.sort(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        fnp.sort(a)
    assert b1.flops_used == b2.flops_used > 0


def test_np_linalg_norm_tracks_flops():
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.linalg.norm(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        fnp.linalg.norm(a)
    assert b1.flops_used == b2.flops_used > 0


def test_np_where_tracks_flops():
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.where(a > 0, a, 0.0)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        fnp.where(a > 0, a, 0.0)
    assert b1.flops_used == b2.flops_used > 0


def test_np_sum_routes_to_whest():
    """np.sum(flopscope) goes through __array_function__ (not __array_ufunc__)."""
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.sum(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        fnp.sum(a)
    assert b1.flops_used == b2.flops_used > 0


# ----- Structural ops on SymmetricTensor: type follows surviving symmetry -----


def test_diagonal_of_3sym_downgrades_when_no_tensor_axis_symmetry_remains():
    """Diagonal of a (n,n,n) tensor with full S_3 symmetry along (0,1,2)
    collapses axes 0/1 (or any pair) into a single diagonal axis, which
    cannot retain a multi-axis permutation group → result must be
    ``FlopscopeArray``, not ``SymmetricTensor``."""
    n = 4
    A = flops.symmetrize(
        fnp.random.randn(n, n, n),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1, 2)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.warns(SymmetryLossWarning):
            d = fnp.diagonal(A)
    assert not isinstance(d, flops.SymmetricTensor)
    assert isinstance(d, fnp.ndarray)


def test_transpose_of_symmetric_preserves_type():
    """Transposing a 2-axis symmetric matrix preserves the symmetry."""
    A = flops.symmetrize(
        fnp.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)):
        AT = A.T
    assert isinstance(AT, flops.SymmetricTensor)


# ----- Helper unit tests -----


def test_to_base_ndarray_strips_whest_subclass():
    from flopscope._ndarray import FlopscopeArray, _to_base_ndarray

    a = fnp.random.randn(4)
    base = _to_base_ndarray(a)
    assert type(base) is np.ndarray
    assert not isinstance(base, FlopscopeArray)
    base[0] = 99.0
    assert a[0] == 99.0  # zero-copy view


def test_to_base_ndarray_preserves_python_scalar():
    from flopscope._ndarray import _to_base_ndarray

    assert _to_base_ndarray(2.0) == 2.0
    assert isinstance(_to_base_ndarray(2.0), float)


def test_to_base_ndarray_tree_strips_in_tuple():
    from flopscope._ndarray import _to_base_ndarray_tree

    a = fnp.random.randn(4)
    b = fnp.random.randn(4)
    out = _to_base_ndarray_tree((a, b))
    assert all(type(x) is np.ndarray for x in out)


def test_to_base_ndarray_tree_strips_in_list():
    from flopscope._ndarray import _to_base_ndarray_tree

    a = fnp.random.randn(4)
    b = fnp.random.randn(4)
    out = _to_base_ndarray_tree([a, b])
    assert all(type(x) is np.ndarray for x in out)


def test_to_base_ndarray_tree_preserves_scalars():
    from flopscope._ndarray import _to_base_ndarray_tree

    out = _to_base_ndarray_tree((1.0, 2, "x"))
    assert out == (1.0, 2, "x")


def test_to_base_ndarray_tree_recurses_into_nested():
    from flopscope._ndarray import _to_base_ndarray_tree

    a = fnp.random.randn(4)
    out = _to_base_ndarray_tree([(a, 1.0), [a]])
    assert type(out[0][0]) is np.ndarray
    assert type(out[1][0]) is np.ndarray


# ----- _PASSTHROUGH lock-in -----

# Realistic-args registry: every name in _PASSTHROUGH_NAMES must have an
# entry here. Adding a name to _PASSTHROUGH_NAMES without a matching
# _BUILD_ARGS entry causes the parameterized test below to fail loudly
# with KeyError, forcing the author to register realistic args.
#
# NOTE: for some entries (promote_types, mintypecode, broadcast_shapes,
# isfortran, isscalar) numpy does not invoke __array_function__ — either
# no FlopscopeArray is in args, or the function is not NEP-18-decorated.
# For these the dispatch-spy assertion below is vacuous; only the 0-FLOP
# assertion has meaningful coverage (against accidental budget charges).
_BUILD_ARGS = {
    # Zero-FLOP type/shape queries:
    "ndim": lambda: (fnp.empty(4),),
    "shape": lambda: (fnp.empty(4),),
    "size": lambda: (fnp.empty(4),),
    # Zero-FLOP type-system queries:
    "result_type": lambda: (fnp.empty(4),),
    "can_cast": lambda: (fnp.empty(4), np.float64),
    "min_scalar_type": lambda: (fnp.empty(4),),
    "promote_types": lambda: (np.int32, np.float64),  # (vacuous spy)
    "find_common_type": lambda: ([np.float64], []),
    "mintypecode": lambda: (["f", "d"],),  # (vacuous spy)
    # Test-harness assertion:
    "array_equal": lambda: (fnp.empty(4), fnp.empty(4)),
    # Zero-FLOP memory-layout queries (#72):
    "may_share_memory": lambda: (fnp.empty(4), fnp.empty(4)),
    "shares_memory": lambda: (fnp.empty(4), fnp.empty(4)),
    "byte_bounds": lambda: (fnp.empty(4),),
    # Zero-FLOP boolean predicates (#72 audit):
    "iscomplexobj": lambda: (fnp.empty(4),),
    "isrealobj": lambda: (fnp.empty(4),),
    "isfortran": lambda: (fnp.empty(4),),  # (vacuous spy)
    "isscalar": lambda: (fnp.empty(4),),  # (vacuous spy)
    # Zero-FLOP shape arithmetic (#72 audit):
    "broadcast_shapes": lambda: ((4,), (4,)),  # (vacuous spy)
}


def _passthrough_names_present_in_numpy():
    """Parametrize source: every _PASSTHROUGH_NAMES entry whose underlying
    np.<name> exists in the active NumPy version (skips byte_bounds /
    find_common_type which were removed in NumPy 2.0)."""
    from flopscope._ndarray import _PASSTHROUGH_NAMES

    return [n for n in _PASSTHROUGH_NAMES if getattr(np, n, None) is not None]


@pytest.mark.parametrize("name", _passthrough_names_present_in_numpy())
def test_passthrough_name_does_not_dispatch_or_charge(monkeypatch, name):
    """Every name in _PASSTHROUGH_NAMES must:
    (a) charge 0 FLOPs under a BudgetContext, and
    (b) NOT enter FlopscopeArray._get_array_function_dispatch.

    Adding a new name to _PASSTHROUGH_NAMES without a matching _BUILD_ARGS
    entry causes a KeyError here, forcing the author to register args."""
    func = getattr(np, name)
    args = _BUILD_ARGS[name]()

    dispatch_called = []
    real = FlopscopeArray._get_array_function_dispatch.__func__

    def spy(cls):
        dispatch_called.append(name)
        return real(cls)

    monkeypatch.setattr(
        FlopscopeArray,
        "_get_array_function_dispatch",
        classmethod(spy),
    )

    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        func(*args)

    assert bc.flops_used == 0, f"{name} charged {bc.flops_used} FLOPs"
    assert not dispatch_called, (
        f"{name} reached the dispatch lookup inside __array_function__; "
        f"passthrough check failed — verify {name!r} is in _PASSTHROUGH_NAMES"
    )


# ----- Cache-verification test (Stage 2 helper added in Step 2.2) -----


def test_signature_kwargs_accepted_is_cached():
    """The signature lookup must be cached — it sits on the per-ufunc
    hot path. PR #51 memoized similar helpers; we do the same here."""
    from flopscope._ndarray import _signature_kwargs_accepted

    # Same callable should return the same frozenset object (cached).
    a = _signature_kwargs_accepted(np.add)
    b = _signature_kwargs_accepted(np.add)
    assert a is b, "_signature_kwargs_accepted is not cached"


# ----- Performance regression guards (against PR #51 hot paths) -----
# Reference: https://github.com/AIcrowd/flopscope/pull/51#issuecomment-4340098399
# These are SEED tests demonstrating the pattern. Add stage-specific
# perf tests in later stages when implementer reasoning identifies
# hot paths their changes touch.


def test_perf_warm_rank8_scalar_comparison_is_fast():
    """Warm rank-8 scalar comparison ``fnp.full((2,)*8, 1) == 1`` must
    finish well under 100ms.

    PR #51 fixed this from ~930ms warm to ~0.04ms warm via:
    (a) ``SymmetryGroup.__eq__`` identity short-circuit,
    (b) per-instance ``_canonical_axis_action`` cache,
    (c) ``@functools.cache`` on ``unique_elements_for_shape``.
    Generous bound (2500× margin over the post-fix figure) catches
    order-of-magnitude regressions without flaking on machine variance.
    """
    import time

    a = fnp.full((2,) * 8, 1)  # rank-8, 256 elements, full S_8 symmetric group
    # Warm-up: prime the per-instance _canonical_axis_action cache and the
    # @functools.cache on unique_elements_for_shape.
    with flops.BudgetContext(flop_budget=int(1e12)):
        _ = a == 1
    # Measure the warm path.
    with flops.BudgetContext(flop_budget=int(1e12)):
        t0 = time.perf_counter()
        _ = a == 1
        elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, (
        f"warm rank-8 scalar comparison took {elapsed * 1000:.1f}ms; "
        f"PR #51 fixed this to ~0.04ms. Are we re-introducing O(|G|) "
        f"work in __eq__ / _canonical_axis_action / "
        f"unique_elements_for_shape, or making OWNDATA-preserving "
        f"copies in __array_ufunc__?"
    )


def test_perf_array_ufunc_dispatch_does_not_copy():
    """100 invocations of ``np.add(flopscope, flopscope)`` on 1024-element
    arrays must complete well under 1 second.

    PR #51 dropped OWNDATA-preserving copies in ``__array_wrap__``,
    ``_asflopscope``, ``_asplainflopscope`` to keep this path cheap. Our
    ``__array_ufunc__`` handler must not re-introduce them by, e.g.,
    calling ``np.array(x, copy=True)`` or wrapping each result in a
    redundant view-cast that triggers a finalize chain.
    """
    import time

    a = fnp.random.randn(1024)
    b = fnp.random.randn(1024)
    # Warm-up.
    with flops.BudgetContext(flop_budget=int(1e12)):
        _ = np.add(a, b)
    # Measure 100 calls.
    with flops.BudgetContext(flop_budget=int(1e15)):
        t0 = time.perf_counter()
        for _ in range(100):
            _ = np.add(a, b)
        elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, (
        f"100x np.add(flopscope, flopscope) took {elapsed:.3f}s; "
        f"are we re-introducing per-call copies or O(|G|) work in "
        f"__array_ufunc__ / _filter_to_np_signature?"
    )


class TestIssue70AutoInferredOutDowngrade:
    """Regression: issue #70 — auto-inferred symmetry on `out=` target.

    np.zeros_like(plain_3x3) auto-infers S_n symmetry; using it as `out=`
    for an asymmetric write used to raise. With the provenance marker,
    inferred-symmetry out= targets silently downgrade to plain ndarray.
    """

    def test_inferred_out_downgrades_silently(self):
        import numpy as np

        import flopscope.numpy as fnp
        from flopscope._symmetric import SymmetricTensor

        plain_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        mask_np = np.array(
            [[True, False, True], [False, True, False], [True, False, True]]
        )
        out = fnp.zeros_like(plain_np)
        assert isinstance(out, SymmetricTensor)
        assert out._symmetry_inferred is True

        # Before the issue-70 fix this raised ``ValueError: out symmetry does
        # not match result symmetry``. The whole point of the canary is that
        # the call now completes without raising. We deliberately do NOT
        # assert the values at mask=False positions: numpy <2.3 leaves
        # unwritten ``out`` cells uninitialized when both ``out=`` and
        # ``where=`` are given on an ndarray subclass; that behavior is
        # outside the scope of this fix.
        np.positive(plain_np, out=out, where=mask_np)
        actual = np.asarray(out)
        np.testing.assert_array_equal(actual[mask_np], plain_np[mask_np])

    def test_explicit_symmetry_out_still_raises(self):
        import numpy as np
        import pytest

        import flopscope as flops
        import flopscope.numpy as fnp

        plain = fnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        out = flops.as_symmetric(
            np.zeros((3, 3)), symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1))
        )
        mask = fnp.array(
            [[True, False, True], [False, True, False], [True, False, True]]
        )
        with pytest.raises(ValueError, match="out symmetry does not match"):
            np.positive(plain, out=out, where=mask)
