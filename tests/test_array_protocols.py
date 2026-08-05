"""Tests verifying numpy's __array_ufunc__ and __array_function__ protocols
route numpy calls through flopscope's counted functions when the operands are
FlopscopeArray (or SymmetricTensor).

Includes adversarial coverage for recursion, out= tuples, kwargs passthrough,
mixed operands, unsupported ufunc methods, and identity preservation.

Translated against post-PR-#51 unified SymmetryGroup API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.ma as ma
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._ndarray import FlopscopeArray
from flopscope.errors import RemoteCallbackWarning, SymmetryLossWarning

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


def _negative_stride_symmetric_out(values):
    """Build an explicit symmetric destination with observable non-C layout."""
    backing = np.empty_like(values)
    view = backing[::-1, ::-1]
    view[...] = values
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    with flops.BudgetContext(flop_budget=int(1e9)):
        return flops.as_symmetric(view, symmetry=symmetry)


def _call_nonforeign_symmetric_out(operation, value, out, *, tracked=False, **kwargs):
    array_module = fnp if tracked else np
    if operation == "unary":
        return array_module.positive(value, out=out, **kwargs)
    if operation == "binary":
        return array_module.add(value, value, out=out, **kwargs)
    raise AssertionError(f"unknown operation: {operation}")


@pytest.mark.parametrize("operation", ["unary", "binary"])
@pytest.mark.parametrize(
    "where",
    [
        pytest.param(False, id="where-false"),
        pytest.param(
            np.array(
                [
                    [True, False, True],
                    [False, True, False],
                    [True, False, True],
                ]
            ),
            id="partial-mask",
        ),
    ],
)
def test_nonforeign_symmetric_out_preserves_initialized_masked_values(operation, where):
    values = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    initial = np.array(
        [[101.0, 102.0, 103.0], [102.0, 104.0, 105.0], [103.0, 105.0, 106.0]]
    )
    expected = initial.copy()
    _call_nonforeign_symmetric_out(operation, values, expected, where=where)

    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    with flops.BudgetContext(flop_budget=int(1e9)):
        value = flops.as_symmetric(values.copy(), symmetry=symmetry)
    out = _negative_stride_symmetric_out(initial)
    original_strides = out.strides

    with flops.BudgetContext(flop_budget=int(1e9)):
        returned = _call_nonforeign_symmetric_out(
            operation, value, out, where=where, tracked=True
        )

    assert returned is out
    np.testing.assert_array_equal(np.asarray(out), expected)
    assert out.strides == original_strides == (-24, -8)
    assert out.flags.writeable


@pytest.mark.parametrize("operation", ["unary", "binary"])
def test_nonforeign_symmetric_out_uses_numpy_output_casting(operation):
    values = np.array([[1.5, 2.5], [2.5, 4.5]])
    raw_out = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(TypeError) as raw_raised:
        _call_nonforeign_symmetric_out(operation, values, raw_out)

    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    with flops.BudgetContext(flop_budget=int(1e9)):
        value = flops.as_symmetric(values.copy(), symmetry=symmetry)
    out = _negative_stride_symmetric_out(np.zeros((2, 2), dtype=np.int64))
    before = np.asarray(out).copy()
    original_strides = out.strides

    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises(type(raw_raised.value)) as raised:
            _call_nonforeign_symmetric_out(operation, value, out, tracked=True)

    assert str(raised.value) == str(raw_raised.value)
    np.testing.assert_array_equal(np.asarray(out), before)
    assert out.strides == original_strides == (-16, -8)
    assert out.flags.writeable


@pytest.mark.parametrize("operation", ["unary", "binary"])
def test_nonforeign_symmetric_out_uses_numpy_readonly_error(operation):
    values = np.array([[1.0, 2.0], [2.0, 4.0]])
    raw_out = np.zeros((2, 2))
    raw_out.flags.writeable = False
    with pytest.raises(ValueError) as raw_raised:
        _call_nonforeign_symmetric_out(operation, values, raw_out)

    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    with flops.BudgetContext(flop_budget=int(1e9)):
        value = flops.as_symmetric(values.copy(), symmetry=symmetry)
    out = _negative_stride_symmetric_out(np.zeros((2, 2)))
    before = np.asarray(out).copy()
    original_strides = out.strides
    out.flags.writeable = False

    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises(ValueError) as raised:
            _call_nonforeign_symmetric_out(operation, value, out, tracked=True)

    assert str(raised.value) == str(raw_raised.value)
    np.testing.assert_array_equal(np.asarray(out), before)
    assert out.strides == original_strides == (-16, -8)
    assert not out.flags.writeable


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


class _SuccessfulProtocolDuck:
    """Protocol participant that records the raw NumPy dispatch it receives."""

    def __init__(self, values, *, raises=None):
        self.values = np.asarray(values)
        self.raises = raises
        self.calls = []

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.calls.append((ufunc.__name__, method))
        if self.raises is not None:
            raise self.raises
        raw_inputs = tuple(
            self.values
            if value is self
            else np.asarray(value)
            if isinstance(value, FlopscopeArray)
            else value
            for value in inputs
        )
        if "out" in kwargs and kwargs["out"] is not None:
            kwargs["out"] = tuple(
                None if value is None else np.asarray(value) for value in kwargs["out"]
            )
        return getattr(ufunc, method)(*raw_inputs, **kwargs)


class _UnaryProtocolDuck:
    """Unary ufunc protocol participant."""

    def __init__(self, values, *, raises=None, decline=False):
        self.values = np.asarray(values)
        self.raises = raises
        self.decline = decline
        self.ufunc_calls = []

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.ufunc_calls.append((ufunc.__name__, method))
        if self.raises is not None:
            raise self.raises
        if self.decline:
            return NotImplemented
        inputs = tuple(self.values if value is self else value for value in inputs)
        return getattr(ufunc, method)(*inputs, **kwargs)


class _StatefulProtocolMeta(type):
    protocol_lookups = 0

    def __getattribute__(cls, name):
        if name == "__array_ufunc__":
            _StatefulProtocolMeta.protocol_lookups += 1
        return super().__getattribute__(name)


class _MetaclassProtocolDuck(metaclass=_StatefulProtocolMeta):
    def __init__(self):
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        return np.asarray([4.0, 6.0], dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.calls += 1
        return np.asarray([99.0, 99.0])


class _OptOutProtocolMeta(type):
    """Record dynamic class-name reads during a ufunc opt-out error."""

    name_lookups = 0

    def __getattribute__(cls, name):
        if name == "__name__":
            _OptOutProtocolMeta.name_lookups += 1
        return super().__getattribute__(name)


class _NonNdarrayUfuncOptOut(metaclass=_OptOutProtocolMeta):
    __array_ufunc__ = None

    def __init__(self):
        self.array_calls = 0
        self.protocol_calls = 0

    def __array__(self, dtype=None, copy=None):
        self.array_calls += 1
        raise AssertionError("__array__ must not be called for a ufunc opt-out")


class _NdarrayUfuncOptOut(np.ndarray):
    __array_ufunc__ = None  # pyright: ignore[reportAssignmentType]


class _RawReturnUfuncDuck:
    """Foreign NEP 13 participant that deliberately bypasses NumPy outputs."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        return np.asarray([1.0, 2.0], dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.calls += 1
        return self.result


class _SymmetricOutProtocolDuck:
    """Foreign ufunc participant that records and optionally writes ``out``."""

    def __init__(
        self,
        values,
        *,
        write=None,
        result=None,
        ignore_out=False,
        expected_out=None,
        expected_input_alias=False,
        write_input=False,
        raises=None,
        raise_before_write=False,
    ):
        self.values = np.asarray(values)
        self.write = write
        self.result = result
        self.ignore_out = ignore_out
        self.expected_out = expected_out
        self.expected_input_alias = expected_input_alias
        self.write_input = write_input
        self.raises = raises
        self.raise_before_write = raise_before_write
        self.seen_out = None

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.seen_out = kwargs.get("out")
        if self.expected_out is not None:
            assert type(self.seen_out) is tuple and len(self.seen_out) == 1
            actual = self.seen_out[0]
            expected = self.expected_out
            assert type(actual) is np.ndarray
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            assert actual.strides == expected.strides
            assert actual.flags.writeable is expected.flags.writeable
            assert actual.flags.aligned is expected.flags.aligned
            if actual.size:
                assert np.shares_memory(actual, np.asarray(expected))
            else:
                assert (
                    actual.__array_interface__["data"][0]
                    == np.asarray(expected).__array_interface__["data"][0]
                )
            assert actual.tobytes(order="C") == np.asarray(expected).tobytes(order="C")
            if self.expected_input_alias:
                if actual.size:
                    assert np.shares_memory(np.asarray(inputs[0]), actual)
                else:
                    assert (
                        np.asarray(inputs[0]).__array_interface__["data"][0]
                        == actual.__array_interface__["data"][0]
                    )
        if self.raise_before_write:
            assert self.raises is not None
            raise self.raises
        if self.ignore_out:
            return self.result
        raw_inputs = tuple(
            self.values if value is self else np.asarray(value) for value in inputs
        )
        if self.write is None:
            return getattr(ufunc, method)(*raw_inputs, **kwargs)
        assert type(self.seen_out) is tuple and len(self.seen_out) == 1
        target = raw_inputs[0] if self.write_input else self.seen_out[0]
        np.copyto(target, np.asarray(self.write), casting="unsafe")
        if self.raises is not None:
            raise self.raises
        if self.result is None:
            return self.seen_out[0]
        return self.result


class _ForeignTuple(tuple):
    pass


def _strided_symmetric_out(layout):
    if layout == "unaligned":
        backing = np.zeros(73, dtype=np.uint8)
        view = np.ndarray(
            (3, 3),
            dtype=np.float64,
            buffer=backing,
            offset=1,
            strides=(24, 8),
        )
        written = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    elif layout == "positive":
        backing = np.zeros((6, 6), dtype=np.float64)
        view = backing[::2, ::2]
        written = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    elif layout == "negative":
        backing = np.zeros((3, 3), dtype=np.float64)
        view = backing[::-1, ::-1]
        written = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    elif layout == "zero":
        backing = np.zeros(1, dtype=np.float64)
        view = np.ndarray((3, 3), dtype=np.float64, buffer=backing, strides=(0, 0))
        written = np.full((3, 3), -0.0)
    elif layout == "overlapping":
        backing = np.zeros(5, dtype=np.float64)
        view = np.ndarray((3, 3), dtype=np.float64, buffer=backing, strides=(8, 8))
        written = np.add.outer(np.arange(3.0), np.arange(3.0)) + 1.0
    else:
        raise AssertionError(f"unknown layout: {layout}")

    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    with flops.BudgetContext(flop_budget=int(1e10)):
        out = flops.as_symmetric(view, symmetry=symmetry)
    return out, written


def test_metaclass_protocol_lookup_matches_raw_numpy_once():
    _StatefulProtocolMeta.protocol_lookups = 0
    raw_duck = _MetaclassProtocolDuck()
    expected = np.negative(raw_duck)
    raw_lookups = _StatefulProtocolMeta.protocol_lookups

    _StatefulProtocolMeta.protocol_lookups = 0
    duck = _MetaclassProtocolDuck()
    with flops.BudgetContext(flop_budget=int(1e10)):
        actual = fnp.negative(duck)

    np.testing.assert_array_equal(actual, expected)
    assert (raw_lookups, raw_duck.calls) == (1, 1)
    assert (_StatefulProtocolMeta.protocol_lookups, duck.calls) == (1, 1)


@pytest.mark.parametrize(
    "raw_call, flops_call",
    [
        (lambda value: np.negative(value), lambda value: fnp.negative(value)),
        (lambda value: np.modf(value), lambda value: fnp.modf(value)),
    ],
)
def test_non_ndarray_ufunc_opt_out_matches_raw_numpy_without_materializing(
    raw_call, flops_call
):
    _OptOutProtocolMeta.name_lookups = 0
    raw_value = _NonNdarrayUfuncOptOut()
    with pytest.raises(TypeError) as raw_raised:
        raw_call(raw_value)
    raw_name_lookups = _OptOutProtocolMeta.name_lookups

    _OptOutProtocolMeta.name_lookups = 0
    value = _NonNdarrayUfuncOptOut()
    with flops.BudgetContext(flop_budget=int(1e10)) as budget:
        with pytest.raises(TypeError) as raised:
            flops_call(value)

    assert str(raised.value) == str(raw_raised.value)
    assert value.array_calls == raw_value.array_calls == 0
    assert value.protocol_calls == raw_value.protocol_calls == 0
    assert _OptOutProtocolMeta.name_lookups == raw_name_lookups
    assert budget.flops_used == 0


@pytest.mark.parametrize(
    "raw_call, flops_call",
    [
        (
            lambda value: np.add(np.array([1.0, 2.0]), value),
            lambda value, a, _b: fnp.add(a, value),
        ),
        (
            lambda value: np.divmod(np.array([5.0, 6.0]), value),
            lambda value, a, _b: fnp.divmod(a, value),
        ),
    ],
)
def test_non_ndarray_ufunc_opt_out_preflights_before_billing(raw_call, flops_call):
    raw_value = _NonNdarrayUfuncOptOut()
    with pytest.raises(TypeError) as raw_raised:
        raw_call(raw_value)

    value = _NonNdarrayUfuncOptOut()
    a = fnp.array([1.0, 2.0])
    b = fnp.array([3.0, 4.0])
    with flops.BudgetContext(flop_budget=int(1e10)) as budget:
        with pytest.raises(TypeError) as raised:
            flops_call(value, a, b)

    assert str(raised.value) == str(raw_raised.value)
    assert value.array_calls == raw_value.array_calls == 0
    assert value.protocol_calls == raw_value.protocol_calls == 0
    assert budget.flops_used == 0


def test_ndarray_ufunc_opt_out_out_preflights_before_billing():
    raw_out = np.zeros(2).view(_NdarrayUfuncOptOut)
    with pytest.raises(TypeError) as raw_raised:
        np.add(np.array([1.0, 2.0]), np.array([3.0, 4.0]), out=raw_out)

    left = fnp.array([1.0, 2.0])
    right = fnp.array([3.0, 4.0])
    out = fnp.zeros(2).view(_NdarrayUfuncOptOut)
    with flops.BudgetContext(flop_budget=int(1e10)) as budget:
        with pytest.raises(TypeError) as raised:
            fnp.add(left, right, out=out)  # pyright: ignore[reportArgumentType]

    assert str(raised.value) == str(raw_raised.value)
    assert budget.flops_used == 0


@pytest.mark.parametrize(
    "raw_call, flops_call",
    [
        (
            lambda duck: np.add(np.array([3.0, 4.0]), duck),
            lambda duck: fnp.add(fnp.array([3.0, 4.0]), duck),
        ),
        (lambda duck: np.negative(duck), lambda duck: fnp.negative(duck)),
        (lambda duck: np.modf(duck), lambda duck: fnp.modf(duck)),
        (
            lambda duck: np.negative(duck, out=np.zeros(2)),
            lambda duck: fnp.negative(duck, out=fnp.zeros(2)),
        ),
        (
            lambda duck: np.add.outer(np.array([3.0, 4.0]), duck),
            lambda duck: np.add.outer(fnp.array([3.0, 4.0]), duck),
        ),
        (
            lambda duck: np.add.at(np.zeros(2), [0, 1], duck),
            lambda duck: np.add.at(fnp.zeros(2), [0, 1], duck),
        ),
    ],
    ids=["call", "unary", "multi", "out", "outer", "at"],
)
def test_foreign_ufunc_return_preserves_raw_result_identity(raw_call, flops_call):
    raw_sentinel = object()
    raw_duck = _RawReturnUfuncDuck(raw_sentinel)
    expected = raw_call(raw_duck)

    sentinel = object()
    duck = _RawReturnUfuncDuck(sentinel)
    with flops.BudgetContext(flop_budget=int(1e10)):
        actual = flops_call(duck)

    assert expected is raw_sentinel
    assert actual is sentinel
    assert raw_duck.calls == duck.calls == 1


@pytest.mark.parametrize(
    "raw_call, flops_call, make_raw_out, make_out",
    [
        (
            lambda duck, out: np.negative(duck, out=out),
            lambda duck, out: fnp.negative(duck, out=out),
            lambda: np.zeros(2),
            lambda: fnp.zeros(2),
        ),
        (
            lambda duck, out: np.modf(duck, out=out),
            lambda duck, out: fnp.modf(duck, out=out),
            lambda: (np.zeros(2), np.zeros(2)),
            lambda: (fnp.zeros(2), fnp.zeros(2)),
        ),
    ],
)
def test_foreign_ufunc_delegation_preserves_flopscope_out_identity(
    raw_call, flops_call, make_raw_out, make_out
):
    raw_out = make_raw_out()
    raw_result = raw_call(_SuccessfulProtocolDuck([1.5, -2.5]), raw_out)

    out = make_out()
    with flops.BudgetContext(flop_budget=int(1e10)):
        result = flops_call(_SuccessfulProtocolDuck([1.5, -2.5]), out)

    if isinstance(out, tuple):
        for raw_actual, raw_destination in zip(raw_result, raw_out, strict=True):
            assert raw_actual is raw_destination
        for actual, destination in zip(result, out, strict=True):
            assert actual is destination
    else:
        assert raw_result is raw_out
        assert result is out
    np.testing.assert_array_equal(np.asarray(out), np.asarray(raw_out))


def test_foreign_ufunc_writes_and_returns_symmetric_out():
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    symmetric_input = flops.symmetrize(
        fnp.array([[1.0, 2.0], [2.0, 3.0]]), symmetry=symmetry
    )
    out = flops.symmetrize(fnp.zeros((2, 2)), symmetry=symmetry)
    duck = _SymmetricOutProtocolDuck(10.0)

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            result = fnp.add(symmetric_input, duck, out=out)

    assert type(duck.seen_out) is tuple and len(duck.seen_out) == 1
    assert type(duck.seen_out[0]) is np.ndarray
    assert np.shares_memory(duck.seen_out[0], np.asarray(out))
    assert result is out
    np.testing.assert_array_equal(np.asarray(out), np.asarray(symmetric_input) + 10.0)


@pytest.mark.parametrize(
    "layout", ["unaligned", "positive", "negative", "zero", "overlapping"]
)
def test_foreign_ufunc_preserves_symmetric_out_layout(layout):
    out, written = _strided_symmetric_out(layout)
    duck = _SymmetricOutProtocolDuck(
        0.0,
        write=written,
        expected_out=out,
        expected_input_alias=True,
        write_input=True,
    )

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            result = fnp.add(out, duck, out=out)

    assert result is out
    np.testing.assert_array_equal(np.asarray(out), written)
    assert np.asarray(out).tobytes(order="C") == written.tobytes(order="C")


def test_foreign_ufunc_ignored_readonly_symmetric_out_preserves_sentinel():
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    symmetric_input = flops.symmetrize(
        fnp.array([[1.0, 2.0], [2.0, 3.0]]), symmetry=symmetry
    )
    out = flops.symmetrize(fnp.array([[4.0, -0.0], [-0.0, 5.0]]), symmetry=symmetry)
    out.flags.writeable = False
    before = np.asarray(out).tobytes()
    sentinel = object()
    duck = _SymmetricOutProtocolDuck(
        10.0,
        result=sentinel,
        ignore_out=True,
        expected_out=out,
    )

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            result = fnp.add(symmetric_input, duck, out=out)

    assert result is sentinel
    assert np.asarray(out).tobytes() == before
    assert out.flags.writeable is False
    assert out.symmetry == symmetry


def test_foreign_ufunc_write_to_readonly_symmetric_out_fails_before_commit():
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    symmetric_input = flops.symmetrize(
        fnp.array([[1.0, 2.0], [2.0, 3.0]]), symmetry=symmetry
    )
    out = flops.symmetrize(fnp.full((2, 2), 7.0), symmetry=symmetry)
    out.flags.writeable = False
    before = np.asarray(out).tobytes()
    duck = _SymmetricOutProtocolDuck(10.0, expected_out=out)

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            with pytest.raises(ValueError, match="read-only"):
                fnp.add(symmetric_input, duck, out=out)

    assert np.asarray(out).tobytes() == before
    assert out.flags.writeable is False
    assert out.symmetry == symmetry


@pytest.mark.parametrize(
    "callback_result", [None, object()], ids=["canonical", "sentinel"]
)
def test_foreign_ufunc_rejects_asymmetric_alias_write_and_rolls_back(
    callback_result,
):
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    out = flops.symmetrize(fnp.full((2, 2), 7.0), symmetry=symmetry)
    before = np.asarray(out).tobytes()
    duck = _SymmetricOutProtocolDuck(
        0.0,
        write=np.array([[1.0, 2.0], [3.0, 4.0]]),
        result=callback_result,
        expected_out=out,
        expected_input_alias=True,
        write_input=True,
    )

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            with pytest.raises(flops.errors.SymmetryError):
                fnp.add(out, duck, out=out)

    assert type(duck.seen_out) is tuple and len(duck.seen_out) == 1
    assert np.asarray(out).tobytes() == before
    assert out.symmetry == symmetry


def test_foreign_ufunc_preserves_sentinel_while_committing_safe_out_write():
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    out = flops.symmetrize(fnp.zeros((2, 2)), symmetry=symmetry)
    sentinel = object()
    written = np.array([[21.0, 22.0], [22.0, 23.0]])
    duck = _SymmetricOutProtocolDuck(
        0.0,
        write=written,
        result=sentinel,
        expected_out=out,
        expected_input_alias=True,
        write_input=True,
    )

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            result = fnp.add(out, duck, out=out)

    assert result is sentinel
    np.testing.assert_array_equal(np.asarray(out), written)
    assert out.symmetry == symmetry


def test_foreign_ufunc_partial_input_out_alias_commits_valid_shared_write():
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    backing = np.zeros((3, 3), dtype=np.float64)
    with flops.BudgetContext(flop_budget=int(1e10)):
        out = flops.as_symmetric(backing[:2, :2], symmetry=symmetry)
        aliased_input = flops.as_symmetric(backing[1:, 1:], symmetry=symmetry)
    sentinel = object()
    duck = _SymmetricOutProtocolDuck(
        0.0,
        write=np.array([[9.0, 0.0], [0.0, 0.0]]),
        result=sentinel,
        expected_out=out,
        expected_input_alias=True,
        write_input=True,
    )

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            result = fnp.add(aliased_input, duck, out=out)

    assert result is sentinel
    np.testing.assert_array_equal(np.asarray(out), [[0.0, 0.0], [0.0, 9.0]])
    assert out.symmetry == symmetry


def test_foreign_ufunc_alias_mutation_then_raise_preserves_exception_and_write():
    from flopscope._write_epoch import epoch_of

    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    out = flops.symmetrize(fnp.zeros((2, 2)), symmetry=symmetry)
    before_epoch = epoch_of(out)
    written = np.array([[1.0, 2.0], [3.0, 4.0]])
    error = RuntimeError("callback failed after mutating aliased input")
    duck = _SymmetricOutProtocolDuck(
        0.0,
        write=written,
        expected_out=out,
        expected_input_alias=True,
        write_input=True,
        raises=error,
    )

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            with pytest.raises(RuntimeError) as raised:
                fnp.add(out, duck, out=out)

    assert raised.value is error
    np.testing.assert_array_equal(np.asarray(out), written)
    assert epoch_of(out) != before_epoch
    assert out.symmetry is None


def test_foreign_ufunc_raise_before_alias_mutation_preserves_epoch_and_symmetry():
    from flopscope._write_epoch import epoch_of

    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    out = flops.symmetrize(fnp.full((2, 2), 7.0), symmetry=symmetry)
    before = np.asarray(out).tobytes()
    before_epoch = epoch_of(out)
    error = RuntimeError("callback failed before mutation")
    duck = _SymmetricOutProtocolDuck(
        0.0,
        expected_out=out,
        expected_input_alias=True,
        raises=error,
        raise_before_write=True,
    )

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            with pytest.raises(RuntimeError) as raised:
                fnp.add(out, duck, out=out)

    assert raised.value is error
    assert np.asarray(out).tobytes() == before
    assert epoch_of(out) == before_epoch
    assert out.symmetry == symmetry


def test_foreign_ufunc_empty_aliased_out_preserves_identity_and_symmetry():
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    with flops.BudgetContext(flop_budget=int(1e10)):
        out = flops.as_symmetric(np.empty((0, 0)), symmetry=symmetry)
    duck = _SymmetricOutProtocolDuck(
        1.0,
        expected_out=out,
        expected_input_alias=True,
    )

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            result = fnp.add(out, duck, out=out)

    assert result is out
    assert out.shape == (0, 0)
    assert out.symmetry == symmetry


def test_foreign_ufunc_nan_payload_change_is_committed_bitwise():
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    out = flops.symmetrize(fnp.zeros((2, 2)), symmetry=symmetry)
    payload_bits = np.full((2, 2), 0x7FF8000000000042, dtype=np.uint64)
    written = payload_bits.view(np.float64)
    sentinel = object()
    duck = _SymmetricOutProtocolDuck(
        0.0,
        write=written,
        result=sentinel,
        expected_out=out,
        expected_input_alias=True,
        write_input=True,
    )

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            result = fnp.add(out, duck, out=out)

    assert result is sentinel
    assert np.asarray(out).tobytes(order="C") == written.tobytes(order="C")
    assert out.symmetry is None


def test_foreign_ufunc_where_mask_preserves_unwritten_symmetric_out_values():
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    symmetric_input = flops.symmetrize(
        fnp.array([[1.0, 2.0], [2.0, 3.0]]), symmetry=symmetry
    )
    out = flops.symmetrize(fnp.full((2, 2), 5.0), symmetry=symmetry)
    mask = np.array([[True, False], [False, True]])
    duck = _SymmetricOutProtocolDuck(10.0, expected_out=out)

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            result = fnp.add(symmetric_input, duck, out=out, where=mask)

    assert result is out
    np.testing.assert_array_equal(np.asarray(out), [[11.0, 5.0], [5.0, 13.0]])
    assert out.symmetry == symmetry


# Both operations are currently registered through ``_counted_unary``;
# parametrization covers numeric-output and boolean-output ufunc loops.
@pytest.mark.parametrize("operation", [fnp.negative, fnp.signbit])
def test_foreign_unary_ufunc_writes_inferred_symmetric_out(operation):
    values = np.array([[1.0, -2.0], [3.0, 4.0]])
    expected = getattr(np, operation.__name__)(values)
    out = fnp.zeros_like(expected)
    assert isinstance(out, flops.SymmetricTensor)
    assert out._symmetry_inferred is True
    duck = _SymmetricOutProtocolDuck(values)

    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.warns(RemoteCallbackWarning):
            result = operation(duck, out=out)

    assert type(duck.seen_out) is tuple and len(duck.seen_out) == 1
    assert type(duck.seen_out[0]) is np.ndarray
    assert result is out
    np.testing.assert_array_equal(np.asarray(out), expected)
    assert out.symmetry is None


def test_foreign_ufunc_at_callback_records_successful_write():
    """A foreign ``ufunc.at`` handler can mutate the target before returning."""
    from flopscope._write_epoch import epoch_of

    class MutatingDuck:
        def __init__(self, result):
            self.result = result
            self.calls = 0

        def __array__(self, dtype=None, copy=None):
            return np.asarray([0.0], dtype=dtype)

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            assert (ufunc, method) == (np.add, "at")
            self.calls += 1
            inputs[0][0] += 7
            return self.result

    raw_result = object()
    raw_target = np.zeros(2)
    raw_duck = MutatingDuck(raw_result)
    expected = np.add.at(raw_target, [0], raw_duck)

    result = object()
    target = fnp.zeros(2)
    before = epoch_of(target)
    duck = MutatingDuck(result)
    with flops.BudgetContext(flop_budget=10**9):
        actual = np.add.at(target, [0], duck)

    assert expected is raw_result
    assert actual is result
    assert raw_duck.calls == duck.calls == 1
    np.testing.assert_array_equal(target, raw_target)
    assert epoch_of(target) != before


def test_foreign_ufunc_at_callback_records_write_before_raising():
    """A callback may mutate ``at``'s target before propagating its exception."""
    from flopscope._write_epoch import epoch_of

    class RaisingMutatingDuck:
        def __init__(self, error):
            self.error = error
            self.calls = 0

        def __array__(self, dtype=None, copy=None):
            return np.asarray([0.0], dtype=dtype)

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            assert (ufunc, method) == (np.add, "at")
            self.calls += 1
            inputs[0][0] += 7
            raise self.error

    raw_error = RuntimeError("raw callback failure")
    raw_target = np.zeros(2)
    raw_duck = RaisingMutatingDuck(raw_error)
    with pytest.raises(RuntimeError) as raw_raised:
        np.add.at(raw_target, [0], raw_duck)

    error = RuntimeError("callback failure")
    target = fnp.zeros(2)
    before = epoch_of(target)
    duck = RaisingMutatingDuck(error)
    with flops.BudgetContext(flop_budget=10**9):
        with pytest.raises(RuntimeError) as raised:
            np.add.at(target, [0], duck)

    assert raw_raised.value is raw_error
    assert raised.value is error
    assert raw_duck.calls == duck.calls == 1
    np.testing.assert_array_equal(target, raw_target)
    assert epoch_of(target) != before


@pytest.mark.parametrize(
    "make_result",
    [
        lambda: (object(),),
        lambda: _ForeignTuple((object(), object())),
    ],
    ids=["short-tuple", "tuple-subclass"],
)
def test_foreign_multi_ufunc_out_preserves_arbitrary_tuple_result_identity(
    make_result,
):
    raw_result = make_result()
    raw_duck = _RawReturnUfuncDuck(raw_result)
    raw_out = (np.zeros(2), np.zeros(2))
    expected = np.modf(raw_duck, out=raw_out)

    result = make_result()
    duck = _RawReturnUfuncDuck(result)
    out = (fnp.zeros(2), fnp.zeros(2))
    with flops.BudgetContext(flop_budget=int(1e10)):
        actual = fnp.modf(duck, out=out)

    assert expected is raw_result
    assert actual is result
    assert type(actual) is type(result)
    assert raw_duck.calls == duck.calls == 1


def test_ufunc_opt_out_precedes_invalid_out_normalization():
    invalid_out: Any = "not-an-array"
    raw_value = _NonNdarrayUfuncOptOut()
    with pytest.raises(TypeError) as raw_raised:
        np.negative(raw_value, out=invalid_out)

    value = _NonNdarrayUfuncOptOut()
    with flops.BudgetContext(flop_budget=int(1e10)) as budget:
        with pytest.raises(TypeError) as raised:
            fnp.negative(value, out=invalid_out)

    assert str(raised.value) == str(raw_raised.value)
    assert value.array_calls == raw_value.array_calls == 0
    assert budget.flops_used == 0


def test_ufunc_wrong_out_arity_precedes_opt_out_preflight():
    raw_value = _NonNdarrayUfuncOptOut()
    raw_out: Any = (np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError) as raw_raised:
        np.negative(raw_value, out=raw_out)

    _OptOutProtocolMeta.name_lookups = 0
    value = _NonNdarrayUfuncOptOut()
    out: Any = (fnp.zeros(2), fnp.zeros(2))
    with flops.BudgetContext(flop_budget=int(1e10)) as budget:
        with pytest.raises(ValueError) as raised:
            fnp.negative(value, out=out)

    assert str(raised.value) == str(raw_raised.value)
    assert value.array_calls == value.protocol_calls == 0
    assert _OptOutProtocolMeta.name_lookups == 0
    assert budget.flops_used == 0


def test_tuple_subclass_out_does_not_preflight_opt_out_contents():
    raw_out = _ForeignTuple((_NonNdarrayUfuncOptOut(),))
    with pytest.raises(TypeError) as raw_raised:
        np.negative(np.ones(2), out=raw_out)

    _OptOutProtocolMeta.name_lookups = 0
    opt_out = _NonNdarrayUfuncOptOut()
    out: Any = _ForeignTuple((opt_out,))
    value = fnp.ones(2)
    with flops.BudgetContext(flop_budget=int(1e10)) as budget:
        with pytest.raises(TypeError) as raised:
            fnp.negative(value, out=out)

    assert type(raised.value) is type(raw_raised.value)
    assert "_NonNdarrayUfuncOptOut" not in str(raised.value)
    assert opt_out.array_calls == opt_out.protocol_calls == 0
    assert _OptOutProtocolMeta.name_lookups == 0
    assert budget.flops_used == 0


def test_wrong_arity_tuple_subclass_does_not_precede_input_opt_out():
    raw_value = _NonNdarrayUfuncOptOut()
    raw_out = _ForeignTuple((np.zeros(2), np.zeros(2)))
    with pytest.raises(TypeError) as raw_raised:
        np.negative(raw_value, out=raw_out)

    value = _NonNdarrayUfuncOptOut()
    out: Any = _ForeignTuple((fnp.zeros(2), fnp.zeros(2)))
    with flops.BudgetContext(flop_budget=int(1e10)) as budget:
        with pytest.raises(TypeError) as raised:
            fnp.negative(value, out=out)

    assert str(raised.value) == str(raw_raised.value)
    assert value.array_calls == value.protocol_calls == 0
    assert budget.flops_used == 0


@pytest.mark.parametrize(
    "name, raw_call, flops_call, expected_protocol",
    [
        (
            "negative",
            lambda duck: np.negative(duck),
            lambda duck: fnp.negative(duck),
            ("negative", "__call__"),
        ),
        (
            "modf",
            lambda duck: np.modf(duck),
            lambda duck: fnp.modf(duck),
            ("modf", "__call__"),
        ),
    ],
)
def test_unary_ufunc_protocol_matches_raw_numpy(
    name, raw_call, flops_call, expected_protocol
):
    raw_duck = _UnaryProtocolDuck([1.5, -2.5])
    expected = raw_call(raw_duck)
    duck = _UnaryProtocolDuck([1.5, -2.5])
    with flops.BudgetContext(flop_budget=int(1e10)):
        actual = flops_call(duck)

    assert raw_duck.ufunc_calls == duck.ufunc_calls == [expected_protocol], name
    if isinstance(expected, tuple):
        for actual_part, expected_part in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(actual_part, expected_part)
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "raw_call, flops_call",
    [
        (lambda duck: np.negative(duck), lambda duck: fnp.negative(duck)),
        (lambda duck: np.modf(duck), lambda duck: fnp.modf(duck)),
    ],
)
def test_unary_ufunc_protocol_exception_identity_matches_raw_numpy(
    raw_call, flops_call
):
    error = RuntimeError("unary protocol boom")
    raw_duck = _UnaryProtocolDuck([1.5, -2.5], raises=error)
    with pytest.raises(RuntimeError) as raw_raised:
        raw_call(raw_duck)

    duck = _UnaryProtocolDuck([1.5, -2.5], raises=error)
    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.raises(RuntimeError) as raised:
            flops_call(duck)
    assert raw_raised.value is raised.value is error
    assert raw_duck.ufunc_calls == duck.ufunc_calls


@pytest.mark.parametrize(
    "call", [lambda duck: fnp.negative(duck), lambda duck: fnp.modf(duck)]
)
def test_unary_ufunc_protocol_notimplemented_matches_raw_numpy(call):
    raw_duck = _UnaryProtocolDuck([1.5, -2.5], decline=True)
    with pytest.raises(TypeError):
        call(raw_duck)

    duck = _UnaryProtocolDuck([1.5, -2.5], decline=True)
    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.raises(TypeError):
            call(duck)
    assert raw_duck.ufunc_calls == duck.ufunc_calls


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
    "name, raw_call, flops_call",
    [
        (
            "call",
            lambda duck: np.add(np.array([1.0, 2.0]), duck),
            lambda duck: fnp.add(fnp.array([1.0, 2.0]), duck),
        ),
        (
            "outer",
            lambda duck: np.add.outer(np.array([1.0, 2.0]), duck),
            lambda duck: np.add.outer(fnp.array([1.0, 2.0]), duck),
        ),
        (
            "reduce",
            lambda duck: np.subtract.reduce(duck, axis=0),
            lambda duck: np.subtract.reduce(duck, axis=0, out=fnp.zeros(())),
        ),
        (
            "accumulate",
            lambda duck: np.subtract.accumulate(duck, axis=0),
            lambda duck: np.subtract.accumulate(duck, axis=0, out=fnp.zeros(2)),
        ),
        (
            "reduceat",
            lambda duck: np.add.reduceat(duck, [0, 1], axis=0),
            lambda duck: np.add.reduceat(duck, [0, 1], axis=0, out=fnp.zeros(2)),
        ),
        (
            "at",
            lambda duck: np.add.at(np.zeros(2), [0, 1], duck),
            lambda duck: np.add.at(fnp.zeros(2), [0, 1], duck),
        ),
    ],
)
def test_successful_ufunc_protocol_matches_raw_numpy_dispatch(
    name, raw_call, flops_call
):
    raw_duck = _SuccessfulProtocolDuck([3.0, 4.0])
    expected = raw_call(raw_duck)
    duck = _SuccessfulProtocolDuck([3.0, 4.0])
    with flops.BudgetContext(flop_budget=int(1e10)):
        actual = flops_call(duck)

    assert (
        duck.calls
        == raw_duck.calls
        == [
            (
                "add" if name not in {"reduce", "accumulate"} else "subtract",
                name if name != "call" else "__call__",
            )
        ]
    )
    if expected is not None:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda duck: fnp.add(fnp.array([1.0, 2.0]), duck),
        lambda duck: np.add.outer(fnp.array([1.0, 2.0]), duck),
        lambda duck: np.subtract.reduce(duck, axis=0, out=fnp.zeros(())),
        lambda duck: np.subtract.accumulate(duck, axis=0, out=fnp.zeros(2)),
        lambda duck: np.add.reduceat(duck, [0, 1], axis=0, out=fnp.zeros(2)),
        lambda duck: np.add.at(fnp.zeros(2), [0, 1], duck),
    ],
    ids=["call", "outer", "reduce", "accumulate", "reduceat", "at"],
)
def test_ufunc_protocol_exception_identity_matches_raw_numpy(call):
    error = RuntimeError("protocol boom")
    duck = _SuccessfulProtocolDuck([3.0, 4.0], raises=error)
    with flops.BudgetContext(flop_budget=int(1e10)):
        with pytest.raises(RuntimeError) as raised:
            call(duck)
    assert raised.value is error
    assert len(duck.calls) == 1


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


def test_np_multiply_outer_b_resolution_resizing_a_bills_the_post_resize_array():
    """Regression pin for the confirmed stale-``a``-billing-view defect in
    ``outer``: ``a``'s billing view used to be captured BEFORE ``b`` was
    resolved. ``outer`` takes two data operands, not one, so resolving the
    SECOND one is itself participant code (an ``__array__`` call, for a
    duck ``b``) that can reach back and mutate the first -- the caller
    controls the closure ``b.__array__`` runs in and can bind it to the
    very ``a`` object passed alongside it. The bill must reflect the array
    ``outer`` actually multiplies against, not the tiny pre-resize
    snapshot taken before ``b`` was ever touched.
    """
    n = 1_000_000
    a = _OwningFloat64(4)

    class _ResizingB:
        def __array__(self, dtype=None, copy=None):
            a.resize(n, refcheck=False)
            return np.ones(4, dtype=np.float64)

    out = fnp.zeros((n, 4))
    with flops.BudgetContext(flop_budget=int(1e10)) as bc:
        np.multiply.outer(a, _ResizingB(), out=out)
    assert a.size == n, "sanity: the resize actually ran"

    with flops.BudgetContext(flop_budget=int(1e10)) as honest_bc:
        np.multiply.outer(
            fnp.asarray(np.full(n, 1.0)),
            fnp.asarray(np.ones(4)),
            out=fnp.zeros((n, 4)),
        )
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
