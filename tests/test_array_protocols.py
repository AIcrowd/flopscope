"""Tests verifying numpy's __array_ufunc__ and __array_function__ protocols
route numpy calls through flopscope's counted functions when the operands are
FlopscopeArray (or SymmetricTensor).

Includes adversarial coverage for recursion, out= tuples, kwargs passthrough,
mixed operands, unsupported ufunc methods, and identity preservation.

Translated against post-PR-#51 unified SymmetryGroup API.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._ndarray import FlopscopeArray

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


# ----- In-place dunder identity preservation -----


def test_inplace_add_preserves_identity():
    """a += b must mutate a, not rebind it. Currently broken on
    post-PR-#51 main — fixed in Task 5."""
    a = fnp.random.randn(8)
    old_id = id(a)
    with flops.BudgetContext(flop_budget=int(1e9)):
        a += 1
    assert id(a) == old_id


def test_inplace_add_scalar_on_symmetric_tensor_preserves_identity():
    """``A_sym += 1.0`` is symmetry-preserving (binary-with-scalar keeps
    every group). It must mutate ``A_sym`` in place AND keep the
    SymmetricTensor type — no spurious downgrade, no rebinding."""
    A = flops.symmetrize(
        fnp.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    old_id = id(A)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        A += 1.0
    assert id(A) == old_id
    assert isinstance(A, flops.SymmetricTensor)
    assert bc.flops_used > 0


def test_inplace_add_refuses_when_symmetry_would_be_destroyed():
    """``A_sym += B_unsymmetric`` would silently overwrite ``A_sym``'s
    bytes with a non-symmetric result. The in-place dunder must refuse
    rather than corrupt the symmetry metadata."""
    A = flops.symmetrize(
        fnp.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    B = fnp.random.randn(4, 4)  # plain FlopscopeArray, no symmetry
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises(ValueError, match="destroy or weaken symmetry"):
            A += B


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
    emits :class:`CostFallbackWarning` once per ``(op, degree)``."""
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
    assert "degree 33" in str(cost_warnings[0].message)


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
        with pytest.raises(TypeError, match="length"):
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


def test_np_type_query_passthrough_does_not_charge_flops():
    """``np.result_type``, ``np.can_cast``, ``np.min_scalar_type``, etc.
    are zero-FLOP type-query functions that must bypass flopscope's
    FLOP-tracking dispatch. They are added to ``_PASSTHROUGH_NAMES`` for
    this reason."""
    a = fnp.empty(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        assert np.result_type(a) == np.float64
        assert np.can_cast(a, np.float64)
        assert np.min_scalar_type(a) is not None
    assert bc.flops_used == 0


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


def test_perf_warm_inplace_add_scalar_on_symmetric_is_fast():
    """Warm ``A_sym += 1.0`` on a rank-4 SymmetricTensor must finish
    well under 50 ms, AND must preserve the symmetry object identity.

    This pins three PR #51 fast paths for the in-place dunder rewrite:
    - class 4 (``_counted_binary`` scalar fast path) — ``A_sym + 1.0``
      flows through the scalar branch that returns the operand's
      symmetry unchanged (same object reference).
    - class 5 (``SymmetryGroup.__eq__`` identity short-circuit) —
      ``_inplace_from_result`` does ``self_sym != result_sym``; for
      scalar ops these are the SAME instance, hitting the identity
      short-circuit and skipping the O(|G|) ``_canonical_axis_action``
      comparison.
    - class 6 (per-instance ``_canonical_axis_action`` cache) — even
      if identity short-circuit somehow misses, the cache still keeps
      subsequent calls O(1).

    If this test fails with elapsed > 50 ms or identity is lost, the
    in-place dunder rewrite is constructing a fresh ``SymmetryGroup``
    for comparison or wrapping the scalar into an array somewhere
    along the dispatch chain.
    """
    import time

    arr = fnp.random.randn(4, 4, 4, 4)
    A_sym = flops.symmetrize(
        arr, symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1, 2, 3))
    )
    original_symmetry_ref = A_sym._symmetry  # pyright: ignore[reportAttributeAccessIssue]
    # Warm-up: prime caches and dispatch tables.
    with flops.BudgetContext(flop_budget=int(1e12)):
        A_sym += 1.0
    # The above mutated A_sym; verify identity preserved through warm-up.
    assert A_sym._symmetry is original_symmetry_ref, (  # pyright: ignore[reportAttributeAccessIssue]
        "warm-up in-place add lost symmetry object identity; "
        "_inplace_from_result is constructing a fresh group somewhere"
    )
    # Measure the warm path.
    with flops.BudgetContext(flop_budget=int(1e12)):
        t0 = time.perf_counter()
        A_sym += 1.0
        elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, (
        f"warm A_sym += 1.0 took {elapsed * 1000:.1f}ms; "
        f"PR #51 made this the scalar fast path. Are we missing the "
        f"identity short-circuit in _inplace_from_result, or wrapping "
        f"the scalar in an array in __iadd__ before dispatch?"
    )
    assert A_sym._symmetry is original_symmetry_ref, (  # pyright: ignore[reportAttributeAccessIssue]
        "warm in-place add lost symmetry object identity; "
        "_inplace_from_result is constructing a fresh group somewhere"
    )


# ----- #72 spot-check (RED in Task 1, deleted in Task 3) -----


def test_may_share_memory_bypasses_array_function_dispatch(monkeypatch):
    """np.may_share_memory must NOT enter __array_function__ dispatch.
    It is a zero-FLOP memory-layout query; #72 audit adds it to
    _PASSTHROUGH_NAMES."""
    dispatch_called = []
    real = FlopscopeArray._get_array_function_dispatch.__func__

    def spy(cls):
        dispatch_called.append("may_share_memory")
        return real(cls)

    monkeypatch.setattr(
        FlopscopeArray,
        "_get_array_function_dispatch",
        classmethod(spy),
    )
    a = fnp.empty(4)
    b = fnp.empty(4)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        np.may_share_memory(a, b)
    assert bc.flops_used == 0
    assert not dispatch_called, (
        "np.may_share_memory entered __array_function__ dispatch; "
        "should be in _PASSTHROUGH_NAMES"
    )
