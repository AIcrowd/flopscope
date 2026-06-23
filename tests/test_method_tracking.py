"""Tests verifying ndarray methods, NumPy protocols, and SymmetricTensor type
consolidation track FLOPs and propagate symmetry correctly.

Each test pins one bug from #58 (method calls), #38 (SymmetricTensor dunders),
or #62 (no-symmetry SymmetricTensor type ambiguity).

Translated against post-PR-#51 unified SymmetryGroup API.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope.errors import SymmetryLossWarning

# ----- #58: ndarray method calls on FlopscopeArray must track FLOPs -----

REDUCTION_METHODS = [
    ("sum", lambda a: a.sum()),
    ("mean", lambda a: a.mean()),
    ("max", lambda a: a.max()),
    ("min", lambda a: a.min()),
    ("prod", lambda a: a.prod()),
    ("std", lambda a: a.std()),
    ("var", lambda a: a.var()),
    ("argmax", lambda a: a.argmax()),
    ("argmin", lambda a: a.argmin()),
    ("cumsum", lambda a: a.cumsum()),
    ("cumprod", lambda a: a.cumprod()),
]


@pytest.mark.parametrize("name,op", REDUCTION_METHODS)
def test_reduction_method_tracks_flops(name, op):
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        op(a)
    assert bc.flops_used > 0, f"{name}() method bypassed FLOP tracking"


@pytest.mark.parametrize("name,op", REDUCTION_METHODS)
def test_reduction_method_matches_function(name, op):
    """a.foo() must produce the same FLOP count as fnp.foo(a)."""
    we_func = getattr(fnp, name)
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        op(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        we_func(a)
    assert b1.flops_used == b2.flops_used, (
        f"{name}: method={b1.flops_used}, function={b2.flops_used}"
    )


# ----- Non-protocol ndarray methods (need explicit overrides) -----

# These methods do NOT route through __array_ufunc__ or __array_function__,
# so a method override on FlopscopeArray is the only way to track them.

NON_PROTOCOL_METHODS = [
    ("dot", lambda a: a.dot(a)),
    ("argsort", lambda a: a.argsort()),
    ("argpartition", lambda a: a.argpartition(2)),
    ("take", lambda a: a.take([0, 1, 2])),
    ("repeat", lambda a: a.repeat(2)),
    # ``compress`` requires a boolean condition the same length as a's first axis.
    ("compress", lambda a: a.compress([True] * a.shape[0])),
    ("conj", lambda a: a.conj()),
    ("conjugate", lambda a: a.conjugate()),
    ("clip", lambda a: a.clip(-1.0, 1.0)),
    ("round", lambda a: a.round()),
    ("trace", lambda a: a.reshape(2, 4).trace()),
    ("ptp", lambda a: a.ptp()),
]


@pytest.mark.parametrize("name,op", NON_PROTOCOL_METHODS)
def test_non_protocol_method_tracks_flops(name, op):
    """Methods that bypass both __array_ufunc__ and __array_function__
    (``a.dot``, ``a.argsort``, etc.) must be explicitly overridden on
    FlopscopeArray to track FLOPs."""
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        op(a)
    assert bc.flops_used > 0, f"{name}() bypassed FLOP tracking"


def test_searchsorted_method_tracks():
    """``searchsorted`` is non-protocol but needs a sorted array; build
    one with plain NumPy *outside* any budget so the method-call FLOPs
    are the only thing in ``bc.flops_used``. Calling ``fnp.sort`` /
    ``fnp.random.randn`` outside a budget would itself raise.
    """
    sorted_arr = np.sort(np.random.randn(8)).view(fnp.ndarray)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        sorted_arr.searchsorted(0.0)
    assert bc.flops_used > 0


# ----- In-place sort / partition: immutable, must raise -----


def test_inplace_sort_raises_immutable():
    """``a.sort()`` must raise; flopscope arrays are immutable.
    Use fnp.sort(arr) for a sorted copy."""
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises(ValueError, match="immutable"):
            a.sort()


def test_inplace_partition_raises_immutable():
    """``a.partition(k)`` must raise; flopscope arrays are immutable.
    Use fnp.partition(arr, k) for a copy."""
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises(ValueError, match="immutable"):
            a.partition(3)


def test_inplace_sort_on_symmetric_raises_immutable():
    """``A_sym.sort()`` must raise (immutable, not a symmetry-specific error)."""
    A = flops.symmetrize(
        fnp.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises(ValueError, match="immutable"):
            A.sort()


def test_inplace_partition_on_symmetric_raises_immutable():
    """``A_sym.partition(k)`` must raise (immutable)."""
    A = flops.symmetrize(
        fnp.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises(ValueError, match="immutable"):
            A.partition(2)


# ----- Adversarial axis / keepdims / dtype / out coverage -----


def test_method_axis_none_tracks():
    a = fnp.random.randn(4, 5)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        a.sum(axis=None)
    assert bc.flops_used > 0


def test_method_negative_axis_tracks():
    a = fnp.random.randn(4, 5)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        a.sum(axis=-1)
    assert bc.flops_used > 0


def test_method_tuple_axis_tracks():
    a = fnp.random.randn(4, 5, 6)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        a.sum(axis=(0, 2))
    assert bc.flops_used > 0


def test_method_keepdims_tracks():
    a = fnp.random.randn(4, 5)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        r = a.sum(axis=0, keepdims=True)
    assert bc.flops_used > 0
    assert r.shape == (1, 5)


def test_method_dtype_tracks():
    a = fnp.random.randn(8).astype(np.float32)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        r = a.sum(dtype=np.float64)
    assert bc.flops_used > 0
    assert r.dtype == np.float64


def test_method_positional_dtype_tracks():
    """``a.sum(0, np.float64)`` passes axis and dtype positionally (NumPy
    signature is ``sum(a, axis, dtype, out, ...)``). The method override
    must forward through to ``_counted_reduction`` without dropping the
    positional dtype arg."""
    a = fnp.random.randn(4, 5).astype(np.float32)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        r = a.sum(0, np.float64)
    assert bc.flops_used > 0
    assert r.dtype == np.float64
    assert r.shape == (5,)


def test_method_out_tracks_and_returns_out():
    a = fnp.random.randn(4, 5)
    out = fnp.zeros(5)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        ret = a.sum(axis=0, out=out)
    assert bc.flops_used > 0
    assert ret is out


def test_method_positional_out_tracks_and_returns_out():
    """``a.sum(axis, dtype, out)`` passes ``out`` *positionally* in the
    fourth slot. The reduction wrapper must (a) detect positional ``out``
    via signature inspection, (b) strip its flopscope subclass before the
    underlying ``np.sum`` call so ``__array_function__`` does not
    re-dispatch and double-charge, and (c) return the original ``out``
    object (identity) rather than a re-wrapped view.
    """
    a = fnp.random.randn(4, 5)
    out = fnp.zeros(5)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        ret = a.sum(0, None, out)  # axis=0, dtype=None, out=out (positional)
    assert bc.flops_used > 0
    assert ret is out
    # Cross-check: same FLOP count as the kwarg form.
    out2 = fnp.zeros(5)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc2:
        a.sum(axis=0, out=out2)
    assert bc.flops_used == bc2.flops_used


def test_method_positional_out_argmax_tracks_and_returns_out():
    """``argmax`` has a different positional layout than ``sum``:
    NumPy's signature is ``argmax(a, axis=None, out=None, *, keepdims)``.
    The wrapper's signature inspection must locate ``out`` at the
    correct positional slot for *each* underlying NumPy function — not
    just the ``sum``-shaped layout."""
    a = fnp.random.randn(4, 5)
    out = fnp.zeros(5, dtype=np.intp)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        ret = a.argmax(0, out)  # axis=0, out=out (positional, no dtype slot)
    assert bc.flops_used > 0
    assert ret is out


# ----- Symmetry propagation through methods -----


def test_method_propagates_symmetry_through_reduction():
    """Issue #58: A.sum(axis=0) on a SymmetricTensor must propagate symmetry."""
    n = 10
    A = flops.symmetrize(
        fnp.random.randn(n, n, n),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1, 2)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)):
        S_method = A.sum(axis=0)
        S_func = fnp.sum(A, axis=0)
    # Compare via SymmetryGroup value-equality (post-PR-#51).
    assert S_method.symmetry == S_func.symmetry  # pyright: ignore[reportAttributeAccessIssue]
    assert S_method.symmetry is not None  # pyright: ignore[reportAttributeAccessIssue] # surviving symmetry


def test_no_double_counting_method_vs_function():
    """a.sum() should not double-charge by routing through both methods and
    me.sum (which itself shouldn't re-dispatch via __array_ufunc__).
    """
    a = fnp.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc1:
        a.sum()
    with flops.BudgetContext(flop_budget=int(1e9)) as bc2:
        fnp.sum(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc3:
        np.sum(a)
    assert bc1.flops_used == bc2.flops_used == bc3.flops_used > 0


# ----- #38: dunder operators on SymmetricTensor must track FLOPs -----


def test_sym_mul_tracks_flops():
    """Issue #38: SymmetricTensor * SymmetricTensor must track FLOPs."""
    n = 10
    A = flops.symmetrize(
        fnp.random.randn(n, n),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        A * A  # pyright: ignore[reportUnusedExpression]
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        fnp.multiply(A, A)
    assert b1.flops_used == b2.flops_used > 0


def test_sym_add_tracks_flops():
    n = 10
    A = flops.symmetrize(
        fnp.random.randn(n, n),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        A + A  # pyright: ignore[reportUnusedExpression]
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        fnp.add(A, A)
    assert b1.flops_used == b2.flops_used > 0


# ----- #62: SymmetricTensor with no surviving symmetry should downgrade -----
# Note: PR #51 implements the downgrade architecturally (via __array_finalize__
# + __array_wrap__); the original v1 plan's tests for empty-axes /
# singleton-axes / order-one as_symmetric construction are no longer expressible
# in the new API (it requires a SymmetryGroup; "empty axes" is not a valid
# input). We keep only the downgrade tests that are still expressible.


def test_no_symmetry_downgrades_to_whest_array_via_diagonal():
    """Issue #62: ``fnp.diagonal`` of a 2-axis-symmetric matrix has no
    surviving symmetry → result is a FlopscopeArray, not a SymmetricTensor."""
    A = flops.symmetrize(
        fnp.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.warns(SymmetryLossWarning):
            d = fnp.diagonal(A)
    assert not isinstance(d, flops.SymmetricTensor), (
        f"got {type(d).__name__} with empty symmetry; expected FlopscopeArray"
    )
    assert isinstance(d, fnp.ndarray)


def test_no_symmetry_downgrades_via_full_reduction():
    """Reducing all symmetric axes should downgrade to FlopscopeArray (or scalar)."""
    A = flops.symmetrize(
        fnp.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.warns(SymmetryLossWarning):
            s = A.sum()  # scalar — but if returned as array, must be FlopscopeArray
    assert not isinstance(s, flops.SymmetricTensor)


# ----- Scalar __getitem__ on SymmetricTensor returns Python scalar -----


def test_symmetric_getitem_scalar_returns_python_scalar():
    """``A[0, 0]`` on a 2D SymmetricTensor returns a single element. It
    must come back as a Python scalar (``np.isscalar`` true), not a 0-d
    ndarray, to match plain ndarray semantics."""
    A = flops.symmetrize(
        fnp.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)):
        v = A[0, 0]
    assert np.isscalar(v), f"expected Python scalar, got {type(v).__name__}"


def test_symmetric_getitem_full_reduction_via_indexing_returns_whest_array():
    """``A[0]`` on a 2D symmetric (axes=(0,1)) collapses one axis; the
    remaining 1D slice has no surviving 2-axis group → FlopscopeArray."""
    A = flops.symmetrize(
        fnp.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.warns(SymmetryLossWarning):
            row = A[0]
    assert not isinstance(row, flops.SymmetricTensor)
    assert isinstance(row, fnp.ndarray)
    assert row.shape == (4,)


# ----- Verbatim issue reproducers -----


def test_issue_58_reproducer():
    """Verbatim from https://github.com/AIcrowd/flopscope/issues/58
    (translated to the post-PR-#51 SymmetryGroup API)."""
    n = 10
    with flops.BudgetContext(flop_budget=int(1e20)) as bc:
        with flops.namespace("init"):
            A = fnp.random.randn(n, n, n)
            A = flops.symmetrize(
                A, symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1, 2))
            )
        with flops.namespace("sum1"):
            S1 = A.sum(axis=0)
        with flops.namespace("sum2"):
            S2 = fnp.sum(A, axis=0)

    by_ns = bc.summary_dict(by_namespace=True)["by_namespace"]
    flops_sum1 = by_ns.get("sum1", {}).get("flops_used", 0)
    flops_sum2 = by_ns.get("sum2", {}).get("flops_used", 0)
    assert flops_sum1 > 0, "sum1 (method) bypassed tracking"
    assert flops_sum1 == flops_sum2, (
        f"method/function FLOP mismatch: {flops_sum1} vs {flops_sum2}"
    )
    assert S1.symmetry == S2.symmetry  # pyright: ignore[reportAttributeAccessIssue]
    assert S1.symmetry is not None  # pyright: ignore[reportAttributeAccessIssue] # surviving symmetry


def test_issue_38_reproducer():
    """Verbatim from https://github.com/AIcrowd/flopscope/issues/38."""
    n = 10
    with flops.BudgetContext(flop_budget=int(1e20)) as bc:
        with flops.namespace("init"):
            A = fnp.random.randn(n, n)
            A = flops.symmetrize(A, symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)))
        with flops.namespace("mult1"):
            B = A * A
        with flops.namespace("mult2"):
            B2 = fnp.multiply(A, A)

    by_ns = bc.summary_dict(by_namespace=True)["by_namespace"]
    flops_m1 = by_ns.get("mult1", {}).get("flops_used", 0)
    flops_m2 = by_ns.get("mult2", {}).get("flops_used", 0)
    assert flops_m1 == flops_m2 > 0


def test_issue_62_reproducer():
    """https://github.com/AIcrowd/flopscope/issues/62 — A SymmetricTensor with no
    surviving symmetry should be a FlopscopeArray, not an empty SymmetricTensor."""
    A = flops.symmetrize(
        fnp.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.warns(SymmetryLossWarning):
            d = fnp.diagonal(A)
    assert not isinstance(d, flops.SymmetricTensor)
    assert isinstance(d, fnp.ndarray)
