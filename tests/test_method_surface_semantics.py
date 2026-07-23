"""Semantic guarantees of FlopscopeArray method overrides.

The billed method surface (``x.copy()``, ``x.choose(...)``) must keep numpy's
native METHOD semantics, not just numpy's values — NumPy's own compat suite
(``tests/numpy_compat/`` + ``--pyargs numpy._core.tests.test_numeric``) caught
two regressions the pure-billing tests missed:

* ``np.require(a, requirements=['OWNDATA'])`` satisfies the 'O' flag via
  ``arr.copy()`` and then checks ``flags['OWNDATA']`` on what it gets back — a
  delegating override that wraps a base buffer as a subclass VIEW can never
  satisfy it.
* ``ndarray.choose(choices, out=arr)`` passes ``out`` as a KEYWORD; the
  fnp.choose wrapper must strip it like the positional arrays or the
  numpy-entry guard trips.
"""

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp


def test_method_copy_owns_its_buffer():
    x = fnp.asarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    with flops.BudgetContext(flop_budget=10**9, quiet=True):
        y = x.copy()
    assert y.flags["OWNDATA"]
    assert type(y).__name__ == type(x).__name__
    np.testing.assert_array_equal(np.asarray(y), np.asarray(x))


def test_method_copy_bills_like_fnp_copy():
    x = fnp.asarray(np.arange(1000, dtype=np.float64))
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as b_method:
        x.copy()
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as b_free:
        fnp.copy(x)
    assert b_method.flops_used == b_free.flops_used


def test_require_owndata_satisfied():
    # Non-square: a square zeros() is auto-detected as a SymmetricTensor,
    # whose copy() is a separate (single-bill) surface — the plain
    # FlopscopeArray path is what numpy's TestRequire exercises.
    arr = fnp.zeros((2, 3))
    with flops.BudgetContext(flop_budget=10**9, quiet=True):
        b = fnp.require(arr, None, ["OWNDATA"])
    assert b.flags["OWNDATA"]


def test_method_choose_accepts_out_keyword():
    selector = fnp.asarray(np.array([0, 1, 0, 1]))
    choices = (
        fnp.asarray(np.zeros(4, dtype=np.float64)),
        fnp.asarray(np.ones(4, dtype=np.float64)),
    )
    out = fnp.asarray(np.empty(4, dtype=np.float64))
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as bc:
        result = selector.choose(choices, out=out)
    np.testing.assert_array_equal(np.asarray(out), [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(np.asarray(result), [0.0, 1.0, 0.0, 1.0])
    # Suite conftest runs unit weights/rates: billed == flop_cost == numel(out).
    assert bc.flops_used == 4


def test_method_choose_out_production_billing():
    from tests.test_dtype_cost import _billed_with_production_rates

    selector = fnp.asarray(np.array([0, 1, 0, 1]))
    choices = (
        fnp.asarray(np.zeros(4, dtype=np.float64)),
        fnp.asarray(np.ones(4, dtype=np.float64)),
    )
    out = fnp.asarray(np.empty(4, dtype=np.float64))
    billed, _ = _billed_with_production_rates(lambda: selector.choose(choices, out=out))
    # gather tier: numel(out) × weight 4.0 × float64 rate 2
    assert billed == 4 * 4 * 2


def test_creation_rejected_before_allocation():
    """Over-budget creation must be rejected by FLOP accounting BEFORE numpy
    allocates the buffer (a tiny budget must never OOM on fnp.ones((1e12,)))."""
    import pytest

    from flopscope.errors import BudgetExhaustedError

    with flops.BudgetContext(flop_budget=10, quiet=True):
        with pytest.raises(BudgetExhaustedError):
            fnp.ones((10**12,))
        with pytest.raises(BudgetExhaustedError):
            fnp.full((10**12,), 7.0)
        with pytest.raises(BudgetExhaustedError):
            fnp.ones_like(fnp.zeros(3), shape=(10**12,))


def test_linspace_family_bills_produced_dtype():
    """With dtype omitted, numpy produces float64 samples from integer
    endpoints (and geomspace promotes even float32) — billing must follow the
    PRODUCED dtype, not the inputs."""
    from tests.test_dtype_cost import _billed_with_production_rates

    billed, _ = _billed_with_production_rates(
        lambda: fnp.linspace(np.int32(0), np.int32(8), 5)
    )
    assert billed == 2 * 5 * 2  # cost 2*num × float64 rate 2

    billed, _ = _billed_with_production_rates(
        lambda: fnp.linspace(np.float32(0), np.float32(8), 5)
    )
    assert billed == 2 * 5  # float32 stays rate 1

    billed, _ = _billed_with_production_rates(
        lambda: fnp.logspace(np.int32(0), np.int32(3), 4)
    )
    assert billed == 4 * 16 * 2  # num × weight 16 × float64 rate 2

    billed, _ = _billed_with_production_rates(
        lambda: fnp.geomspace(np.float32(1), np.float32(8), 4)
    )
    assert billed == 4 * 16 * 2  # geomspace promotes float32 -> float64
