"""The vector axis must be removed before broadcasting cross-product batches."""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope.errors import BudgetExhaustedError


def _run(fn, a, b, axis):
    with flops.budget(10**8, quiet=True) as budget:
        tracked_a, tracked_b = fnp.asarray(a), fnp.asarray(b)
        before = budget.flops_used
        result = fn(tracked_a, tracked_b, axis=axis)
        charged = budget.flops_used - before
    return np.asarray(result), charged


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int32, np.complex64, np.complex128]
)
@pytest.mark.parametrize(
    "shape_a,shape_b,axis",
    [
        (
            (3, 4),
            (3,),
            0,
        ),  # valid after moving vector axis, raw shapes cannot broadcast
        ((3, 2), (3,), 0),
        ((3, 1), (3,), 0),  # raw shapes broadcast to (3,3), actual result is only (3,1)
        ((3, 1, 2), (3,), 0),
        ((2, 3, 4), (3, 4), -2),
        ((4, 3), (3,), -1),  # unchanged default-axis control
    ],
)
def test_moved_vector_axis_values_and_billing(dtype, shape_a, shape_b, axis):
    a = np.arange(np.prod(shape_a), dtype=np.float64).astype(dtype).reshape(shape_a)
    b = np.arange(np.prod(shape_b), dtype=np.float64).astype(dtype).reshape(shape_b)
    expected = np.linalg.cross(a, b, axis=axis)
    result, charge = _run(fnp.linalg.cross, a, b, axis)
    canonical, canonical_charge = _run(fnp.cross, a, b, axis)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result, canonical)
    assert charge == canonical_charge


@pytest.mark.parametrize(
    "shape_a,shape_b,axis",
    [
        (
            (2,),
            (2,),
            -1,
        ),  # numpy.cross allows2-vectors; linalg.cross must still reject them
        ((3, 4), (3, 5), 0),  # incompatible non-vector dimensions
        ((3,), (3,), 1),  # invalid axis
    ],
)
def test_invalid_cross_rejected_before_charge(shape_a, shape_b, axis):
    a, b = np.ones(shape_a), np.ones(shape_b)
    with pytest.raises((ValueError, IndexError)):
        np.linalg.cross(a, b, axis=axis)
    with flops.budget(10**8, quiet=True) as budget:
        tracked_a, tracked_b = fnp.asarray(a), fnp.asarray(b)
        before = budget.flops_used
        with pytest.raises((ValueError, IndexError)):
            fnp.linalg.cross(tracked_a, tracked_b, axis=axis)
        assert budget.flops_used == before


def test_corrected_singleton_batch_fits_its_dense_budget():
    a = np.arange(3, dtype=np.float64).reshape(3, 1)
    b = np.array([1.0, 2.0, 3.0])
    # Respect the configured dtype weights (the suite uses unit test weights).
    # Raw-shape broadcasting used to bill three times the canonical charge.
    _, expected_charge = _run(fnp.cross, a, b, 0)
    with flops.budget(expected_charge, quiet=True) as budget:
        actual = fnp.linalg.cross(a, b, axis=0)
        assert budget.flops_used == expected_charge
    np.testing.assert_array_equal(np.asarray(actual), np.linalg.cross(a, b, axis=0))


def test_moved_axis_still_enforces_budget_upfront():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.array([1.0, 2.0, 3.0])
    _, expected_charge = _run(fnp.cross, a, b, 0)
    with flops.budget(expected_charge - 1, quiet=True) as budget:
        with pytest.raises(BudgetExhaustedError):
            fnp.linalg.cross(a, b, axis=0)
        assert budget.flops_used == 0
