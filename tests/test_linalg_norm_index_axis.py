"""Scalar norm axes accept the integer index protocol, like NumPy."""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope.errors import BudgetExhaustedError


class IndexAxis:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def _call(name, array, axis, **kwargs):
    with flops.budget(10**7, quiet=True) as budget:
        tracked = fnp.asarray(array)
        before = budget.flops_used
        value = getattr(fnp.linalg, name)(tracked, axis=axis, **kwargs)
        charge = budget.flops_used - before
    return np.asarray(value), charge


@pytest.mark.parametrize("name", ["norm", "vector_norm"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
@pytest.mark.parametrize(
    "axis", [np.int64(0), np.int32(-1), np.uint32(0), np.array(0), IndexAxis(-1)]
)
def test_index_axis_values_and_billing(name, dtype, axis):
    array = np.arange(12, dtype=np.float64).astype(dtype).reshape(3, 4)
    expected = getattr(np.linalg, name)(array, axis=axis, keepdims=True)
    actual, charge = _call(name, array, axis, keepdims=True)
    canonical, canonical_charge = _call(name, array, axis.__index__(), keepdims=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, canonical)
    assert charge == canonical_charge


@pytest.mark.parametrize("name", ["norm", "vector_norm"])
@pytest.mark.parametrize("axis", [None, 0, -1, (np.int64(0),)])
def test_existing_axis_forms_unchanged(name, axis):
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    actual, _ = _call(name, array, axis)
    np.testing.assert_array_equal(actual, getattr(np.linalg, name)(array, axis=axis))


@pytest.mark.parametrize("name", ["norm", "vector_norm"])
def test_index_axis_general_p_keeps_group_charge(name):
    array = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    actual, charge = _call(name, array, np.int64(-1), ord=3)
    canonical, canonical_charge = _call(name, array, -1, ord=3)
    np.testing.assert_array_equal(actual, canonical)
    assert charge == canonical_charge


@pytest.mark.parametrize("name", ["norm", "vector_norm"])
def test_index_axis_still_checks_budget_before_execution(name):
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    _, charge = _call(name, array, 0)
    with flops.budget(charge - 1, quiet=True) as budget:
        with pytest.raises(BudgetExhaustedError):
            getattr(fnp.linalg, name)(array, axis=np.int64(0))
        assert budget.flops_used == 0


@pytest.mark.parametrize("name", ["norm", "vector_norm"])
@pytest.mark.parametrize("axis", [np.int64(2), np.int32(-3)])
def test_invalid_index_axis_preserves_numpy_error_without_charge(name, axis):
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    with pytest.raises((ValueError, IndexError)):
        getattr(np.linalg, name)(array, axis=axis)
    with flops.budget(10**7, quiet=True) as budget:
        with pytest.raises((ValueError, IndexError)):
            getattr(fnp.linalg, name)(array, axis=axis)
        assert budget.flops_used == 0
