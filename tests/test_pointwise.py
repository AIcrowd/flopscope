"""Tests for counted pointwise and reduction operations."""

import numpy
import pytest

from flopscope._budget import BudgetContext
from flopscope._pointwise import (
    add,
    argmax,
    clip,
    cumsum,
    dot,
    exp,
    log,
    matmul,
    maximum,
    mean,
    sqrt,
    std,
    sum,
)


def test_exp_result():
    x = numpy.array([0.0, 1.0, 2.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(exp(x), numpy.exp(x))


def test_exp_flop_cost():
    x = numpy.ones((10, 20))
    with BudgetContext(flop_budget=10**6) as budget:
        exp(x)
        assert budget.flops_used == 200


def test_sqrt_result():
    x = numpy.array([1.0, 4.0, 9.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(sqrt(x), numpy.sqrt(x))


def test_add_result():
    a = numpy.array([1.0, 2.0])
    b = numpy.array([3.0, 4.0])
    expected = a + b
    with BudgetContext(flop_budget=10**6) as budget:
        result = add(a, b)
        assert budget.flops_used == 2
    assert numpy.allclose(result, expected)


def test_add_broadcast_cost():
    a = numpy.ones((3, 4))
    b = numpy.ones((4,))
    with BudgetContext(flop_budget=10**6) as budget:
        result = add(a, b)
        assert result.shape == (3, 4)
        assert budget.flops_used == 12


def test_maximum_result():
    a = numpy.array([1.0, 5.0, 3.0])
    b = numpy.array([2.0, 4.0, 6.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(maximum(a, b), numpy.maximum(a, b))


def test_clip_result():
    x = numpy.array([-1.0, 0.5, 2.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(clip(x, 0.0, 1.0), numpy.clip(x, 0.0, 1.0))


def test_sum_full():
    x = numpy.ones((5, 3))
    with BudgetContext(flop_budget=10**6) as budget:
        result = sum(x)
        # Updated for orbit-mapping cost model (PR #91 Task 7).
        # prod(shape) - 1 = 15 - 1 = 14 additions for a full reduction.
        assert budget.flops_used == 14
    assert float(result) == 15.0


def test_sum_axis():
    x = numpy.ones((5, 3))
    with BudgetContext(flop_budget=10**6) as budget:
        result = sum(x, axis=0)
        assert result.shape == (3,)
        # Updated for orbit-mapping cost model (PR #91 Task 7).
        # Reduces 5->1 for each of 3 cols: 3 * (5-1) = 12 additions.
        assert budget.flops_used == 12


def test_mean_cost():
    x = numpy.ones((10, 20))
    with BudgetContext(flop_budget=10**6) as budget:
        mean(x, axis=0)
        # Task 9: mean charges sum_cost + num_output_orbits divides.
        # Reduces 10->1 for each of 20 cols: 20 * (10-1) = 180 additions.
        # Output shape is (20,) dense → 20 divides.  Total = 200.
        assert budget.flops_used == 200


def test_std_cost():
    x = numpy.ones((10, 20))
    with BudgetContext(flop_budget=10**6) as budget:
        std(x, axis=0)
        # Updated for orbit-mapping cost model (PR #91 Task 7).
        # Reduces 10->1 for each of 20 cols: 20 * (10-1) = 180 additions.
        assert budget.flops_used == 180


def test_argmax_result():
    x = numpy.array([1.0, 5.0, 3.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert argmax(x) == 1


def test_cumsum():
    x = numpy.array([1.0, 2.0, 3.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(cumsum(x), [1, 3, 6])


def test_dot_result():
    a = numpy.ones((3, 4))
    b = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        result = dot(a, b)
        assert numpy.allclose(result, numpy.dot(a, b))
        # direct-event model with off-by-one correction:
        # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
        # = 60 + 60 - 15 = 105 (textbook 2n^3 - n^2 form).
        # First cell of each output orbit is a free copy.
        assert budget.flops_used == 105


def test_matmul_result():
    a = numpy.ones((3, 4))
    b = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(matmul(a, b), numpy.matmul(a, b))


def test_counted_op_outside_context():
    # Operations now auto-activate the global default budget instead of raising
    result = exp(numpy.ones((3,)))
    assert result.shape == (3,)


def test_nan_warning():
    import flopscope as flops
    from flopscope._config import get_setting

    original = get_setting("check_nan_inf")
    try:
        flops.configure(check_nan_inf=True)
        with BudgetContext(flop_budget=10**6):
            with pytest.warns(match="NaN"):
                log(numpy.array([0.0]))
    finally:
        flops.configure(check_nan_inf=original)
