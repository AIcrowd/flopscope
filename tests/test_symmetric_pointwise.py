"""Tests for symmetry-aware pointwise operations."""

import numpy

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._budget import BudgetContext
from flopscope._symmetric import SymmetricTensor, as_symmetric


class TestUnarySymmetry:
    def test_exp_symmetric_cost(self):
        data = numpy.eye(10)
        S = as_symmetric(data, symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = fnp.exp(S)
            assert budget.flops_used == 55  # 10*11/2

    def test_exp_symmetric_returns_symmetric(self):
        data = numpy.eye(4)
        S = as_symmetric(data, symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)))
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = fnp.exp(S)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry.axes == (0, 1)

    def test_log_symmetric_cost(self):
        data = numpy.eye(5) + 1
        S = as_symmetric(data, symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            fnp.log(S)
            assert budget.flops_used == 15

    def test_plain_array_unchanged(self):
        data = numpy.ones((10, 10))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            fnp.exp(data)
            assert budget.flops_used == 100


class TestBinarySymmetry:
    def test_add_both_symmetric_same_axes(self):
        A = as_symmetric(
            numpy.eye(5), symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1))
        )
        B = as_symmetric(
            numpy.eye(5) * 2,
            symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
        )
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = fnp.add(A, B)
            assert budget.flops_used == 15
            assert isinstance(result, SymmetricTensor)

    def test_add_different_dims_no_symmetry(self):
        A = as_symmetric(
            numpy.eye(5), symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1))
        )
        B = numpy.ones((5, 5))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = fnp.add(A, B)
            assert budget.flops_used == 25
            assert not isinstance(result, SymmetricTensor)

    def test_multiply_scalar_preserves_symmetry(self):
        A = as_symmetric(
            numpy.eye(5), symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1))
        )
        scalar = numpy.asarray(3.0)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = fnp.multiply(A, scalar)
            assert isinstance(result, SymmetricTensor)

    def test_dunder_with_dense_operand_matches_function_and_drops_symmetry(self):
        A = as_symmetric(
            numpy.eye(5), symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1))
        )
        B = numpy.arange(25.0).reshape(5, 5)
        with BudgetContext(flop_budget=10**6, quiet=True) as function_budget:
            expected = fnp.multiply(A, B)
        with BudgetContext(flop_budget=10**6, quiet=True) as dunder_budget:
            result = A * B
        assert function_budget.flops_used == dunder_budget.flops_used
        assert not isinstance(expected, SymmetricTensor)
        assert not isinstance(result, SymmetricTensor)
        assert numpy.allclose(result, expected)


class TestReductionSymmetry:
    def test_sum_symmetric_cost(self):
        # Updated to orbit-mapping count (was n(n+1)/2 unique-element count).
        # Symmetric (10,10) has 55 orbits; full reduction: 55-1=54 additions.
        data = numpy.eye(10)
        S = as_symmetric(data, symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            fnp.sum(S)
            assert budget.flops_used == 54

    def test_sum_returns_plain(self):
        data = numpy.eye(4)
        S = as_symmetric(data, symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)))
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = fnp.sum(S)
            assert not isinstance(result, SymmetricTensor)
