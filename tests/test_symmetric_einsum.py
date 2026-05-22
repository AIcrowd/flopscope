"""Tests for symmetry-aware einsum."""

import numpy

import flopscope as flops
from flopscope._budget import BudgetContext
from flopscope._einsum import einsum
from flopscope._symmetric import SymmetricTensor, as_symmetric


class TestEinsumSymmetricInput:
    def test_symmetric_input_reduces_cost(self):
        S = as_symmetric(
            numpy.ones((10, 10, 5)),
            symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
        )
        v = numpy.ones(5)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            einsum("ijk,k->ij", S, v)
            # new direct-event model: total=550, gaming bound=num_terms*dense=2*500=1000
            dense_gaming_bound = 2 * 10 * 10 * 5 * 2  # 2 operands * dense_baseline
            assert budget.flops_used < dense_gaming_bound
            assert budget.flops_used > 0

    def test_plain_input_unchanged(self):
        A = numpy.eye(10)
        v = numpy.ones(10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            einsum("ij,j->i", A, v)
            # direct-event model with off-by-one correction:
            # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
            # = 100 + 100 - 10 = 190. First cell of each output orbit is free.
            assert budget.flops_used == 190


class TestEinsumSymmetricOutput:
    def test_symmetry_returns_symmetric_tensor(self):
        X = numpy.ones((5, 10))
        target = flops.SymmetryGroup.symmetric(axes=(0, 1))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetry=target)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == target

    def test_without_symmetry_returns_plain(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij,jk->ik", A, B)
            assert not isinstance(result, SymmetricTensor)


class TestEinsumSymmetryParam:
    def test_symmetry_param_returns_symmetric_tensor(self):
        X = numpy.ones((5, 10))
        g = flops.SymmetryGroup.symmetric(axes=(0, 1))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetry=g)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == g

    def test_symmetry_accepts_exact_group_shorthand(self):
        X = numpy.ones((5, 10))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetry=(0, 1))
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == flops.SymmetryGroup.symmetric(axes=(0, 1))


def test_total_never_exceeds_k_times_dense_baseline():
    """Even with declared symmetries, total <= k * dense_baseline always (gaming-resistance)."""
    import numpy as np

    import flopscope as fps

    A = np.zeros((4, 4, 4))
    A_sym = fps.as_symmetric(A, symmetry=(0, 1, 2))
    cost = fps.einsum_accumulation_cost("ijk,abc->ic", A_sym, A_sym)
    upper_bound = cost.num_terms * cost.dense_baseline
    assert cost.total <= upper_bound
