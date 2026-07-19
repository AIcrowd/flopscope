"""Regression tests for issue #145: zero-sized contractions must not
produce negative FLOP charges."""

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp


def test_empty_inner_matmul_costs_zero():
    cost = flops.einsum_accumulation_cost(
        "ij,jk->ik", np.empty((2, 0)), np.empty((0, 3))
    )
    assert cost.total == 0


def test_empty_scalar_dot_costs_zero():
    cost = flops.einsum_accumulation_cost("i,i->", np.empty((0,)), np.empty((0,)))
    assert cost.total == 0


def test_empty_reduction_costs_zero():
    assert flops.reduction_accumulation_cost(np.empty((0,))).total == 0
    assert flops.reduction_accumulation_cost(np.empty((0, 5)), axis=0).total == 0


def test_empty_matmul_does_not_refund_budget():
    with flops.BudgetContext(flop_budget=10, quiet=True) as budget:
        result = fnp.matmul(np.empty((2, 0)), np.empty((0, 3)))
    assert result.shape == (2, 3)
    assert budget.flops_used == 0
    assert budget.flops_remaining == 10


def test_empty_reduction_preserves_extra_ops():
    cost = flops.reduction_accumulation_cost(np.empty((0,)), extra_ops=3)
    assert cost.total == 3


def test_empty_symmetric_contraction_costs_zero():
    # The k>=3 contraction path selects the cheapest of several cost
    # candidates; the joint-burnside candidate applies the same free
    # initial-copy correction and must also charge zero for an empty
    # contracted dimension with a symmetric output pair.
    left = flops.as_symmetric(fnp.zeros((4, 4, 0)), symmetry=(0, 1))
    cost = flops.einsum_accumulation_cost(
        "ijk,kl,lm->ijm", left, fnp.zeros((0, 5)), fnp.zeros((5, 2))
    )
    assert cost.total == 0


def test_empty_symmetric_contraction_does_not_refund_budget():
    left = flops.as_symmetric(fnp.zeros((4, 4, 0)), symmetry=(0, 1))
    with flops.BudgetContext(flop_budget=10, quiet=True) as budget:
        result = fnp.einsum(
            "ijk,kl,lm->ijm", left, fnp.zeros((0, 5)), fnp.zeros((5, 2))
        )
    assert result.shape == (4, 4, 2)
    assert budget.flops_used == 0
    assert budget.flops_remaining == 10
