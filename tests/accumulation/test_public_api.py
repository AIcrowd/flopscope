"""Tests for the public einsum_accumulation_cost surface."""

import numpy as np

import flopscope as fps


def test_einsum_accumulation_cost_is_publicly_exposed():
    assert hasattr(fps, "einsum_accumulation_cost")
    assert hasattr(fps, "AccumulationCost")
    assert hasattr(fps, "ComponentCost")
    assert hasattr(fps, "RegimeStep")


def test_einsum_accumulation_cost_simple_matmul():
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    cost = fps.einsum_accumulation_cost("ij,jk->ik", A, B)
    assert isinstance(cost, fps.AccumulationCost)
    assert cost.dense_baseline == 27
    # No symmetry → trivial components (i, j, k each as singletons).
    # ∏M = 27, ∏α = 27, ∏num_output_orbits = 3·1·3 = 9
    # total = (k-1)·M + α − ∏num_output_orbits = 27 + 27 − 9 = 45
    # (matches the textbook 2·n³ − n² for n=3.)
    assert cost.total == 45


def test_einsum_accumulation_cost_does_not_require_budget_context():
    """Pure inspection — no BudgetContext needed."""
    A = np.zeros((3, 3))
    cost = fps.einsum_accumulation_cost("ii->", A)
    assert cost.total > 0


def test_einsum_accumulation_cost_accepts_partition_budget_override():
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    cost = fps.einsum_accumulation_cost("ij,jk->ik", A, B, partition_budget=50_000)
    assert cost.total == 45  # see test_einsum_accumulation_cost_simple_matmul


def test_einsum_accumulation_cost_with_symmetric_input():
    A = np.zeros((4, 4))
    A_sym = fps.as_symmetric(A, symmetry=(0, 1))
    cost = fps.einsum_accumulation_cost("ij,j->i", A_sym, np.zeros(4))
    assert cost.fallback_used is False
    assert cost.total > 0
