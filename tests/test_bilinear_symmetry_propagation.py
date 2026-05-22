"""Tests for issue #59 — bilinear wrappers must propagate operand symmetry
and apply joint-operand cost savings via the einsum cost path."""

from __future__ import annotations

import numpy as np

import flopscope
from flopscope import SymmetryGroup, BudgetContext


# --- _resolve_cost_and_output_symmetry helper (Task 1) -------------------

def test_helper_returns_costinfo_for_symmetric_matmul():
    """Helper must compute symmetry-aware cost and infer output symmetry for A @ A."""
    from flopscope._einsum import _resolve_cost_and_output_symmetry

    n = 10
    A_raw = np.random.RandomState(0).randn(n, n)
    A = flopscope.symmetrize(A_raw, symmetry=SymmetryGroup.symmetric(axes=(0, 1)))
    with BudgetContext(flop_budget=int(1e20)):
        info = _resolve_cost_and_output_symmetry("ij,jk->ik", A, A)
    assert info.accumulation.total > 0
    assert info.output_symmetry is not None, (
        "A @ A with symmetric A must infer output symmetry"
    )
    assert info.canonical_subscripts == "ij,jk->ik"
    assert info.shapes == ((n, n), (n, n))


def test_helper_returns_none_symmetry_for_distinct_matmul():
    """For two distinct symmetric matrices (not aliased), output symmetry is None."""
    from flopscope._einsum import _resolve_cost_and_output_symmetry

    n = 6
    rs = np.random.RandomState(0)
    A_raw = rs.randn(n, n)
    B_raw = rs.randn(n, n)
    A = flopscope.symmetrize(A_raw, symmetry=SymmetryGroup.symmetric(axes=(0, 1)))
    B = flopscope.symmetrize(B_raw, symmetry=SymmetryGroup.symmetric(axes=(0, 1)))
    with BudgetContext(flop_budget=int(1e20)):
        info = _resolve_cost_and_output_symmetry("ij,jk->ik", A, B)
    assert info.accumulation.total > 0
    assert info.output_symmetry is None, (
        "matmul(A, B) with distinct A, B (no identity-alias) must not "
        "infer joint output symmetry"
    )
