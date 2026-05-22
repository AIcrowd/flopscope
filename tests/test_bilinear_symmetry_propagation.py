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


# --- _flops.einsum_cost side-fix (Task 3) ---------------------------------

def test_einsum_cost_forwards_identity_pattern():
    """Public-introspection einsum_cost must respect identity_pattern to
    detect A @ A joint savings. Before the fix it always passed
    identity_pattern=None, so cost matched A @ B (distinct ops)."""
    from flopscope._flops import einsum_cost

    n = 10
    sym = SymmetryGroup.symmetric(axes=(0, 1))
    shape = (n, n)
    # Cost with identity_pattern indicating both positions share one operand.
    cost_aliased = einsum_cost(
        "ij,jk->ik",
        shapes=[shape, shape],
        operand_symmetries=[sym, sym],
        identity_pattern=((0, 1),),
    )
    # Cost without the alias (treats operands as independent).
    cost_distinct = einsum_cost(
        "ij,jk->ik",
        shapes=[shape, shape],
        operand_symmetries=[sym, sym],
        identity_pattern=None,
    )
    assert cost_aliased <= cost_distinct, (
        "Aliased A@A cost should be no greater than two-distinct-operand cost"
    )
