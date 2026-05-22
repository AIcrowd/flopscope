"""Tests for issue #59 — bilinear wrappers must propagate operand symmetry
and apply joint-operand cost savings via the einsum cost path."""

from __future__ import annotations

import numpy as np

import flopscope
import flopscope.numpy as fnp
from flopscope import SymmetryGroup, SymmetricTensor, BudgetContext


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


# --- matmul (Task 4 — closes #59) ----------------------------------------

def _flops(bc, ns):
    bn = bc.summary_dict(by_namespace=True)["by_namespace"]
    return bn.get(ns, {}).get("flops_used", 0)


def test_matmul_sym_self_matches_einsum():
    """A @ A with symmetric A: cost and output type match einsum."""
    n = 10
    sym = SymmetryGroup.symmetric(axes=(0, 1))
    rs = np.random.RandomState(0)
    with BudgetContext(flop_budget=int(1e20)) as bc:
        A_raw = fnp.array(rs.randn(n, n))
        A = flopscope.symmetrize(A_raw, symmetry=sym)
        with flopscope.namespace("mm"):
            X = A @ A
        with flopscope.namespace("ein"):
            Y = fnp.einsum("ij,jk->ik", A, A)
    assert _flops(bc, "mm") == _flops(bc, "ein"), (
        f"matmul cost {_flops(bc, 'mm')} must equal einsum cost "
        f"{_flops(bc, 'ein')} for A @ A with symmetric A"
    )
    assert isinstance(X, SymmetricTensor), f"A @ A expected SymmetricTensor, got {type(X).__name__}"
    assert isinstance(Y, SymmetricTensor)


def test_matmul_two_distinct_sym_not_symmetric():
    """matmul(A, B) with distinct symmetric A, B does NOT yield SymmetricTensor."""
    n = 6
    sym = SymmetryGroup.symmetric(axes=(0, 1))
    rs = np.random.RandomState(0)
    with BudgetContext(flop_budget=int(1e20)):
        A_raw = fnp.array(rs.randn(n, n))
        B_raw = fnp.array(rs.randn(n, n))
        A = flopscope.symmetrize(A_raw, symmetry=sym)
        B = flopscope.symmetrize(B_raw, symmetry=sym)
        X = A @ B
    assert not isinstance(X, SymmetricTensor), (
        "matmul(A, B) with distinct symmetric A, B must NOT be wrapped as "
        "SymmetricTensor — AB is symmetric only if A and B commute, which "
        "is non-generic."
    )


def test_matmul_at_transpose_not_detected():
    """A @ A.T with non-symmetric A: oracle correctly misses the alias.

    This is the documented limitation — .T returns a view (different id),
    so the identity_pattern detector treats A and A.T as distinct."""
    n = 4
    rs = np.random.RandomState(0)
    with BudgetContext(flop_budget=int(1e20)):
        A = fnp.array(rs.randn(n, n))   # NOT symmetric
        X = A @ A.T
    assert not isinstance(X, SymmetricTensor), (
        "A @ A.T detection is out of scope; oracle uses Python id() which "
        "differs between A and A.T (view-aliasing). Documented limitation."
    )


def test_issue_59_reproducer():
    """Verbatim reproducer from issue #59."""
    n = 10
    sym = SymmetryGroup.symmetric(axes=(0, 1))
    with BudgetContext(flop_budget=int(1e20)) as bc:
        with flopscope.namespace("init"):
            A_raw = fnp.array(np.random.RandomState(0).randn(n, n))
            A = flopscope.symmetrize(A_raw, symmetry=sym)
        with flopscope.namespace("mm"):
            X = A @ A
        with flopscope.namespace("einsum"):
            Y = fnp.einsum("ij,jk->ik", A, A)
    assert _flops(bc, "mm") == _flops(bc, "einsum"), (
        f"Issue #59: mm flops ({_flops(bc, 'mm')}) must equal einsum flops "
        f"({_flops(bc, 'einsum')})"
    )
    assert isinstance(X, SymmetricTensor), (
        f"Issue #59: type(X) must be SymmetricTensor, got {type(X).__name__}"
    )
    assert isinstance(Y, SymmetricTensor)


# --- dot (Task 5) --------------------------------------------------------

def test_dot_sym_self_matches_einsum():
    """dot(A, A) with symmetric A: cost and output type match einsum."""
    n = 8
    sym = SymmetryGroup.symmetric(axes=(0, 1))
    rs = np.random.RandomState(0)
    with BudgetContext(flop_budget=int(1e20)) as bc:
        A_raw = fnp.array(rs.randn(n, n))
        A = flopscope.symmetrize(A_raw, symmetry=sym)
        with flopscope.namespace("dt"):
            X = fnp.dot(A, A)
        with flopscope.namespace("ein"):
            Y = fnp.einsum("ij,jk->ik", A, A)
    assert _flops(bc, "dt") == _flops(bc, "ein"), (
        f"dot cost {_flops(bc, 'dt')} != einsum cost {_flops(bc, 'ein')}"
    )
    assert isinstance(X, SymmetricTensor)
    assert isinstance(Y, SymmetricTensor)


# --- outer (Task 6) ------------------------------------------------------

def test_outer_sym_self_matches_einsum():
    """outer(v, v) cost and output type match einsum('i,j->ij', v, v)."""
    n = 12
    rs = np.random.RandomState(0)
    with BudgetContext(flop_budget=int(1e20)) as bc:
        v = fnp.array(rs.randn(n))
        with flopscope.namespace("ou"):
            X = fnp.outer(v, v)
        with flopscope.namespace("ein"):
            Y = fnp.einsum("i,j->ij", v, v)
    assert _flops(bc, "ou") == _flops(bc, "ein"), (
        f"outer cost {_flops(bc, 'ou')} != einsum cost {_flops(bc, 'ein')}"
    )
    assert isinstance(X, SymmetricTensor)
    assert isinstance(Y, SymmetricTensor)


# --- inner (Task 7) ------------------------------------------------------

def test_inner_sym_self_matches_einsum_1d():
    """inner(v, v) cost matches einsum('i,i->', v, v)."""
    n = 8
    rs = np.random.RandomState(0)
    with BudgetContext(flop_budget=int(1e20)) as bc:
        v = fnp.array(rs.randn(n))
        with flopscope.namespace("in"):
            x = fnp.inner(v, v)
        with flopscope.namespace("ein"):
            y = fnp.einsum("i,i->", v, v)
    assert _flops(bc, "in") == _flops(bc, "ein")


# --- vdot (Task 8) -------------------------------------------------------

def test_vdot_sym_self_matches_einsum():
    """vdot(v, v) cost matches einsum('i,i->', v, v)."""
    n = 8
    rs = np.random.RandomState(0)
    with BudgetContext(flop_budget=int(1e20)) as bc:
        v = fnp.array(rs.randn(n))
        with flopscope.namespace("vd"):
            x = fnp.vdot(v, v)
        with flopscope.namespace("ein"):
            y = fnp.einsum("i,i->", v, v)
    assert _flops(bc, "vd") == _flops(bc, "ein")
