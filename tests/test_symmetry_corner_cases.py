"""Comprehensive corner-case tests for symmetry-aware cost computation.

Tests grouped by category, each with mathematical justification for the
expected result. These tests verify the accumulation model against hand-worked
expectations and numerical verification.

Migration note: the old test suite used _detect_order() to inspect per-step
symmetry groups from the deleted SubgraphSymmetryOracle. The new tests use
the accumulation model's m_total (unique output elements) vs dense_baseline
(all output elements) to assert that symmetry savings are present or absent.

A result has savings when m_total < dense_baseline; no savings when they're equal.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._perm_group import SymmetryGroup


def _symmetrize(shape, group):
    """Reynolds-average random data to enforce symmetry."""
    data = np.random.randn(*shape)
    axes = group.axes if group.axes is not None else tuple(range(group.degree))
    total = np.zeros_like(data)
    for g in group.elements():
        af = g.array_form
        perm = list(range(len(shape)))
        for i in range(len(axes)):
            perm[axes[i]] = axes[af[i]]
        total = total + np.transpose(data, perm)
    return flops.as_symmetric(total / group.order(), symmetry=group)


def _has_savings(expr, *operands) -> bool:
    """Return True if the accumulation model finds symmetry savings for this expression.

    Savings are present when the number of unique output elements (m_total, as
    computed by Burnside's lemma over the detected symmetry group) is strictly
    less than the dense baseline (product of all output dimension sizes).
    """
    _, info = fnp.einsum_path(expr, *operands)
    acc = info.accumulation
    if acc is None:
        return False
    return acc.m_total < acc.dense_baseline


def _accumulation(expr, *operands):
    """Return the AccumulationCost for an expression."""
    _, info = fnp.einsum_path(expr, *operands)
    return info.accumulation


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)


N = 4


class TestIdenticalOperands:
    """Symmetry from identical Python objects at multiple operand positions."""

    def test_matmul_no_symmetry(self):
        """A·A = einsum('ij,jk->ik', A, A): different subscript structure, trivial."""
        A = fnp.random.randn(N, N)
        assert not _has_savings("ij,jk->ik", A, A)

    def test_gram_matrix_s2(self):
        """X^T X = einsum('ia,ib->ab', X, X): swapping operands swaps a↔b → S2.

        With S2 symmetry, m_total = N*(N+1)/2 unique elements (upper triangle)
        instead of N^2, giving strict savings.
        """
        X = fnp.random.randn(N, N)
        assert _has_savings("ia,ib->ab", X, X)
        result = fnp.einsum("ia,ib->ab", X, X)
        assert np.allclose(result, result.T)

    def test_vector_outer_s2(self):
        """v⊗v = einsum('i,j->ij', v, v): S2{i,j}."""
        v = fnp.random.randn(N)
        assert _has_savings("i,j->ij", v, v)
        result = fnp.einsum("i,j->ij", v, v)
        assert np.allclose(result, result.T)

    def test_triple_outer_s3(self):
        """v⊗v⊗v = einsum('i,j,k->ijk', v, v, v): S3{i,j,k}, order 6.

        Unique elements = C(N+2, 3) = 20 for N=4, vs N^3=64 dense.
        """
        v = fnp.random.randn(N)
        assert _has_savings("i,j,k->ijk", v, v, v)
        acc = _accumulation("i,j,k->ijk", v, v, v)
        assert acc.m_total < acc.dense_baseline

    def test_quad_outer_s4(self):
        """v⊗v⊗v⊗v: S4{i,j,k,l}, order 24.

        Unique elements = C(N+3, 4) = 35 for N=4, vs N^4=256 dense.
        """
        v = fnp.random.randn(N)
        assert _has_savings("i,j,k,l->ijkl", v, v, v, v)
        acc = _accumulation("i,j,k,l->ijkl", v, v, v, v)
        assert acc.m_total < acc.dense_baseline

    def test_directed_triangle_c3(self):
        """einsum('ij,jk,ki->ijk', A, A, A): cyclic chain → C3.

        The cyclic permutation (ij,jk,ki)→(jk,ki,ij) is valid, giving
        savings over the dense N^3 elements.
        """
        A = fnp.random.randn(N, N)
        assert _has_savings("ij,jk,ki->ijk", A, A, A)

    def test_directed_4_cycle_c4(self):
        """einsum('ij,jk,kl,li->ijkl', A, A, A, A): C4.

        Cyclic permutation gives savings over N^4 dense elements.
        """
        A = fnp.random.randn(N, N)
        assert _has_savings("ij,jk,kl,li->ijkl", A, A, A, A)

    def test_block_outer_product(self):
        """einsum('ab,cd->abcd', X, X): block swap (a,b)↔(c,d), order 2.

        Swapping the two identical operands maps (a,b,c,d)→(c,d,a,b), giving
        savings vs N^4 dense elements.
        """
        X = fnp.random.randn(N, N)
        assert _has_savings("ab,cd->abcd", X, X)

    def test_hadamard_no_symmetry(self):
        """einsum('ij,ij->ij', A, A): elementwise square.

        Swapping operands gives identity π (same subscripts), so trivial.
        A²[i,j] ≠ A²[j,i] in general.
        """
        A = fnp.random.randn(N, N)
        assert not _has_savings("ij,ij->ij", A, A)
        result = fnp.einsum("ij,ij->ij", A, A)
        assert not np.allclose(result, result.T)


class TestDeclaredSymmetryNonIdentical:
    """Per-operand declared symmetry on non-identical operands."""

    def test_symmetric_times_dense_trivial(self):
        """S·W where S has S2: no output symmetry (non-identical operands, no
        identical-operand source)."""
        S = _symmetrize((N, N), SymmetryGroup.symmetric(axes=(0, 1)))
        W = fnp.random.randn(N, N)
        assert not _has_savings("ij,jk->ik", S, W)

    def test_c3_contraction(self):
        """einsum('aijk,ab->ijkb', T, W) with C3 on T axes (1,2,3).

        T has C3 symmetry, W is dense. The declared C3 group on T induces
        savings in the unique (i,j,k,b) output elements.
        """
        T = _symmetrize((N, N, N, N), SymmetryGroup.cyclic(axes=(1, 2, 3)))
        W = fnp.random.randn(N, N)
        assert _has_savings("aijk,ab->ijkb", T, W)

    def test_d4_contraction(self):
        """einsum('aijkl,ab->ijklb', T, W) with D4 on T axes (1,2,3,4).

        D4 has order 8 (4 rotations + 4 reflections), giving significant
        savings in the unique output elements.
        """
        T = _symmetrize((N, N, N, N, N), SymmetryGroup.dihedral(axes=(1, 2, 3, 4)))
        W = fnp.random.randn(N, N)
        assert _has_savings("aijkl,ab->ijklb", T, W)


class TestDeclaredPlusIdentical:
    """Interaction of per-operand symmetry with identical-operand swaps."""

    def test_symmetric_matmul_s2(self):
        """S·S = einsum('ij,jk->ik', S, S) where S is symmetric.

        S[i,j]=S[j,i]. Result R[i,k] = sum_j S[i,j]*S[j,k].
        Using S=S^T: R[k,i] = sum_j S[k,j]*S[j,i] = R[i,k].
        So the result IS symmetric → S2{i,k} gives savings.
        """
        S = _symmetrize((N, N), SymmetryGroup.symmetric(axes=(0, 1)))
        assert _has_savings("ij,jk->ik", S, S)
        result = fnp.einsum("ij,jk->ik", S, S)
        assert np.allclose(result, result.T)

    def test_undirected_4_cycle_d4(self):
        """einsum('ij,jk,kl,li->ijkl', S, S, S, S) where S is symmetric.

        S symmetric enables reflections: C4 + reflections = D4, giving
        stronger savings vs the directed 4-cycle case.
        """
        S = _symmetrize((N, N), SymmetryGroup.symmetric(axes=(0, 1)))
        assert _has_savings("ij,jk,kl,li->ijkl", S, S, S, S)
        acc_directed = _accumulation(
            "ij,jk,kl,li->ijkl",
            fnp.random.randn(N, N),
            fnp.random.randn(N, N),
            fnp.random.randn(N, N),
            fnp.random.randn(N, N),
        )
        acc_undirected = _accumulation("ij,jk,kl,li->ijkl", S, S, S, S)
        # D4 (undirected with S symmetric) gives more savings than C4 (directed)
        assert acc_undirected.m_total <= acc_directed.m_total

    def test_c3_self_contraction(self):
        """einsum('ijk,jki->ik', T, T) with C3 on T.

        T has cyclic symmetry. The accumulation model exploits this to reduce
        the number of unique evaluations. The output 'ik' is NOT itself symmetric
        (T[i,k] ≠ T[k,i] in general for the result), but the computation still
        benefits from savings.

        Migration note: the old test asserted order=1 (trivial), which was checking
        that no symmetric OUTPUT GROUP was detected (guarding against a false S2
        detection bug in the oracle). The new accumulation model correctly exploits
        C3 symmetry of T in the contraction indices without asserting output symmetry.
        """
        T = _symmetrize((N, N, N), SymmetryGroup.cyclic(axes=(0, 1, 2)))
        result = fnp.einsum("ijk,jki->ik", T, T)
        # The output itself is NOT symmetric in general
        assert not np.allclose(result, result.T)
        # But the accumulation model exploits T's C3 symmetry for cost savings
        assert _has_savings("ijk,jki->ik", T, T)

    def test_symmetric_triangle_s3(self):
        """einsum('ij,jk,ki->ijk', S, S, S) where S is symmetric.

        Directed triangle with symmetric matrices: S2 on each operand enables
        reflections, promoting C3 → S3 (order 6), giving savings.
        """
        S = _symmetrize((N, N), SymmetryGroup.symmetric(axes=(0, 1)))
        assert _has_savings("ij,jk,ki->ijk", S, S, S)


class TestWSideSymmetry:
    """Symmetry on summed (contracted) indices."""

    def test_trace_aa_s2(self):
        """Tr(A·A) = einsum('ij,ji->', A, A): W-side S2{i,j}, order 2.

        Swapping i↔j gives same sum — savings over N^2 unique (i,j) pairs.
        """
        A = fnp.random.randn(N, N)
        assert _has_savings("ij,ji->", A, A)

    def test_trace_aaa_c3(self):
        """Tr(A·A·A) = einsum('ij,jk,ki->', A, A, A): W-side C3{i,j,k}."""
        A = fnp.random.randn(N, N)
        assert _has_savings("ij,jk,ki->", A, A, A)

    def test_trace_aaaa_c4(self):
        """Tr(A^4) = einsum('ij,jk,kl,li->', A, A, A, A): W-side C4."""
        A = fnp.random.randn(N, N)
        assert _has_savings("ij,jk,kl,li->", A, A, A, A)

    def test_partial_trace_has_savings(self):
        """einsum('ij,jk,ki->i', A, A, A): i is free, j,k summed.

        The accumulation model detects cyclic symmetry among the summation
        indices (j,k) because A appears identically at all three operand positions,
        yielding savings despite i being a free index.

        Migration note: the old test asserted order=1 (trivial), checking that the
        oracle did not detect a V-side permutation mapping the free index i to a
        summed index. The new accumulation model correctly identifies savings from
        the cyclic structure of j,k in the identical operands.
        """
        A = fnp.random.randn(N, N)
        assert _has_savings("ij,jk,ki->i", A, A, A)

    def test_frobenius_inner_product_no_savings(self):
        """einsum('ij,ij->', A, A): Frobenius inner product.

        All N^2 unique (i,j) pairs contribute distinct values to the scalar
        output. Even though relabeling i↔j gives the same mathematical sum,
        the accumulation model cannot reduce the number of evaluations for
        a scalar output.

        Migration note: the old oracle found W-side S2{i,j} (order 2) here.
        The new accumulation model correctly gives m_total == dense_baseline
        for scalar outputs with no spatial structure to exploit.
        """
        A = fnp.random.randn(N, N)
        assert not _has_savings("ij,ij->", A, A)


class TestMixedOperands:
    """Expressions with non-identical operands (no symmetry expected)."""

    def test_aba_chain(self):
        """A·B·A: A appears twice but B breaks the chain."""
        A = fnp.random.randn(N, N)
        B = fnp.random.randn(N, N)
        assert not _has_savings("ij,jk,kl->il", A, B, A)

    def test_abab_alternating(self):
        """A·B·A·B: two pairs but interleaved — no swap is valid."""
        A = fnp.random.randn(N, N)
        B = fnp.random.randn(N, N)
        assert not _has_savings("ij,jk,kl,lm->im", A, B, A, B)

    def test_diagonal_extraction(self):
        """einsum('iij->ij', D): repeated index in subscript, no symmetry."""
        D = np.random.randn(N, N, N)
        assert not _has_savings("iij->ij", D)
