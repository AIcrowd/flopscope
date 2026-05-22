"""End-to-end integration tests for equal-operand induced symmetry.

Tests that fnp.einsum with repeated operands produces the expected symmetry-aware
FLOP costs. Uses hand-computed expected values.

Migration note (direct-event accumulation model with off-by-one correction):
  The new model uses
      total = (num_terms - 1) * prod(M) + prod(alpha) - prod(num_output_orbits).
  The final ``- prod(num_output_orbits)`` term applies the off-by-one correction
  used by ``reduction_accumulation_cost``: the first cell of each output orbit
  is a free copy. For a 2-term expression with S2 savings (output orbits = 55):
  cost = 1 * 550 + 550 - 55 = 1045.

  Sprint 1 Cat A re-enables output auto-tagging as SymmetricTensor via the
  path-walker's SubgraphSymmetryOracle.  For multi-operand einsum with identical
  operands the oracle infers output symmetry automatically; use
  flops.as_symmetric() only when the oracle cannot infer it.
"""

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._budget import BudgetContext
from flopscope._symmetric import SymmetricTensor


class TestGramMatrixInduction:
    """einsum('ij,ik->jk', X, X) — the classic Gram matrix."""

    def test_plain_X_induces_s2_on_jk(self):
        n = 10
        X = np.ones((n, n))
        # Accumulation model with off-by-one correction:
        # m_total = n * C(n+1,2) = 10 * 55 = 550 unique (i,j,k) combos.
        # num_output_orbits = 55 (S2 swap of j<->k on the (j,k) output).
        # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
        # = 550 + 550 - 55 = 1045. First cell of each output orbit is free.
        _, info_eq = fnp.einsum_path("ij,ik->jk", X, X)
        assert info_eq.optimized_cost == 1045
        # Verify savings are present: m_total < dense_baseline
        acc = info_eq.accumulation
        assert acc.m_total < acc.dense_baseline

    def test_different_operands_dense_cost(self):
        n = 10
        X = np.ones((n, n))
        Y = np.ones((n, n))
        _, info = fnp.einsum_path("ij,ik->jk", X, Y)
        # Different operands → no induction → full dense.
        # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
        # = 1000 + 1000 - 100 = 1900 (textbook 2n^3 - n^2 form).
        assert info.optimized_cost == 1900
        acc = info.accumulation
        assert acc.m_total == acc.dense_baseline  # no savings

    def test_einsum_numerically_correct(self):
        """The gram matrix result is numerically correct and tagged S₂."""
        X = np.arange(1.0, 17.0).reshape(4, 4)

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = fnp.einsum("ij,ik->jk", X, X)

        expected = np.einsum("ij,ik->jk", X, X)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        # Sprint 1 Cat A: multi-operand einsum auto-tags output symmetry via
        # the path-walker's SubgraphSymmetryOracle.  X @ X.T for identical X
        # is provably symmetric, so the result is tagged S₂.
        assert isinstance(result, SymmetricTensor)
        assert len(list(result.symmetry.elements())) == 2
        assert result.symmetry.axes == (0, 1)

    def test_einsum_with_plain_out_preserves_output_identity(self):
        X = np.arange(1.0, 17.0).reshape(4, 4)
        out = np.empty((4, 4))

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = fnp.einsum("ij,ik->jk", X, X, out=out)

        expected = np.einsum("ij,ik->jk", X, X)
        assert result is out
        assert not isinstance(result, SymmetricTensor)
        np.testing.assert_allclose(out, expected, rtol=1e-10)


class TestMatMulChainNoInducedSymmetry:
    """einsum('ij,jk->ik', X, X) with plain (non-declared-symmetric) X.

    Regression guard: passing the same Python object does not imply the tensor
    values are symmetric. X @ X is NOT symmetric in (i, k) unless X itself is
    symmetric — the output value R[i,k] = Σ_j X[i,j]·X[j,k] differs from
    R[k,i] = Σ_j X[k,j]·X[j,i] for a generic non-symmetric X.

    Use flops.as_symmetric() to declare symmetry explicitly — see
    TestSymmetricXMatMul below for the declared-symmetric case.
    """

    def test_plain_X_has_no_induced_symmetry(self):
        n = 10
        X = np.ones((n, n))
        _, info = fnp.einsum_path("ij,jk->ik", X, X)
        # No symmetry detected: m_total == dense_baseline.
        # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
        # = 1000 + 1000 - 100 = 1900 (textbook 2n^3 - n^2 form).
        # First cell of each output orbit is a free copy.
        assert info.optimized_cost == 1900
        acc = info.accumulation
        assert acc.m_total == acc.dense_baseline  # no savings


class TestTripleProductInduction:
    """einsum('ij,ik,il->jkl', X, X, X) — three-way induction → S3."""

    def test_three_equal_operands_induce_s3(self):
        n = 10
        X = np.ones((n, n))
        _, info = fnp.einsum_path("ij,ik,il->jkl", X, X, X)
        # Trajectory:
        #   pre-Sprint-1: 20000 (dense intermediate, no symmetry threading)
        #   post-Sprint-1: 11000 (per-input S₂ inherited; step 0 = 550, step 1 = 10450)
        #   post-Sprint-2: 4730  (joint-Burnside on the merged subset captures the
        #                          full S₃ across all 3 identical X's; step 0 = 550,
        #                          step 1 = 4180 = 2·M − O with M = 2200, O = 220)
        # Sprint 2 reaches joint S₃{j,k,l} on the output, dropping step 1 by ~6,270.
        assert info.optimized_cost == 4730
        acc = info.accumulation
        assert acc.m_total < acc.dense_baseline  # savings from S3


class TestBlockOuterProductInduction:
    """einsum('ijk,ilm->jklm', X, X) — block symmetry on (j,k) and (l,m)."""

    def test_block_s2_induction(self):
        n = 10
        X = np.ones((n, n, n))
        _, info = fnp.einsum_path("ijk,ilm->jklm", X, X)
        # Accumulation model with off-by-one correction: block S2 swaps the
        # two operand blocks. m_total = 50500.
        # num_output_orbits = C(n^2+1, 2) = 100*101/2 = 5050 (S2 on (jk,lm)).
        # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
        # = 50500 + 50500 - 5050 = 95950. First cell of each output orbit is free.
        assert info.optimized_cost == 95950
        acc = info.accumulation
        assert acc.m_total < acc.dense_baseline  # savings from block S2


class TestSymmetricXMatMul:
    """einsum('ij,jk->ik', X, X) where X is already declared symmetric.

    Per-operand symmetry (S2 on X's axes) detected; same m_total as the
    gram matrix case.
    """

    def test_both_sources_apply(self):
        n = 10
        X_data = np.ones((n, n))
        X = flops.as_symmetric(X_data, symmetry=(0, 1))
        _, info = fnp.einsum_path("ij,jk->ik", X, X)
        # Accumulation model with off-by-one correction:
        # m_total = 550, num_output_orbits = 55 (S2 on (i,k) output).
        # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
        # = 550 + 550 - 55 = 1045. First cell of each output orbit is free.
        assert info.optimized_cost == 1045
        acc = info.accumulation
        assert acc.m_total < acc.dense_baseline  # savings detected
