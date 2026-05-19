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

  Output auto-tagging as SymmetricTensor has also been removed (the oracle that
  inferred output symmetry from equal-operand detection is gone). Results are plain
  FlopscopeArrays; use flops.as_symmetric() explicitly if you need the tag.
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
        """The gram matrix result is numerically correct."""
        X = np.arange(1.0, 17.0).reshape(4, 4)

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = fnp.einsum("ij,ik->jk", X, X)

        expected = np.einsum("ij,ik->jk", X, X)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        # The accumulation model detects savings but does NOT auto-tag output
        # as SymmetricTensor (the oracle that did that has been removed).
        assert not isinstance(result, SymmetricTensor)

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
        # Updated for Task 17b: SubgraphSymmetryOracle now threads symmetry across
        # binary steps (was conservative dense-intermediate value of 20000).
        # With oracle: step1=550 (ij,ik->jk with S2 inherited) + step2=10450 (jk,il->jkl).
        # Total = 11000 (tighter symmetry-aware path cost).
        assert info.optimized_cost == 11000
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
