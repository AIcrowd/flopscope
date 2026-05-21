"""Regression tests for multi-operand einsum output symmetry tagging (Sprint 1, Cat A).

These verify that fnp.einsum(...) returns a SymmetricTensor when the path-walker's
oracle has derived a non-trivial output_group on the last step.  Before Sprint 1,
multi-operand einsums returned plain FlopscopeArray regardless of provable output
symmetry (Wilson PR #91 issues #4 and the output-tagging half of #6).
"""

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._symmetric import SymmetricTensor


def test_hadamard_of_S2_inherits_S2():
    """Wilson #4 exact: einsum('ij,ij->ij', S, S) should yield S₂ output.

    For symmetric S, (S*S)[i,j] = S[i,j]·S[i,j] = S[j,i]·S[j,i] = (S*S)[j,i]
    so the Hadamard product is symmetric.  fnp.zeros((4,4)) auto-tags as S₂.
    """
    S = fnp.zeros((4, 4))
    T = fnp.einsum("ij,ij->ij", S, S)
    assert isinstance(T, SymmetricTensor), (
        f"expected SymmetricTensor with symmetry, got {type(T).__name__}"
    )
    assert T.symmetry is not None
    # Order-2 group on the 2 axes of the output
    assert len(list(T.symmetry.elements())) == 2
    assert T.symmetry.axes == (0, 1)


def test_K_W_W_W_inherits_S3():
    """Wilson #6 output-tag half: K (S₃) · W · W · W → ijk should yield S₃ output.

    The oracle reports `S2{i,j} × - → S3{i,j,k}` on the last step.  The result
    tensor should be tagged S₃ on (i,j,k).
    """
    n = 6
    K = fnp.random.default_rng(0).standard_normal((n, n, n))
    K = flops.symmetrize(K, symmetry=(0, 1, 2))
    W = fnp.random.default_rng(0).standard_normal((n, n))
    R = fnp.einsum("abc,ai,bj,ck->ijk", K, W, W, W)
    assert isinstance(R, SymmetricTensor), (
        f"expected SymmetricTensor with symmetry, got {type(R).__name__}"
    )
    assert R.symmetry is not None
    assert len(list(R.symmetry.elements())) == 6  # |S₃| = 6
    assert R.symmetry.axes == (0, 1, 2)


def test_distinct_dense_operands_no_symmetry():
    """Negative case: distinct dense operands should NOT acquire spurious symmetry.

    Three distinct random matrices contracted linearly — no operand swap,
    no declared symmetry, the result should remain a plain FlopscopeArray.
    """
    rng = fnp.random.default_rng(0)
    x = rng.standard_normal((5, 6))
    y = rng.standard_normal((6, 7))
    z = rng.standard_normal((7, 8))
    R = fnp.einsum("ij,jk,kl->il", x, y, z)
    assert getattr(R, "symmetry", None) is None, (
        f"unexpected symmetry {getattr(R, 'symmetry', None)} on distinct-operand result"
    )


def test_single_operand_path_preserved():
    """Regression guard: single-operand einsum (no summing) preserves symmetry tagging.

    The existing single-operand inference path handles cases where every operand label
    survives in the output (no axes summed out).  Cases with summed axes hit a
    separate bug in `_relabel_group_to_output` that is out of scope for Sprint 1;
    see the project tracker for the missing-label restriction follow-up.
    """
    T = fnp.zeros((4, 4, 4))  # auto-tagged S₃
    R = fnp.einsum("ijk->ijk", T)
    assert isinstance(R, SymmetricTensor)
    assert R.symmetry is not None
    assert len(list(R.symmetry.elements())) == 6  # S₃ preserved
    assert R.symmetry.axes == (0, 1, 2)


def test_partial_axes_symmetry_on_output():
    """When only some output axes carry symmetry, the SymmetricTensor's axes
    should reflect that — not be padded to cover all output axes.

    Example: einsum('ab,cd->abcd', X, X) for identical X gives a coupled
    (a,c)(b,d) generator — symmetry on all 4 axes via a single non-trivial
    permutation.  We just verify symmetry exists and the axes match.
    """
    X = fnp.random.default_rng(0).standard_normal((4, 4))
    R = fnp.einsum("ab,cd->abcd", X, X)
    assert isinstance(R, SymmetricTensor)
    assert R.symmetry is not None
    # Group should have at least one non-identity element acting on the output axes
    assert len(list(R.symmetry.elements())) >= 2
    assert R.symmetry.axes is not None
    assert set(R.symmetry.axes).issubset({0, 1, 2, 3})


def test_scalar_output_no_tagging_attempted():
    """Scalar einsums (no output labels) should return a 0-d array, no symmetry
    tagging attempted.  einsum('ij,ji->', A, A) collapses to a scalar.
    """
    A = fnp.random.default_rng(0).standard_normal((5, 5))
    R = fnp.einsum("ij,ji->", A, A)
    # Scalar — no shape to tag symmetry on
    assert np.asarray(R).shape == (), (
        f"expected scalar, got shape {np.asarray(R).shape}"
    )
    # Either no symmetry attribute, or symmetry is None (both fine for scalar)
    assert getattr(R, "symmetry", None) is None
