"""Regression tests for joint-identity per-step cost reductions (Sprint 2 Cat C).

Verifies the merged-subset oracle's joint group reduces per-step cost in
cases where Sprint 1's per-input fingerprints overcount — specifically when
identical-operand swaps span step boundaries (intermediates trace back to
the same original operands).
"""

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp


def _einsum_path_info(subscripts, *operands):
    with flops.BudgetContext(10**12, quiet=True):
        _, info = fnp.einsum_path(subscripts, *operands)
    return info


def test_4cycle_step2_full_close():
    """§5(a) step 2: lik,jki->ijkl with shared S₂ S.

    Sprint 1 reduced step 2 from 256 → 160 (per-input S₂{i,k} on each
    intermediate).  Sprint 2 reduces 160 → 55 via the merged-subset D₄ joint
    group, which sees that all 4 original S's are identical, contributing
    operand-swap generators that per-input cannot synthesize across step
    boundaries.
    """
    S = fnp.zeros((4, 4))  # auto-tagged S₂
    info = _einsum_path_info("ij,jk,kl,li->ijkl", S, S, S, S)
    assert info.steps[2].flop_cost == 55, (
        f"step 2 cost {info.steps[2].flop_cost} should be 55 after Sprint 2"
    )


def test_4cycle_total_drops_to_135():
    """§5(a) total = 40 + 40 + 55 = 135 after Sprint 2 (was 240 after Sprint 1)."""
    S = fnp.zeros((4, 4))
    info = _einsum_path_info("ij,jk,kl,li->ijkl", S, S, S, S)
    assert info.optimized_cost == 135


def test_wilson6_step0_per_input_remains_primary():
    """Wilson #6 step 0: V = (b,c,i), W = (a), K has S₃ on (a,b,c), W is dense.
    Joint group = S₂{b,c} (the S₃ elements moving a get filtered out because W
    is dense).  Both per-input and joint give the same answer; cost stays at
    ~10,450.  This test guards against regression of Sprint 1's behavior.
    """
    n = 10
    K = fnp.random.default_rng(0).standard_normal((n, n, n))
    K = flops.symmetrize(K, symmetry=(0, 1, 2))
    W = fnp.random.default_rng(0).standard_normal((n, n))
    info = _einsum_path_info("abc,ai,bj,ck->ijk", K, W, W, W)
    assert 10_000 <= info.steps[0].flop_cost <= 11_000, info.steps[0].flop_cost


def test_cross_V_W_falls_back_to_per_input():
    """`cross-c3-partial` preset: abc->ab with cyclic C₃ on (a,b,c).
    The cycle moves c (W) → a (V), so V is NOT preserved by the joint group.
    The new helper returns None; per-input via the partition-count regime
    gives the correct (non-dense) cost — Sprint 2 must NOT regress this.
    """
    T = flops.as_symmetric(
        np.zeros((4, 4, 4)),
        symmetry=flops.SymmetryGroup.cyclic(axes=(0, 1, 2)),
    )
    info = _einsum_path_info("abc->ab", T)
    assert info.check_consistency()
    # Must be strictly less than dense (4^3 = 64) — partition-count found savings
    assert info.optimized_cost < 64


def test_triangle_shared_A_uses_joint():
    """ij,jk,ki->ijk with three identical dense A.  Joint group includes
    cyclic operand-swap symmetry across steps.  Sprint 2 should match or
    improve Sprint 1's cost of 88.
    """
    A = fnp.random.default_rng(0).standard_normal((4, 4))
    info = _einsum_path_info("ij,jk,ki->ijk", A, A, A)
    assert info.optimized_cost <= 88, info.optimized_cost
    assert info.check_consistency()


def test_dense_only_path_unchanged():
    """All-distinct dense operands: both per-input and joint give dense; min = dense."""
    rng = fnp.random.default_rng(0)
    x = rng.standard_normal((6, 7))
    y = rng.standard_normal((7, 8))
    z = rng.standard_normal((8, 9))
    info = _einsum_path_info("ij,jk,kl->il", x, y, z)
    for s in info.steps:
        assert s.flop_cost == s.dense_flop_cost, (
            f"step {info.steps.index(s)} cost {s.flop_cost} != dense {s.dense_flop_cost}"
        )
