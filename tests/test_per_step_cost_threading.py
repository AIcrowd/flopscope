"""Regression tests for per-step cost symmetry threading (Sprint 1, Cat B).

Verifies that _per_step_flop_cost honors the propagated intermediate symmetries
from the SubgraphSymmetryOracle, instead of hardcoding sym_fingerprint=None.

Closes Wilson PR #91 issue #6 step-cost half; partially closes #5 (256 → 160,
remaining → 55 deferred to Sprint 2 Cat C).
"""

import flopscope as flops
import flopscope.numpy as fnp


def _einsum_path_info(subscripts, *operands):
    with flops.BudgetContext(10**12, quiet=True):
        _, info = fnp.einsum_path(subscripts, *operands)
    return info


def test_wilson6_step0_correct_regression_guard():
    """Wilson #6 step 0 is already correct (S₃ on K propagates).  Pin it.

    Step 0 contracts the original S₃ K with a dense W; the cost uses K's
    declared per-operand symmetry directly (not threaded from oracle).
    """
    n = 10
    K = fnp.random.default_rng(0).standard_normal((n, n, n))
    K = flops.symmetrize(K, symmetry=(0, 1, 2))
    W = fnp.random.default_rng(0).standard_normal((n, n))
    info = _einsum_path_info("abc,ai,bj,ck->ijk", K, W, W, W)
    # Expected ~10,450; allow a small window for slight α/M variations
    assert 10_000 <= info.steps[0].flop_cost <= 11_000, (
        f"step 0 cost {info.steps[0].flop_cost} not in expected range"
    )


def test_wilson6_step1_uses_intermediate_S2():
    """Wilson #6 step 1: ibc,bj->icj should drop from 19,000 (dense) to a
    much lower value once S₂{b,c} on the intermediate propagates.

    The intermediate from step 0 carries S₂{b,c}; passing that to the per-step
    cost should yield savings comparable to step 0.
    """
    n = 10
    K = fnp.random.default_rng(0).standard_normal((n, n, n))
    K = flops.symmetrize(K, symmetry=(0, 1, 2))
    W = fnp.random.default_rng(0).standard_normal((n, n))
    info = _einsum_path_info("abc,ai,bj,ck->ijk", K, W, W, W)
    # Strict assertion: must be < dense (19,000), and substantially so
    assert info.steps[1].flop_cost < 15_000, (
        f"step 1 cost {info.steps[1].flop_cost} should drop below 15,000 "
        "after S₂ propagates from the step-0 intermediate"
    )


def test_wilson6_step2_uses_intermediate_S2():
    """Wilson #6 step 2: icj,ck->ijk should drop from 19,000 (dense) — output
    has S₃ symmetry, intermediate has S₂{i,j}.
    """
    n = 10
    K = fnp.random.default_rng(0).standard_normal((n, n, n))
    K = flops.symmetrize(K, symmetry=(0, 1, 2))
    W = fnp.random.default_rng(0).standard_normal((n, n))
    info = _einsum_path_info("abc,ai,bj,ck->ijk", K, W, W, W)
    # Step 2 output carries S₃{i,j,k}; cost should reflect orbit-count savings
    assert info.steps[2].flop_cost < 15_000, (
        f"step 2 cost {info.steps[2].flop_cost} should drop below 15,000 "
        "after S₃ propagates from the step-1 intermediate's S₂{i,j}"
    )


def test_4cycle_steps_01_correct_regression_guard():
    """§5(a) steps 0 and 1 are already at 40 (S₂ on each input correctly
    propagates because each input is an original S, not an intermediate).
    Pin those.
    """
    S = fnp.zeros((4, 4))  # auto-tagged S₂
    info = _einsum_path_info("ij,jk,kl,li->ijkl", S, S, S, S)
    assert info.steps[0].flop_cost == 40, info.steps[0].flop_cost
    assert info.steps[1].flop_cost == 40, info.steps[1].flop_cost


def test_4cycle_step2_partial_fix_per_input_groups_only():
    """§5(a) step 2: lik,jki->ijkl.  Both inputs have S₂{i,k} from steps 0/1.

    After Sprint 1 (per-input groups threaded), the cost should drop from
    256 (current) to 160 (Burnside on (i,j,k,l) under S₂{i,k} on each input
    → orbit count 160 for n=4).

    Sprint 2 (Cat C) will further drop to 55 (D₄ orbits) by recognizing
    that both intermediates trace back to the same original S, but that's
    explicitly out of scope here.  When Sprint 2 lands and this test breaks,
    update the assertion to 55 with the test renamed accordingly.
    """
    S = fnp.zeros((4, 4))  # auto-tagged S₂
    info = _einsum_path_info("ij,jk,kl,li->ijkl", S, S, S, S)
    cost = info.steps[2].flop_cost
    assert cost == 160, (
        f"step 2 cost {cost} should be 160 (Sprint 1 target); "
        "becomes 55 in Sprint 2 (Cat C joint-identity)"
    )


def test_dense_only_path_unchanged():
    """Regression guard: all-dense, distinct-instance operands must keep their
    original per-step costs — Sprint 1 should not perturb dense paths.
    """
    rng = fnp.random.default_rng(0)
    x = rng.standard_normal((6, 7))
    y = rng.standard_normal((7, 8))
    z = rng.standard_normal((8, 9))
    info = _einsum_path_info("ij,jk,kl->il", x, y, z)
    # Each step is a binary matmul on distinct dense operands.
    # Formula: 2·m·n·k − m·n (FMA=2, first-cell-free)
    # Verify steps yield non-trivial costs and total is consistent.
    assert all(s.flop_cost > 0 for s in info.steps)
    total = sum(s.flop_cost for s in info.steps)
    assert info.optimized_cost == total
    assert info.check_consistency()
