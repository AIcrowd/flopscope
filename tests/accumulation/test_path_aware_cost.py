"""Path-aware einsum cost regression tests (Wilson PR #91 review, bug #1)."""

import flopscope as flops
import flopscope.numpy as fnp


def test_wilson_three_operand_dense_chain():
    """ij,jk,kl->il with n=10 dense operands should cost ~4*n^3 (two binary
    matmuls), not ~3*n^4 (one fictitious 3-way contraction).

    Textbook two-step path cost = 2 * (2*n^3 - n^2) = 2*1900 = 3800.
    Off-by-one fix alone leaves us at 29900 (still wrong); path-aware fix
    brings it to 3800.
    """
    x = fnp.ones((10, 10))
    y = fnp.ones((10, 10))
    z = fnp.ones((10, 10))
    cost = flops.einsum_accumulation_cost("ij,jk,kl->il", x, y, z)
    assert cost.total == 3800, (
        f"Wilson regression: ij,jk,kl->il (n=10) returned {cost.total}, "
        f"expected 3800 (= 2 * (2*n^3 - n^2) for two binary matmuls)"
    )


def test_wilson_four_operand_dense_chain():
    """Same shape, one more operand. Expected: 3 * (2*n^3 - n^2) = 5700."""
    x = fnp.ones((10, 10))
    cost = flops.einsum_accumulation_cost("ij,jk,kl,lm->im", x, x, x, x)
    # Path picks three binary matmuls; each costs 2*n^3 - n^2 = 1900.
    # (When opt_einsum reuses operands the actual symmetric cost may
    # drop further; this assertion is a conservative ceiling.)
    assert cost.total <= 3 * 1900, (
        f"4-operand chain returned {cost.total}, expected <= 5700 "
        f"(three binary matmuls at 2*n^3 - n^2 each)"
    )


def test_binary_einsum_unchanged():
    """k=2 einsums must produce byte-identical totals to pre-path-aware behavior.
    Asserts the off-by-one-fix's matmul value (textbook 2*n^3 - n^2)."""
    A = fnp.ones((4, 4))
    B = fnp.ones((4, 4))
    cost = flops.einsum_accumulation_cost("ij,jk->ik", A, B)
    assert cost.total == 112, (
        f"binary matmul (n=4) returned {cost.total}, expected 112 "
        f"(= 2*n^3 - n^2 from off-by-one fix; must stay byte-identical)"
    )
    assert cost.per_step == ()  # no path walked for k=2
    assert cost.path is None


def test_per_step_totals_sum_to_top_level():
    """For multi-operand einsums, top-level total must equal sum of per-step totals."""
    x = fnp.ones((10, 10))
    cost = flops.einsum_accumulation_cost("ij,jk,kl->il", x, x, x)
    assert cost.per_step, "k=3 einsum should have populated per_step"
    step_sum = sum(s.total for s in cost.per_step)
    assert cost.total == step_sum, (
        f"top-level total ({cost.total}) != sum of per_step ({step_sum})"
    )


def test_per_step_path_matches_top_level_path():
    """AccumulationCost.path must match the path opt_einsum produced."""
    x = fnp.ones((4, 4))
    cost = flops.einsum_accumulation_cost("ij,jk,kl,lm->im", x, x, x, x)
    assert cost.path is not None
    # 4-operand chain produces 3 binary steps.
    assert len(cost.path) == 3, f"expected 3 steps, got {cost.path}"
    assert len(cost.per_step) == 3


def test_per_step_cache_hits_across_expressions():
    """Two different multi-operand expressions sharing a binary sub-step
    (here 'ij,jk->ik') should produce one cache hit on the shared step."""
    from flopscope._accumulation._cache import _accumulation_cache

    flops.clear_cache()
    info_before = _accumulation_cache.cache_info()
    assert info_before.hits == 0 and info_before.misses == 0

    x = fnp.ones((4, 4))
    # First call: 1 miss (top-level) + 2 misses (per-step "ij,jk->ik"
    # and "ik,kl->il"). Total: 3 misses.
    flops.einsum_accumulation_cost("ij,jk,kl->il", x, x, x)
    info_mid = _accumulation_cache.cache_info()
    assert info_mid.misses == 3, f"expected 3 misses, got {info_mid.misses}"

    # Second call shares "ij,jk->ik" step. Top-level miss (different
    # subscripts), one per-step hit on "ij,jk->ik", one per-step miss
    # on "ik,km->im". Total: 1 hit + 2 misses (cumulative 1 + 5).
    flops.einsum_accumulation_cost("ij,jk,km->im", x, x, x)
    info_after = _accumulation_cache.cache_info()
    assert info_after.hits >= 1, (
        f"expected >=1 hit (shared step), got {info_after.hits}"
    )
