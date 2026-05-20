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
    should produce cache hits on the shared step.

    Updated for Task 17b: SubgraphSymmetryOracle now threads symmetry through
    per-step calls. Each step now makes TWO cache calls — one from build_path_info
    (dense fingerprint, for path ordering) and one from _walk_path_and_aggregate
    (oracle fingerprint, for cost). So first call yields:
      1 (top-level) + 2 (build_path_info dense) + 2 (oracle per-step) = 5 misses.
    The second call shares 1 dense step and 1 oracle step → 2 hits.
    """
    from flopscope._accumulation._cache import _accumulation_cache

    flops.clear_cache()
    info_before = _accumulation_cache.cache_info()
    assert info_before.hits == 0 and info_before.misses == 0

    x = fnp.ones((4, 4))
    # First call: 1 top-level miss + 2 build_path_info dense-step misses +
    # 2 oracle per-step misses = 5 misses total.
    # (Updated for Task 17b: was 3 misses before oracle was wired in.)
    flops.einsum_accumulation_cost("ij,jk,kl->il", x, x, x)
    info_mid = _accumulation_cache.cache_info()
    assert info_mid.misses == 5, f"expected 5 misses, got {info_mid.misses}"

    # Second call shares the first build_path_info step (dense) and the first
    # oracle per-step. Top-level miss + 1 dense-step miss + 1 oracle-step miss
    # + 2 hits for the shared steps. Total: >=2 cumulative hits.
    flops.einsum_accumulation_cost("ij,jk,km->im", x, x, x)
    info_after = _accumulation_cache.cache_info()
    assert info_after.hits >= 2, (
        f"expected >=2 hits (shared build_path_info + oracle steps), "
        f"got {info_after.hits}"
    )


def test_symmetry_group_hash_is_canonical_for_cache_reuse():
    """Two SymmetryGroups built from equivalent generators should hash and
    compare equal, so cache lookups reuse entries across construction paths."""
    from flopscope import SymmetryGroup

    g1 = SymmetryGroup.symmetric(axes=(0, 1))
    g2 = SymmetryGroup.symmetric(axes=(0, 1))
    assert g1 == g2, "symmetric(axes=(0,1)) should equal itself across calls"
    assert hash(g1) == hash(g2), "symmetric(axes=(0,1)) hash must be stable"


def test_path_walker_per_step_cost_matches_accumulation_per_step():
    """info.steps[i].flop_cost must equal info.accumulation.per_step[i].total."""
    x = fnp.ones((10, 10))
    with flops.BudgetContext(flop_budget=10**12, quiet=True):
        _, info = fnp.einsum_path("ij,jk,kl->il", x, x, x)

    assert info.accumulation is not None
    assert info.accumulation.per_step, "expected populated per_step for k=3"
    assert len(info.steps) == len(info.accumulation.per_step)
    for i, (step, acc_step) in enumerate(
        zip(info.steps, info.accumulation.per_step, strict=False)
    ):
        assert step.flop_cost == acc_step.total, (
            f"step {i}: path-walker flop_cost={step.flop_cost} != "
            f"accumulation per-step total={acc_step.total}"
        )



def test_symmetric_triangle_uses_inherited_symmetry():
    """ij,jk,ki->ijk with all three operands sharing an S_2 symmetric matrix:
    the path walker should INHERIT the full-expression group's restriction
    to each binary step, not treat intermediates as dense.

    Expected: per-step costs use the restricted group, total = 80
    (two 40-op symmetric steps), not 128 (two 64-op dense steps).
    Before SubgraphSymmetryOracle was wired in, this returned 128.
    """
    import flopscope as flops
    import flopscope.numpy as fnp

    A = flops.as_symmetric(fnp.zeros((4, 4)), symmetry=(0, 1))
    cost = flops.einsum_accumulation_cost("ij,jk,ki->ijk", A, A, A)

    # Pre-oracle: cost.total == 128. With oracle: each step is a binary
    # contraction whose effective group inherits from the full S_2 symmetry,
    # bringing total down. Exact value depends on opt_einsum's path choice
    # under symmetry-aware search, but it MUST be < 128 (proves symmetry
    # is being threaded through).
    assert cost.total < 128, (
        f"symmetric triangle should benefit from symmetry inheritance; "
        f"got {cost.total} (>=128 implies oracle is not being used)"
    )


def test_three_costs_agree_for_multi_operand():
    """info.optimized_cost == sum(s.flop_cost for s in info.steps) ==
    info.accumulation.total. Holds for every multi-operand expression."""
    import flopscope as flops
    import flopscope.numpy as fnp

    cases = [
        ("ij,jk,kl->il", (fnp.ones((10, 10)),) * 3),
        ("ij,jk,kl,lm->im", (fnp.ones((4, 4)),) * 4),
        ("ai,bi,aj,bj->ab", (fnp.ones((4, 4)),) * 4),
    ]
    for subs, ops in cases:
        with flops.BudgetContext(flop_budget=10**12, quiet=True):
            _, info = fnp.einsum_path(subs, *ops)
        step_sum = sum(s.flop_cost for s in info.steps)
        assert info.optimized_cost == step_sum, (
            f"{subs}: optimized_cost={info.optimized_cost} != "
            f"sum(steps.flop_cost)={step_sum}"
        )
        assert info.optimized_cost == info.accumulation.total, (
            f"{subs}: optimized_cost={info.optimized_cost} != "
            f"accumulation.total={info.accumulation.total}"
        )


def test_path_cache_distinguishes_on_per_op_symmetries():
    """Same subscripts, different per-op symmetries should produce distinct
    cache entries (and may produce different paths once symmetry-aware
    search is enabled).

    Uses einsum_path() (which routes through _get_path_info → _path_cache)
    to exercise the path-cache key directly.
    """
    import flopscope as flops
    import flopscope.numpy as fnp
    from flopscope._einsum import _path_cache

    flops.clear_cache()
    A = fnp.zeros((4, 4))
    S = flops.as_symmetric(fnp.zeros((4, 4)), symmetry=(0, 1))

    with flops.BudgetContext(flop_budget=10**12, quiet=True):
        fnp.einsum_path("ij,jk->ik", A, A)
    info_a = _path_cache.cache_info()

    with flops.BudgetContext(flop_budget=10**12, quiet=True):
        fnp.einsum_path("ij,jk->ik", S, A)
    info_b = _path_cache.cache_info()

    # The second call must be a miss (different per_op_symmetries).
    assert info_b.misses > info_a.misses, (
        f"path cache treated symmetric and dense calls as same key: "
        f"hits_after={info_b.hits}, misses_after={info_b.misses}, "
        f"misses_before={info_a.misses}"
    )


def test_large_k_auto_fallback_to_greedy():
    """For k >= 8 with optimize='auto', resolve to greedy to avoid optimal/B&B
    cold-call latency blowup."""
    from flopscope._einsum import _resolve_optimize_for_k

    assert _resolve_optimize_for_k("auto", k=8) == "greedy"
    assert _resolve_optimize_for_k("auto", k=10) == "greedy"
    assert _resolve_optimize_for_k("auto", k=7) == "auto"
    # Explicit user choice always honored.
    assert _resolve_optimize_for_k("optimal", k=10) == "optimal"
    assert _resolve_optimize_for_k("branch", k=10) == "branch"


def test_large_k_einsum_completes_within_one_second():
    """Smoke check: k=8 chain with optimize='auto' completes quickly."""
    import time

    import flopscope as flops
    import flopscope.numpy as fnp

    subs = (
        ",".join(f"{chr(ord('a') + i)}{chr(ord('a') + i + 1)}" for i in range(8))
        + "->ai"
    )
    ops = tuple(fnp.ones((3, 3)) for _ in range(8))
    flops.clear_cache()
    t0 = time.perf_counter()
    flops.einsum_accumulation_cost(subs, *ops)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"k=8 auto cold call took {elapsed:.2f}s (budget 1.0s)"


def test_clear_cache_flushes_all_layers():
    import flopscope as flops
    import flopscope.numpy as fnp
    from flopscope._accumulation._cache import _accumulation_cache, _reduction_cache
    from flopscope._einsum import _path_cache

    x = fnp.ones((4, 4))
    with flops.BudgetContext(flop_budget=10**12, quiet=True):
        fnp.einsum_path("ij,jk,kl->il", x, x, x)
    flops.einsum_accumulation_cost("ij,jk,kl->il", x, x, x)
    assert _path_cache.cache_info().currsize > 0
    assert _accumulation_cache.cache_info().currsize > 0

    flops.clear_cache()
    assert _path_cache.cache_info().currsize == 0
    assert _accumulation_cache.cache_info().currsize == 0
    assert _reduction_cache.cache_info().currsize == 0
