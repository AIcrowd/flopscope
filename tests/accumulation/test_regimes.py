"""Tests for the four mixed regimes in _regimes.py."""

from flopscope._accumulation._ladder import RegimeContext
from flopscope._accumulation._regimes import FUNCTIONAL_PROJECTION_REGIME
from flopscope._perm_group import _dimino
from flopscope._perm_group import _Permutation as Permutation


def _ctx(
    *,
    labels,
    va,
    wa,
    elements,
    generators,
    sizes,
    visible_positions,
    partition_budget=100_000,
):
    return RegimeContext(
        labels=tuple(labels),
        va=tuple(va),
        wa=tuple(wa),
        elements=tuple(elements),
        generators=tuple(generators),
        sizes=tuple(sizes),
        visible_positions=tuple(visible_positions),
        partition_budget=partition_budget,
    )


# ── functionalProjection ──────────────────────────────────────────────


def test_functional_projection_fires_when_v_is_setwise_invariant():
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    ctx = _ctx(
        labels=("i", "j"),
        va=("i", "j"),
        wa=(),
        elements=elements,
        generators=(swap,),
        sizes=(4, 4),
        visible_positions=(0, 1),
    )
    verdict = FUNCTIONAL_PROJECTION_REGIME.recognize(ctx)
    assert verdict.fired is True


def test_functional_projection_refuses_when_g_moves_v_to_w():
    cycle = Permutation([1, 2, 0])  # 0→1→2→0; visible {0,1} not preserved
    elements = _dimino((cycle,))
    ctx = _ctx(
        labels=("i", "j", "k"),
        va=("i", "j"),
        wa=("k",),
        elements=elements,
        generators=(cycle,),
        sizes=(3, 3, 3),
        visible_positions=(0, 1),
    )
    verdict = FUNCTIONAL_PROJECTION_REGIME.recognize(ctx)
    assert verdict.fired is False
    assert "output label into a summed label" in verdict.reason


def test_functional_projection_compute_returns_burnside_count():
    swap = Permutation([1, 0])
    identity = Permutation.identity(2)
    ctx = _ctx(
        labels=("i", "j"),
        va=("i", "j"),
        wa=(),
        elements=(identity, swap),
        generators=(swap,),
        sizes=(4, 4),
        visible_positions=(0, 1),
    )
    out = FUNCTIONAL_PROJECTION_REGIME.compute(ctx)
    # S_2 on (4, 4): 4·5/2 = 10
    assert out.count == 10
    assert len(out.sub_steps) == 1
    assert out.sub_steps[0]["step"] == "projection-functional"
    assert out.sub_steps[0]["count"] == 10


# ── singleton ─────────────────────────────────────────────────────────


from flopscope._accumulation._regimes import SINGLETON_REGIME


def test_singleton_recognizes_when_v_size_is_one():
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    ctx = _ctx(
        labels=("i", "j"),
        va=("i",),
        wa=("j",),
        elements=elements,
        generators=(swap,),
        sizes=(4, 4),
        visible_positions=(0,),
    )
    verdict = SINGLETON_REGIME.recognize(ctx)
    assert verdict.fired is True


def test_singleton_refuses_when_v_size_not_one():
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    ctx = _ctx(
        labels=("i", "j"),
        va=("i", "j"),
        wa=(),
        elements=elements,
        generators=(swap,),
        sizes=(4, 4),
        visible_positions=(0, 1),
    )
    verdict = SINGLETON_REGIME.recognize(ctx)
    assert verdict.fired is False
    assert "|V| = 2" in verdict.reason


def test_singleton_compute_for_d2_r1_s2_example():
    """JS reference test: partition_count test #1 — S_2 on (i, j) with V=(i), W=(j),
    sizes (6, 6). Expected α = 36 (n²)."""
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    ctx = _ctx(
        labels=("i", "j"),
        va=("i",),
        wa=("j",),
        elements=elements,
        generators=(swap,),
        sizes=(6, 6),
        visible_positions=(0,),
    )
    out = SINGLETON_REGIME.compute(ctx)
    assert out.count == 36


def test_singleton_compute_for_s3_to_s2_reduction():
    """JS reference test: partition_count test #2 — S_3 on (i,j,k) with V=(i,j), W=(k)
    is OUTSIDE singleton (|V|=2). This test verifies refusal, not computation."""
    s01 = Permutation([1, 0, 2])
    s12 = Permutation([0, 2, 1])
    elements = _dimino((s01, s12))
    ctx = _ctx(
        labels=("i", "j", "k"),
        va=("i", "j"),
        wa=("k",),
        elements=elements,
        generators=(s01, s12),
        sizes=(4, 4, 4),
        visible_positions=(0, 1),
    )
    verdict = SINGLETON_REGIME.recognize(ctx)
    assert verdict.fired is False


# ── young ──────────────────────────────────────────────────────────────


from flopscope._accumulation._regimes import YOUNG_REGIME


def test_young_recognizes_full_sym_with_uniform_sizes():
    """G = S_3 on (i, j, k); both V and W nonempty; |V| ≥ 2; sizes uniform."""
    s01 = Permutation([1, 0, 2])
    s12 = Permutation([0, 2, 1])
    elements = _dimino((s01, s12))
    ctx = _ctx(
        labels=("i", "j", "k"),
        va=("i", "j"),
        wa=("k",),
        elements=elements,
        generators=(s01, s12),
        sizes=(4, 4, 4),
        visible_positions=(0, 1),
    )
    verdict = YOUNG_REGIME.recognize(ctx)
    assert verdict.fired is True


def test_young_refuses_when_v_size_below_2():
    s01 = Permutation([1, 0, 2])
    s12 = Permutation([0, 2, 1])
    elements = _dimino((s01, s12))
    ctx = _ctx(
        labels=("i", "j", "k"),
        va=("i",),
        wa=("j", "k"),
        elements=elements,
        generators=(s01, s12),
        sizes=(4, 4, 4),
        visible_positions=(0,),
    )
    verdict = YOUNG_REGIME.recognize(ctx)
    assert verdict.fired is False
    assert "|V|" in verdict.reason


def test_young_refuses_when_g_not_full_symmetric():
    """C_3 has 3 elements but |L|! = 6, so young refuses."""
    cyclic = Permutation([1, 2, 0])
    elements = _dimino((cyclic,))
    ctx = _ctx(
        labels=("i", "j", "k"),
        va=("i", "j"),
        wa=("k",),
        elements=elements,
        generators=(cyclic,),
        sizes=(4, 4, 4),
        visible_positions=(0, 1),
    )
    verdict = YOUNG_REGIME.recognize(ctx)
    assert verdict.fired is False


def test_young_refuses_with_mixed_sizes():
    s01 = Permutation([1, 0, 2])
    s12 = Permutation([0, 2, 1])
    elements = _dimino((s01, s12))
    ctx = _ctx(
        labels=("i", "j", "k"),
        va=("i", "j"),
        wa=("k",),
        elements=elements,
        generators=(s01, s12),
        sizes=(4, 4, 5),
        visible_positions=(0, 1),
    )
    verdict = YOUNG_REGIME.recognize(ctx)
    assert verdict.fired is False
    assert "mixed" in verdict.reason.lower()


def test_young_compute_multiset_formula():
    """For S_3 on (i,j,k) with V=(i,j), W=(k), n=4:
    α = C(4+2-1, 2) · C(4+1-1, 1) = C(5, 2) · 4 = 10 · 4 = 40"""
    s01 = Permutation([1, 0, 2])
    s12 = Permutation([0, 2, 1])
    elements = _dimino((s01, s12))
    ctx = _ctx(
        labels=("i", "j", "k"),
        va=("i", "j"),
        wa=("k",),
        elements=elements,
        generators=(s01, s12),
        sizes=(4, 4, 4),
        visible_positions=(0, 1),
    )
    out = YOUNG_REGIME.compute(ctx)
    assert out.count == 40


# ── partitionCount ────────────────────────────────────────────────────


from flopscope._accumulation._regimes import PARTITION_COUNT_REGIME


def test_partition_count_recognizes_when_under_budget():
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    ctx = _ctx(
        labels=("i", "j"),
        va=("i",),
        wa=("j",),
        elements=elements,
        generators=(swap,),
        sizes=(6, 6),
        visible_positions=(0,),
        partition_budget=100,
    )
    verdict = PARTITION_COUNT_REGIME.recognize(ctx)
    assert verdict.fired is True


def test_partition_count_refuses_when_over_budget():
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    ctx = _ctx(
        labels=("i", "j"),
        va=("i",),
        wa=("j",),
        elements=elements,
        generators=(swap,),
        sizes=(6, 6),
        visible_positions=(0,),
        partition_budget=0,
    )
    verdict = PARTITION_COUNT_REGIME.recognize(ctx)
    assert verdict.fired is False
    assert "budget" in verdict.reason


def test_partition_count_d2_r1_s2_matches_singleton():
    """JS partition-count test #1: S_2 on (i,j), V=(i), W=(j), sizes (6,6).
    Expected α = 36 (matches singleton regime)."""
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    ctx = _ctx(
        labels=("i", "j"),
        va=("i",),
        wa=("j",),
        elements=elements,
        generators=(swap,),
        sizes=(6, 6),
        visible_positions=(0,),
    )
    out = PARTITION_COUNT_REGIME.compute(ctx)
    assert out.count == 36


def test_partition_count_s3_reduction_alpha_40():
    """JS partition-count test #2: S_3 on (i,j,k) with V=(i,j), W=(k), sizes (4,4,4).
    Expected α = 40 (matches young regime)."""
    s01 = Permutation([1, 0, 2])
    s12 = Permutation([0, 2, 1])
    elements = _dimino((s01, s12))
    ctx = _ctx(
        labels=("i", "j", "k"),
        va=("i", "j"),
        wa=("k",),
        elements=elements,
        generators=(s01, s12),
        sizes=(4, 4, 4),
        visible_positions=(0, 1),
    )
    out = PARTITION_COUNT_REGIME.compute(ctx)
    assert out.count == 40


def test_partition_count_heterogeneous_partitions_alpha_90():
    """JS partition-count test #3: S_2 × S_2 on (i,j,k,l), V=(i,j), W=(k,l), sizes (3,3,5,5).
    Expected α = 90."""
    vis_swap = Permutation([1, 0, 2, 3])
    sum_swap = Permutation([0, 1, 3, 2])
    elements = _dimino((vis_swap, sum_swap))
    ctx = _ctx(
        labels=("i", "j", "k", "l"),
        va=("i", "j"),
        wa=("k", "l"),
        elements=elements,
        generators=(vis_swap, sum_swap),
        sizes=(3, 3, 5, 5),
        visible_positions=(0, 1),
    )
    out = PARTITION_COUNT_REGIME.compute(ctx)
    assert out.count == 90


def test_partition_count_emits_subtrace_per_partition():
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    ctx = _ctx(
        labels=("i", "j"),
        va=("i",),
        wa=("j",),
        elements=elements,
        generators=(swap,),
        sizes=(6, 6),
        visible_positions=(0,),
    )
    out = PARTITION_COUNT_REGIME.compute(ctx)
    assert len(out.sub_steps) >= 1
    for step in out.sub_steps:
        # Schema check — each substep names its pattern + counts.
        assert "partition_key" in step
        assert "blocks" in step
        assert "typed_labelings" in step
        assert "block_action_size" in step
        assert "input_orbit_count" in step
        assert "output_orbit_count" in step
        assert "contribution" in step
