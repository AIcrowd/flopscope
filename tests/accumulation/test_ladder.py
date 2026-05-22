"""Tests for compute_accumulation — the per-component regime dispatcher."""

from flopscope._accumulation._ladder import compute_accumulation
from flopscope._perm_group import _dimino
from flopscope._perm_group import _Permutation as Permutation


def test_dispatcher_trivial_short_circuit_for_empty_elements():
    result = compute_accumulation(
        labels=("i", "j"),
        va=("i",),
        wa=("j",),
        elements=(),
        generators=(),
        sizes=(3, 4),
        visible_positions=(0,),
    )
    assert result.count == 12  # 3 * 4
    assert result.regime_id == "trivial"
    assert result.shape == "trivial"
    assert len(result.trace) == 1
    assert result.trace[0].decision == "fired"


def test_dispatcher_trivial_short_circuit_for_identity_only():
    identity = Permutation.identity(2)
    result = compute_accumulation(
        labels=("i", "j"),
        va=("i",),
        wa=("j",),
        elements=(identity,),
        generators=(),
        sizes=(3, 4),
        visible_positions=(0,),
    )
    assert result.count == 12
    assert result.regime_id == "trivial"


def test_dispatcher_picks_functional_projection_for_v_invariant_action():
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    result = compute_accumulation(
        labels=("i", "j"),
        va=("i", "j"),
        wa=(),
        elements=elements,
        generators=(swap,),
        sizes=(4, 4),
        visible_positions=(0, 1),
    )
    # S_2 on (4,4): n(n+1)/2 = 10
    assert result.count == 10
    assert result.regime_id == "functionalProjection"
    assert result.shape == "allVisible"


def test_dispatcher_picks_singleton_for_single_visible_label():
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    result = compute_accumulation(
        labels=("i", "j"),
        va=("i",),
        wa=("j",),
        elements=elements,
        generators=(swap,),
        sizes=(6, 6),
        visible_positions=(0,),
    )
    assert result.count == 36
    assert result.regime_id == "singleton"
    assert result.shape == "mixed"


def test_dispatcher_picks_young_for_full_sym_uniform():
    s01 = Permutation([1, 0, 2])
    s12 = Permutation([0, 2, 1])
    elements = _dimino((s01, s12))
    result = compute_accumulation(
        labels=("i", "j", "k"),
        va=("i", "j"),
        wa=("k",),
        elements=elements,
        generators=(s01, s12),
        sizes=(4, 4, 4),
        visible_positions=(0, 1),
    )
    assert result.count == 40
    assert result.regime_id == "young"


def test_dispatcher_picks_partition_count_when_others_refuse():
    """Z_2 cross-swap (0<->2, 1<->3): functionalProjection refuses (moves visible->summed),
    singleton refuses (|V|=2), young refuses (mixed sizes). Falls through to partitionCount."""
    cross = Permutation([2, 3, 0, 1])
    elements = _dimino((cross,))
    result = compute_accumulation(
        labels=("i", "j", "k", "l"),
        va=("i", "j"),
        wa=("k", "l"),
        elements=elements,
        generators=(cross,),
        sizes=(3, 5, 3, 5),
        visible_positions=(0, 1),
    )
    assert result.count == 225
    assert result.regime_id == "partitionCount"


def test_dispatcher_returns_unavailable_when_partition_budget_exceeded():
    cross = Permutation([2, 3, 0, 1])
    elements = _dimino((cross,))
    result = compute_accumulation(
        labels=("i", "j", "k", "l"),
        va=("i", "j"),
        wa=("k", "l"),
        elements=elements,
        generators=(cross,),
        sizes=(3, 5, 3, 5),
        visible_positions=(0, 1),
        partition_budget=0,
    )
    assert result.count is None
    assert result.regime_id == "unavailable"


def test_dispatcher_records_full_refused_trace_before_fired():
    """Z_2 cross-swap causes functional refused, singleton refused, young refused,
    partitionCount fired."""
    cross = Permutation([2, 3, 0, 1])
    elements = _dimino((cross,))
    result = compute_accumulation(
        labels=("i", "j", "k", "l"),
        va=("i", "j"),
        wa=("k", "l"),
        elements=elements,
        generators=(cross,),
        sizes=(3, 5, 3, 5),
        visible_positions=(0, 1),
    )
    assert result.regime_id == "partitionCount"
    decisions_by_id = {step.regime_id: step.decision for step in result.trace}
    assert decisions_by_id["functionalProjection"] == "refused"
    assert decisions_by_id["singleton"] == "refused"
    assert decisions_by_id["young"] == "refused"
    assert decisions_by_id["partitionCount"] == "fired"
