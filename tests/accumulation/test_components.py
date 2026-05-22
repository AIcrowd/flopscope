"""Tests for _components.py — port of componentDecomposition.js."""

from flopscope._accumulation._components import decompose_into_components
from flopscope._accumulation._detection import DetectedGroup
from flopscope._perm_group import _dimino
from flopscope._perm_group import _Permutation as Permutation


def test_decompose_trivial_group_each_label_its_own_component():
    """Trivial G → each label is its own component."""
    detected = DetectedGroup(
        all_labels=("i", "j", "k"),
        generators=(),
        elements=(Permutation.identity(3),),
        group_name="trivial",
        action_summary="trivial",
        valid_pi_results=(),
    )
    components = decompose_into_components(
        detected_group=detected,
        v_labels=frozenset({"i", "k"}),
        w_labels=frozenset({"j"}),
        sizes=(3, 4, 5),
    )
    assert len(components) == 3
    by_labels = {c.labels: c for c in components}
    assert ("i",) in by_labels
    assert ("j",) in by_labels
    assert ("k",) in by_labels


def test_decompose_s2_groups_two_labels_into_one_component():
    """S_2 on (i, j), with k as a free label → 2 components: {i,j} and {k}."""
    swap = Permutation([1, 0, 2])
    elements = _dimino((swap,))
    detected = DetectedGroup(
        all_labels=("i", "j", "k"),
        generators=(swap,),
        elements=tuple(elements),
        group_name="S2{i,j}",
        action_summary="V-only",
        valid_pi_results=(),
    )
    components = decompose_into_components(
        detected_group=detected,
        v_labels=frozenset({"i", "j"}),
        w_labels=frozenset({"k"}),
        sizes=(4, 4, 5),
    )
    assert len(components) == 2


def test_component_carries_va_wa_and_visible_positions():
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    detected = DetectedGroup(
        all_labels=("i", "j"),
        generators=(swap,),
        elements=tuple(elements),
        group_name="S2{i,j}",
        action_summary="cross-V/W",
        valid_pi_results=(),
    )
    components = decompose_into_components(
        detected_group=detected,
        v_labels=frozenset({"i"}),
        w_labels=frozenset({"j"}),
        sizes=(6, 6),
    )
    assert len(components) == 1
    c = components[0]
    assert c.labels == ("i", "j")
    assert c.va == ("i",)
    assert c.wa == ("j",)
    assert c.visible_positions == (0,)


def test_component_restricted_generators_have_local_degree():
    """A direct product G = S_2 × S_2 on (i,j) and (k,l) decomposes into two S_2 components."""
    swap_ij = Permutation([1, 0, 2, 3])
    swap_kl = Permutation([0, 1, 3, 2])
    elements = _dimino((swap_ij, swap_kl))
    detected = DetectedGroup(
        all_labels=("i", "j", "k", "l"),
        generators=(swap_ij, swap_kl),
        elements=tuple(elements),
        group_name="S2{i,j}×S2{k,l}",
        action_summary="V-only × W-only",
        valid_pi_results=(),
    )
    components = decompose_into_components(
        detected_group=detected,
        v_labels=frozenset({"i", "j"}),
        w_labels=frozenset({"k", "l"}),
        sizes=(3, 3, 5, 5),
    )
    assert len(components) == 2
    for c in components:
        for gen in c.generators:
            assert gen.size == len(c.labels)
            assert gen.size == 2  # each component has 2 labels
