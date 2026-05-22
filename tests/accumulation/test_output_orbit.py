"""Tests for _output_orbit.py — port of outputOrbit.js."""

from flopscope._accumulation._output_orbit import (
    apply_permutation_to_tuple_array,
    canonical_tuple_under_group,
    preserves_position_set,
    projection_is_functional,
    restrict_stabilizer_to_positions,
    restrict_to_positions,
    tuple_array_key,
    visible_tuple_from_full_tuple,
)
from flopscope._perm_group import _Permutation as Permutation


def test_tuple_array_key_uses_pipe_separator():
    assert tuple_array_key((0, 1, 2)) == "0|1|2"
    assert tuple_array_key(()) == ""


def test_preserves_position_set_returns_true_when_set_invariant():
    swap = Permutation([1, 0, 2])  # swap positions 0,1
    assert preserves_position_set(swap, (0, 1)) is True
    assert preserves_position_set(swap, (0,)) is False  # 0 → 1, leaves the set


def test_apply_permutation_to_tuple_array_uses_source_to_target():
    # Convention: out[perm.arr[source]] = tuple[source]
    perm = Permutation([1, 2, 0])  # 0→1, 1→2, 2→0
    tup = ("a", "b", "c")
    result = apply_permutation_to_tuple_array(tup, perm)
    # tup[0]='a' goes to position 1; tup[1]='b' goes to position 2; tup[2]='c' goes to position 0
    assert result == ["c", "a", "b"]


def test_visible_tuple_extracts_visible_positions():
    full = (10, 20, 30, 40)
    visible = visible_tuple_from_full_tuple(full, (0, 2))
    assert visible == [10, 30]


def test_restrict_to_positions_returns_local_permutation():
    # Identity on global indices → identity on local positions
    perm = Permutation([2, 1, 0])  # swaps 0 and 2
    restricted = restrict_to_positions(perm, (0, 2))
    assert restricted is not None
    assert tuple(restricted.array_form) == (1, 0)


def test_restrict_to_positions_returns_none_when_set_not_preserved():
    perm = Permutation([1, 0, 2])  # 0 ↔ 1
    assert restrict_to_positions(perm, (0, 2)) is None


def test_restrict_stabilizer_to_positions_dedupes_kernel():
    # Two GLOBALLY distinct permutations that both restrict to the local identity
    # on V = {0, 1} should dedupe to a single local element.
    # p_identity is the global identity.
    # p_outside swaps positions 2 and 3 (outside V), so it preserves V pointwise
    # and its restriction to V is the local identity.
    p_identity = Permutation([0, 1, 2, 3])
    p_outside = Permutation([0, 1, 3, 2])  # swaps indices 2 and 3
    result = restrict_stabilizer_to_positions((p_identity, p_outside), (0, 1))
    assert len(result) == 1
    assert result[0].is_identity


def test_restrict_stabilizer_empty_positions_returns_identity():
    p = Permutation([1, 0])
    result = restrict_stabilizer_to_positions((p,), ())
    assert len(result) == 1
    assert result[0].size == 0


def test_canonical_tuple_under_group_picks_lex_min():
    swap = Permutation([1, 0])
    elements = (Permutation([0, 1]), swap)
    assert canonical_tuple_under_group([2, 1], elements) == "1|2"
    assert canonical_tuple_under_group([1, 2], elements) == "1|2"


def test_canonical_tuple_under_empty_group_returns_input_key():
    assert canonical_tuple_under_group([3, 1, 2], ()) == "3|1|2"


def test_projection_is_functional_when_all_g_preserve_v():
    # Two perms, both fixing positions {0, 1}
    p1 = Permutation([1, 0, 2])  # swaps 0,1; preserves {0,1}
    p2 = Permutation([0, 1, 2])  # identity
    assert projection_is_functional((p1, p2), (0, 1)) is True


def test_projection_is_not_functional_when_some_g_moves_v_to_w():
    cycle = Permutation([1, 2, 0])  # 0→1→2→0
    assert projection_is_functional((cycle,), (0, 1)) is False
