"""Tests for _partition.py — port of partition/typedPartitions.js (Task 5: basic utilities)."""

import pytest

from flopscope._accumulation._partition import (
    block_domains,
    falling_factorial,
    generate_typed_set_partitions,
    normalize_partition,
    num_blocks,
    partition_key,
    typed_labeling_count,
)


def test_falling_factorial_base_cases():
    assert falling_factorial(5, 0) == 1
    assert falling_factorial(5, 1) == 5
    assert falling_factorial(5, 2) == 20  # 5 * 4
    assert falling_factorial(5, 3) == 60  # 5 * 4 * 3


def test_falling_factorial_zero_when_m_exceeds_n():
    assert falling_factorial(1, 2) == 0
    assert falling_factorial(3, 5) == 0


def test_falling_factorial_rejects_negative_m():
    with pytest.raises(ValueError, match="negative m"):
        falling_factorial(5, -1)


def test_normalize_partition_renumbers_blocks_by_first_appearance():
    assert normalize_partition([2, 5, 2, 5]) == [0, 1, 0, 1]
    assert normalize_partition([0, 1, 2]) == [0, 1, 2]
    assert normalize_partition([3, 3, 3]) == [0, 0, 0]


def test_partition_key_uses_normalized_form():
    assert partition_key([2, 5, 2, 5]) == "0|1|0|1"
    assert partition_key([0, 1, 0, 1]) == "0|1|0|1"  # already normalized


def test_num_blocks_counts_distinct_block_ids():
    assert num_blocks([0, 1, 0, 1]) == 2
    assert num_blocks([0, 1, 2, 3]) == 4
    assert num_blocks([0, 0, 0]) == 1


def test_block_domains_groups_by_size():
    # Partition [0, 1, 0, 1] over sizes [3, 5, 3, 5] → block 0 has size 3, block 1 has size 5
    domains = block_domains([0, 1, 0, 1], (3, 5, 3, 5))
    assert domains == {0: 3, 1: 5}


def test_block_domains_rejects_mixed_sizes():
    # Block 0 mixes positions of sizes 3 and 5 — invalid typed partition.
    with pytest.raises(ValueError, match="mixes dimensions"):
        block_domains([0, 0, 1], (3, 5, 4))


def test_typed_labeling_count_uniform_sizes():
    # Two blocks, both domain 5: 5 * 4 = 20
    assert typed_labeling_count([0, 1], (5, 5)) == 20


def test_typed_labeling_count_mixed_domains():
    # Two blocks of domain 3 (need 3*2=6) and one block of domain 5 (need 5)
    # Partition [0,1,2] over (3,3,5): block 0 size 3, block 1 size 3, block 2 size 5
    # countsByDomain: {3: 2, 5: 1}
    # falling(3, 2) * falling(5, 1) = 6 * 5 = 30
    assert typed_labeling_count([0, 1, 2], (3, 3, 5)) == 30


def test_generate_typed_set_partitions_disjoint_when_sizes_differ():
    # Sizes (2, 3) — different domains can't merge → only the discrete partition.
    partitions = generate_typed_set_partitions((2, 3))
    assert partitions == [[0, 1]]


def test_generate_typed_set_partitions_full_when_sizes_match():
    # Sizes (4, 4) — both partitions of 2 are valid.
    partitions = generate_typed_set_partitions((4, 4))
    keys = sorted(partition_key(p) for p in partitions)
    assert keys == ["0|0", "0|1"]


def test_generate_typed_set_partitions_three_position_uniform():
    # Sizes (n, n, n) — all 5 set partitions of {1,2,3} (Bell(3) = 5).
    partitions = generate_typed_set_partitions((4, 4, 4))
    keys = sorted(partition_key(p) for p in partitions)
    assert keys == ["0|0|0", "0|0|1", "0|1|0", "0|1|1", "0|1|2"]


def test_generate_typed_set_partitions_heterogeneous_three():
    # Sizes (3, 3, 5) — block of size-3 positions can merge or not; size-5 stands alone.
    # Possible partitions: (0,1,2)=all-distinct, (0,0,1)=first two merge.
    partitions = generate_typed_set_partitions((3, 3, 5))
    keys = sorted(partition_key(p) for p in partitions)
    assert keys == ["0|0|1", "0|1|2"]


# ── Task 6 additions ──────────────────────────────────────────────────────


from flopscope._accumulation._partition import (
    apply_permutation_to_partition,
    count_map_orbits_under_h,
    induced_block_action_size,
    induced_block_permutation,
    induced_prefix_map,
    induced_prefix_maps,
    inverse_array,
    map_array_from_key,
    map_key,
    partition_orbit_reps,
)
from flopscope._perm_group import _Permutation as Permutation


def test_inverse_array_inverts():
    perm = Permutation([2, 0, 1])
    inv = inverse_array(perm)
    # perm: 0→2, 1→0, 2→1. Inverse: 2→0, 0→1, 1→2 → [1, 2, 0]
    assert inv == [1, 2, 0]


def test_apply_permutation_to_partition_relabels_positions():
    # Partition [0, 1, 0] under swap(0,1): position 0 gets the block from position 1 (=1),
    # position 1 gets the block from position 0 (=0), position 2 stays = 0.
    # After move: [1, 0, 0]. Normalized: [0, 1, 1].
    swap = Permutation([1, 0, 2])
    result = apply_permutation_to_partition([0, 1, 0], swap)
    assert result == [0, 1, 1]


def test_partition_orbit_reps_collapses_under_s2():
    # Two-block partitions of two equal-size positions: [0,0] and [0,1].
    # Under S_2 they're each fixed (swap doesn't change all-merged or all-split).
    s2 = Permutation([1, 0])
    elements = (Permutation.identity(2), s2)
    partitions = [[0, 0], [0, 1]]
    reps = partition_orbit_reps(partitions, elements)
    assert len(reps) == 2  # both fixed; both are their own rep


def test_induced_block_permutation_returns_None_when_partition_not_fixed():
    # [0, 1, 0] under swap(0,1): becomes [1, 0, 0] → normalized [0, 1, 1] ≠ [0, 1, 0].
    swap = Permutation([1, 0, 2])
    assert induced_block_permutation([0, 1, 0], swap) is None


def test_induced_block_permutation_returns_block_perm_when_fixed():
    # Partition [0, 0, 1] under swap(0, 1): block 0 stays (positions 0,1 in same block).
    # Block 1 stays (position 2 fixed).
    swap = Permutation([1, 0, 2])
    result = induced_block_permutation([0, 0, 1], swap)
    # Block 0's representative is position 0; perm sends 0→1, position 1's block is 0.
    # Block 1's representative is position 2; perm sends 2→2, position 2's block is 1.
    # Block perm: [0, 1] → identity on blocks.
    assert result == "0|1"


def test_induced_block_action_size_uses_image_not_raw_stabilizer():
    # Partition [0, 0]: both positions in one block. Any swap stays in the same partition,
    # but its action on the single block is identity. Block action size = 1.
    swap = Permutation([1, 0])
    elements = (Permutation.identity(2), swap)
    assert induced_block_action_size([0, 0], elements) == 1


def test_induced_block_action_size_for_distinguishable_blocks():
    # Partition [0, 1]: swap permutes the two blocks. Block action size = 2.
    swap = Permutation([1, 0])
    elements = (Permutation.identity(2), swap)
    assert induced_block_action_size([0, 1], elements) == 2


def test_induced_prefix_map_uses_inverse_perm():
    # Partition [0, 1, 2], swap of positions (0, 1), visible positions (0, 1).
    # inv = [1, 0, 2]; prefix = [partition[inv[0]], partition[inv[1]]] = [partition[1], partition[0]] = [1, 0]
    swap = Permutation([1, 0, 2])
    assert induced_prefix_map([0, 1, 2], swap, (0, 1)) == [1, 0]


def test_induced_prefix_maps_dedupes_via_keys():
    # Trivial action — every permutation maps the partition to the same prefix.
    identity = Permutation.identity(3)
    maps = induced_prefix_maps([0, 1, 2], (identity,), (0, 1))
    assert maps == frozenset(["0|1"])


def test_map_key_and_map_array_from_key_round_trip():
    assert map_key([1, 0, 2]) == "1|0|2"
    assert map_array_from_key("1|0|2") == [1, 0, 2]
    assert map_array_from_key("") == []


def test_count_map_orbits_under_h_partitions_under_action():
    # H = S_2 acting on 2 visible positions. Maps {"0|1", "1|0", "0|2"}.
    # Under H = {id, swap}: "0|1" ↔ "1|0" (one orbit), "0|2" alone.
    h_swap = Permutation([1, 0])
    h_id = Permutation.identity(2)
    h_elements = (h_id, h_swap)
    maps = frozenset(["0|1", "1|0", "0|2"])
    assert count_map_orbits_under_h(maps, h_elements) == 2
