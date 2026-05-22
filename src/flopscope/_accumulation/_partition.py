"""Typed set partition utilities for the partitionCount regime.

Port of website/components/symmetry-aware-einsum-contractions/engine/partition/typedPartitions.js.

A typed equality pattern is a partition of label positions where blocks may only merge
positions sharing a domain class (same numeric size n_s). The partitionCount regime sums
over typed-partition orbits; each contribution is `falling_factorial(n_s, b_s) / |Ḡ_x̃|`
times the number of stored output representatives reachable.

This module currently provides the basic utilities. Orbit dedup and induced-block-action
helpers land in Task 6.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence


def falling_factorial(n: int, m: int) -> int:
    """Falling factorial n*(n-1)*...*(n-m+1). Zero when m > n; identity 1 when m = 0."""
    if m < 0:
        raise ValueError(f"falling_factorial received negative m={m}")
    if m > n:
        return 0
    result = 1
    for i in range(m):
        result *= n - i
    return result


def normalize_partition(partition: Sequence[int]) -> list[int]:
    """Renumber block IDs by first appearance, so equivalent partitions get equal lists."""
    remap: dict[int, int] = {}
    next_id = 0
    out: list[int] = []
    for block in partition:
        if block not in remap:
            remap[block] = next_id
            next_id += 1
        out.append(remap[block])
    return out


def partition_key(partition: Sequence[int]) -> str:
    """Stable key for partition equality, after normalization. Mirrors JS partitionKey."""
    return "|".join(str(b) for b in normalize_partition(partition))


def num_blocks(partition: Sequence[int]) -> int:
    """Number of distinct block IDs in `partition`."""
    return len(set(partition))


def block_domains(partition: Sequence[int], sizes: Sequence[int]) -> dict[int, int]:
    """Map block ID → common domain size of all positions in that block.
    Raises ValueError if any block contains positions of unequal size.
    """
    domains: dict[int, int] = {}
    for position, block in enumerate(partition):
        domain = sizes[position]
        existing = domains.get(block)
        if existing is not None and existing != domain:
            raise ValueError(
                f"partition block {block} mixes dimensions {existing} and {domain}"
            )
        domains[block] = domain
    return domains


def typed_labeling_count(partition: Sequence[int], sizes: Sequence[int]) -> int:
    """Number of injective concrete labelings of a typed partition.

    For each domain class with ``b_s`` blocks, contributes ``falling(n_s, b_s)``;
    multiplied across domains.
    """
    domains = block_domains(partition, sizes)
    counts_by_domain: dict[int, int] = {}
    for domain in domains.values():
        counts_by_domain[domain] = counts_by_domain.get(domain, 0) + 1

    result = 1
    for domain_size, block_count in counts_by_domain.items():
        result *= falling_factorial(domain_size, block_count)
    return result


def generate_typed_set_partitions(
    sizes: Sequence[int],
) -> list[list[int]]:
    """Enumerate all typed equality patterns over `sizes`.

    A typed partition only merges positions with equal `sizes[i]`. Returns a
    list of normalized partitions (block IDs renumbered by first appearance).
    Result is deduplicated by partition_key.

    Cached by sizes tuple — most repeated einsum calls share shapes.
    """
    return _generate_typed_set_partitions_cached(tuple(sizes))


@functools.lru_cache(maxsize=256)
def _generate_typed_set_partitions_cached(
    sizes: tuple[int, ...],
) -> list[list[int]]:
    results: list[list[int]] = []
    current: list[int] = []

    def visit(position: int, block_count: int) -> None:
        if position == len(sizes):
            results.append(normalize_partition(current))
            return
        # Try merging into each existing block of the same domain
        for block in range(block_count):
            first_position_in_block = next(
                (i for i, b in enumerate(current) if b == block), -1
            )
            if (
                first_position_in_block >= 0
                and sizes[first_position_in_block] == sizes[position]
            ):
                current.append(block)
                visit(position + 1, block_count)
                current.pop()
        # Open a fresh block
        current.append(block_count)
        visit(position + 1, block_count + 1)
        current.pop()

    visit(0, 0)

    # Deduplicate by normalized key (the JS uses Map<key, partition>).
    by_key: dict[str, list[int]] = {}
    for partition in results:
        by_key[partition_key(partition)] = partition
    return list(by_key.values())


# ── Orbit and induced-block-action utilities ─────────────────────────────────

from collections.abc import Iterable

from flopscope._perm_group import _Permutation as Permutation

from ._output_orbit import apply_permutation_to_tuple_array


def inverse_array(perm: Permutation) -> list[int]:
    """Return the inverse of `perm` as an array (target -> source)."""
    inv = [0] * perm.size
    for source, target in enumerate(perm.array_form):
        inv[target] = source
    return inv


def apply_permutation_to_partition(
    partition: Sequence[int], perm: Permutation
) -> list[int]:
    """Apply perm to a partition's POSITIONS (not block IDs).

    Convention: position p in the result holds the block of position perm⁻¹(p).
    Mirrors the JS `applyPermutationToPartition`.
    """
    moved: list[int | None] = [None] * len(partition)
    for source in range(len(partition)):
        moved[perm.array_form[source]] = partition[source]
    # All slots filled by construction (perm is a bijection); cast tightens type.
    return normalize_partition(list(moved))  # type: ignore[arg-type]


def partition_orbit_reps(
    partitions: Sequence[Sequence[int]], elements: Iterable[Permutation]
) -> list[list[int]]:
    """Return one representative per G-orbit on partitions, in input order.
    Mirrors JS partitionOrbitReps using a dict-by-key dedup pattern.
    """
    elements_tuple = tuple(elements)
    remaining: dict[str, list[int]] = {partition_key(p): list(p) for p in partitions}
    reps: list[list[int]] = []
    for key, partition in list(remaining.items()):
        if key not in remaining:
            continue
        reps.append(partition)
        for element in elements_tuple:
            moved = apply_permutation_to_partition(partition, element)
            remaining.pop(partition_key(moved), None)
    return reps


def induced_block_permutation(
    partition: Sequence[int], perm: Permutation
) -> str | None:
    """Return the action of `perm` on `partition`'s blocks, encoded as a string key.
    Returns None if `perm` doesn't preserve `partition` (i.e. its image differs).
    """
    moved = apply_permutation_to_partition(partition, perm)
    if partition_key(moved) != partition_key(partition):
        return None

    representative_by_block: dict[int, int] = {}
    for position, block in enumerate(partition):
        if block not in representative_by_block:
            representative_by_block[block] = position

    blocks = sorted(representative_by_block.keys())
    arr: list[int] = []
    for block in blocks:
        source_position = representative_by_block[block]
        target_position = perm.array_form[source_position]
        arr.append(partition[target_position])
    return "|".join(str(b) for b in arr)


def induced_block_action_size(
    partition: Sequence[int], elements: Iterable[Permutation]
) -> int:
    """Size of the IMAGE of Stab_G(partition) on the blocks (not the raw stabilizer order).

    This is the |Ḡ_x̃| in the partition-count formula. Generators that permute positions
    within a block act trivially on blocks, so |Ḡ_x̃| ≤ |Stab_G(x̃)|. Using the wrong one
    breaks the integer division ∏_s (n_s)_{b_s(x̃)} / |Ḡ_x̃|.
    """
    actions: set[str] = set()
    for element in elements:
        action_key = induced_block_permutation(partition, element)
        if action_key is not None:
            actions.add(action_key)
    return len(actions) or 1


def map_key(map_array: Sequence[int]) -> str:
    """Stable key for an induced prefix map array."""
    return "|".join(str(v) for v in map_array)


def map_array_from_key(key: str) -> list[int]:
    """Inverse of map_key."""
    if key == "":
        return []
    return [int(part) for part in key.split("|")]


def induced_prefix_map(
    partition: Sequence[int],
    perm: Permutation,
    visible_positions: Sequence[int],
) -> list[int]:
    """For each visible position v, return the block of perm⁻¹(v) in `partition`.

    Mirrors JS `inducedPrefixMap`. The prefix-map captures which input-block each
    visible coordinate's preimage belongs to under `perm`.
    """
    inv = inverse_array(perm)
    return [partition[inv[visible_position]] for visible_position in visible_positions]


def induced_prefix_maps(
    partition: Sequence[int],
    elements: Iterable[Permutation],
    visible_positions: Sequence[int],
) -> frozenset[str]:
    """Set of induced prefix-map keys across all elements of G."""
    return frozenset(
        map_key(induced_prefix_map(partition, element, visible_positions))
        for element in elements
    )


def _act_on_map_by_output_permutation(
    map_array: Sequence[int], h_element: Permutation
) -> list[int]:
    """Apply an output-side permutation h to a prefix map. Same convention as tuple action."""
    return apply_permutation_to_tuple_array(list(map_array), h_element)


def count_map_orbits_under_h(
    map_keys: Iterable[str], h_elements: Iterable[Permutation]
) -> int:
    """Count orbits of H acting on the prefix-map set. Used as |A_x̃ / H_a| in the
    partition-count formula.
    """
    h_tuple = tuple(h_elements)
    remaining = set(map_keys)
    count = 0
    for key in list(remaining):
        if key not in remaining:
            continue
        count += 1
        map_array = map_array_from_key(key)
        for h in h_tuple:
            remaining.discard(map_key(_act_on_map_by_output_permutation(map_array, h)))
    return count
