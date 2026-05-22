"""H = Stab_G(V)|_V helpers and canonical tuple operations.

Port of website/components/symmetry-aware-einsum-contractions/engine/outputOrbit.js.

Permutation convention matches `_perm_group.py`: source -> target arrays.
For tuple action: out[perm.arr[source]] = tuple[source].
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from flopscope._perm_group import _Permutation as Permutation


def tuple_array_key(tup: Sequence) -> str:
    """Stable string key for a tuple, used for orbit dedup. Mirrors JS tupleArrayKey."""
    return "|".join(str(v) for v in tup)


def preserves_position_set(perm: Permutation, positions: Sequence[int]) -> bool:
    """True iff perm maps the position set into itself (setwise stabilizer test)."""
    position_set = set(positions)
    for source in positions:
        if perm.array_form[source] not in position_set:
            return False
    return True


def restrict_to_positions(
    perm: Permutation, positions: Sequence[int]
) -> Permutation | None:
    """Restrict perm to the given positions, returning a local-coordinate Permutation.
    Returns None when perm doesn't preserve the position set.
    """
    if not preserves_position_set(perm, positions):
        return None
    local_index = {
        global_pos: local_pos for local_pos, global_pos in enumerate(positions)
    }
    arr = [local_index[perm.array_form[global_source]] for global_source in positions]
    return Permutation(arr)


def restrict_stabilizer_to_positions(
    elements: Iterable[Permutation], positions: Sequence[int]
) -> tuple[Permutation, ...]:
    """Restrict every G element that preserves `positions` to the local action on `positions`.
    Deduplicates by string key (kernel of restriction collapses to one local element)."""
    degree = len(positions)
    if degree == 0:
        return (Permutation.identity(0),)

    by_key: dict[str, Permutation] = {}
    for element in elements:
        restricted = restrict_to_positions(element, positions)
        if restricted is not None:
            by_key[",".join(str(v) for v in restricted.array_form)] = restricted

    if not by_key:
        identity = Permutation.identity(degree)
        return (identity,)

    return tuple(by_key.values())


def apply_permutation_to_tuple_array(tup: Sequence, perm: Permutation) -> list:
    """Apply perm to a tuple under the source-to-target convention:
    out[perm.array_form[source]] = tup[source].
    Mirrors JS applyPermutationToTupleArray.
    """
    next_tuple: list = [None] * len(tup)
    for source in range(len(tup)):
        next_tuple[perm.array_form[source]] = tup[source]
    return next_tuple


def canonical_tuple_under_group(tup: Sequence, elements: Iterable[Permutation]) -> str:
    """Return the lex-smallest tuple key over the orbit of `tup` under `elements`."""
    elements_tuple = tuple(elements)
    if not elements_tuple:
        return tuple_array_key(tup)
    best: str | None = None
    for element in elements_tuple:
        moved = apply_permutation_to_tuple_array(tup, element)
        key = tuple_array_key(moved)
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def visible_tuple_from_full_tuple(
    full_tuple: Sequence, visible_positions: Sequence[int]
) -> list:
    """Project a full assignment tuple to its visible-label coordinates."""
    return [full_tuple[position] for position in visible_positions]


def projection_is_functional(
    elements: Iterable[Permutation], visible_positions: Sequence[int]
) -> bool:
    """True iff every g in elements preserves the visible position set as a set.
    When True, projection π_V descends to a well-defined map X/G → Y/H.
    """
    return all(preserves_position_set(g, visible_positions) for g in elements)
