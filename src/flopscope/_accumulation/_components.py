"""Connected-component decomposition of G_pt's label-interaction graph.

Port of website/components/symmetry-aware-einsum-contractions/engine/componentDecomposition.js.

Each component gets its own restricted group (generators + dimino closure), V/W split,
sizes, and visible_positions. The cost orchestrator invokes the regime ladder per component.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from flopscope._perm_group import _dimino
from flopscope._perm_group import _Permutation as Permutation

from ._detection import DetectedGroup, _classify_group_name


@dataclass(frozen=True)
class Component:
    """One independent block of G_pt's action on labels."""

    indices: tuple[int, ...]  # positions in all_labels
    labels: tuple[str, ...]
    va: tuple[str, ...]
    wa: tuple[str, ...]
    sizes: tuple[int, ...]
    visible_positions: tuple[int, ...]
    generators: tuple[Permutation, ...]
    elements: tuple[Permutation, ...]
    order: int
    group_name: str


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


def _build_label_interaction_components(
    all_labels: Sequence[str],
    generators: Sequence[Permutation],
) -> list[list[int]]:
    """Connected components of the label-interaction graph: labels are connected
    if any single generator moves them together."""
    n = len(all_labels)
    uf = _UnionFind(n)
    for gen in generators:
        moved = [i for i in range(n) if gen.array_form[i] != i]
        for j in range(1, len(moved)):
            uf.union(moved[0], moved[j])
        for i in range(n):
            target = gen.array_form[i]
            if target != i:
                uf.union(i, target)

    component_map: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        component_map.setdefault(root, []).append(i)
    return [sorted(indices) for indices in component_map.values()]


def _restrict_permutation(
    perm: Permutation, indices: Sequence[int]
) -> Permutation | None:
    """Restrict perm to the given global indices. Returns None if perm doesn't preserve
    the index set."""
    local_idx = {global_idx: local_pos for local_pos, global_idx in enumerate(indices)}
    arr: list[int] = []
    for global_idx in indices:
        target = perm.array_form[global_idx]
        if target not in local_idx:
            return None
        arr.append(local_idx[target])
    return Permutation(arr)


def decompose_into_components(
    *,
    detected_group: DetectedGroup,
    v_labels: frozenset[str],
    w_labels: frozenset[str],
    sizes: Sequence[int],
) -> tuple[Component, ...]:
    """Pure: G_pt + V/W → independent components. Used by einsum and (future) reduction code paths."""
    all_labels = detected_group.all_labels
    raw_components = _build_label_interaction_components(
        all_labels, detected_group.generators
    )

    components: list[Component] = []
    for indices in raw_components:
        labels = tuple(all_labels[i] for i in indices)
        va = tuple(lbl for lbl in labels if lbl in v_labels)
        wa = tuple(lbl for lbl in labels if lbl in w_labels)
        comp_sizes = tuple(sizes[i] for i in indices)
        visible_positions = tuple(labels.index(lbl) for lbl in va)

        restricted_gens: list[Permutation] = []
        seen_keys: set[tuple[int, ...]] = set()
        for gen in detected_group.generators:
            r = _restrict_permutation(gen, indices)
            if r is None:
                continue
            if r.is_identity:
                continue
            key = tuple(r.array_form)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            restricted_gens.append(r)

        if restricted_gens:
            elements = _dimino(tuple(restricted_gens))
        else:
            elements = (Permutation.identity(len(indices)),)

        order = len(elements)
        group_name = _classify_group_name(labels, restricted_gens, elements)

        components.append(
            Component(
                indices=tuple(indices),
                labels=labels,
                va=va,
                wa=wa,
                sizes=comp_sizes,
                visible_positions=visible_positions,
                generators=tuple(restricted_gens),
                elements=tuple(elements),
                order=order,
                group_name=group_name,
            )
        )

    return tuple(components)
