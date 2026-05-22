"""Bipartite graph + incidence matrix construction for the σ-loop.

Port of website/components/symmetry-aware-einsum-contractions/engine/algorithm.js
(buildBipartite + buildIncidenceMatrix only — runSigmaLoop lives in _detection.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class UVertex:
    op_idx: int
    class_id: int
    labels: frozenset[str]


@dataclass(frozen=True)
class BipartiteGraph:
    u_vertices: tuple[UVertex, ...]
    incidence: tuple[dict[str, int], ...]  # one per u-vertex
    u_operand: tuple[int, ...]
    operand_labels: tuple[frozenset[str], ...]
    all_labels: tuple[str, ...]  # sorted union
    free_labels: frozenset[str]
    summed_labels: frozenset[str]
    identical_groups: tuple[tuple[int, ...], ...]
    num_operands: int


@dataclass(frozen=True)
class IncidenceMatrix:
    matrix: tuple[tuple[int, ...], ...]
    labels: tuple[str, ...]
    col_fingerprints: dict[str, tuple[int, ...]]
    fp_to_labels: dict[tuple[int, ...], frozenset[str]]


def build_bipartite(
    *,
    subscripts: Sequence[str],
    output: str,
    operand_names: Sequence[str],
) -> BipartiteGraph:
    """Build the bipartite graph: one U-vertex per axis of each operand.

    Mirrors algorithm.js#buildBipartite. No axis merging — per-operand symmetry
    is handled later by the σ-loop's wreath enumeration, not by collapsing axes here.
    """
    num_ops = len(subscripts)
    u_vertices: list[UVertex] = []
    incidence: list[dict[str, int]] = []
    u_operand: list[int] = []
    operand_labels: list[frozenset[str]] = []

    for op_idx, sub in enumerate(subscripts):
        operand_labels.append(frozenset(sub))
        for axis_idx, ch in enumerate(sub):
            u_vertices.append(
                UVertex(op_idx=op_idx, class_id=axis_idx, labels=frozenset({ch}))
            )
            incidence.append({ch: 1})
            u_operand.append(op_idx)

    all_labels_set: set[str] = set()
    for sub in subscripts:
        all_labels_set.update(sub)
    all_labels = tuple(sorted(all_labels_set))
    output_set = set(output)
    free_labels = frozenset(lbl for lbl in all_labels if lbl in output_set)
    summed_labels = frozenset(lbl for lbl in all_labels if lbl not in output_set)

    # Group operand positions by name (Python id() equivalent at the einsum-level
    # is "same operand_name").
    name_to_positions: dict[str, list[int]] = {}
    for i, name in enumerate(operand_names):
        name_to_positions.setdefault(name, []).append(i)
    identical_groups = tuple(
        tuple(positions)
        for positions in name_to_positions.values()
        if len(positions) >= 2
    )

    return BipartiteGraph(
        u_vertices=tuple(u_vertices),
        incidence=tuple(incidence),
        u_operand=tuple(u_operand),
        operand_labels=tuple(operand_labels),
        all_labels=all_labels,
        free_labels=free_labels,
        summed_labels=summed_labels,
        identical_groups=identical_groups,
        num_operands=num_ops,
    )


def build_incidence_matrix(graph: BipartiteGraph) -> IncidenceMatrix:
    """Build the dense incidence matrix and column fingerprints.

    Mirrors algorithm.js#buildIncidenceMatrix.
    """
    labels = graph.all_labels
    matrix = tuple(
        tuple(graph.incidence[row_idx].get(lbl, 0) for lbl in labels)
        for row_idx in range(len(graph.u_vertices))
    )
    col_fingerprints: dict[str, tuple[int, ...]] = {}
    fp_to_labels_mut: dict[tuple[int, ...], set[str]] = {}
    for c, label in enumerate(labels):
        fp = tuple(row[c] for row in matrix)
        col_fingerprints[label] = fp
        fp_to_labels_mut.setdefault(fp, set()).add(label)
    fp_to_labels = {fp: frozenset(s) for fp, s in fp_to_labels_mut.items()}

    return IncidenceMatrix(
        matrix=matrix,
        labels=labels,
        col_fingerprints=col_fingerprints,
        fp_to_labels=fp_to_labels,
    )
