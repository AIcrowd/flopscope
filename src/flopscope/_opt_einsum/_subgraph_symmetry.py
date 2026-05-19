"""Subset-keyed subgraph symmetry detection for einsum intermediates.

One oracle per contract_path call. Given the original operand list,
subscript parts, per-operand declared symmetries, and output subscript,
builds a bipartite graph once and exposes ``.sym(subset)`` which returns
a ``SubsetSymmetry`` with ``.output`` (V-side) and ``.inner`` (W-side)
symmetries, computed lazily on first access and cached.

Each axis of each operand gets its own U-vertex in the bipartite graph
(no axis merging). The σ-loop iterates over generators from three sources:
(A) per-operand internal symmetry generators, (B) identical-operand swap
generators, and (C) coordinated axis relabeling for identical operands
with the same subscript (W-side only). Dimino's algorithm builds the full
row-permutation group from these generators, and π is derived for each
group element via column-fingerprint hash lookup.

See docs/explanation/subgraph-symmetry.md for the algorithm walkthrough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flopscope._perm_group import SymmetryGroup
from flopscope._perm_group import _Permutation as Perm

from ._symmetry import SubsetSymmetry

_MISSING = object()


def _derive_pi_canonical(
    sigma_col_of: dict[str, tuple[int, ...]],
    fp_to_labels: dict[tuple[int, ...], set[str]],
    v_labels: frozenset[str],
    w_labels: frozenset[str],
) -> dict[str, str] | None:
    """Build π by canonical hash lookup. Returns None on failure.

    For each label ℓ, looks up σ(M)'s column fingerprint in fp_to_labels
    and picks the lex-first unused candidate. Validates that π is a
    bijection and preserves the V/W partition.
    """
    pi: dict[str, str] = {}
    used: set[str] = set()
    all_labels = v_labels | w_labels

    for label in sorted(all_labels):
        fp = sigma_col_of[label]
        candidates = fp_to_labels.get(fp)
        if not candidates:
            return None
        pick = None
        for c in sorted(candidates):
            if c not in used:
                pick = c
                break
        if pick is None:
            return None
        pi[label] = pick
        used.add(pick)

    # Validate V→V and W→W.
    for lbl, target in pi.items():
        if lbl in v_labels and target not in v_labels:
            return None
        if lbl in w_labels and target not in w_labels:
            return None

    return pi


def _collect_pi_permutations(
    graph: EinsumBipartite,
    sub: _Subgraph,
    row_order: tuple[int, ...],
    col_of: dict[str, tuple[int, ...]],
    fp_to_labels: dict[tuple[int, ...], set[str]],
) -> tuple[list[Perm], list[Perm]]:
    """Collect V and W permutation generators via the expanded σ-loop.

    Generators come from two sources:

    Source A — per-operand internal symmetry generators.  For each operand
    in the subset that has declared groups, each generator's array form
    (mapped through ``group.axes``) permutes U-vertex positions within
    that operand's block.

    Source B — identical-operand swap generators.  For each pair of
    adjacent operands in an identical-operand group, an adjacent
    transposition that swaps their entire U-vertex blocks.

    For each generator, the induced column permutation π is derived via
    ``_derive_pi_canonical``.

    Returns
    -------
    v_perms : list[Perm]
        Non-identity permutations on V labels (output / free side).
    w_perms : list[Perm]
        Non-identity permutations on W labels (inner / summed side).
    """
    v_perms: list[Perm] = []
    w_perms: list[Perm] = []
    all_labels = sub.v_labels | sub.w_labels
    v_sorted = tuple(sorted(sub.v_labels))
    w_sorted = tuple(sorted(sub.w_labels))
    v_idx = {lbl: i for i, lbl in enumerate(v_sorted)}
    w_idx = {lbl: i for i, lbl in enumerate(w_sorted)}

    n_rows = len(row_order)
    identity_row = tuple(range(n_rows))

    # Map: operand_idx -> list of positions in row_order belonging to it.
    op_to_u_indices: dict[int, list[int]] = {}
    for pos, u_idx in enumerate(row_order):
        op = graph.u_operand[u_idx]
        op_to_u_indices.setdefault(op, []).append(pos)

    # Collect row-permutation generators, then derive π for each.
    row_perm_generators: list[tuple[int, ...]] = []

    # --- Source A: per-operand internal symmetry generators ---
    for op_idx in sorted({graph.u_operand[u] for u in row_order}):
        groups = graph.per_op_groups[op_idx]
        if groups is None:
            continue
        positions = op_to_u_indices.get(op_idx, [])
        if not positions:
            continue
        for group in groups:
            if group._labels is None:
                continue
            # Map group axis indices to positions within this operand's
            # block in row_order. group.axes[i] is the tensor axis index
            # that group position i acts on. Since we no longer merge,
            # each axis position in the subscript maps 1:1 to a U-vertex.
            # The operand's subscript gives us the axis->position mapping.
            subscript = graph.operand_subscripts[op_idx]
            # group._labels are the subscript chars the group acts on.
            # For each generator, we need to map its array_form through
            # the axis indirection to produce a row permutation.
            # Build: group_pos -> row_order position
            # group._labels[g_pos] is the char at group position g_pos.
            # We need to find which axis position in the subscript that
            # char corresponds to, respecting group.axes.
            if group.axes is not None:
                # group.axes[g_pos] = tensor axis index
                gpos_to_rowpos = {}
                for g_pos in range(group.degree):
                    axis_idx = group.axes[g_pos]
                    if axis_idx < len(positions):
                        gpos_to_rowpos[g_pos] = positions[axis_idx]
            elif group._labels is not None:
                # No axes, but _labels maps group positions to subscript
                # chars.  Find the subscript position of each label.
                gpos_to_rowpos = {}
                for g_pos in range(group.degree):
                    lbl = group._labels[g_pos]
                    # Find position of this label in the subscript
                    sub_pos = subscript.find(lbl)
                    if sub_pos >= 0 and sub_pos < len(positions):
                        gpos_to_rowpos[g_pos] = positions[sub_pos]
            else:
                # Default: group position i acts on operand axis i
                gpos_to_rowpos = {}
                for g_pos in range(group.degree):
                    if g_pos < len(positions):
                        gpos_to_rowpos[g_pos] = positions[g_pos]

            for gen in group.generators:
                arr = gen.array_form
                # Build row permutation: start from identity, then
                # permute the positions that this generator acts on.
                row_perm = list(identity_row)
                is_identity = True
                for g_pos in range(len(arr)):
                    if arr[g_pos] != g_pos:
                        src = gpos_to_rowpos.get(g_pos)
                        dst = gpos_to_rowpos.get(arr[g_pos])
                        if src is not None and dst is not None:
                            row_perm[src] = identity_row[dst]
                            is_identity = False
                if not is_identity:
                    row_perm_generators.append(tuple(row_perm))

    # --- Source B: identical-operand swap generators ---
    for group in sub.id_groups:
        group_sorted = sorted(group)
        for idx in range(len(group_sorted) - 1):
            op_a = group_sorted[idx]
            op_b = group_sorted[idx + 1]
            pos_a = op_to_u_indices.get(op_a, [])
            pos_b = op_to_u_indices.get(op_b, [])
            if len(pos_a) != len(pos_b):
                continue  # block sizes must match
            row_perm = list(identity_row)
            for pa, pb in zip(pos_a, pos_b, strict=False):
                row_perm[pa] = identity_row[pb]
                row_perm[pb] = identity_row[pa]
            row_perm_generators.append(tuple(row_perm))

    # --- Source C: coordinated axis relabeling for identical operands ---
    # When identical operands share the same subscript pattern, permuting
    # axes uniformly across all copies is equivalent to relabeling dummy
    # indices.  Only valid when BOTH labels involved in the swap are
    # summed (W-side) — relabeling free (V-side) labels changes the output.
    # Generate adjacent transpositions on W-only axis pairs, applied to
    # every copy simultaneously.
    for group in sub.id_groups:
        group_sorted = sorted(group)
        # Check all operands in this group have the same subscript
        subs_list = [graph.operand_subscripts[op] for op in group_sorted]
        if len(set(subs_list)) != 1:
            continue  # different subscripts — can't do coordinated relabeling
        subscript = subs_list[0]
        rank = len(subscript)
        if rank < 2:
            continue
        # Find which axis positions have W-only (summed) labels
        w_axes = [ax for ax in range(rank) if subscript[ax] in sub.w_labels]
        if len(w_axes) < 2:
            continue
        # Generate adjacent transpositions on W-only axes
        for idx in range(len(w_axes) - 1):
            ax_a = w_axes[idx]
            ax_b = w_axes[idx + 1]
            row_perm = list(identity_row)
            is_identity = True
            for op_idx in group_sorted:
                positions = op_to_u_indices.get(op_idx, [])
                if ax_a >= len(positions) or ax_b >= len(positions):
                    continue
                pa, pb = positions[ax_a], positions[ax_b]
                row_perm[pa] = identity_row[pb]
                row_perm[pb] = identity_row[pa]
                is_identity = False
            if not is_identity:
                row_perm_generators.append(tuple(row_perm))

    # --- Build a group on row positions and enumerate all elements ---
    if not row_perm_generators:
        return v_perms, w_perms

    row_gens = [Perm(list(g)) for g in row_perm_generators]
    row_group = SymmetryGroup(*row_gens)

    for sigma_perm in row_group.elements():
        sigma_row_perm = sigma_perm.array_form
        # Skip identity σ — it always gives π = identity.
        if all(sigma_row_perm[k] == k for k in range(n_rows)):
            continue

        # Compute σ(M)'s column fingerprints.
        sigma_col_of: dict[str, tuple[int, ...]] = {}
        for label in all_labels:
            sigma_col_of[label] = tuple(
                graph.incidence[row_order[sigma_row_perm[k]]].get(label, 0)
                for k in range(n_rows)
            )

        # Derive π.
        pi = _derive_pi_canonical(
            sigma_col_of, fp_to_labels, sub.v_labels, sub.w_labels
        )
        if pi is None:
            continue

        # Restrict π to V labels — emit Perm if non-identity.
        if sub.v_labels and any(pi.get(lbl, lbl) != lbl for lbl in sub.v_labels):
            arr = [v_idx[pi.get(lbl, lbl)] for lbl in v_sorted]
            v_perms.append(Perm(arr))

        # Restrict π to W labels — emit Perm if non-identity.
        if sub.w_labels and any(pi.get(lbl, lbl) != lbl for lbl in sub.w_labels):
            arr = [w_idx[pi.get(lbl, lbl)] for lbl in w_sorted]
            w_perms.append(Perm(arr))

    return v_perms, w_perms


@dataclass(frozen=True)
class EinsumBipartite:
    """Bipartite graph representation of an einsum expression.

    Left vertices U: one per (operand_idx, axis_position). Each axis of
    each operand gets its own U-vertex (no merging). For a dense operand
    with subscript "ai" we get two U vertices — one for axis 0 ({a}),
    one for axis 1 ({i}). A fully symmetric operand T with subscript
    "ij" also gets two U vertices — one per axis.

    Right vertices are labels, partitioned at the top level into
    free_labels (V, the final output) and summed_labels (W, contracted
    at the top level). Subset induction may reclassify labels from W
    to V when they cross the cut.
    """

    # Parallel tuples over U vertices:
    u_vertices: tuple[tuple[int, int], ...]  # (operand_idx, class_id)
    u_labels: tuple[frozenset[str], ...]  # which labels this class contains
    u_operand: tuple[int, ...]  # operand index
    incidence: tuple[dict[str, int], ...]  # {label -> multiplicity}

    # Right vertices, top-level partition:
    free_labels: frozenset[str]  # V at the top level
    summed_labels: frozenset[str]  # W at the top level

    # Python-identity groups: partition of [0..num_operands),
    # non-singleton blocks enumerate identical operands.
    identical_operand_groups: tuple[tuple[int, ...], ...]

    # Per-operand label set, needed for subset induction to compute
    # crossing labels efficiently without re-scanning incidence.
    operand_labels: tuple[frozenset[str], ...]

    # Per-operand subscript string, needed by the block path helper
    # to reconstruct free-to-one-operand label sets in positional order.
    operand_subscripts: tuple[str, ...]  # parallel to operand_labels

    # Declared symmetry groups per operand, preserved from construction
    # so the fingerprint fast path can use exact declared groups instead
    # of always promoting to S_k.
    per_op_groups: tuple[tuple[SymmetryGroup, ...] | None, ...]


def _build_bipartite(
    operands: list[Any],
    subscript_parts: list[str],
    per_op_groups: list[list[SymmetryGroup] | None],
    output_chars: str,
) -> EinsumBipartite:
    """Construct the bipartite graph for an einsum expression.

    Parameters
    ----------
    operands : list
        Original operand objects; Python identity is used to detect
        repeated operands.
    subscript_parts : list[str]
        Per-operand subscript strings (e.g., ["ij", "jk"]).
    per_op_groups : list
        Declared symmetry for each operand, as a list of
        SymmetryGroup objects (with ``_labels`` set), or None.
    output_chars : str
        Output subscript string.
    """
    u_vertices: list[tuple[int, int]] = []
    u_labels: list[frozenset[str]] = []
    u_operand: list[int] = []
    incidence: list[dict[str, int]] = []
    operand_labels: list[frozenset[str]] = []

    for op_idx, sub in enumerate(subscript_parts):
        operand_labels.append(frozenset(sub))
        per_op_groups[op_idx]

        # Each axis gets its own U-vertex (no merging). The σ-loop handles
        # symmetry detection via per-operand generators instead.
        class_of_position: dict[int, int] = {k: k for k in range(len(sub))}

        # Build one U vertex per axis, with incidence = label multiplicity.
        num_classes = len(sub)
        class_incidence: list[dict[str, int]] = [{} for _ in range(num_classes)]
        class_labels: list[set[str]] = [set() for _ in range(num_classes)]
        for k, c in enumerate(sub):
            cls = class_of_position[k]
            class_incidence[cls][c] = class_incidence[cls].get(c, 0) + 1
            class_labels[cls].add(c)

        for cls in range(num_classes):
            u_vertices.append((op_idx, cls))
            u_labels.append(frozenset(class_labels[cls]))
            u_operand.append(op_idx)
            incidence.append(class_incidence[cls])

    # Partition labels into free (V) vs summed (W) at the top level
    output_set = frozenset(output_chars)
    all_labels = frozenset().union(*operand_labels) if operand_labels else frozenset()
    free_labels = all_labels & output_set
    summed_labels = all_labels - output_set

    # Identical-operand groups via Python id
    id_to_positions: dict[int, list[int]] = {}
    for op_idx, op in enumerate(operands):
        id_to_positions.setdefault(id(op), []).append(op_idx)
    identical_operand_groups = tuple(
        tuple(positions)
        for positions in id_to_positions.values()
        if len(positions) >= 2
    )

    return EinsumBipartite(
        u_vertices=tuple(u_vertices),
        u_labels=tuple(u_labels),
        u_operand=tuple(u_operand),
        incidence=tuple(incidence),
        free_labels=free_labels,
        summed_labels=summed_labels,
        identical_operand_groups=identical_operand_groups,
        operand_labels=tuple(operand_labels),
        operand_subscripts=tuple(subscript_parts),
        per_op_groups=tuple(
            tuple(gs) if gs is not None else None for gs in per_op_groups
        ),
    )


@dataclass(frozen=True)
class _Subgraph:
    """Induced subgraph of an EinsumBipartite on a subset of operands.

    u_local: list of U-vertex indices (indices into graph.u_vertices) that
             belong to operands in the subset.
    v_labels: labels that are free at this step (output or crossing the cut).
    w_labels: labels that are summed entirely within the subset.
    id_groups: identical-operand groups restricted to the subset.
    """

    u_local: tuple[int, ...]
    v_labels: frozenset[str]
    w_labels: frozenset[str]
    id_groups: tuple[tuple[int, ...], ...]


def _induce_subgraph(graph: EinsumBipartite, subset: frozenset[int]) -> _Subgraph:
    u_local = tuple(
        idx for idx, op_idx in enumerate(graph.u_operand) if op_idx in subset
    )

    labels_in_subset: set[str] = set()
    for idx in u_local:
        labels_in_subset.update(graph.incidence[idx].keys())

    # Labels appearing in operands outside the subset (crossing the cut).
    outside_labels: set[str] = set()
    for op_idx, op_lbls in enumerate(graph.operand_labels):
        if op_idx not in subset:
            outside_labels.update(op_lbls)

    # V at this step = labels_in_subset ∩ (free_labels ∪ outside_labels)
    v_labels = frozenset(labels_in_subset & (graph.free_labels | outside_labels))
    w_labels = frozenset(labels_in_subset - v_labels)

    id_groups = tuple(
        tuple(sorted(set(g) & subset))
        for g in graph.identical_operand_groups
        if len(set(g) & subset) >= 2
    )

    return _Subgraph(
        u_local=u_local,
        v_labels=v_labels,
        w_labels=w_labels,
        id_groups=id_groups,
    )


class SubgraphSymmetryOracle:
    """Subset-keyed symmetry oracle for einsum intermediates.

    One oracle per contract_path call. Symmetries are computed lazily
    on first access to a subset and cached in memory. Returns a
    ``SubsetSymmetry`` with ``.output`` (V-side) and ``.inner`` (W-side).
    """

    def __init__(
        self,
        operands: list[Any],
        subscript_parts: list[str],
        per_op_groups: list[list[SymmetryGroup] | None],
        output_chars: str,
    ) -> None:
        self._graph = _build_bipartite(
            operands=operands,
            subscript_parts=subscript_parts,
            per_op_groups=per_op_groups,
            output_chars=output_chars,
        )
        self._cache: dict[frozenset[int], SubsetSymmetry] = {}

    def sym(self, subset: frozenset[int]) -> SubsetSymmetry:
        cached = self._cache.get(subset, _MISSING)
        if cached is not _MISSING:
            return cached  # type: ignore[return-value]
        result = _compute_subset_symmetry(self._graph, subset)
        self._cache[subset] = result
        return result


def _compute_subset_symmetry(
    graph: EinsumBipartite,
    subset: frozenset[int],
) -> SubsetSymmetry:
    sub = _induce_subgraph(graph, subset)
    if not sub.v_labels and not sub.w_labels:
        return SubsetSymmetry(None, None)

    # Column fingerprints for π derivation.
    row_order = sub.u_local
    all_labels = sub.v_labels | sub.w_labels
    col_of: dict[str, tuple[int, ...]] = {}
    for label in all_labels:
        col_of[label] = tuple(graph.incidence[u].get(label, 0) for u in row_order)

    fp_to_labels: dict[tuple[int, ...], set[str]] = {}
    for lbl, fp in col_of.items():
        fp_to_labels.setdefault(fp, set()).add(lbl)

    # Collect exact π generators via σ-loop.
    v_perms, w_perms = _collect_pi_permutations(
        graph, sub, row_order, col_of, fp_to_labels
    )
    v_sorted = tuple(sorted(sub.v_labels))
    w_sorted = tuple(sorted(sub.w_labels))

    # Build V-side group (only from σ-loop results, no fast path).
    v_group: SymmetryGroup | None = None
    if v_perms:
        v_group = SymmetryGroup(*v_perms, axes=tuple(range(len(v_sorted))))
        v_group._labels = v_sorted

    # Build W-side group (only from σ-loop results, no fast path).
    w_group: SymmetryGroup | None = None
    if w_perms:
        w_group = SymmetryGroup(*w_perms, axes=tuple(range(len(w_sorted))))
        w_group._labels = w_sorted

    return SubsetSymmetry(output=v_group, inner=w_group)
