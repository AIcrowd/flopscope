"""σ-loop, π-canonical derivation, and whole-expression G_pt construction.

Port of website/components/symmetry-aware-einsum-contractions/engine/algorithm.js
(runSigmaLoop only) and fullGroup.js (in Task 17).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from flopscope._perm_group import _Permutation as Permutation

from ._bipartite import BipartiteGraph, IncidenceMatrix
from ._wreath import WreathElement


@dataclass(frozen=True)
class SigmaResult:
    """One row of the σ-loop. Either σ is identity (skipped, π=identity), or σ is
    non-identity and we either accepted a π or rejected (no fingerprint match)."""

    is_valid: bool
    is_identity: bool
    skipped: bool
    pi: dict[str, str] | None
    pi_kind: Literal["identity", "v-only", "w-only", "cross-v-w"] | None
    reason: str | None = None
    sigma_row_perm: tuple[int, ...] | None = None


def derive_pi_canonical(
    sigma_col_of: dict[str, tuple[int, ...]],
    fp_to_labels: dict[tuple[int, ...], frozenset[str]],
    v_labels: frozenset[str],
    w_labels: frozenset[str],
) -> dict[str, str] | None:
    """For each label, look up σ(M)'s column fingerprint in fp_to_labels and pick the
    lex-first unused candidate. Validates bijectivity. Returns None when no
    consistent π exists.

    Note: π may legitimately mix V and W labels — cross-V/W actions are part of
    the detected symmetry. The deprecated 'partition-preserving rejection' is NOT
    reintroduced here. Mirrors algorithm.js#derivePi.
    """
    pi: dict[str, str] = {}
    used: set[str] = set()
    for label in sorted(v_labels | w_labels):
        fp = sigma_col_of[label]
        candidates = fp_to_labels.get(fp)
        if not candidates:
            return None
        pick = next((c for c in sorted(candidates) if c not in used), None)
        if pick is None:
            return None
        pi[label] = pick
        used.add(pick)
    return pi


def classify_pi(
    pi: dict[str, str],
    v_labels: frozenset[str],
    w_labels: frozenset[str],
) -> dict[str, object]:
    """Classify a π's action: identity, v-only (preserves W pointwise),
    w-only (preserves V pointwise), or cross-v-w."""
    pi_is_identity = all(pi[lbl] == lbl for lbl in pi)
    moves_v = any(pi[lbl] != lbl for lbl in v_labels if lbl in pi)
    moves_w = any(pi[lbl] != lbl for lbl in w_labels if lbl in pi)
    crosses_vw = any(
        (lbl in v_labels and pi[lbl] in w_labels)
        or (lbl in w_labels and pi[lbl] in v_labels)
        for lbl in pi
    )

    if pi_is_identity:
        kind = "identity"
    elif crosses_vw:
        kind = "cross-v-w"
    elif moves_v and not moves_w:
        kind = "v-only"
    elif moves_w and not moves_v:
        kind = "w-only"
    else:
        kind = "cross-v-w"  # both V and W move, but no V↔W swap — still classified as cross

    return {
        "piIsIdentity": pi_is_identity,
        "piKind": kind,
        "crosses": crosses_vw,
        "movesV": moves_v,
        "movesW": moves_w,
    }


def run_sigma_loop(
    graph: BipartiteGraph,
    matrix_data: IncidenceMatrix,
    wreath_elements: Sequence[WreathElement],
) -> tuple[SigmaResult, ...]:
    """Run the σ-loop: for each wreath element, derive π and classify it.
    Mirrors algorithm.js#runSigmaLoop."""
    results: list[SigmaResult] = []
    v_labels = graph.free_labels
    w_labels = graph.summed_labels
    all_labels = graph.all_labels

    for element in wreath_elements:
        sigma_row_perm = tuple(element.row_perm.array_form)
        is_identity = all(v == i for i, v in enumerate(sigma_row_perm))

        if is_identity:
            identity_pi = {lbl: lbl for lbl in all_labels}
            results.append(
                SigmaResult(
                    is_valid=True,
                    is_identity=True,
                    skipped=True,
                    pi=identity_pi,
                    pi_kind="identity",
                    sigma_row_perm=sigma_row_perm,
                )
            )
            continue

        # Compute σ(M) column fingerprints.
        sigma_col_of: dict[str, tuple[int, ...]] = {}
        for label in all_labels:
            sigma_col_of[label] = tuple(
                graph.incidence[sigma_row_perm[k]].get(label, 0)
                for k in range(len(sigma_row_perm))
            )

        pi = derive_pi_canonical(
            sigma_col_of, matrix_data.fp_to_labels, v_labels, w_labels
        )
        if pi is None:
            results.append(
                SigmaResult(
                    is_valid=False,
                    is_identity=False,
                    skipped=False,
                    pi=None,
                    pi_kind=None,
                    reason="No matching π (fingerprint mismatch)",
                    sigma_row_perm=sigma_row_perm,
                )
            )
            continue

        classification = classify_pi(pi, v_labels, w_labels)
        results.append(
            SigmaResult(
                is_valid=True,
                is_identity=False,
                skipped=False,
                pi=pi,
                pi_kind=classification["piKind"],  # type: ignore[arg-type]
                sigma_row_perm=sigma_row_perm,
            )
        )

    return tuple(results)


# ── build_full_group ─────────────────────────────────────────────────


import math

from flopscope._perm_group import _dimino


@dataclass(frozen=True)
class DetectedGroup:
    """The whole-expression G_pt."""

    all_labels: tuple[str, ...]
    generators: tuple[Permutation, ...]
    elements: tuple[Permutation, ...]
    group_name: str
    action_summary: str
    valid_pi_results: tuple[SigmaResult, ...]


def _minimal_generators(
    candidates: Sequence[Permutation],
) -> list[Permutation]:
    """Greedy minimal generating set: keep only generators that grow the closure."""
    if len(candidates) <= 1:
        return list(candidates)
    selected: list[Permutation] = []
    current_size = 1
    for candidate in candidates:
        trial = (*selected, candidate)
        elements = _dimino(trial)
        if len(elements) > current_size:
            selected.append(candidate)
            current_size = len(elements)
    return selected


def _classify_group_name(
    labels: Sequence[str],
    generators: Sequence[Permutation],
    elements: Sequence[Permutation],
) -> str:
    """Mirror of fullGroup.js + algorithm.js#classifyGroupName.

    Recognizes S_n, C_n, D_n, Z_2, S_2, and falls back to PermGroup⟨...⟩."""
    order = len(elements)
    degree = len(labels)
    if degree < 2 or order <= 1:
        return "trivial"

    moved_set: set[int] = set()
    for el in elements:
        for i, v in enumerate(el.array_form):
            if v != i:
                moved_set.add(i)
    moved_indices = sorted(moved_set)
    moved_labels = [labels[i] for i in moved_indices] if moved_indices else list(labels)
    effective_degree = len(moved_labels) or degree
    label_set = "{" + ",".join(moved_labels) + "}"

    if order == math.factorial(effective_degree):
        return f"S{effective_degree}{label_set}"
    if order == effective_degree and effective_degree >= 3:
        # Cyclic check
        for el in elements:
            cycles = el.cyclic_form
            if len(cycles) == 1 and len(cycles[0]) == effective_degree:
                return f"C{effective_degree}{label_set}"
    if order == 2 * effective_degree and effective_degree >= 3:
        return (
            f"D{effective_degree}{label_set}"  # heuristic — matches JS classification
        )
    if order == 2 and degree == 2:
        return f"S2{label_set}"
    if order == 2 and effective_degree > 2:
        return f"Z2{label_set}"

    # Fallback: build cycle notation inline using cyclic_form + label substitution
    def _gen_cycle_str(gen: Permutation, lbl_list: Sequence[str]) -> str:
        cycles = gen.cyclic_form
        if not cycles:
            return "()"
        return "".join(
            "(" + " ".join(lbl_list[idx] for idx in cycle) + ")" for cycle in cycles
        )

    gen_str = ", ".join(_gen_cycle_str(g, labels) for g in generators)
    return f"PermGroup⟨{gen_str}⟩"


def _action_summary(
    pi_kinds_seen: set[str],
) -> str:
    """Summarize the action of G as 'V-only' / 'W-only' / 'cross-V/W' / 'trivial'."""
    if not pi_kinds_seen or pi_kinds_seen <= {"identity"}:
        return "trivial"
    if "cross-v-w" in pi_kinds_seen:
        return "cross-V/W"
    if "v-only" in pi_kinds_seen and "w-only" in pi_kinds_seen:
        return "V-only × W-only"
    if "v-only" in pi_kinds_seen:
        return "V-only"
    if "w-only" in pi_kinds_seen:
        return "W-only"
    return "trivial"


def build_full_group(
    sigma_results: Sequence[SigmaResult],
    *,
    all_labels: Sequence[str],
) -> DetectedGroup:
    """Collect valid πs as full-degree permutations on `all_labels`, dedupe, find
    a minimal generating set via greedy growth, run dimino. Mirrors fullGroup.js#buildFullGroup."""
    label_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    candidates: list[Permutation] = []
    seen_keys: set[tuple[int, ...]] = set()
    pi_kinds_seen: set[str] = set()
    valid: list[SigmaResult] = []

    for r in sigma_results:
        if not r.is_valid:
            continue
        if r.pi is not None:
            valid.append(r)
        if r.skipped or r.pi_kind == "identity":
            if r.pi_kind is not None:
                pi_kinds_seen.add(r.pi_kind)
            continue
        if r.pi is None:
            continue
        if r.pi_kind is not None:
            pi_kinds_seen.add(r.pi_kind)
        arr = tuple(label_idx[r.pi[lbl]] for lbl in all_labels)
        if arr in seen_keys:
            continue
        seen_keys.add(arr)
        candidates.append(Permutation(list(arr)))

    minimal = _minimal_generators(candidates)
    if minimal:
        elements = _dimino(tuple(minimal))
    else:
        elements = [Permutation.identity(len(all_labels))] if all_labels else []

    group_name = _classify_group_name(all_labels, minimal, elements)

    return DetectedGroup(
        all_labels=tuple(all_labels),
        generators=tuple(minimal),
        elements=tuple(elements),
        group_name=group_name,
        action_summary=_action_summary(pi_kinds_seen),
        valid_pi_results=tuple(valid),
    )
