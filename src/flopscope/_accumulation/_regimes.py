"""The four mixed-shape accumulation regimes plus functionalProjection.

Port of website/components/symmetry-aware-einsum-contractions/engine/regimes/*.js.

Each regime is a Regime instance with `recognize(ctx) -> Verdict` and
`compute(ctx) -> RegimeOutput`. The dispatcher in _ladder.py iterates them
in priority order; functionalProjection is checked first (Stage 2a) and the
remaining three form the mixed-shape ladder (Stage 2b).
"""

from __future__ import annotations

from ._burnside import size_aware_burnside
from ._ladder import Regime, RegimeContext, RegimeOutput, Verdict
from ._output_orbit import projection_is_functional

# ── functionalProjection ─────────────────────────────────────────────


def _functional_projection_recognize(ctx: RegimeContext) -> Verdict:
    if projection_is_functional(ctx.elements, ctx.visible_positions):
        return Verdict(
            fired=True,
            reason="each product orbit reaches exactly one stored output representative",
        )
    return Verdict(
        fired=False,
        reason="some pointwise symmetry moves an output label into a summed label",
    )


def _functional_projection_compute(ctx: RegimeContext) -> RegimeOutput:
    count = size_aware_burnside(ctx.elements, ctx.sizes)
    return RegimeOutput(
        count=count,
        sub_steps=(
            {
                "step": "projection-functional",
                "reason": "G preserves V setwise; projection descends to output reps",
                "count": count,
            },
        ),
    )


FUNCTIONAL_PROJECTION_REGIME: Regime = Regime(
    id="functionalProjection",
    recognize=_functional_projection_recognize,
    compute=_functional_projection_compute,
)


# ── singleton (B.5: |V| = 1) ─────────────────────────────────────────


def _label_orbit(elements, label_idx: int) -> list[int]:
    """G-orbit of a single label position."""
    seen = {label_idx}
    changed = True
    while changed:
        changed = False
        for g in elements:
            for p in list(seen):
                q = g.array_form[p]
                if q not in seen:
                    seen.add(q)
                    changed = True
    return sorted(seen)


def _cycles_on_subset(perm, subset: list[int]) -> int:
    """Count cycles of perm restricted to a perm-invariant subset."""
    subset_set = set(subset)
    seen: set[int] = set()
    cycles = 0
    for start in subset:
        if start in seen:
            continue
        cycles += 1
        cur = start
        while cur not in seen:
            if cur not in subset_set:
                raise ValueError("subset not invariant under perm")
            seen.add(cur)
            cur = perm.array_form[cur]
    return cycles


def _subset_cycle_product(perm, subset: list[int], sizes) -> int:
    """∏ n_c over cycles of perm in `subset`. Each cycle's labels must share a size."""
    subset_set = set(subset)
    seen: set[int] = set()
    product = 1
    for start in subset:
        if start in seen:
            continue
        cycle: list[int] = []
        cur = start
        while cur not in seen:
            if cur not in subset_set:
                raise ValueError("subset not invariant under perm")
            seen.add(cur)
            cycle.append(cur)
            cur = perm.array_form[cur]
        n0 = sizes[cycle[0]]
        for i in cycle:
            if sizes[i] != n0:
                raise ValueError("singleton: cycle in R has mixed sizes")
        product *= n0
    return product


def _singleton_recognize(ctx: RegimeContext) -> Verdict:
    if len(ctx.va) == 1:
        return Verdict(fired=True, reason="|V| = 1")
    return Verdict(fired=False, reason=f"|V| = {len(ctx.va)}, not 1")


def _singleton_compute(ctx: RegimeContext) -> RegimeOutput:
    v_pos = ctx.visible_positions[0]
    omega = _label_orbit(ctx.elements, v_pos)
    n_omega = ctx.sizes[v_pos]
    for idx in omega:
        if ctx.sizes[idx] != n_omega:
            raise ValueError(
                f"singleton: orbit of label has mixed sizes at {ctx.labels[idx]}"
            )
    omega_set = set(omega)
    rest = [i for i in range(len(ctx.labels)) if i not in omega_set]

    total = 0
    for g in ctx.elements:
        rest_factor = _subset_cycle_product(g, rest, ctx.sizes)
        c_omega = _cycles_on_subset(g, omega)
        total += rest_factor * (n_omega**c_omega - (n_omega - 1) ** c_omega)
    count = (n_omega * total) // len(ctx.elements)
    return RegimeOutput(count=count, sub_steps=())


SINGLETON_REGIME: Regime = Regime(
    id="singleton",
    recognize=_singleton_recognize,
    compute=_singleton_compute,
)


# ── young (B.6: G = Sym(L), uniform sizes, |V| ≥ 2) ──────────────────


import math


def _multiset_count(n: int, k: int) -> int:
    """Number of size-k multisets from [n]. C(n + k - 1, k)."""
    if k == 0:
        return 1
    num = 1
    den = 1
    for i in range(k):
        num *= n + k - 1 - i
        den *= i + 1
    return num // den


def _young_recognize(ctx: RegimeContext) -> Verdict:
    if not ctx.elements or len(ctx.elements) <= 1:
        return Verdict(fired=False, reason="|G| <= 1")
    if len(ctx.va) < 2:
        return Verdict(fired=False, reason="|V| < 2; singleton handles this")

    expected_full_sym = math.factorial(len(ctx.labels))
    if len(ctx.elements) != expected_full_sym:
        return Verdict(
            fired=False,
            reason=f"|G|={len(ctx.elements)} != |L|!={expected_full_sym}",
        )

    label_to_idx = {lbl: i for i, lbl in enumerate(ctx.labels)}
    v_idx_set = {label_to_idx[lbl] for lbl in ctx.va}
    has_cross = any(
        any(g.array_form[label_to_idx[lbl]] not in v_idx_set for lbl in ctx.va)
        for g in ctx.elements
    )
    if not has_cross:
        return Verdict(fired=False, reason="no cross-V/W element")

    if not ctx.sizes:
        return Verdict(fired=False, reason="no sizes provided")

    n_l = ctx.sizes[0]
    if any(s != n_l for s in ctx.sizes):
        return Verdict(fired=False, reason="mixed label sizes")

    return Verdict(fired=True, reason="G = Sym(L); Young equation applies")


def _young_compute(ctx: RegimeContext) -> RegimeOutput:
    n_l = ctx.sizes[0]
    visible_multisets = _multiset_count(n_l, len(ctx.va))
    summed_multisets = _multiset_count(n_l, len(ctx.wa))
    count = visible_multisets * summed_multisets
    return RegimeOutput(
        count=count,
        sub_steps=(
            {
                "step": "full-symmetric-output-orbit-formula",
                "n": n_l,
                "v_count": len(ctx.va),
                "w_count": len(ctx.wa),
                "visible_multisets": visible_multisets,
                "summed_multisets": summed_multisets,
                "count": count,
            },
        ),
    )


YOUNG_REGIME: Regime = Regime(
    id="young",
    recognize=_young_recognize,
    compute=_young_compute,
)


# ── partitionCount (B.7: typed equality-pattern enumeration) ─────────


from ._output_orbit import restrict_stabilizer_to_positions
from ._partition import (
    count_map_orbits_under_h,
    generate_typed_set_partitions,
    induced_block_action_size,
    induced_prefix_maps,
    num_blocks,
    partition_key,
    partition_orbit_reps,
    typed_labeling_count,
)


def _partition_count_recognize(ctx: RegimeContext) -> Verdict:
    partitions = generate_typed_set_partitions(ctx.sizes)
    if len(partitions) > ctx.partition_budget:
        return Verdict(
            fired=False,
            reason=(
                f"typed partition count {len(partitions)} exceeds "
                f"budget {ctx.partition_budget}"
            ),
        )
    return Verdict(
        fired=True,
        reason=f"typed partition count over {len(partitions)} equality patterns",
    )


def _partition_count_compute(ctx: RegimeContext) -> RegimeOutput:
    h_elements = restrict_stabilizer_to_positions(ctx.elements, ctx.visible_positions)
    partitions = generate_typed_set_partitions(ctx.sizes)
    reps = partition_orbit_reps(partitions, ctx.elements)

    total = 0
    sub_steps: list[dict] = []
    for partition in reps:
        labelings = typed_labeling_count(partition, ctx.sizes)
        block_action = induced_block_action_size(partition, ctx.elements)
        if labelings % block_action != 0:
            raise ValueError(
                f"partition {partition_key(partition)} has labelings={labelings} "
                f"not divisible by block action size={block_action} — "
                f"invariant violation in G"
            )
        input_orbits = labelings // block_action
        maps = induced_prefix_maps(partition, ctx.elements, ctx.visible_positions)
        output_orbits = count_map_orbits_under_h(maps, h_elements)
        term = input_orbits * output_orbits
        total += term
        sub_steps.append(
            {
                "partition_key": partition_key(partition),
                "blocks": num_blocks(partition),
                "typed_labelings": labelings,
                "block_action_size": block_action,
                "input_orbit_count": input_orbits,
                "induced_map_count": len(maps),
                "output_orbit_count": output_orbits,
                "contribution": term,
            }
        )

    return RegimeOutput(count=total, sub_steps=tuple(sub_steps))


PARTITION_COUNT_REGIME: Regime = Regime(
    id="partitionCount",
    recognize=_partition_count_recognize,
    compute=_partition_count_compute,
)


MIXED_REGIMES: tuple[Regime, ...] = (
    SINGLETON_REGIME,
    YOUNG_REGIME,
    PARTITION_COUNT_REGIME,
)
