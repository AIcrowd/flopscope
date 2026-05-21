"""AccumulationCost orchestrator + per-component cost wrapping.

Aggregates the ladder primitive (compute_accumulation) into
einsum-shaped cost reports. Future reduction code reuses run_ladder_per_component
and adds its own aggregator (aggregate_reduction) that uses different cost arithmetic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ._burnside import size_aware_burnside
from ._components import Component
from ._ladder import (
    AccumulationResult,
    RegimeId,
    RegimeStep,
    Shape,
    compute_accumulation,
)
from ._output_orbit import restrict_stabilizer_to_positions


@dataclass(frozen=True)
class ComponentCost:
    """Per-component cost. alpha is None when this component's regime returned
    unavailable (partition budget exceeded with brute-force disabled by policy)."""

    labels: tuple[str, ...]
    va: tuple[str, ...]
    wa: tuple[str, ...]
    sizes: tuple[int, ...]

    m: int
    alpha: int | None
    dense_count: int
    num_output_orbits: int

    regime_id: RegimeId
    shape: Shape

    group_name: str
    group_order: int

    regime_trace: tuple[RegimeStep, ...]
    unavailable_reason: str | None = None

    def describe(self) -> dict[str, str]:
        """LaTeX strings built on demand (Task 23 fills in the body)."""
        from ._cost_descriptions import describe_component

        return describe_component(self)


def run_ladder_per_component(
    components: Sequence[Component],
    *,
    partition_budget: int,
) -> tuple[ComponentCost, ...]:
    """For each Component, run the ladder + Burnside, return ComponentCosts.

    Pure transformation, no aggregation policy. Reused by both einsum and (future)
    reduction code paths.
    """
    out: list[ComponentCost] = []
    for c in components:
        result: AccumulationResult = compute_accumulation(
            labels=c.labels,
            va=c.va,
            wa=c.wa,
            elements=c.elements,
            generators=c.generators,
            sizes=c.sizes,
            visible_positions=c.visible_positions,
            partition_budget=partition_budget,
        )
        # M is always computable via Burnside, even when α is unavailable.
        if c.elements and len(c.elements) > 0:
            m = size_aware_burnside(c.elements, c.sizes)
        else:
            m = math.prod(c.sizes) if c.sizes else 1
        dense_count = math.prod(c.sizes) if c.sizes else 1

        # num_output_orbits: orbits of the visible-label tuples under the
        # output stabilizer (the subgroup of G that preserves V setwise,
        # restricted to V). Used by aggregate_einsum to apply the same
        # "first cell of each orbit is a free copy" off-by-one correction
        # that reduction_accumulation_cost already applies.
        v_sizes = tuple(c.sizes[p] for p in c.visible_positions)
        if not v_sizes:
            num_output_orbits = 1
        elif c.elements and len(c.elements) > 0:
            h_elements = restrict_stabilizer_to_positions(
                c.elements, c.visible_positions
            )
            num_output_orbits = size_aware_burnside(h_elements, v_sizes)
        else:
            num_output_orbits = math.prod(v_sizes)

        unavailable_reason: str | None = None
        if result.regime_id == "unavailable":
            # The "unavailable" trace step's reason is the ladder's last word.
            unavailable_step = next(
                (s for s in result.trace if s.regime_id == "unavailable"),
                None,
            )
            if unavailable_step is not None:
                unavailable_reason = unavailable_step.reason

        out.append(
            ComponentCost(
                labels=c.labels,
                va=c.va,
                wa=c.wa,
                sizes=c.sizes,
                m=m,
                alpha=result.count,
                dense_count=dense_count,
                num_output_orbits=num_output_orbits,
                regime_id=result.regime_id,
                shape=result.shape,
                group_name=c.group_name,
                group_order=c.order,
                regime_trace=result.trace,
                unavailable_reason=unavailable_reason,
            )
        )
    return tuple(out)


# ── AccumulationCost + aggregate_einsum ──────────────────────────────


import warnings

from flopscope.errors import CostFallbackWarning


@dataclass(frozen=True)
class AccumulationCost:
    """Whole-einsum cost. When any component is unavailable, total falls back to
    the dense baseline (k · dense_baseline) and a CostFallbackWarning fires."""

    total: int
    mu: int | None
    alpha: int | None
    m_total: int
    dense_baseline: int
    num_terms: int

    per_component: tuple[ComponentCost, ...]

    fallback_used: bool
    unavailable_components: tuple[int, ...] = ()
    unavailable_reason: str | None = None

    # NEW: path-aware decomposition. Empty for k<=2 (no path walked).
    per_step: tuple[AccumulationCost, ...] = ()
    path: tuple[tuple[int, ...], ...] | None = None

    def describe(self) -> dict:
        """Human-readable + LaTeX summary, built on demand."""
        from ._cost_descriptions import describe_total

        return describe_total(self)

    @property
    def savings_ratio(self) -> float:
        """total / (k · dense_baseline). 1.0 means no savings; lower is better."""
        denom = self.num_terms * self.dense_baseline
        return self.total / denom if denom > 0 else 1.0


def aggregate_einsum(
    component_costs: Sequence[ComponentCost],
    *,
    num_terms: int,
    dense_baseline: int,
) -> AccumulationCost:
    """Aggregate per-component costs into the einsum cost:
    ``total = (k-1)·∏M + ∏α − ∏num_output_orbits``.

    The final ``−∏num_output_orbits`` term applies the same off-by-one
    correction that ``reduction_accumulation_cost`` uses: the first cell of
    each output orbit is a free copy, and only the remaining accumulations
    cost. When there is no actual reduction (``∏α == ∏num_output_orbits``),
    the α contribution collapses to zero, leaving only the ``(k-1)·∏M``
    multiplication chain.

    Fallback policy: if any component has alpha=None, total = k · dense_baseline
    (the no-symmetry direct-event count) and a CostFallbackWarning fires.
    """
    failing = [i for i, c in enumerate(component_costs) if c.alpha is None]

    m_total = 1
    for c in component_costs:
        m_total *= c.m

    if not failing:
        alpha_product = 1
        output_orbit_product = 1
        for c in component_costs:
            assert c.alpha is not None  # for type narrowing
            alpha_product *= c.alpha
            output_orbit_product *= c.num_output_orbits
        mu = (num_terms - 1) * m_total
        total = mu + alpha_product - output_orbit_product
        return AccumulationCost(
            total=total,
            mu=mu,
            alpha=alpha_product,
            m_total=m_total,
            dense_baseline=dense_baseline,
            num_terms=num_terms,
            per_component=tuple(component_costs),
            fallback_used=False,
        )

    # Fallback: charge dense.
    fallback_total = num_terms * dense_baseline
    first_failing = component_costs[failing[0]]
    reason = first_failing.unavailable_reason or "partition_budget exceeded"
    failing_labels = ", ".join(first_failing.labels)
    warnings.warn(
        CostFallbackWarning(
            f"einsum: component {list(failing)} ({failing_labels}) returned "
            f"unavailable — charging dense cost {fallback_total} = "
            f"{num_terms} × {dense_baseline}. Failing reason: {reason}. "
            f"Raise via flopscope.configure(partition_budget=...) to attempt "
            f"exact counting."
        ),
        stacklevel=4,
    )
    return AccumulationCost(
        total=fallback_total,
        mu=None,
        alpha=None,
        m_total=m_total,
        dense_baseline=dense_baseline,
        num_terms=num_terms,
        per_component=tuple(component_costs),
        fallback_used=True,
        unavailable_components=tuple(failing),
        unavailable_reason=reason,
    )


# ── End-to-end orchestrator ──────────────────────────────────────────


from collections.abc import Sequence as _Seq
from typing import Any as _Any
from typing import cast as _cast

from ._bipartite import build_bipartite, build_incidence_matrix
from ._components import decompose_into_components
from ._detection import build_full_group, run_sigma_loop
from ._wreath import enumerate_wreath


def _build_size_map(
    input_parts: _Seq[str],
    shapes: _Seq[_Seq[int]],
) -> dict[str, int]:
    """Build label → size from operand shapes. Validates that each label
    appears with consistent sizes across operands."""
    size_map: dict[str, int] = {}
    for part, shape in zip(input_parts, shapes, strict=True):
        for label, dim in zip(part, shape, strict=True):
            existing = size_map.get(label)
            if existing is not None and existing != dim:
                raise ValueError(
                    f"label '{label}' has inconsistent sizes {existing} and {dim} "
                    f"across operands"
                )
            size_map[label] = dim
    return size_map


def _operand_names_from_identity_pattern(
    num_ops: int,
    identity_pattern: tuple[tuple[int, ...], ...] | None,
) -> tuple[str, ...]:
    """Generate operand names that respect the identity pattern.
    Operands sharing the same id (per identity_pattern) get the same name."""
    if identity_pattern is None:
        return tuple(f"op_{i}" for i in range(num_ops))
    name_of: dict[int, str] = {}
    for group in identity_pattern:
        shared_name = f"op_grp_{group[0]}"
        for pos in group:
            name_of[pos] = shared_name
    return tuple(name_of.get(i, f"op_{i}") for i in range(num_ops))


def _per_op_symmetry_for_wreath(sym: _Any) -> _Any:
    """Pass declared symmetry through to enumerate_h. SymmetryGroup objects
    pass through unchanged; strings (like 'symmetric') also pass through."""
    return sym


def _walk_path_and_aggregate(
    *,
    canonical_subscripts: str,
    input_parts: _Seq[str],
    output_subscript: str,
    shapes: _Seq[_Seq[int]],
    per_op_symmetries: _Seq[_Any],
    identity_pattern: tuple[tuple[int, ...], ...] | None,
    partition_budget: int,
    dense_baseline: int,
    full_expression_component_costs: tuple[ComponentCost, ...] | None = None,
) -> AccumulationCost:
    """Walk opt_einsum's binary contraction path and sum per-step costs.

    Decomposes a k>=3 einsum into binary contractions via opt_einsum.contract_path.
    Each binary step calls compute_accumulation_cost recursively (k=2 path),
    treating intermediate tensors as dense (no symmetry carried forward).

    ``full_expression_component_costs``: the ComponentCost tuple from the caller's
    wreath/sigma computation of the FULL k-ary expression. Stored verbatim in
    per_component so that JS-parity tests (which inspect per_component directly)
    remain unaffected by the path decomposition.

    Returns an AccumulationCost with per_step and path populated.
    """
    import opt_einsum as _oe

    from flopscope._opt_einsum._contract import build_path_info

    num_ops = len(input_parts)

    # Build shape-only operands for opt_einsum (it only needs shape info).
    # We use shapes=True to avoid materializing tensors.
    import numpy as _np

    dummy_operands = [_np.empty(shape) for shape in shapes]
    upstream_path, upstream_info = _oe.contract_path(
        canonical_subscripts,
        *dummy_operands,
        optimize="auto",
    )

    path_info = build_path_info(
        upstream_path,
        upstream_info,
        size_dict=upstream_info.size_dict,
    )

    # Build SubgraphSymmetryOracle for the full expression so per-step
    # binary contractions inherit their symmetry from the declared groups
    # rather than treating every intermediate as dense.
    from flopscope._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle

    def _group_to_oracle_list(sym: _Any, subscript: str) -> list | None:
        """Convert a per_op_symmetry entry to the oracle's per_op_groups format.

        The oracle's Source A generator loop requires ``group._labels`` to be
        set (the subscript chars the group acts on, in axis order).  When the
        declared SymmetryGroup has ``axes`` but ``_labels is None``, we derive
        _labels from the operand subscript and the group's axes mapping.
        """
        from flopscope._perm_group import SymmetryGroup as _SG

        if sym is None:
            return None
        if not isinstance(sym, _SG):
            # String symmetries (e.g. 'symmetric') are not SymmetryGroup objects;
            # skip them here — the oracle only consumes SymmetryGroup generators.
            return None
        # Ensure _labels is set so the oracle's Source A generator loop can
        # map group-axis positions to subscript characters.
        if sym._labels is None and sym.axes is not None:
            # axes[g_pos] = tensor axis index; subscript[axis_idx] = char.
            try:
                derived_labels = tuple(subscript[ax] for ax in sym.axes)
            except IndexError:
                derived_labels = None
            if derived_labels is not None:
                # Build a new group with _labels set (avoid mutating the original).
                import copy as _copy

                labeled_group = _copy.copy(sym)
                labeled_group._labels = derived_labels
                return [labeled_group]
        return [sym]

    # Build oracle operands that mirror the original identity pattern:
    # identical operands (same Python id in the original call) must share
    # the same Python object here so the oracle's Source B generator (identical-
    # operand swaps) fires correctly for subset queries.
    oracle_operands = list(dummy_operands)  # start with distinct fresh objects
    if identity_pattern:
        for group in identity_pattern:
            # All positions in `group` referred to the same object originally.
            # Reuse the dummy for the first position for all positions in the group.
            canonical_obj = oracle_operands[group[0]]
            for pos in group[1:]:
                oracle_operands[pos] = canonical_obj

    oracle = SubgraphSymmetryOracle(
        operands=oracle_operands,
        subscript_parts=list(input_parts),
        per_op_groups=[
            _group_to_oracle_list(s, sub)
            for s, sub in zip(per_op_symmetries, input_parts, strict=False)
        ],
        output_chars=output_subscript,
    )

    def _subset_sym_fingerprint(subset: frozenset[int], step_subscript: str) -> tuple:
        """Return an accumulation-cache fingerprint for a step-input subset.

        Queries the oracle for the V-side (output) group of the subset of
        original operands, then serialises it in the (axes, gens) format
        expected by get_accumulation_cost_cached's sym_fingerprint.
        Returns None (dense) when the oracle finds no symmetry.

        Oracle groups encode generators in label-sorted space (_labels gives the
        sorted char tuple and each generator acts on positions of that tuple).
        The cache reconstructs a group with _labels derived from the step
        subscript in axis order, so the generator would be misinterpreted unless
        we convert it to tensor-axis space first.
        """
        from flopscope._perm_group import SymmetryGroup as _SG
        from flopscope._perm_group import _PermutationCompat as _Perm

        ss = oracle.sym(subset)
        grp = ss.output  # V-side (free-label) group for this subset
        if grp is None or not isinstance(grp, _SG):
            return None  # type: ignore[return-value]
        if not grp.generators:
            return None  # type: ignore[return-value]

        # Convert oracle's label-sorted generators to tensor-axis-space so that
        # when the cache reconstructs the group (with _labels derived from the
        # step subscript in axis order), the generator is correctly interpreted.
        if grp._labels is not None:
            labels = grp._labels  # sorted char tuple
            char_to_axis = {c: i for i, c in enumerate(step_subscript)}
            label_to_lpos = {lbl: k for k, lbl in enumerate(labels)}
            new_gens = []
            for gen in grp.generators:
                arr = list(gen.array_form)
                axis_gen = list(range(len(step_subscript)))
                for ax, char in enumerate(step_subscript):
                    if char in label_to_lpos:
                        lpos = label_to_lpos[char]
                        target_char = labels[arr[lpos]]
                        if target_char in char_to_axis:
                            axis_gen[ax] = char_to_axis[target_char]
                new_gens.append(_Perm(axis_gen))
            axes = tuple(range(len(step_subscript)))
            gens = tuple(tuple(g.array_form) for g in new_gens)
        else:
            axes = grp.axes if grp.axes is not None else tuple(range(grp.degree))
            gens = tuple(tuple(g.array_form) for g in grp.generators)

        if not gens:
            return None  # type: ignore[return-value]
        return (axes, gens)

    # SSA-to-subset: tracks which original operand positions each current
    # operand covers. Starts as singletons; merged as the path progresses.
    # We walk path_info.path in parallel with path_info.steps.
    current_subsets: list[frozenset[int]] = [frozenset({i}) for i in range(num_ops)]

    from ._cache import get_accumulation_cost_cached

    per_step_costs: list[AccumulationCost] = []
    for step_idx, step in enumerate(path_info.steps):
        step_subscript = step.subscript
        if "->" in step_subscript:
            step_lhs, step_output = step_subscript.split("->", 1)
        else:
            step_lhs, step_output = step_subscript, ""
        step_input_parts = step_lhs.split(",")
        step_shapes = step.input_shapes

        # Derive which original operands each step input covers.
        # path_info.path[step_idx] gives the current-list positions to contract.
        raw_path_entry = path_info.path[step_idx]
        # Sort descending so that popping by index doesn't shift earlier indices.
        contract_positions = tuple(sorted(raw_path_entry, reverse=True))

        # Gather the input subsets matching the opt_einsum operand order in the
        # step subscript. opt_einsum pops positions from highest to lowest (to
        # avoid index shifting), so the subscript's left-to-right operand order
        # corresponds to descending position order in the path entry.
        # E.g. path entry (0,1) with subscript "jk,ij->..." means position 1
        # (higher) contributes "jk" first and position 0 contributes "ij" second.
        step_input_subsets = [
            current_subsets[pos]
            for pos in sorted(
                raw_path_entry, reverse=True
            )  # descending = subscript order
        ]

        # Build per-step sym_fingerprint by querying the oracle per input subset.
        # Pass the step subscript so generators can be converted from oracle
        # label-sorted space to tensor-axis space matching the step subscript.
        step_sym_fp: tuple = tuple(
            _subset_sym_fingerprint(subset, sub_part)
            for subset, sub_part in zip(
                step_input_subsets, step_input_parts, strict=False
            )
        )

        # Derive the step's identity_pattern from the original expression's
        # identity_pattern. A step identity exists only when BOTH inputs are:
        # (a) singleton subsets (original operands, not intermediates), AND
        # (b) in the same identity group in the original expression.
        # The wreath/sigma framework correctly handles identical operands even
        # when they appear with different subscripts in the step — the
        # swap generator combined with per-operand declared symmetry produces
        # the correct joint group.
        step_identity_pattern: tuple[tuple[int, ...], ...] | None = None
        if identity_pattern:
            # Map each singleton original position to its local step index.
            orig_to_local: dict[int, int] = {}
            for local_idx_inner, subset in enumerate(step_input_subsets):
                if len(subset) == 1:
                    orig_to_local[next(iter(subset))] = local_idx_inner
            # For each original identity group, restrict to this step's singletons.
            local_groups = []
            for orig_group in identity_pattern:
                step_members = tuple(
                    orig_to_local[op] for op in orig_group if op in orig_to_local
                )
                if len(step_members) >= 2:
                    local_groups.append(step_members)
            if local_groups:
                step_identity_pattern = tuple(local_groups)

        step_canonical = ",".join(step_input_parts) + "->" + step_output

        # Route per-step binary calls through the shared LRU cache so that
        # identical sub-steps across different top-level expressions hit once.
        step_cost = get_accumulation_cost_cached(
            canonical_subscripts=step_canonical,
            input_parts=tuple(step_input_parts),
            output_subscript=step_output,
            shapes=tuple(tuple(s) for s in step_shapes),
            sym_fingerprint=step_sym_fp,
            identity_pattern=step_identity_pattern,
            partition_budget=partition_budget,
        )

        # Sprint 1 Cat B: if the per-input fingerprint detected no savings
        # (step_cost == dense FMA baseline), query the oracle for the merged
        # subset's OUTPUT group.  This captures symmetry that arises from the
        # global context (e.g. identical W matrices) but cannot be detected from
        # the individual input subsets alone (e.g. when the input's b-c swap
        # crosses the V/W boundary in this binary step).
        #
        # Discriminator: we apply the correction ONLY when step_cost equals the
        # dense FMA cost — meaning per-input analysis found no savings.  This
        # prevents double-counting in steps that already benefited from per-input
        # symmetry reduction.
        if oracle is not None:
            # Compute dense FMA for this binary step:
            #   dense_fma = 2 * dense_baseline - output_dense
            step_size_dict: dict[str, int] = {}
            for sub, shape in zip(step_input_parts, step_shapes, strict=False):
                for char, sz in zip(sub, shape, strict=False):
                    step_size_dict[char] = sz
            step_all_chars = set("".join(step_input_parts))
            step_dense_baseline_val = math.prod(
                step_size_dict.get(c, 1) for c in step_all_chars
            )
            step_output_dense = math.prod(step_size_dict.get(c, 1) for c in step_output)
            step_dense_fma = 2 * step_dense_baseline_val - step_output_dense

            if step_cost.total == step_dense_fma and step_dense_fma > 0:
                # No savings detected; check global oracle's merged output group.
                merged_subset_local = frozenset().union(*step_input_subsets)
                try:
                    ss_merged = oracle.sym(merged_subset_local)
                    merged_output_grp = ss_merged.output if ss_merged else None
                except Exception:
                    merged_output_grp = None

                if merged_output_grp is not None and merged_output_grp.generators:
                    from flopscope._perm_group import _dimino
                    from flopscope._perm_group import _PermutationCompat as _Perm

                    # Convert oracle's label-sorted generators to tensor-axis
                    # space for the step output subscript.
                    labels = merged_output_grp._labels
                    if labels is not None:
                        char_to_axis = {c: i for i, c in enumerate(step_output)}
                        label_to_lpos = {lbl: k for k, lbl in enumerate(labels)}
                        axis_gens: list[_Perm] = []
                        for gen in merged_output_grp.generators:
                            arr = list(gen.array_form)
                            axis_gen = list(range(len(step_output)))
                            for ax, char in enumerate(step_output):
                                if char in label_to_lpos:
                                    lpos = label_to_lpos[char]
                                    tgt = labels[arr[lpos]]
                                    if tgt in char_to_axis:
                                        axis_gen[ax] = char_to_axis[tgt]
                            axis_gens.append(_Perm(axis_gen))
                    else:
                        axis_gens = list(merged_output_grp.generators)

                    try:
                        all_elems = _dimino(tuple(axis_gens))
                        output_sizes = tuple(
                            step_size_dict.get(c, 1) for c in step_output
                        )
                        corrected_orbits = size_aware_burnside(all_elems, output_sizes)
                        if corrected_orbits < step_output_dense:
                            # Orbit reduction found; compute corrected cost.
                            step_w_chars = step_all_chars - set(step_output)
                            step_w_size = (
                                math.prod(
                                    step_size_dict.get(c, 1) for c in step_w_chars
                                )
                                if step_w_chars
                                else 1
                            )
                            corrected_total = corrected_orbits * (2 * step_w_size - 1)
                            if corrected_total < step_cost.total:
                                # Replace step_cost total while preserving other fields.
                                step_cost = AccumulationCost(
                                    total=corrected_total,
                                    mu=step_w_size * corrected_orbits
                                    - corrected_orbits,
                                    alpha=step_w_size * corrected_orbits,
                                    m_total=step_w_size * corrected_orbits,
                                    dense_baseline=step_dense_baseline_val,
                                    num_terms=step_cost.num_terms,
                                    per_component=step_cost.per_component,
                                    fallback_used=step_cost.fallback_used,
                                    unavailable_components=step_cost.unavailable_components,
                                    unavailable_reason=step_cost.unavailable_reason,
                                    per_step=step_cost.per_step,
                                    path=step_cost.path,
                                )
                    except Exception:
                        pass  # Burnside failed (e.g. dimino budget); keep original

        per_step_costs.append(step_cost)

        # Update current_subsets: remove contracted inputs (highest index first
        # to preserve lower indices), then append the merged output subset.
        merged_subset: frozenset[int] = frozenset().union(*step_input_subsets)
        for pos in contract_positions:
            current_subsets.pop(pos)
        current_subsets.append(merged_subset)

    total = sum(s.total for s in per_step_costs)
    mu_total = sum((s.mu or 0) for s in per_step_costs)
    alpha_total = sum((s.alpha or 0) for s in per_step_costs)
    # m_total for the path-level result must reflect the number of unique output
    # elements in the FULL k-ary expression, not the product of per-step intermediate
    # m_total values (which would multiply intermediate tensor sizes together and
    # always exceed dense_baseline, making _has_savings() return False).
    # full_expression_component_costs holds the full-expression per_component data
    # computed by the wreath/sigma pass before the path decomposition.
    if full_expression_component_costs:
        m_total = math.prod(c.m for c in full_expression_component_costs)
    else:
        # Fallback: sum of per-step m_total values is still wrong, but if we have
        # no component data just use dense_baseline as a conservative estimate.
        m_total = dense_baseline
    fallback_used = any(s.fallback_used for s in per_step_costs)
    unavailable_components: tuple[int, ...] = tuple(
        i for i, s in enumerate(per_step_costs) if s.fallback_used
    )
    unavailable_reason = next(
        (s.unavailable_reason for s in per_step_costs if s.unavailable_reason),
        None,
    )

    return AccumulationCost(
        total=total,
        mu=mu_total if mu_total > 0 else None,
        alpha=alpha_total if alpha_total > 0 else None,
        m_total=m_total,
        dense_baseline=dense_baseline,
        num_terms=num_ops,
        per_component=full_expression_component_costs or (),
        fallback_used=fallback_used,
        unavailable_components=unavailable_components,
        unavailable_reason=unavailable_reason,
        per_step=tuple(per_step_costs),
        path=tuple(tuple(p) for p in path_info.path),
    )


def compute_accumulation_cost(
    *,
    canonical_subscripts: str,
    input_parts: _Seq[str],
    output_subscript: str,
    shapes: _Seq[_Seq[int]],
    per_op_symmetries: _Seq[_Any] | None,
    identity_pattern: tuple[tuple[int, ...], ...] | None,
    partition_budget: int | None = None,
) -> AccumulationCost:
    """End-to-end whole-expression cost.

    Inputs:
      canonical_subscripts: the einsum string after canonicalization
        (e.g. 'ij,jk->ik').
      input_parts: per-operand subscript strings.
      output_subscript: output labels (may be empty).
      shapes: per-operand shape tuples.
      per_op_symmetries: parallel to operands; SymmetryGroup objects, strings
        ('symmetric', 'cyclic', 'dihedral'), or None.
      identity_pattern: tuple of operand-position tuples that share id.
      partition_budget: per-component partition cap; defaults to global setting.

    Dimino-budget fallback: any internal call to ``_dimino`` that exceeds the
    configured ``dimino_budget`` (default 500_000) raises
    :class:`_DiminoBudgetExceeded`. This function catches the exception,
    emits a :class:`CostFallbackWarning`, and returns an
    :class:`AccumulationCost` with ``fallback_used=True`` and
    ``total = k * dense_baseline`` (the no-symmetry direct-event count) —
    the same shape the partition-budget fallback uses.
    """
    from flopscope._config import get_setting
    from flopscope._perm_group import _DiminoBudgetExceeded

    num_ops = len(input_parts)
    if per_op_symmetries is None:
        per_op_symmetries = (None,) * num_ops
    if partition_budget is None:
        partition_budget = _cast(int, get_setting("partition_budget"))

    operand_names = _operand_names_from_identity_pattern(num_ops, identity_pattern)

    graph = build_bipartite(
        subscripts=tuple(input_parts),
        output=output_subscript,
        operand_names=operand_names,
    )
    matrix_data = build_incidence_matrix(graph)

    # Build per-operand wreath inputs.
    axis_ranks = tuple(len(part) for part in input_parts)
    u_offsets = tuple(sum(axis_ranks[:i]) for i in range(num_ops))
    grouped_ops: set[int] = set()
    for grp in graph.identical_groups:
        grouped_ops.update(grp)
    singleton_groups = [(i,) for i in range(num_ops) if i not in grouped_ops]
    identical_groups_all = (*graph.identical_groups, *singleton_groups)

    # Pre-compute sizes / dense_baseline up-front so the bail-handler can use
    # them without redoing graph work.
    size_map = _build_size_map(input_parts, shapes)
    sizes = tuple(size_map[lbl] for lbl in graph.all_labels)
    dense_baseline = math.prod(sizes) if sizes else 1

    try:
        wreath_elements = list(
            enumerate_wreath(
                identical_groups=identical_groups_all,
                per_op_symmetry=tuple(
                    _per_op_symmetry_for_wreath(s) for s in per_op_symmetries
                ),
                axis_ranks=axis_ranks,
                u_offsets=u_offsets,
            )
        )

        sigma_results = run_sigma_loop(graph, matrix_data, tuple(wreath_elements))
        detected = build_full_group(sigma_results, all_labels=graph.all_labels)

        # Decompose into components.
        components = decompose_into_components(
            detected_group=detected,
            v_labels=graph.free_labels,
            w_labels=graph.summed_labels,
            sizes=sizes,
        )

        component_costs = run_ladder_per_component(
            components,
            partition_budget=partition_budget,
        )
    except _DiminoBudgetExceeded as exc:
        fallback_total = num_ops * dense_baseline
        reason = (
            f"dimino_budget exceeded ({exc.seen_count} > {exc.budget}); "
            f"auto-inferred symmetry group is too large to enumerate exactly"
        )
        warnings.warn(
            CostFallbackWarning(
                f"accumulation: {reason} — charging dense cost {fallback_total} "
                f"= {num_ops} × {dense_baseline}. Raise via "
                f"flopscope.configure(dimino_budget=...) to attempt exact counting."
            ),
            stacklevel=4,
        )
        return AccumulationCost(
            total=fallback_total,
            mu=None,
            alpha=None,
            m_total=1,
            dense_baseline=dense_baseline,
            num_terms=num_ops,
            per_component=(),
            fallback_used=True,
            unavailable_components=(),
            unavailable_reason=reason,
        )

    if num_ops >= 3:
        return _walk_path_and_aggregate(
            canonical_subscripts=canonical_subscripts,
            input_parts=input_parts,
            output_subscript=output_subscript,
            shapes=shapes,
            per_op_symmetries=per_op_symmetries,
            identity_pattern=identity_pattern,
            partition_budget=partition_budget,
            dense_baseline=dense_baseline,
            full_expression_component_costs=component_costs,
        )
    return aggregate_einsum(
        component_costs=component_costs,
        num_terms=num_ops,
        dense_baseline=dense_baseline,
    )


# ── Reduction-cost API hook (designed for, not implemented) ──────────


def aggregate_reduction(
    component_costs: Sequence[ComponentCost],
    *,
    op_factor: int = 1,
    dense_baseline: int,
    output_dense: int,
    extra_ops: int = 0,
) -> AccumulationCost:
    """Aggregate per-component costs into a ufunc.reduce cost.

    Formula:
        total = op_factor × (∏α_c - output_dense) + extra_ops

    ``output_dense`` is the **orbit-aware** output count: the orchestrator
    passes ``_num_output_orbits(input_shape, axes_summed, symmetry)`` through
    this slot. For dense inputs this equals ``prod(output_shape)`` so the
    name is consistent with the locked-stub docstring.

    Fallback (any component unavailable):
        total = output_dense × (input_axis_size - 1) × op_factor + extra_ops
    where input_axis_size = dense_baseline / output_dense.

    See .aicrowd/superpowers/specs/2026-05-13-symmetry-aware-reduction-cost-design.md.
    """
    failing = [i for i, c in enumerate(component_costs) if c.alpha is None]

    m_total = 1
    for c in component_costs:
        m_total *= c.m

    if failing:
        # Floor-division is safe: the orchestrator passes
        # dense_baseline = prod(input_shape) and output_dense = num_output_orbits,
        # which evenly divides for symmetry=None and represents the orbit-discounted
        # input-axis size for symmetric inputs.
        input_axis_size = dense_baseline // output_dense if output_dense else 0
        fallback_total = (
            output_dense * max(0, input_axis_size - 1) * op_factor + extra_ops
        )
        first = component_costs[failing[0]]
        reason = first.unavailable_reason or "partition_budget exceeded"
        labels = ", ".join(first.labels)
        warnings.warn(
            CostFallbackWarning(
                f"reduction: component {list(failing)} ({labels}) returned "
                f"unavailable — charging dense cost {fallback_total}. "
                f"Failing reason: {reason}."
            ),
            stacklevel=4,
        )
        return AccumulationCost(
            total=fallback_total,
            mu=None,
            alpha=None,
            m_total=m_total,
            dense_baseline=dense_baseline,
            num_terms=1,
            per_component=tuple(component_costs),
            fallback_used=True,
            unavailable_components=tuple(failing),
            unavailable_reason=reason,
        )

    alpha_product = 1
    for c in component_costs:
        assert c.alpha is not None  # for type narrowing
        alpha_product *= c.alpha

    # Invariant: output_dense (num_output_orbits) <= alpha_product. The
    # orchestrator computes both consistently. If you ever see a negative
    # total here, the orchestrator passed an inconsistent output_dense.
    total = op_factor * (alpha_product - output_dense) + extra_ops
    return AccumulationCost(
        total=total,
        mu=0,  # k=1 single-operand reduction
        alpha=alpha_product,
        m_total=m_total,
        dense_baseline=dense_baseline,
        num_terms=1,
        per_component=tuple(component_costs),
        fallback_used=False,
    )
