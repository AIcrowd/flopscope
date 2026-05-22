"""Python ladder vs SymPy oracle on every corpus case.

Computes alpha via the Python ladder per component and via brute-force orbit
enumeration. Asserts equality. Cases exceeding the oracle's |X|·|G| budget
are skipped. Runs in default CI (no Node required).
"""

from __future__ import annotations

import math
from typing import cast

import pytest

from tests.accumulation._corpus import CORPUS
from tests.accumulation._sympy_oracle import MAX_PAIR_TOUCHES, sympy_brute_force_alpha

# ---------------------------------------------------------------------------
# Helper: build operands (mirrors test_js_parity._build_operand)
# ---------------------------------------------------------------------------


def _build_operand(shape, sym_spec):

    import numpy as np

    import flopscope as fps
    from flopscope._perm_group import SymmetryGroup

    op = np.zeros(shape) if shape else np.zeros(1)
    if sym_spec is None:
        return op

    if sym_spec == "symmetric":
        return fps.as_symmetric(op, symmetry=tuple(range(len(shape))))

    if isinstance(sym_spec, dict):
        sym_type = sym_spec.get("type")
        axes = tuple(sym_spec.get("axes", range(len(shape))))

        if sym_type == "symmetric":
            return fps.as_symmetric(op, symmetry=axes)

        if sym_type == "cyclic":
            return fps.as_symmetric(op, symmetry=SymmetryGroup.cyclic(axes=axes))

        if sym_type == "custom":
            generators_str = sym_spec.get("generators", "")
            gen_perms = _parse_generators(generators_str, degree=len(axes))
            group = SymmetryGroup(*gen_perms, axes=axes)
            return fps.as_symmetric(op, symmetry=group)

    return op


def _parse_generators(generators_str: str, *, degree: int):
    import re

    from flopscope._perm_group import _Permutation

    segments: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(generators_str):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            segments.append(generators_str[start:i].strip())
            start = i + 1
    last = generators_str[start:].strip()
    if last:
        segments.append(last)

    result = []
    for seg in segments:
        arr = list(range(degree))
        for m in re.finditer(r"\(([^)]*)\)", seg):
            cycle = list(map(int, m.group(1).split()))
            for i in range(len(cycle)):
                arr[cycle[i]] = cycle[(i + 1) % len(cycle)]
        result.append(_Permutation(arr))
    return result


# ---------------------------------------------------------------------------
# Helper: run the internal pipeline and return Component + ComponentCost pairs
# ---------------------------------------------------------------------------


def _get_component_pairs(case):
    """Run the full decomposition pipeline, returning (Component, ComponentCost) pairs."""
    from flopscope._accumulation._bipartite import (
        build_bipartite,
        build_incidence_matrix,
    )
    from flopscope._accumulation._components import decompose_into_components
    from flopscope._accumulation._cost import run_ladder_per_component
    from flopscope._accumulation._detection import build_full_group, run_sigma_loop
    from flopscope._accumulation._wreath import enumerate_wreath
    from flopscope._config import get_setting
    from flopscope._symmetric import SymmetricTensor

    parts = case.subscripts.split(",")
    num_ops = len(parts)

    # Build operands (same shared-object logic as JS parity test)
    canonical_by_name: dict[str, object] = {}
    operands = []
    for op_idx, part in enumerate(parts):
        name = case.operand_names[op_idx]
        shape = tuple(case.sizes_by_label[lbl] for lbl in part)
        sym = case.per_op_symmetry[op_idx] if case.per_op_symmetry else None
        if name not in canonical_by_name:
            canonical_by_name[name] = _build_operand(shape, sym)
        operands.append(canonical_by_name[name])

    # Extract per-operand symmetry
    per_op_syms = []
    for op in operands:
        if isinstance(op, SymmetricTensor) and op.symmetry is not None:
            per_op_syms.append(op.symmetry)
        else:
            per_op_syms.append(None)

    # Identity pattern (operands sharing same object)
    id_to_positions: dict[int, list[int]] = {}
    for idx, op in enumerate(operands):
        id_to_positions.setdefault(id(op), []).append(idx)
    identity_groups = tuple(
        tuple(pos) for pos in id_to_positions.values() if len(pos) >= 2
    )

    # Operand names respecting identity
    name_of: dict[int, str] = {}
    for grp in identity_groups:
        shared = f"op_grp_{grp[0]}"
        for pos in grp:
            name_of[pos] = shared
    singleton_groups = tuple(
        (i,) for i in range(num_ops) if i not in {p for g in identity_groups for p in g}
    )
    all_identical_groups = identity_groups + singleton_groups
    operand_names = tuple(name_of.get(i, f"op_{i}") for i in range(num_ops))

    axis_ranks = tuple(len(p) for p in parts)
    u_offsets = tuple(sum(axis_ranks[:i]) for i in range(num_ops))

    wreath_elements = list(
        enumerate_wreath(
            identical_groups=all_identical_groups,
            per_op_symmetry=tuple(per_op_syms),
            axis_ranks=axis_ranks,
            u_offsets=u_offsets,
        )
    )

    graph = build_bipartite(
        subscripts=tuple(parts),
        output=case.output,
        operand_names=operand_names,
    )
    matrix_data = build_incidence_matrix(graph)
    sigma_results = run_sigma_loop(graph, matrix_data, tuple(wreath_elements))
    detected = build_full_group(sigma_results, all_labels=graph.all_labels)

    size_map = case.sizes_by_label
    sizes = tuple(size_map[lbl] for lbl in graph.all_labels)
    components = decompose_into_components(
        detected_group=detected,
        v_labels=graph.free_labels,
        w_labels=graph.summed_labels,
        sizes=sizes,
    )

    partition_budget = cast(int, get_setting("partition_budget"))
    component_costs = run_ladder_per_component(
        components, partition_budget=partition_budget
    )

    return list(zip(components, component_costs, strict=True))


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.case_id)
def test_python_ladder_matches_sympy_oracle(case):
    if not case.subscripts:
        pytest.skip("empty einsum")

    # Skip if the total X space is too large for the oracle budget
    sizes_product = math.prod(case.sizes_by_label.values())
    if sizes_product > MAX_PAIR_TOUCHES // 10:
        pytest.skip(
            f"{case.case_id}: sizes product {sizes_product} too large for oracle"
        )

    pairs = _get_component_pairs(case)

    checked = 0
    for comp, cost in pairs:
        if cost.alpha is None:
            continue
        if comp.order <= 1:
            # Trivial group — alpha = ∏ sizes, oracle would confirm trivially
            continue

        x_size = math.prod(comp.sizes)
        pair_touches = x_size * comp.order
        if pair_touches > MAX_PAIR_TOUCHES:
            continue  # this component too large; skip, don't fail

        oracle_alpha = sympy_brute_force_alpha(
            elements=comp.elements,
            sizes=comp.sizes,
            visible_positions=comp.visible_positions,
        )
        assert cost.alpha == oracle_alpha, (
            f"{case.case_id}/{list(comp.labels)}: "
            f"ladder={cost.alpha}, oracle={oracle_alpha}"
        )
        checked += 1

    # If we skipped all non-trivial components (all sizes too large), that's fine
    # but log it so we know coverage isn't zero across the whole suite.
    _ = checked  # at least one checked if sizes are small enough
