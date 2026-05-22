"""Python-vs-JS parity tests: assert per-component (m, alpha, regime_id) match.

Skipped automatically when Node.js is not on PATH.
"""

from __future__ import annotations

import pytest

from tests.accumulation._corpus import CORPUS
from tests.accumulation._js_oracle import is_available, run_js_oracle

pytestmark = pytest.mark.skipif(
    not is_available(), reason="Node.js / JS oracle not available"
)


# ---------------------------------------------------------------------------
# Helper: build SymmetricTensor operands from corpus per_op_symmetry specs
# ---------------------------------------------------------------------------


def _build_operand(shape, sym_spec):
    """Build a flopscope operand (SymmetricTensor or numpy array) from a corpus sym_spec."""
    import numpy as np

    import flopscope as fps
    from flopscope._perm_group import SymmetryGroup

    op = np.zeros(shape) if shape else np.zeros(1)

    if sym_spec is None:
        return op

    if sym_spec == "symmetric":
        axes = tuple(range(len(shape)))
        return fps.as_symmetric(op, symmetry=axes)

    if isinstance(sym_spec, dict):
        sym_type = sym_spec.get("type")
        axes = tuple(sym_spec.get("axes", range(len(shape))))

        if sym_type == "symmetric":
            return fps.as_symmetric(op, symmetry=axes)

        if sym_type == "cyclic":
            group = SymmetryGroup.cyclic(axes=axes)
            return fps.as_symmetric(op, symmetry=group)

        if sym_type == "custom":
            generators_str = sym_spec.get("generators", "")
            # Parse cycle notation string: "(0 1), (2 3)" → list of Permutations
            gen_perms = _parse_generators(generators_str, degree=len(axes))
            group = SymmetryGroup(*gen_perms, axes=axes)
            return fps.as_symmetric(op, symmetry=group)

    return op


def _parse_generators(generators_str: str, *, degree: int):
    """Parse cycle-notation generator string into _Permutation list."""
    from flopscope._perm_group import _Permutation

    # Split on commas outside parentheses
    segments = []
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
        # Each segment like "(0 1)(2 3)"
        arr = list(range(degree))
        import re

        for m in re.finditer(r"\(([^)]*)\)", seg):
            cycle = list(map(int, m.group(1).split()))
            for i in range(len(cycle)):
                arr[cycle[i]] = cycle[(i + 1) % len(cycle)]
        result.append(_Permutation(arr))
    return result


def _compute_python_cost(case):
    import flopscope as fps

    if not case.subscripts:
        pytest.skip("empty einsum")

    parts = case.subscripts.split(",")

    # Build one canonical operand per unique name so that identical-operand
    # detection (which uses Python object id()) matches JS name-based detection.
    canonical_by_name: dict[str, object] = {}
    operands = []
    for op_idx, part in enumerate(parts):
        name = case.operand_names[op_idx]
        shape = tuple(case.sizes_by_label[lbl] for lbl in part)
        sym = case.per_op_symmetry[op_idx] if case.per_op_symmetry else None
        if name not in canonical_by_name:
            canonical_by_name[name] = _build_operand(shape, sym)
        operands.append(canonical_by_name[name])

    return fps.einsum_accumulation_cost(
        case.subscripts + "->" + case.output,
        *operands,
    )


# ---------------------------------------------------------------------------
# Parametrized parity tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.case_id)
def test_python_matches_js_per_component(case):
    # TODO: extend JS engine to support path-aware accumulation (spec §10 —
    # JS path-awareness deferred to a follow-up PR). Until then, only
    # compare binary einsums.
    num_ops = len(case.operand_names)
    if num_ops >= 3:
        import pytest

        pytest.skip(
            f"{case.case_id}: k={num_ops} JS-parity check deferred "
            f"(JS engine still single-step; see spec §10)"
        )

    py_cost = _compute_python_cost(case)
    js_result = run_js_oracle(
        subscripts=case.subscripts,
        output=case.output,
        operand_names=case.operand_names,
        per_op_symmetry=case.per_op_symmetry,
        sizes_by_label=case.sizes_by_label,
    )

    py_components = py_cost.per_component
    js_components = js_result["components"]

    assert len(py_components) == len(js_components), (
        f"{case.case_id}: Python has {len(py_components)} components, "
        f"JS has {len(js_components)}"
    )

    py_by_labels = {tuple(c.labels): c for c in py_components}
    js_by_labels = {tuple(c["labels"]): c for c in js_components}

    assert set(py_by_labels.keys()) == set(js_by_labels.keys()), (
        f"{case.case_id}: label sets differ.\n"
        f"  Python: {set(py_by_labels.keys())}\n"
        f"  JS:     {set(js_by_labels.keys())}"
    )

    for labels, py_c in py_by_labels.items():
        js_c = js_by_labels[labels]
        assert py_c.m == js_c["m"], (
            f"{case.case_id}/{list(labels)}: m mismatch: Python={py_c.m}, JS={js_c['m']}"
        )
        assert py_c.alpha == js_c["alpha"], (
            f"{case.case_id}/{list(labels)}: alpha mismatch: Python={py_c.alpha}, JS={js_c['alpha']}"
        )
        assert py_c.regime_id == js_c["regimeId"], (
            f"{case.case_id}/{list(labels)}: regime mismatch: Python={py_c.regime_id!r}, JS={js_c['regimeId']!r}"
        )
