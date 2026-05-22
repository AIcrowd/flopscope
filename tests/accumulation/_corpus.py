"""Hand-curated test corpus mirroring the JS EXAMPLES preset suite.

Each CorpusCase is a self-contained description of a symmetry-aware einsum
that is tested against both the Python ladder and the JS engine.

Per-op symmetry encoding (matches run.mjs expectations):
  - None           → no symmetry
  - 'symmetric'    → full S_k on all axes
  - {'type': 'symmetric', 'axes': [...]}          → partial symmetric
  - {'type': 'cyclic', 'axes': [...]}              → cyclic group
  - {'type': 'custom', 'axes': [...], 'generators': '(0 1), (2 3)'}  → custom

sizes_by_label uses small-to-medium sizes to keep both the JS engine and the
SymPy oracle within their budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    subscripts: str
    output: str
    operand_names: tuple[str, ...]
    per_op_symmetry: tuple[Any, ...] | None
    sizes_by_label: dict[str, int]
    expected_regimes: frozenset[str]
    description: str = ""


# ---------------------------------------------------------------------------
# Trivial regime
# ---------------------------------------------------------------------------

CORPUS: list[CorpusCase] = [
    CorpusCase(
        case_id="matrix-chain",
        subscripts="ij,jk",
        output="ik",
        operand_names=("A", "A"),
        per_op_symmetry=None,
        sizes_by_label={"i": 4, "j": 4, "k": 4},
        expected_regimes=frozenset({"trivial"}),
        description="A·A no symmetry → trivial group, each label its own component",
    ),
    CorpusCase(
        case_id="frobenius",
        subscripts="ij,ij",
        output="",
        operand_names=("A", "A"),
        per_op_symmetry=None,
        sizes_by_label={"i": 4, "j": 4},
        expected_regimes=frozenset({"trivial"}),
        description="Frobenius inner product — operand swap induces identity relabeling",
    ),
    CorpusCase(
        case_id="mixed-chain",
        subscripts="ij,jk,kl",
        output="il",
        operand_names=("A", "B", "A"),
        per_op_symmetry=None,
        sizes_by_label={"i": 4, "j": 4, "k": 4, "l": 4},
        expected_regimes=frozenset({"trivial"}),
        description="A·B·A — middle B pins incidence pattern, no non-trivial symmetry",
    ),
    # ---------------------------------------------------------------------------
    # functionalProjection regime
    # ---------------------------------------------------------------------------
    CorpusCase(
        case_id="bilinear-trace",
        subscripts="ik,jl",
        output="ij",
        operand_names=("A", "A"),
        per_op_symmetry=None,
        sizes_by_label={"i": 4, "j": 4, "k": 4, "l": 4},
        expected_regimes=frozenset({"functionalProjection"}),
        description="Identical ops give Z2 diagonal; every g preserves V → functionalProjection",
    ),
    CorpusCase(
        case_id="trace-product",
        subscripts="ij,ji",
        output="",
        operand_names=("A", "A"),
        per_op_symmetry=None,
        sizes_by_label={"i": 4, "j": 4},
        expected_regimes=frozenset({"functionalProjection"}),
        description="Tr(A·A) — S2{i,j} on summed side, functionalProjection fires",
    ),
    CorpusCase(
        case_id="direct-s2-s2",
        subscripts="abcd",
        output="ab",
        operand_names=("T",),
        per_op_symmetry=(
            {"type": "custom", "axes": [0, 1, 2, 3], "generators": "(0 1), (2 3)"},
        ),
        sizes_by_label={"a": 4, "b": 4, "c": 4, "d": 4},
        expected_regimes=frozenset({"functionalProjection"}),
        description="S2{a,b} × S2{c,d}: every g preserves V → functionalProjection",
    ),
    CorpusCase(
        case_id="direct-s2-c3",
        subscripts="abcde",
        output="ab",
        operand_names=("T",),
        per_op_symmetry=(
            {"type": "custom", "axes": [0, 1, 2, 3, 4], "generators": "(0 1), (2 3 4)"},
        ),
        sizes_by_label={"a": 4, "b": 4, "c": 4, "d": 4, "e": 4},
        expected_regimes=frozenset({"functionalProjection"}),
        description="S2{a,b} × C3{c,d,e}: every g preserves V → functionalProjection",
    ),
    # ---------------------------------------------------------------------------
    # singleton regime
    # ---------------------------------------------------------------------------
    CorpusCase(
        case_id="cross-s2",
        subscripts="ij,k",
        output="ik",
        operand_names=("A", "B"),
        per_op_symmetry=({"type": "symmetric", "axes": [0, 1]}, None),
        sizes_by_label={"i": 4, "j": 4, "k": 4},
        expected_regimes=frozenset({"singleton", "trivial"}),
        description="A symmetric → (i j) crosses V/W; S2{i,j} component → singleton",
    ),
    CorpusCase(
        case_id="cross-s3",
        subscripts="ijk",
        output="i",
        operand_names=("T",),
        per_op_symmetry=({"type": "symmetric", "axes": [0, 1, 2]},),
        sizes_by_label={"i": 4, "j": 4, "k": 4},
        expected_regimes=frozenset({"singleton"}),
        description="S3{i,j,k} crosses V/W, |V|=1 → singleton equation",
    ),
    CorpusCase(
        case_id="cyclic-cross",
        subscripts="ijk",
        output="i",
        operand_names=("T",),
        per_op_symmetry=({"type": "cyclic", "axes": [0, 1, 2]},),
        sizes_by_label={"i": 4, "j": 4, "k": 4},
        expected_regimes=frozenset({"singleton"}),
        description="C3{i,j,k} crosses V/W, |V|=1 → singleton equation",
    ),
    # ---------------------------------------------------------------------------
    # young regime
    # ---------------------------------------------------------------------------
    CorpusCase(
        case_id="young-s3",
        subscripts="abc",
        output="ab",
        operand_names=("T",),
        per_op_symmetry=({"type": "symmetric", "axes": [0, 1, 2]},),
        sizes_by_label={"a": 4, "b": 4, "c": 4},
        expected_regimes=frozenset({"young"}),
        description="Full S3 with cross V/W elements and |V|=2 → Young regime",
    ),
    CorpusCase(
        case_id="young-s4-v2w2",
        subscripts="abcd",
        output="ab",
        operand_names=("T",),
        per_op_symmetry=({"type": "symmetric", "axes": [0, 1, 2, 3]},),
        sizes_by_label={"a": 4, "b": 4, "c": 4, "d": 4},
        expected_regimes=frozenset({"young"}),
        description="Full S4 on rank-4 symmetric T, |V|=2, |W|=2 → Young closed form",
    ),
    CorpusCase(
        case_id="young-s4-v3w1",
        subscripts="abcd",
        output="abc",
        operand_names=("T",),
        per_op_symmetry=({"type": "symmetric", "axes": [0, 1, 2, 3]},),
        sizes_by_label={"a": 4, "b": 4, "c": 4, "d": 4},
        expected_regimes=frozenset({"young"}),
        description="Full S4 on rank-4 symmetric T, |V|=3, |W|=1 → Young closed form",
    ),
    # ---------------------------------------------------------------------------
    # partitionCount regime
    # ---------------------------------------------------------------------------
    CorpusCase(
        case_id="cross-c3-partial",
        subscripts="abc",
        output="ab",
        operand_names=("T",),
        per_op_symmetry=({"type": "cyclic", "axes": [0, 1, 2]},),
        sizes_by_label={"a": 4, "b": 4, "c": 4},
        expected_regimes=frozenset({"partitionCount"}),
        description="C3 cross-V/W, |V|=2 — Young refuses (|G|=3≠3!), partitionCount handles it",
    ),
]
