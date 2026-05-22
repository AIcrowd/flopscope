"""Shape classifier — Stage 1 of the cost ladder.

Port of website/components/symmetry-aware-einsum-contractions/engine/shapeLayer.js.

The shape classifier is structural and runs before the regime ladder. It produces
a label that goes onto the per-component output for diagnostic display alongside
the regime ID. (regime_id is computational; shape is structural.)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from flopscope._perm_group import _Permutation as Permutation

Shape = Literal["trivial", "allVisible", "allSummed", "mixed"]

SHAPES: tuple[Shape, ...] = ("trivial", "allVisible", "allSummed", "mixed")


def detect_shape(
    *,
    va: Sequence[str],
    wa: Sequence[str],
    elements: Sequence[Permutation],
) -> Shape:
    """Classify a component's structural shape from its V/W partition and group size."""
    if not elements or len(elements) <= 1:
        return "trivial"
    if len(wa) == 0:
        return "allVisible"
    if len(va) == 0:
        return "allSummed"
    return "mixed"
