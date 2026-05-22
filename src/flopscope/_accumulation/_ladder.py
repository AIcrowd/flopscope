"""Regime ladder dispatcher and supporting data types.

Port of website/components/symmetry-aware-einsum-contractions/engine/accumulationCount.js
plus the regime contract from regimes/index.js.

The ladder runs per-component:
    Stage 1: trivial short-circuit (|G| ≤ 1)
    Stage 2a: functionalProjection takes priority (covers shape ∈ {allVisible, allSummed,
              and mixed-but-functional})
    Stage 2b: mixed regimes ladder — singleton, young, partitionCount
    Fallthrough: 'unavailable' (brute-force orbit B.8 is excluded by policy)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from flopscope._perm_group import _Permutation as Permutation

from ._shape import Shape

RegimeId = Literal[
    "trivial",
    "functionalProjection",
    "singleton",
    "young",
    "partitionCount",
    "unavailable",
]

Decision = Literal["fired", "refused"]


@dataclass(frozen=True)
class RegimeContext:
    """Input to a regime's recognize() and compute()."""

    labels: tuple[str, ...]
    va: tuple[str, ...]
    wa: tuple[str, ...]
    elements: tuple[Permutation, ...]
    generators: tuple[Permutation, ...]
    sizes: tuple[int, ...]
    visible_positions: tuple[int, ...]
    partition_budget: int


@dataclass(frozen=True)
class Verdict:
    """Output of regime.recognize()."""

    fired: bool
    reason: str


@dataclass(frozen=True)
class RegimeOutput:
    """Output of regime.compute()."""

    count: int
    sub_steps: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Regime:
    """A regime is identified by id and consists of recognize + compute callables."""

    id: RegimeId
    recognize: Callable[[RegimeContext], Verdict]
    compute: Callable[[RegimeContext], RegimeOutput]


@dataclass(frozen=True)
class RegimeStep:
    """One entry in the regime-dispatch trace."""

    regime_id: RegimeId
    decision: Decision
    reason: str
    sub_steps: tuple[dict, ...] = ()


@dataclass(frozen=True)
class AccumulationResult:
    """Output of compute_accumulation() — the ladder's primitive output for one
    component. Reused by future reduction-cost.

    `count` is None when regime_id == 'unavailable' (partition budget exceeded
    with brute-force disabled by policy)."""

    count: int | None
    regime_id: RegimeId
    shape: Shape
    trace: tuple[RegimeStep, ...]


# ── Dispatcher ───────────────────────────────────────────────────────


import math

from flopscope._config import get_setting

from ._regimes import FUNCTIONAL_PROJECTION_REGIME, MIXED_REGIMES
from ._shape import detect_shape


def compute_accumulation(
    *,
    labels: Sequence[str],
    va: Sequence[str],
    wa: Sequence[str],
    elements: Sequence[Permutation],
    generators: Sequence[Permutation],
    sizes: Sequence[int],
    visible_positions: Sequence[int],
    partition_budget: int | None = None,
) -> AccumulationResult:
    """Run the regime ladder for a single component.

    Stages mirror accumulationCount.js:
      1. trivial short-circuit for |G| <= 1
      2a. functionalProjection priority check (covers shape allVisible/allSummed
          and mixed-but-functional)
      2b. mixed-shape ladder: singleton, young, partitionCount

    Returns AccumulationResult with count=None when no regime fires within
    the partition budget (brute-force orbit B.8 is excluded by policy).
    """
    if partition_budget is None:
        partition_budget = cast(int, get_setting("partition_budget"))

    shape = detect_shape(va=va, wa=wa, elements=elements)

    # Stage 1: trivial short-circuit
    if not elements or len(elements) <= 1:
        return AccumulationResult(
            count=math.prod(sizes) if sizes else 1,
            regime_id="trivial",
            shape=shape,
            trace=(RegimeStep("trivial", "fired", "|G| = 1"),),
        )

    ctx = RegimeContext(
        labels=tuple(labels),
        va=tuple(va),
        wa=tuple(wa),
        elements=tuple(elements),
        generators=tuple(generators),
        sizes=tuple(sizes),
        visible_positions=tuple(visible_positions),
        partition_budget=partition_budget,
    )

    trace: list[RegimeStep] = []

    # Stage 2a: functionalProjection priority
    verdict = FUNCTIONAL_PROJECTION_REGIME.recognize(ctx)
    if verdict.fired:
        out = FUNCTIONAL_PROJECTION_REGIME.compute(ctx)
        trace.append(
            RegimeStep(
                "functionalProjection",
                "fired",
                verdict.reason,
                out.sub_steps,
            )
        )
        return AccumulationResult(
            out.count, "functionalProjection", shape, tuple(trace)
        )
    trace.append(RegimeStep("functionalProjection", "refused", verdict.reason))

    # Stage 2b: mixed-shape ladder
    for regime in MIXED_REGIMES:
        verdict = regime.recognize(ctx)
        if not verdict.fired:
            trace.append(RegimeStep(regime.id, "refused", verdict.reason))
            continue
        out = regime.compute(ctx)
        trace.append(RegimeStep(regime.id, "fired", verdict.reason, out.sub_steps))
        return AccumulationResult(out.count, regime.id, shape, tuple(trace))

    # Fallthrough: brute-force is excluded by policy → unavailable
    trace.append(
        RegimeStep(
            "unavailable",
            "fired",
            "no exact regime fired within partition budget; brute-force disabled by policy",
        )
    )
    return AccumulationResult(None, "unavailable", shape, tuple(trace))
