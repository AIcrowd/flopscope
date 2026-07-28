"""Diff two observation records dimension by dimension.

Each dimension is compared and reported independently so that allowlisting a
dtype divergence still lets a value divergence in the same case fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Every dimension the harness can report, in report order.
DIMENSIONS: tuple[str, ...] = (
    "outcome",
    "exc_type",
    "exc_bases",
    "value",
    "dtype",
    "shape",
    "container",
    "pytype",
    "flops",
)

#: Compared only when BOTH sides returned a value.
_RETURNED_ONLY = ("value", "dtype", "shape", "container", "pytype")

#: Compared only when BOTH sides raised.
_RAISED_ONLY = ("exc_type", "exc_bases")


@dataclass(frozen=True)
class Divergence:
    """One dimension on which the two backends disagreed for one case."""

    case_id: str
    dimension: str
    inproc: Any
    client: Any

    def key(self) -> tuple[str, str]:
        """Allowlist key: divergences are allowlisted per (case, dimension)."""
        return (self.case_id, self.dimension)


def compare_observations(case_id: str, inproc: dict, client: dict) -> list[Divergence]:
    """Return every dimension on which *inproc* and *client* disagree."""
    out: list[Divergence] = []

    def add(dimension: str) -> None:
        out.append(
            Divergence(case_id, dimension, inproc.get(dimension), client.get(dimension))
        )

    if inproc.get("outcome") != client.get("outcome"):
        add("outcome")
        # Outcomes differ, so the shape-specific dimensions are not comparable.
        # FLOPs still are: charging for a call that failed on one side only is
        # exactly the billing defect this harness exists to catch.
        if inproc.get("flops") != client.get("flops"):
            add("flops")
        return out

    outcome = inproc.get("outcome")
    if outcome == "returned":
        for dimension in _RETURNED_ONLY:
            if inproc.get(dimension) != client.get(dimension):
                add(dimension)
    elif outcome == "raised":
        for dimension in _RAISED_ONLY:
            if inproc.get(dimension) != client.get(dimension):
                add(dimension)

    if inproc.get("flops") != client.get("flops"):
        add("flops")
    return out
