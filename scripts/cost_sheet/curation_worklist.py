"""Emit the curation work-list for the cost-model sheet.

One-shot analysis (may be deleted once curation lands): joins the committed
sheet (input/measurement coverage) with the registry (formula quality) and
writes the JSON work-list the curation workflow fans out over.

An op lands on the list when any reason applies:

- ``no-input``       charged sheet row with no canonical input resolved
                     (``example_input == "(pending curation)"``, empty
                     ``raw_flop_cost``).
- ``input-fails``    charged sheet row whose input resolved but whose
                     measurement raised (``example_input`` present, empty
                     ``raw_flop_cost``).
- ``vague-formula``  registry ``cost_formula`` matches a vagueness marker,
                     or is empty for a charged op.

Input coverage is read from the committed sheet rather than re-derived: the
sheet already records which charged ops measured successfully (ops resolving
via category defaults have a ``raw_flop_cost``), so only rows with an empty
``raw_flop_cost`` still need input curation.

Usage:
    uv run python scripts/cost_sheet/curation_worklist.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from flopscope._registry import REGISTRY

REPO = Path(__file__).resolve().parents[2]
SHEET_CSV = REPO / "docs" / "reference" / "cost-model-sheet.csv"
OUT_JSON = REPO / "scratch" / "cost-sheet-worklist.json"

PENDING = "(pending curation)"

# Vagueness markers. `numel`/`per-element` are deliberately NOT markers:
# `numel(output)` is the exact formula for elementwise ops, not a vague one.
# `^$` documents that an empty formula counts as vague (see formula_is_vague).
VAGUE = re.compile(r"per operand|weight-calibrated|TODO|calibrated|^$", re.I)


def formula_is_vague(formula: str, charged: bool) -> bool:
    """True when the formula needs curation.

    The vagueness markers apply to every op; an EMPTY formula (the regex's
    ``^$`` case) is only flagged for charged ops -- free/blacklisted ops bill
    nothing and carry no ``cost_formula`` by design, so they owe none.
    """
    if formula == "":
        return charged
    return bool(VAGUE.search(formula))


def read_sheet() -> list[dict[str, str]]:
    with SHEET_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def _weight(raw: str) -> float | str:
    try:
        return float(raw)
    except ValueError:
        return raw  # "" when the production-weight snapshot failed for the op


def build_worklist() -> list[dict]:
    rows = {r["op"]: r for r in read_sheet()}
    out: list[dict] = []
    for op in sorted(set(REGISTRY) | set(rows)):
        entry = REGISTRY.get(op, {})
        row = rows.get(op, {})
        if row:
            charged = row["status"] == "charged"
        else:  # registry op missing from the sheet (drift gate should prevent this)
            charged = entry.get("category", "").startswith("counted")
        formula = (
            entry.get("cost_formula", "") if entry else row.get("flop_cost_formula", "")
        )

        reasons: list[str] = []
        if charged and row and row["raw_flop_cost"] == "":
            reasons.append(
                "no-input" if row["example_input"] == PENDING else "input-fails"
            )
        if formula_is_vague(formula, charged):
            reasons.append("vague-formula")
        if not reasons:
            continue
        out.append(
            {
                "op": op,
                "category": entry.get("category", row.get("category", "")),
                "reasons": reasons,
                "current_formula": formula,
                "example_input": row.get("example_input", ""),
                "weight": _weight(row.get("weight", "")),
            }
        )
    return out


def main() -> int:
    worklist = build_worklist()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(worklist, indent=1) + "\n")

    counts = Counter(reason for e in worklist for reason in e["reasons"])
    print(f"{len(worklist)} ops need curation -> {OUT_JSON.relative_to(REPO)}")
    for reason in ("no-input", "input-fails", "vague-formula"):
        print(f"  {reason}: {counts.get(reason, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
