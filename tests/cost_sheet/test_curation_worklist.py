from __future__ import annotations

import csv
import re

from flopscope._registry import REGISTRY
from scripts.cost_sheet import curation_worklist as cw

# Independent restatement of the vagueness markers (empty case handled inline)
# so the assertion does not just mirror the module under test.
_MARKERS = re.compile(r"per operand|weight-calibrated|TODO|calibrated", re.I)


def test_worklist_flags_exactly_the_uncurated_ops():
    with cw.SHEET_CSV.open(newline="") as f:
        sheet = {r["op"]: r for r in csv.DictReader(f)}
    worklist = cw.build_worklist()

    assert worklist, "worklist must be non-empty while curation is pending"

    # Every input-coverage reason is backed by an empty raw cost in the sheet.
    for e in worklist:
        if {"no-input", "input-fails"} & set(e["reasons"]):
            row = sheet[e["op"]]
            assert row["status"] == "charged", e["op"]
            assert row["raw_flop_cost"] == "", e["op"]

    # No op with a measured raw cost AND a non-vague formula appears.
    listed = {e["op"] for e in worklist}
    for op, row in sheet.items():
        formula = REGISTRY.get(op, {}).get("cost_formula", "")
        measured = row["status"] == "charged" and row["raw_flop_cost"] != ""
        if measured and formula and not _MARKERS.search(formula):
            assert op not in listed, f"{op} is curated yet appears in the worklist"
