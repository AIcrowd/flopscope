from __future__ import annotations

import csv
import re

from flopscope._registry import REGISTRY
from scripts.cost_sheet import curation_worklist as cw

# Independent restatement of the vagueness markers (empty case handled inline)
# so the assertion does not just mirror the module under test.
_MARKERS = re.compile(r"per operand|weight-calibrated|TODO|calibrated", re.I)


def test_worklist_is_empty_now_that_curation_landed():
    # Curation is complete: every charged op has a measured raw cost and a
    # curated, non-vague cost_formula. A non-empty worklist means a new or
    # changed op owes curation (each entry's "reasons" says what is missing).
    worklist = cw.build_worklist()
    assert worklist == [], [(e["op"], e["reasons"]) for e in worklist]


def test_every_charged_op_is_curated_in_sheet_and_registry():
    # Independent restatement (sheet + registry directly, not build_worklist):
    # each charged row measured successfully and its formula is non-vague.
    with cw.SHEET_CSV.open(newline="") as f:
        sheet = {r["op"]: r for r in csv.DictReader(f)}
    assert sheet, "committed sheet is missing or empty"
    for op, row in sheet.items():
        if row["status"] != "charged":
            continue
        formula = REGISTRY.get(op, {}).get("cost_formula", "")
        assert formula, f"{op}: charged op without a cost_formula"
        assert not _MARKERS.search(formula), f"{op}: vague formula {formula!r}"
        assert row["raw_flop_cost"] != "", f"{op}: charged op without a measured cost"
