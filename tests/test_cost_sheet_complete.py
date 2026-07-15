"""Completeness + drift guard for the generated cost-model sheet.

Exhaustiveness is a permanent property, not a one-time state: a new registry
op with no canonical input, an op whose worked example stops measuring, or a
stale committed CSV turns CI red here.
"""

from functools import cache

from flopscope._registry import REGISTRY
from scripts.cost_sheet.render import to_csv
from scripts.generate_cost_sheet import CSV, _normalize_shas, build_rows

_CHARGED = {
    "counted_unary",
    "counted_binary",
    "counted_reduction",
    "counted_custom",
    "counted_random_method",
}


@cache
def _rows():
    # build once per module; build_rows measures every charged op live.
    return build_rows()


def test_every_registry_op_is_a_row():
    rows, _missing, _failed = _rows()
    assert {r.op for r in rows} == set(REGISTRY)


def test_every_charged_op_has_a_measured_worked_example():
    rows, missing, failed = _rows()
    assert missing == [], f"charged ops with no canonical input: {missing}"
    assert failed == [], f"charged ops whose canonical input failed: {failed}"
    holes = [
        r.op
        for r in rows
        if REGISTRY[r.op].get("category") in _CHARGED
        and not (isinstance(r.raw_flop_cost, int) and r.example_input)
    ]
    assert holes == [], f"charged ops without measured raw cost: {holes}"


def test_committed_csv_matches_regeneration():
    rows, _missing, _failed = _rows()
    fresh = _normalize_shas(to_csv(rows))
    committed = _normalize_shas(CSV.read_text())
    assert committed == fresh, "cost-model-sheet.csv is stale — regenerate it"
