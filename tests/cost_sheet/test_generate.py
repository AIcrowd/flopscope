from __future__ import annotations

import re
from pathlib import Path

import pytest

from flopscope._registry import REGISTRY
from scripts import generate_cost_sheet as gen
from scripts.cost_sheet import render
from scripts.cost_sheet.schema import CostRow

_CHARGED = {
    "counted_unary",
    "counted_binary",
    "counted_reduction",
    "counted_custom",
    "counted_random_method",
}

BuildResult = tuple[list[CostRow], list[str], list[tuple[str, str]]]


@pytest.fixture(scope="module")
def built() -> BuildResult:
    return gen.build_rows()


def test_build_rows_covers_every_registry_op(built: BuildResult) -> None:
    rows, _missing, _failed = built
    assert len(rows) == len(REGISTRY)
    assert {r.op for r in rows} == set(REGISTRY)


def test_resolved_charged_ops_are_measured(built: BuildResult) -> None:
    rows, missing, failed = built
    unmeasured = set(missing) | {op for op, _err in failed}
    measured = [
        r
        for r in rows
        if REGISTRY[r.op].get("category") in _CHARGED and r.op not in unmeasured
    ]
    assert measured, "no charged op was measured at all"
    for r in measured:
        assert isinstance(r.raw_flop_cost, int), r.op
        assert r.example_input, r.op
        assert r.example_input != "(pending curation)", r.op


def test_unmeasured_charged_rows_are_marked(built: BuildResult) -> None:
    rows, missing, failed = built
    by_op = {r.op: r for r in rows}
    for op in missing:
        assert by_op[op].example_input == "(pending curation)"
        assert by_op[op].raw_flop_cost == ""
        assert by_op[op].billed_fp64 == ""
    for op, err in failed:
        assert err, op  # error class recorded
        assert by_op[op].raw_flop_cost == ""


def test_build_rows_deterministic(built: BuildResult) -> None:
    rows2, _m2, _f2 = gen.build_rows()
    assert render.to_csv(built[0]) == render.to_csv(rows2)


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(gen, "CSV", tmp_path / "sheet.csv")
    monkeypatch.setattr(gen, "DT_CSV", tmp_path / "rates.csv")
    monkeypatch.setattr(gen, "HTML", tmp_path / "sheet.html")
    return tmp_path / "sheet.csv"


def test_check_passes_after_generation_and_normalizes_shas(
    built: BuildResult, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    csv_path = _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(gen, "build_rows", lambda: built)
    assert gen.main(check=False) == 0
    assert gen.main(check=True) == 0

    # Rewriting the committed permalink SHAs to a different 40-hex value must
    # NOT trip the check: check compares content modulo the embedded commit.
    text = csv_path.read_text()
    fake = re.sub(r"blob/[0-9a-f]{40}/", "blob/" + "f" * 40 + "/", text)
    assert fake != text, "expected at least one permalink in the sheet"
    csv_path.write_text(fake)
    assert gen.main(check=True) == 0

    # A moved registry line (permalink line number) IS real drift.
    drifted = re.sub(r"#L(\d+)", lambda m: f"#L{int(m.group(1)) + 1}", fake, count=1)
    assert drifted != fake
    csv_path.write_text(drifted)
    assert gen.main(check=True) == 1


def test_check_fails_when_artifacts_missing(
    built: BuildResult, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(gen, "build_rows", lambda: built)
    assert gen.main(check=True) == 1
