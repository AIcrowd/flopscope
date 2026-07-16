"""Config-level tests for the generalized Sheets uploader.

These tests assert preset configuration and pure row/column logic only.
No real ``gws`` CLI invocation (and therefore no network) ever happens
here: importing ``scripts.upload_to_sheets`` must be side-effect free, and
the one test that drives ``upload_data`` does so against an in-memory
``gws`` stub.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

import scripts.upload_to_sheets as upload_mod
from scripts.upload_to_sheets import (
    _COST_MODEL_BILLED_COLUMNS,
    _COST_MODEL_GROUPS,
    PRESETS,
    SheetPreset,
    _append_ship_empty_columns,
    _contiguous_runs,
    _cost_model_format_requests,
    _dropdown_requests,
    _find_reviewer_columns,
    _reorder_columns,
    upload_data,
)

# The reviewer-ergonomics layout of the cost-model sheet, pinned exactly:
# identity, reviewer block, cost model, evidence, billed matrix, provenance.
_REVIEW_LAYOUT: tuple[str, ...] = (
    "op",
    "module",
    "status",
    "category",
    "looks-right?",
    "proposed-change",
    "reviewer-notes",
    "weight",
    "flop_cost_formula",
    "complex_factor",
    "dtype_rate_rule",
    "example_input",
    "raw_flop_cost",
    "raw_flop_cost_2x",
    "billed_int16",
    "billed_fp32",
    "billed_fp64",
    "billed_complex128",
    "complex_penalty",
    "notes",
    "numpy_range",
    "registry_ref",
    "cost_impl_ref",
)


def test_cost_model_preset_exists_and_is_keyed_by_op() -> None:
    p = PRESETS["cost-model"]
    assert p.title and p.key_column == "op"
    assert "cost-model-sheet.csv" in str(p.csv_path)
    # reviewer-annotation columns the sheet ships empty for the team to fill
    assert "looks-right?" in p.preserve_columns


def test_cost_model_preset_full_config() -> None:
    p = PRESETS["cost-model"]
    assert p.title == "flopscope Cost Model Review"
    assert set(p.preserve_columns) == {
        "looks-right?",
        "proposed-change",
        "reviewer-notes",
    }
    assert p.csv_path.name == "cost-model-sheet.csv"
    assert p.csv_path.exists(), "generator output missing (tracked in git)"
    assert tuple(p.dropdown_columns["looks-right?"]) == ("yes", "no", "unsure")
    # The annotation columns are not in the generated CSV; the uploader
    # appends them as empty columns so the sheet ships ready to review.
    assert p.ship_empty_columns == ("looks-right?", "proposed-change", "reviewer-notes")


def test_cost_model_column_order_is_the_review_layout() -> None:
    """The uploaded sheet's column order is pinned to the review layout."""
    p = PRESETS["cost-model"]
    assert p.column_order == _REVIEW_LAYOUT
    # The order names every uploaded column exactly once: all CSV columns
    # plus the three shipped-empty annotation columns, nothing else.
    with open(p.csv_path, newline="") as f:
        csv_header = next(csv.reader(f))
    assert set(_REVIEW_LAYOUT) == set(csv_header) | set(p.ship_empty_columns)
    assert len(_REVIEW_LAYOUT) == len(csv_header) + len(p.ship_empty_columns)
    # Presentation-only: the committed CSV artifact keeps its own order.
    assert csv_header != list(_REVIEW_LAYOUT)[: len(csv_header)]


def test_weights_preset_reproduces_legacy_config() -> None:
    """Back-compat: the weights preset must match the pre-refactor constants."""
    p = PRESETS["weights"]
    assert p.title == "flopscope FLOP Weight Calibration Review"
    assert p.key_column == "Operation"
    assert p.csv_path.name == "weights.csv"
    assert p.csv_path.exists()
    # The four reviewer-owned columns previously hardcoded as _ALWAYS_PRESERVE.
    assert set(p.preserve_columns) == {
        "Reviewer Weight",
        "Reviewer Notes",
        "Review Status",
        "Post Review Action",
    }
    # Existing Status dropdown options, unchanged.
    assert tuple(p.dropdown_columns["Status"]) == (
        "benchmarked",
        "alias",
        "excluded",
        "free",
        "blacklisted",
        "blacklisted-by-reviewer",
        "keep",
    )
    # weights.csv already ships its reviewer column ("Reviewer Weight");
    # the uploader must not append anything extra.
    assert p.ship_empty_columns == ()
    # No column_order: the weights sheet keeps its byte-identical legacy
    # layout (its format hook is positional against the live sheet).
    assert p.column_order is None


def test_ship_empty_columns_appended_once() -> None:
    rows = [["op", "weight"], ["abs", "1.0"], ["add", "1.0"]]
    cols = ("looks-right?", "proposed-change", "reviewer-notes")
    out = _append_ship_empty_columns(rows, cols)
    assert out[0] == [
        "op",
        "weight",
        "looks-right?",
        "proposed-change",
        "reviewer-notes",
    ]
    assert out[1] == ["abs", "1.0", "", "", ""]
    assert out[2] == ["add", "1.0", "", "", ""]
    # Idempotent: columns already present are not duplicated.
    assert _append_ship_empty_columns(out, cols) == out
    # No-op preset (weights) leaves rows untouched.
    assert _append_ship_empty_columns(rows, ()) == rows


def test_find_reviewer_columns_respects_preset_preserve() -> None:
    sheet = ["op", "weight", "looks-right?", "reviewer-extra"]
    csv_cols = ["op", "weight", "looks-right?"]
    # "looks-right?" is CSV-shipped but reviewer-owned (preserve set);
    # "reviewer-extra" is a column the reviewer added on the sheet.
    idx = _find_reviewer_columns(sheet, csv_cols, frozenset({"looks-right?"}))
    assert idx == [2, 3]
    # Without the preserve marker, the CSV-shipped column would be overwritten.
    idx_no_preserve = _find_reviewer_columns(sheet, csv_cols, frozenset())
    assert idx_no_preserve == [3]


def test_reorder_columns_pure_semantics() -> None:
    """Named headers lead in order; unknown tail follows; missing skipped."""
    rows = [
        ["a", "b", "c", "d"],
        ["1", "2", "3", "4"],
        ["5", "6", "7"],  # ragged row: padded to header width, then permuted
    ]
    out = _reorder_columns(rows, ("c", "zz", "a"))  # "zz" not in data: skipped
    assert out[0] == ["c", "a", "b", "d"]  # b/d keep their relative order
    assert out[1] == ["3", "1", "2", "4"]
    assert out[2] == ["7", "5", "6", ""]
    # The input is not mutated.
    assert rows[0] == ["a", "b", "c", "d"]
    # No column_order (weights preset) and empty order are strict no-ops.
    assert _reorder_columns(rows, None) is rows
    assert _reorder_columns(rows, ()) is rows
    # Already in order: returned unchanged.
    ordered = [["a", "b"], ["1", "2"]]
    assert _reorder_columns(ordered, ("a",)) is ordered
    # Empty data is fine.
    assert _reorder_columns([], ("a",)) == []


def test_contiguous_runs_coalesce() -> None:
    assert _contiguous_runs([]) == []
    assert _contiguous_runs([4, 2, 3, 9, 0]) == [(0, 1), (2, 5), (9, 10)]
    assert _contiguous_runs([1, 1, 2]) == [(1, 3)]


class _FakeGws:
    """In-memory stand-in for the gws CLI wrapper used by upload_data."""

    def __init__(self, sheet_rows: list[list[str]]) -> None:
        self.sheet_rows = sheet_rows
        self.uploaded: list[list[str]] = []
        self.cleared = False

    def __call__(self, *args: str, json_body: dict | None = None) -> dict:
        if "get" in args:
            return {"values": self.sheet_rows}
        if "clear" in args:
            self.cleared = True
            return {}
        if "update" in args and json_body is not None:
            # _upload_all_rows writes sequential row chunks; concatenating
            # them reconstructs the full uploaded grid.
            self.uploaded.extend(json_body["values"])
            return {}
        raise AssertionError(f"unexpected gws call: {args}")


def test_reviewer_annotations_survive_reorder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Re-upload under a NEW column_order keeps annotations key-aligned.

    The live sheet holds the pre-reorder layout (annotations at the tail,
    verdicts filled in, plus a reviewer-added scratch column); the fresh
    upload declares a column_order and a CSV whose rows moved. Reviewer
    values must land in the reordered columns, aligned by the "op" key.
    """
    preset = SheetPreset(
        title="mini cost model",
        key_column="op",
        dropdown_columns={},
        preserve_columns=frozenset({"looks-right?"}),
        csv_path=tmp_path / "unused.csv",
        ship_empty_columns=("looks-right?",),
        column_order=("op", "looks-right?", "weight"),
    )
    sheet = [
        ["op", "weight", "notes", "looks-right?", "scratch"],
        ["add", "9.9", "stale", "yes", "keep-me"],
        ["abs", "9.8", "stale2", "no", "hmm"],
    ]
    fake = _FakeGws(sheet)
    monkeypatch.setattr(upload_mod, "gws", fake)

    # Fresh CSV: abs/add swapped vs the sheet, one new row, updated values.
    rows = [
        ["op", "weight", "notes"],
        ["abs", "1.0", "n-abs"],
        ["add", "2.0", "n-add"],
        ["new", "3.0", "n-new"],
    ]
    # Mirror main(): append shipped-empty columns, then reorder at load time.
    rows = _append_ship_empty_columns(rows, preset.ship_empty_columns)
    rows = _reorder_columns(rows, preset.column_order)

    headers = upload_data("sheet-id", rows, preset)

    assert fake.cleared
    # Named columns lead; unnamed CSV column ("notes") and the reviewer's
    # own sheet-only column ("scratch") follow, keeping relative order.
    assert headers == ["op", "looks-right?", "weight", "notes", "scratch"]
    assert fake.uploaded[0] == headers
    # CSV fields refreshed; reviewer fields realigned by key across the row
    # swap and the insertion — all inside the new column order.
    assert fake.uploaded[1:] == [
        ["abs", "no", "1.0", "n-abs", "hmm"],
        ["add", "yes", "2.0", "n-add", "keep-me"],
        ["new", "", "3.0", "n-new", ""],
    ]


def test_dropdown_targets_looks_right_after_reorder() -> None:
    """Dropdown plumbing resolves by header name, so it lands post-reorder."""
    p = PRESETS["cost-model"]
    headers = list(_REVIEW_LAYOUT)  # what upload_data returns after reorder
    reqs = _dropdown_requests(p, headers, num_rows=10, sheet_id=3)
    assert len(reqs) == 1
    rng = reqs[0]["setDataValidation"]["range"]
    assert rng["startColumnIndex"] == headers.index("looks-right?") == 4
    assert rng["endColumnIndex"] == 5
    assert rng["sheetId"] == 3


def _walk_sheet_ids(obj: object) -> list[object]:
    """Collect every value stored under a "sheetId" key, recursively."""
    found: list[object] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "sheetId":
                found.append(value)
            else:
                found.extend(_walk_sheet_ids(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_walk_sheet_ids(item))
    return found


def test_cost_model_format_hook_resolves_columns_by_name() -> None:
    """The hook targets columns wherever they are — never by fixed index."""
    headers = list(_REVIEW_LAYOUT)
    random.Random(20260716).shuffle(headers)  # deterministic scramble
    assert headers != list(_REVIEW_LAYOUT)
    num_rows, sheet_id = 101, 777
    requests = _cost_model_format_requests(headers, num_rows, len(headers), sheet_id)

    # Every request range carries the threaded data-tab sheetId.
    for req in requests:
        ids = _walk_sheet_ids(req)
        assert ids, f"request without a sheetId: {req}"
        assert set(ids) == {sheet_id}

    # Group fills cover exactly each group's columns, wherever they landed:
    # pastel body fill on data rows, saturated header fill on row 0.
    for names, body_fill, header_fill in _COST_MODEL_GROUPS:
        expected = {headers.index(n) for n in names}
        body_cols: set[int] = set()
        header_cols: set[int] = set()
        for req in requests:
            cell = (req.get("repeatCell") or {}).get("cell", {})
            bg = cell.get("userEnteredFormat", {}).get("backgroundColor")
            if bg is None:
                continue
            rng = req["repeatCell"]["range"]
            cols = set(range(rng["startColumnIndex"], rng["endColumnIndex"]))
            if bg == body_fill and rng["startRowIndex"] == 1:
                body_cols |= cols
            elif bg == header_fill and rng["startRowIndex"] == 0:
                header_cols |= cols
        assert body_cols == expected, f"body fill misses for group {names}"
        assert header_cols == expected, f"header fill misses for group {names}"

    # Conditional formats resolved by name.
    def rules_for(text: str) -> list[dict]:
        out: list[dict] = []
        for req in requests:
            rule = (req.get("addConditionalFormatRule") or {}).get("rule")
            if not rule:
                continue
            values = rule["booleanRule"]["condition"]["values"]
            if [v["userEnteredValue"] for v in values] == [text]:
                out.append(rule)
        return out

    verdict_col = headers.index("looks-right?")
    for verdict in ("yes", "no", "unsure"):
        rules = rules_for(verdict)
        assert len(rules) == 1
        assert rules[0]["ranges"][0]["startColumnIndex"] == verdict_col

    status_col = headers.index("status")
    for status in ("free", "blacklisted"):
        rules = rules_for(status)
        assert len(rules) == 1
        assert rules[0]["ranges"][0]["startColumnIndex"] == status_col

    raises_cols = {r["ranges"][0]["startColumnIndex"] for r in rules_for("raises")}
    assert raises_cols == {headers.index(n) for n in _COST_MODEL_BILLED_COLUMNS}

    # Column widths follow their headers (spec-pinned sizes).
    widths: dict[int, int] = {}
    for req in requests:
        dim = req.get("updateDimensionProperties")
        if not dim:
            continue
        assert dim["range"]["dimension"] == "COLUMNS"
        widths[dim["range"]["startIndex"]] = dim["properties"]["pixelSize"]
    assert widths[headers.index("op")] == 200
    assert widths[headers.index("flop_cost_formula")] == 420
    assert widths[headers.index("notes")] == 320
    assert widths[headers.index("example_input")] == 220
    assert widths[headers.index("looks-right?")] == 160
    assert widths[headers.index("proposed-change")] == 220
    assert widths[headers.index("reviewer-notes")] == 220

    # Exactly one basic filter spanning the whole data range.
    filters = [req["setBasicFilter"] for req in requests if "setBasicFilter" in req]
    assert len(filters) == 1
    assert filters[0]["filter"]["range"] == {
        "sheetId": sheet_id,
        "startRowIndex": 0,
        "endRowIndex": num_rows,
        "startColumnIndex": 0,
        "endColumnIndex": len(headers),
    }

    # The hook must not re-freeze rows/columns (apply_formatting owns that).
    assert not any("updateSheetProperties" in req for req in requests)


def _canonical_upload_rows(ops: list[str]) -> list[list[str]]:
    """Build upload rows for the real cost-model preset, as ``main()`` loads them.

    Data cells are ``f"{op}/{header}"`` so tests can tell fresh CSV data
    from stale live-sheet values at a glance; the shipped-empty annotation
    columns are appended and the preset's column_order applied, mirroring
    the load path in ``main()``.
    """
    p = PRESETS["cost-model"]
    ship_empty = set(p.ship_empty_columns)
    csv_header = [h for h in _REVIEW_LAYOUT if h not in ship_empty]
    rows = [csv_header] + [
        [op if h == "op" else f"{op}/{h}" for h in csv_header] for op in ops
    ]
    rows = _append_ship_empty_columns(rows, p.ship_empty_columns)
    return _reorder_columns(rows, p.column_order)


def test_reupload_restores_canonical_column_lost_to_drag_copy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Incident replay: `category` drag-copied over `looks-right?` live.

    The live sheet holds 23 cells with `category` twice and NO
    `looks-right?`. The re-upload must anchor on the canonical layout: all
    23 canonical columns in canonical order, `looks-right?` restored
    (empty, dropdown-targetable again), `category` carrying CSV data
    exactly once, and the reviewer's annotations in the other two
    annotation columns preserved by op key.
    """
    p = PRESETS["cost-model"]
    canonical = list(_REVIEW_LAYOUT)
    damaged = list(canonical)
    damaged[canonical.index("looks-right?")] = "category"

    def live_row(op: str, proposed: str, note: str) -> list[str]:
        vals = {h: f"stale-{op}/{h}" for h in canonical}
        vals.update({"op": op, "proposed-change": proposed, "reviewer-notes": note})
        return [vals[h] for h in damaged]  # both `category` cells alike

    sheet = [
        damaged,
        live_row("add", "flip weight", "seen it"),
        live_row("abs", "", "check complex"),
    ]
    fake = _FakeGws(sheet)
    monkeypatch.setattr(upload_mod, "gws", fake)

    headers = upload_data("sheet-id", _canonical_upload_rows(["abs", "add", "new"]), p)

    assert headers == canonical
    assert fake.uploaded[0] == canonical
    by_op = {r[0]: dict(zip(canonical, r, strict=True)) for r in fake.uploaded[1:]}
    assert set(by_op) == {"abs", "add", "new"}
    for op in ("abs", "add", "new"):
        # The lost annotation column is back, shipped empty...
        assert by_op[op]["looks-right?"] == ""
        # ...and `category` (like every data column) is CSV data, once.
        assert by_op[op]["category"] == f"{op}/category"
        assert by_op[op]["weight"] == f"{op}/weight"
    # Annotations in the surviving reviewer columns realigned by key.
    assert by_op["add"]["proposed-change"] == "flip weight"
    assert by_op["add"]["reviewer-notes"] == "seen it"
    assert by_op["abs"]["reviewer-notes"] == "check complex"
    assert by_op["new"]["proposed-change"] == ""
    assert "duplicate canonical column 'category'" in capsys.readouterr().out
    # The restored column is a dropdown target again — the incident's
    # "dropdown column not on sheet; skipping" path is unreachable now.
    reqs = _dropdown_requests(p, headers, num_rows=4, sheet_id=0)
    assert len(reqs) == 1
    rng = reqs[0]["setDataValidation"]["range"]
    assert rng["startColumnIndex"] == canonical.index("looks-right?") == 4
    assert "skipping" not in capsys.readouterr().out


def test_reupload_ships_annotation_columns_missing_from_live_sheet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-ergonomics live sheet (no annotation columns) gains all three."""
    p = PRESETS["cost-model"]
    canonical = list(_REVIEW_LAYOUT)
    data_only = [h for h in canonical if h not in p.preserve_columns]
    assert len(data_only) == len(canonical) - 3
    sheet = [data_only] + [
        [op if h == "op" else f"old-{op}/{h}" for h in data_only]
        for op in ("abs", "add")
    ]
    fake = _FakeGws(sheet)
    monkeypatch.setattr(upload_mod, "gws", fake)

    headers = upload_data("sheet-id", _canonical_upload_rows(["abs", "add"]), p)

    assert headers == canonical
    assert fake.uploaded[0] == canonical
    assert len(fake.uploaded) == 3
    for row in fake.uploaded[1:]:
        vals = dict(zip(canonical, row, strict=True))
        for col in ("looks-right?", "proposed-change", "reviewer-notes"):
            assert vals[col] == ""  # shipped fresh, empty
        assert vals["weight"] == f"{vals['op']}/weight"  # data refreshed from CSV


def test_duplicate_reviewer_extra_column_keeps_first_and_warns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A duplicated reviewer-owned extra column: first kept, warned, no crash."""
    p = PRESETS["cost-model"]
    canonical = list(_REVIEW_LAYOUT)
    live = canonical + ["scratch", "scratch"]

    def live_row(op: str, first: str, second: str) -> list[str]:
        vals = {h: f"stale-{op}/{h}" for h in canonical}
        vals.update({"op": op, "looks-right?": f"verdict-{op}"})
        return [vals[h] for h in canonical] + [first, second]

    sheet = [
        live,
        live_row("add", "keep-me", "shadow"),
        live_row("abs", "also-keep", "nope"),
    ]
    fake = _FakeGws(sheet)
    monkeypatch.setattr(upload_mod, "gws", fake)

    headers = upload_data("sheet-id", _canonical_upload_rows(["abs", "add"]), p)

    # Canonical block leads in canonical order; the reviewer's extra column
    # survives exactly once, appended after it.
    assert headers == canonical + ["scratch"]
    assert fake.uploaded[0] == headers
    by_op = {r[0]: dict(zip(headers, r, strict=True)) for r in fake.uploaded[1:]}
    assert by_op["add"]["scratch"] == "keep-me"  # FIRST occurrence's values win
    assert by_op["abs"]["scratch"] == "also-keep"
    # Canonical annotation values still preserved by key alongside the dupe.
    assert by_op["add"]["looks-right?"] == "verdict-add"
    assert by_op["abs"]["looks-right?"] == "verdict-abs"
    assert "duplicate reviewer column 'scratch'" in capsys.readouterr().out


def test_dropdown_warning_kept_for_unknown_configured_columns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dropdown columns that genuinely never exist post-merge still warn."""
    p = SheetPreset(
        title="t",
        key_column="op",
        dropdown_columns={"ghost": ("a", "b")},
        preserve_columns=frozenset(),
        csv_path=Path("unused.csv"),
    )
    assert _dropdown_requests(p, ["op", "weight"], num_rows=2, sheet_id=0) == []
    assert "dropdown column 'ghost' not on sheet" in capsys.readouterr().out
