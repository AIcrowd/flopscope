"""Config-level tests for the generalized Sheets uploader.

These tests assert preset configuration and pure row/column logic only.
No ``gws`` CLI invocation (and therefore no network) ever happens here:
importing ``scripts.upload_to_sheets`` must be side-effect free.
"""

from __future__ import annotations

from scripts.upload_to_sheets import (
    PRESETS,
    _append_ship_empty_columns,
    _find_reviewer_columns,
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
    csv = ["op", "weight", "looks-right?"]
    # "looks-right?" is CSV-shipped but reviewer-owned (preserve set);
    # "reviewer-extra" is a column the reviewer added on the sheet.
    idx = _find_reviewer_columns(sheet, csv, frozenset({"looks-right?"}))
    assert idx == [2, 3]
    # Without the preserve marker, the CSV-shipped column would be overwritten.
    idx_no_preserve = _find_reviewer_columns(sheet, csv, frozenset())
    assert idx_no_preserve == [3]
