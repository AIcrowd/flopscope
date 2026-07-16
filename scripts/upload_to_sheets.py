#!/usr/bin/env python3
"""Upload a flopscope review CSV to Google Sheets with formatting and dropdowns.

Schema-agnostic: each uploadable sheet is described by a ``SheetPreset``
(title, key column, dropdown columns, reviewer-owned columns, CSV path).
Re-uploads preserve reviewer-owned columns, realigned by the preset's key
column so reviewer annotations survive row insertions/removals/reordering.

Requires: gws CLI (https://github.com/googleworkspace/cli) authenticated via
`gws auth login`.

Usage::

    python scripts/upload_to_sheets.py                            # weights (default)
    python scripts/upload_to_sheets.py --preset cost-model
    python scripts/upload_to_sheets.py --preset cost-model --spreadsheet-id <id>
    python scripts/upload_to_sheets.py --csv path/to/other.csv    # override preset CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Name of the tab that holds the uploaded CSV data. Freshly created
# spreadsheets pin it to sheetId 0; on pre-existing spreadsheets its actual
# sheetId is resolved by _ensure_data_sheet and threaded through formatting.
DATA_SHEET_TITLE = "All Operations"

# Reviewer-owned columns of the weights sheet. "Reviewer Weight" ships (empty)
# inside weights.csv; the other three exist only on the live sheet.
_WEIGHTS_PRESERVE = frozenset(
    {
        "Reviewer Weight",
        "Reviewer Notes",
        "Review Status",
        "Post Review Action",
    }
)

# Reviewer-annotation columns of the cost-model sheet. The generated CSV does
# not ship them; the uploader appends them as empty columns on upload.
_COST_MODEL_ANNOTATIONS = ("looks-right?", "proposed-change", "reviewer-notes")


@dataclass(frozen=True)
class SheetPreset:
    """Declarative description of one uploadable review sheet.

    Attributes:
        title: Spreadsheet title used when creating a new spreadsheet.
        key_column: Header of the column that uniquely identifies a row
            (e.g. "Operation" / "op"). Reviewer data is realigned by this
            key across re-uploads.
        dropdown_columns: Mapping of column header -> allowed values; each
            gets a ONE_OF_LIST data-validation dropdown, resolved by header
            name on the uploaded sheet.
        preserve_columns: Headers that belong to reviewers even if they
            appear in our CSV — never overwritten by CSV data on re-upload
            (the sheet's values are the source of truth).
        csv_path: Default CSV to upload (overridable via --csv).
        ship_empty_columns: Columns to append (empty) to the CSV at load
            time. Used when the CSV generator does not ship the reviewer
            columns itself; combined with preserve_columns this makes the
            sheet ship ready-to-annotate while keeping annotations safe.
        format_hook: Optional callable (headers, num_rows, num_cols,
            sheet_id) -> batchUpdate requests for preset-specific formatting
            (colors, widths, conditional rules), targeting the data tab
            identified by sheet_id.
        summary_builder: Optional callable (csv rows) -> summary rows. When
            set, a "Review Summary" tab is created and populated on initial
            spreadsheet creation.
    """

    title: str
    key_column: str
    dropdown_columns: dict[str, tuple[str, ...]]
    preserve_columns: frozenset[str]
    csv_path: Path
    ship_empty_columns: tuple[str, ...] = ()
    format_hook: Callable[[list[str], int, int, int], list[dict]] | None = None
    summary_builder: Callable[[list[list[str]]], list[list[str]]] | None = None


def gws(*args: str, json_body: dict | None = None) -> dict:
    """Run a gws CLI command and return parsed JSON output.

    For large JSON bodies, writes to a temp file to avoid CLI arg length limits.
    """
    cmd = ["gws"] + list(args)
    tmp_file = None
    if json_body is not None:
        body_str = json.dumps(json_body)
        # If body is large, write to temp file and use @file syntax
        if len(body_str) > 50_000:
            tmp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            tmp_file.write(body_str)
            tmp_file.close()
            cmd += ["--json", f"@{tmp_file.name}"]
        else:
            cmd += ["--json", body_str]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        if tmp_file:
            Path(tmp_file.name).unlink(missing_ok=True)

    # Try stdout first, then stderr
    for output in [result.stdout, result.stderr]:
        idx = output.find("{")
        if idx == -1:
            continue
        # Find the first complete JSON object
        depth = 0
        for i, ch in enumerate(output[idx:], idx):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(output[idx : i + 1])
                        if "error" in parsed:
                            print(
                                f"  gws API error: {parsed['error'].get('message', '')[:200]}",
                                file=sys.stderr,
                            )
                        return parsed
                    except json.JSONDecodeError:
                        break

    if result.returncode != 0:
        print(f"gws error (exit {result.returncode}):", file=sys.stderr)
        print(f"  cmd: {' '.join(cmd[:6])}...", file=sys.stderr)
        print(f"  stderr: {result.stderr[:300]}", file=sys.stderr)
        sys.exit(1)
    return {}


def load_csv(path: Path) -> list[list[str]]:
    """Load CSV as a list of rows (each row is a list of strings)."""
    with open(path) as f:
        reader = csv.reader(f)
        return list(reader)


def _append_ship_empty_columns(
    rows: list[list[str]], columns: Sequence[str]
) -> list[list[str]]:
    """Append reviewer-annotation columns that the CSV does not ship.

    The weights CSV ships its reviewer-owned column ("Reviewer Weight") as an
    empty column inside the CSV itself. Generated CSVs (cost-model) stay free
    of review columns, so the uploader appends them here — header names plus
    an empty cell on every data row — before upload. Downstream,
    ``preserve_columns`` marks them reviewer-owned so re-uploads never
    overwrite what reviewers typed. Idempotent: names already present in the
    header are skipped; with no missing names the rows are returned unchanged.
    """
    if not rows:
        return rows
    header = rows[0]
    missing = [c for c in columns if c not in header]
    if not missing:
        return rows
    new_header = header + missing
    width = len(new_header)
    return [new_header] + [row + [""] * (width - len(row)) for row in rows[1:]]


def create_spreadsheet(preset: SheetPreset) -> str:
    """Create a new Google Sheets spreadsheet and return its ID."""
    print(f"Creating spreadsheet: {preset.title}")
    sheets: list[dict] = [{"properties": {"title": DATA_SHEET_TITLE, "sheetId": 0}}]
    if preset.summary_builder is not None:
        sheets.append({"properties": {"title": "Review Summary", "sheetId": 1}})
    resp = gws(
        "sheets",
        "spreadsheets",
        "create",
        json_body={
            "properties": {"title": preset.title},
            "sheets": sheets,
        },
    )
    sid = resp.get("spreadsheetId", "")
    if not sid:
        print("ERROR: Could not create spreadsheet", file=sys.stderr)
        sys.exit(1)
    print(f"  Spreadsheet ID: {sid}")
    print(f"  URL: https://docs.google.com/spreadsheets/d/{sid}")
    return sid


def _ensure_data_sheet(sid: str) -> int:
    """Make sure the data tab exists; return its sheetId.

    A user-created spreadsheet starts with a single default tab (``Sheet1``),
    while every read/write here addresses ``DATA_SHEET_TITLE``. If the data
    tab is absent: rename the lone existing tab (the common "fresh sheet
    shared for review" case), otherwise add a new tab. The returned sheetId
    (existing tab's id / rename target's id / the ``addSheet`` reply's new
    id) is what formatting must target — on a multi-tab spreadsheet the data
    tab is generally NOT sheetId 0.
    """
    meta = gws(
        "sheets",
        "spreadsheets",
        "get",
        "--params",
        json.dumps({"spreadsheetId": sid, "fields": "sheets.properties"}),
    )
    sheets = [s.get("properties", {}) for s in meta.get("sheets", [])]
    for props in sheets:
        if props.get("title") == DATA_SHEET_TITLE:
            return int(props.get("sheetId", 0))
    data_sheet_id: int | None
    if len(sheets) == 1:
        data_sheet_id = int(sheets[0].get("sheetId", 0))
        print(f"  Renaming tab {sheets[0].get('title')!r} -> {DATA_SHEET_TITLE!r}")
        req = {
            "updateSheetProperties": {
                "properties": {
                    "sheetId": data_sheet_id,
                    "title": DATA_SHEET_TITLE,
                },
                "fields": "title",
            }
        }
    else:
        data_sheet_id = None
        print(f"  Adding tab {DATA_SHEET_TITLE!r}")
        req = {"addSheet": {"properties": {"title": DATA_SHEET_TITLE}}}
    resp = gws(
        "sheets",
        "spreadsheets",
        "batchUpdate",
        "--params",
        json.dumps({"spreadsheetId": sid}),
        json_body={"requests": [req]},
    )
    if data_sheet_id is None:
        replies = resp.get("replies") or [{}]
        added = replies[0].get("addSheet", {}).get("properties", {})
        data_sheet_id = added.get("sheetId")
        if data_sheet_id is None:
            print(
                "ERROR: addSheet reply did not include the new tab's sheetId",
                file=sys.stderr,
            )
            sys.exit(1)
    return int(data_sheet_id)


def _read_sheet_all(
    sid: str, sheet_name: str = DATA_SHEET_TITLE
) -> tuple[list[str], list[list[str]]]:
    """Read all data from a sheet. Returns (headers, data_rows)."""
    resp = gws(
        "sheets",
        "spreadsheets",
        "values",
        "get",
        "--params",
        json.dumps(
            {
                "spreadsheetId": sid,
                "range": f"'{sheet_name}'!A1:ZZ",
            }
        ),
    )
    all_rows = resp.get("values", [])
    if not all_rows:
        return [], []
    return all_rows[0], all_rows[1:]


def _find_reviewer_columns(
    sheet_headers: list[str],
    csv_headers: list[str],
    preserve: frozenset[str],
) -> list[int]:
    """Return sheet column indices that should be preserved (reviewer-owned).

    A column is reviewer-owned if:
    - Its header is NOT in our CSV headers, OR
    - Its header is in ``preserve`` (e.g. "Reviewer Weight" which is in the
      weights CSV but always empty — the reviewer fills it in on the sheet).
    """
    csv_set = set(csv_headers) - preserve
    return [i for i, h in enumerate(sheet_headers) if h not in csv_set or h in preserve]


def upload_data(sid: str, rows: list[list[str]], preset: SheetPreset) -> list[str]:
    """Upload CSV data to the data sheet, preserving reviewer-added columns.

    Algorithm:
    1. Read the sheet's current state (headers + all data).
    2. Identify reviewer-owned columns (headers not in our CSV, plus the
       preset's ``preserve_columns`` even when CSV-shipped).
    3. Build a map: key -> {reviewer_col_header: value, ...} keyed by the
       preset's key column so alignment is by key, not row position.
    4. Clear the sheet and write our CSV data (all columns including any
       shipped-empty reviewer placeholders).
    5. Write reviewer data back, aligned by key to match the new row order.

    Returns the header row actually written to the sheet.
    """
    csv_headers = rows[0]
    csv_data = rows[1:]
    print(f"Uploading {len(csv_data)} data rows ({len(csv_headers)} CSV columns)...")

    # --- Step 1: Read current sheet state ---
    sheet_headers, sheet_data = _read_sheet_all(sid)

    if not sheet_headers:
        print("  Fresh sheet, uploading all columns...")
        _upload_all_rows(sid, rows)
        return list(csv_headers)

    # --- Step 2: Identify reviewer columns ---
    reviewer_col_indices = _find_reviewer_columns(
        sheet_headers, csv_headers, preset.preserve_columns
    )
    reviewer_col_names = [sheet_headers[i] for i in reviewer_col_indices]
    reviewer_col_set = set(reviewer_col_names)

    if reviewer_col_names:
        print(f"  Found reviewer columns: {reviewer_col_names}")
    else:
        print("  No reviewer columns found.")

    # --- Step 3: Build key -> reviewer data map ---
    # Find the key column on the sheet (falls back to the first column).
    try:
        sheet_key_idx = sheet_headers.index(preset.key_column)
    except ValueError:
        sheet_key_idx = 0

    reviewer_data: dict[str, dict[str, str]] = {}
    for row in sheet_data:
        if not row or len(row) <= sheet_key_idx:
            continue
        key = row[sheet_key_idx]
        if not key:
            continue
        reviewer_data[key] = {}
        for col_idx in reviewer_col_indices:
            col_name = sheet_headers[col_idx]
            value = row[col_idx] if col_idx < len(row) else ""
            reviewer_data[key][col_name] = value

    non_empty = sum(
        1 for key_vals in reviewer_data.values() for v in key_vals.values() if v.strip()
    )
    print(
        f"  Captured {non_empty} non-empty reviewer values across "
        f"{len(reviewer_data)} rows."
    )

    # --- Step 4: Clear sheet and write CSV data ---
    # Clear the entire data range first
    gws(
        "sheets",
        "spreadsheets",
        "values",
        "clear",
        "--params",
        json.dumps(
            {
                "spreadsheetId": sid,
                "range": f"'{DATA_SHEET_TITLE}'!A1:ZZ",
            }
        ),
    )

    # Build output column order: preserve the sheet's original column layout.
    # For each sheet column, either pull from CSV (by header name) or from
    # the reviewer data (by key). This keeps reviewer columns in their
    # original positions (e.g. F and G stay as F and G).
    csv_header_to_idx = {h: i for i, h in enumerate(csv_headers)}

    # Determine output columns: sheet's existing order, but skip the CSV's
    # shipped-empty reviewer placeholders since the sheet has the real ones.
    out_col_sources: list[tuple[str, str]] = []  # (header, source: "csv"|"reviewer")
    for sheet_col_name in sheet_headers:
        if sheet_col_name in preset.preserve_columns:
            out_col_sources.append((sheet_col_name, "reviewer"))
        elif (
            sheet_col_name in csv_header_to_idx
            and sheet_col_name not in reviewer_col_set
        ):
            out_col_sources.append((sheet_col_name, "csv"))
        else:
            out_col_sources.append((sheet_col_name, "reviewer"))

    # Add any CSV columns not already on the sheet (new columns)
    sheet_header_set = set(sheet_headers)
    for csv_h in csv_headers:
        if csv_h not in sheet_header_set and csv_h not in preset.preserve_columns:
            out_col_sources.append((csv_h, "csv"))

    # CSV key column index (falls back to the first column).
    try:
        csv_key_idx = csv_headers.index(preset.key_column)
    except ValueError:
        csv_key_idx = 0

    # Build rows
    out_headers = [src[0] for src in out_col_sources]
    out_data = []
    for csv_row in csv_data:
        key = csv_row[csv_key_idx] if len(csv_row) > csv_key_idx else ""
        reviewer_vals = reviewer_data.get(key, {})
        row = []
        for col_name, source in out_col_sources:
            if source == "csv":
                idx = csv_header_to_idx.get(col_name)
                row.append(
                    csv_row[idx] if idx is not None and idx < len(csv_row) else ""
                )
            else:  # reviewer
                row.append(reviewer_vals.get(col_name, ""))
        out_data.append(row)

    # --- Step 5: Apply reviewer weights to Active Weight locally ---
    # Weights-preset enrichment: where the reviewer provided a numeric weight,
    # use it as Active Weight. This avoids per-cell API writes after upload.
    # No-op for presets without both columns (e.g. cost-model).
    active_idx = (
        out_headers.index("Active Weight") if "Active Weight" in out_headers else -1
    )
    reviewer_idx = (
        out_headers.index("Reviewer Weight") if "Reviewer Weight" in out_headers else -1
    )
    if active_idx >= 0 and reviewer_idx >= 0:
        applied = 0
        for row in out_data:
            rw = row[reviewer_idx].strip() if row[reviewer_idx] else ""
            if rw and rw != "?":
                try:
                    float(rw)
                    row[active_idx] = rw
                    applied += 1
                except ValueError:
                    pass
        print(f"  Applied {applied} reviewer weights to Active Weight (locally).")

    all_out = [out_headers] + out_data
    _upload_all_rows(sid, all_out)

    print(
        f"  Uploaded {len(out_data)} rows: {len(out_col_sources)} columns "
        f"({sum(1 for _, s in out_col_sources if s == 'csv')} CSV, "
        f"{sum(1 for _, s in out_col_sources if s == 'reviewer')} reviewer), "
        f"aligned by {preset.key_column!r}."
    )
    return out_headers


def _upload_all_rows(sid: str, rows: list[list[str]]) -> None:
    """Upload rows to the sheet in chunks."""
    CHUNK = 50
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start : start + CHUNK]
        row_start = start + 1
        gws(
            "sheets",
            "spreadsheets",
            "values",
            "update",
            "--params",
            json.dumps(
                {
                    "spreadsheetId": sid,
                    "range": f"'{DATA_SHEET_TITLE}'!A{row_start}",
                    "valueInputOption": "USER_ENTERED",
                }
            ),
            json_body={"values": chunk},
        )


def _color(r: float, g: float, b: float) -> dict:
    """Create a color dict for the Sheets API (0-1 scale)."""
    return {"red": r, "green": g, "blue": b}


def _cond_rule(
    sheet_id: int, col: int, num_rows: int, condition_type: str, values: list, fmt: dict
) -> dict:
    """Build a conditional format rule for a column."""
    rule = {
        "addConditionalFormatRule": {
            "rule": {
                "ranges": [
                    {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": num_rows,
                        "startColumnIndex": col,
                        "endColumnIndex": col + 1,
                    }
                ],
                "booleanRule": {
                    "condition": {
                        "type": condition_type,
                        "values": values,
                    },
                    "format": fmt,
                },
            },
            "index": 0,
        }
    }
    return rule


def _text_eq_rule(
    sheet_id: int, col: int, num_rows: int, text: str, bg: dict, fg: dict | None = None
) -> dict:
    """Conditional format: cell text equals a specific value."""
    fmt = {"backgroundColor": bg}
    if fg:
        fmt["textFormat"] = {"foregroundColor": fg}
    return _cond_rule(
        sheet_id,
        col,
        num_rows,
        "TEXT_EQ",
        [{"userEnteredValue": text}],
        fmt,
    )


def _number_between_rule(
    sheet_id: int, col: int, num_rows: int, lo: str, hi: str, bg: dict
) -> dict:
    """Conditional format: number between lo and hi."""
    return _cond_rule(
        sheet_id,
        col,
        num_rows,
        "NUMBER_BETWEEN",
        [{"userEnteredValue": lo}, {"userEnteredValue": hi}],
        {"backgroundColor": bg},
    )


def _number_rule(
    sheet_id: int,
    col: int,
    num_rows: int,
    cond_type: str,
    value: str,
    bg: dict,
    fg: dict | None = None,
) -> dict:
    """Conditional format: number comparison."""
    fmt = {"backgroundColor": bg}
    if fg:
        fmt["textFormat"] = {"foregroundColor": fg}
    return _cond_rule(
        sheet_id,
        col,
        num_rows,
        cond_type,
        [{"userEnteredValue": value}],
        fmt,
    )


def _gradient_rule(
    sheet_id: int,
    col: int,
    num_rows: int,
    min_color: dict,
    mid_color: dict,
    max_color: dict,
) -> dict:
    """Color scale (gradient) conditional format for a column.

    Automatically adapts to the min/max values in the column — no hardcoded
    ranges. Uses a 3-point scale: min → mid → max.
    """
    return {
        "addConditionalFormatRule": {
            "rule": {
                "ranges": [
                    {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": num_rows,
                        "startColumnIndex": col,
                        "endColumnIndex": col + 1,
                    }
                ],
                "gradientRule": {
                    "minpoint": {
                        "color": min_color,
                        "type": "MIN",
                    },
                    "midpoint": {
                        "color": mid_color,
                        "type": "PERCENTILE",
                        "value": "50",
                    },
                    "maxpoint": {
                        "color": max_color,
                        "type": "MAX",
                    },
                },
            },
            "index": 0,
        }
    }


def _dropdown_requests(
    preset: SheetPreset, headers: list[str], num_rows: int, sheet_id: int
) -> list[dict]:
    """ONE_OF_LIST data-validation dropdowns, resolved by header name."""
    requests: list[dict] = []
    for col_name, options in preset.dropdown_columns.items():
        if col_name not in headers:
            print(f"  Warning: dropdown column {col_name!r} not on sheet; skipping.")
            continue
        col = headers.index(col_name)
        requests.append(
            {
                "setDataValidation": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": num_rows,
                        "startColumnIndex": col,
                        "endColumnIndex": col + 1,
                    },
                    "rule": {
                        "condition": {
                            "type": "ONE_OF_LIST",
                            "values": [{"userEnteredValue": o} for o in options],
                        },
                        "showCustomUi": True,
                        "strict": False,
                    },
                }
            }
        )
    return requests


def _weights_format_requests(
    headers: list[str], num_rows: int, num_cols: int, sheet_id: int
) -> list[dict]:
    """Weights-sheet formatting (unchanged legacy behavior).

    Column indices here are positional on purpose: they target the LIVE
    weights sheet layout (which has reviewer-inserted columns H-J), exactly
    as the pre-preset version of this script did. ``headers`` is unused.
    """
    requests: list[dict] = []

    # ---- Header row formatting ----
    # Section A (cols 0-9, A-J: review columns): dark blue-gray bg, white text, bold
    requests.append(
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": 10,
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": _color(0.2, 0.3, 0.4),
                        "textFormat": {
                            "foregroundColor": _color(1, 1, 1),
                            "bold": True,
                            "fontSize": 10,
                        },
                    }
                },
                "fields": "userEnteredFormat.backgroundColor,userEnteredFormat.textFormat",
            }
        }
    )
    # Section B (cols 10+, K onward: evidence columns): lighter gray bg
    requests.append(
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": 10,
                    "endColumnIndex": num_cols,
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": _color(0.6, 0.6, 0.65),
                        "textFormat": {
                            "foregroundColor": _color(1, 1, 1),
                            "bold": True,
                            "fontSize": 10,
                        },
                    }
                },
                "fields": "userEnteredFormat.backgroundColor,userEnteredFormat.textFormat",
            }
        }
    )

    # ---- Reviewer Weight column (G=6): light yellow bg ----
    requests.append(
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": num_rows,
                    "startColumnIndex": 6,
                    "endColumnIndex": 7,
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": _color(1.0, 0.98, 0.8),
                    }
                },
                "fields": "userEnteredFormat.backgroundColor",
            }
        }
    )

    # ---- Conditional formatting: Status column (B, index 1) ----
    WHITE = _color(1, 1, 1)
    status_rules = [
        ("benchmarked", _color(0.85, 0.93, 0.83), None),
        ("alias", _color(0.82, 0.88, 0.95), None),
        ("excluded", _color(0.9, 0.9, 0.9), None),
        ("free", _color(0.96, 0.96, 0.96), None),
        ("blacklisted", _color(0.87, 0.36, 0.34), WHITE),
        ("blacklisted-by-reviewer", _color(0.6, 0.15, 0.15), WHITE),
        ("keep", _color(0.2, 0.55, 0.24), WHITE),
    ]
    for text, bg, fg in status_rules:
        requests.append(_text_eq_rule(sheet_id, 1, num_rows, text, bg, fg))

    # ---- Conditional formatting: Weight columns (E=4, F=5, G=6) ----
    # Use gradient (color scale) so the coloring adapts to each column's
    # actual value range — no hardcoded thresholds.
    # Green (low weight = cheap) → Yellow (mid) → Red (high weight = expensive)
    WEIGHT_GREEN = _color(0.72, 0.88, 0.72)  # cheap ops
    WEIGHT_YELLOW = _color(1.0, 0.95, 0.6)  # mid-range
    WEIGHT_RED = _color(0.92, 0.45, 0.4)  # expensive ops
    for col_idx in (4, 5, 6):  # E, F, G
        requests.append(
            _gradient_rule(
                sheet_id,
                col_idx,
                num_rows,
                min_color=WEIGHT_GREEN,
                mid_color=WEIGHT_YELLOW,
                max_color=WEIGHT_RED,
            )
        )

    # ---- "?" markers in Reviewer Weight (G=6) ----
    requests.append(
        _text_eq_rule(
            sheet_id,
            6,
            num_rows,
            "?",
            bg=_color(0.85, 0.75, 0.95),  # light purple
        )
    )

    # ---- Review Status (I=8) ----
    review_status_rules = [
        ("accepted", _color(0.72, 0.88, 0.72), None),  # green
        ("pending", _color(1.0, 0.95, 0.6), None),  # yellow
        ("rejected", _color(0.95, 0.7, 0.65), None),  # red
        ("needs-discussion", _color(0.82, 0.88, 0.95), None),  # light blue
    ]
    for text, bg, fg in review_status_rules:
        requests.append(_text_eq_rule(sheet_id, 8, num_rows, text, bg, fg))

    # ---- Confidence (L=11) ----
    conf_rules = [
        ("high", _color(0.72, 0.88, 0.72)),
        ("medium", _color(1.0, 0.95, 0.6)),
        ("low", _color(0.95, 0.7, 0.65)),
    ]
    for text, bg in conf_rules:
        requests.append(_text_eq_rule(sheet_id, 11, num_rows, text, bg))

    # ---- Perf/Timing Agreement (Q=16) ----
    # Green: 0.5-2.0, Yellow: 0.2-0.5 or 2.0-5.0, Red: <0.2 or >5.0
    requests.append(
        _number_between_rule(
            sheet_id, 16, num_rows, "0.5", "2.0", _color(0.72, 0.88, 0.72)
        )
    )
    requests.append(
        _number_between_rule(
            sheet_id, 16, num_rows, "0.2", "0.5", _color(1.0, 0.95, 0.6)
        )
    )
    requests.append(
        _number_between_rule(
            sheet_id, 16, num_rows, "2.0", "5.0", _color(1.0, 0.95, 0.6)
        )
    )
    requests.append(
        _number_rule(
            sheet_id, 16, num_rows, "NUMBER_LESS", "0.2", _color(0.95, 0.7, 0.65)
        )
    )
    requests.append(
        _number_rule(
            sheet_id, 16, num_rows, "NUMBER_GREATER", "5.0", _color(0.95, 0.7, 0.65)
        )
    )

    # ---- Column widths ----
    # Matches sheet layout: A-Z (see column order above)
    col_widths = {
        0: 200,  # A: Operation
        1: 120,  # B: Status
        2: 140,  # C: Category
        3: 200,  # D: Cost Formula
        4: 110,  # E: Active Weight
        5: 120,  # F: Empirical Weight
        6: 120,  # G: Reviewer Weight
        7: 200,  # H: Reviewer Notes
        8: 120,  # I: Review Status
        9: 250,  # J: Post Review Action
        10: 200,  # K: Effective Cost Example
        11: 100,  # L: Confidence
        12: 400,  # M: Notes
        13: 250,  # N: Exclusion Reason
        14: 100,  # O: HW FP Instructions
        15: 100,  # P: Timing Weight
        16: 100,  # Q: Perf/Timing Agreement
        17: 100,  # R: CV
        18: 250,  # S: Benchmark Command
        19: 140,  # T: Benchmark Size
        20: 180,  # U: Total Perf Instructions
        21: 120,  # V: Total Timing
        22: 350,  # W: Implementation URL
        23: 100,  # X: Weight Tier
        24: 70,  # Y: Repeats
    }
    for col_idx, width in col_widths.items():
        if col_idx < num_cols:
            requests.append(
                {
                    "updateDimensionProperties": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "COLUMNS",
                            "startIndex": col_idx,
                            "endIndex": col_idx + 1,
                        },
                        "properties": {"pixelSize": width},
                        "fields": "pixelSize",
                    }
                }
            )

    # ---- Wrap text on Notes column ----
    requests.append(
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": num_rows,
                    "startColumnIndex": 8,
                    "endColumnIndex": 9,
                },
                "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP"}},
                "fields": "userEnteredFormat.wrapStrategy",
            }
        }
    )
    return requests


def _cost_model_format_requests(
    headers: list[str], num_rows: int, num_cols: int, sheet_id: int
) -> list[dict]:
    """Cost-model-sheet formatting; columns are resolved by header name."""
    requests: list[dict] = []

    def col_of(name: str) -> int | None:
        return headers.index(name) if name in headers else None

    # ---- Header row: single dark section across all current columns ----
    requests.append(
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": max(len(headers), num_cols),
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": _color(0.2, 0.3, 0.4),
                        "textFormat": {
                            "foregroundColor": _color(1, 1, 1),
                            "bold": True,
                            "fontSize": 10,
                        },
                    }
                },
                "fields": "userEnteredFormat.backgroundColor,userEnteredFormat.textFormat",
            }
        }
    )

    # ---- Annotation columns: light yellow "fill me in" cue ----
    for name in _COST_MODEL_ANNOTATIONS:
        col = col_of(name)
        if col is None:
            continue
        requests.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": num_rows,
                        "startColumnIndex": col,
                        "endColumnIndex": col + 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": _color(1.0, 0.98, 0.8),
                        }
                    },
                    "fields": "userEnteredFormat.backgroundColor",
                }
            }
        )

    # ---- Conditional formatting: looks-right? verdicts ----
    verdict_col = col_of("looks-right?")
    if verdict_col is not None:
        verdict_rules = [
            ("yes", _color(0.72, 0.88, 0.72)),  # green
            ("no", _color(0.95, 0.7, 0.65)),  # red
            ("unsure", _color(1.0, 0.95, 0.6)),  # yellow
        ]
        for text, bg in verdict_rules:
            requests.append(_text_eq_rule(sheet_id, verdict_col, num_rows, text, bg))

    # ---- Column widths (by header name) ----
    col_widths = {
        "op": 190,
        "module": 90,
        "status": 100,
        "category": 160,
        "flop_cost_formula": 240,
        "weight": 80,
        "complex_factor": 110,
        "dtype_rate_rule": 120,
        "example_input": 130,
        "raw_flop_cost": 110,
        "raw_flop_cost_2x": 120,
        "billed_int16": 100,
        "billed_fp32": 100,
        "billed_fp64": 100,
        "billed_complex128": 130,
        "complex_penalty": 120,
        "notes": 380,
        "numpy_range": 110,
        "registry_ref": 320,
        "cost_impl_ref": 320,
        "looks-right?": 110,
        "proposed-change": 260,
        "reviewer-notes": 320,
    }
    for name, width in col_widths.items():
        col = col_of(name)
        if col is None:
            continue
        requests.append(
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": col,
                        "endIndex": col + 1,
                    },
                    "properties": {"pixelSize": width},
                    "fields": "pixelSize",
                }
            }
        )

    # ---- Wrap text on prose columns ----
    for name in ("notes", "proposed-change", "reviewer-notes"):
        col = col_of(name)
        if col is None:
            continue
        requests.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": num_rows,
                        "startColumnIndex": col,
                        "endColumnIndex": col + 1,
                    },
                    "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP"}},
                    "fields": "userEnteredFormat.wrapStrategy",
                }
            }
        )

    return requests


def apply_formatting(
    sid: str,
    preset: SheetPreset,
    headers: list[str],
    num_rows: int,
    num_cols: int,
    sheet_id: int,
) -> None:
    """Apply dropdowns and preset formatting to the data sheet.

    ``headers`` is the header row actually written by ``upload_data`` (CSV
    columns plus shipped-empty and preserved reviewer columns); dropdown
    targets are resolved against it by name. ``num_rows``/``num_cols`` are
    the uploaded CSV dimensions, used by preset format hooks for ranges.
    ``sheet_id`` is the data tab's sheetId (0 on freshly created
    spreadsheets; whatever ``_ensure_data_sheet`` resolved on pre-existing
    ones) — every structural request targets it.
    """
    print("Applying formatting...")
    requests = []

    # ---- Clear the data tab's existing conditional formatting first ----
    # Without this, rules accumulate across uploads and stale rules
    # override the new ones (Sheets evaluates top-down, first match wins).
    # We locate the data tab in the metadata by sheetId — NOT sheets[0],
    # which on a multi-tab spreadsheet may be an unrelated user tab — then
    # read its rule count and delete them all in reverse order.
    try:
        sheet_meta = gws(
            "sheets",
            "spreadsheets",
            "get",
            "--params",
            json.dumps(
                {
                    "spreadsheetId": sid,
                    "fields": "sheets(properties.sheetId,conditionalFormats)",
                }
            ),
        )
        if isinstance(sheet_meta, str):
            sheet_meta = json.loads(sheet_meta)
        data_tab = next(
            (
                s
                for s in sheet_meta.get("sheets", [])
                if s.get("properties", {}).get("sheetId", 0) == sheet_id
            ),
            {},
        )
        existing_rules = data_tab.get("conditionalFormats", [])
        if existing_rules:
            # Delete in reverse order so indices stay valid
            for i in range(len(existing_rules) - 1, -1, -1):
                requests.append(
                    {
                        "deleteConditionalFormatRule": {
                            "sheetId": sheet_id,
                            "index": i,
                        }
                    }
                )
            print(f"  Clearing {len(existing_rules)} stale conditional format rules...")
    except Exception as e:
        print(f"  Warning: could not read existing rules: {e}")

    # ---- Freeze header row + key column ----
    requests.append(
        {
            "updateSheetProperties": {
                "properties": {
                    "sheetId": sheet_id,
                    "gridProperties": {
                        "frozenRowCount": 1,
                        "frozenColumnCount": 1,
                    },
                },
                "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount",
            }
        }
    )

    # ---- Dropdowns (from the preset, resolved by header name) ----
    requests += _dropdown_requests(preset, headers, num_rows, sheet_id)

    # ---- Preset-specific formatting (colors, widths, conditional rules) ----
    if preset.format_hook is not None:
        requests += preset.format_hook(headers, num_rows, num_cols, sheet_id)

    # ---- Send batch updates in chunks (avoid CLI arg length limits) ----
    CHUNK_SIZE = 10
    for i in range(0, len(requests), CHUNK_SIZE):
        chunk = requests[i : i + CHUNK_SIZE]
        print(f"  Sending batch {i // CHUNK_SIZE + 1} ({len(chunk)} requests)...")
        gws(
            "sheets",
            "spreadsheets",
            "batchUpdate",
            "--params",
            json.dumps({"spreadsheetId": sid}),
            json_body={"requests": chunk},
        )
    print(f"  Formatting applied ({len(requests)} requests total).")


def _weights_summary_rows(rows: list[list[str]]) -> list[list[str]]:
    """Build the weights Review Summary tab content (unchanged legacy)."""
    data_rows = rows[1:]

    # Build summary data
    from collections import Counter

    statuses = Counter(r[1] for r in data_rows)  # col B
    categories = Counter(r[2] for r in data_rows if r[2])  # col C
    tiers = Counter(
        r[19] for r in data_rows if len(r) > 19 and r[19]
    )  # col T (Weight Tier)
    confs = Counter(r[7] for r in data_rows if r[7])  # col H

    summary = [
        ["flopscope FLOP Weight Calibration — Review Summary", ""],
        ["", ""],
        ["Status", "Count"],
    ]
    for status in ["benchmarked", "alias", "excluded", "free", "blacklisted"]:
        summary.append([status, str(statuses.get(status, 0))])
    summary.append(["", ""])
    summary.append(["Category", "Count"])
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        summary.append([cat, str(count)])
    summary.append(["", ""])
    summary.append(["Weight Tier", "Count"])
    for tier in ["negligible", "baseline", "moderate", "heavy", "extreme"]:
        summary.append([tier, str(tiers.get(tier, 0))])
    summary.append(["", ""])
    summary.append(["Confidence", "Count"])
    for conf in ["high", "medium", "low"]:
        summary.append([conf, str(confs.get(conf, 0))])
    summary.append(["", ""])
    summary.append(["Total operations", str(len(data_rows))])
    summary.append(["", ""])
    summary.append(["Instructions:", ""])
    summary.append(["1. Review the 'Weight' column (E) in 'All Operations'", ""])
    summary.append(
        ["2. Change Status dropdown to 'keep' or 'blacklisted-by-reviewer'", ""]
    )
    summary.append(["3. Enter your preferred weight in 'Reviewer Weight' (F)", ""])
    summary.append(
        ["4. Weight = 1.0 means same cost as np.add per analytical FLOP", ""]
    )
    summary.append(
        ["5. Weight < 1.0 means cheaper than np.add per analytical FLOP", ""]
    )
    summary.append(["6. Weight > 1.0 means more expensive (e.g., sin=18.39)", ""])
    return summary


def create_summary_sheet(sid: str, rows: list[list[str]], preset: SheetPreset) -> None:
    """Populate the Review Summary sheet, if the preset defines one."""
    if preset.summary_builder is None:
        return
    print("Creating summary sheet...")
    summary = preset.summary_builder(rows)

    gws(
        "sheets",
        "spreadsheets",
        "values",
        "update",
        "--params",
        json.dumps(
            {
                "spreadsheetId": sid,
                "range": "'Review Summary'!A1",
                "valueInputOption": "RAW",
            }
        ),
        json_body={"values": summary},
    )
    print("  Summary sheet created.")


PRESETS: dict[str, SheetPreset] = {
    "weights": SheetPreset(
        title="flopscope FLOP Weight Calibration Review",
        key_column="Operation",
        dropdown_columns={
            "Status": (
                "benchmarked",
                "alias",
                "excluded",
                "free",
                "blacklisted",
                "blacklisted-by-reviewer",
                "keep",
            ),
        },
        preserve_columns=_WEIGHTS_PRESERVE,
        csv_path=REPO_ROOT / "src" / "flopscope" / "data" / "weights.csv",
        format_hook=_weights_format_requests,
        summary_builder=_weights_summary_rows,
    ),
    "cost-model": SheetPreset(
        title="flopscope Cost Model Review",
        key_column="op",
        dropdown_columns={
            "looks-right?": ("yes", "no", "unsure"),
        },
        preserve_columns=frozenset(_COST_MODEL_ANNOTATIONS),
        csv_path=REPO_ROOT / "docs" / "reference" / "cost-model-sheet.csv",
        ship_empty_columns=_COST_MODEL_ANNOTATIONS,
        format_hook=_cost_model_format_requests,
    ),
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Upload a flopscope review CSV to Google Sheets"
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="weights",
        help="Which sheet preset to upload (default: weights)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Override the preset's CSV path",
    )
    parser.add_argument(
        "--spreadsheet-id",
        type=str,
        default=None,
        help="Update an existing spreadsheet instead of creating a new one. "
        "Reviewer-added columns are preserved and realigned by the preset's "
        "key column.",
    )
    args = parser.parse_args(argv)

    preset = PRESETS[args.preset]
    csv_path: Path = args.csv if args.csv is not None else preset.csv_path

    rows = load_csv(csv_path)
    rows = _append_ship_empty_columns(rows, preset.ship_empty_columns)
    print(f"Loaded {len(rows)} rows, {len(rows[0])} columns from {csv_path}")

    if args.spreadsheet_id:
        sid = args.spreadsheet_id
        print(f"Updating existing spreadsheet: {sid}")
        data_sheet_id = _ensure_data_sheet(sid)
        headers = upload_data(sid, rows, preset)
        apply_formatting(
            sid,
            preset,
            headers,
            num_rows=len(rows),
            num_cols=len(rows[0]),
            sheet_id=data_sheet_id,
        )
    else:
        sid = create_spreadsheet(preset)
        headers = upload_data(sid, rows, preset)
        apply_formatting(
            sid,
            preset,
            headers,
            num_rows=len(rows),
            num_cols=len(rows[0]),
            # create_spreadsheet pins the data tab to sheetId 0.
            sheet_id=0,
        )
        create_summary_sheet(sid, rows, preset)

    url = f"https://docs.google.com/spreadsheets/d/{sid}"
    print(f"\nDone! Spreadsheet URL:\n  {url}")


if __name__ == "__main__":
    main()
