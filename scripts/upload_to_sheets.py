#!/usr/bin/env python3
"""Upload weights.csv to Google Sheets with formatting, dropdowns, and color coding.

Requires: gws CLI (https://github.com/googleworkspace/cli) authenticated via `gws auth login`.

Usage::

    python scripts/upload_to_sheets.py
    python scripts/upload_to_sheets.py --csv src/flopscope/data/weights.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "src" / "flopscope" / "data" / "weights.csv"
TITLE = "flopscope FLOP Weight Calibration Review"


import tempfile


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


def create_spreadsheet() -> str:
    """Create a new Google Sheets spreadsheet and return its ID."""
    print(f"Creating spreadsheet: {TITLE}")
    resp = gws(
        "sheets",
        "spreadsheets",
        "create",
        json_body={
            "properties": {"title": TITLE},
            "sheets": [
                {"properties": {"title": "All Operations", "sheetId": 0}},
                {"properties": {"title": "Review Summary", "sheetId": 1}},
            ],
        },
    )
    sid = resp.get("spreadsheetId", "")
    if not sid:
        print("ERROR: Could not create spreadsheet", file=sys.stderr)
        sys.exit(1)
    print(f"  Spreadsheet ID: {sid}")
    print(f"  URL: https://docs.google.com/spreadsheets/d/{sid}")
    return sid


def _read_sheet_all(
    sid: str, sheet_name: str = "All Operations"
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


# Column headers that belong to the reviewer even if they appear in our CSV.
# These columns must never be overwritten by CSV data — their values on the
# sheet are the source of truth (filled in by reviewers).
_ALWAYS_PRESERVE = frozenset(
    {
        "Reviewer Weight",
        "Reviewer Notes",
        "Review Status",
        "Post Review Action",
    }
)


def _find_reviewer_columns(
    sheet_headers: list[str],
    csv_headers: list[str],
) -> list[int]:
    """Return sheet column indices that should be preserved (reviewer-owned).

    A column is reviewer-owned if:
    - Its header is NOT in our CSV headers, OR
    - Its header is in _ALWAYS_PRESERVE (e.g. "Reviewer Weight" which is
      in our CSV but always empty — the reviewer fills it in on the sheet).
    """
    csv_set = set(csv_headers) - _ALWAYS_PRESERVE
    return [
        i
        for i, h in enumerate(sheet_headers)
        if h not in csv_set or h in _ALWAYS_PRESERVE
    ]


def upload_data(sid: str, rows: list[list[str]]) -> None:
    """Upload CSV data to Sheet 1, preserving reviewer-added columns.

    Algorithm:
    1. Read the sheet's current state (headers + all data).
    2. Identify reviewer-added columns (headers not in our CSV).
    3. Build a map: operation_name -> {reviewer_col_header: value, ...}
       keyed by the Operation column so alignment is by name, not row.
    4. Clear the sheet and write our CSV data (all columns including
       our "Reviewer Weight" placeholder).
    5. Write reviewer data back, aligned by operation name to match
       the new row order.
    """
    csv_headers = rows[0]
    csv_data = rows[1:]
    print(f"Uploading {len(csv_data)} data rows ({len(csv_headers)} CSV columns)...")

    # --- Step 1: Read current sheet state ---
    sheet_headers, sheet_data = _read_sheet_all(sid)

    if not sheet_headers:
        print("  Fresh sheet, uploading all columns...")
        _upload_all_rows(sid, rows)
        return

    # --- Step 2: Identify reviewer columns ---
    reviewer_col_indices = _find_reviewer_columns(sheet_headers, csv_headers)
    reviewer_col_names = [sheet_headers[i] for i in reviewer_col_indices]
    reviewer_col_set = set(reviewer_col_names)

    if reviewer_col_names:
        print(f"  Found reviewer columns: {reviewer_col_names}")
    else:
        print("  No reviewer columns found.")

    # --- Step 3: Build operation -> reviewer data map ---
    # Find the Operation column on the sheet (should be index 0)
    try:
        op_col_idx = sheet_headers.index("Operation")
    except ValueError:
        op_col_idx = 0

    reviewer_data: dict[str, dict[str, str]] = {}
    for row in sheet_data:
        if not row or len(row) <= op_col_idx:
            continue
        op_name = row[op_col_idx]
        if not op_name:
            continue
        reviewer_data[op_name] = {}
        for col_idx in reviewer_col_indices:
            col_name = sheet_headers[col_idx]
            value = row[col_idx] if col_idx < len(row) else ""
            reviewer_data[op_name][col_name] = value

    non_empty = sum(
        1 for op_vals in reviewer_data.values() for v in op_vals.values() if v.strip()
    )
    print(
        f"  Captured {non_empty} non-empty reviewer values across {len(reviewer_data)} ops."
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
                "range": "'All Operations'!A1:ZZ",
            }
        ),
    )

    # Build output column order: preserve the sheet's original column layout.
    # For each sheet column, either pull from CSV (by header name) or from
    # the reviewer data (by operation name). This keeps reviewer columns
    # in their original positions (e.g. F and G stay as F and G).
    csv_header_to_idx = {h: i for i, h in enumerate(csv_headers)}

    # Determine output columns: sheet's existing order, but skip CSV's
    # "Reviewer Weight" since it's always empty and the sheet has the real one.
    out_col_sources: list[tuple[str, str]] = []  # (header, source: "csv"|"reviewer")
    for sheet_col_name in sheet_headers:
        if sheet_col_name in _ALWAYS_PRESERVE:
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
        if csv_h not in sheet_header_set and csv_h not in _ALWAYS_PRESERVE:
            out_col_sources.append((csv_h, "csv"))

    # Build rows
    out_headers = [src[0] for src in out_col_sources]
    out_data = []
    for csv_row in csv_data:
        op_name = csv_row[0] if csv_row else ""
        reviewer_vals = reviewer_data.get(op_name, {})
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
    # Where the reviewer provided a numeric weight, use it as Active Weight.
    # This avoids per-cell API writes after upload.
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
        f"aligned by operation name."
    )


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
                    "range": f"'All Operations'!A{row_start}",
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


def apply_formatting(sid: str, num_rows: int, num_cols: int) -> None:
    """Apply all formatting, dropdowns, and conditional rules."""
    print("Applying formatting...")
    sheet_id = 0
    requests = []

    # ---- Clear ALL existing conditional formatting first ----
    # Without this, rules accumulate across uploads and stale rules
    # override the new ones (Sheets evaluates top-down, first match wins).
    # We read the current count and delete them all in reverse order.
    try:
        sheet_meta = gws(
            "sheets",
            "spreadsheets",
            "get",
            "--params",
            json.dumps(
                {
                    "spreadsheetId": sid,
                    "fields": "sheets.conditionalFormats",
                }
            ),
        )
        if isinstance(sheet_meta, str):
            sheet_meta = json.loads(sheet_meta)
        existing_rules = sheet_meta["sheets"][0].get("conditionalFormats", [])
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

    # ---- Freeze header row + column A ----
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

    # ---- Status dropdown (col B, index 1) ----
    requests.append(
        {
            "setDataValidation": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": num_rows,
                    "startColumnIndex": 1,
                    "endColumnIndex": 2,
                },
                "rule": {
                    "condition": {
                        "type": "ONE_OF_LIST",
                        "values": [
                            {"userEnteredValue": "benchmarked"},
                            {"userEnteredValue": "alias"},
                            {"userEnteredValue": "excluded"},
                            {"userEnteredValue": "free"},
                            {"userEnteredValue": "blacklisted"},
                            {"userEnteredValue": "blacklisted-by-reviewer"},
                            {"userEnteredValue": "keep"},
                        ],
                    },
                    "showCustomUi": True,
                    "strict": False,
                },
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


def create_summary_sheet(sid: str, rows: list[list[str]]) -> None:
    """Populate the Review Summary sheet with formulas."""
    print("Creating summary sheet...")
    header = rows[0]
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
    summary.append(["5. Weight < 1.0 means cheaper than np.add per analytical FLOP", ""])
    summary.append(["6. Weight > 1.0 means more expensive (e.g., sin=18.39)", ""])

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


def main():
    parser = argparse.ArgumentParser(description="Upload weights CSV to Google Sheets")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to weights CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--spreadsheet-id",
        type=str,
        default=None,
        help="Update an existing spreadsheet instead of creating a new one. "
        "Reviewer-added columns are preserved and realigned by operation name.",
    )
    args = parser.parse_args()

    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} rows, {len(rows[0])} columns from {args.csv}")

    if args.spreadsheet_id:
        sid = args.spreadsheet_id
        print(f"Updating existing spreadsheet: {sid}")
        upload_data(sid, rows)
        apply_formatting(sid, num_rows=len(rows), num_cols=len(rows[0]))
    else:
        sid = create_spreadsheet()
        upload_data(sid, rows)
        apply_formatting(sid, num_rows=len(rows), num_cols=len(rows[0]))
        create_summary_sheet(sid, rows)

    url = f"https://docs.google.com/spreadsheets/d/{sid}"
    print(f"\nDone! Spreadsheet URL:\n  {url}")


if __name__ == "__main__":
    main()
