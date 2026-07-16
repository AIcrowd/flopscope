#!/usr/bin/env python3
"""Upload a flopscope review CSV to Google Sheets with formatting and dropdowns.

Schema-agnostic: each uploadable sheet is described by a ``SheetPreset``
(title, key column, dropdown columns, reviewer-owned columns, CSV path).
Re-uploads anchor the output layout on the canonical columns (so canonical
columns lost on the live sheet come back) and preserve reviewer-owned
columns, realigned by the preset's key column so reviewer annotations
survive row insertions/removals/reordering.

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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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

# Presentation-only column order for the cost-model REVIEW sheet: identity
# first, then the reviewer block where the eye lands, then the cost model,
# its evidence, the billed matrix, and the provenance tail. Applied at load
# time by the uploader; the committed CSV artifact keeps its own order.
_COST_MODEL_COLUMN_ORDER = (
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

# Billed-outcome columns: the per-dtype billing matrix plus the derived
# penalty column. Cells hold numbers, "—" (not applicable) or "raises".
_COST_MODEL_BILLED_COLUMNS = (
    "billed_int16",
    "billed_fp32",
    "billed_fp64",
    "billed_complex128",
    "complex_penalty",
)


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
        column_order: Optional presentation-only column order, by header
            name. Applied at load time (after ship_empty_columns) and to the
            merged output on re-uploads, so fresh and update paths both land
            in this order. Headers not named keep their relative order after
            the named ones; named headers missing from the data are skipped.
            The CSV artifact on disk is untouched.
        note_tooltips: Mapping of target header -> source header. On every
            data row, the SOURCE column's cell text is written as the cell
            note (Sheets' native hover tooltip) on that row's TARGET-column
            cell — e.g. hovering an op name shows the row's ``notes`` prose
            without scrolling. Columns are resolved by header name on the
            final uploaded layout; presentation-only, values untouched.
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
    column_order: tuple[str, ...] | None = None
    note_tooltips: dict[str, str] = field(default_factory=dict)
    format_hook: Callable[[list[str], int, int, int], list[dict]] | None = None
    summary_builder: Callable[[list[list[str]]], list[list[str]]] | None = None


def gws(*args: str, json_body: dict | None = None) -> dict:
    """Run a gws CLI command and return parsed JSON output."""
    cmd = ["gws"] + list(args)
    if json_body is not None:
        # Always inline: this gws build has no @file syntax (it would parse
        # the literal "@/tmp/..." as JSON and fail). Callers keep individual
        # bodies small (values writes chunk at 50 rows; note requests chunk
        # at _NOTE_ROWS_PER_REQUEST), far below OS argument limits.
        cmd += ["--json", json.dumps(json_body)]

    result = subprocess.run(cmd, capture_output=True, text=True)

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


def _reorder_columns(
    rows: list[list[str]], column_order: Sequence[str] | None
) -> list[list[str]]:
    """Reorder columns by header name (a presentation-only permutation).

    Headers named in ``column_order`` come first, in that order; names not
    present in the data are skipped. All remaining headers follow, keeping
    their existing relative order. Every row is permuted identically (short
    rows are padded to header width first), so cell/column association is
    preserved. No-op — returning ``rows`` unchanged — when ``column_order``
    is falsy, ``rows`` is empty, or the data already matches the order.
    """
    if not column_order or not rows:
        return rows
    header = rows[0]
    named: list[str] = []
    for name in column_order:
        if name in header and name not in named:
            named.append(name)
    named_set = set(column_order)
    tail = [h for h in header if h not in named_set]
    new_header = named + tail
    if new_header == header:
        return rows
    perm = [header.index(h) for h in new_header]
    width = len(header)
    out: list[list[str]] = []
    for row in rows:
        padded = row + [""] * (width - len(row))
        out.append([padded[i] for i in perm])
    return out


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


def _dedupe_headers(
    sheet_headers: list[str], canonical: set[str]
) -> tuple[list[str], list[int]]:
    """Drop repeated live headers, keeping each name's first occurrence.

    A stray duplicated header — e.g. a drag-copy slip that stamped one
    column's header over a neighbour — must not corrupt the merge. Returns
    the de-duplicated header names plus, for each, its column index on the
    live sheet, so values are still read from the right cells. A repeated
    canonical header is ignored (canonical data wins on re-upload); a
    repeated reviewer-owned header keeps its first occurrence's values.
    Both cases print a warning naming the column.
    """
    names: list[str] = []
    indices: list[int] = []
    seen: set[str] = set()
    for i, name in enumerate(sheet_headers):
        if name in seen:
            if name in canonical:
                print(
                    f"  Warning: duplicate canonical column {name!r} on the "
                    "live sheet; ignoring the duplicate (canonical data wins)."
                )
            else:
                print(
                    f"  Warning: duplicate reviewer column {name!r} on the "
                    "live sheet; keeping the first occurrence."
                )
            continue
        seen.add(name)
        names.append(name)
        indices.append(i)
    return names, indices


def upload_data(
    sid: str, rows: list[list[str]], preset: SheetPreset
) -> list[list[str]]:
    """Upload CSV data to the data sheet, preserving reviewer-added columns.

    Algorithm:
    1. Read the sheet's current state (headers + all data), de-duplicating
       repeated live headers (first occurrence wins) so a drag-copy slip
       cannot corrupt the merge.
    2. Identify reviewer-owned columns (headers not in our CSV, plus the
       preset's ``preserve_columns`` even when CSV-shipped).
    3. Build a map: key -> {reviewer_col_header: value, ...} keyed by the
       preset's key column so alignment is by key, not row position.
    4. Clear the sheet and write the merged grid, anchored on the CANONICAL
       headers (CSV columns plus shipped-empty annotation columns): every
       canonical column ships, whatever state the live sheet is in. A
       canonical column missing from the live sheet comes back fresh —
       empty for annotation (``preserve_columns``) columns, CSV data for
       data columns. Reviewer-owned EXTRA live columns are kept.
    5. Write reviewer data back, aligned by key to match the new row order.

    When the preset declares a ``column_order``, the merged output is
    permuted into that order before upload, so update uploads land in the
    declared layout even if the live sheet predates it. Reviewer alignment
    is by header name + key, so annotations survive the permutation.

    Returns the full grid actually written to the sheet — header row first,
    then every data row in final order. Downstream row-aligned presentation
    (e.g. ``note_tooltips``) must read THIS grid, never the input CSV rows,
    so it tracks the merged, reordered upload.
    """
    rows = _reorder_columns(rows, preset.column_order)
    csv_headers = rows[0]
    csv_data = rows[1:]
    print(f"Uploading {len(csv_data)} data rows ({len(csv_headers)} CSV columns)...")

    # --- Step 1: Read current sheet state ---
    sheet_headers, sheet_data = _read_sheet_all(sid)

    if not sheet_headers:
        print("  Fresh sheet, uploading all columns...")
        _upload_all_rows(sid, rows)
        return rows

    # --- Step 2: Identify reviewer columns ---
    # Work on a de-duplicated view of the live header row (first occurrence
    # wins); ``live_indices`` maps every kept header back to its live
    # column, so reviewer values are captured from the right cells even
    # when the live sheet carries duplicated headers.
    canonical_set = set(csv_headers)
    live_headers, live_indices = _dedupe_headers(sheet_headers, canonical_set)
    reviewer_positions = _find_reviewer_columns(
        live_headers, csv_headers, preset.preserve_columns
    )
    reviewer_col_indices = [live_indices[i] for i in reviewer_positions]
    reviewer_col_names = [live_headers[i] for i in reviewer_positions]
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

    # Build the output layout anchored on the CANONICAL headers — the CSV
    # columns plus shipped-empty annotation columns, with ``column_order``
    # applied — never on whatever layout the live sheet ended up with.
    # Live columns are consumed in their (de-duplicated) live order here;
    # canonical columns the live sheet lost are re-added below; and the
    # final ``_reorder_columns`` pass lands presets whose ``column_order``
    # names every canonical column (cost-model) in the canonical layout,
    # reviewer-owned extras appended after it. Presets without a
    # ``column_order`` (weights) keep their live layout on an intact sheet
    # and self-heal missing canonical columns at the tail.
    csv_header_to_idx = {h: i for i, h in enumerate(csv_headers)}

    out_col_sources: list[tuple[str, str]] = []  # (header, source: "csv"|"reviewer")
    for col_name in live_headers:
        if col_name in preset.preserve_columns:
            out_col_sources.append((col_name, "reviewer"))
        elif col_name in canonical_set and col_name not in reviewer_col_set:
            out_col_sources.append((col_name, "csv"))
        else:
            out_col_sources.append((col_name, "reviewer"))

    # Canonical columns missing from the live sheet ship fresh: annotation
    # (preserve) columns come back EMPTY, data columns carry CSV data. This
    # is the self-heal path — without it, a canonical column lost on the
    # live sheet (e.g. its header drag-copied over) stayed lost on every
    # re-upload, taking its dropdown with it.
    live_header_set = set(live_headers)
    healed = [h for h in csv_headers if h not in live_header_set]
    if healed:
        print(f"  Restoring canonical columns missing from the sheet: {healed}")
    for csv_h in healed:
        source = "reviewer" if csv_h in preset.preserve_columns else "csv"
        out_col_sources.append((csv_h, source))

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

    # Permute the merged output into the preset's declared column order
    # (no-op without one). Done after the merge so it applies uniformly to
    # header and data rows — reviewer values stay glued to their columns.
    all_out = _reorder_columns([out_headers] + out_data, preset.column_order)
    _upload_all_rows(sid, all_out)

    print(
        f"  Uploaded {len(out_data)} rows: {len(out_col_sources)} columns "
        f"({sum(1 for _, s in out_col_sources if s == 'csv')} CSV, "
        f"{sum(1 for _, s in out_col_sources if s == 'reviewer')} reviewer), "
        f"aligned by {preset.key_column!r}."
    )
    return all_out


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


# Defensive cap on note length. Sheets allows far longer notes; the registry
# notes are short prose (a few hundred chars), so this only guards against a
# pathological row bloating the batchUpdate payload.
_NOTE_MAX_LEN = 2000
# Rows per updateCells note request: keeps each batchUpdate body well under
# inline-argument limits (~200 rows x ~200 chars ~= 18 KB).
_NOTE_ROWS_PER_REQUEST = 200


def _note_tooltip_requests(
    preset: SheetPreset,
    headers: list[str],
    data_rows: Sequence[Sequence[str]],
    sheet_id: int,
) -> list[dict]:
    """updateCells requests that mirror source columns as hover notes.

    For each ``target -> source`` pair in the preset's ``note_tooltips``,
    one request stamps every data-row cell of the TARGET column with the
    same row's SOURCE-column text as its cell note (Sheets' native hover
    tooltip). Columns are resolved by header name; ``data_rows`` must be
    the data grid ``upload_data`` actually wrote (header row excluded), so
    notes stay row-aligned with the merged, reordered upload. EVERY data
    row gets a note entry — an empty source cell writes an empty note,
    clearing any stale note a previous upload left when rows shifted.
    ``fields: "note"`` scopes the write to notes only; cell values and
    formatting are untouched.
    """
    requests: list[dict] = []
    if not data_rows:
        return requests
    for target, source in preset.note_tooltips.items():
        missing = [name for name in (target, source) if name not in headers]
        if missing:
            print(f"  Warning: tooltip column(s) {missing} not on sheet; skipping.")
            continue
        target_col = headers.index(target)
        source_col = headers.index(source)
        note_rows: list[dict] = []
        for row in data_rows:
            text = row[source_col] if source_col < len(row) else ""
            note_rows.append({"values": [{"note": text[:_NOTE_MAX_LEN]}]})
        # Chunk rows per request so no single batchUpdate body balloons past
        # what the gws CLI accepts as an inline --json argument.
        for start in range(0, len(note_rows), _NOTE_ROWS_PER_REQUEST):
            chunk = note_rows[start : start + _NOTE_ROWS_PER_REQUEST]
            requests.append(
                {
                    "updateCells": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 1 + start,
                            "endRowIndex": 1 + start + len(chunk),
                            "startColumnIndex": target_col,
                            "endColumnIndex": target_col + 1,
                        },
                        "rows": chunk,
                        "fields": "note",
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


def _contiguous_runs(indexes: Sequence[int]) -> list[tuple[int, int]]:
    """Coalesce column indexes into sorted ``(start, end-exclusive)`` runs.

    Purely an API-call reducer: adjacent columns share one repeatCell range
    instead of one request each. Works on any index set, so name-resolved
    groups stay correct even when the live sheet's columns are scattered.
    """
    runs: list[tuple[int, int]] = []
    for i in sorted(set(indexes)):
        if runs and runs[-1][1] == i:
            runs[-1] = (runs[-1][0], i + 1)
        else:
            runs.append((i, i + 1))
    return runs


# Cost-model column groups: (headers, data-row fill, header-cell fill).
# Pastel body under a saturated same-hue header; resolved by header NAME at
# format time — never by position — so the styling follows the columns.
_COST_MODEL_GROUPS: tuple[tuple[tuple[str, ...], dict, dict], ...] = (
    # identity: what op is this row about
    (
        ("op", "module", "status", "category"),
        _color(1.0, 1.0, 1.0),  # white
        _color(0.85, 0.85, 0.87),  # light gray
    ),
    # reviewer block: the three columns reviewers fill in
    (
        _COST_MODEL_ANNOTATIONS,
        _color(1.0, 0.973, 0.863),  # cornsilk (#FFF8DC)
        _color(0.961, 0.843, 0.431),  # saturated gold (#F5D76E)
    ),
    # cost model: what we charge
    (
        ("weight", "flop_cost_formula", "complex_factor", "dtype_rate_rule"),
        _color(0.875, 0.922, 0.973),  # light blue
        _color(0.62, 0.77, 0.906),  # saturated blue
    ),
    # evidence: the measured example backing the formula
    (
        ("example_input", "raw_flop_cost", "raw_flop_cost_2x"),
        _color(0.925, 0.906, 0.965),  # light lavender
        _color(0.78, 0.729, 0.898),  # saturated lavender
    ),
    # billed matrix: per-dtype outcomes + derived penalty
    (
        _COST_MODEL_BILLED_COLUMNS,
        _color(0.988, 0.925, 0.871),  # light peach
        _color(0.953, 0.78, 0.62),  # saturated peach
    ),
    # provenance: prose + links back to the source
    (
        ("notes", "numpy_range", "registry_ref", "cost_impl_ref"),
        _color(0.945, 0.945, 0.949),  # light gray
        _color(0.82, 0.82, 0.839),  # saturated gray
    ),
)


def _cost_model_format_requests(
    headers: list[str], num_rows: int, num_cols: int, sheet_id: int
) -> list[dict]:
    """Cost-model-sheet formatting; columns are resolved by header name.

    Reviewer-ergonomics layout: per-group background fills (pastel data
    rows, saturated same-hue header cells), verdict/status/"raises"
    conditional formats, per-column widths, wrapped prose columns, and a
    basic filter over the whole data range so reviewers can sort/filter.
    Headers missing from the sheet are skipped, so the hook stays correct
    whatever order (or subset) the live sheet ends up with. The frozen
    header row/key column comes from ``apply_formatting`` — not repeated
    here.
    """
    requests: list[dict] = []
    full_width = max(len(headers), num_cols)

    def col_of(name: str) -> int | None:
        return headers.index(name) if name in headers else None

    def cols_of(names: Sequence[str]) -> list[int]:
        return [headers.index(n) for n in names if n in headers]

    # ---- Header row: bold text (group hues below fill the backgrounds) ----
    requests.append(
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": full_width,
                },
                "cell": {
                    "userEnteredFormat": {
                        "textFormat": {"bold": True, "fontSize": 10},
                    }
                },
                "fields": "userEnteredFormat.textFormat",
            }
        }
    )

    # ---- Group fills: pastel data rows + saturated header cells ----
    for names, body_fill, header_fill in _COST_MODEL_GROUPS:
        for start, end in _contiguous_runs(cols_of(names)):
            for row_lo, row_hi, fill in (
                (1, num_rows, body_fill),
                (0, 1, header_fill),
            ):
                requests.append(
                    {
                        "repeatCell": {
                            "range": {
                                "sheetId": sheet_id,
                                "startRowIndex": row_lo,
                                "endRowIndex": row_hi,
                                "startColumnIndex": start,
                                "endColumnIndex": end,
                            },
                            "cell": {"userEnteredFormat": {"backgroundColor": fill}},
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
            ("unsure", _color(0.988, 0.867, 0.545)),  # amber
        ]
        for text, bg in verdict_rules:
            requests.append(_text_eq_rule(sheet_id, verdict_col, num_rows, text, bg))

    # ---- Conditional formatting: status tiers ----
    status_col = col_of("status")
    if status_col is not None:
        status_rules = [
            ("free", _color(0.898, 0.957, 0.898)),  # pale green
            ("blacklisted", _color(0.72, 0.72, 0.72)),  # mid gray
        ]
        for text, bg in status_rules:
            requests.append(_text_eq_rule(sheet_id, status_col, num_rows, text, bg))

    # ---- Conditional formatting: "raises" cells in the billed matrix ----
    for name in _COST_MODEL_BILLED_COLUMNS:
        col = col_of(name)
        if col is None:
            continue
        requests.append(
            _text_eq_rule(
                sheet_id,
                col,
                num_rows,
                "raises",
                bg=_color(0.93, 0.93, 0.93),  # light gray
                fg=_color(0.55, 0.55, 0.55),  # gray text
            )
        )

    # ---- Column widths (by header name) ----
    col_widths = {
        "op": 200,
        "module": 90,
        "status": 100,
        "category": 150,
        "looks-right?": 160,
        "proposed-change": 220,
        "reviewer-notes": 220,
        "weight": 110,
        "flop_cost_formula": 420,
        "complex_factor": 110,
        "dtype_rate_rule": 120,
        "example_input": 220,
        "raw_flop_cost": 110,
        "raw_flop_cost_2x": 110,
        "billed_int16": 110,
        "billed_fp32": 110,
        "billed_fp64": 110,
        "billed_complex128": 130,
        "complex_penalty": 110,
        "notes": 320,
        "numpy_range": 110,
        "registry_ref": 140,
        "cost_impl_ref": 140,
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

    # ---- Basic filter over the full data range (sort/filter per column) ----
    # setBasicFilter replaces any existing basic filter, so re-uploads do
    # not stack filters.
    requests.append(
        {
            "setBasicFilter": {
                "filter": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": num_rows,
                        "startColumnIndex": 0,
                        "endColumnIndex": full_width,
                    }
                }
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
    data_rows: Sequence[Sequence[str]] = (),
) -> None:
    """Apply dropdowns and preset formatting to the data sheet.

    ``headers`` is the header row actually written by ``upload_data`` (CSV
    columns plus shipped-empty and preserved reviewer columns); dropdown
    targets are resolved against it by name. ``num_rows``/``num_cols`` are
    the uploaded CSV dimensions, used by preset format hooks for ranges.
    ``sheet_id`` is the data tab's sheetId (0 on freshly created
    spreadsheets; whatever ``_ensure_data_sheet`` resolved on pre-existing
    ones) — every structural request targets it. ``data_rows`` is the data
    grid ``upload_data`` returned (header row excluded); the preset's
    ``note_tooltips`` read source-column text from it, row-aligned with
    what was actually uploaded.
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

    # ---- Hover notes: mirror source-column text as cell notes ----
    requests += _note_tooltip_requests(preset, headers, data_rows, sheet_id)

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
        column_order=_COST_MODEL_COLUMN_ORDER,
        # Hovering an op name shows that row's `notes` prose as a cell note,
        # so reviewers read the caveats without scrolling to column T.
        note_tooltips={"op": "notes"},
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
    rows = _reorder_columns(rows, preset.column_order)
    print(f"Loaded {len(rows)} rows, {len(rows[0])} columns from {csv_path}")

    if args.spreadsheet_id:
        sid = args.spreadsheet_id
        print(f"Updating existing spreadsheet: {sid}")
        data_sheet_id = _ensure_data_sheet(sid)
        uploaded = upload_data(sid, rows, preset)
        apply_formatting(
            sid,
            preset,
            uploaded[0],
            num_rows=len(rows),
            num_cols=len(rows[0]),
            sheet_id=data_sheet_id,
            data_rows=uploaded[1:],
        )
    else:
        sid = create_spreadsheet(preset)
        uploaded = upload_data(sid, rows, preset)
        apply_formatting(
            sid,
            preset,
            uploaded[0],
            num_rows=len(rows),
            num_cols=len(rows[0]),
            # create_spreadsheet pins the data tab to sheetId 0.
            sheet_id=0,
            data_rows=uploaded[1:],
        )
        create_summary_sheet(sid, rows, preset)

    url = f"https://docs.google.com/spreadsheets/d/{sid}"
    print(f"\nDone! Spreadsheet URL:\n  {url}")


if __name__ == "__main__":
    main()
