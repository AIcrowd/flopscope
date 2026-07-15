"""Generate the committed cost-model sheet (CSV + HTML + dtype-rate table).

Iterates every registry op, measures each charged op on its canonical input
(raw unit-weight cost, 2x-scale cost, production-billed cost per dtype), and
writes:

- docs/reference/cost-model-sheet.csv        (one row per registry op)
- docs/reference/cost-model-dtype-rates.csv  (global dtype-rate table)
- website/public/cost-model-sheet.html       (self-contained filterable page)

Usage:
    uv run python scripts/generate_cost_sheet.py          # regenerate
    uv run python scripts/generate_cost_sheet.py --check  # drift gate

``--check`` re-derives the CSVs and diffs them against the committed files,
comparing modulo the commit SHA embedded in permalinks (a regeneration at a
later commit must not count as drift; a moved registry LINE does).
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Allow `python scripts/generate_cost_sheet.py`: the interpreter puts scripts/
# on sys.path, not the repo root that the `scripts.cost_sheet` imports need.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from flopscope._registry import REGISTRY
from flopscope._weights import get_weight, load_weights, reset_weights
from scripts.cost_sheet import canonical_inputs, measure, refs, render
from scripts.cost_sheet.schema import CostRow

CSV = REPO / "docs" / "reference" / "cost-model-sheet.csv"
DT_CSV = REPO / "docs" / "reference" / "cost-model-dtype-rates.csv"
HTML = REPO / "website" / "public" / "cost-model-sheet.html"

_CHARGED = {
    "counted_unary",
    "counted_binary",
    "counted_reduction",
    "counted_custom",
    "counted_random_method",
}
_SHA_RE = re.compile(r"blob/[0-9a-f]{40}/")


def _module_of(op: str) -> str:
    return op.split(".")[0] if "." in op else "numpy"


def _status(cat: str) -> str:
    return {
        "free": "free",
        "free_random_method": "free",
        "blacklisted": "blacklisted",
    }.get(cat, "charged")


def _penalty(billed: dict) -> str:
    c, f32 = billed["complex128"], billed["float32"]
    if isinstance(c, int) and isinstance(f32, int) and f32:
        return f"{c / f32:.1f}"
    return "—"


def _dtype_rule(op: str, e: dict) -> str:
    # Default heuristic by category; per-op curation refines this later.
    cat = e.get("category")
    if cat == "counted_random_method":
        return "output"
    if cat in ("counted_unary", "counted_binary", "counted_custom"):
        return "operands"
    if cat == "counted_reduction":
        return "operands/accumulator"
    return "neutral"


def _numpy_range(e: dict) -> str:
    lo, hi = e.get("min_numpy", ""), e.get("max_numpy", "")
    return f"{lo}..{hi}" if (lo or hi) else ""


def build_rows() -> tuple[list[CostRow], list[str], list[tuple[str, str]]]:
    """One row per registry op.

    Returns ``(rows, missing, failed)``: *missing* lists charged ops with no
    canonical input yet (pending curation); *failed* lists ``(op, error
    class)`` for ops whose weight lookup or measurement raised. Both kinds
    still get a row, with the measured cells left empty.
    """
    sha = refs.current_sha()
    missing: list[str] = []
    failed: list[tuple[str, str]] = []

    # Snapshot production weights BEFORE measuring anything: measure_raw /
    # measure_billed reset+reload the active weights as they run, so calling
    # get_weight() inside the loop would silently return unit 1.0 after the
    # first measurement.
    load_weights()
    weights: dict[str, float | str] = {}
    for op, e in REGISTRY.items():
        if _status(e.get("category", "")) == "charged":
            try:
                weights[op] = get_weight(op)
            except Exception as exc:
                weights[op] = ""
                failed.append((op, type(exc).__name__))
    reset_weights()

    rows: list[CostRow] = []
    for op, e in REGISTRY.items():
        cat = e.get("category", "")
        status = _status(cat)
        line = refs.registry_entry_line(op)
        reg_ref = refs.permalink(refs.REGISTRY_REL, line, sha) if line else ""
        base: dict = {
            "op": op,
            "module": _module_of(op),
            "status": status,
            "category": cat,
            "flop_cost_formula": e.get("cost_formula", ""),
            "weight": (weights.get(op, "") if status == "charged" else 0.0),
            "complex_factor": str(e.get("complex_factor", "")),
            "dtype_rate_rule": _dtype_rule(op, e),
            "example_input": "",
            "raw_flop_cost": 0,
            "raw_flop_cost_2x": "",
            "billed_int16": "",
            "billed_fp32": "",
            "billed_fp64": "",
            "billed_complex128": "",
            "complex_penalty": "—",
            "notes": e.get("notes", ""),
            "numpy_range": _numpy_range(e),
            "registry_ref": reg_ref,
            "cost_impl_ref": "",
        }
        if status == "charged":
            spec = None
            try:
                spec = canonical_inputs.resolve(op, e)
                if spec is None:
                    missing.append(op)
                    base.update(example_input="(pending curation)", raw_flop_cost="")
                else:
                    m = measure.measure_op(spec.make, spec.scalable)
                    impl = m["cost_impl"]
                    base.update(
                        example_input=spec.describe,
                        raw_flop_cost=m["raw_flop_cost"],
                        raw_flop_cost_2x=str(m["raw_flop_cost_2x"]),
                        billed_int16=str(m["billed"]["int16"]),
                        billed_fp32=str(m["billed"]["float32"]),
                        billed_fp64=str(m["billed"]["float64"]),
                        billed_complex128=str(m["billed"]["complex128"]),
                        complex_penalty=_penalty(m["billed"]),
                        cost_impl_ref=(
                            refs.permalink(impl[0], impl[1], sha) if impl else ""
                        ),
                    )
            except Exception as exc:
                failed.append((op, type(exc).__name__))
                base.update(
                    example_input=(
                        spec.describe if spec is not None else "(pending curation)"
                    ),
                    raw_flop_cost="",
                )
        rows.append(CostRow(**base))
    return rows, missing, failed


def _dt_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    # "\n" (not the csv default "\r\n"): this string is committed to git.
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["dtype", "rate"])
    for r in rows:
        w.writerow([r["dtype"], r["rate"]])
    return buf.getvalue()


def _normalize_shas(text: str) -> str:
    return _SHA_RE.sub("blob/SHA/", text)


def main(check: bool) -> int:
    rows, missing, failed = build_rows()
    unmeasured = set(missing) | {op for op, _err in failed}
    measured = sum(1 for r in rows if r.status == "charged" and r.op not in unmeasured)
    print(
        f"rows={len(rows)} measured={measured} "
        f"pending-curation={len(missing)} failed={len(failed)}"
    )
    if failed:
        print(
            "failed: " + ", ".join(f"{op} ({err})" for op, err in failed),
            file=sys.stderr,
        )

    csv_text = render.to_csv(rows)
    dt_rows = render.dtype_rates_table()
    dt_text = _dt_csv(dt_rows)

    if check:
        stale: list[str] = []
        for path, want in ((CSV, csv_text), (DT_CSV, dt_text)):
            try:
                have = path.read_text()
            except FileNotFoundError:
                stale.append(path.name)
                continue
            if _normalize_shas(have) != _normalize_shas(want):
                stale.append(path.name)
        if stale:
            print(
                f"stale: {', '.join(stale)} — regenerate with "
                "`uv run python scripts/generate_cost_sheet.py`",
                file=sys.stderr,
            )
            return 1
        return 0

    for path in (CSV, DT_CSV, HTML):
        path.parent.mkdir(parents=True, exist_ok=True)
    CSV.write_text(csv_text)
    DT_CSV.write_text(dt_text)
    HTML.write_text(render.to_html(rows, dt_rows, np.__version__, refs.current_sha()))
    print(f"wrote {len(rows)} rows to {CSV}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the cost-model sheet.")
    ap.add_argument(
        "--check",
        action="store_true",
        help="diff against the committed CSVs instead of writing; exit 1 on drift",
    )
    raise SystemExit(main(ap.parse_args().check))
