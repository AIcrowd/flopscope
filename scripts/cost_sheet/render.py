from __future__ import annotations

import csv
import dataclasses
import io
import json

import flopscope._weights as W
from flopscope._weights import load_weights, reset_weights
from scripts.cost_sheet.schema import COLUMNS, LEGEND, CostRow


def to_csv(rows: list[CostRow]) -> str:
    buf = io.StringIO()
    # "\n" (not the csv default "\r\n"): this string is committed to git.
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for r in rows:
        w.writerow([getattr(r, c) for c in COLUMNS])
    return buf.getvalue()


def dtype_rates_table() -> list[dict]:
    """Global dtype-rate table, read from the packaged production weights."""
    load_weights()
    try:
        # Reading the private _ACTIVE_DTYPE_RATES is accepted here: this is
        # repo-internal generator tooling, not shipped library code.
        return [{"dtype": k, "rate": v} for k, v in W._ACTIVE_DTYPE_RATES.items()]
    finally:
        reset_weights()  # don't leak production weights into whatever runs next


def to_html(
    rows: list[CostRow], dtype_rows: list[dict], numpy_version: str, sha: str
) -> str:
    data = [dataclasses.asdict(r) for r in rows]
    # A single self-contained page: embedded JSON + a small filter script.
    return (
        _HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
        .replace("__DTYPES__", json.dumps(dtype_rows))
        .replace("__LEGEND__", json.dumps(LEGEND))
        .replace("__NUMPY__", numpy_version)
        .replace("__SHA__", sha)
    )


_HTML_TEMPLATE = """<!doctype html><meta charset=utf-8>
<title>flopscope cost model</title>
<p>numpy __NUMPY__ · commit __SHA__</p>
<input id=q placeholder="filter ops">
<table id=t></table>
<h2>dtype rates</h2>
<table id=d></table>
<script>
const DATA=__DATA__, DTYPES=__DTYPES__, LEGEND=__LEGEND__;
const cols=Object.keys(LEGEND);
function render(f){const t=document.getElementById('t');
 const rows=DATA.filter(r=>!f||r.op.includes(f));
 t.innerHTML='<tr>'+cols.map(c=>`<th title="${LEGEND[c]}">${c}</th>`).join('')+'</tr>'+
  rows.map(r=>'<tr>'+cols.map(c=>`<td>${r[c]}</td>`).join('')+'</tr>').join('');}
document.getElementById('q').oninput=e=>render(e.target.value);
render('');
document.getElementById('d').innerHTML='<tr><th>dtype</th><th>rate</th></tr>'+
 DTYPES.map(r=>`<tr><td>${r.dtype}</td><td>${r.rate}</td></tr>`).join('');
</script>"""
