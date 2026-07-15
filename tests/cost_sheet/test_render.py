from typing import Any

from scripts.cost_sheet.render import dtype_rates_table, to_csv, to_html
from scripts.cost_sheet.schema import COLUMNS, CostRow


def _row(**kw: Any) -> CostRow:
    base: dict[str, Any] = dict.fromkeys(COLUMNS, "")
    base.update(op="add", raw_flop_cost=1000, weight=1.0)
    base.update(kw)
    return CostRow(**base)


def test_csv_has_header_and_rows():
    csv = to_csv([_row(), _row(op="sum")])
    lines = csv.strip().splitlines()
    assert lines[0] == ",".join(COLUMNS)
    assert len(lines) == 3


def test_dtype_rates_table_has_18_rows():
    rows = dtype_rates_table()
    assert len(rows) == 18
    assert {"dtype", "rate"} <= set(rows[0])


def test_dtype_rates_table_does_not_leak_production_weights():
    # Discriminating isolation test: dtype_rates_table() loads production
    # weights to read the rates, but must reset afterwards so whatever runs
    # next (raw measurements) stays in unit mode.
    import flopscope._weights as W

    dtype_rates_table()
    assert W._ACTIVE_DTYPE_RATES == {}


def test_html_is_self_contained_and_embeds_data():
    html = to_html(
        [_row(op="matmul")], dtype_rates_table(), numpy_version="2.3.1", sha="abc1234"
    )
    assert html.startswith("<!doctype html>")
    assert "2.3.1" in html and "abc1234" in html
    assert "<script>" in html
    assert '"matmul"' in html  # row data embedded as JSON
    assert '"float64"' in html  # companion dtype-rate table embedded
    # Self-contained: no external resources fetched at view time.
    assert "src=" not in html and "href=" not in html
