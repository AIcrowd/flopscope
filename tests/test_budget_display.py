"""Tests for Rich budget display and plain-text fallback."""

from unittest.mock import patch

import pytest

import flopscope as flops
from flopscope._budget import BudgetContext
from flopscope._display import (
    _display_totals,
    _plain_text_summary,
    budget_live,
    render_budget_summary,
)


def test_plain_text_summary_default_omits_namespace_section():
    with BudgetContext(flop_budget=1000, namespace="train", quiet=True) as ctx:
        with ctx.deduct("add", flop_cost=100, subscripts=None, shapes=(), dtypes=()):
            pass
        with flops.namespace("precompute"):
            with ctx.deduct(
                "mul", flop_cost=200, subscripts=None, shapes=(), dtypes=()
            ):
                pass

    text = _plain_text_summary()
    assert "300" in text.replace(",", "")
    assert "By namespace:" not in text
    assert "add" in text
    assert "mul" in text


def test_plain_text_summary_by_namespace_shows_dotted_rows():
    with BudgetContext(flop_budget=1000, namespace="predict", quiet=True) as ctx:
        with ctx.deduct("mul", flop_cost=50, subscripts=None, shapes=(), dtypes=()):
            pass
        with flops.namespace("precompute"):
            with ctx.deduct(
                "add", flop_cost=100, subscripts=None, shapes=(), dtypes=()
            ):
                pass

    with BudgetContext(flop_budget=500, quiet=True) as ctx:
        with ctx.deduct("sum", flop_cost=25, subscripts=None, shapes=(), dtypes=()):
            pass

    text = _plain_text_summary(by_namespace=True)
    assert "By namespace:" in text
    assert "predict.precompute" in text
    assert "predict" in text
    assert "(unlabeled)" in text


def test_plain_text_summary_no_data():
    text = _plain_text_summary()
    assert "No budget data" in text


def test_render_budget_summary_falls_back_to_text():
    """When Rich is not available, render_budget_summary returns plain text."""
    with BudgetContext(flop_budget=1000, namespace="test", quiet=True) as ctx:
        with flops.namespace("nested"):
            with ctx.deduct(
                "add", flop_cost=100, subscripts=None, shapes=(), dtypes=()
            ):
                pass

    with patch.dict(
        "sys.modules",
        {"rich": None, "rich.panel": None, "rich.table": None, "rich.text": None},
    ):
        result = render_budget_summary(by_namespace=True)
        assert isinstance(result, str)
        assert "test.nested" in result


def test_render_budget_summary_with_rich():
    """When Rich is available, render_budget_summary returns a Rich renderable."""
    pytest.importorskip("rich")

    with BudgetContext(flop_budget=1000, namespace="test", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=(), dtypes=())

    result = render_budget_summary()
    from rich.panel import Panel

    assert isinstance(result, Panel)


def test_budget_live_is_context_manager():
    """budget_live() returns a context manager."""
    live = budget_live()
    assert hasattr(live, "__enter__")
    assert hasattr(live, "__exit__")


class _PoisonedRecords(list):
    def __iter__(self):
        raise AssertionError("display traversed namespace records")


def test_display_totals_do_not_iterate_namespace_records() -> None:
    import flopscope._budget as budget_module

    with BudgetContext(100, quiet=True):
        pass
    records = budget_module._accumulator._records
    budget_module._accumulator._records = _PoisonedRecords(records)
    try:
        assert _display_totals()["budget"] == 100
    finally:
        budget_module._accumulator._records = records


def test_display_totals_combine_live_implicit_global_and_explicit_budget() -> None:
    import flopscope._budget as budget_module

    global_ctx = budget_module._get_global_default()
    global_ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
    records = budget_module._accumulator._records
    budget_module._accumulator._records = _PoisonedRecords(records)
    try:
        with BudgetContext(100, quiet=True) as explicit:
            explicit.deduct(
                "multiply", flop_cost=7, subscripts=None, shapes=(), dtypes=()
            )
            totals = _display_totals()
    finally:
        budget_module._accumulator._records = records

    assert totals == {
        "has_explicit_budget": True,
        "budget": 100,
        "used": 12,
        "remaining": 88,
        "color": "green",
    }


def test_namespaced_implicit_global_budget_remains_hidden_from_explicit_total() -> None:
    import flopscope._budget as budget_module

    global_ctx = budget_module._get_global_default()
    with flops.namespace("phase"):
        global_ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
        with BudgetContext(100, quiet=True) as explicit:
            explicit.deduct(
                "multiply", flop_cost=7, subscripts=None, shapes=(), dtypes=()
            )
            totals = _display_totals()

    assert totals == {
        "has_explicit_budget": True,
        "budget": 100,
        "used": 12,
        "remaining": 88,
        "color": "green",
    }


def test_large_unlabeled_explicit_budget_remains_visible() -> None:
    explicit_budget = int(1e15)
    with BudgetContext(explicit_budget, quiet=True) as explicit:
        explicit.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
        totals = _display_totals()

    assert totals == {
        "has_explicit_budget": True,
        "budget": explicit_budget,
        "used": 5,
        "remaining": explicit_budget - 5,
        "color": "green",
    }


def test_display_totals_count_reentry_and_post_reset_deltas_exactly_once() -> None:
    ctx = BudgetContext(100, quiet=True)
    with ctx:
        ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
    assert _display_totals()["budget"] == 100
    assert _display_totals()["used"] == 5

    with ctx:
        ctx.deduct("multiply", flop_cost=7, subscripts=None, shapes=(), dtypes=())
        reentered_live = _display_totals()
    reentered_closed = _display_totals()

    assert reentered_live["budget"] == 100
    assert reentered_live["used"] == 12
    assert reentered_closed["budget"] == 100
    assert reentered_closed["used"] == 12

    with ctx:
        flops.budget_reset()
        ctx.deduct("subtract", flop_cost=3, subscripts=None, shapes=(), dtypes=())
        reset_live = _display_totals()
    reset_closed = _display_totals()

    assert reset_live["budget"] == 100
    assert reset_live["used"] == 3
    assert reset_closed["budget"] == 100
    assert reset_closed["used"] == 3


def test_display_totals_reset_discards_closed_aggregate() -> None:
    with BudgetContext(100, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
    assert _display_totals()["used"] == 5

    flops.budget_reset()

    assert _display_totals() == {
        "has_explicit_budget": False,
        "budget": 0,
        "used": 0,
        "remaining": 0,
        "color": "green",
    }
