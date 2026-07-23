"""Tests for Rich budget display and plain-text fallback."""

from unittest.mock import patch

import pytest

import flopscope as flops
from flopscope._budget import BudgetContext
from flopscope._display import _plain_text_summary, budget_live, render_budget_summary


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
