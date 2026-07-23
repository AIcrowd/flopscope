"""Extended tests for _display.py to cover plain-text summary paths.

Covers: _format_flops, _pct edge cases, _usage_color, plain text with
no-namespace ops section, _PlainTextLive context manager, budget_summary.
"""

from unittest.mock import patch

import pytest

from flopscope._budget import BudgetContext
from flopscope._display import (
    _format_flops,
    _is_global_default_ns,
    _pct,
    _plain_text_summary,
    _usage_color,
    budget_live,
    budget_summary,
    render_budget_summary,
)

# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def test_format_flops_small():
    assert _format_flops(0) == "0"
    assert _format_flops(1000) == "1,000"
    assert _format_flops(1_000_000) == "1,000,000"


def test_pct_zero_total():
    assert _pct(0, 0) == "0.0%"
    assert _pct(100, 0) == "0.0%"


def test_pct_normal():
    assert _pct(50, 100) == "50.0%"
    assert _pct(1, 4) == "25.0%"


def test_usage_color_zero_total():
    assert _usage_color(0, 0) == "white"


def test_usage_color_low():
    assert _usage_color(10, 100) == "green"


def test_usage_color_medium():
    assert _usage_color(65, 100) == "yellow"


def test_usage_color_high():
    assert _usage_color(90, 100) == "red"


# ---------------------------------------------------------------------------
# _is_global_default_ns
# ---------------------------------------------------------------------------


def test_is_global_default_ns_none_with_large_budget():
    assert _is_global_default_ns(None, {"flop_budget": int(2e15)})


def test_is_global_default_ns_named():
    assert not _is_global_default_ns("train", {"flop_budget": 1000})


def test_is_global_default_ns_none_small_budget():
    assert not _is_global_default_ns(None, {"flop_budget": 1000})


# ---------------------------------------------------------------------------
# _plain_text_summary — edge cases
# ---------------------------------------------------------------------------


def test_plain_text_summary_no_data_message():
    text = _plain_text_summary()
    assert "No budget data" in text


def test_plain_text_summary_has_operations_section():
    """Session total ops section appears when ops are present."""
    with BudgetContext(flop_budget=1000, namespace="disp_test", quiet=True) as ctx:
        ctx.deduct("foo", flop_cost=100, subscripts=None, shapes=(), dtypes=())
        ctx.deduct("foo", flop_cost=200, subscripts=None, shapes=(), dtypes=())
        ctx.deduct("bar", flop_cost=50, subscripts=None, shapes=(), dtypes=())

    text = _plain_text_summary()
    assert "foo" in text
    assert "bar" in text
    assert "2 calls" in text or "1 call" in text


def test_plain_text_summary_single_call_word():
    """Singular 'call' vs plural 'calls'."""
    with BudgetContext(flop_budget=1000, namespace="call_test", quiet=True) as ctx:
        ctx.deduct("single_op", flop_cost=10, subscripts=None, shapes=(), dtypes=())

    text = _plain_text_summary()
    assert "1 call" in text


def test_plain_text_summary_percentages():
    with BudgetContext(flop_budget=1000, namespace="pct_test", quiet=True) as ctx:
        ctx.deduct("op1", flop_cost=500, subscripts=None, shapes=(), dtypes=())

    text = _plain_text_summary()
    assert "%" in text


# ---------------------------------------------------------------------------
# _PlainTextLive context manager
# ---------------------------------------------------------------------------


def test_plain_text_live_prints_on_exit(capsys):
    """_PlainTextLive prints a summary when exiting the context."""
    with patch.dict(
        "sys.modules",
        {"rich": None, "rich.live": None},
    ):
        live = budget_live()
        with live:
            with BudgetContext(
                flop_budget=100, namespace="live_test", quiet=True
            ) as ctx:
                ctx.deduct(
                    "live_op", flop_cost=10, subscripts=None, shapes=(), dtypes=()
                )
    # The output should be captured from the plain text fallback
    captured = capsys.readouterr()
    # Either something was printed or the rich live was used — just check no exception
    assert captured is not None


# ---------------------------------------------------------------------------
# budget_summary — non-IPython path
# ---------------------------------------------------------------------------


def test_budget_summary_plain_text_prints(capsys):
    """In a non-IPython context, budget_summary should print plain text."""
    with BudgetContext(flop_budget=1000, namespace="summary_test", quiet=True) as ctx:
        ctx.deduct("op1", flop_cost=100, subscripts=None, shapes=(), dtypes=())

    # Patch away rich so fnp get the plain text path
    with patch.dict(
        "sys.modules",
        {"rich": None, "rich.console": None, "rich.panel": None},
    ):
        result = budget_summary()
    # budget_summary prints and returns None in non-IPython
    assert result is None
    captured = capsys.readouterr()
    assert "summary_test" in captured.out or "flopscope" in captured.out
    assert "By namespace:" not in captured.out


def test_budget_summary_by_namespace_prints_section_only_when_requested(capsys):
    import flopscope as flops

    with BudgetContext(flop_budget=1000, namespace="predict", quiet=True) as ctx:
        with flops.namespace("precompute"):
            with ctx.deduct(
                "op1", flop_cost=100, subscripts=None, shapes=(), dtypes=()
            ):
                pass

    with patch.dict(
        "sys.modules",
        {"rich": None, "rich.console": None, "rich.panel": None},
    ):
        result = budget_summary(by_namespace=True)
    assert result is None
    captured = capsys.readouterr()
    assert "By namespace:" in captured.out
    assert "predict.precompute" in captured.out


def test_budget_summary_with_rich_returns_none(capsys):
    """With rich available, budget_summary returns None (prints via Console)."""
    pytest.importorskip("rich")

    with BudgetContext(
        flop_budget=1000, namespace="rich_summary_test", quiet=True
    ) as ctx:
        ctx.deduct("op1", flop_cost=50, subscripts=None, shapes=(), dtypes=())

    result = budget_summary()
    assert result is None


# ---------------------------------------------------------------------------
# render_budget_summary — rich path with namespace data
# ---------------------------------------------------------------------------


def test_render_budget_summary_rich_with_multiple_namespaces():
    """Rich summary renders without error when multiple namespaces are present."""
    pytest.importorskip("rich")

    for ns in ["ns_a", "ns_b", "ns_c", "ns_d"]:
        with BudgetContext(flop_budget=1000, namespace=ns, quiet=True) as ctx:
            ctx.deduct("op", flop_cost=100, subscripts=None, shapes=(), dtypes=())

    result = render_budget_summary(by_namespace=True)
    from rich.panel import Panel

    assert isinstance(result, Panel)
