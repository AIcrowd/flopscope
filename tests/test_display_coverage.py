"""Tests for flopscope._display to increase coverage to ~95%."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from flopscope._budget import BudgetContext
from flopscope._display import (
    _display_totals,
    _format_flops,
    _is_global_default_ns,
    _pct,
    _plain_text_summary,
    _PlainTextLive,
    _usage_color,
    budget_live,
    budget_summary,
    render_budget_summary,
)

# ---------------------------------------------------------------------------
# Helper: create a budget with some operations recorded in the accumulator
# ---------------------------------------------------------------------------


def _make_budget(flop_budget=10_000, namespace=None, ops=None):
    """Enter and exit a BudgetContext, deducting ops so the accumulator records it."""
    ops = ops or [("matmul", 500), ("add", 100)]
    with BudgetContext(flop_budget=flop_budget, quiet=True, namespace=namespace) as b:
        for op_name, cost in ops:
            b.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(), dtypes=())


# ======================================================================
# _format_flops
# ======================================================================


class TestFormatFlops:
    def test_zero(self):
        assert _format_flops(0) == "0"

    def test_small(self):
        assert _format_flops(42) == "42"

    def test_thousands(self):
        assert _format_flops(1_234_567) == "1,234,567"


# ======================================================================
# _pct
# ======================================================================


class TestPct:
    def test_zero_total(self):
        assert _pct(100, 0) == "0.0%"

    def test_zero_used(self):
        assert _pct(0, 1000) == "0.0%"

    def test_half(self):
        assert _pct(500, 1000) == "50.0%"

    def test_full(self):
        assert _pct(1000, 1000) == "100.0%"


# ======================================================================
# _usage_color
# ======================================================================


class TestUsageColor:
    def test_zero_budget(self):
        assert _usage_color(0, 0) == "white"

    def test_low_usage(self):
        # ratio < 0.5 => green
        assert _usage_color(10, 100) == "green"

    def test_mid_usage(self):
        # 0.5 <= ratio < 0.8 => yellow
        assert _usage_color(50, 100) == "yellow"
        assert _usage_color(79, 100) == "yellow"

    def test_high_usage(self):
        # ratio >= 0.8 => red
        assert _usage_color(80, 100) == "red"
        assert _usage_color(100, 100) == "red"

    def test_boundary_half(self):
        # exactly 0.5 => yellow (not green)
        assert _usage_color(50, 100) == "yellow"

    def test_boundary_eighty(self):
        # exactly 0.8 => red (not yellow)
        assert _usage_color(80, 100) == "red"


# ======================================================================
# _is_global_default_ns
# ======================================================================


class TestIsGlobalDefaultNs:
    def test_none_ns_big_budget(self):
        assert _is_global_default_ns(None, {"flop_budget": int(1e15)}) is True

    def test_none_ns_small_budget(self):
        assert _is_global_default_ns(None, {"flop_budget": 1000}) is False

    def test_named_ns(self):
        assert _is_global_default_ns("training", {"flop_budget": int(1e15)}) is False

    def test_no_budget_key(self):
        assert _is_global_default_ns(None, {}) is False


# ======================================================================
# _plain_text_summary
# ======================================================================


class TestPlainTextSummary:
    def test_no_data(self):
        result = _plain_text_summary()
        assert result == "No budget data recorded yet."

    def test_single_namespace(self):
        _make_budget(flop_budget=10_000, ops=[("matmul", 3000), ("add", 1000)])
        result = _plain_text_summary()
        assert "flopscope FLOP Budget Summary" in result
        assert "10,000" in result
        assert "4,000" in result
        assert "matmul" in result
        assert "add" in result

    def test_multiple_namespaces(self):
        _make_budget(flop_budget=5000, namespace="train", ops=[("matmul", 2000)])
        _make_budget(flop_budget=3000, namespace="eval", ops=[("add", 500)])
        result = _plain_text_summary(by_namespace=True)
        assert "By namespace:" in result
        assert "train" in result
        assert "eval" in result

    def test_default_namespace_label(self):
        _make_budget(flop_budget=5000, namespace=None, ops=[("matmul", 100)])
        result = _plain_text_summary(by_namespace=True)
        assert "(unlabeled)" in result

    def test_single_call_word(self):
        _make_budget(flop_budget=5000, ops=[("matmul", 100)])
        result = _plain_text_summary()
        # "matmul" deducted once => "1 call"
        assert "1 call]" in result

    def test_multiple_calls_word(self):
        _make_budget(flop_budget=5000, ops=[("matmul", 100), ("matmul", 200)])
        result = _plain_text_summary()
        assert "2 calls]" in result

    def test_all_operations_section(self):
        _make_budget(flop_budget=5000, ops=[("matmul", 100)])
        result = _plain_text_summary()
        assert "  By operation:" in result


# ======================================================================
# _rich_namespace_table
# ======================================================================


class TestRichNamespaceTable:
    def test_returns_table(self):
        pytest.importorskip("rich")
        from flopscope._display import _rich_namespace_table

        ns_data = {
            "flop_budget": 10_000,
            "flops_used": 3000,
            "operations": {
                "matmul": {"flop_cost": 2000, "calls": 5},
                "add": {"flop_cost": 1000, "calls": 1},
            },
        }
        from rich.table import Table

        table = _rich_namespace_table("train", ns_data, "green")
        assert isinstance(table, Table)

    def test_empty_ops(self):
        pytest.importorskip("rich")
        from rich.table import Table

        from flopscope._display import _rich_namespace_table

        ns_data = {
            "flop_budget": 5000,
            "flops_used": 0,
            "operations": {},
        }
        table = _rich_namespace_table("empty", ns_data, "white")
        assert isinstance(table, Table)

    def test_single_call_text(self):
        """Operations with calls==1 should show '1 call', not '1 calls'."""
        pytest.importorskip("rich")
        from flopscope._display import _rich_namespace_table

        ns_data = {
            "flop_budget": 5000,
            "flops_used": 100,
            "operations": {"exp": {"flop_cost": 100, "calls": 1}},
        }
        table = _rich_namespace_table("ns", ns_data, "green")
        # Render table to string to check content
        import io

        from rich.console import Console

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        console.print(table)
        output = buf.getvalue()
        assert "1 call" in output

    def test_multiple_calls_text(self):
        pytest.importorskip("rich")
        from flopscope._display import _rich_namespace_table

        ns_data = {
            "flop_budget": 5000,
            "flops_used": 300,
            "operations": {"matmul": {"flop_cost": 300, "calls": 3}},
        }
        table = _rich_namespace_table("ns", ns_data, "yellow")
        import io

        from rich.console import Console

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        console.print(table)
        output = buf.getvalue()
        assert "3 calls" in output


# ======================================================================
# _rich_summary
# ======================================================================


class TestRichSummary:
    def test_display_totals_exclude_implicit_global_default_budget(self):
        from flopscope._budget import _get_global_default

        _make_budget(flop_budget=5000, namespace="train", ops=[("matmul", 1000)])
        global_ctx = _get_global_default()
        global_ctx.deduct("add", flop_cost=10, subscripts=None, shapes=(), dtypes=())

        totals = _display_totals()
        assert totals["has_explicit_budget"] is True
        assert totals["budget"] == 5000
        assert totals["used"] == 1010
        assert totals["remaining"] == 3990

    def test_no_data_panel(self):
        pytest.importorskip("rich")
        from rich.panel import Panel

        from flopscope._display import _rich_summary

        result = _rich_summary()
        assert isinstance(result, Panel)
        # Render and check "No budget data" message
        import io

        from rich.console import Console

        buf = io.StringIO()
        Console(file=buf, force_terminal=True, width=120).print(result)
        assert "No budget data" in buf.getvalue()

    def test_single_namespace_panel(self):
        pytest.importorskip("rich")
        from rich.panel import Panel

        from flopscope._display import _rich_summary

        _make_budget(flop_budget=10_000, namespace="train", ops=[("matmul", 3000)])
        result = _rich_summary(by_namespace=True)
        assert isinstance(result, Panel)

    def test_multi_namespace_columns(self):
        """Multiple namespace buckets render in the attribution table."""
        pytest.importorskip("rich")
        from rich.panel import Panel

        from flopscope._display import _rich_summary

        _make_budget(flop_budget=5000, namespace="a", ops=[("matmul", 100)])
        _make_budget(flop_budget=5000, namespace="b", ops=[("add", 200)])
        result = _rich_summary(by_namespace=True)
        assert isinstance(result, Panel)

    def test_more_than_3_namespaces_no_columns(self):
        """More than 3 namespace buckets still render in one summary panel."""
        pytest.importorskip("rich")
        from rich.panel import Panel

        from flopscope._display import _rich_summary

        for ns in ["a", "b", "c", "d"]:
            _make_budget(flop_budget=3000, namespace=ns, ops=[("matmul", 100)])
        result = _rich_summary(by_namespace=True)
        assert isinstance(result, Panel)

    def test_global_default_ns_excluded_from_budget(self):
        """Namespace rendering uses attribution buckets rather than pseudo-budgets."""
        pytest.importorskip("rich")
        from flopscope._budget import _get_global_default
        from flopscope._display import _rich_summary

        # A named explicit namespace
        _make_budget(flop_budget=5000, namespace="train", ops=[("matmul", 1000)])
        global_ctx = _get_global_default()
        global_ctx.deduct("add", flop_cost=10, subscripts=None, shapes=(), dtypes=())
        result = _rich_summary(by_namespace=True)
        import io

        from rich.console import Console

        buf = io.StringIO()
        Console(file=buf, force_terminal=True, width=120).print(result)
        output = buf.getvalue()
        assert "By namespace" in output
        assert "(unlabeled)" in output

    def test_unscoped_label(self):
        """None namespace should render as '(unlabeled)' in Rich output."""
        pytest.importorskip("rich")
        from flopscope._display import _rich_summary

        _make_budget(flop_budget=5000, namespace=None, ops=[("matmul", 100)])
        import io

        from rich.console import Console

        buf = io.StringIO()
        Console(file=buf, force_terminal=True, width=120).print(
            _rich_summary(by_namespace=True)
        )
        assert "(unlabeled)" in buf.getvalue()

    def test_no_explicit_budget(self):
        """A pure default-budget session still renders as a Rich panel."""
        pytest.importorskip("rich")
        from rich.panel import Panel

        from flopscope._budget import _get_global_default
        from flopscope._display import _rich_summary

        global_ctx = _get_global_default()
        global_ctx.deduct(
            "matmul", flop_cost=100, subscripts=None, shapes=(), dtypes=()
        )
        result = _rich_summary(by_namespace=True)
        assert isinstance(result, Panel)


# ======================================================================
# render_budget_summary
# ======================================================================


class TestRenderBudgetSummary:
    def test_rich_path(self):
        """When Rich is installed, returns a Rich Panel."""
        pytest.importorskip("rich")
        from rich.panel import Panel

        _make_budget(flop_budget=5000, ops=[("matmul", 100)])
        result = render_budget_summary()
        assert isinstance(result, Panel)

    def test_no_rich_fallback(self):
        """When Rich is not installed, returns plain text."""
        _make_budget(flop_budget=5000, ops=[("matmul", 100)])
        with patch.dict(sys.modules, {"rich": None}):
            # Force ImportError path
            import flopscope._display as disp_mod

            # Directly test the fallback by simulating ImportError
            with patch.object(disp_mod, "_rich_summary", side_effect=ImportError):
                pass
        # Test via the function itself with import mocked out
        result = render_budget_summary()
        # With Rich actually installed, this returns a Panel; test the other branch:
        from flopscope._display import render_budget_summary as rbs

        # Patch 'rich' import to raise ImportError
        orig_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "rich":
                raise ImportError("no rich")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = rbs()
        assert isinstance(result, str)
        assert "flopscope FLOP Budget Summary" in result


# ======================================================================
# _PlainTextLive
# ======================================================================


class TestPlainTextLive:
    def test_context_manager(self, capsys):
        _make_budget(flop_budget=5000, ops=[("matmul", 100)])
        live = _PlainTextLive()
        with live:
            pass
        captured = capsys.readouterr()
        assert "flopscope FLOP Budget Summary" in captured.out

    def test_enter_returns_self(self):
        live = _PlainTextLive()
        result = live.__enter__()
        assert result is live
        live.__exit__(None, None, None)

    def test_exit_returns_none(self, capsys):
        live = _PlainTextLive()
        live.__enter__()
        result = live.__exit__(None, None, None)
        assert result is None


# ======================================================================
# budget_live
# ======================================================================


class TestBudgetLive:
    def test_rich_path_context_manager(self):
        """With Rich installed, budget_live returns a _RichBudgetLive."""
        pytest.importorskip("rich")
        _make_budget(flop_budget=5000, ops=[("matmul", 100)])
        live = budget_live()
        # Should have __enter__ and __exit__
        assert hasattr(live, "__enter__")
        assert hasattr(live, "__exit__")
        # Class name should be _RichBudgetLive
        assert "RichBudgetLive" in type(live).__name__

    def test_rich_path_enter_exit(self):
        """_RichBudgetLive enters and exits the Rich Live context."""
        pytest.importorskip("rich")
        _make_budget(flop_budget=5000, ops=[("matmul", 100)])
        live = budget_live()
        # Patch the Rich Live to avoid actual terminal output
        with patch("rich.live.Live.start"):
            with patch("rich.live.Live.stop"):
                with patch("rich.live.Live.__enter__", return_value=MagicMock()):
                    with patch("rich.live.Live.__exit__", return_value=None):
                        ctx = live.__enter__()
                        assert ctx is live
                        result = live.__exit__(None, None, None)
                        assert result is None

    def test_fallback_path(self, capsys):
        """When Rich.live is not importable, falls back to _PlainTextLive."""
        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "rich.live":
                raise ImportError("no rich.live")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            live = budget_live()
        assert isinstance(live, _PlainTextLive)

    def test_rich_live_full_cycle(self, capsys):
        """Full enter/exit cycle with Rich Live mocked."""
        pytest.importorskip("rich")
        _make_budget(flop_budget=5000, ops=[("matmul", 100)])

        mock_live_instance = MagicMock()
        mock_live_instance.__enter__ = MagicMock(return_value=mock_live_instance)
        mock_live_instance.__exit__ = MagicMock(return_value=None)

        with patch("rich.live.Live", return_value=mock_live_instance):
            live = budget_live()
            with live:
                pass
        # The mock Live's __enter__ and __exit__ should have been called
        mock_live_instance.__enter__.assert_called_once()
        mock_live_instance.__exit__.assert_called_once()

    def test_rich_live_exit_updates(self):
        """_RichBudgetLive.__exit__ should call update before closing."""
        pytest.importorskip("rich")
        _make_budget(flop_budget=5000, ops=[("matmul", 100)])

        mock_live_instance = MagicMock()
        mock_live_instance.__enter__ = MagicMock(return_value=mock_live_instance)
        mock_live_instance.__exit__ = MagicMock(return_value=None)

        with patch("rich.live.Live", return_value=mock_live_instance):
            live = budget_live()
            live.__enter__()
            live.__exit__(None, None, None)
        # update() called with the final rich summary
        mock_live_instance.update.assert_called_once()


# ======================================================================
# budget_summary
# ======================================================================


class TestBudgetSummary:
    def test_console_plain_text(self, capsys):
        """In a regular console (no IPython), should print and return None."""
        _make_budget(flop_budget=5000, ops=[("matmul", 100)])
        # Ensure get_ipython is not defined (standard console)
        with patch.dict(sys.modules, {}):
            result = budget_summary()
        # Should print via Rich console (since Rich is installed) and return None
        assert result is None

    def test_console_plain_text_no_rich(self, capsys):
        """When Rich is not installed and in console mode, prints plain text."""
        _make_budget(flop_budget=5000, ops=[("matmul", 100)])
        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "rich" or name.startswith("rich."):
                raise ImportError("no rich")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = budget_summary()
        assert result is None
        captured = capsys.readouterr()
        assert "flopscope FLOP Budget Summary" in captured.out

    def test_jupyter_returns_renderable(self):
        """In a Jupyter environment, should return the renderable without printing."""
        _make_budget(flop_budget=5000, ops=[("matmul", 100)])
        # Simulate Jupyter by making get_ipython available in builtins
        import builtins

        original = getattr(builtins, "get_ipython", None)
        try:
            builtins.get_ipython = lambda: MagicMock()  # pyright: ignore[reportAttributeAccessIssue]
            result = budget_summary()
            # Should return the renderable (Panel) without printing
            assert result is not None
        finally:
            if original is None:
                if hasattr(builtins, "get_ipython"):
                    delattr(builtins, "get_ipython")
            else:
                builtins.get_ipython = original  # pyright: ignore[reportAttributeAccessIssue]

    def test_console_with_rich_prints_via_console(self, capsys):
        """When Rich is installed and result is a Panel, uses Console().print()."""
        _make_budget(flop_budget=5000, ops=[("matmul", 100)])
        # Make sure get_ipython raises NameError (console mode)
        result = budget_summary()
        assert result is None
        # Something should have been printed
        captured = capsys.readouterr()
        # Rich Console output goes to stdout
        assert len(captured.out) > 0
