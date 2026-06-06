"""Unit tests for BudgetContext and OpRecord.

All tests mock the connection — no server required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import msgpack
import pytest


@pytest.fixture(autouse=True)
def _reset_active_context():
    """Reset the module-level _active_context guard between tests."""
    import flopscope._budget as bmod

    old = bmod._active_context
    bmod._active_context = None
    yield
    bmod._active_context = old


@pytest.fixture(autouse=True)
def _reset_dispatch():
    import flopscope._dispatch as _d

    _d.reset_dispatch()
    yield
    _d.reset_dispatch()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pack_response(payload: dict) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def _make_mock_conn(response: dict) -> MagicMock:
    """Return a mock Connection whose send_recv always returns *response*."""
    conn = MagicMock()
    conn.send_recv.return_value = response
    return conn


# ---------------------------------------------------------------------------
# OpRecord
# ---------------------------------------------------------------------------


class TestOpRecord:
    """OpRecord stores op metadata and is accessible via attributes."""

    def test_op_name(self):
        from flopscope._budget import OpRecord

        rec = OpRecord(op_name="dot", flop_cost=100, cumulative=500)
        assert rec.op_name == "dot"

    def test_flop_cost(self):
        from flopscope._budget import OpRecord

        rec = OpRecord(op_name="matmul", flop_cost=2000, cumulative=3000)
        assert rec.flop_cost == 2000

    def test_cumulative(self):
        from flopscope._budget import OpRecord

        rec = OpRecord(op_name="add", flop_cost=10, cumulative=110)
        assert rec.cumulative == 110


# ---------------------------------------------------------------------------
# BudgetContext – attribute defaults
# ---------------------------------------------------------------------------


class TestBudgetContextAttributes:
    """BudgetContext stores parameters without connecting."""

    def test_flop_budget_stored(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        assert ctx.flop_budget == 1000

    def test_flop_multiplier_default(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=500)
        assert ctx.flop_multiplier == 1.0

    def test_flop_multiplier_custom(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=500, flop_multiplier=2.5)
        assert ctx.flop_multiplier == 2.5

    def test_flops_used_starts_zero(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        assert ctx.flops_used == 0

    def test_flops_remaining_equals_budget_minus_used(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        assert ctx.flops_remaining == 1000

    def test_quiet_default_false(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=100)
        assert ctx.quiet is False

    def test_quiet_custom(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=100, quiet=True)
        assert ctx.quiet is True


# ---------------------------------------------------------------------------
# BudgetContext – _update_budget
# ---------------------------------------------------------------------------


class TestUpdateBudget:
    """_update_budget patches local flops_used from a server-response dict."""

    def test_update_flops_used(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        ctx._update_budget({"flops_used": 300})
        assert ctx.flops_used == 300

    def test_flops_remaining_after_update(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        ctx._update_budget({"flops_used": 400})
        assert ctx.flops_remaining == 600

    def test_update_ignores_missing_key(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        ctx._update_budget({})  # no flops_used key — should not raise
        assert ctx.flops_used == 0

    def test_update_multiple_times(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        ctx._update_budget({"flops_used": 100})
        ctx._update_budget({"flops_used": 250})
        assert ctx.flops_used == 250
        assert ctx.flops_remaining == 750


# ---------------------------------------------------------------------------
# BudgetContext – context manager (__enter__ / __exit__)
# ---------------------------------------------------------------------------


class TestBudgetContextManager:
    """__enter__ sends budget_open; __exit__ sends budget_close."""

    def test_enter_sends_budget_open(self):

        from flopscope._budget import BudgetContext

        mock_conn = _make_mock_conn({"status": "ok", "flops_used": 0})
        with patch("flopscope._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=500)
            result = ctx.__enter__()
            assert result is ctx
            mock_conn.send_recv.assert_called_once()
            # Verify the payload encodes budget_open
            sent_bytes = mock_conn.send_recv.call_args[0][0]
            decoded = msgpack.unpackb(sent_bytes, raw=False)
            assert decoded["op"] == "budget_open"
            assert decoded["kwargs"]["flop_budget"] == 500

    def test_enter_updates_flops_used_from_response(self):
        from flopscope._budget import BudgetContext

        mock_conn = _make_mock_conn({"status": "ok", "flops_used": 50})
        with patch("flopscope._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=1000)
            ctx.__enter__()
            assert ctx.flops_used == 50

    def test_enter_returns_self(self):
        from flopscope._budget import BudgetContext

        mock_conn = _make_mock_conn({"status": "ok", "flops_used": 0})
        with patch("flopscope._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=200)
            returned = ctx.__enter__()
            assert returned is ctx

    def test_exit_sends_budget_close(self):
        from flopscope._budget import BudgetContext

        open_conn = _make_mock_conn({"status": "ok", "flops_used": 0})
        close_resp = {"status": "ok", "flops_used": 75}
        open_conn.send_recv.side_effect = [
            {"status": "ok", "flops_used": 0},
            close_resp,
        ]
        with patch("flopscope._budget.get_connection", return_value=open_conn):
            ctx = BudgetContext(flop_budget=200)
            ctx.__enter__()
            ctx.__exit__(None, None, None)
            # Second call should be budget_close
            assert open_conn.send_recv.call_count == 2
            close_bytes = open_conn.send_recv.call_args_list[1][0][0]
            decoded = msgpack.unpackb(close_bytes, raw=False)
            assert decoded["op"] == "budget_close"

    def test_context_manager_with_statement(self):
        from flopscope._budget import BudgetContext

        responses = [
            {"status": "ok", "flops_used": 0},  # budget_open
            {"status": "ok", "flops_used": 100},  # budget_close
        ]
        mock_conn = MagicMock()
        mock_conn.send_recv.side_effect = responses

        with patch("flopscope._budget.get_connection", return_value=mock_conn):
            with BudgetContext(flop_budget=1000) as ctx:
                assert isinstance(ctx, BudgetContext)
            assert mock_conn.send_recv.call_count == 2


# ---------------------------------------------------------------------------
# BudgetContext – summary
# ---------------------------------------------------------------------------


class TestBudgetContextSummary:
    """summary() sends budget_status, updates cache, returns a string."""

    def test_summary_sends_budget_status(self):
        from flopscope._budget import BudgetContext

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "flops_used": 200,
                "flop_budget": 1000,
            }
        )
        with patch("flopscope._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=1000)
            ctx.summary()
            sent_bytes = mock_conn.send_recv.call_args[0][0]
            decoded = msgpack.unpackb(sent_bytes, raw=False)
            assert decoded["op"] == "budget_status"

    def test_summary_updates_flops_used(self):
        from flopscope._budget import BudgetContext

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "result": {"flops_used": 350, "flop_budget": 1000},
            }
        )
        with patch("flopscope._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=1000)
            ctx.summary()
            assert ctx.flops_used == 350

    def test_summary_returns_string(self):
        from flopscope._budget import BudgetContext

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "result": {"flops_used": 100, "flop_budget": 500},
            }
        )
        with patch("flopscope._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=500)
            result = ctx.summary()
            assert isinstance(result, str)

    def test_summary_contains_budget_info(self):
        from flopscope._budget import BudgetContext

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "flops_used": 100,
                "flop_budget": 500,
            }
        )
        with patch("flopscope._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=500)
            result = ctx.summary()
            # Should contain numbers related to budget usage
            assert "100" in result or "500" in result


class TestDecomposeTiming:
    """_decompose_timing splits wall into (wall, backend, overhead, residual)."""

    def test_identity_normal(self):
        from flopscope._budget import _decompose_timing

        # wall=1.0s, dispatch=0.6s, backend(kernel)=0.4s
        wall, backend, overhead, residual = _decompose_timing(
            wall_ns=1_000_000_000, dispatch_ns=600_000_000, kernel_ns=400_000_000
        )
        assert backend == pytest.approx(0.4)
        assert overhead == pytest.approx(0.2)  # 0.6 - 0.4
        assert residual == pytest.approx(0.4)  # 1.0 - 0.6
        assert wall == pytest.approx(backend + overhead + residual)

    def test_clamps_overhead_when_kernel_exceeds_dispatch(self):
        from flopscope._budget import _decompose_timing

        wall, backend, overhead, residual = _decompose_timing(
            wall_ns=1_000_000_000, dispatch_ns=300_000_000, kernel_ns=500_000_000
        )
        assert overhead == 0.0  # max(0, 0.3 - 0.5)
        assert backend == pytest.approx(0.5)
        assert residual == pytest.approx(0.7)  # max(0, 1.0 - 0.3)

    def test_clamps_residual_when_dispatch_exceeds_wall(self):
        from flopscope._budget import _decompose_timing

        wall, backend, overhead, residual = _decompose_timing(
            wall_ns=100_000_000, dispatch_ns=500_000_000, kernel_ns=300_000_000
        )
        assert residual == 0.0  # max(0, 0.1 - 0.5)
        assert backend == pytest.approx(0.3)
        assert overhead == pytest.approx(0.2)  # 0.5 - 0.3

    def test_empty_context(self):
        from flopscope._budget import _decompose_timing

        wall, backend, overhead, residual = _decompose_timing(
            wall_ns=500_000_000, dispatch_ns=0, kernel_ns=0
        )
        assert backend == 0.0
        assert overhead == 0.0
        assert residual == pytest.approx(0.5)


class TestExtractComputeNs:
    """_extract_compute_ns pulls server compute time out of a close response."""

    def test_full_response(self):
        from flopscope._budget import _extract_compute_ns

        resp = {"result": {"comms_summary": {"total_compute_time_ns": 12345}}}
        assert _extract_compute_ns(resp) == 12345

    def test_missing_comms_summary(self):
        from flopscope._budget import _extract_compute_ns

        assert _extract_compute_ns({"result": {}}) == 0

    def test_missing_result(self):
        from flopscope._budget import _extract_compute_ns

        assert _extract_compute_ns({"status": "ok"}) == 0

    def test_non_dict(self):
        from flopscope._budget import _extract_compute_ns

        assert _extract_compute_ns(None) == 0


class TestBudgetContextTimingProperties:
    """The proxy BudgetContext exposes the four timing properties.

    ``test_properties_exist_with_defaults`` is the regression canary for the
    production bug where the proxy had NO timing attributes: it uses direct
    attribute access, which raises AttributeError if a property is missing.
    ``test_evaluator_getattr_contract`` documents the evaluator's exact getattr
    read pattern (it does not, on its own, catch a missing attribute).
    """

    def test_properties_exist_with_defaults(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)  # not entered
        assert ctx.wall_time_s is None
        assert ctx.flopscope_backend_time == 0.0
        assert ctx.flopscope_overhead_time == 0.0
        assert ctx.residual_wall_time is None

    def test_evaluator_getattr_contract(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        # exactly how whestbench-evaluator/_child_entry.py reads them
        assert float(getattr(ctx, "flopscope_backend_time", 0.0)) == 0.0
        assert float(getattr(ctx, "flopscope_overhead_time", 0.0)) == 0.0
        assert getattr(ctx, "wall_time_s", None) is None
        assert getattr(ctx, "residual_wall_time", None) is None
