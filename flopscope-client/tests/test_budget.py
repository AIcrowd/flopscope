"""Unit tests for BudgetContext and OpRecord.

All tests mock the connection — no server required.
"""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import msgpack
import pytest

import flopscope._budget as budget_module
from flopscope._budget import BudgetContext
from flopscope.errors import FlopscopeServerError
from tests.test_authoritative_budget_summary import (
    _canonical_summary,
    _summary_response,
)


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


def _open_mock_context(
    monkeypatch,
    *,
    namespace: str | None = None,
) -> tuple[BudgetContext, MagicMock]:
    conn = MagicMock()
    conn.send_recv.return_value = {"status": "ok"}
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)
    ctx = BudgetContext(100, namespace=namespace)
    ctx.__enter__()
    return ctx, conn


def _close_response(used: int) -> dict:
    return {
        "status": "ok",
        "result": {
            "budget_breakdown": _canonical_summary(flops_used=used, by_namespace=True),
            "budget_summary": "summary",
            "comms_summary": {},
        },
    }


def _two_invocation_responses(*, first_used: int, second_used: int) -> list[dict]:
    return [
        {"status": "ok"},
        _close_response(first_used),
        {"status": "ok"},
        _close_response(second_used),
    ]


def _fake_dispatch_clock(monkeypatch):
    clock = {"ns": 0}

    def advance(ns: int) -> None:
        clock["ns"] += ns

    monkeypatch.setattr(
        budget_module.time,
        "perf_counter_ns",
        lambda: clock["ns"],
    )
    return clock, advance


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

    def test_client_budget_context_has_no_multiplier(self):
        import flopscope as flops

        with pytest.raises(TypeError):
            flops.BudgetContext(flop_budget=1000, flop_multiplier=0.0)


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

    def test_enter_resets_flops_used_for_fresh_server_session(self):
        from flopscope._budget import BudgetContext

        mock_conn = _make_mock_conn({"status": "ok", "flops_used": 50})
        with patch("flopscope._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=1000)
            ctx._flops_used = 50
            ctx.__enter__()
            assert ctx.flops_used == 0

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
        close_resp = _close_response(75)
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
            _close_response(100),
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


def test_unentered_context_matches_core_empty_mapping_without_network() -> None:
    ctx = BudgetContext(100, namespace="phase")
    assert ctx.summary_dict(False) == {
        "flop_budget": 100,
        "flops_used": 0,
        "flops_remaining": 100,
        "operations": {},
        "wall_time_s": None,
        "flopscope_backend_time_s": 0.0,
        "flopscope_overhead_time_s": 0.0,
        "residual_wall_time_s": None,
    }
    assert ctx.summary_dict(True)["by_namespace"] == {}


def test_live_summary_preserves_property_none_semantics(monkeypatch) -> None:
    ctx, conn = _open_mock_context(monkeypatch)
    monkeypatch.setattr(
        ctx,
        "_normalize_context_timing",
        lambda summary, **_: summary,
    )
    conn.send_recv.return_value = _summary_response(
        flops_used=7,
        wall_time_s=2.0,
        residual_wall_time_s=1.0,
        by_namespace=False,
    )
    result = ctx.summary_dict(False)
    request = msgpack.unpackb(conn.send_recv.call_args.args[0], raw=False)
    assert request["kwargs"] == {
        "scope": "active_context",
        "by_namespace": False,
    }
    assert result["wall_time_s"] == 2.0
    assert ctx.wall_time_s is None
    assert ctx.residual_wall_time_s is None
    assert ctx.flops_used == 7


def test_live_summary_attributes_validation_and_copy_time_to_overhead(
    monkeypatch,
) -> None:
    clock, advance = _fake_dispatch_clock(monkeypatch)
    ctx, conn = _open_mock_context(monkeypatch)
    response = _summary_response(by_namespace=True)
    response["display_totals"]["client_context_compute_ns"] = 0
    conn.send_recv.return_value = response

    validate = budget_module._validate_summary_response
    copy_mapping = budget_module.deepcopy

    def timed_validate(*args, **kwargs):
        advance(100_000_000)
        return validate(*args, **kwargs)

    def timed_copy(value):
        advance(200_000_000)
        return copy_mapping(value)

    monkeypatch.setattr(budget_module, "_validate_summary_response", timed_validate)
    monkeypatch.setattr(budget_module, "deepcopy", timed_copy)

    summary = ctx.summary_dict(True)

    assert summary["wall_time_s"] == pytest.approx(0.7)
    assert summary["flopscope_overhead_time_s"] == pytest.approx(0.7)
    assert summary["residual_wall_time_s"] == 0.0
    assert budget_module.total_dispatch_ns() == clock["ns"]


def test_close_caches_authoritative_mapping_and_never_calls_network_again(
    monkeypatch,
) -> None:
    ctx, conn = _open_mock_context(monkeypatch)
    monkeypatch.setattr(
        ctx,
        "_normalize_context_timing",
        lambda summary, **_: summary,
    )
    close_mapping = _canonical_summary(flops_used=9, by_namespace=True)
    conn.send_recv.return_value = {
        "status": "ok",
        "result": {
            "budget_breakdown": close_mapping,
            "budget_summary": "summary",
            "comms_summary": {},
        },
    }
    ctx.__exit__(None, None, None)
    calls_after_close = conn.send_recv.call_count
    cached_with_namespaces = ctx.summary_dict(True)
    assert cached_with_namespaces == close_mapping
    cached_with_namespaces["operations"]["add"]["calls"] = 99
    assert ctx.summary_dict(True)["operations"]["add"]["calls"] == 1
    assert ctx.summary_dict(False) == {
        key: value for key, value in close_mapping.items() if key != "by_namespace"
    }
    assert conn.send_recv.call_count == calls_after_close


def test_failed_reopen_preserves_previous_closed_view(monkeypatch) -> None:
    ctx = BudgetContext(100)
    previous = _canonical_summary(flops_used=9, by_namespace=True)
    ctx._install_closed_summary(previous)
    conn = MagicMock()
    conn.send_recv.side_effect = RuntimeError("open failed")
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)

    with pytest.raises(RuntimeError, match="open failed"):
        ctx.__enter__()

    assert ctx.summary_dict(True) == previous
    assert ctx.flops_used == 9


def test_closed_mapping_properties_and_transport_overhead_share_one_view(
    monkeypatch,
) -> None:
    clock = iter([1_000_000_000, 6_000_000_000])
    dispatch = iter([0, 2_000_000_000])
    monkeypatch.setattr(budget_module.time, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(budget_module, "total_dispatch_ns", lambda: next(dispatch))
    monkeypatch.setattr(budget_module, "dispatch_span", nullcontext)
    ctx, conn = _open_mock_context(monkeypatch)
    response = _close_response(used=9)
    response["result"]["comms_summary"] = {"total_compute_time_ns": 1_000_000_000}
    conn.send_recv.return_value = response

    ctx.__exit__(None, None, None)
    summary = ctx.summary_dict()
    assert summary["wall_time_s"] == ctx.wall_time_s == 5.0
    assert summary["flopscope_backend_time_s"] == (ctx.flopscope_backend_time_s) == 1.0
    assert (
        summary["flopscope_overhead_time_s"] == (ctx.flopscope_overhead_time_s) == 1.0
    )
    assert summary["residual_wall_time_s"] == (ctx.residual_wall_time_s) == 3.0


def test_close_validation_copy_and_compute_metadata_are_overhead(monkeypatch) -> None:
    clock, advance = _fake_dispatch_clock(monkeypatch)
    ctx, conn = _open_mock_context(monkeypatch)
    response = _close_response(used=9)
    response["result"]["comms_summary"] = {"total_compute_time_ns": 0}
    conn.send_recv.return_value = response

    validate = budget_module._validated_close_summary
    extract = budget_module._extract_compute_ns
    copy_mapping = budget_module.deepcopy

    def timed_validate(value):
        advance(100_000_000)
        return validate(value)

    def timed_extract(value):
        advance(100_000_000)
        return extract(value)

    def timed_copy(value):
        advance(200_000_000)
        return copy_mapping(value)

    monkeypatch.setattr(budget_module, "_validated_close_summary", timed_validate)
    monkeypatch.setattr(budget_module, "_extract_compute_ns", timed_extract)
    monkeypatch.setattr(budget_module, "deepcopy", timed_copy)

    ctx.__exit__(None, None, None)

    summary = ctx.summary_dict()
    assert summary["wall_time_s"] == ctx.wall_time_s == pytest.approx(0.4)
    assert (
        summary["flopscope_overhead_time_s"]
        == ctx.flopscope_overhead_time_s
        == pytest.approx(0.4)
    )
    assert summary["residual_wall_time_s"] == ctx.residual_wall_time_s == 0.0
    assert budget_module.total_dispatch_ns() == clock["ns"]


@pytest.mark.parametrize("closed", [False, True])
def test_cached_mapping_copy_is_overhead_for_another_live_context(
    monkeypatch, closed
) -> None:
    clock, advance = _fake_dispatch_clock(monkeypatch)
    inspected = BudgetContext(100)
    if closed:
        inspected._closed_summary = _canonical_summary(flops_used=9, by_namespace=True)
    live = BudgetContext(100)
    live._wall_start_ns = 0
    live._dispatch_baseline_ns = 0
    monkeypatch.setattr(budget_module, "_active_context", live)

    copy_mapping = budget_module.deepcopy

    def timed_copy(value):
        advance(200_000_000)
        return copy_mapping(value)

    monkeypatch.setattr(budget_module, "deepcopy", timed_copy)
    inspected.summary_dict(True)
    monkeypatch.setattr(budget_module, "deepcopy", copy_mapping)

    normalized = live._normalize_context_timing(_canonical_summary(), kernel_ns=0)
    assert normalized["flopscope_overhead_time_s"] == pytest.approx(0.2)
    assert normalized["residual_wall_time_s"] == 0.0


def test_summary_text_formatting_is_overhead_for_live_context(monkeypatch) -> None:
    import flopscope._display as display_module

    _, advance = _fake_dispatch_clock(monkeypatch)
    ctx = BudgetContext(100)
    ctx._is_open = True
    ctx._wall_start_ns = 0
    ctx._dispatch_baseline_ns = 0
    monkeypatch.setattr(budget_module, "_active_context", ctx)
    monkeypatch.setattr(
        budget_module,
        "_request_budget_summary",
        lambda **_: (
            _canonical_summary(),
            {"client_context_compute_ns": 0},
        ),
    )

    def timed_format(*args, **kwargs):
        advance(300_000_000)
        return "formatted"

    monkeypatch.setattr(display_module, "_format_budget_summary_text", timed_format)

    assert ctx.summary() == "formatted"
    normalized = ctx._normalize_context_timing(_canonical_summary(), kernel_ns=0)
    assert normalized["flopscope_overhead_time_s"] == pytest.approx(0.3)
    assert normalized["residual_wall_time_s"] == 0.0


def test_malformed_acknowledged_close_cleans_up_and_context_is_reusable(
    monkeypatch,
) -> None:
    ctx = BudgetContext(100)
    conn = MagicMock()
    conn.send_recv.side_effect = [
        {"status": "ok"},
        {"status": "ok", "result": {}},
        {"status": "ok"},
        _close_response(used=5),
    ]
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)

    ctx.__enter__()
    with pytest.raises(FlopscopeServerError, match="malformed budget_summary response"):
        ctx.__exit__(None, None, None)
    assert ctx._is_open is False
    assert budget_module._active_context is None

    ctx.__enter__()
    ctx.__exit__(None, None, None)
    assert ctx.flops_used == 5


def test_unacknowledged_close_failure_remains_retryable(monkeypatch) -> None:
    ctx = BudgetContext(100)
    conn = MagicMock()
    conn.send_recv.side_effect = [
        {"status": "ok"},
        RuntimeError("close failed"),
        _close_response(used=5),
    ]
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)

    ctx.__enter__()
    with pytest.raises(RuntimeError, match="close failed"):
        ctx.__exit__(None, None, None)
    assert ctx._is_open is True
    assert budget_module._active_context is ctx

    ctx.__exit__(None, None, None)
    assert ctx._is_open is False
    assert budget_module._active_context is None


@pytest.mark.parametrize(
    "value",
    [None, True, "1", -1, 1.0, float("nan"), float("inf")],
)
def test_invalid_acknowledged_close_compute_time_cleans_up(monkeypatch, value) -> None:
    ctx = BudgetContext(100)
    response = _close_response(used=5)
    response["result"]["comms_summary"] = {"total_compute_time_ns": value}
    conn = MagicMock()
    conn.send_recv.side_effect = [{"status": "ok"}, response]
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)

    ctx.__enter__()
    with pytest.raises(
        FlopscopeServerError,
        match="compute time must be a non-negative integer",
    ):
        ctx.__exit__(None, None, None)

    assert ctx._is_open is False
    assert budget_module._active_context is None


def test_decorator_reuse_replaces_not_maxes_previous_invocation(monkeypatch) -> None:
    ctx = BudgetContext(100)
    responses = _two_invocation_responses(first_used=80, second_used=5)
    conn = MagicMock()
    conn.send_recv.side_effect = responses
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)

    @ctx
    def run():
        return None

    run()
    assert ctx.flops_used == 80
    run()
    assert ctx.flops_used == 5
    assert ctx.summary_dict()["flops_used"] == 5


def test_budget_open_transports_literal_root_namespace(monkeypatch) -> None:
    ctx, conn = _open_mock_context(monkeypatch, namespace="predict..raw")
    request = msgpack.unpackb(conn.send_recv.call_args_list[0].args[0], raw=False)
    assert request["kwargs"]["namespace"] == "predict..raw"


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

    @pytest.mark.parametrize(
        "value",
        [None, True, "1", -1, 1.0, float("nan"), float("inf")],
    )
    def test_present_compute_time_must_be_exact_nonnegative_int(self, value):
        from flopscope._budget import _extract_compute_ns

        response = {"result": {"comms_summary": {"total_compute_time_ns": value}}}
        with pytest.raises(
            FlopscopeServerError,
            match="compute time must be a non-negative integer",
        ):
            _extract_compute_ns(response)


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
        assert ctx.flopscope_backend_time_s == 0.0
        assert ctx.flopscope_overhead_time_s == 0.0
        assert ctx.residual_wall_time_s is None

    def test_evaluator_getattr_contract(self):
        from flopscope._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        # exactly how whestbench-evaluator/_child_entry.py reads them
        assert float(getattr(ctx, "flopscope_backend_time_s", 0.0)) == 0.0
        assert float(getattr(ctx, "flopscope_overhead_time_s", 0.0)) == 0.0
        assert getattr(ctx, "wall_time_s", None) is None
        assert getattr(ctx, "residual_wall_time_s", None) is None
