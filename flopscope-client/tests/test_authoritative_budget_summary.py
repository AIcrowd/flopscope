"""Tests for authoritative client-side budget summary requests."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import MagicMock

import msgpack
import pytest
from flopscope._protocol import (
    AUTHORITATIVE_BUDGET_SUMMARY_CAPABILITY,
    encode_budget_summary,
)

import flopscope._budget as budget_module
from flopscope.errors import FlopscopeServerError


def _canonical_summary(
    *,
    flops_used: int = 7,
    by_namespace: bool = False,
    wall_time_s: float | None = 1.0,
    residual_wall_time_s: float | None = 0.7,
) -> dict:
    result = {
        "flop_budget": 100,
        "flops_used": flops_used,
        "flops_remaining": 100 - flops_used,
        "operations": {
            "add": {
                "flop_cost": flops_used,
                "calls": 1,
                "flopscope_backend_time_s": 0.2,
                "flopscope_overhead_time_s": 0.1,
            }
        },
        "wall_time_s": wall_time_s,
        "flopscope_backend_time_s": 0.2,
        "flopscope_overhead_time_s": 0.1,
        "residual_wall_time_s": residual_wall_time_s,
    }
    if by_namespace:
        result["by_namespace"] = {
            "phase": {
                "flops_used": flops_used,
                "calls": 1,
                "flopscope_backend_time_s": 0.2,
                "flopscope_overhead_time_s": 0.1,
                "operations": deepcopy(result["operations"]),
            }
        }
    return result


def _display_totals(used: int = 7) -> dict:
    return {
        "has_explicit_budget": True,
        "budget": 100,
        "used": used,
        "client_context_compute_ns": 100_000,
    }


def _summary_response(
    *,
    flops_used: int = 7,
    by_namespace: bool = False,
    wall_time_s: float | None = 1.0,
    residual_wall_time_s: float | None = 0.7,
) -> dict:
    return {
        "status": "ok",
        "result": _canonical_summary(
            flops_used=flops_used,
            by_namespace=by_namespace,
            wall_time_s=wall_time_s,
            residual_wall_time_s=residual_wall_time_s,
        ),
        "display_totals": _display_totals(flops_used),
        "comms_overhead_ns": 0,
    }


def test_encode_budget_summary_preserves_requested_namespace_flag() -> None:
    for value in (False, True):
        decoded = msgpack.unpackb(
            encode_budget_summary("session", by_namespace=value), raw=False
        )
        assert decoded == {
            "op": "budget_summary",
            "args": None,
            "kwargs": {"scope": "session", "by_namespace": value},
        }


def test_global_summary_uses_server_result_without_local_merge(monkeypatch) -> None:
    response = _summary_response(flops_used=7, by_namespace=False)
    conn = MagicMock()
    conn.send_recv.return_value = response
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)
    result = budget_module.budget_summary_dict(False)
    request = msgpack.unpackb(conn.send_recv.call_args.args[0], raw=False)
    conn.require_capability.assert_called_once_with(
        AUTHORITATIVE_BUDGET_SUMMARY_CAPABILITY
    )
    conn.send_recv.assert_called_once()
    assert request["kwargs"] == {"scope": "session", "by_namespace": False}
    assert result["flops_used"] == 7


def test_global_summary_does_not_fall_back_without_capability(monkeypatch) -> None:
    conn = MagicMock()
    conn.require_capability.side_effect = FlopscopeServerError("unsupported")
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)

    with pytest.raises(FlopscopeServerError, match="unsupported"):
        budget_module.budget_summary_dict(False)

    conn.send_recv.assert_not_called()


def test_returned_mapping_is_a_deep_defensive_copy(monkeypatch) -> None:
    response = _summary_response(flops_used=7, by_namespace=True)
    conn = MagicMock()
    conn.send_recv.return_value = response
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)
    first = budget_module.budget_summary_dict(True)
    first["operations"]["add"]["calls"] = 999
    second = budget_module.budget_summary_dict(True)
    assert second["operations"]["add"]["calls"] == 1


def test_private_summary_helper_defensively_copies_display_metadata(
    monkeypatch,
) -> None:
    response = _summary_response()
    conn = MagicMock()
    conn.send_recv.return_value = response
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)

    summary, display_totals = budget_module._request_budget_summary(
        scope="session", by_namespace=False
    )
    summary["flops_used"] = 99
    display_totals["used"] = 99

    assert response["result"]["flops_used"] == 7
    assert response["display_totals"]["used"] == 7


def test_complete_connection_envelope_is_accepted(monkeypatch) -> None:
    response = _summary_response()
    response.update(
        {
            "budget": {
                "flop_budget": 100,
                "flops_used": 7,
                "flops_remaining": 93,
            },
            "_round_trip_ns": 10,
            "_request_bytes": 20,
            "_response_bytes": 30,
        }
    )
    conn = MagicMock()
    conn.send_recv.return_value = response
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)

    assert budget_module.budget_summary_dict(False)["flops_used"] == 7


@pytest.mark.parametrize(
    "budget_info",
    [
        {},
        {"flops_used": True},
        {"flops_used": "9"},
        {"flops_used": -1},
        {"flops_used": 1.5},
        {"flops_used": []},
    ],
)
def test_active_budget_sync_ignores_invalid_flops_used(
    monkeypatch, budget_info
) -> None:
    ctx = budget_module.BudgetContext(flop_budget=100)
    monkeypatch.setattr(budget_module, "_active_context", ctx)

    budget_module._sync_active_context_from_response({"budget": budget_info})

    assert ctx.flops_used == 0


def test_active_budget_sync_preserves_monotonic_valid_updates(monkeypatch) -> None:
    ctx = budget_module.BudgetContext(flop_budget=100)
    monkeypatch.setattr(budget_module, "_active_context", ctx)

    budget_module._sync_active_context_from_response({"budget": {"flops_used": 12}})
    budget_module._sync_active_context_from_response({"budget": {"flops_used": 9}})

    assert ctx.flops_used == 12


def test_malformed_summary_budget_reaches_envelope_validator(monkeypatch) -> None:
    response = _summary_response()
    response["budget"] = {
        "flop_budget": 100,
        "flops_used": "bad",
        "flops_remaining": 93,
    }
    ctx = budget_module.BudgetContext(flop_budget=100)
    monkeypatch.setattr(budget_module, "_active_context", ctx)
    conn = MagicMock()

    def send_recv(_request):
        budget_module._sync_active_context_from_response(response)
        return response

    conn.send_recv.side_effect = send_recv
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)

    with pytest.raises(
        FlopscopeServerError,
        match="malformed budget_summary response: budget metadata",
    ):
        budget_module.budget_summary_dict(False)

    assert ctx.flops_used == 0


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {"status": "ok"},
        {
            **_summary_response(),
            "status": "unexpected",
        },
        {
            key: value
            for key, value in _summary_response().items()
            if key != "comms_overhead_ns"
        },
        {
            **_summary_response(),
            "comms_overhead_ns": True,
        },
        {
            **_summary_response(),
            "comms_overhead_ns": 1,
        },
        {
            **_summary_response(),
            "unknown": 1,
        },
        {
            **_summary_response(),
            "budget": {"flop_budget": 100},
        },
        {
            **_summary_response(),
            "budget": {
                "flop_budget": True,
                "flops_used": 7,
                "flops_remaining": 93,
            },
        },
        {
            **_summary_response(),
            "budget": {
                "flop_budget": 100,
                "flops_used": -1,
                "flops_remaining": 101,
            },
        },
        {
            **_summary_response(),
            "_round_trip_ns": -1,
        },
        {
            **_summary_response(),
            "_request_bytes": True,
        },
        {
            **_summary_response(),
            "_response_bytes": 1.5,
        },
    ],
)
def test_malformed_success_envelope_raises_flopscope_server_error(
    monkeypatch, response
) -> None:
    conn = MagicMock()
    conn.send_recv.return_value = response
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)

    with pytest.raises(FlopscopeServerError, match="malformed budget_summary response"):
        budget_module.budget_summary_dict(False)


@pytest.mark.parametrize(
    ("by_namespace", "mutate"),
    [
        (False, lambda r: r["result"].__setitem__("flop_budget", True)),
        (False, lambda r: r["result"].__setitem__("flops_used", -1)),
        (False, lambda r: r["result"].__setitem__("flops_remaining", -1)),
        (False, lambda r: r["result"].__setitem__("wall_time_s", True)),
        (False, lambda r: r["result"].__setitem__("wall_time_s", -1.0)),
        (False, lambda r: r["result"].__setitem__("wall_time_s", float("nan"))),
        (
            False,
            lambda r: r["result"].__setitem__("residual_wall_time_s", float("inf")),
        ),
        (
            False,
            lambda r: r["result"]["operations"]["add"].__setitem__("flop_cost", -1),
        ),
        (
            False,
            lambda r: r["result"]["operations"]["add"].__setitem__("calls", True),
        ),
        (
            False,
            lambda r: r["result"]["operations"]["add"].__setitem__(
                "flopscope_backend_time_s", float("nan")
            ),
        ),
        (
            False,
            lambda r: r["result"]["operations"]["add"].__setitem__(
                "flopscope_overhead_time_s", -1.0
            ),
        ),
        (
            True,
            lambda r: r["result"]["by_namespace"]["phase"].__setitem__(
                "flops_used", -1
            ),
        ),
        (
            True,
            lambda r: r["result"]["by_namespace"]["phase"].__setitem__("calls", True),
        ),
        (
            True,
            lambda r: r["result"]["by_namespace"]["phase"].__setitem__(
                "flopscope_backend_time_s", float("inf")
            ),
        ),
        (
            True,
            lambda r: r["result"]["by_namespace"]["phase"]["operations"][
                "add"
            ].__setitem__("flopscope_overhead_time_s", True),
        ),
        (False, lambda r: r["display_totals"].__setitem__("budget", True)),
        (False, lambda r: r["display_totals"].__setitem__("used", -1)),
        (
            False,
            lambda r: r["display_totals"].__setitem__(
                "client_context_compute_ns", True
            ),
        ),
        (
            False,
            lambda r: r["display_totals"].__setitem__("client_context_compute_ns", -1),
        ),
    ],
)
def test_summary_rejects_noncanonical_numeric_values(
    monkeypatch, by_namespace, mutate
) -> None:
    response = _summary_response(by_namespace=by_namespace)
    mutate(response)
    conn = MagicMock()
    conn.send_recv.return_value = response
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)

    with pytest.raises(FlopscopeServerError, match="malformed budget_summary response"):
        budget_module.budget_summary_dict(by_namespace)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("result"),
        lambda r: r["result"].pop("operations"),
        lambda r: r["result"].__setitem__("by_namespace", {}),
        lambda r: r.__setitem__("display_totals", {"budget": "bad"}),
    ],
)
def test_malformed_summary_response_raises_flopscope_server_error(
    monkeypatch, mutate
) -> None:
    response = _summary_response(flops_used=7, by_namespace=False)
    mutate(response)
    conn = MagicMock()
    conn.send_recv.return_value = response
    monkeypatch.setattr(budget_module, "get_connection", lambda: conn)
    with pytest.raises(FlopscopeServerError, match="malformed budget_summary response"):
        budget_module.budget_summary_dict(False)
