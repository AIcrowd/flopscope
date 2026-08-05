"""Tests for the authoritative server-side budget summary RPC."""

from __future__ import annotations

from unittest.mock import MagicMock

import msgpack
import pytest
from flopscope_server._request_handler import RequestHandler
from flopscope_server._server import FlopscopeServer
from flopscope_server._session import Session

import flopscope
from flopscope._budget import get_active_budget

TOKEN = "test-control-token"


class FakeClock:
    def __init__(self) -> None:
        self.ns = 0

    def tick(self, ns: int) -> None:
        self.ns += ns

    def perf_counter_ns(self) -> int:
        return self.ns

    def perf_counter(self) -> float:
        return self.ns / 1e9


@pytest.fixture
def fake_clock(monkeypatch):
    import flopscope_server._server as server_module

    clock = FakeClock()
    monkeypatch.setattr(server_module, "perf_counter_ns", clock.perf_counter_ns)
    monkeypatch.setattr(flopscope._budget.time, "perf_counter", clock.perf_counter)
    return clock


@pytest.fixture(autouse=True)
def _isolated_core_summary():
    flopscope.budget_reset()
    yield
    active = get_active_budget()
    if active is not None:
        active.__exit__(None, None, None)
    flopscope.budget_reset()


def _open_direct(
    server: FlopscopeServer,
    *,
    flop_budget: int = 100,
    namespace: str | None = None,
) -> None:
    server._session = Session(
        flop_budget=flop_budget,
        conn_store=server._conn_store,
        namespace=namespace,
    )
    server._handler = RequestHandler(server._session)


def _charge_direct(server: FlopscopeServer, cost: int = 5) -> None:
    assert server._session is not None
    with server._session.budget_context.deduct(
        "add",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(),
    ):
        pass


def _close_direct(server: FlopscopeServer) -> None:
    response = msgpack.unpackb(
        server._handle_budget_close(0, 0),
        raw=False,
        strict_map_key=False,
    )
    assert response["status"] == "ok"


@pytest.mark.parametrize("namespace", [None, "phase", "predict..raw"])
def test_budget_open_accepts_literal_root_namespace(namespace) -> None:
    server = FlopscopeServer()
    raw = msgpack.packb(
        {
            "op": "budget_open",
            "kwargs": {"flop_budget": 100, "namespace": namespace},
        },
        use_bin_type=True,
    )
    response = msgpack.unpackb(server._process_request(raw), raw=False)
    assert response["status"] == "ok"
    assert server._session is not None
    assert server._session.budget_context.namespace == namespace
    _close_direct(server)


@pytest.mark.parametrize("namespace", [1, False, [], {}])
def test_budget_open_rejects_non_string_namespace(namespace) -> None:
    server = FlopscopeServer()
    raw = msgpack.packb(
        {
            "op": "budget_open",
            "kwargs": {"flop_budget": 100, "namespace": namespace},
        },
        use_bin_type=True,
    )
    response = msgpack.unpackb(server._process_request(raw), raw=False)
    assert response["status"] == "error"
    assert response["error_type"] == "InvalidRequestError"
    assert server._session is None


@pytest.mark.parametrize("kwargs", [[], False, 0, ""])
def test_budget_open_rejects_falsy_non_dict_kwargs(kwargs) -> None:
    server = FlopscopeServer()
    response = msgpack.unpackb(
        server._handle_budget_open({"op": "budget_open", "kwargs": kwargs}, 0, 0),
        raw=False,
    )
    assert response["status"] == "error"
    assert response["error_type"] == "InvalidRequestError"
    assert server._session is None


@pytest.fixture
def server_with_closed_cost() -> FlopscopeServer:
    server = FlopscopeServer()
    _open_direct(server)
    _charge_direct(server, 5)
    _close_direct(server)
    return server


@pytest.fixture
def server_with_closed_and_active_cost() -> FlopscopeServer:
    server = FlopscopeServer()
    _open_direct(server)
    _charge_direct(server, 5)
    _close_direct(server)
    _open_direct(server)
    _charge_direct(server, 7)
    return server


@pytest.fixture
def server_with_stale_closed_session() -> FlopscopeServer:
    server = FlopscopeServer()
    _open_direct(server)
    _charge_direct(server, 5)
    assert server._session is not None
    server._session.close()
    return server


def _request(server, *, scope="session", by_namespace=False, **extra):
    kwargs = {"scope": scope, "by_namespace": by_namespace, **extra}
    raw = msgpack.packb(
        {"op": "budget_summary", "args": None, "kwargs": kwargs},
        use_bin_type=True,
    )
    return msgpack.unpackb(
        server._process_request(raw), raw=False, strict_map_key=False
    )


def _reset_request(server, token=None):
    kwargs = {} if token is None else {"control_token": token}
    raw = msgpack.packb(
        {"op": "budget_summary_reset", "kwargs": kwargs},
        use_bin_type=True,
    )
    return msgpack.unpackb(server._process_request(raw), raw=False)


@pytest.fixture
def closed_token_server() -> FlopscopeServer:
    server = FlopscopeServer(control_token=TOKEN)
    _open_direct(server)
    _charge_direct(server, 5)
    _close_direct(server)
    return server


@pytest.fixture
def active_token_server() -> FlopscopeServer:
    server = FlopscopeServer(control_token=TOKEN)
    _open_direct(server)
    _charge_direct(server, 5)
    return server


def test_encode_budget_summary_response_has_frozen_envelope() -> None:
    from flopscope_server._protocol import encode_budget_summary_response

    raw = encode_budget_summary_response(
        {"flops_used": 3},
        display_totals={
            "has_explicit_budget": True,
            "budget": 10,
            "used": 3,
            "client_context_compute_ns": 42,
        },
        budget={"flop_budget": 10, "flops_used": 3, "flops_remaining": 7},
    )
    decoded = msgpack.unpackb(raw, raw=False)
    assert decoded == {
        "status": "ok",
        "result": {"flops_used": 3},
        "display_totals": {
            "has_explicit_budget": True,
            "budget": 10,
            "used": 3,
            "client_context_compute_ns": 42,
        },
        "budget": {"flop_budget": 10, "flops_used": 3, "flops_remaining": 7},
        "comms_overhead_ns": 0,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "op": "budget_summary",
                "args": [1],
                "kwargs": {"scope": "session", "by_namespace": False},
            },
            "args",
        ),
        ({"op": "budget_summary", "kwargs": None}, "kwargs must be a dict"),
        (
            {"op": "budget_summary", "kwargs": {"scope": "session"}},
            "exactly",
        ),
        (
            {
                "op": "budget_summary",
                "kwargs": {
                    "scope": "session",
                    "by_namespace": False,
                    "duration_s": 99,
                },
            },
            "duration_s",
        ),
        (
            {
                "op": "budget_summary",
                "kwargs": {"scope": "session", "by_namespace": False},
                "duration_s": 99,
            },
            "top-level",
        ),
        (
            {
                "op": "budget_summary",
                "kwargs": {"scope": "other", "by_namespace": False},
            },
            "scope",
        ),
        (
            {
                "op": "budget_summary",
                "kwargs": {"scope": "session", "by_namespace": 1},
            },
            "boolean",
        ),
    ],
)
def test_budget_summary_rejects_noncanonical_requests(payload, message) -> None:
    server = FlopscopeServer()
    response = msgpack.unpackb(
        server._process_request(msgpack.packb(payload, use_bin_type=True)),
        raw=False,
    )
    assert response["status"] == "error"
    assert response["error_type"] == "InvalidRequestError"
    assert message in response["message"]


def test_session_scope_is_available_between_contexts(server_with_closed_cost) -> None:
    response = _request(server_with_closed_cost, scope="session", by_namespace=True)
    assert response["status"] == "ok"
    assert response["result"]["flops_used"] > 0
    assert "by_namespace" in response["result"]
    assert "display_totals" not in response["result"]
    assert "budget" not in response


def test_active_context_scope_requires_live_session() -> None:
    response = _request(FlopscopeServer(), scope="active_context")
    assert response["status"] == "error"
    assert response["error_type"] == "NoBudgetContextError"


def test_active_context_scope_rejects_stale_closed_session(
    server_with_stale_closed_session,
) -> None:
    response = _request(server_with_stale_closed_session, scope="active_context")

    assert response["status"] == "error"
    assert response["error_type"] == "NoBudgetContextError"


def test_session_scope_ignores_stale_closed_session_metadata(
    server_with_stale_closed_session,
) -> None:
    response = _request(server_with_stale_closed_session, scope="session")

    assert response["status"] == "ok"
    assert response["result"]["flops_used"] > 0
    assert "budget" not in response
    assert response["display_totals"]["client_context_compute_ns"] is None


def test_closed_session_rejects_active_context_summary() -> None:
    session = Session(flop_budget=100)
    session.close()

    with pytest.raises(flopscope.NoBudgetContextError):
        session.budget_summary_dict(by_namespace=False)


def test_active_scope_is_only_active_context(
    server_with_closed_and_active_cost,
) -> None:
    server = server_with_closed_and_active_cost
    response = _request(server, scope="active_context", by_namespace=True)
    direct = server._session.budget_context.summary_dict(by_namespace=True)
    assert response["result"]["flop_budget"] == direct["flop_budget"]
    assert response["result"]["flops_used"] == direct["flops_used"]
    assert response["result"]["operations"] == direct["operations"]
    assert response["budget"] == server._session.budget_status()
    assert (
        response["display_totals"]["client_context_compute_ns"]
        == (server._session.comms_tracker.summary()["total_compute_time_ns"])
    )


def test_summary_outer_overhead_appears_once_in_next_snapshot(
    fake_clock, monkeypatch
) -> None:
    import flopscope_server._server as server_module

    import flopscope._budget as core_budget

    active_server = FlopscopeServer()
    _open_direct(active_server)
    _charge_direct(active_server, 7)
    original_decode = server_module.decode_request
    original_snapshot = core_budget._accumulator.snapshot
    original_encode = server_module.encode_budget_summary_response

    def decode(raw):
        fake_clock.tick(2_000_000)
        return original_decode(raw)

    def snapshot(*args, **kwargs):
        fake_clock.tick(3_000_000)
        return original_snapshot(*args, **kwargs)

    def encode(*args, **kwargs):
        fake_clock.tick(5_000_000)
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(server_module, "decode_request", decode)
    monkeypatch.setattr(core_budget._accumulator, "snapshot", snapshot)
    monkeypatch.setattr(server_module, "encode_budget_summary_response", encode)

    first = _request(active_server, scope="session")["result"]
    second = _request(active_server, scope="session")["result"]
    third = _request(active_server, scope="session")["result"]

    first_overhead = first["flopscope_overhead_time_s"]
    delta_2 = second["flopscope_overhead_time_s"] - first_overhead
    delta_3 = third["flopscope_overhead_time_s"] - second["flopscope_overhead_time_s"]
    assert delta_2 == pytest.approx(0.010)
    assert delta_3 == pytest.approx(delta_2)


def test_between_session_summary_does_not_mutate_closed_aggregate(
    server_with_closed_cost,
) -> None:
    closed_server = server_with_closed_cost
    before = _request(closed_server, scope="session")["result"]
    _request(closed_server, scope="session")
    after = _request(closed_server, scope="session")["result"]
    assert after == before


def test_unknown_duration_is_rejected_and_not_attributed(
    server_with_closed_and_active_cost,
) -> None:
    active_server = server_with_closed_and_active_cost
    assert active_server._session is not None
    before = active_server._session.budget_context.flopscope_overhead_time_s
    response = _request(active_server, duration_s=1000.0)
    after = active_server._session.budget_context.flopscope_overhead_time_s
    assert response["error_type"] == "InvalidRequestError"
    assert after == before


def test_summary_serialization_failure_is_encoded_and_recorded_once(
    server_with_closed_and_active_cost, monkeypatch
) -> None:
    import flopscope_server._server as server_module

    active_server = server_with_closed_and_active_cost
    monkeypatch.setattr(
        server_module,
        "encode_budget_summary_response",
        lambda *a, **k: (_ for _ in ()).throw(TypeError("boom")),
    )
    assert active_server._session is not None
    before = active_server._session.comms_tracker.summary()["request_count"]
    response = _request(active_server)
    after = active_server._session.comms_tracker.summary()["request_count"]
    assert response["error_type"] == "FlopscopeServerError"
    assert "TypeError" in response["message"]
    assert after == before + 1


def test_close_uses_one_final_mapping_and_includes_prior_summary_overhead(
    server_with_closed_and_active_cost, monkeypatch
) -> None:
    server = server_with_closed_and_active_cost
    assert server._session is not None
    _request(server, scope="active_context")
    context = server._session.budget_context
    committed_before_close = context.flopscope_overhead_time_s
    wrapped = MagicMock(wraps=context.summary_dict)
    monkeypatch.setattr(context, "summary_dict", wrapped)

    response = msgpack.unpackb(
        server._handle_budget_close(0, 0),
        raw=False,
        strict_map_key=False,
    )
    breakdown = response["result"]["budget_breakdown"]
    text = response["result"]["budget_summary"]
    assert wrapped.call_count == 1
    assert breakdown["flopscope_overhead_time_s"] >= committed_before_close
    assert f"{breakdown['flops_used']:,}" in text
    assert all(name in text for name in breakdown["operations"])


def test_reset_requires_control_token(closed_token_server) -> None:
    before = _request(closed_token_server, scope="session")["result"]
    rejected = _reset_request(closed_token_server)
    after = _request(closed_token_server, scope="session")["result"]
    assert rejected["error_type"] == "UnauthorizedControlError"
    assert after["flops_used"] == before["flops_used"]


def test_reset_rejects_wrong_control_token_without_mutating_summary(
    closed_token_server,
) -> None:
    before = _request(closed_token_server, scope="session")["result"]

    rejected = _reset_request(closed_token_server, "wrong-token")

    after = _request(closed_token_server, scope="session")["result"]
    assert rejected["error_type"] == "UnauthorizedControlError"
    assert after["flops_used"] == before["flops_used"]


@pytest.mark.parametrize("malformed_kwargs", ["not-a-dict", ["not-a-dict"]])
def test_reset_rejects_malformed_kwargs_without_mutating_summary(
    closed_token_server,
    malformed_kwargs,
) -> None:
    before = _request(closed_token_server, scope="session")["result"]
    raw = msgpack.packb(
        {"op": "budget_summary_reset", "kwargs": malformed_kwargs},
        use_bin_type=True,
    )

    rejected = msgpack.unpackb(
        closed_token_server._process_request(raw),
        raw=False,
    )

    after = _request(closed_token_server, scope="session")["result"]
    assert rejected["status"] == "error"
    assert rejected["error_type"] == "UnauthorizedControlError"
    assert after["flops_used"] == before["flops_used"]


def test_reset_rejects_server_session_even_with_token(active_token_server) -> None:
    before = _request(active_token_server, scope="session")["result"]

    response = _reset_request(active_token_server, TOKEN)

    after = _request(active_token_server, scope="session")["result"]
    assert response["error_type"] == "RuntimeError"
    assert "active" in response["message"]
    assert after["flops_used"] == before["flops_used"]


def test_reset_rejects_non_session_core_context() -> None:
    server = FlopscopeServer(control_token=TOKEN)
    with flopscope.BudgetContext(100, quiet=True):
        response = _reset_request(server, TOKEN)
    assert response["error_type"] == "RuntimeError"


def test_authorized_reset_returns_single_envelope_and_zeros_epoch(
    closed_token_server,
) -> None:
    response = _reset_request(closed_token_server, TOKEN)
    assert response == {
        "status": "ok",
        "result": None,
        "budget": 0,
        "comms_overhead_ns": 0,
    }
    summary = _request(closed_token_server, scope="session")["result"]
    assert summary["flop_budget"] == 0
    assert summary["flops_used"] == 0
    assert summary["operations"] == {}


def test_authorized_reset_accepts_byte_control_token(closed_token_server) -> None:
    response = _reset_request(closed_token_server, TOKEN.encode())

    assert response["status"] == "ok"
    summary = _request(closed_token_server, scope="session")["result"]
    assert summary["flops_used"] == 0


def test_authorized_reset_accepts_top_level_control_token(
    closed_token_server,
) -> None:
    raw = msgpack.packb(
        {
            "op": "budget_summary_reset",
            "kwargs": {},
            "control_token": TOKEN,
        },
        use_bin_type=True,
    )

    response = msgpack.unpackb(
        closed_token_server._process_request(raw),
        raw=False,
    )

    assert response["status"] == "ok"
    summary = _request(closed_token_server, scope="session")["result"]
    assert summary["flops_used"] == 0


def test_authorized_reset_allows_stale_closed_server_session(
    server_with_stale_closed_session,
) -> None:
    server_with_stale_closed_session._control_token = TOKEN

    response = _reset_request(server_with_stale_closed_session, TOKEN)

    assert response["status"] == "ok"
    summary = _request(server_with_stale_closed_session, scope="session")["result"]
    assert summary["flops_used"] == 0
