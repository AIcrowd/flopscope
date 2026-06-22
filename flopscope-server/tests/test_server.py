"""Integration tests for FlopscopeServer — real ZMQ over TCP."""

from __future__ import annotations

import threading
import time

import msgpack
import numpy as np
import pytest
import zmq
from flopscope_server._server import FlopscopeServer, _normalize_arg, _normalize_msg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SERVER_URL = "tcp://127.0.0.1:15555"


def _send(sock: zmq.Socket, msg: dict) -> dict:
    """Send a msgpack request and return the decoded response.

    Decode with ``strict_map_key=False`` to match how the real consumers decode
    server responses — the flopscope-client (``_protocol.py``) and the whestbench
    worker both pass ``strict_map_key=False``. The ``budget_close`` summary's
    ``by_namespace`` breakdown legitimately uses a ``None`` key for unlabeled ops,
    which the strict default rejects.
    """
    sock.send(msgpack.packb(msg, use_bin_type=True))
    raw = sock.recv()
    return msgpack.unpackb(raw, raw=False, strict_map_key=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def server_and_client():
    """Start a FlopscopeServer in a daemon thread and yield a REQ client socket."""
    server = FlopscopeServer(url=SERVER_URL, session_timeout_s=60.0)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(0.2)  # let the server bind

    ctx = zmq.Context()
    client = ctx.socket(zmq.REQ)
    client.setsockopt(zmq.RCVTIMEO, 5000)
    client.connect(SERVER_URL)

    yield server, client

    server.stop()
    client.close(linger=0)
    ctx.term()
    t.join(timeout=3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_budget_open_and_status(server_and_client):
    """budget_open creates a session; budget_status returns remaining FLOPs."""
    _server, client = server_and_client

    resp = _send(client, {"op": "budget_open", "flop_budget": 500_000})
    assert resp["status"] == "ok"
    assert resp["result"]["session"] == "opened"
    assert resp["budget"] == 500_000

    resp = _send(client, {"op": "budget_status"})
    assert resp["status"] == "ok"
    assert resp["result"]["flops_remaining"] == 500_000


def test_create_array_and_fetch(server_and_client):
    """Create an array via create_from_data and fetch it back."""
    _server, client = server_and_client

    _send(client, {"op": "budget_open", "flop_budget": 1_000_000})

    arr = np.array([1.0, 2.0, 3.0], dtype="float64")
    resp = _send(
        client,
        {
            "op": "create_from_data",
            "data": arr.tobytes(),
            "dtype": "float64",
            "shape": [3],
        },
    )
    assert resp["status"] == "ok"
    handle = resp["result"]["id"]

    resp = _send(client, {"op": "fetch", "id": handle})
    assert resp["status"] == "ok"
    fetched = np.frombuffer(resp["data"], dtype=resp["dtype"]).reshape(resp["shape"])
    np.testing.assert_array_equal(fetched, arr)


def test_fetch_slice_scalar_uses_raw_transport(server_and_client):
    """Scalar fetch_slice stays on the raw data/shape/dtype transport path."""
    _server, client = server_and_client

    _send(client, {"op": "budget_open", "flop_budget": 1_000_000})

    arr = np.array(2.0, dtype="float64")
    resp = _send(
        client,
        {
            "op": "create_from_data",
            "data": arr.tobytes(),
            "dtype": "float64",
            "shape": [],
        },
    )
    handle = resp["result"]["id"]

    resp = _send(client, {"op": "fetch_slice", "id": handle, "slices": []})
    assert resp["status"] == "ok"
    assert resp["shape"] == []
    assert resp["dtype"] == "float64"
    assert "result" not in resp
    value = (
        np.frombuffer(resp["data"], dtype=resp["dtype"]).reshape(resp["shape"]).item()
    )
    assert value == 2.0


def test_operation_chain_ones_exp_fetch(server_and_client):
    """ones -> exp -> fetch, verify values are exp(1)."""
    _server, client = server_and_client

    _send(client, {"op": "budget_open", "flop_budget": 10_000_000})

    # ones
    resp = _send(client, {"op": "ones", "args": [(4,)], "kwargs": {}})
    assert resp["status"] == "ok"
    ones_handle = resp["result"]["id"]

    # exp
    resp = _send(client, {"op": "exp", "args": [ones_handle], "kwargs": {}})
    assert resp["status"] == "ok"
    exp_handle = resp["result"]["id"]

    # fetch
    resp = _send(client, {"op": "fetch", "id": exp_handle})
    assert resp["status"] == "ok"
    result = np.frombuffer(resp["data"], dtype=resp["dtype"]).reshape(resp["shape"])
    np.testing.assert_allclose(result, np.e * np.ones(4), rtol=1e-7)


def test_budget_close_returns_summary(server_and_client):
    """budget_close returns a summary with budget and comms info."""
    _server, client = server_and_client

    _send(client, {"op": "budget_open", "flop_budget": 1_000_000})

    # Do some work so the summary is non-trivial
    _send(client, {"op": "ones", "args": [(5,)], "kwargs": {}})

    resp = _send(client, {"op": "budget_close"})
    assert resp["status"] == "ok"
    result = resp["result"]
    assert "budget_summary" in result
    assert "budget_breakdown" in result
    assert "comms_summary" in result
    assert isinstance(result["budget_breakdown"], dict)
    assert result["comms_summary"]["request_count"] >= 1


def test_budget_close_returns_structured_namespace_breakdown(server_and_client):
    """budget_close returns machine-readable namespace data when labels exist."""
    server, client = server_and_client

    _send(client, {"op": "budget_open", "flop_budget": 1_000_000})

    ctx = server._session.budget_context
    ctx._push_namespace("phase")
    try:
        ctx.deduct("add", flop_cost=1, subscripts=None, shapes=())
    finally:
        ctx._pop_namespace("phase")

    resp = _send(client, {"op": "budget_close"})
    assert resp["status"] == "ok"
    result = resp["result"]
    assert result["budget_breakdown"]["by_namespace"]["phase"]["flops_used"] == 1
    assert result["budget_breakdown"]["by_namespace"]["phase"]["calls"] == 1


def test_error_no_session(server_and_client):
    """Operations without an active session return NoBudgetContextError."""
    _server, client = server_and_client

    resp = _send(client, {"op": "ones", "args": [(3,)], "kwargs": {}})
    assert resp["status"] == "error"
    assert resp["error_type"] == "NoBudgetContextError"


def test_error_unknown_op(server_and_client):
    """Unknown ops return InvalidRequestError."""
    _server, client = server_and_client

    _send(client, {"op": "budget_open", "flop_budget": 1_000_000})

    resp = _send(client, {"op": "nonexistent_banana_op"})
    assert resp["status"] == "error"
    assert resp["error_type"] == "InvalidRequestError"
    assert "unknown op" in resp["message"]


def test_error_invalid_msgpack(server_and_client):
    """Sending garbage bytes returns an InvalidRequestError."""
    _server, client = server_and_client

    client.send(b"\x00\x01\x02garbage")
    raw = client.recv()
    resp = msgpack.unpackb(raw, raw=False)
    assert resp["status"] == "error"
    assert resp["error_type"] == "InvalidRequestError"


def test_budget_close_without_session(server_and_client):
    """budget_close without an open session returns an error."""
    _server, client = server_and_client

    resp = _send(client, {"op": "budget_close"})
    assert resp["status"] == "error"
    assert resp["error_type"] == "NoBudgetContextError"


def test_session_reopen_blocked(server_and_client):
    """FIX 1: budget_open with an active session must return an error."""
    _server, client = server_and_client

    _send(client, {"op": "budget_open", "flop_budget": 100_000})

    # A second budget_open MUST fail instead of silently resetting
    resp = _send(client, {"op": "budget_open", "flop_budget": 200_000})
    assert resp["status"] == "error"
    assert resp["error_type"] == "RuntimeError"
    assert "already open" in resp["message"]

    # Original session is still active
    resp = _send(client, {"op": "budget_status"})
    assert resp["result"]["flop_budget"] == 100_000

    # Close and reopen should work
    _send(client, {"op": "budget_close"})
    resp = _send(client, {"op": "budget_open", "flop_budget": 200_000})
    assert resp["status"] == "ok"
    assert resp["result"]["flop_budget"] == 200_000


# ---------------------------------------------------------------------------
# FIX 2: _normalize_arg preserves binary data
# ---------------------------------------------------------------------------


def test_normalize_arg_preserves_binary_float64():
    """FIX 2: small binary data (e.g. 8-byte float64) must NOT be decoded."""
    import struct

    data = struct.pack("<d", 3.14)  # 8 bytes, may be valid UTF-8
    result = _normalize_arg(data)
    assert isinstance(result, bytes), (
        "binary float64 data was incorrectly decoded to str"
    )
    assert result == data


def test_normalize_arg_decodes_handle_id():
    """FIX 2: short ASCII handle IDs like b'a0' must still be decoded."""
    result = _normalize_arg(b"a0")
    assert result == "a0"
    assert isinstance(result, str)


def test_normalize_arg_decodes_dtype():
    """FIX 2: short ASCII dtype strings must still be decoded."""
    result = _normalize_arg(b"float64")
    assert result == "float64"
    assert isinstance(result, str)


def test_normalize_arg_preserves_high_bytes():
    """FIX 2: bytes with high bytes (>127) must be kept as bytes."""
    data = bytes([0x80, 0x90, 0xA0])  # high bytes
    result = _normalize_arg(data)
    assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# FIX 8: kwargs normalization uses _normalize_arg
# ---------------------------------------------------------------------------


def test_normalize_msg_kwargs_handle_dict():
    """FIX 8: kwargs values containing handle dicts are normalized recursively."""
    msg = {
        "op": "some_op",
        "args": [],
        "kwargs": {
            b"out": {b"__handle__": b"a5"},
        },
    }
    _normalize_msg(msg)
    # The dict inside kwargs should have its keys/values normalized
    out_val = msg["kwargs"]["out"]
    assert isinstance(out_val, dict)
    assert "__handle__" in out_val
    assert out_val["__handle__"] == "a5"


# ---------------------------------------------------------------------------
# kernel_ns timing tests (pure numpy kernel attribution)
# ---------------------------------------------------------------------------


def test_compute_time_is_kernel_only():
    """A compute op records kernel time that does not exceed the full handle() wall."""
    import time

    import numpy as np
    from flopscope_server._request_handler import RequestHandler
    from flopscope_server._session import Session

    session = Session(flop_budget=10**12)
    handler = RequestHandler(session)
    h = session.store_array(np.ones((256, 256)))

    t0 = time.perf_counter_ns()
    handler.handle({"op": "dot", "args": [h, h], "kwargs": None})
    handle_ns = time.perf_counter_ns() - t0

    assert handler.kernel_ns > 0
    assert handler.kernel_ns <= handle_ns
    session.close()


def test_fetch_contributes_no_kernel():
    """A fetch op is data movement, not a numpy kernel — kernel_ns stays 0."""
    import numpy as np
    from flopscope_server._request_handler import RequestHandler
    from flopscope_server._session import Session

    session = Session(flop_budget=10**12)
    handler = RequestHandler(session)
    h = session.store_array(np.ones((8, 8)))

    handler.handle({"op": "fetch", "id": h})
    assert handler.kernel_ns == 0
    session.close()


def test_handle_persists_across_budget_sessions(server_and_client):
    """A handle minted in MLP #1's budget session resolves in MLP #2's session.

    Mirrors PR #108's tests/integration/test_flopscope_session_handle_lifetime.py
    assertion (c): the warm child holds a module-level handle across a
    budget_close/budget_open, and maximum(var_c, floor) must succeed.
    """
    _server, client = server_and_client

    # MLP #1: create a 0-d "floor" array (scalar 0.0), capture its handle, close.
    _send(client, {"op": "budget_open", "flop_budget": 1_000_000})
    floor_arr = np.array(0.0, dtype="float32")
    resp = _send(
        client,
        {
            "op": "create_from_data",
            "data": floor_arr.tobytes(),
            "dtype": "float32",
            "shape": [],
        },
    )
    assert resp["status"] == "ok"
    floor_id = resp["result"]["id"]
    assert resp["result"]["shape"] == []

    # Burn FLOPs in MLP #1 so the per-MLP budget reset is observable in MLP #2.
    ones_resp = _send(client, {"op": "ones", "args": [(8,)], "kwargs": {}})
    _send(client, {"op": "exp", "args": [ones_resp["result"]["id"]], "kwargs": {}})
    mlp1_status = _send(client, {"op": "budget_status"})
    assert mlp1_status["result"]["flops_used"] > 0
    _send(client, {"op": "budget_close"})

    # MLP #2 on the same connection: a new array must NOT reuse floor's id, and
    # the floor handle must still resolve to its original 0-d array.
    _send(client, {"op": "budget_open", "flop_budget": 1_000_000})

    # Per-MLP budget integrity: each budget_open starts a FRESH FLOP counter —
    # MLP #1's spend must NOT carry over (only the handle store persists, not
    # the budget). This is the dual of the persistence invariant below.
    mlp2_status = _send(client, {"op": "budget_status"})
    assert mlp2_status["result"]["flops_used"] == 0
    assert mlp2_status["result"]["flops_remaining"] == 1_000_000

    weights_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
    resp = _send(
        client,
        {
            "op": "create_from_data",
            "data": weights_arr.tobytes(),
            "dtype": "float32",
            "shape": [5],
        },
    )
    assert resp["status"] == "ok"
    weights_id = resp["result"]["id"]
    assert weights_id != floor_id  # monotonic id, never reused

    # floor_id must still resolve to its original 0-d array
    fetched = _send(client, {"op": "fetch", "id": floor_id})
    assert fetched["status"] == "ok", fetched
    assert fetched["shape"] == []  # still the original 0-d floor, not aliased
    assert fetched["dtype"] == "float32"

    # maximum(var_c, floor) across the budget boundary must succeed
    var_c_arr = np.array([-1.0, 0.5, 2.0], dtype="float32")
    resp = _send(
        client,
        {
            "op": "create_from_data",
            "data": var_c_arr.tobytes(),
            "dtype": "float32",
            "shape": [3],
        },
    )
    assert resp["status"] == "ok"
    var_c_id = resp["result"]["id"]

    result = _send(
        client,
        {"op": "maximum", "args": [var_c_id, floor_id], "kwargs": {}},
    )
    assert result["status"] == "ok", result
    assert result["result"]["shape"] == [3]

    _send(client, {"op": "budget_close"})
