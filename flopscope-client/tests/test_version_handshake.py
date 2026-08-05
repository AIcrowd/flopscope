"""Tests for the client-side lazy version handshake."""

from __future__ import annotations

import msgpack
import pytest
from flopscope._protocol import AUTHORITATIVE_BUDGET_SUMMARY_CAPABILITY

from flopscope import _connection, _protocol
from flopscope.errors import FlopscopeServerError

CAPABILITY = AUTHORITATIVE_BUDGET_SUMMARY_CAPABILITY


def _client_xyz() -> str:
    import flopscope

    return flopscope.__version__.split("+", 1)[0]


def test_authoritative_budget_summary_capability_wire_value() -> None:
    assert AUTHORITATIVE_BUDGET_SUMMARY_CAPABILITY == "authoritative_budget_summary_v1"


def test_encode_hello_serializes_client_version():
    """encode_hello produces a hello op with a client_version kwarg."""
    raw = _protocol.encode_hello("0.3.0")
    decoded = msgpack.unpackb(raw, raw=False)
    assert decoded["op"] == "hello"
    assert decoded["kwargs"]["client_version"] == "0.3.0"


class _FakeSocket:
    """Stand-in for a zmq REQ socket: records sends, replays scripted recvs."""

    def __init__(self, recv_payloads: list[bytes]) -> None:
        self.sent: list[bytes] = []
        self._recv_payloads = list(recv_payloads)
        self.closed_with_linger: int | None = None

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self) -> bytes:
        return self._recv_payloads.pop(0)

    def close(self, *, linger: int = 0) -> None:
        self.closed_with_linger = linger


def _make_connection(socket: _FakeSocket) -> _connection.Connection:
    conn = _connection.Connection()
    conn._socket = socket
    return conn


def test_ensure_handshaked_happy_path_sets_flag_and_sends_hello():
    server_ok = msgpack.packb(
        {"status": "ok", "server_version": _client_xyz()}, use_bin_type=True
    )
    sock = _FakeSocket([server_ok])
    conn = _make_connection(sock)

    assert conn._handshake_done is False
    conn._ensure_handshaked()
    assert conn._handshake_done is True

    first = msgpack.unpackb(sock.sent[0], raw=False)
    assert first["op"] == "hello"
    assert "client_version" in first["kwargs"]


def test_malformed_msgpack_rejects_handshake_and_resets_socket() -> None:
    sock = _FakeSocket([b"\xc1"])
    conn = _make_connection(sock)
    conn._capabilities = frozenset({CAPABILITY})

    with pytest.raises(ConnectionError, match="malformed handshake response"):
        conn._ensure_handshaked()

    assert sock.closed_with_linger == 0
    assert conn._socket is None
    assert conn._handshake_done is False
    assert conn._capabilities == frozenset()


@pytest.mark.parametrize("response", [[], "ok"])
def test_non_mapping_response_rejects_handshake_and_resets_socket(response) -> None:
    raw = msgpack.packb(response, use_bin_type=True)
    sock = _FakeSocket([raw])
    conn = _make_connection(sock)
    conn._capabilities = frozenset({CAPABILITY})

    with pytest.raises(ConnectionError, match="malformed handshake response"):
        conn._ensure_handshaked()

    assert sock.closed_with_linger == 0
    assert conn._socket is None
    assert conn._handshake_done is False
    assert conn._capabilities == frozenset()


@pytest.mark.parametrize("server_version", [None, "", 10, ["0.10.0"]])
def test_malformed_server_version_rejects_handshake_and_resets_socket(
    server_version,
) -> None:
    response = {"status": "ok"}
    if server_version is not None:
        response["server_version"] = server_version
    raw = msgpack.packb(response, use_bin_type=True)
    sock = _FakeSocket([raw])
    conn = _make_connection(sock)
    conn._capabilities = frozenset({CAPABILITY})

    with pytest.raises(ConnectionError) as excinfo:
        conn._ensure_handshaked()

    message = str(excinfo.value)
    assert "malformed server_version" in message
    assert "matching client/server versions" in message
    assert sock.closed_with_linger == 0
    assert conn._socket is None
    assert conn._handshake_done is False
    assert conn._capabilities == frozenset()


def test_mismatched_server_version_rejects_handshake_and_resets_socket() -> None:
    server_version = "0.0.0"
    raw = msgpack.packb(
        {"status": "ok", "server_version": server_version}, use_bin_type=True
    )
    sock = _FakeSocket([raw])
    conn = _make_connection(sock)
    conn._capabilities = frozenset({CAPABILITY})

    with pytest.raises(ConnectionError) as excinfo:
        conn._ensure_handshaked()

    message = str(excinfo.value)
    assert _client_xyz() in message
    assert server_version in message
    assert "Install matching client/server versions" in message
    assert sock.closed_with_linger == 0
    assert conn._socket is None
    assert conn._handshake_done is False
    assert conn._capabilities == frozenset()


def test_handshake_stores_capabilities() -> None:
    server_ok = msgpack.packb(
        {
            "status": "ok",
            "server_version": _client_xyz(),
            "capabilities": [CAPABILITY],
        },
        use_bin_type=True,
    )
    conn = _make_connection(_FakeSocket([server_ok]))
    conn._ensure_handshaked()
    assert conn._capabilities == frozenset({CAPABILITY})


def test_missing_capabilities_means_unsupported_peer() -> None:
    server_ok = msgpack.packb(
        {"status": "ok", "server_version": _client_xyz()}, use_bin_type=True
    )
    conn = _make_connection(_FakeSocket([server_ok]))
    conn._ensure_handshaked()
    with pytest.raises(
        FlopscopeServerError,
        match="requires authoritative remote-summary support",
    ):
        conn.require_capability(CAPABILITY)


@pytest.mark.parametrize("bad", ["capability", [1], {"x": True}])
def test_malformed_capabilities_rejects_handshake(bad) -> None:
    server_ok = msgpack.packb(
        {
            "status": "ok",
            "server_version": _client_xyz(),
            "capabilities": bad,
        },
        use_bin_type=True,
    )
    conn = _make_connection(_FakeSocket([server_ok]))
    with pytest.raises(ConnectionError, match="malformed capabilities"):
        conn._ensure_handshaked()
    assert conn._handshake_done is False
    assert conn._capabilities == frozenset()


def test_socket_reset_and_close_clear_capabilities() -> None:
    for method in ("_reset_socket", "close"):
        conn = _make_connection(_FakeSocket([]))
        conn._handshake_done = True
        conn._capabilities = frozenset({CAPABILITY})
        getattr(conn, method)()
        assert conn._handshake_done is False
        assert conn._capabilities == frozenset()


def test_ensure_handshaked_mismatch_raises_connection_error():
    err = msgpack.packb(
        {
            "status": "error",
            "error_type": "VersionMismatch",
            "message": ("flopscope-client 0.3.0 cannot talk to flopscope-server 0.4.0"),
        },
        use_bin_type=True,
    )
    sock = _FakeSocket([err])
    conn = _make_connection(sock)

    with pytest.raises(ConnectionError) as excinfo:
        conn._ensure_handshaked()
    msg = str(excinfo.value)
    assert "0.3.0" in msg
    assert "0.4.0" in msg


def test_ensure_handshaked_is_idempotent():
    """Calling _ensure_handshaked twice only sends one hello."""
    server_ok = msgpack.packb(
        {"status": "ok", "server_version": _client_xyz()}, use_bin_type=True
    )
    sock = _FakeSocket([server_ok])
    conn = _make_connection(sock)

    conn._ensure_handshaked()
    conn._ensure_handshaked()  # second call must not send again
    assert len(sock.sent) == 1


def test_send_recv_triggers_handshake_first():
    """The first send_recv on a fresh Connection performs the handshake."""
    server_ok_hello = msgpack.packb(
        {"status": "ok", "server_version": _client_xyz()}, use_bin_type=True
    )
    server_ok_op = msgpack.packb(
        {"status": "ok", "result": 42, "budget": 0, "comms_overhead_ns": 0},
        use_bin_type=True,
    )
    sock = _FakeSocket([server_ok_hello, server_ok_op])
    conn = _make_connection(sock)

    # Drain any GC-enqueued frees so send_recv's flush is a no-op; otherwise it
    # would consume a scripted recv (and an extra send) before the real op.
    import gc

    from flopscope import _handles

    gc.collect()
    _handles.drain_pending()
    # An arbitrary "op" request — not a hello — should still get handshaked first.
    op_request = msgpack.packb({"op": "budget_status"}, use_bin_type=True)
    response = conn.send_recv(op_request)
    assert response["status"] == "ok"
    assert conn._handshake_done is True
    # Order matters: hello first, then the real op.
    assert msgpack.unpackb(sock.sent[0], raw=False)["op"] == "hello"
    assert msgpack.unpackb(sock.sent[1], raw=False)["op"] == "budget_status"


def test_handshake_sends_stripped_prerelease_version(monkeypatch):
    """The client sends its +local-stripped PUBLIC version (rc kept) in the hello.

    A prerelease like 0.8.0rc0 carrying a dynamic +local build tag must be sent
    as "0.8.0rc0" so it string-matches the server's stripped version — the rc
    suffix is preserved, only the +local segment is dropped.
    """
    import flopscope

    monkeypatch.setattr(flopscope, "__version__", "0.8.0rc0+np9.9")
    server_ok = msgpack.packb(
        {"status": "ok", "server_version": "0.8.0rc0"}, use_bin_type=True
    )
    sock = _FakeSocket([server_ok])
    conn = _make_connection(sock)

    conn._ensure_handshaked()
    assert conn._handshake_done is True
    hello = msgpack.unpackb(sock.sent[0], raw=False)
    assert hello["kwargs"]["client_version"] == "0.8.0rc0"
