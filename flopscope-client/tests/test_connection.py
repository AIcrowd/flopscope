"""Tests for flopscope._connection and flopscope._protocol."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import msgpack
import pytest

# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


class TestProtocol:
    """Tests for message encoding/decoding functions."""

    def test_encode_request_basic(self):
        from flopscope._protocol import encode_request

        raw = encode_request("dot")
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded == {"op": "dot", "args": None, "kwargs": None}

    def test_encode_request_with_args(self):
        from flopscope._protocol import encode_request

        raw = encode_request("add", args=[1, 2])
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded["op"] == "add"
        assert decoded["args"] == [1, 2]
        assert decoded["kwargs"] is None

    def test_encode_request_with_kwargs(self):
        from flopscope._protocol import encode_request

        raw = encode_request("solve", kwargs={"assume_a": "pos"})
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded["op"] == "solve"
        assert decoded["args"] is None
        assert decoded["kwargs"] == {"assume_a": "pos"}

    def test_encode_request_with_args_and_kwargs(self):
        from flopscope._protocol import encode_request

        raw = encode_request("op", args=[1], kwargs={"k": "v"})
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded["args"] == [1]
        assert decoded["kwargs"] == {"k": "v"}

    def test_encode_create_from_data(self):
        from flopscope._protocol import encode_create_from_data

        data = b"\x00\x01\x02\x03"
        raw = encode_create_from_data(data, [2, 2], "float64")
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded["op"] == "create_from_data"
        args = decoded["args"]
        assert args[0] == data
        assert args[1] == [2, 2]
        assert args[2] == "float64"

    def test_encode_budget_open(self):
        from flopscope._protocol import encode_budget_open

        raw = encode_budget_open(1000)
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded["op"] == "budget_open"
        assert decoded["kwargs"]["flop_budget"] == 1000

    def test_encode_budget_close(self):
        from flopscope._protocol import encode_budget_close

        raw = encode_budget_close()
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded["op"] == "budget_close"

    def test_encode_budget_status(self):
        from flopscope._protocol import encode_budget_status

        raw = encode_budget_status()
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded["op"] == "budget_status"

    def test_encode_fetch(self):
        from flopscope._protocol import encode_fetch

        raw = encode_fetch("handle-abc")
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded["op"] == "fetch"
        assert decoded["kwargs"]["handle_id"] == "handle-abc"

    def test_encode_free(self):
        from flopscope._protocol import encode_free

        raw = encode_free(["h1", "h2", "h3"])
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded["op"] == "free"
        assert decoded["kwargs"]["handles"] == ["h1", "h2", "h3"]

    def test_decode_response_success(self):
        from flopscope._protocol import decode_response

        payload = msgpack.packb({"status": "ok", "result": 42})
        result = decode_response(payload)
        assert result["status"] == "ok"
        assert result["result"] == 42

    def test_decode_response_normalizes_bytes_keys(self):
        from flopscope._protocol import decode_response

        # msgpack raw=True would give bytes keys; simulate that
        payload = msgpack.packb(
            {"status": "ok", "handle": "abc-123"}, use_bin_type=True
        )
        result = decode_response(payload)
        assert "status" in result
        assert result["status"] == "ok"

    def test_decode_response_error_structure(self):
        from flopscope._protocol import decode_response

        payload = msgpack.packb(
            {"status": "error", "error_type": "ValueError", "message": "bad input"},
            use_bin_type=True,
        )
        result = decode_response(payload)
        assert result["status"] == "error"
        assert result["error_type"] == "ValueError"
        assert result["message"] == "bad input"

    def test_decode_response_keeps_binary_data_as_bytes(self):
        from flopscope._protocol import decode_response

        # Binary data (e.g. raw array bytes) should stay as bytes
        binary_payload = bytes(range(256))  # 256 bytes of raw binary
        payload = msgpack.packb(
            {"status": "ok", "data": binary_payload},
            use_bin_type=True,
        )
        result = decode_response(payload)
        assert isinstance(result["data"], bytes)
        assert result["data"] == binary_payload

    def test_decode_response_short_string_becomes_str(self):
        from flopscope._protocol import decode_response

        # Short readable string should be decoded as str
        payload = msgpack.packb(
            {"status": "ok", "handle": "abc-handle-123"},
            use_bin_type=True,
        )
        result = decode_response(payload)
        assert isinstance(result["handle"], str)

    def test_decode_response_allows_unscoped_namespace_keys(self):
        from flopscope._protocol import decode_response

        payload = msgpack.packb(
            {
                "status": "ok",
                "result": {
                    "budget_breakdown": {
                        "by_namespace": {
                            None: {"flops_used": 1, "calls": 1},
                            "phase": {"flops_used": 2, "calls": 1},
                        }
                    }
                },
            },
            use_bin_type=True,
        )
        result = decode_response(payload)
        assert (
            result["result"]["budget_breakdown"]["by_namespace"][None]["flops_used"]
            == 1
        )
        assert (
            result["result"]["budget_breakdown"]["by_namespace"]["phase"]["calls"] == 1
        )

    def test_normalize_preserves_small_binary_data(self):
        """FIX 2 (client): small binary array data must NOT be decoded."""
        import struct

        from flopscope._protocol import _normalize

        data = struct.pack("<d", 3.14)  # 8 bytes float64
        result = _normalize(data)
        assert isinstance(result, bytes), "binary float64 data was decoded to str"

    def test_normalize_decodes_ascii_handle(self):
        """FIX 2 (client): short ASCII handle IDs are decoded to str."""
        from flopscope._protocol import _normalize

        result = _normalize(b"a0")
        assert result == "a0"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Connection config tests
# ---------------------------------------------------------------------------


class TestConnectionConfig:
    """Tests for Connection configuration and URL resolution."""

    def test_default_url(self):
        from flopscope._connection import Connection

        # Clear env var if set
        env = {k: v for k, v in os.environ.items() if k != "FLOPSCOPE_SERVER_URL"}
        with patch.dict(os.environ, env, clear=True):
            conn = Connection()
            assert conn.url == "ipc:///tmp/flopscope.sock"

    def test_url_from_env_var(self):
        from flopscope._connection import Connection

        with patch.dict(os.environ, {"FLOPSCOPE_SERVER_URL": "tcp://localhost:5555"}):
            conn = Connection()
            assert conn.url == "tcp://localhost:5555"

    def test_url_from_argument_takes_precedence(self):
        from flopscope._connection import Connection

        with patch.dict(os.environ, {"FLOPSCOPE_SERVER_URL": "tcp://localhost:5555"}):
            conn = Connection(url="ipc:///tmp/other.sock")
            assert conn.url == "ipc:///tmp/other.sock"

    def test_custom_timeout(self):
        from flopscope._connection import Connection

        conn = Connection(timeout_ms=5000)
        assert conn.timeout_ms == 5000

    def test_default_timeout(self):
        from flopscope._connection import Connection

        conn = Connection()
        assert conn.timeout_ms == 30000

    def test_close_without_connect(self):
        """close() on a never-connected Connection should not raise."""
        from flopscope._connection import Connection

        conn = Connection()
        conn.close()  # should not raise


# ---------------------------------------------------------------------------
# Module-level singleton tests
# ---------------------------------------------------------------------------


class TestConnectionSingleton:
    """Tests for get_connection/reset_connection."""

    def test_get_connection_returns_connection(self):
        from flopscope._connection import Connection, get_connection, reset_connection

        reset_connection()
        conn = get_connection()
        assert isinstance(conn, Connection)

    def test_get_connection_returns_same_instance(self):
        from flopscope._connection import get_connection, reset_connection

        reset_connection()
        conn1 = get_connection()
        conn2 = get_connection()
        assert conn1 is conn2

    def test_reset_connection_creates_new_instance(self):
        from flopscope._connection import get_connection, reset_connection

        reset_connection()
        conn1 = get_connection()
        reset_connection()
        conn2 = get_connection()
        assert conn1 is not conn2

    def test_reset_connection_closes_old(self):
        """reset_connection should close the previous socket if connected."""
        from flopscope._connection import get_connection, reset_connection

        reset_connection()
        conn = get_connection()
        conn.close = MagicMock()
        # Manually inject it back so reset_connection sees it
        import flopscope._connection as mod

        mod._connection = conn
        reset_connection()
        conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# send_recv tests (mock ZMQ socket)
# ---------------------------------------------------------------------------


class TestSendRecv:
    """Tests for Connection.send_recv with a mocked ZMQ socket."""

    def _make_response(self, payload: dict) -> bytes:
        return msgpack.packb(payload, use_bin_type=True)

    def test_send_recv_success(self):
        from flopscope._connection import Connection
        from flopscope._protocol import encode_request

        conn = Connection()
        mock_socket = MagicMock()
        mock_socket.recv.return_value = self._make_response(
            {"status": "ok", "result": 99}
        )
        conn._socket = mock_socket
        conn._handshake_done = True  # bypass the lazy version handshake

        raw = encode_request("test_op")
        result = conn.send_recv(raw)
        assert result["status"] == "ok"
        assert result["result"] == 99
        mock_socket.send.assert_called_once_with(raw)

    def test_send_recv_tracks_timing(self):
        from flopscope._connection import Connection
        from flopscope._protocol import encode_request

        conn = Connection()
        mock_socket = MagicMock()
        mock_socket.recv.return_value = self._make_response({"status": "ok"})
        conn._socket = mock_socket
        conn._handshake_done = True  # bypass the lazy version handshake

        raw = encode_request("test_op")
        result = conn.send_recv(raw)
        assert "_round_trip_ns" in result
        assert "_request_bytes" in result
        assert "_response_bytes" in result
        assert result["_request_bytes"] == len(raw)

    def test_send_recv_raises_on_error_response(self):
        from flopscope._connection import Connection
        from flopscope._protocol import encode_request

        from flopscope.errors import FlopscopeServerError

        conn = Connection()
        mock_socket = MagicMock()
        mock_socket.recv.return_value = self._make_response(
            {"status": "error", "error_type": "FlopscopeServerError", "message": "boom"}
        )
        conn._socket = mock_socket
        conn._handshake_done = True  # bypass the lazy version handshake

        raw = encode_request("op")
        with pytest.raises(FlopscopeServerError, match="boom"):
            conn.send_recv(raw)

    def test_send_recv_raises_budget_exhausted(self):
        from flopscope._connection import Connection
        from flopscope._protocol import encode_request

        from flopscope.errors import BudgetExhaustedError

        conn = Connection()
        mock_socket = MagicMock()
        mock_socket.recv.return_value = self._make_response(
            {
                "status": "error",
                "error_type": "BudgetExhaustedError",
                "message": "Budget exhausted: operation 'dot' requires 100 FLOPs but only 0 FLOPs remain.",
            }
        )
        conn._socket = mock_socket
        conn._handshake_done = True  # bypass the lazy version handshake

        raw = encode_request("dot")
        with pytest.raises(BudgetExhaustedError):
            conn.send_recv(raw)
