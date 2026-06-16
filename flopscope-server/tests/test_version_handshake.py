"""Tests for the server-side version handshake (`hello` op).

The hello op is handled at the FlopscopeServer level (alongside
budget_open / budget_close) because it's session-lifecycle: the
handshake must succeed before any budget session can be opened.
"""

from __future__ import annotations

import msgpack
from flopscope_server import _protocol
from flopscope_server._server import FlopscopeServer

import flopscope


def _leading_xyz(v: str) -> str:
    """Strip any +np... suffix from flopscope.__version__."""
    return v.split("+", 1)[0]


def test_hello_in_protocol_ops():
    """The `hello` op must be on the permitted op whitelist."""
    assert "hello" in _protocol._PROTOCOL_OPS
    assert "hello" in _protocol.WHITELIST


def test_handle_hello_matching_version_returns_ok():
    """Matching client_version → ok response with server_version."""
    server = FlopscopeServer.__new__(FlopscopeServer)
    server._session = None
    server._handler = None
    server._last_activity = 0.0

    client_version = _leading_xyz(flopscope.__version__)
    response_bytes = server._handle_hello(
        {"op": "hello", "kwargs": {"client_version": client_version}}
    )
    decoded = msgpack.unpackb(response_bytes, raw=False)
    assert decoded["status"] == "ok"
    assert decoded["server_version"] == client_version


def test_handle_hello_mismatch_returns_version_mismatch_error():
    """Mismatched client_version → error response with VersionMismatch."""
    server = FlopscopeServer.__new__(FlopscopeServer)
    server._session = None
    server._handler = None
    server._last_activity = 0.0

    response_bytes = server._handle_hello(
        {"op": "hello", "kwargs": {"client_version": "0.0.99"}}
    )
    decoded = msgpack.unpackb(response_bytes, raw=False)
    assert decoded["status"] == "error"
    assert decoded["error_type"] == "VersionMismatch"
    server_xyz = _leading_xyz(flopscope.__version__)
    assert "0.0.99" in decoded["message"]
    assert server_xyz in decoded["message"]


def test_handle_hello_missing_client_version_is_mismatch():
    """A hello with no client_version kwarg is treated as a VersionMismatch."""
    server = FlopscopeServer.__new__(FlopscopeServer)
    server._session = None
    server._handler = None
    server._last_activity = 0.0

    response_bytes = server._handle_hello({"op": "hello", "kwargs": {}})
    decoded = msgpack.unpackb(response_bytes, raw=False)
    assert decoded["status"] == "error"
    assert decoded["error_type"] == "VersionMismatch"


def test_hello_works_before_budget_open():
    """The hello op must succeed even when no session is active."""
    server = FlopscopeServer.__new__(FlopscopeServer)
    server._session = None
    server._handler = None
    server._last_activity = 0.0

    client_version = _leading_xyz(flopscope.__version__)
    raw_request = msgpack.packb(
        {"op": "hello", "kwargs": {"client_version": client_version}},
        use_bin_type=True,
    )
    response_bytes = server._process_request(raw_request)
    decoded = msgpack.unpackb(response_bytes, raw=False)
    assert decoded["status"] == "ok"
    assert decoded["server_version"] == client_version
    # And no session was opened as a side effect.
    assert server._session is None


def test_handle_hello_prerelease_version_uses_string_compare(monkeypatch):
    """A prerelease server version (e.g. 0.8.0rc0) is matched by full public version.

    Guards that the handshake string-compares the +local-stripped version and
    never int-parses X.Y.Z (which would raise on the `rc` suffix), and that the
    full prerelease is significant: rc0 vs rc1 (and rc0 vs the final release) is
    a genuine mismatch, not a same-core match.
    """
    monkeypatch.setattr(flopscope, "__version__", "0.8.0rc0+np9.9")
    server = FlopscopeServer.__new__(FlopscopeServer)
    server._session = None
    server._handler = None
    server._last_activity = 0.0

    # Same public prerelease → ok; the +np9.9 local segment is stripped server-side.
    ok = msgpack.unpackb(
        server._handle_hello({"op": "hello", "kwargs": {"client_version": "0.8.0rc0"}}),
        raw=False,
    )
    assert ok["status"] == "ok"
    assert ok["server_version"] == "0.8.0rc0"

    # rc0 vs rc1 and rc0 vs the final release must both be VersionMismatch.
    for other in ("0.8.0rc1", "0.8.0"):
        err = msgpack.unpackb(
            server._handle_hello({"op": "hello", "kwargs": {"client_version": other}}),
            raw=False,
        )
        assert err["status"] == "error", other
        assert err["error_type"] == "VersionMismatch", other
