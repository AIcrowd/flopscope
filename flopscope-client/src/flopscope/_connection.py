"""ZMQ REQ socket wrapper for the flopscope client."""

from __future__ import annotations

import os
import time

import zmq

from flopscope._protocol import decode_response
from flopscope.errors import raise_from_response

_DEFAULT_URL = "ipc:///tmp/flopscope.sock"
_DEFAULT_TIMEOUT_MS = 30_000


class Connection:
    """Lazy-connecting ZMQ REQ socket wrapper.

    Parameters
    ----------
    url:
        ZMQ endpoint to connect to.  Defaults to the ``FLOPSCOPE_SERVER_URL``
        environment variable, or ``ipc:///tmp/flopscope.sock`` if unset.
    timeout_ms:
        Send/receive timeout in milliseconds.  Defaults to 30 000 ms.
    """

    def __init__(
        self, url: str | None = None, timeout_ms: int = _DEFAULT_TIMEOUT_MS
    ) -> None:
        self.url: str = url or os.environ.get("FLOPSCOPE_SERVER_URL", _DEFAULT_URL)
        self.timeout_ms: int = timeout_ms
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._handshake_done: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> zmq.Socket:
        """Return an open, connected REQ socket (lazy initialisation)."""
        if self._socket is not None:
            return self._socket
        self._context = zmq.Context.instance()
        sock = self._context.socket(zmq.REQ)
        sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        sock.connect(self.url)
        self._socket = sock
        return sock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _ensure_handshaked(self) -> None:
        """Send a `hello` once; raise ConnectionError on version mismatch.

        Compares the client's leading X.Y.Z (`flopscope.__version__` minus
        any `+np<numpy>` suffix) against the server. On success, sets
        ``self._handshake_done = True``. Idempotent: subsequent calls
        return immediately without sending again.

        Raised exceptions
        -----------------
        ConnectionError
            If the server reports VersionMismatch, returns a malformed
            response, or fails to respond.
        """
        if self._handshake_done:
            return

        from flopscope import __version__ as _flopscope_version
        from flopscope._protocol import encode_hello

        client_xyz = _flopscope_version.split("+", 1)[0]
        sock = self._ensure_connected()

        try:
            sock.send(encode_hello(client_xyz))
            raw = sock.recv()
        except zmq.Again as err:
            self._reset_socket()
            raise ConnectionError(
                f"flopscope-client {client_xyz} handshake timeout: "
                "server did not respond"
            ) from err

        response = decode_response(raw)
        if (
            response.get("status") == "error"
            and response.get("error_type") == "VersionMismatch"
        ):
            raise ConnectionError(
                f"flopscope-client {client_xyz} cannot talk to this server: "
                f"{response.get('message', 'version mismatch')}"
            )
        if response.get("status") != "ok":
            raise ConnectionError(
                f"unexpected handshake response from server: {response!r}"
            )
        self._handshake_done = True

    def send_recv(self, raw_request: bytes) -> dict:
        """Send *raw_request* and return the decoded response dict.

        On the first call against a fresh Connection, performs a lazy
        version handshake with the server (a `hello` op carrying the
        client's leading X.Y.Z) before forwarding the real request.

        Augments the response with timing/size metadata:

        * ``_round_trip_ns`` – round-trip time in nanoseconds
        * ``_request_bytes``  – size of the outgoing payload in bytes
        * ``_response_bytes`` – size of the incoming payload in bytes

        Raises the appropriate exception if the response has
        ``"status": "error"``.
        """
        from flopscope.errors import FlopscopeServerError

        sock = self._ensure_connected()
        self._ensure_handshaked()

        t0 = time.monotonic_ns()
        try:
            sock.send(raw_request)
            raw_response: bytes = sock.recv()
        except zmq.Again as err:
            # Socket is in a bad state after timeout — reset it
            self._reset_socket()
            raise FlopscopeServerError(
                "server timeout: no response within timeout period"
            ) from err
        t1 = time.monotonic_ns()

        response = decode_response(raw_response)
        response["_round_trip_ns"] = t1 - t0
        response["_request_bytes"] = len(raw_request)
        response["_response_bytes"] = len(raw_response)

        if response.get("status") == "error":
            raise_from_response(
                response.get("error_type", "FlopscopeServerError"),
                response.get("message", ""),
            )

        return response

    def _reset_socket(self) -> None:
        """Close and discard the socket so it will be recreated on next use."""
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._handshake_done = False

    def close(self) -> None:
        """Close the ZMQ socket, if open."""
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._handshake_done = False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_connection: Connection | None = None


def get_connection() -> Connection:
    """Return the module-level singleton :class:`Connection`."""
    global _connection
    if _connection is None:
        _connection = Connection()
    return _connection


def reset_connection() -> None:
    """Close and discard the singleton connection (primarily for testing)."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
