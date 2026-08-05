"""ZMQ REQ socket wrapper for the flopscope client."""

from __future__ import annotations

import os
import time
from typing import NoReturn

import zmq

from flopscope._dispatch import dispatch_span
from flopscope._handles import drain_pending
from flopscope._protocol import decode_response, encode_free
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
        self._capabilities: frozenset[str] = frozenset()
        self._flushing_frees: bool = False

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

    def _fail_handshake(
        self, message: str, *, cause: BaseException | None = None
    ) -> NoReturn:
        """Reset invalid handshake state and raise a client-facing error."""
        self._reset_socket()
        raise ConnectionError(message) from cause

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

        try:
            response_obj: object = decode_response(raw)
        except Exception as err:
            self._fail_handshake(
                f"malformed handshake response from server: {err}", cause=err
            )
        if not isinstance(response_obj, dict):
            self._fail_handshake(
                "malformed handshake response from server: expected a mapping, "
                f"got {response_obj!r}"
            )
        response = response_obj
        if (
            response.get("status") == "error"
            and response.get("error_type") == "VersionMismatch"
        ):
            self._fail_handshake(
                f"flopscope-client {client_xyz} cannot talk to this server: "
                f"{response.get('message', 'version mismatch')}"
            )
        if response.get("status") != "ok":
            self._fail_handshake(
                f"unexpected handshake response from server: {response!r}"
            )
        server_version = response.get("server_version")
        if not isinstance(server_version, str) or not server_version:
            self._fail_handshake(
                "malformed server_version in server handshake: expected a "
                f"non-empty string matching {client_xyz!r}, got {server_version!r}. "
                "Install matching client/server versions."
            )
        if server_version != client_xyz:
            self._fail_handshake(
                f"flopscope-client {client_xyz} cannot talk to flopscope-server "
                f"{server_version}: versions must match. Install matching "
                "client/server versions."
            )
        raw_capabilities = response.get("capabilities", [])
        if not isinstance(raw_capabilities, list) or not all(
            isinstance(value, str) for value in raw_capabilities
        ):
            self._fail_handshake(
                f"malformed capabilities in server handshake: {raw_capabilities!r}"
            )
        self._capabilities = frozenset(raw_capabilities)
        self._handshake_done = True

    def require_capability(self, name: str) -> None:
        """Raise an actionable error when the connected server lacks *name*."""
        self._ensure_connected()
        self._ensure_handshaked()
        if name not in self._capabilities:
            from flopscope.errors import FlopscopeServerError

            raise FlopscopeServerError(
                "budget_summary_dict() requires authoritative remote-summary "
                "support. The connected FLOPScope server does not provide it; "
                "install matching client/server versions or avoid this accessor "
                "in prediction code."
            )

    def _flush_pending_frees(self) -> None:
        """Release GC'd server handles, batched onto the current round-trip.

        Enqueued handle ids (from RemoteArray finalizers) are sent as a single
        ``free`` op before the real request. Called from inside send_recv's
        ``dispatch_span`` so the tiny round-trip bills as flopscope overhead,
        not participant residual. Best-effort: on ANY error the socket is reset
        so the REQ/REP state machine is clean for the request that follows.

        Lost-on-error: drain_pending() clears the queue BEFORE the send, so if
        the free send/recv fails, those handle ids are discarded (never
        re-enqueued). The server handles then linger until the session closes.
        Intentional — a missed free is a slow memory reclaim, not a correctness
        bug, and re-enqueuing risks unbounded growth on a persistently bad socket.
        """
        if self._flushing_frees:
            return
        handles = drain_pending()
        if not handles:
            return
        self._flushing_frees = True
        try:
            sock = self._ensure_connected()
            self._ensure_handshaked()
            sock.send(encode_free(handles))
            sock.recv()  # consume {status:ok}; content ignored
        except Exception:
            self._reset_socket()
        finally:
            self._flushing_frees = False

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

        with dispatch_span():
            self._flush_pending_frees()
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

            # Every compute-op response carries the server's authoritative
            # budget; fold it into the active BudgetContext so flops_used is
            # current after each op with no extra round trip. Lazy import: the
            # budget module imports this one on its own paths.
            #
            # This runs BEFORE the error check on purpose. An operation can be
            # billed and then fail -- charged by the kernel, then refused while
            # its result is packed -- so an error response's budget has already
            # advanced. Syncing after the raise would never reach those, and a
            # caller that catches the exception would go on to read a
            # flops_used from before the charge.
            from flopscope._budget import _sync_active_context_from_response

            _sync_active_context_from_response(response)

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
        self._capabilities = frozenset()

    def close(self) -> None:
        """Close the ZMQ socket, if open."""
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._handshake_done = False
        self._capabilities = frozenset()


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
