"""FlopscopeServer -- ZMQ REP loop that dispatches requests to a Session + RequestHandler."""

from __future__ import annotations

import secrets
import sys
from time import monotonic, perf_counter_ns

import msgpack
import zmq

import flopscope
from flopscope_server._connection_store import ConnectionStore
from flopscope_server._protocol import (
    InvalidRequestError,
    decode_request,
    encode_error_response,
    encode_response,
    validate_request,
)
from flopscope_server._request_handler import RequestHandler
from flopscope_server._session import Session


class FlopscopeServer:
    """ZMQ REP server for the flopscope budget-controlled compute service.

    Parameters
    ----------
    url:
        ZMQ endpoint to bind (e.g. ``"ipc:///tmp/flopscope.sock"`` or
        ``"tcp://127.0.0.1:15555"``).
    session_timeout_s:
        Seconds of inactivity after which an open session is reaped.
    """

    def __init__(
        self,
        url: str = "ipc:///tmp/flopscope.sock",
        session_timeout_s: float = 60.0,
        control_token: str | None = None,
    ) -> None:
        self._url = url
        self._session_timeout_s = session_timeout_s
        self._control_token = control_token
        self._running = False
        self._session: Session | None = None
        self._handler: RequestHandler | None = None
        self._last_activity: float = 0.0
        # Array + generator handle storage lives for the whole connection
        # (process), not the per-MLP budget session, so module-level handles
        # survive across MLPs (issue #107). Injected into every Session below.
        self._conn_store = ConnectionStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the blocking ZMQ REP loop.

        Creates a ZMQ Context and REP socket, binds to ``self._url``, and
        loops until :meth:`stop` is called.
        """
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1 s poll interval
        sock.bind(self._url)
        self._running = True

        try:
            while self._running:
                try:
                    raw = sock.recv()
                except zmq.Again:
                    self._reap_session_if_stale()
                    continue

                response = self._process_request(raw)
                sock.send(response)
        finally:
            sock.close(linger=0)
            ctx.term()

    def stop(self) -> None:
        """Signal the run loop to exit."""
        self._running = False

    # ------------------------------------------------------------------
    # Request processing
    # ------------------------------------------------------------------

    def _process_request(self, raw: bytes) -> bytes:
        """Decode, dispatch, encode a single request.

        Returns msgpack bytes ready to send back to the client.
        """
        t0 = perf_counter_ns()

        # --- Decode ---
        try:
            msg = decode_request(raw)
        except InvalidRequestError as exc:
            return encode_error_response("InvalidRequestError", str(exc))

        # Normalise remaining bytes→str fields that decode_request leaves as
        # raw bytes (it only converts _STRING_FIELDS: op, dtype, request_id).
        _normalize_msg(msg)

        t1 = perf_counter_ns()
        op = msg["op"]

        # --- Pre-session ops (no active session required) ---
        if op == "hello":
            return self._handle_hello(msg)

        # --- Session lifecycle ops (privileged: require the control token) ---
        if op in ("budget_open", "budget_close"):
            if not self._control_token_ok(msg):
                return encode_error_response(
                    "UnauthorizedControlError",
                    "session lifecycle is grader-controlled; participant code "
                    "cannot open, close, or reset a budget",
                )
            if op == "budget_open":
                return self._handle_budget_open(msg, t0, t1)
            return self._handle_budget_close(t0, t1)

        # --- Require active session for everything else ---
        if self._session is None:
            return encode_error_response(
                "NoBudgetContextError",
                "no active session -- send budget_open first",
            )

        # --- Validate (whitelist check) ---
        try:
            validate_request(msg)
        except InvalidRequestError as exc:
            return encode_error_response("InvalidRequestError", str(exc))

        # --- Dispatch ---
        assert self._handler is not None
        result = self._handler.handle(msg)

        t2 = perf_counter_ns()

        # --- Encode response ---
        is_fetch = op in ("fetch", "fetch_slice")
        response_bytes: bytes

        try:
            if result.get("status") == "error":
                response_bytes = encode_error_response(
                    result["error_type"], result["message"]
                )
            elif "data" in result:
                # fetch / fetch_slice -- use raw response packing
                payload = {
                    "status": "ok",
                    "data": result["data"],
                    "shape": result["shape"],
                    "dtype": result["dtype"],
                    "comms_overhead_ns": 0,  # placeholder, updated below
                }
                response_bytes = msgpack.packb(payload, use_bin_type=True)
            else:
                response_bytes = encode_response(
                    result.get("result"),
                    result.get(
                        "budget", self._session.budget_remaining if self._session else 0
                    ),
                    comms_overhead_ns=0,  # placeholder
                )
        except Exception as enc_err:
            # Response serialization failed -- return error instead of crashing
            response_bytes = encode_error_response(
                "FlopscopeServerError",
                f"failed to serialize response: {type(enc_err).__name__}",
            )

        t3 = perf_counter_ns()

        # --- Record comms overhead ---
        comms_ns = (t1 - t0) + (t3 - t2)
        compute_ns = self._handler.kernel_ns

        self._session.comms_tracker.record_request(
            bytes_received=len(raw),
            bytes_sent=len(response_bytes),
            comms_overhead_ns=comms_ns,
            compute_time_ns=compute_ns,
            is_fetch=is_fetch,
        )

        self._last_activity = monotonic()
        return response_bytes

    # ------------------------------------------------------------------
    # Token gate
    # ------------------------------------------------------------------

    def _control_token_ok(self, msg: dict) -> bool:
        """True if the request carries the configured control token.

        When no token is configured (``control_token is None`` — local/dev or
        the server's own test suite), control is unrestricted. Production always
        configures one via ``--token-fd``.
        """
        if self._control_token is None:
            return True
        token = msg.get("control_token")
        if token is None:
            kwargs = msg.get("kwargs") or {}
            token = kwargs.get("control_token")
        # _normalize_arg only ASCII-decodes byte strings of length ≤ 32; a
        # 64-char token_hex(32) exceeds that threshold and stays as bytes even
        # after _normalize_msg.  Decode it here so compare_digest gets two strs.
        if isinstance(token, bytes):
            try:
                token = token.decode("utf-8")
            except UnicodeDecodeError:
                return False
        if not isinstance(token, str):
            return False
        return secrets.compare_digest(token, self._control_token)

    # ------------------------------------------------------------------
    # Budget lifecycle handlers
    # ------------------------------------------------------------------

    def _handle_budget_open(self, msg: dict, t0: int, t1: int) -> bytes:
        """Open a new session; error if one is already open."""
        if self._session is not None and self._session.is_open:
            return encode_error_response(
                "RuntimeError",
                "session already open -- send budget_close first",
            )

        # flop_budget may be top-level or nested in kwargs. There is no
        # multiplier: the server never scales op costs by a client-supplied
        # factor (that was the budget-bypass vector). Cost = flop_cost × weight.
        flop_budget = msg.get("flop_budget")
        if flop_budget is None:
            kwargs = msg.get("kwargs") or {}
            flop_budget = kwargs.get("flop_budget", 1_000_000)
        self._session = Session(flop_budget=flop_budget, conn_store=self._conn_store)
        self._handler = RequestHandler(self._session)
        self._last_activity = monotonic()

        t2 = perf_counter_ns()
        response_bytes = encode_response(
            {"session": "opened", "flop_budget": flop_budget},
            budget=flop_budget,
            comms_overhead_ns=0,
        )
        t3 = perf_counter_ns()

        comms_ns = (t1 - t0) + (t3 - t2)
        compute_ns = 0  # session creation is not a numpy kernel
        self._session.comms_tracker.record_request(
            bytes_received=0,
            bytes_sent=len(response_bytes),
            comms_overhead_ns=comms_ns,
            compute_time_ns=compute_ns,
            is_fetch=False,
        )
        return response_bytes

    def _handle_budget_close(self, t0: int, t1: int) -> bytes:
        """Close the active session and return a summary."""
        if self._session is None or not self._session.is_open:
            return encode_error_response(
                "NoBudgetContextError",
                "no active session to close",
            )

        summary = self._session.close()
        self._session = None
        self._handler = None

        t2 = perf_counter_ns()
        response_bytes = encode_response(summary, budget=0, comms_overhead_ns=0)
        t3 = perf_counter_ns()
        # No session to record to (already closed); that's fine.
        return response_bytes

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    def _handle_hello(self, msg: dict) -> bytes:
        """Compare client and server flopscope versions.

        Returns an ok response carrying the server's leading X.Y.Z when
        the client reports a matching version, or a ``VersionMismatch``
        error otherwise. The handshake is intentionally session-free so
        clients can detect mismatch before opening a budget.
        """
        kwargs = msg.get("kwargs") or {}
        client_version = (
            kwargs.get("client_version") if isinstance(kwargs, dict) else None
        )
        server_xyz = flopscope.__version__.split("+", 1)[0]

        if not isinstance(client_version, str) or not client_version:
            return encode_error_response(
                "VersionMismatch",
                f"client did not report a flopscope version; server is {server_xyz}",
            )

        if client_version != server_xyz:
            return encode_error_response(
                "VersionMismatch",
                (
                    f"flopscope-client {client_version} cannot talk to "
                    f"flopscope-server {server_xyz}: versions must match. "
                    "Install matched versions: see "
                    "https://pypi.org/project/flopscope/ for the latest."
                ),
            )

        return msgpack.packb(
            {"status": "ok", "server_version": server_xyz},
            use_bin_type=True,
        )

    # ------------------------------------------------------------------
    # Session reaping
    # ------------------------------------------------------------------

    def _reap_session_if_stale(self) -> None:
        """Close and discard the session if it has been idle too long."""
        if self._session is None or not self._session.is_open:
            return
        if monotonic() - self._last_activity > self._session_timeout_s:
            print(
                "[flopscope-server] session timed out -- reaping",
                file=sys.stderr,
            )
            self._session.close()
            self._session = None
            self._handler = None


# ---------------------------------------------------------------------------
# Message normalisation helper
# ---------------------------------------------------------------------------


def _decode_if_bytes(v: object) -> object:
    """Decode bytes to str if possible, otherwise return as-is."""
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            return v
    return v


def _normalize_arg(a: object) -> object:
    """Normalize a single arg: decode short bytes to str, normalize dict keys/values.

    Binary data payloads (array bytes) must stay as bytes.  Heuristic: only
    decode if short AND contains no null bytes (handles, dtype strings, etc.
    are always short ASCII).
    """
    if isinstance(a, bytes):
        # Only decode bytes that are short AND contain only ASCII printable
        # characters (no bytes > 127).  This catches handle IDs and dtype
        # strings but never touches binary array data (which often contains
        # high bytes even if it happens to be valid UTF-8).
        if len(a) > 0 and len(a) <= 32 and all(32 <= b < 128 for b in a):
            return a.decode("ascii")
        return a  # keep as raw bytes (likely binary payload)
    if isinstance(a, dict):
        return {_decode_if_bytes(k): _normalize_arg(v) for k, v in a.items()}
    if isinstance(a, list):
        return [_normalize_arg(x) for x in a]
    return a


def _normalize_msg(msg: dict) -> None:
    """In-place normalise a decoded request dict.

    ``decode_request`` only converts a small set of known string fields
    (op, dtype, request_id) from bytes to str.  Other string-valued
    fields -- ``id``, ``ids``, and items inside ``args`` -- may still be
    bytes after msgpack decoding with ``raw=True``.  This helper converts
    them so the downstream RequestHandler receives plain strings.
    """
    # id (used by fetch, fetch_slice, free)
    if "id" in msg:
        msg["id"] = _decode_if_bytes(msg["id"])

    # ids (used by free)
    if "ids" in msg and isinstance(msg["ids"], list):
        msg["ids"] = [_decode_if_bytes(x) for x in msg["ids"]]

    # args -- handle IDs are short ASCII strings, may also contain dicts
    if "args" in msg and isinstance(msg["args"], list):
        msg["args"] = [_normalize_arg(a) for a in msg["args"]]

    # kwargs keys are already str (decode_request normalises all keys).
    # kwargs *values* may contain handles, dicts, or lists that need full
    # normalization (not just _decode_if_bytes).
    if "kwargs" in msg and isinstance(msg["kwargs"], dict):
        msg["kwargs"] = {
            _decode_if_bytes(k): _normalize_arg(v) for k, v in msg["kwargs"].items()
        }
