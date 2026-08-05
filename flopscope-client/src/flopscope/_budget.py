"""Client-side BudgetContext proxy that delegates to the flopscope server."""

from __future__ import annotations

import time
from copy import deepcopy
from math import isfinite
from typing import NoReturn, TypeGuard

from flopscope._connection import get_connection
from flopscope._dispatch import dispatch_span, total_dispatch_ns
from flopscope._protocol import (
    AUTHORITATIVE_BUDGET_SUMMARY_CAPABILITY,
    encode_budget_close,
    encode_budget_open,
    encode_budget_summary,
)

_SUMMARY_KEYS = {
    "flop_budget",
    "flops_used",
    "flops_remaining",
    "operations",
    "wall_time_s",
    "flopscope_backend_time_s",
    "flopscope_overhead_time_s",
    "residual_wall_time_s",
}

# Module-level guard: only one BudgetContext can be active at a time.
_active_context = None


def _malformed(message: str) -> NoReturn:
    from flopscope.errors import FlopscopeServerError

    raise FlopscopeServerError(f"malformed budget_summary response: {message}")


def _is_nonnegative_int(value: object) -> TypeGuard[int]:
    return type(value) is int and value >= 0


def _is_finite_nonnegative_numeric(value: object) -> bool:
    if type(value) is int:
        return value >= 0
    return type(value) is float and isfinite(value) and value >= 0.0


def _validate_operation_bucket(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "flop_cost",
        "calls",
        "flopscope_backend_time_s",
        "flopscope_overhead_time_s",
    }:
        _malformed("invalid operation bucket")
    if not _is_nonnegative_int(value["flop_cost"]) or not _is_nonnegative_int(
        value["calls"]
    ):
        _malformed("operation counts must be non-negative integers")
    if not all(
        _is_finite_nonnegative_numeric(value[name])
        for name in ("flopscope_backend_time_s", "flopscope_overhead_time_s")
    ):
        _malformed("operation timing must be finite and non-negative")


def _validate_summary_mapping(value: object, *, by_namespace: bool) -> dict:
    if not isinstance(value, dict):
        _malformed("result must be a dict")
    expected = _SUMMARY_KEYS | ({"by_namespace"} if by_namespace else set())
    if set(value) != expected:
        _malformed(f"result keys must be {sorted(expected)}")
    for name in ("flop_budget", "flops_used", "flops_remaining"):
        if not _is_nonnegative_int(value[name]):
            _malformed(f"{name} must be a non-negative integer")
    if not isinstance(value["operations"], dict):
        _malformed("operations must be a dict")
    for name, bucket in value["operations"].items():
        if not isinstance(name, str):
            _malformed("operation names must be strings")
        _validate_operation_bucket(bucket)
    for name in ("flopscope_backend_time_s", "flopscope_overhead_time_s"):
        if not _is_finite_nonnegative_numeric(value[name]):
            _malformed(f"{name} must be finite and non-negative")
    for name in ("wall_time_s", "residual_wall_time_s"):
        if value[name] is not None and not _is_finite_nonnegative_numeric(value[name]):
            _malformed(f"{name} must be finite, non-negative, or None")
    if by_namespace:
        if not isinstance(value["by_namespace"], dict):
            _malformed("by_namespace must be a dict")
        for namespace, bucket in value["by_namespace"].items():
            if namespace is not None and not isinstance(namespace, str):
                _malformed("namespace keys must be strings or None")
            if not isinstance(bucket, dict) or set(bucket) != {
                "flops_used",
                "calls",
                "flopscope_backend_time_s",
                "flopscope_overhead_time_s",
                "operations",
            }:
                _malformed("invalid namespace bucket")
            if not isinstance(bucket["operations"], dict):
                _malformed("namespace operations must be a dict")
            if not _is_nonnegative_int(bucket["flops_used"]) or not _is_nonnegative_int(
                bucket["calls"]
            ):
                _malformed("namespace counts must be non-negative integers")
            for timing_name in (
                "flopscope_backend_time_s",
                "flopscope_overhead_time_s",
            ):
                if not _is_finite_nonnegative_numeric(bucket[timing_name]):
                    _malformed("namespace timing must be finite and non-negative")
            for op_name, op_bucket in bucket["operations"].items():
                if not isinstance(op_name, str):
                    _malformed("namespace operation names must be strings")
                _validate_operation_bucket(op_bucket)
    return value


def _validate_display_totals(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "has_explicit_budget",
        "budget",
        "used",
        "client_context_compute_ns",
    }:
        _malformed("invalid display_totals")
    if not isinstance(value["has_explicit_budget"], bool):
        _malformed("has_explicit_budget must be boolean")
    if not _is_nonnegative_int(value["budget"]) or not _is_nonnegative_int(
        value["used"]
    ):
        _malformed("display totals must be non-negative integers")
    compute_ns = value["client_context_compute_ns"]
    if compute_ns is not None and not _is_nonnegative_int(compute_ns):
        _malformed("client context compute time must be a non-negative integer or null")
    return value


def _validate_budget_metadata(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "flop_budget",
        "flops_used",
        "flops_remaining",
    }:
        _malformed("invalid budget metadata")
    if not all(_is_nonnegative_int(value[name]) for name in value):
        _malformed("budget metadata must contain non-negative integers")


def _validate_summary_response(
    value: object, *, by_namespace: bool
) -> tuple[dict, dict]:
    if not isinstance(value, dict):
        _malformed("response must be a dict")
    required = {"status", "result", "display_totals", "comms_overhead_ns"}
    optional = {
        "budget",
        "_round_trip_ns",
        "_request_bytes",
        "_response_bytes",
    }
    keys = set(value)
    if not required <= keys or not keys <= required | optional:
        _malformed("invalid response envelope keys")
    if value["status"] != "ok":
        _malformed("status must be 'ok'")
    if type(value["comms_overhead_ns"]) is not int or value["comms_overhead_ns"] != 0:
        _malformed("comms_overhead_ns must be the integer 0")
    if "budget" in value:
        _validate_budget_metadata(value["budget"])
    for name in ("_round_trip_ns", "_request_bytes", "_response_bytes"):
        if name in value and not _is_nonnegative_int(value[name]):
            _malformed(f"{name} must be a non-negative integer")
    summary = _validate_summary_mapping(value["result"], by_namespace=by_namespace)
    display_totals = _validate_display_totals(value["display_totals"])
    return summary, display_totals


def _sync_active_context_from_response(response: object) -> None:
    """Refresh the active budget's cached ``flops_used`` from an op response.

    Every compute-op response carries the server's authoritative budget under a
    ``"budget"`` key. Consuming it here keeps :attr:`BudgetContext.flops_used`
    current after every operation with no extra round trip, so a caller that
    inspects it between ops sees a live value rather than a stale one.

    Responses without a ``"budget"`` key — fetches, frees, the version
    handshake, and the budget-lifecycle ops, which manage the cache through
    their own paths — are ignored. Called from the single send/recv chokepoint,
    so it must stay cheap and never raise.
    """
    if _active_context is None:
        return
    if isinstance(response, dict):
        budget_info = response.get("budget")
        if isinstance(budget_info, dict):
            _active_context._update_budget(budget_info)


def _extract_compute_ns(close_response: object) -> int:
    """Pull total server compute (ns) out of a ``budget_close`` response.

    Returns 0 if the ``result.comms_summary.total_compute_time_ns`` path is
    absent. A present value must be an exact non-negative integer.
    """
    if not isinstance(close_response, dict):
        return 0
    result = close_response.get("result")
    if not isinstance(result, dict):
        return 0
    comms = result.get("comms_summary")
    if not isinstance(comms, dict):
        return 0
    if "total_compute_time_ns" not in comms:
        return 0
    compute_ns = comms["total_compute_time_ns"]
    if not _is_nonnegative_int(compute_ns):
        _malformed("client context compute time must be a non-negative integer")
    return compute_ns


def _validated_close_summary(response: object) -> dict:
    if not isinstance(response, dict):
        _malformed("budget_close response must be a dict")
    if response.get("status") != "ok":
        _malformed("budget_close status must be 'ok'")
    result = response.get("result")
    if not isinstance(result, dict):
        _malformed("budget_close result must be a dict")
    return _validate_summary_mapping(result.get("budget_breakdown"), by_namespace=True)


def _decompose_timing(
    wall_ns: int, dispatch_ns: int, kernel_ns: int
) -> tuple[float, float, float, float]:
    """Decompose context wall into (wall, backend, overhead, residual) seconds.

    - backend  = pure server numpy kernel (``kernel_ns``)
    - overhead = all other flopscope machinery: client dispatch + wire +
      server marshaling/store/framing = ``dispatch − kernel``; not billed
    - residual = the participant's own Python = ``wall − dispatch``; billed

    Each is clamped to >= 0 for cross-clock skew. In the normal regime
    (``kernel <= dispatch <= wall``) no clamp fires and
    ``wall == backend + overhead + residual`` exactly.
    """
    wall_s = wall_ns / 1e9
    backend_s = max(0, kernel_ns) / 1e9
    dispatch_s = dispatch_ns / 1e9
    overhead_s = max(0.0, dispatch_s - backend_s)
    residual_s = max(0.0, wall_s - dispatch_s)
    return wall_s, backend_s, overhead_s, residual_s


class OpRecord:
    """Record of a single operation's FLOP cost.

    Parameters
    ----------
    op_name:
        Name of the operation (e.g. ``"dot"``).
    flop_cost:
        FLOPs charged for this operation.
    cumulative:
        Total FLOPs used after this operation.
    """

    def __init__(self, op_name: str, flop_cost: int, cumulative: int) -> None:
        self.op_name = op_name
        self.flop_cost = flop_cost
        self.cumulative = cumulative

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OpRecord(op_name={self.op_name!r}, "
            f"flop_cost={self.flop_cost}, cumulative={self.cumulative})"
        )


class BudgetContext:
    """Context manager that opens/closes a FLOP budget on the server.

    Parameters
    ----------
    flop_budget:
        Maximum FLOPs allowed within this context.
    quiet:
        If ``True``, suppress informational output.  Defaults to ``False``.
    namespace:
        Optional label for grouping budget records.

    Example
    -------
    >>> with BudgetContext(flop_budget=1_000_000) as ctx:
    ...     result = flopscope.dot(a, b)
    ...     print(ctx.summary())
    """

    def __init__(
        self,
        flop_budget: int,
        quiet: bool = False,
        namespace: str | None = None,
    ) -> None:
        self._flop_budget = flop_budget
        self._quiet = quiet
        self._namespace = namespace
        self._flops_used: int = 0
        self._closed_summary: dict | None = None
        self._is_open: bool = False
        self._previous_context = None
        # Timing split — populated on __exit__. None until then for wall/residual,
        # 0.0 for backend/overhead, mirroring the in-process flopscope contract.
        self._wall_time_s: float | None = None
        self._flopscope_backend_time: float = 0.0
        self._flopscope_overhead_time: float = 0.0
        self._residual_wall_time: float | None = None
        self._wall_start_ns: int | None = None
        self._dispatch_baseline_ns: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def flop_budget(self) -> int:
        """Maximum FLOP allowance for this context."""
        return self._flop_budget

    @property
    def flops_used(self) -> int:
        """FLOPs consumed so far.

        Kept current from the budget the server returns on every operation, so
        this reflects the count as of the most recent op without an extra round
        trip. Reading it between operations is safe — it is not stale.
        """
        return self._flops_used

    @property
    def flops_remaining(self) -> int:
        """FLOPs remaining in the budget (``budget - used``).

        Tracks :attr:`flops_used`, so it too is current as of the most recent
        operation.
        """
        return self._flop_budget - self._flops_used

    @property
    def quiet(self) -> bool:
        """Whether informational output is suppressed."""
        return self._quiet

    @property
    def namespace(self) -> str | None:
        """Optional namespace label for this context."""
        return self._namespace

    @property
    def wall_time_s(self) -> float | None:
        """Total wall-clock seconds spanned by the context (None until closed)."""
        return self._wall_time_s

    @property
    def flopscope_backend_time_s(self) -> float:
        """Seconds of real op compute on the server (0.0 until closed)."""
        return self._flopscope_backend_time

    @property
    def flopscope_overhead_time_s(self) -> float:
        """Seconds of flopscope transport overhead — serialization + network +
        server-side comms (0.0 until closed). Not billed."""
        return self._flopscope_overhead_time

    @property
    def residual_wall_time_s(self) -> float | None:
        """Seconds of participant Python outside flopscope calls (None until
        closed). The billed bucket: C_m = F_m + lambda * residual."""
        return self._residual_wall_time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_budget(self, budget_info: dict) -> None:
        """Update the local ``flops_used`` cache from a server-response dict.

        Parameters
        ----------
        budget_info:
            Dict that may contain a ``"flops_used"`` key. Missing or malformed
            values are silently ignored: this synchronization hook runs before
            response errors are raised and therefore must never mask the
            intended client-facing error.

        Spent FLOPs never come back, so the cache only ever moves up. Requests
        are synchronous, so in practice values arrive in order and the clamp
        does nothing -- but this cache is now refreshed from error responses
        too, and a cache that could move down would be a way to read a budget
        as less spent than it is.
        """
        flops_used = budget_info.get("flops_used")
        if _is_nonnegative_int(flops_used):
            self._flops_used = max(self._flops_used, flops_used)

    def _empty_summary(self, *, by_namespace: bool) -> dict:
        result = {
            "flop_budget": self._flop_budget,
            "flops_used": 0,
            "flops_remaining": self._flop_budget,
            "operations": {},
            "wall_time_s": None,
            "flopscope_backend_time_s": 0.0,
            "flopscope_overhead_time_s": 0.0,
            "residual_wall_time_s": None,
        }
        if by_namespace:
            result["by_namespace"] = {}
        return result

    def _update_live_from_summary(self, summary: dict) -> None:
        self._flops_used = int(summary["flops_used"])

    def _install_closed_summary(self, summary: dict) -> None:
        # ``_normalize_context_timing`` returns a fresh defensive mapping.
        # Take ownership so close does not perform a second unbounded copy.
        self._closed_summary = summary
        self._flops_used = int(summary["flops_used"])
        self._wall_time_s = summary["wall_time_s"]
        self._flopscope_backend_time = summary["flopscope_backend_time_s"]
        self._flopscope_overhead_time = summary["flopscope_overhead_time_s"]
        self._residual_wall_time = summary["residual_wall_time_s"]

    def _normalize_context_timing(self, summary: dict, *, kernel_ns: int) -> dict:
        """Return one client-context timing view without changing server state."""
        with dispatch_span():
            result = deepcopy(summary)
        if self._wall_start_ns is None:
            return result
        wall_ns = max(time.perf_counter_ns() - self._wall_start_ns, 0)
        dispatch_ns = max(total_dispatch_ns() - self._dispatch_baseline_ns, 0)
        wall, backend, overhead, residual = _decompose_timing(
            wall_ns, dispatch_ns, kernel_ns
        )
        result.update(
            wall_time_s=wall,
            flopscope_backend_time_s=backend,
            flopscope_overhead_time_s=overhead,
            residual_wall_time_s=residual,
        )
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary_dict(self, by_namespace: bool = False) -> dict:
        if self._is_open:
            summary, metadata = _request_budget_summary(
                scope="active_context", by_namespace=by_namespace
            )
            summary = self._normalize_context_timing(
                summary,
                kernel_ns=metadata["client_context_compute_ns"] or 0,
            )
            self._update_live_from_summary(summary)
            return summary
        if self._closed_summary is not None:
            with dispatch_span():
                result = deepcopy(self._closed_summary)
                if not by_namespace:
                    result.pop("by_namespace", None)
            return result
        with dispatch_span():
            return deepcopy(self._empty_summary(by_namespace=by_namespace))

    def summary(self, by_namespace: bool = False) -> str:
        data = self.summary_dict(by_namespace=by_namespace)
        with dispatch_span():
            from flopscope._display import _format_budget_summary_text

            header = "flopscope FLOP Budget Summary"
            if self.namespace:
                header += f" [{self.namespace}]"
            return _format_budget_summary_text(
                data,
                by_namespace=by_namespace,
                header=header,
            )

    # ------------------------------------------------------------------
    # Decorator support
    # ------------------------------------------------------------------

    def __call__(self, func):
        """Use BudgetContext as a decorator."""
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> BudgetContext:
        """Open the budget on the server and update the local cache."""
        global _active_context
        if _active_context is not None and _active_context is not _global_default:
            raise RuntimeError(
                "Nested BudgetContext is not supported. "
                "Only one context can be active at a time."
            )
        self._previous_context = _active_context
        conn = get_connection()
        wall_start_ns = time.perf_counter_ns()
        dispatch_baseline_ns = total_dispatch_ns()
        with dispatch_span():
            conn.send_recv(encode_budget_open(self._flop_budget, self._namespace))
        self._wall_start_ns = wall_start_ns
        self._dispatch_baseline_ns = dispatch_baseline_ns
        self._flops_used = 0
        self._closed_summary = None
        self._wall_time_s = None
        self._flopscope_backend_time = 0.0
        self._flopscope_overhead_time = 0.0
        self._residual_wall_time = None
        self._is_open = True
        _active_context = self
        return self

    def __exit__(self, *args: object) -> None:
        """Close the budget, compute the timing split, store summary."""
        global _active_context
        if self._is_open:
            conn = get_connection()
            close_acknowledged = False
            try:
                with dispatch_span():
                    response = conn.send_recv(encode_budget_close())
                    close_acknowledged = (
                        isinstance(response, dict) and response.get("status") == "ok"
                    )
                    summary = _validated_close_summary(response)
                    kernel_ns = _extract_compute_ns(response)
                summary = self._normalize_context_timing(
                    summary,
                    kernel_ns=kernel_ns,
                )
                self._install_closed_summary(summary)
            finally:
                if close_acknowledged:
                    self._is_open = False
                    _active_context = self._previous_context
            return
        _active_context = self._previous_context

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BudgetContext(flop_budget={self._flop_budget}, "
            f"flops_used={self._flops_used})"
        )


def budget(flop_budget, quiet=False, namespace=None):
    """Create a BudgetContext usable as both a context manager and decorator."""
    return BudgetContext(
        flop_budget=flop_budget,
        quiet=quiet,
        namespace=namespace,
    )


def _request_budget_summary(*, scope: str, by_namespace: bool) -> tuple[dict, dict]:
    conn = get_connection()
    with dispatch_span():
        conn.require_capability(AUTHORITATIVE_BUDGET_SUMMARY_CAPABILITY)
        response = conn.send_recv(
            encode_budget_summary(scope, by_namespace=by_namespace)
        )
        summary, display_totals = _validate_summary_response(
            response, by_namespace=by_namespace
        )
        summary = deepcopy(summary)
        display_totals = deepcopy(display_totals)
    return summary, display_totals


def budget_summary_dict(by_namespace=False):
    """Return the server-authoritative session budget summary."""
    summary, _ = _request_budget_summary(scope="session", by_namespace=by_namespace)
    return summary


# Note: No budget_reset() in the client — participants must not clear usage.


_global_default = None


def _get_default_budget_amount():
    import os

    raw = os.environ.get("FLOPSCOPE_DEFAULT_BUDGET")
    if raw is not None:
        return int(float(raw))
    return int(1e15)


def _get_global_default():
    global _global_default, _active_context
    if _global_default is None:
        _global_default = BudgetContext(
            flop_budget=_get_default_budget_amount(),
            quiet=True,
            namespace=None,
        )
        # Open it on the server. Defensive: keep the round-trip inside a
        # dispatch span so it counts as overhead (never billed residual) if this
        # implicit global-default path is ever wired up. It is currently
        # unreferenced, but the invariant is "every send_recv lives in a span".
        conn = get_connection()
        with dispatch_span():
            response = conn.send_recv(encode_budget_open(_global_default._flop_budget))
            _global_default._update_budget(response)
        _global_default._is_open = True
        _active_context = _global_default
    return _global_default


def _reset_global_default():
    global _global_default, _active_context
    if _global_default is not None and _active_context is _global_default:
        _active_context = None
    _global_default = None
