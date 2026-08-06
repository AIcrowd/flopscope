"""Client-side BudgetContext proxy that delegates to the flopscope server."""

from __future__ import annotations

import time
from typing import NamedTuple

from flopscope._connection import get_connection
from flopscope._dispatch import dispatch_span, total_dispatch_ns
from flopscope._protocol import (
    encode_budget_close,
    encode_budget_open,
    encode_budget_status,
)
from flopscope.errors import NoBudgetContextError

# Module-level guard: only one BudgetContext can be active at a time.
_active_context = None


class BudgetSnapshot(NamedTuple):
    """An immutable, local snapshot of the active budget counters."""

    flop_budget: int
    flops_used: int
    flops_remaining: int


def _update_active_budget_from_response(response: object) -> None:
    """Refresh the active context cache from a successful operation response."""
    if _active_context is None or not isinstance(response, dict):
        return
    budget_info = response.get("budget")
    if isinstance(budget_info, dict):
        if "flops_used" in budget_info:
            _active_context._update_budget(budget_info)
            return
        budget_info = budget_info.get("flops_remaining")
    if budget_info is not None:
        try:
            _active_context._update_budget(
                {"flops_used": _active_context.flop_budget - int(budget_info)}
            )
        except (TypeError, ValueError):
            pass


def _extract_compute_ns(close_response: object) -> int:
    """Pull total server compute (ns) out of a ``budget_close`` response.

    Returns 0 if the ``result.comms_summary.total_compute_time_ns`` path is
    absent or unparseable (defensive; the version handshake makes it present in
    practice).
    """
    if not isinstance(close_response, dict):
        return 0
    result = close_response.get("result")
    if not isinstance(result, dict):
        return 0
    comms = result.get("comms_summary")
    if not isinstance(comms, dict):
        return 0
    try:
        return int(comms.get("total_compute_time_ns", 0))
    except (TypeError, ValueError):
        return 0


def _extract_close_budget(close_response: object) -> dict:
    """Pull the ``budget_breakdown`` dict (which carries ``flops_used``) out of a
    ``budget_close`` response.

    The authoritative FLOP count is nested at ``result.budget_breakdown`` —
    unlike ``budget_status``, which exposes ``flops_used`` directly under
    ``result``. Returns ``{}`` if the path is absent (defensive; the version
    handshake makes it present in practice).
    """
    if not isinstance(close_response, dict):
        return {}
    result = close_response.get("result")
    if not isinstance(result, dict):
        return {}
    breakdown = result.get("budget_breakdown")
    return breakdown if isinstance(breakdown, dict) else {}


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
        self._close_summary: str | None = None
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
        self._diagnostic_flops_baseline: int = 0
        self._diagnostic_budget_recorded: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def flop_budget(self) -> int:
        """Maximum FLOP allowance for this context."""
        return self._flop_budget

    @property
    def flops_used(self) -> int:
        """FLOPs consumed so far (cached locally, updated from server responses)."""
        return self._flops_used

    @property
    def flops_remaining(self) -> int:
        """FLOPs remaining in the budget (``budget - used``)."""
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
            Dict that may contain a ``"flops_used"`` key.  Missing key is
            silently ignored.
        """
        if "flops_used" in budget_info:
            self._flops_used = int(budget_info["flops_used"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Query the server for current budget status and return a formatted string.

        Also updates the local ``flops_used`` cache.

        Returns
        -------
        str
            Human-readable summary of budget usage.
        """
        conn = get_connection()
        with dispatch_span():
            response = conn.send_recv(encode_budget_status())
        # Budget status is nested inside "result" key
        result = response.get("result", {})
        self._update_budget(result)
        budget = result.get("flop_budget", self._flop_budget)
        used = self._flops_used
        remaining = int(budget) - used
        return f"BudgetContext: {used}/{budget} FLOPs used ({remaining} remaining)"

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
        # A BudgetContext can be reused as a decorator.  Each entry opens a
        # fresh server budget, so its diagnostic delta starts at that session's
        # cached zero rather than the prior session's closing total.
        self._diagnostic_flops_baseline = 0
        self._diagnostic_budget_recorded = False
        self._previous_context = _active_context
        conn = get_connection()
        self._wall_start_ns = time.perf_counter_ns()
        self._dispatch_baseline_ns = total_dispatch_ns()
        with dispatch_span():
            response = conn.send_recv(encode_budget_open(self._flop_budget))
            self._update_budget(response)
        self._is_open = True
        _active_context = self
        return self

    def __exit__(self, *args: object) -> None:
        """Close the budget, compute the timing split, store summary."""
        global _active_context
        if self._is_open:
            conn = get_connection()
            with dispatch_span():
                response = conn.send_recv(encode_budget_close())
                # flops_used is nested at result.budget_breakdown in the close
                # response; refresh the cache from there so a plain `with` block
                # reports the server's count without a separate summary() call.
                self._update_budget(_extract_close_budget(response))
            # _wall_start_ns is always set by __enter__ before _is_open=True; the
            # fallback only guards the never-exercised "exit without enter" path.
            start_ns = (
                self._wall_start_ns
                if self._wall_start_ns is not None
                else time.perf_counter_ns()
            )
            wall_ns = time.perf_counter_ns() - start_ns
            dispatch_ns = total_dispatch_ns() - self._dispatch_baseline_ns
            kernel_ns = _extract_compute_ns(response)
            (
                self._wall_time_s,
                self._flopscope_backend_time,
                self._flopscope_overhead_time,
                self._residual_wall_time,
            ) = _decompose_timing(wall_ns, dispatch_ns, kernel_ns)
            self._close_summary = (
                f"BudgetContext closed: {self._flops_used}/{self._flop_budget} "
                f"FLOPs used"
            )
            self._is_open = False
            _accumulator.record(self)
        _active_context = self._previous_context

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BudgetContext(flop_budget={self._flop_budget}, "
            f"flops_used={self._flops_used})"
        )


# ------------------------------------------------------------------
# Accumulator
# ------------------------------------------------------------------


class BudgetAccumulator:
    """Incrementally collects diagnostic totals across budget sessions."""

    def __init__(self):
        self.reset()

    def record(self, ctx):
        # `or 0.0` coerces Optional timing fields on a never-closed context.
        baseline = getattr(ctx, "_diagnostic_flops_baseline", 0)
        budget_recorded = getattr(ctx, "_diagnostic_budget_recorded", False)
        flop_budget = 0 if budget_recorded else ctx.flop_budget
        flops_used = max(ctx.flops_used - baseline, 0)
        self._flop_budget += flop_budget
        self._flops_used += flops_used
        self._wall_time_s += getattr(ctx, "wall_time_s", 0.0) or 0.0
        self._backend_time_s += getattr(ctx, "flopscope_backend_time_s", 0.0) or 0.0
        self._overhead_time_s += getattr(ctx, "flopscope_overhead_time_s", 0.0) or 0.0
        self._residual_time_s += getattr(ctx, "residual_wall_time_s", 0.0) or 0.0

        bucket = self._by_namespace.setdefault(
            ctx.namespace,
            {"flop_budget": 0, "flops_used": 0, "operations": {}},
        )
        bucket["flop_budget"] += flop_budget
        bucket["flops_used"] += flops_used
        if hasattr(ctx, "_diagnostic_flops_baseline"):
            ctx._diagnostic_flops_baseline = ctx.flops_used
            ctx._diagnostic_budget_recorded = True

    def get_data(self, by_namespace=False):
        result = {
            "flop_budget": self._flop_budget,
            "flops_used": self._flops_used,
            "flops_remaining": self._flop_budget - self._flops_used,
            "operations": {},
            "wall_time_s": self._wall_time_s,
            "flopscope_backend_time_s": self._backend_time_s,
            "flopscope_overhead_time_s": self._overhead_time_s,
            "residual_wall_time_s": self._residual_time_s,
        }
        if by_namespace:
            result["by_namespace"] = {
                namespace: {
                    **bucket,
                    "operations": dict(bucket["operations"]),
                }
                for namespace, bucket in self._by_namespace.items()
            }
        return result

    def reset(self):
        self._flop_budget = 0
        self._flops_used = 0
        self._wall_time_s = 0.0
        self._backend_time_s = 0.0
        self._overhead_time_s = 0.0
        self._residual_time_s = 0.0
        self._by_namespace = {}


_accumulator = BudgetAccumulator()


def budget(flop_budget, quiet=False, namespace=None):
    """Create a BudgetContext usable as both a context manager and decorator."""
    return BudgetContext(
        flop_budget=flop_budget,
        quiet=quiet,
        namespace=namespace,
    )


def budget_summary_dict(by_namespace=False):
    """Return aggregated budget data across all recorded contexts."""
    return _accumulator.get_data(by_namespace=by_namespace)


def current_budget() -> BudgetSnapshot:
    """Return cached counters for the active budget without an RPC."""
    if _active_context is None:
        raise NoBudgetContextError()
    return BudgetSnapshot(
        flop_budget=int(_active_context.flop_budget),
        flops_used=int(_active_context.flops_used),
        flops_remaining=int(_active_context.flops_remaining),
    )


def budget_reset() -> None:
    """Clear local diagnostic summaries without changing server enforcement."""
    _accumulator.reset()
    if _active_context is not None:
        _active_context._diagnostic_flops_baseline = _active_context.flops_used
        _active_context._diagnostic_budget_recorded = False


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
