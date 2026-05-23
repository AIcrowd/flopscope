"""Budget context manager and operation recording for flopscope."""

from __future__ import annotations

import functools
import threading
import time
import weakref
from typing import Any, Literal, NamedTuple

from flopscope.errors import BudgetExhaustedError


class OpRecord(NamedTuple):
    """Record of a single counted operation."""

    op_name: str
    subscripts: str | None
    shapes: tuple
    flop_cost: int
    cumulative: int
    namespace: str | None = None
    flopscope_context_start_offset_s: float | None = (
        None  # seconds since active BudgetContext start
    )
    flopscope_backend_duration_s: float | None = (
        None  # wall-clock seconds of the backend call
    )
    flopscope_overhead_duration_s: float | None = (
        None  # per-op flopscope dispatch time (preamble + deduct body + bookkeeping + postamble)
    )


# ---------------------------------------------------------------------------
# Why cooperative (not signal-based) deadline enforcement?
#
# We deliberately avoid SIGALRM / signal-based preemption because:
# 1. Python signal handlers only run between bytecodes — they cannot
#    interrupt C extensions (numpy/LAPACK/BLAS), which are exactly the
#    operations where time limits matter most.
# 2. signal.alarm() is POSIX-only (no Windows) and integer-second only.
# 3. Signals are main-thread-only and can interfere with numpy internals.
#
# The hard enforcement boundary is the container/OS level: flopscope
# submissions run inside Docker containers with kernel-level time limits
# (cgroups / rlimit) that deliver SIGKILL when exceeded.
#
# The in-library wall_time_limit_s is a UX feature: it gives participants
# a clean, informative TimeExhaustedError (with op name, elapsed time,
# and configured limit) rather than a brutal container kill.
#
# The deadline is checked:
# 1. Pre-op: in BudgetContext.deduct() before the numpy call starts.
# 2. Post-op: in _OpTimer.__exit__() after the numpy call completes.
#
# This bounds overshoot to the duration of a single numpy call.
# ---------------------------------------------------------------------------


class _OpTimer:
    """Timer for a counted op's numpy call window.

    Used as a context manager around the numpy call::

        with budget.deduct(op_name, ...):
            result = _call_numpy(np_func, ...)

    On __exit__, the block's wall time is split into:
      - backend duration: sum of _call_numpy durations reported during the block
      - in-block overhead: block_duration - backend duration
    """

    __slots__ = (
        "_budget",
        "_op_index",
        "_block_t0",
        "_backend_duration_s",
        "_prev_timer",
    )

    def __init__(self, budget: BudgetContext, op_index: int):
        self._budget = budget
        self._op_index = op_index
        self._block_t0: float | None = None
        self._backend_duration_s: float = 0.0
        self._prev_timer: _OpTimer | None = None

    def __enter__(self) -> _OpTimer:
        self._block_t0 = time.perf_counter()
        # Stack discipline supports the rare case of nested deduct() blocks
        self._prev_timer = self._budget._current_op_timer
        self._budget._current_op_timer = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        if self._block_t0 is not None:
            block_duration = time.perf_counter() - self._block_t0
            in_block_overhead = max(block_duration - self._backend_duration_s, 0.0)

            self._budget._total_flopscope_backend_time += self._backend_duration_s
            self._budget._total_flopscope_overhead_time += in_block_overhead

            op = self._budget._op_log[self._op_index]
            self._budget._op_log[self._op_index] = op._replace(
                flopscope_backend_duration_s=self._backend_duration_s,
                flopscope_overhead_duration_s=(op.flopscope_overhead_duration_s or 0.0)
                + in_block_overhead,
            )

            # Post-op deadline check (preserves existing behavior)
            if (
                exc_type is None
                and self._budget._deadline is not None
                and time.perf_counter() > self._budget._deadline
            ):
                from flopscope.errors import TimeExhaustedError

                raise TimeExhaustedError(
                    op.op_name,
                    elapsed_s=time.perf_counter() - self._budget._start_time,  # type: ignore[operator]
                    limit_s=self._budget._wall_time_limit_s,  # type: ignore[arg-type]
                )

        self._budget._current_op_timer = self._prev_timer
        return False


def _call_numpy(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke a numpy callable, attributing only its wall time to backend time.

    All numpy calls inside counted-op wrappers MUST go through this helper so
    that view-casts, copyto, errstate setup, and other flopscope-internal work
    surrounding the call are correctly bucketed as flopscope_overhead.

    Reports the call's backend duration to the active _OpTimer (via
    budget._current_op_timer). When no timer is active (e.g. helper called
    from a non-counted code path), it is a transparent passthrough.

    Annotated as ``Any -> Any`` to match the existing untyped-internal-helper
    convention used by ``_call_with_optional_out``. Wrappers' explicit return
    annotations carry the public type contract.
    """
    t0 = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        d = time.perf_counter() - t0
        budget = get_active_budget()
        if budget is not None and budget._current_op_timer is not None:
            budget._current_op_timer._backend_duration_s += d


def _counted_wrapper(fn):
    """Decorator that brackets a flopscope wrapper and bills its non-numpy,
    non-nested-overhead time to flopscope_overhead_time.

    Formula: wall - backend_delta - overhead_delta. Handles nesting naturally
    (outer attributes only its own remainder), so no re-entrancy guard.

    Per-op attribution: wrapper-own overhead is distributed equally across
    ops created during this call (typically exactly 1 across this codebase).
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        from flopscope._validation import require_budget

        budget = require_budget()
        fs_t0 = time.perf_counter()
        backend_baseline = budget._total_flopscope_backend_time
        overhead_baseline = budget._total_flopscope_overhead_time
        ops_before = len(budget._op_log)
        try:
            return fn(*args, **kwargs)
        finally:
            wall = time.perf_counter() - fs_t0
            backend_delta = budget._total_flopscope_backend_time - backend_baseline
            overhead_delta = budget._total_flopscope_overhead_time - overhead_baseline
            wrapper_own_overhead = max(wall - backend_delta - overhead_delta, 0.0)
            budget._total_flopscope_overhead_time += wrapper_own_overhead
            ops_added = list(range(ops_before, len(budget._op_log)))
            if ops_added and wrapper_own_overhead > 0:
                per_op = wrapper_own_overhead / len(ops_added)
                for idx in ops_added:
                    op = budget._op_log[idx]
                    budget._op_log[idx] = op._replace(
                        flopscope_overhead_duration_s=(
                            op.flopscope_overhead_duration_s or 0.0
                        )
                        + per_op
                    )

    return wrapped


# ----- Stack-walk tripwire for issue #69 -----
#
# Every @_counted_wrapper call creates a new inner `wrapped` function
# instance, but Python compiles its body once — `wrapped.__code__` is the
# SAME `code` object across all decorator invocations. We capture it once
# here as a stable marker. `__array_function__` and `__array_ufunc__` walk
# the call stack looking for a frame with this code object: if found, the
# protocol was triggered from inside an fnp wrapper (= a wrapper forgot to
# strip a FlopscopeArray before calling raw numpy) — that's a bug, raise.

import sys as _sys


def _capture_wrapped_code():
    """Capture the `code` object of `_counted_wrapper`'s inner `wrapped`."""
    @_counted_wrapper
    def _probe(*args, **kwargs):
        pass
    return _probe.__code__


_WRAPPED_CO = _capture_wrapped_code()


def _called_from_wrapper() -> bool:
    """True iff a `_counted_wrapper.wrapped` frame appears in the call stack.

    Used by `FlopscopeArray.__array_function__` and `__array_ufunc__` to
    distinguish "user wrote np.<f>(whest) at top level" (depth=0, auto-route
    with warning) from "an fnp wrapper forgot to strip and leaked WhestArray
    into raw numpy" (depth>0, raise loudly).

    Implementation: walks `frame.f_back` chain and compares `f.f_code is
    _WRAPPED_CO` (single C-level pointer comparison per frame). Cost is
    O(stack depth) and only paid on actual protocol entries.
    """
    f = _sys._getframe(1)  # skip this frame
    while f is not None:
        if f.f_code is _WRAPPED_CO:
            return True
        f = f.f_back
    return False


_thread_local = threading.local()
_all_budget_contexts: weakref.WeakSet[BudgetContext] = weakref.WeakSet()


def get_active_budget() -> BudgetContext | None:
    """Return the active BudgetContext, or None if outside any context."""
    return getattr(_thread_local, "active_budget", None)


class _NamespaceScope:
    __slots__ = ("_budget", "_segment")

    def __init__(self, budget: BudgetContext, segment: str):
        self._budget = budget
        self._segment = segment

    def __enter__(self) -> BudgetContext:
        fs_t0 = time.perf_counter()
        try:
            self._budget._push_namespace(self._segment)
            return self._budget
        finally:
            self._budget._total_flopscope_overhead_time += time.perf_counter() - fs_t0

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        fs_t0 = time.perf_counter()
        try:
            self._budget._pop_namespace(self._segment)
            return False
        finally:
            self._budget._total_flopscope_overhead_time += time.perf_counter() - fs_t0


def _validate_namespace_segment(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("namespace segment must be a string")
    segment = name.strip()
    if not segment:
        raise ValueError("namespace segment must be non-empty")
    if "." in segment:
        raise ValueError("namespace segment must not contain '.'")
    return segment


def namespace(name: str) -> _NamespaceScope:
    """Create a nested namespace scope for the active budget context.

    Parameters
    ----------
    name : str
        Namespace segment to append to the active budget namespace. The
        segment must be non-empty and must not contain ``"."``.

    Returns
    -------
    _NamespaceScope
        Context manager that attributes counted operations to the nested
        namespace while the scope is active.

    Raises
    ------
    NoBudgetContextError
        If called outside an active :class:`BudgetContext`.
    ValueError
        If ``name`` is not a valid namespace segment.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>>
    >>> with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
    ...     with flops.namespace("encoder"):
    ...         _ = fnp.add(fnp.array([1.0]), fnp.array([2.0]))
    >>> budget.op_log[-1].namespace
    'encoder'
    """
    from flopscope.errors import NoBudgetContextError

    budget = get_active_budget()
    if budget is None:
        raise NoBudgetContextError()
    return _NamespaceScope(budget, _validate_namespace_segment(name))


def _update_operation_summary(ops: dict[str, dict], op: OpRecord) -> None:
    bucket = ops.setdefault(
        op.op_name,
        {
            "flop_cost": 0,
            "calls": 0,
            "flopscope_backend_time_s": 0.0,
            "flopscope_overhead_time_s": 0.0,
        },
    )
    bucket["flop_cost"] += op.flop_cost
    bucket["calls"] += 1
    if op.flopscope_backend_duration_s is not None:
        bucket["flopscope_backend_time_s"] += op.flopscope_backend_duration_s
    if op.flopscope_overhead_duration_s is not None:
        bucket["flopscope_overhead_time_s"] += op.flopscope_overhead_duration_s


def _summarize_operations(op_log: list[OpRecord]) -> dict[str, dict]:
    ops: dict[str, dict] = {}
    for op in op_log:
        _update_operation_summary(ops, op)
    return ops


def _summarize_by_namespace(op_log: list[OpRecord]) -> dict[str | None, dict]:
    by_namespace: dict[str | None, dict] = {}
    for op in op_log:
        bucket = by_namespace.setdefault(
            op.namespace,
            {
                "flops_used": 0,
                "calls": 0,
                "flopscope_backend_time_s": 0.0,
                "flopscope_overhead_time_s": 0.0,
                "operations": {},
            },
        )
        bucket["flops_used"] += op.flop_cost
        bucket["calls"] += 1
        if op.flopscope_backend_duration_s is not None:
            bucket["flopscope_backend_time_s"] += op.flopscope_backend_duration_s
        if op.flopscope_overhead_duration_s is not None:
            bucket["flopscope_overhead_time_s"] += op.flopscope_overhead_duration_s
        _update_operation_summary(bucket["operations"], op)
    return by_namespace


def _timing_summary(
    wall_time_s: float | None,
    flopscope_backend_time_s: float | None,
    overhead_time_s: float | None,
) -> tuple[float | None, float, float, float | None]:
    backend = flopscope_backend_time_s or 0.0
    overhead = overhead_time_s or 0.0
    if wall_time_s is None:
        return None, backend, overhead, None
    residual = wall_time_s - backend - overhead
    if residual < 0 and abs(residual) < 1e-12:
        residual = 0.0
    return wall_time_s, backend, overhead, max(residual, 0.0)


class BudgetContext:
    """Context manager for FLOP budget enforcement.

    Parameters
    ----------
    flop_budget : int
        Maximum number of FLOPs allowed. Must be > 0.
    flop_multiplier : float, optional
        Multiplier applied to all FLOP costs. Default 1.
    quiet : bool, optional
        When ``True``, suppress the startup banner printed on context entry.
    namespace : str | None, optional
        Root namespace prefix used for operation attribution inside this
        context. Nested ``flops.namespace(...)`` scopes append dotted segments.
    wall_time_limit_s : float | None, optional
        Cooperative wall-clock limit for the entire context. The timer starts
        when the context is entered and is checked before and after each
        counted NumPy call. If the deadline is exceeded, flopscope raises
        ``TimeExhaustedError`` at the next operation boundary. This is a
        diagnostic UX limit, not a hard preemptive kill.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>>
    >>> with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
    ...     x = fnp.array([1.0, 2.0, 3.0])
    ...     _ = fnp.einsum("i->", x)
    >>> budget.summary_dict()["flops_used"] > 0
    True
    """

    def __init__(
        self,
        flop_budget: int,
        flop_multiplier: float = 1.0,
        quiet: bool = False,
        namespace: str | None = None,
        wall_time_limit_s: float | None = None,
    ):
        # Capture creation time as the very first thing so any work done in
        # __init__ (validation, list/dict allocation) is included in the
        # eventual wall_time_s span, billed to flopscope_overhead_time. See
        # issue #82.
        self._creation_time = time.perf_counter()
        if flop_budget <= 0:
            raise ValueError(f"flop_budget must be > 0, got {flop_budget}")
        self._flop_budget = flop_budget
        self._flop_multiplier = flop_multiplier
        self._flops_used = 0
        self._op_log: list[OpRecord] = []
        self._quiet = quiet
        self._root_namespace = namespace
        self._namespace_stack: list[str] = []
        self._previous_budget: BudgetContext | None = None
        self._wall_time_limit_s = wall_time_limit_s
        self._start_time: float | None = None
        self._deadline: float | None = None
        self._wall_time_s: float | None = None
        self._total_flopscope_backend_time: float = 0.0
        self._total_flopscope_overhead_time: float = 0.0
        self._pre_enter_overhead: float = 0.0
        self._current_op_timer: _OpTimer | None = None
        self._recorded_flops_used = 0
        self._recorded_op_count = 0
        self._recorded_flopscope_backend_time = 0.0
        self._recorded_overhead_time: float = 0.0
        self._budget_recorded = False
        _all_budget_contexts.add(self)

    @property
    def flop_budget(self) -> int:
        return self._flop_budget

    @property
    def flops_used(self) -> int:
        return self._flops_used

    @property
    def flops_remaining(self) -> int:
        return self._flop_budget - self._flops_used

    @property
    def flop_multiplier(self) -> float:
        return self._flop_multiplier

    @property
    def op_log(self) -> list[OpRecord]:
        return self._op_log

    @property
    def namespace(self) -> str | None:
        if not self._namespace_stack:
            return self._root_namespace
        suffix = ".".join(self._namespace_stack)
        if self._root_namespace is None:
            return suffix
        return f"{self._root_namespace}.{suffix}"

    @property
    def wall_time_limit_s(self) -> float | None:
        return self._wall_time_limit_s

    @property
    def wall_time_s(self) -> float | None:
        """Total wall-clock seconds spanned by the context.

        Measured from ``__init__`` start to the end of ``__exit__`` (after the
        accumulator-record and active-budget restoration work). ``None`` until
        ``__exit__`` has run.

        The decomposition ``wall_time_s == flopscope_backend_time
        + flopscope_overhead_time + residual_wall_time`` holds within numerical
        tolerance. The pre-``__enter__`` slice (``__init__`` body + banner print)
        and the post-``__exit__`` body slice (accumulator-record,
        active-budget restore) are both attributed to
        ``flopscope_overhead_time``.
        """
        return self._wall_time_s

    @property
    def elapsed_s(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def flopscope_backend_time(self) -> float:
        return self._total_flopscope_backend_time

    @property
    def flopscope_overhead_time(self) -> float:
        """Wall-clock seconds spent inside flopscope's own dispatch code.

        Includes wrapper preambles, BudgetContext.deduct() body, _OpTimer
        bookkeeping, the flopscope-internal parts of the timed block (view-
        casts, copyto, dispatch), wrapper postambles (including
        maybe_check_nan_inf when opted in), and namespace push/pop.

        Measured per op via the @_counted_wrapper decorator. Aggregated per
        namespace via summary_dict(by_namespace=True).
        """
        return self._total_flopscope_overhead_time

    @property
    def residual_wall_time(self) -> float | None:
        """Wall time minus flopscope backend and overhead time.

        This is the measured wall-clock remainder outside the backend calls and
        flopscope's own dispatch/accounting work: user Python between ops,
        time.sleep, GC pauses, and un-instrumented NumPy.
        """
        if self._wall_time_s is None:
            return None
        residual = (
            self._wall_time_s
            - self._total_flopscope_backend_time
            - self._total_flopscope_overhead_time
        )
        if residual < 0 and abs(residual) < 1e-12:
            return 0.0
        return max(residual, 0.0)

    def deduct(
        self, op_name: str, *, flop_cost: int, subscripts: str | None, shapes: tuple
    ) -> _OpTimer:
        """Deduct FLOPs from the budget and return a timer context manager."""
        fs_t0 = time.perf_counter()
        appended = False
        try:
            from flopscope._weights import get_weight

            weight = get_weight(op_name)
            adjusted_cost = int(flop_cost * self._flop_multiplier * weight)
            if adjusted_cost > self.flops_remaining:
                raise BudgetExhaustedError(
                    op_name,
                    flop_cost=adjusted_cost,
                    flops_remaining=self.flops_remaining,
                )
            self._flops_used += adjusted_cost

            now = time.perf_counter()
            context_start_offset_s = (
                now - self._start_time if self._start_time is not None else None
            )

            self._op_log.append(
                OpRecord(
                    op_name=op_name,
                    subscripts=subscripts,
                    shapes=shapes,
                    flop_cost=adjusted_cost,
                    cumulative=self._flops_used,
                    namespace=self.namespace,
                    flopscope_context_start_offset_s=context_start_offset_s,
                )
            )
            appended = True

            if self._deadline is not None and now > self._deadline:
                from flopscope.errors import TimeExhaustedError

                raise TimeExhaustedError(
                    op_name,
                    elapsed_s=now - self._start_time,  # type: ignore[operator]
                    limit_s=self._wall_time_limit_s,  # type: ignore[arg-type]
                )

            op_index = len(self._op_log) - 1
            return _OpTimer(self, op_index=op_index)
        finally:
            deduct_body_time = time.perf_counter() - fs_t0
            self._total_flopscope_overhead_time += deduct_body_time
            if appended:
                op = self._op_log[-1]
                self._op_log[-1] = op._replace(
                    flopscope_overhead_duration_s=(
                        op.flopscope_overhead_duration_s or 0.0
                    )
                    + deduct_body_time
                )

    def summary_dict(self, by_namespace: bool = False) -> dict:
        """Return structured summary data for this budget context.

        Returns a dict with keys ``flop_budget``, ``flops_used``,
        ``flops_remaining``, ``operations``, ``wall_time_s``,
        ``flopscope_backend_time_s``, ``flopscope_overhead_time_s``,
        ``residual_wall_time_s``, and optionally ``by_namespace`` with per-
        namespace buckets that each include the same timing keys.

        Decomposition: ``wall_time_s == flopscope_backend_time_s
        + flopscope_overhead_time_s + residual_wall_time_s`` (within numerical
        tolerance).
        """
        wall_time = self._wall_time_s
        if wall_time is None and self._start_time is not None:
            wall_time = self.elapsed_s
        wall_time, backend_time, overhead_time, residual_wall_time = _timing_summary(
            wall_time,
            self._total_flopscope_backend_time,
            self._total_flopscope_overhead_time,
        )

        result = {
            "flop_budget": self._flop_budget,
            "flops_used": self._flops_used,
            "flops_remaining": self.flops_remaining,
            "operations": _summarize_operations(self._op_log),
            "wall_time_s": wall_time,
            "flopscope_backend_time_s": backend_time,
            "flopscope_overhead_time_s": overhead_time,
            "residual_wall_time_s": residual_wall_time,
        }
        if by_namespace:
            result["by_namespace"] = _summarize_by_namespace(self._op_log)
        return result

    def summary(self, by_namespace: bool = False) -> str:
        """Return a pretty-printed FLOP budget summary."""
        from flopscope._display import _format_budget_summary_text

        header = "flopscope FLOP Budget Summary"
        if self.namespace:
            header += f" [{self.namespace}]"
        return _format_budget_summary_text(
            self.summary_dict(by_namespace=by_namespace),
            by_namespace=by_namespace,
            header=header,
        )

    def _push_namespace(self, segment: str) -> None:
        self._namespace_stack.append(segment)

    def _pop_namespace(self, expected: str) -> None:
        actual = self._namespace_stack.pop()
        if actual != expected:
            raise RuntimeError(
                f"Namespace stack corrupted: expected {expected!r}, got {actual!r}"
            )

    def _snapshot_record(self) -> NamespaceRecord:
        wall_time = self.wall_time_s
        if wall_time is None and self._start_time is not None:
            wall_time = self.elapsed_s

        backend_delta = (
            self._total_flopscope_backend_time - self._recorded_flopscope_backend_time
        )
        if backend_delta < 0 and abs(backend_delta) < 1e-12:
            backend_delta = 0.0

        overhead_delta = (
            self._total_flopscope_overhead_time - self._recorded_overhead_time
        )
        if overhead_delta < 0 and abs(overhead_delta) < 1e-12:
            overhead_delta = 0.0

        return NamespaceRecord(
            namespace=self.namespace,
            flop_budget=0 if self._budget_recorded else self.flop_budget,
            flops_used=max(self._flops_used - self._recorded_flops_used, 0),
            op_log=list(self._op_log[self._recorded_op_count :]),
            wall_time_s=wall_time,
            total_flopscope_backend_time=max(backend_delta, 0.0),
            total_flopscope_overhead_time=max(overhead_delta, 0.0),
        )

    def _has_unrecorded_activity(self) -> bool:
        return self._flops_used > self._recorded_flops_used

    def _mark_recorded(self) -> None:
        self._recorded_flops_used = self._flops_used
        self._recorded_op_count = len(self._op_log)
        self._recorded_flopscope_backend_time = self._total_flopscope_backend_time
        self._recorded_overhead_time = self._total_flopscope_overhead_time
        self._budget_recorded = True

    def _mark_reset_baseline(self) -> None:
        self._recorded_flops_used = self._flops_used
        self._recorded_op_count = len(self._op_log)
        self._recorded_flopscope_backend_time = self._total_flopscope_backend_time
        self._recorded_overhead_time = self._total_flopscope_overhead_time
        self._budget_recorded = False

    def __enter__(self) -> BudgetContext:
        current = get_active_budget()
        if current is not None and current is not _global_default:
            raise RuntimeError("Cannot nest BudgetContexts")
        self._previous_budget = current  # save (may be global default or None)
        _thread_local.active_budget = self
        self._wall_time_s = None
        self._deadline = None
        if not self._quiet:
            import sys

            import flopscope

            banner = (
                f"flopscope {flopscope.__version__} "
                f"(numpy {flopscope.__numpy_version__} backend) | "
                f"budget: {self._flop_budget:.2e} FLOPs"
            )
            if self._wall_time_limit_s is not None:
                banner += f" | time limit: {self._wall_time_limit_s:.1f}s"
            print(banner, file=sys.stderr)
        # Mark wall start AFTER all enter setup (including the banner print).
        # The slice from _creation_time → _start_time is pre-enter overhead;
        # __exit__ adds it to flopscope_overhead_time. See issue #82.
        self._start_time = time.perf_counter()
        self._pre_enter_overhead = self._start_time - self._creation_time
        if self._wall_time_limit_s is not None:
            self._deadline = self._start_time + self._wall_time_limit_s
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # body_end snapshots the user-code window (start_time → here). Then
        # we do the post-exit accumulator work and snapshot post_exit_end.
        # wall_time_s spans creation → post_exit_end, and the pre-enter +
        # post-exit slices are billed to flopscope_overhead_time. See #82.
        body_end = time.perf_counter()
        if self._start_time is not None:
            _accumulator.record(self)
            _thread_local.active_budget = self._previous_budget
            post_exit_end = time.perf_counter()
            self._wall_time_s = post_exit_end - self._creation_time
            self._total_flopscope_overhead_time += self._pre_enter_overhead + (
                post_exit_end - body_end
            )
        else:
            # Defensive: __exit__ called without __enter__.
            _thread_local.active_budget = self._previous_budget
        return None

    def __call__(self, func):
        """Use BudgetContext as a decorator."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


def budget(
    flop_budget: int,
    flop_multiplier: float = 1.0,
    quiet: bool = False,
    namespace: str | None = None,
    wall_time_limit_s: float | None = None,
) -> BudgetContext:
    """Create a ``BudgetContext`` usable as a context manager or decorator.

    This helper accepts the same arguments as ``BudgetContext(...)`` and is
    convenient when you want a short, function-style entrypoint.

    Parameters
    ----------
    flop_budget : int
        Maximum number of FLOPs allowed inside the context.
    flop_multiplier : float, optional
        Multiplier applied to every charged FLOP cost. Default ``1.0``.
    quiet : bool, optional
        If ``True``, suppress the startup banner on context entry.
    namespace : str or None, optional
        Root namespace prefix for attribution.
    wall_time_limit_s : float or None, optional
        Cooperative wall-clock limit checked at operation boundaries.

    Returns
    -------
    BudgetContext
        A context manager that can also be used as a decorator.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>> with flops.budget(1_000):
    ...     _ = fnp.add(fnp.array([1.0]), fnp.array([2.0]))
    """
    return BudgetContext(
        flop_budget=flop_budget,
        flop_multiplier=flop_multiplier,
        quiet=quiet,
        namespace=namespace,
        wall_time_limit_s=wall_time_limit_s,
    )


# ---------------------------------------------------------------------------
# Global default BudgetContext
# ---------------------------------------------------------------------------

_global_default: BudgetContext | None = None


def _get_default_budget_amount() -> int:
    """Read default budget from env var, falling back to 1e15."""
    import os

    raw = os.environ.get("FLOPSCOPE_DEFAULT_BUDGET")
    if raw is not None:
        return int(float(raw))
    return int(1e15)


def _get_global_default() -> BudgetContext:
    """Return the global default BudgetContext, creating it lazily."""
    global _global_default
    if _global_default is None:
        _global_default = BudgetContext(
            flop_budget=_get_default_budget_amount(),
            quiet=True,
            namespace=None,
        )
        _thread_local.active_budget = _global_default
    return _global_default


def _reset_global_default() -> None:
    """Reset the global default context. For testing and core library use."""
    global _global_default
    if (
        _global_default is not None
        and getattr(_thread_local, "active_budget", None) is _global_default
    ):
        _thread_local.active_budget = None
    _global_default = None


# ---------------------------------------------------------------------------
# Session-level accumulator
# ---------------------------------------------------------------------------


class NamespaceRecord(NamedTuple):
    """Snapshot of a BudgetContext's state at close time."""

    namespace: str | None
    flop_budget: int
    flops_used: int
    op_log: list[OpRecord]
    wall_time_s: float | None = None
    total_flopscope_backend_time: float | None = None
    total_flopscope_overhead_time: float | None = None


def _snapshot_namespace_record(ctx: BudgetContext) -> NamespaceRecord:
    return ctx._snapshot_record()


class BudgetAccumulator:
    """Collects budget records across multiple BudgetContext sessions."""

    def __init__(self) -> None:
        self._records: list[NamespaceRecord] = []

    def record(self, ctx: BudgetContext) -> None:
        """Snapshot a BudgetContext and store it."""
        self._records.append(_snapshot_namespace_record(ctx))
        ctx._mark_recorded()

    def get_data(self, by_namespace: bool = False) -> dict:
        """Return aggregated budget data across all recorded contexts."""
        total_budget = 0
        total_used = 0
        total_wall_time: float | None = None
        total_backend: float | None = None
        total_overhead: float | None = None
        all_ops: list[OpRecord] = []

        for rec in self._records:
            total_budget += rec.flop_budget
            total_used += rec.flops_used
            all_ops.extend(rec.op_log)
            if rec.wall_time_s is not None:
                total_wall_time = (total_wall_time or 0.0) + rec.wall_time_s
            if rec.total_flopscope_backend_time is not None:
                total_backend = (
                    total_backend or 0.0
                ) + rec.total_flopscope_backend_time
            if rec.total_flopscope_overhead_time is not None:
                total_overhead = (
                    total_overhead or 0.0
                ) + rec.total_flopscope_overhead_time

        wall_time, backend_time, overhead_time, residual_wall_time = _timing_summary(
            total_wall_time, total_backend, total_overhead
        )

        result = {
            "flop_budget": total_budget,
            "flops_used": total_used,
            "flops_remaining": total_budget - total_used,
            "operations": _summarize_operations(all_ops),
            "wall_time_s": wall_time,
            "flopscope_backend_time_s": backend_time,
            "flopscope_overhead_time_s": overhead_time,
            "residual_wall_time_s": residual_wall_time,
        }

        if by_namespace:
            result["by_namespace"] = _summarize_by_namespace(all_ops)

        return result

    def reset(self) -> None:
        """Clear all recorded data."""
        self._records.clear()


_accumulator = BudgetAccumulator()


def _snapshot_records() -> list[NamespaceRecord]:
    records = list(_accumulator._records)
    active = get_active_budget()
    if _global_default is not None and _global_default._has_unrecorded_activity():
        records.append(_snapshot_namespace_record(_global_default))
    if (
        active is not None
        and active is not _global_default
        and active._has_unrecorded_activity()
    ):
        records.append(_snapshot_namespace_record(active))
    return records


def budget_summary_dict(by_namespace: bool = False) -> dict:
    """Return aggregated budget data across all recorded contexts.

    Parameters
    ----------
    by_namespace : bool, optional
        If ``True``, include a ``"by_namespace"`` key with per-namespace
        breakdowns. Default ``False``.

    Returns
    -------
    dict
        Dictionary with keys ``"flop_budget"``, ``"flops_used"``,
        ``"flops_remaining"``, ``"operations"``, ``"wall_time_s"``,
        ``"flopscope_backend_time_s"``, ``"flopscope_overhead_time_s"``,
        ``"residual_wall_time_s"``, and optionally ``"by_namespace"``.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>> with flops.BudgetContext(flop_budget=100):
    ...     _ = fnp.add(fnp.array([1.0]), fnp.array([2.0]))
    >>> summary = flops.budget_summary_dict()
    >>> sorted(summary)
    ['flop_budget', 'flops_remaining', 'flops_used', 'flopscope_backend_time_s', 'flopscope_overhead_time_s', 'operations', 'residual_wall_time_s', 'wall_time_s']
    """
    acc_copy = BudgetAccumulator()
    acc_copy._records = _snapshot_records()
    return acc_copy.get_data(by_namespace=by_namespace)


def budget_reset() -> None:
    """Clear accumulated session-wide budget data.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Removes all recorded budget summaries and resets live baselines for
        active contexts.

    Notes
    -----
    This is primarily useful in tests, notebooks, and long-lived processes
    where you want a fresh session-wide summary.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>>
    >>> flops.budget_reset()
    >>> with flops.BudgetContext(flop_budget=100, quiet=True):
    ...     _ = fnp.add(fnp.array([1.0]), fnp.array([2.0]))
    >>> flops.budget_summary_dict()["flops_used"] > 0
    True
    >>> flops.budget_reset()
    >>> flops.budget_summary_dict()["flops_used"]
    0
    """
    _accumulator.reset()
    for ctx in list(_all_budget_contexts):
        ctx._mark_reset_baseline()
