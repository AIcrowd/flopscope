"""Client-side dispatch-time accumulator.

Times the *full* client dispatch of each flopscope op (request encode → socket
send/recv → response decode → result reconstruction) and accumulates it into a
single per-process counter. ``BudgetContext`` snapshots a baseline on enter and
reads the delta on exit; ``overhead = dispatch - backend`` and
``residual = wall - dispatch``.

Nesting is handled with a baseline/delta trick (ported from the in-process
``_counted_wrapper``): each span adds only its own remainder
(``wall - inner_already_counted``), so a span that contains nested spans counts
each op's wall exactly once. Single participant process, single-threaded (the
server allows one session at a time), so a module-level counter is safe.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager

_total_dispatch_ns: int = 0


def _now_ns() -> int:
    """Monotonic nanosecond clock (indirected so tests can fake it)."""
    return time.perf_counter_ns()


def total_dispatch_ns() -> int:
    """Total client dispatch nanoseconds accumulated so far this process."""
    return _total_dispatch_ns


def reset_dispatch() -> None:
    """Reset the accumulator (tests only)."""
    global _total_dispatch_ns
    _total_dispatch_ns = 0


@contextmanager
def dispatch_span():
    """Bracket one full op dispatch; add only this span's own remainder."""
    global _total_dispatch_ns
    t0 = _now_ns()
    baseline = _total_dispatch_ns
    try:
        yield
    finally:
        wall = _now_ns() - t0
        inner = _total_dispatch_ns - baseline
        # inner > wall is impossible with a monotonic clock; max() guards only
        # against cross-clock skew / faked-clock edge cases.
        _total_dispatch_ns += max(0, wall - inner)


def timed_dispatch(fn):
    """Decorator: time the full dispatch of *fn* into the accumulator."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with dispatch_span():
            return fn(*args, **kwargs)

    return wrapped
