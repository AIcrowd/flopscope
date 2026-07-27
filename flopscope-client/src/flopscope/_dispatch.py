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

Scope
-----
The accumulator measures flopscope's own dispatch cost, so only flopscope's own
code contributes to it. A span opened from outside this package runs normally
but adds nothing, and is counted by ``participant_span_count()``.

This keeps the timing split well-defined: the accumulator is subtracted from
wall time to separate framework cost from caller cost, so admitting spans around
non-flopscope code would make that split meaningless. Restricting the API
surface is not sufficient on its own — module-level state stays reachable
through ordinary imports — so the invariant is enforced where the span is
opened.

Provenance is taken from the *defining module of the wrapped callable*, not from
the immediate caller: ``timed_dispatch`` opens its span from inside this module,
so the immediate caller is always internal and carries no information.

The test is IDENTITY of the callable's ``__globals__`` against the live
flopscope module namespaces. ``__module__`` and ``__code__.co_filename`` are
ordinary writable attributes and so cannot carry this invariant.
"""

from __future__ import annotations

import functools
import sys
import time
from contextlib import contextmanager

_total_dispatch_ns: int = 0
_participant_spans: int = 0

# id(module.__dict__) for every imported flopscope module. Rebuilt when
# sys.modules grows, so lazily-imported submodules are picked up; recomputing it
# per call would put an O(len(sys.modules)) scan on the hot path.
_internal_globals: frozenset[int] = frozenset()
_modules_seen: int = -1


def _now_ns() -> int:
    """Monotonic nanosecond clock (indirected so tests can fake it)."""
    return time.perf_counter_ns()


def total_dispatch_ns() -> int:
    """Total client dispatch nanoseconds accumulated so far this process."""
    return _total_dispatch_ns


def participant_span_count() -> int:
    """Spans opened from outside flopscope, which contributed nothing.

    Expected to be zero: callers have no reason to open a dispatch span, since
    the accumulator exists to measure flopscope's own cost. Exposed so an
    embedding harness can assert that invariant held.
    """
    return _participant_spans


def reset_dispatch() -> None:
    """Reset the accumulators (tests only)."""
    global _total_dispatch_ns, _participant_spans
    _total_dispatch_ns = 0
    _participant_spans = 0


def _refresh_internal_globals() -> frozenset[int]:
    global _internal_globals, _modules_seen
    if len(sys.modules) != _modules_seen:
        _modules_seen = len(sys.modules)
        _internal_globals = frozenset(
            id(mod.__dict__)
            for name, mod in list(sys.modules.items())
            if (name == "flopscope" or name.startswith("flopscope."))
            and getattr(mod, "__dict__", None) is not None
        )
    return _internal_globals


def _globals_are_internal(namespace: object) -> bool:
    return id(namespace) in _refresh_internal_globals()


def _callable_is_internal(fn: object) -> bool:
    """Whether *fn* was defined inside the flopscope package."""
    # Unwrap the shapes flopscope actually decorates: bound/unbound methods and
    # functools.partial. Anything with no __globals__ (C builtins, arbitrary
    # objects with __call__) is treated as external, which is the safe default.
    seen = 0
    while seen < 10:
        seen += 1
        inner = getattr(fn, "__func__", None) or getattr(fn, "func", None)
        if inner is None or inner is fn:
            break
        fn = inner
    namespace = getattr(fn, "__globals__", None)
    return namespace is not None and _globals_are_internal(namespace)


def _count_participant_span() -> None:
    global _participant_spans
    _participant_spans += 1


def _caller_is_internal(depth: int) -> bool:
    """Whether the frame *depth* levels above this one is flopscope's own code."""
    try:
        return _globals_are_internal(sys._getframe(depth).f_globals)
    except (ValueError, AttributeError):  # pragma: no cover - no Python frame
        return False


@contextmanager
def _null_span():
    """Run the body without touching the accumulator."""
    yield


@contextmanager
def _counted_span():
    """Bracket one full op dispatch; add only this span's own remainder.

    Carries the same caller check as ``dispatch_span``. It is the primitive that
    actually mutates the accumulator, so leaving it unguarded would place the
    check on the wrappers rather than on the thing being protected, and a span
    opened here from outside would both count and leave the span tally at zero.
    """
    global _total_dispatch_ns
    if not _caller_is_internal(depth=3):  # caller -> contextmanager -> here
        _count_participant_span()
        yield
        return
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


def dispatch_span():
    """Bracket one full op dispatch, if the caller is flopscope itself.

    Not a generator: the provenance check must read the CALLER's frame, and by
    the time a ``@contextmanager`` body runs the caller is contextlib.
    """
    if _caller_is_internal(depth=2):  # our caller, not us
        return _counted_span()
    _count_participant_span()
    return _null_span()


def timed_dispatch(fn):
    """Decorator: time the full dispatch of *fn* into the accumulator.

    For a callable defined outside flopscope this returns *fn* unchanged and
    records the span, so the wrapper is a no-op rather than an error: behaviour
    and return values are untouched.
    """
    if not _callable_is_internal(fn):
        _count_participant_span()
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _counted_span():
            return fn(*args, **kwargs)

    return wrapped
