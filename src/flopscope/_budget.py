"""Budget context manager and operation recording for flopscope."""

from __future__ import annotations

import functools
import inspect
import sys as _sys
import threading
import time
import weakref
from typing import Any, Literal, NamedTuple

import numpy as _np

from flopscope._write_epoch import note_write
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
    resolved_dtype: str | None = None  # np.result_type name over declared operands


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
      - backend duration: direct _call_numpy durations reported during the block
      - in-block overhead: the remainder after direct backend, already-recorded
        nested flopscope work, and user callback time
    """

    __slots__ = (
        "_budget",
        "_op_index",
        "_block_t0",
        "_backend_duration_s",
        "_backend_baseline",
        "_overhead_baseline",
        "_usercode_baseline",
        "_prev_timer",
    )

    def __init__(self, budget: BudgetContext, op_index: int):
        self._budget = budget
        self._op_index = op_index
        self._block_t0: float | None = None
        self._backend_duration_s: float = 0.0
        self._backend_baseline: float = 0.0
        self._overhead_baseline: float = 0.0
        self._usercode_baseline: float = 0.0
        self._prev_timer: _OpTimer | _DeferredOpTimer | None = None

    def __enter__(self) -> _OpTimer:
        self._block_t0 = time.perf_counter()
        self._backend_baseline = self._budget._total_flopscope_backend_time
        self._overhead_baseline = self._budget._total_flopscope_overhead_time
        self._usercode_baseline = self._budget._total_user_code_time
        # Stack discipline supports the rare case of nested deduct() blocks
        self._prev_timer = self._budget._current_op_timer
        self._budget._current_op_timer = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        if self._block_t0 is not None:
            block_duration = time.perf_counter() - self._block_t0
            nested = (
                self._budget._total_flopscope_backend_time - self._backend_baseline
            ) + (self._budget._total_flopscope_overhead_time - self._overhead_baseline)
            user_code = self._budget._total_user_code_time - self._usercode_baseline
            in_block_overhead = max(
                block_duration - self._backend_duration_s - nested - user_code, 0.0
            )

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


class _DeferredOpTimer:
    """Timer for ops whose FLOP cost is only known after the numpy call runs.

    Used as ``with budget.deduct_after(name, ...) as op: result =
    _call_numpy(...); op.set_cost(...)``. Times the block like ``_OpTimer`` (so
    the numpy call is recorded as backend), then on exit performs the
    cost-dependent work ``deduct()`` normally does up front: weight -> budget
    check -> charge -> append the ``OpRecord``. Charging at exit matches the
    existing run-then-charge behavior of these ops (they already ran numpy
    before the budget check), so a single-op overshoot still raises
    ``BudgetExhaustedError`` without recording the op.

    ``dtypes`` can likewise only be known after the call for ops whose billed
    dtype is the *output's* promoted dtype (e.g. an assembly op fed
    heterogeneous inputs) -- call ``op.set_dtypes(...)`` to override the
    ``deduct_after()``-time declaration before the block exits.
    """

    __slots__ = (
        "_budget",
        "_op_name",
        "_subscripts",
        "_shapes",
        "_dtypes",
        "_complex_factor_override",
        "_cost",
        "_block_t0",
        "_backend_duration_s",
        "_backend_baseline",
        "_overhead_baseline",
        "_usercode_baseline",
        "_prev_timer",
    )

    def __init__(
        self,
        budget: BudgetContext,
        op_name: str,
        subscripts: str | None,
        shapes: tuple,
        *,
        dtypes: tuple | None = None,
        complex_factor_override: float | None = None,
    ):
        self._budget = budget
        self._op_name = op_name
        self._subscripts = subscripts
        self._shapes = shapes
        self._dtypes = dtypes
        self._complex_factor_override = complex_factor_override
        self._cost: int | None = None
        self._block_t0: float | None = None
        self._backend_duration_s: float = 0.0
        self._backend_baseline: float = 0.0
        self._overhead_baseline: float = 0.0
        self._usercode_baseline: float = 0.0
        self._prev_timer: _OpTimer | _DeferredOpTimer | None = None

    def set_cost(self, flop_cost: int) -> None:
        self._cost = flop_cost

    def set_dtypes(self, dtypes: tuple) -> None:
        """Override the ``dtypes`` declared at ``deduct_after()`` call time.

        Use when the billed dtype is only known once the numpy call has
        run -- e.g. an assembly op (``choose``/``block``/``bmat``) whose
        output dtype is the promoted dtype of its (possibly deeply nested)
        inputs, cheaper to read off the produced result than to re-derive
        from the call's raw arguments. Without this, such ops would have to
        declare ``dtypes=()`` up front, which resolves to the dtype-neutral
        rate 1.0 / complex factor 1.0 -- silently discounting float64 and
        complex inputs regardless of what the registry declares for the op.
        """
        from flopscope._dtype_billing import refuse_non_numeric_dtype

        refuse_non_numeric_dtype(self._op_name, *dtypes)
        self._dtypes = dtypes

    def __enter__(self) -> _DeferredOpTimer:
        self._block_t0 = time.perf_counter()
        self._backend_baseline = self._budget._total_flopscope_backend_time
        self._overhead_baseline = self._budget._total_flopscope_overhead_time
        self._usercode_baseline = self._budget._total_user_code_time
        self._prev_timer = self._budget._current_op_timer
        self._budget._current_op_timer = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        block_duration = time.perf_counter() - self._block_t0  # type: ignore[operator]
        nested = (
            self._budget._total_flopscope_backend_time - self._backend_baseline
        ) + (self._budget._total_flopscope_overhead_time - self._overhead_baseline)
        user_code = self._budget._total_user_code_time - self._usercode_baseline
        in_block_overhead = max(
            block_duration - self._backend_duration_s - nested - user_code, 0.0
        )
        # Pop first so a nested _call_numpy after this block attributes to the
        # restored (outer) timer, not this exited one.
        self._budget._current_op_timer = self._prev_timer
        # Attribute the measured time regardless of how we exit.
        self._budget._total_flopscope_backend_time += self._backend_duration_s
        self._budget._total_flopscope_overhead_time += in_block_overhead
        if exc_type is not None:
            return False  # propagate; nothing charged or recorded
        if self._cost is None:
            raise RuntimeError(
                f"deduct_after({self._op_name!r}): set_cost() was never called"
            )
        self._budget._charge_op(
            self._op_name,
            self._cost,
            self._subscripts,
            self._shapes,
            dtypes=self._dtypes,
            complex_factor_override=self._complex_factor_override,
            backend_duration_s=self._backend_duration_s,
            overhead_duration_s=in_block_overhead,
        )
        return False


# numpy callables that write through their first argument rather than ``out=``.
# Reaching one of these means the destination buffer's contents changed, so any
# symmetry tag observing it must be voided -- see flopscope._write_epoch.
_MUTATES_FIRST_ARG = frozenset(
    {
        _np.copyto,
        _np.put,
        _np.putmask,
        _np.place,
        _np.fill_diagonal,
        _np.put_along_axis,
    }
)


class _PythonCallbackTracker:
    """Profile Python callback roots entered from one raw NumPy call frame."""

    __slots__ = (
        "_budget",
        "_op_timer",
        "_raw_numpy_call_frame_id",
        "_previous_profile",
        "_roots",
        "_restoring",
        "callback_wall_s",
    )

    def __init__(
        self,
        budget: BudgetContext,
        op_timer: _OpTimer | _DeferredOpTimer,
        raw_numpy_call_frame: Any,
        previous_profile: Any,
    ):
        self._budget = budget
        self._op_timer = op_timer
        self._raw_numpy_call_frame_id = id(raw_numpy_call_frame)
        self._previous_profile = previous_profile
        self._roots: dict[Any, tuple[float, float, float, float, float]] = {}
        self._restoring = False
        self.callback_wall_s = 0.0

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        if self._restoring:
            return
        if event == "call" and id(frame.f_back) == self._raw_numpy_call_frame_id:
            self._roots[frame] = (
                time.perf_counter(),
                self._budget._total_flopscope_backend_time,
                self._budget._total_flopscope_overhead_time,
                self._budget._total_user_code_time,
                self._op_timer._backend_duration_s,
            )
        try:
            if self._previous_profile is not None:
                self._previous_profile(frame, event, arg)
        finally:
            if not self._restoring and _sys.getprofile() is not self:
                _sys.setprofile(self)
            if event == "return":
                snapshot = self._roots.pop(frame, None)
                if snapshot is not None:
                    t0, backend0, overhead0, user_code0, timer_backend0 = snapshot
                    wall = time.perf_counter() - t0
                    nested = (self._budget._total_flopscope_backend_time - backend0) + (
                        self._budget._total_flopscope_overhead_time - overhead0
                    )
                    nested += (self._budget._total_user_code_time - user_code0) + (
                        self._op_timer._backend_duration_s - timer_backend0
                    )
                    self.callback_wall_s += wall
                    self._budget._total_user_code_time += max(wall - nested, 0.0)


def _call_numpy_impl(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    track_python_callbacks: bool,
) -> Any:
    """Shared implementation for raw NumPy calls with optional callback timing."""
    out = kwargs.get("out")
    if out is not None:
        note_write(out)
    if fn in _MUTATES_FIRST_ARG and args:
        note_write(args[0])

    budget = get_active_budget()
    op_timer = budget._current_op_timer if budget is not None else None
    tracker: _PythonCallbackTracker | None = None
    previous_profile: Any = None
    fallback_backend0 = 0.0
    fallback_overhead0 = 0.0
    fallback_user_code0 = 0.0
    fallback_timer_backend0 = 0.0
    non_callable_profiler_fallback = False
    t0: float | None = None
    try:
        if track_python_callbacks and budget is not None and op_timer is not None:
            previous_profile = _sys.getprofile()
            if previous_profile is None or callable(previous_profile):
                tracker = _PythonCallbackTracker(
                    budget, op_timer, _sys._getframe(), previous_profile
                )
                _sys.setprofile(tracker)
            else:
                non_callable_profiler_fallback = True
                fallback_backend0 = budget._total_flopscope_backend_time
                fallback_overhead0 = budget._total_flopscope_overhead_time
                fallback_user_code0 = budget._total_user_code_time
                fallback_timer_backend0 = op_timer._backend_duration_s
        t0 = time.perf_counter()
        return fn(*args, **kwargs)
    finally:
        duration = 0.0
        try:
            duration = time.perf_counter() - t0 if t0 is not None else 0.0
        finally:
            if tracker is not None:
                tracker._restoring = True
                try:
                    tracker._roots.clear()
                    tracker._previous_profile = None
                finally:
                    _sys.setprofile(previous_profile)
        if tracker is not None:
            assert op_timer is not None
            op_timer._backend_duration_s += max(duration - tracker.callback_wall_s, 0.0)
        elif non_callable_profiler_fallback:
            assert budget is not None and op_timer is not None
            already_classified = (
                budget._total_flopscope_backend_time - fallback_backend0
            ) + (budget._total_flopscope_overhead_time - fallback_overhead0)
            already_classified += (
                budget._total_user_code_time - fallback_user_code0
            ) + (op_timer._backend_duration_s - fallback_timer_backend0)
            op_timer._backend_duration_s += max(duration - already_classified, 0.0)
        else:
            budget = get_active_budget()
            if budget is not None and budget._current_op_timer is not None:
                budget._current_op_timer._backend_duration_s += duration


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
    return _call_numpy_impl(fn, args, kwargs, track_python_callbacks=False)


def _call_numpy_with_python_callbacks(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke NumPy while filing direct Python protocol callbacks as residual.

    While a counted operation's timer is active, temporarily profiles Python
    frames entered directly from this raw NumPy call. Their full wall time is
    removed from the outer backend duration, while only their non-nested
    flopscope time is added to the user-code bucket used by
    ``_counted_wrapper``. Outside an active timed op this is a transparent
    passthrough, just like ``_call_numpy``. If CPython exposes a non-callable
    active C profiler, unclassified callback time remains backend because the
    C callback cannot be safely multiplexed with a Python profile hook; nested
    work keeps its original timing classification.
    """
    return _call_numpy_impl(fn, args, kwargs, track_python_callbacks=True)


def _call_user_code(budget: BudgetContext, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run user-supplied code (a callback or iterable) and attribute its
    *non-nested* wall time to residual rather than flopscope overhead.

    Snapshots backend+overhead, runs ``fn``, and adds ``wall − nested`` to
    ``budget._total_user_code_time`` (where ``nested`` is any backend/overhead
    accrued by flopscope ops the callback itself ran). ``_counted_wrapper``
    subtracts this delta from its overhead remainder, so the time lands in
    ``residual_wall_time_s`` (= wall − backend − overhead).
    Unlike ``_call_numpy``, the caller passes ``budget`` explicitly (it is
    always in scope inside ``_counted_wrapper``).
    """
    b0 = budget._total_flopscope_backend_time
    o0 = budget._total_flopscope_overhead_time
    t0 = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        wall = time.perf_counter() - t0
        nested = (budget._total_flopscope_backend_time - b0) + (
            budget._total_flopscope_overhead_time - o0
        )
        budget._total_user_code_time += max(wall - nested, 0.0)


#: Per-function cache of ``inspect.signature(fn)``, keyed by function
#: identity. Used to detect a POSITIONALLY-passed ``dtype=`` (e.g.
#: ``zeros(shape, object)``); each wrapped function's signature is fixed
#: for the life of the process, so this is paid once per wrapper, not once
#: per call.
_SIGNATURE_CACHE: dict[Any, inspect.Signature | None] = {}

#: Sentinel distinguishing "no dtype= argument at all" from "dtype=None was
#: passed explicitly": ``None`` is numpy's own "no override" default and
#: must fall through unrefused, while a genuinely absent argument never
#: needs to be checked.
_NO_DTYPE = object()

#: The base ``ndarray.dtype`` getset descriptor, captured once so
#: ``_refuse_non_numeric_operands`` can read an array's real dtype directly
#: rather than through ordinary attribute lookup. Plain ``value.dtype``
#: resolves through ``type(value)``'s MRO, and a ``np.ndarray`` subclass
#: can shadow the C-level slot with its own Python ``@property`` -- calling
#: the base class's descriptor via ``__get__`` bypasses that MRO entry and
#: reads the real dtype unconditionally.
_NDARRAY_DTYPE_DESCRIPTOR = _np.ndarray.dtype

#: Depth cap for ``_refuse_non_numeric_operands``'s list/tuple scan. Deep enough
#: for any legitimate nesting flopscope's own wrappers pass around, but
#: well below Python's recursion limit, so a self-referential container
#: (``a = []; a.append(a)``) cannot turn the scan into a ``RecursionError``
#: -- it just stops looking, and NumPy's own construction raises its usual
#: ``ValueError`` instead.
_DTYPE_SCAN_MAX_DEPTH = 64


def _resolve_dtype_kwarg_value(fn: Any, args: tuple, kwargs: dict) -> Any:
    """Return the effective ``dtype=`` argument bound to *fn*'s call, however
    it was supplied -- keyword (the common case) or positional (e.g. the
    array-creation family's ``zeros(shape, dtype)`` slot). Returns
    ``_NO_DTYPE`` when no ``dtype`` parameter was supplied or the binding
    cannot be determined (mismatched arity, an ``fn`` signature that cannot
    be introspected, ...) -- callers must treat that identically to "no
    dtype requested" rather than raising here, since a malformed call should
    surface numpy's/the wrapper's own error, not a speculative one from this
    backstop.
    """
    if "dtype" in kwargs:
        return kwargs["dtype"]
    try:
        sig = _SIGNATURE_CACHE[fn]
    except KeyError:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None
        _SIGNATURE_CACHE[fn] = sig
    if sig is None or "dtype" not in sig.parameters:
        # No `dtype` PARAMETER in fn's own signature at all -- true for most
        # counted ops. A positional bind to a name absent from the signature
        # is structurally impossible (a `**kwargs` catch-all stores under
        # its own name, already handled by the `"dtype" in kwargs` check
        # above), so skip `bind_partial` -- a real binding pass, not free --
        # entirely in that case; that's the hot-path cost this cache exists
        # to avoid.
        return _NO_DTYPE
    try:
        bound = sig.bind_partial(*args, **kwargs)
    except TypeError:
        return _NO_DTYPE
    return bound.arguments.get("dtype", _NO_DTYPE)


#: Leaf types ``_is_inert_dtype_spec`` accepts inside a list/tuple dtype
#: specifier without recursing further. Each is data, not a duck-typed
#: proxy, so encountering one cannot run participant code -- unlike an
#: object exposing a ``.dtype`` property, which is exactly what this
#: check exists to keep untouched.
_INERT_DTYPE_SPEC_LEAF_TYPES = (str, bytes, type, int, _np.dtype)


def _is_inert_dtype_spec(value: Any, _depth: int = 0) -> bool:
    """Return whether *value* is built only from plainly-inert leaves --
    ``str``/``bytes``/``type``/``int``/``np.dtype``/``None``, or a
    list/tuple nesting the same -- so resolving it with ``np.dtype()``
    cannot run participant code.

    A structured dtype spec is a list/tuple of field tuples, and a field's
    format can itself be a nested structured spec or a subarray shape
    tuple, so this recurses. Depth is capped the same as the operand scan:
    legitimate specs are shallow, and a pathological one is declared
    non-inert rather than walked unbounded.
    """
    if isinstance(value, _INERT_DTYPE_SPEC_LEAF_TYPES) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        if _depth >= _DTYPE_SCAN_MAX_DEPTH:
            return False
        return all(_is_inert_dtype_spec(item, _depth + 1) for item in value)
    return False


def _plain_dtype_like(value: Any) -> _np.dtype | None:
    """Coerce *value* to a ``np.dtype`` WITHOUT touching a ``.dtype``
    property.

    A dtype-LIKE argument's ``.dtype`` can be a stateful property (a
    duck-typed proxy numpy's own ``np.dtype()`` constructor accepts) that
    must be resolved exactly once per call -- reading it here and letting
    the op's own dtype-billing logic read it again could report a
    different dtype the second time. Ops with their own correct ``dtype=``
    handling own that single resolution; this backstop must not add another.

    Only specifiers that are cheap and side-effect-free to resolve are
    handled here: an actual ``np.dtype``, a dtype-code string, a numpy/
    Python scalar type, ``None``, or a list/tuple structured-dtype
    specifier built entirely from such inert leaves (see
    ``_is_inert_dtype_spec``). A list/tuple with a non-inert leaf --
    e.g. a duck-typed ``.dtype`` proxy nested inside a field spec -- is
    left unresolved here for the same reason a bare one is: touching it
    could run participant code. Anything else is left unchecked, for the
    op's own resolution.
    """
    if isinstance(value, _np.dtype):
        return value
    try:
        if isinstance(value, str):
            return _np.dtype(value)
        if isinstance(value, bytes):
            # Accepted by np.dtype() at runtime (it decodes the code
            # string), but neither the __new__ overloads nor
            # numpy.typing.DTypeLike itself declare a bare `bytes` spec.
            return _np.dtype(value)  # type: ignore[call-overload]
        if isinstance(value, type):
            return _np.dtype(value)
        if value is None:
            return _np.dtype(value)
        if isinstance(value, (list, tuple)) and _is_inert_dtype_spec(value):
            return _np.dtype(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    return None


def _refuse_non_numeric_operands(
    fn: Any, args: tuple, kwargs: dict, op_name: str
) -> None:
    """Backstop for ops whose ``dtypes=`` declaration never reaches
    ``deduct()``'s non-numeric-dtype refusal.

    ``op_name`` names the call for the error message and is a SEPARATE
    parameter from ``fn`` -- deliberately, not just read off ``fn.__name__``.
    Many wrappers are built by a factory that defines an inner closure named
    generically (``def wrapper(...): ...``), decorates it, and only then
    renames the DECORATED object returned by the decorator (``wrapper.
    __name__ = op_name``, after ``@_counted_wrapper`` already ran). That
    rename lands on a different object than ``fn`` -- the wrapped closure the
    decorator captured -- so reading ``fn.__name__`` here would report the
    closure's literal definition-time name (``"wrapper"``) forever, no
    matter what the factory renames afterward. The caller passes the live
    name of the object actually being called instead.

    Two gaps land here, both upstream of any FLOP charge:

    * Dtype-neutral movement ops (``reshape``, ``transpose``, ``copy``,
      ``concatenate``, ``take``, ``choice``, ``permutation``, ...) declare
      ``dtypes=()`` because their cost genuinely does not depend on dtype --
      but that also means a non-numeric-dtype operand they relocate never
      reaches any dtype check at all.
    * Array-creation ops (``zeros``, ``empty``, ``asarray``, ``stack``, ...)
      can be asked to manufacture a non-numeric array directly via a
      ``dtype=`` argument the wrapper never folds into its billing
      ``dtypes`` tuple.
      Only specifiers ``_plain_dtype_like`` can resolve without touching a
      ``.dtype`` property are caught here -- an ``np.dtype``, a dtype-code
      string, a type, ``None``, or an inert list/tuple structured spec.
      A dtype-LIKE object whose own ``.dtype`` is a property is left to the
      op's own resolution, which for ``zeros``/``empty`` does not exist.

    Only genuine ``np.ndarray`` instances are inspected, and only through
    ``_NDARRAY_DTYPE_DESCRIPTOR`` rather than plain attribute access:

    * flopscope's own internals pass ordinary non-array defaults through
      counted wrappers (e.g. ``linalg.norm``'s ``ord=None``), and
      ``np.asarray(None)`` is itself an object-dtype 0-d array -- coercing
      every argument via ``np.asarray``/``np.dtype`` would refuse perfectly
      ordinary calls.
    * A caller can hand this check any object exposing a ``.dtype``
      property, and plain ``getattr`` would run that property's body as a
      side effect of the ban's own check, whether or not the check
      ultimately raises. A ``np.ndarray`` subclass can also shadow the
      C-level ``dtype`` slot with such a property, so even a type-gated
      ``value.dtype`` isn't safe -- reading through the base descriptor's
      own ``__get__`` bypasses the subclass's MRO entry and returns the
      real dtype unconditionally.
    """
    from flopscope._dtype_billing import refuse_non_numeric_dtype

    # `scanned`: every container id this scan has already fully walked, kept
    # for the whole call so a container reachable by more than one path (a
    # shared, non-cyclic sublist) is not walked again -- it is the same
    # object with the same contents each time, so the one walk that runs the
    # first time it is reached already finds anything inside it.
    scanned: set[int] = set()
    # `active`: container ids currently on the recursion path (added before
    # descending, removed on the way back out). A container whose id is
    # already `active` is its own ancestor -- a genuine cycle, not merely a
    # shared reference -- and cannot be realized as an array at all. NumPy's
    # own construction would reach the same conclusion, but only after
    # exploring every branch down to its own dimension limit, which is
    # exponential in depth for a container with more than one self-reference;
    # raising here reaches the same outcome without that cost.
    active: set[int] = set()

    def check(value: Any, _depth: int = 0) -> None:
        if isinstance(value, _np.ndarray):
            refuse_non_numeric_dtype(op_name, _NDARRAY_DTYPE_DESCRIPTOR.__get__(value))
        elif isinstance(value, (list, tuple)) and _depth < _DTYPE_SCAN_MAX_DEPTH:
            marker = id(value)
            if marker in active:
                raise ValueError(
                    f"{op_name}: cannot construct an array from a "
                    "self-referential sequence"
                )
            if marker in scanned:
                return
            scanned.add(marker)
            active.add(marker)
            try:
                for item in value:
                    check(item, _depth + 1)
            finally:
                active.discard(marker)

    dtype_kwarg = _resolve_dtype_kwarg_value(fn, args, kwargs)
    if dtype_kwarg is not _NO_DTYPE and dtype_kwarg is not None:
        plain = _plain_dtype_like(dtype_kwarg)
        if plain is not None:
            refuse_non_numeric_dtype(op_name, plain)

    for value in args:
        check(value)
    for key, value in kwargs.items():
        if key != "dtype":
            check(value)


def refuse_non_numeric_source(op_name: str, value: Any) -> None:
    """Refuse *value* if it carries a non-numeric dtype, without ever
    casting a payload through a numeric dtype to find out.

    Complements ``_refuse_non_numeric_operands`` above: that backstop only
    recognizes a non-numeric array already boxed as ``np.ndarray`` (or
    nested inside a list/tuple looking for one) -- it deliberately does not
    realize a bare payload or a raw Python sequence of them, since doing
    that for every argument of every counted op would cost a full array
    conversion on the hot path for no reason. Call sites that are about to
    cast a genuinely uninspected source through a dtype -- a fill value, a
    distribution parameter, a materialized iterator -- call this at that one
    point instead.

    Recognized scalar types (``None``, ``bool``, ``int``, ``float``,
    ``complex``, ``str``, ``bytes``, a numpy scalar) are skipped untouched --
    ``np.asarray(None)`` is itself an object-dtype 0-d array, so probing
    every ordinary default would refuse perfectly normal calls, the same
    trap ``_refuse_non_numeric_operands`` avoids. A genuine ``np.ndarray`` is
    checked directly through the base ``dtype`` descriptor, so a hostile
    subclass cannot shadow it. Anything else -- a bare payload object, or a
    list/tuple of them -- is realized with a dtype-free ``np.asarray()``,
    which stores object pointers rather than casting, so no per-element
    caller code runs before the check.
    """
    if value is None or isinstance(
        value, (bool, int, float, complex, str, bytes, _np.generic)
    ):
        return
    from flopscope._dtype_billing import refuse_non_numeric_dtype

    if isinstance(value, _np.ndarray):
        refuse_non_numeric_dtype(op_name, _NDARRAY_DTYPE_DESCRIPTOR.__get__(value))
        return
    refuse_non_numeric_dtype(op_name, _np.asarray(value).dtype)


def _counted_wrapper(fn):
    """Decorator that brackets a flopscope wrapper and bills its non-numpy,
    non-nested-overhead time to flopscope_overhead_time_s.

    Formula: wall - backend_delta - overhead_delta. Handles nesting naturally
    (outer attributes only its own remainder), so no re-entrancy guard.

    Per-op attribution: wrapper-own overhead is distributed equally across
    ops created during this call (typically exactly 1 across this codebase).

    Also runs ``_refuse_non_numeric_operands`` as the first thing inside the
    timed block -- see its docstring. This fires for every registered op,
    not just billed ones: even a 0-FLOP op like ``zeros`` can be asked to
    manufacture a non-numeric array, which the cost model cannot price.
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        from flopscope._validation import require_budget

        budget = require_budget()
        fs_t0 = time.perf_counter()
        backend_baseline = budget._total_flopscope_backend_time
        overhead_baseline = budget._total_flopscope_overhead_time
        usercode_baseline = budget._total_user_code_time
        ops_before = len(budget._op_log)
        try:
            # wrapped.__name__, not fn.__name__: a factory that renames its
            # returned wrapper after decorating (`wrapper.__name__ = op_name`)
            # renames THIS object, not the closure `fn` still refers to --
            # see _refuse_non_numeric_operands's docstring. Read live, at call
            # time, so it reflects whatever the factory's rename left behind.
            _refuse_non_numeric_operands(fn, args, kwargs, wrapped.__name__)
            return fn(*args, **kwargs)
        finally:
            wall = time.perf_counter() - fs_t0
            backend_delta = budget._total_flopscope_backend_time - backend_baseline
            overhead_delta = budget._total_flopscope_overhead_time - overhead_baseline
            usercode_delta = budget._total_user_code_time - usercode_baseline
            wrapper_own_overhead = max(
                wall - backend_delta - overhead_delta - usercode_delta, 0.0
            )
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
        quiet: bool = False,
        namespace: str | None = None,
        wall_time_limit_s: float | None = None,
    ):
        # Capture creation time as the very first thing so any work done in
        # __init__ (validation, list/dict allocation) is included in the
        # eventual wall_time_s span, billed to flopscope_overhead_time_s. See
        # issue #82.
        self._creation_time = time.perf_counter()
        if flop_budget <= 0:
            raise ValueError(f"flop_budget must be > 0, got {flop_budget}")
        self._flop_budget = flop_budget
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
        self._total_user_code_time: float = 0.0
        self._pre_enter_overhead: float = 0.0
        self._current_op_timer: _OpTimer | _DeferredOpTimer | None = None
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

        The decomposition ``wall_time_s == flopscope_backend_time_s
        + flopscope_overhead_time_s + residual_wall_time_s`` holds within numerical
        tolerance. The pre-``__enter__`` slice (``__init__`` body + banner print)
        and the post-``__exit__`` body slice (accumulator-record,
        active-budget restore) are both attributed to
        ``flopscope_overhead_time_s``.
        """
        return self._wall_time_s

    @property
    def elapsed_s(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def flopscope_backend_time_s(self) -> float:
        """Wall-clock seconds of the counted NumPy/BLAS/LAPACK backend calls.

        Includes the numpy call of data-movement ops (e.g. ``tile``, ``take``,
        ``pad``, ``resize``), which run inside ``deduct_after``.
        """
        return self._total_flopscope_backend_time

    @property
    def flopscope_overhead_time_s(self) -> float:
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
    def residual_wall_time_s(self) -> float | None:
        """Wall time minus flopscope backend and overhead time.

        This is the measured wall-clock remainder outside the backend calls and
        flopscope's own dispatch/accounting work: user Python between ops,
        time.sleep, GC pauses, and un-instrumented NumPy. User-callback ops
        (``apply_along_axis``, ``apply_over_axes``, ``piecewise``,
        ``fromfunction``, ``fromiter``) attribute their callback wall time here.
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

    def _charge_op(
        self,
        op_name: str,
        flop_cost: int,
        subscripts: str | None,
        shapes: tuple,
        *,
        dtypes: tuple | None = None,
        complex_factor_override: float | None = None,
        backend_duration_s: float | None = None,
        overhead_duration_s: float | None = None,
    ) -> None:
        """Weight/dtype-rate/complex-factor -> budget-check -> charge -> append OpRecord -> post-op deadline check.

        Shared by deduct() (durations filled in later by _OpTimer.__exit__, so left
        None here) and _DeferredOpTimer.__exit__ (durations already known). Resolves
        the billing dtype and looks up its rate/complex-factor before the budget
        check, so an UnsupportedDtypeError raises before any FLOPs are recorded.
        Raises BudgetExhaustedError before charging on overshoot; raises
        TimeExhaustedError after the record is appended if the deadline has passed.

        ``dtypes=None`` (the default) bills as dtype-neutral (rate 1.0, factor
        1.0). Both public callers (``deduct`` and ``_DeferredOpTimer.__exit__``,
        reached via ``deduct_after``) now require a caller-supplied ``dtypes``
        tuple and raise ``TypeError`` before reaching here if it is ``None``, so
        this default is unreachable from them; it remains as this internal
        choke point's own defensive fallback.
        """
        from flopscope._dtype_billing import (
            complex_factor_for,
            rate_for,
            resolve_billing_dtype,
        )
        from flopscope._weights import get_weight

        weight = get_weight(op_name)
        resolved = resolve_billing_dtype(dtypes) if dtypes is not None else None
        if resolved is None:
            dtype_rate = 1.0
            complex_factor = 1.0
        else:
            dtype_rate = rate_for(resolved)
            complex_factor = (
                complex_factor_override
                if complex_factor_override is not None
                else complex_factor_for(op_name, resolved)
            )
        adjusted_cost = int(flop_cost * dtype_rate * complex_factor * weight)
        if adjusted_cost > self.flops_remaining:
            raise BudgetExhaustedError(
                op_name, flop_cost=adjusted_cost, flops_remaining=self.flops_remaining
            )
        self._flops_used += adjusted_cost
        now = time.perf_counter()
        offset = now - self._start_time if self._start_time is not None else None
        self._op_log.append(
            OpRecord(
                op_name=op_name,
                subscripts=subscripts,
                shapes=shapes,
                flop_cost=adjusted_cost,
                cumulative=self._flops_used,
                namespace=self.namespace,
                flopscope_context_start_offset_s=offset,
                flopscope_backend_duration_s=backend_duration_s,
                flopscope_overhead_duration_s=overhead_duration_s,
                resolved_dtype=resolved.name if resolved is not None else None,
            )
        )
        if self._deadline is not None and now > self._deadline:
            from flopscope.errors import TimeExhaustedError

            raise TimeExhaustedError(
                op_name,
                elapsed_s=now - self._start_time,  # type: ignore[operator]
                limit_s=self._wall_time_limit_s,  # type: ignore[arg-type]
            )

    def deduct(
        self,
        op_name: str,
        *,
        flop_cost: int,
        subscripts: str | None,
        shapes: tuple,
        dtypes: tuple | None,
        complex_factor_override: float | None = None,
    ) -> _OpTimer:
        """Deduct FLOPs from the budget and return a timer context manager."""
        if dtypes is None:
            raise TypeError(
                f"deduct({op_name!r}): dtypes= is required; pass () for a "
                "dtype-neutral op"
            )
        from flopscope._dtype_billing import refuse_non_numeric_dtype

        refuse_non_numeric_dtype(op_name, *dtypes)
        fs_t0 = time.perf_counter()
        n0 = len(self._op_log)
        try:
            self._charge_op(
                op_name,
                flop_cost,
                subscripts,
                shapes,
                dtypes=dtypes,
                complex_factor_override=complex_factor_override,
            )
            return _OpTimer(self, op_index=len(self._op_log) - 1)
        finally:
            deduct_body_time = time.perf_counter() - fs_t0
            self._total_flopscope_overhead_time += deduct_body_time
            if len(self._op_log) > n0:
                op = self._op_log[-1]
                self._op_log[-1] = op._replace(
                    flopscope_overhead_duration_s=(
                        op.flopscope_overhead_duration_s or 0.0
                    )
                    + deduct_body_time
                )

    def deduct_after(
        self,
        op_name: str,
        *,
        subscripts: str | None,
        shapes: tuple,
        dtypes: tuple | None,
        complex_factor_override: float | None = None,
    ) -> _DeferredOpTimer:
        """Like :meth:`deduct`, but the FLOP cost is supplied via ``op.set_cost``
        inside the block and charged at block exit. Use for ops whose cost
        depends on the result; the numpy call runs inside the timer (via
        ``_call_numpy``) and is recorded as backend time.
        """
        if dtypes is None:
            raise TypeError(
                f"deduct({op_name!r}): dtypes= is required; pass () for a "
                "dtype-neutral op"
            )
        from flopscope._dtype_billing import refuse_non_numeric_dtype

        refuse_non_numeric_dtype(op_name, *dtypes)
        return _DeferredOpTimer(
            self,
            op_name,
            subscripts,
            shapes,
            dtypes=dtypes,
            complex_factor_override=complex_factor_override,
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
        wall_time, backend_time, overhead_time, residual_wall_time_s = _timing_summary(
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
            "residual_wall_time_s": residual_wall_time_s,
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
        # __exit__ adds it to flopscope_overhead_time_s. See issue #82.
        self._start_time = time.perf_counter()
        self._pre_enter_overhead = self._start_time - self._creation_time
        if self._wall_time_limit_s is not None:
            self._deadline = self._start_time + self._wall_time_limit_s
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # body_end snapshots the user-code window (start_time → here). Then
        # we do the post-exit accumulator work and snapshot post_exit_end.
        # wall_time_s spans creation → post_exit_end, and the pre-enter +
        # post-exit slices are billed to flopscope_overhead_time_s. See #82.
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

        wall_time, backend_time, overhead_time, residual_wall_time_s = _timing_summary(
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
            "residual_wall_time_s": residual_wall_time_s,
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
