"""Input validation and NaN/Inf checking for flopscope operations."""

from __future__ import annotations

import builtins as _builtins
import warnings
from typing import Any

import numpy as np

from flopscope._budget import get_active_budget
from flopscope.errors import FlopscopeWarning


def require_budget():
    """Return the active budget, auto-activating the global default if needed."""
    budget = get_active_budget()
    if budget is not None:
        return budget
    from flopscope._budget import _get_global_default

    return _get_global_default()


def _normalize_out(out: object, op_name: str, *, nout: int = 1) -> Any:
    """Reduce ``out=`` to the destination array itself, or refuse it.

    numpy's ufunc protocol lets a caller pass the destination either bare or
    inside a tuple of length ``nout``; ``np.multiply(a, b, out=(dest,))`` and
    ``out=dest`` mean the same thing. flopscope has to see through the tuple
    for itself, because it reads ``out`` several times before numpy ever gets
    it -- to pick the billing dtype, to check symmetry, and to decide what to
    hand back. A tuple slipping past those reads is not a cosmetic difference:
    the destination's dtype stops participating in the rate, so a contraction
    into a wider buffer bills as if the buffer were not there.

    Worse on the einsum path, which never forwards ``out`` to numpy at all:
    there a container reaches ``_np.asarray(container)``, which builds a NEW
    array, so the result lands in that temporary, the real destination keeps
    its old contents, and the caller gets the untouched container back having
    paid in full.

    So: unwrap what numpy would unwrap, refuse everything else, and do it
    before a single FLOP is charged.

    Returns the value ``out`` should be for the rest of the call -- ``None``,
    the bare destination, or (for a multi-output op) the tuple unchanged.
    Typed ``Any`` because which of those it is depends on ``nout``, and
    every caller assigns it straight back over its own ``out`` parameter.
    """
    if out is None or isinstance(out, np.ndarray):
        return out
    # ``type(out) is tuple``, not ``isinstance``: numpy refuses a namedtuple
    # or any tuple subclass here, and being more permissive than numpy buys
    # nothing. ``_builtins.all`` rather than ``all``: ``all`` is itself a
    # counted flopscope operation, and the modules that call this helper
    # rebind the name to it -- validating an argument must never bill.
    if (
        type(out) is tuple
        and len(out) == nout
        and _builtins.all(o is None or isinstance(o, np.ndarray) for o in out)
    ):
        return out[0] if nout == 1 else out
    if nout != 1:
        length = len(out) if isinstance(out, (tuple, list)) else "?"
        raise TypeError(
            f"multi-output {op_name} requires out= to be a tuple of length "
            f"{nout}; got {type(out).__name__} of length {length}"
        )
    # numpy's own wording ("return arrays must be of ArrayType") is kept in
    # the message on purpose. Code in the wild matches on it -- numpy's fft
    # test suite among it -- and intercepting the argument earlier than numpy
    # does should not change what the failure looks like to a caller.
    raise TypeError(
        f"{op_name}(): return arrays must be of ArrayType -- out= must be an "
        f"array, or a 1-tuple holding one, not {type(out).__name__}. Pass the "
        f"destination array itself, not a container holding it."
    )


def validate_ndarray(*arrays: object) -> None:
    """Validate that all arguments are numpy ndarrays."""
    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(arr).__name__}")


def coerce_array(x):
    """Convert input to ndarray if not already one, matching NumPy's behavior."""
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def coerce_arrays(*arrays):
    """Convert multiple inputs to ndarrays."""
    return tuple(coerce_array(a) for a in arrays)


def check_nan_inf(result: np.ndarray, op_name: str) -> None:
    """Issue a warning if result contains NaN or Inf values.

    Skips dtypes that don't support `np.isnan`/`np.isinf` (e.g. object,
    integer, complex with object content) — these can never contain NaN
    or Inf as ndarray values, so the check is a no-op.
    """
    if not isinstance(result, np.ndarray):
        return
    # np.isnan/np.isinf only support float and complex dtypes. For other
    # dtypes (object, integer, bool, structured), there are no NaN/Inf
    # values to detect, so skip the check.
    if result.dtype.kind not in ("f", "c"):
        return
    # Strip flopscope subclasses so np.isnan / np.isinf do not re-dispatch
    # through __array_ufunc__ and recurse into me.isnan / me.isfinite,
    # which in turn would call check_nan_inf again.
    plain = result.view(np.ndarray) if type(result) is not np.ndarray else result
    nan_count = int(np.isnan(plain).sum())
    inf_count = int(np.isinf(plain).sum())
    if nan_count > 0 or inf_count > 0:
        warnings.warn(
            f"{op_name} produced {nan_count} NaN and {inf_count} Inf values "
            f"in output of shape {result.shape}",
            FlopscopeWarning,
            stacklevel=3,
        )


def maybe_check_nan_inf(result: object, op_name: str) -> None:
    """Run :func:`check_nan_inf` only if the global ``check_nan_inf`` setting is on.

    Production scoring runs with the setting off (default) and pays no
    per-op O(n) scan cost.  Debug callers opt in via
    ``flopscope.configure(check_nan_inf=True)``.
    """
    from flopscope._config import get_setting

    if not get_setting("check_nan_inf"):
        return
    if isinstance(result, np.ndarray):
        check_nan_inf(result, op_name)
