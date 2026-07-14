# src/flopscope/_unwrap.py
"""Unwrap wrapper with FLOP counting."""

from __future__ import annotations

import numpy as _np
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._ndarray import FlopscopeArray, _to_base_ndarray
from flopscope._validation import require_budget


def unwrap_cost(shape: tuple[int, ...]) -> int:
    """FLOP cost of phase unwrapping.

    Parameters
    ----------
    shape : tuple of int
        Input array shape.

    Returns
    -------
    int
        Estimated FLOP count: ``11 * numel(input)``.

    Notes
    -----
    NumPy's unwrap (``numpy.lib._function_base_impl.unwrap``) performs 13
    one-FLOP ufunc passes over the N-1 element ``dd = diff(p)`` array, then
    one final add of the cumulative correction into the output.  Counting
    each named ufunc call as one pass:

    1.  ``diff``            — subtract adjacent elements (N-1 elements)
    2.  ``dd - low``        — subtract scalar ``interval_low`` from dd
    3.  ``mod(..., period)``— elementwise modulo
    4.  ``+ interval_low``  — add scalar back
    5.  ``ddmod == low``    — elementwise compare (boundary check)
    6.  ``dd > 0``          — elementwise compare
    7.  ``& ``              — bitwise-and of two bool arrays
    8.  ``copyto/select``   — conditional write (boundary fix) **FREE** (3-arg where)
    9.  ``ddmod - dd``      — elementwise subtract (ph_correct)
    10. ``abs(dd)``         — elementwise absolute value
    11. ``< discont``       — elementwise compare
    12. ``copyto/select``   — conditional write (small-jump zeroing) **FREE** (3-arg where)
    13. ``cumsum``          — prefix-sum scan

    The ``p[slice1] + ph_correct.cumsum(axis)`` expression (final output
    materialization) involves one add pass but is treated as part of the
    output-write cost — following the convention that the output buffer
    fill is attributed to the issuing op.  Charging all 11 *non-free* op-passes
    against ``numel(input)`` rather than ``N-1`` avoids tracking the
    edge-element correction (one extra element) and gives a clean formula.

    Steps 8 and 12 (``copyto/select`` via 3-arg ``where``) are pure data
    movement (selecting by a given mask) and are therefore free per the
    cost-model rule introduced with the where-arity branching (Task 4,
    2026-06-15).  Prior value was ``13 * numel`` (included 2 free selects);
    before that it was ``7 * numel`` (Audit-completion Task 4, 2026-06-12).
    """
    numel = 1
    for d in shape:
        numel *= d
    return max(11 * numel, 1)


@_counted_wrapper
def unwrap(
    p: ArrayLike,
    discont: float | None = None,
    axis: int = -1,
    *,
    period: float = 6.283185307179586,
) -> FlopscopeArray:
    budget = require_budget()
    if not isinstance(p, _np.ndarray):
        p = _np.asarray(p)
    cost = unwrap_cost(p.shape)
    kwargs = {"axis": axis, "period": period}
    if discont is not None:
        kwargs["discont"] = discont
    with budget.deduct(
        "unwrap", flop_cost=cost, subscripts=None, shapes=(p.shape,), dtypes=(p.dtype,)
    ):
        result = _call_numpy(_np.unwrap, _to_base_ndarray(p), **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(unwrap, _np.unwrap, "counted_custom", "11 * numel(input) FLOPs")

import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
