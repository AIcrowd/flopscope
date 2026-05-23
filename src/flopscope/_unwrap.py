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
        Estimated FLOP count: ``7 * numel(input)``.

    Notes
    -----
    NumPy's unwrap (``numpy.lib._function_base_impl.unwrap``) does
    approximately 6 full-array ufunc passes: ``diff``, ``mod``, ``add``,
    ``subtract``, ``cumsum``, and ``where``. The ``where`` pass with two
    real-array arguments charges 2*(N-1) elements rather than N-1,
    bringing the effective total to ~7*(N-1). We charge a fixed
    ``7 * numel`` to approximate the sum without tracking numpy's
    internal implementation byte-by-byte. Issue #69.
    """
    numel = 1
    for d in shape:
        numel *= d
    return max(7 * numel, 1)


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
    with budget.deduct("unwrap", flop_cost=cost, subscripts=None, shapes=(p.shape,)):
        result = _call_numpy(_np.unwrap, _to_base_ndarray(p), **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(unwrap, _np.unwrap, "counted_custom", "7 * numel(input) FLOPs")

import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
