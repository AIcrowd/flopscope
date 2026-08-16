# src/flopscope/fft/_free.py
"""Zero-FLOP FFT utility operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as _np
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._ndarray import FlopscopeArray, _to_base_ndarray
from flopscope._validation import require_budget


@_counted_wrapper
def fftfreq(n: int, d: float = 1.0, device: Any = None) -> FlopscopeArray:
    """FFT sample frequencies. Cost: n FLOPs.

    ``np.fft.fftfreq`` scales an integer index grid by ``1/(n*d)`` — one
    float multiply per returned frequency, i.e. ``n`` FLOPs (the grid length).
    """
    budget = require_budget()
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    with budget.deduct(
        "fft.fftfreq",
        flop_cost=max(int(n), 1),
        subscripts=None,
        shapes=((n,),),
        # fftfreq grids are always float64; bill that width.
        dtypes=(_np.dtype(_np.float64),),
    ):
        result = _call_numpy(_np.fft.fftfreq, n, d=d, **kwargs)
    return result  # type: ignore[reportReturnType]


attach_docstring(
    fftfreq, _np.fft.fftfreq, "counted_custom", "n FLOPs (index grid / (n*d))"
)


@_counted_wrapper
def rfftfreq(n: int, d: float = 1.0, device: Any = None) -> FlopscopeArray:
    """Real FFT sample frequencies. Cost: n//2 + 1 FLOPs.

    ``np.fft.rfftfreq`` scales ``arange(0, n//2 + 1)`` by ``1/(n*d)`` — one
    float divide per returned frequency, i.e. ``n//2 + 1`` FLOPs (grid length).
    """
    budget = require_budget()
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    grid = int(n) // 2 + 1
    with budget.deduct(
        "fft.rfftfreq",
        flop_cost=max(grid, 1),
        subscripts=None,
        shapes=((grid,),),
        # fftfreq grids are always float64; bill that width.
        dtypes=(_np.dtype(_np.float64),),
    ):
        result = _call_numpy(_np.fft.rfftfreq, n, d=d, **kwargs)
    return result  # type: ignore[reportReturnType]


attach_docstring(
    rfftfreq,
    _np.fft.rfftfreq,
    "counted_custom",
    "n//2 + 1 FLOPs (rfft index grid / (n*d))",
)


@_counted_wrapper
def fftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> FlopscopeArray:
    """Shift zero-frequency component to center. Cost: numel(output)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    # numpy.fft.fftshift is a roll (shape- and dtype-preserving reindex, no
    # arithmetic), so numel(output) == x_arr.size and result.dtype == x_arr.dtype
    # unconditionally -- billing from the pre-call array keeps the numpy call
    # inside the timed deduct() block.
    cost = max(x_arr.size, 1)
    with budget.deduct(
        "fft.fftshift",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=(x_arr.dtype,),
    ):
        result = _call_numpy(_np.fft.fftshift, _to_base_ndarray(x), axes=axes)
    return result  # type: ignore[reportReturnType]


attach_docstring(fftshift, _np.fft.fftshift, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def ifftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> FlopscopeArray:
    """Inverse of fftshift. Cost: numel(output)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    # Same roll-based reindex as fftshift -- see the comment there.
    cost = max(x_arr.size, 1)
    with budget.deduct(
        "fft.ifftshift",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=(x_arr.dtype,),
    ):
        result = _call_numpy(_np.fft.ifftshift, _to_base_ndarray(x), axes=axes)
    return result  # type: ignore[reportReturnType]


attach_docstring(ifftshift, _np.fft.ifftshift, "counted_custom", "numel(output) FLOPs")

import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

# All four wrappers handed back the raw ndarray ``_call_numpy`` produced, so
# arithmetic on an fftfreq grid or a shifted array billed 0 in-process while the
# grader billed it through the client's RemoteArray (#193).
#
# Module-wide rather than four hand-edited return lines because this module
# holds nothing but those four metered ops: the default ``check_module=True``
# filter skips every other public name here (``attach_docstring``,
# ``require_budget``, ``FlopscopeArray``, ``Sequence``), all of which are
# imports. Pinned by
# ``test_fft_free_module_wrap_reaches_only_the_four_metered_ops``.
#
# Wire-neutral on the grader because none of the four can return a scalar --
# the one shape where a wrap flips a by-value RemoteScalar into an array
# handle, which is a repricing. ``fftfreq``/``rfftfreq`` always build a rank-1
# grid, and ``fftshift``/``ifftshift`` of a 0-d input is refused inside
# ``numpy.roll`` before the return line is reached. Both measured, not assumed
# -- see ``test_fft_free_ops_have_no_scalar_shape_to_trap_on`` and
# ``test_fft_shift_refuses_zero_d_input_before_flopscope_returns``.
#
# No ``skip_names``: none of the four takes ``out=``, so there is no caller
# buffer whose identity a wrap could break.
_wrap_module_returns(_sys.modules[__name__])
