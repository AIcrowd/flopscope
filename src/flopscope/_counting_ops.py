"""Counted wrappers for trace, histogram, and generation operations.

These are operations that look "free" but involve genuine computation.
Each function charges a FLOP cost to the active budget.
"""

from __future__ import annotations

import builtins as _builtins
import inspect as _inspect
import math as _math
from collections.abc import Sequence
from typing import Any

import numpy as _np
from numpy.typing import ArrayLike, DTypeLike

from flopscope._budget import _call_numpy, _call_user_code, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._flops import _ceil_log2
from flopscope._ndarray import FlopscopeArray, _to_base_ndarray, _to_base_ndarray_tree
from flopscope._validation import require_budget
from flopscope.errors import _warn_remote_callback

# Estimator surcharge multipliers for string-bins histogram calls (FLOPs/elem above 2n min/max floor)
# Keys: numpy string estimator names; values: extra cost per element for the estimator step
# References: percentile precedent numel@1.0 (fd/auto=1n), std=4n (scott), doane~6n, stone=max(100,isqrt(n))
_HIST_ESTIMATOR_COST: dict[str, int] = {
    "sturges": 0,
    "sqrt": 0,
    "rice": 0,
    "fd": 1,
    "auto": 1,
    "scott": 4,
    "doane": 6,
}

# ---------------------------------------------------------------------------
# Reductions disguised as free
# ---------------------------------------------------------------------------


@_counted_wrapper
def trace(
    a: ArrayLike,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: DTypeLike | None = None,
    out: FlopscopeArray | None = None,
) -> FlopscopeArray:
    budget = require_budget()
    a = _np.asarray(a)
    # Normalise negative axes
    ndim = a.ndim
    ax1 = axis1 % ndim if ndim > 0 else 0
    ax2 = axis2 % ndim if ndim > 0 else 0
    if ndim >= 2 and a.shape[ax1] > 0 and a.shape[ax2] > 0:
        k = _builtins.max(_builtins.min(a.shape[ax1], a.shape[ax2]), 1)
        # Number of independent trace evaluations = product of all other axes
        n_traces = a.size // (a.shape[ax1] * a.shape[ax2])
        cost = k * n_traces
    else:
        cost = 0
    out_stripped = _to_base_ndarray(out) if out is not None else None
    billing_dtypes: tuple = (a.dtype,)
    if dtype is not None:
        billing_dtypes += (_np.dtype(dtype),)
    if out_stripped is not None:
        billing_dtypes += (out_stripped.dtype,)
    with budget.deduct(
        "trace",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(
            _np.trace,
            _to_base_ndarray(a),
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            dtype=dtype,
            out=out_stripped,
        )
    return out if out is not None else result  # type: ignore[return-value]


attach_docstring(
    trace,
    _np.trace,
    "counted_custom",
    "min(a.shape[axis1], a.shape[axis2]) × batch FLOPs (diagonal sum per matrix)",
)


@_counted_wrapper
def allclose(a: ArrayLike, b: ArrayLike, **kwargs: Any) -> bool:
    budget = require_budget()
    a = _np.asarray(a)
    b = _np.asarray(b)
    out_shape = _np.broadcast_shapes(a.shape, b.shape)
    numel = 1
    for d in out_shape:
        numel *= d
    # 6 FLOPs/elem tolerance core (sub + 2*abs + mul + add + cmp) + (numel-1) all-reduce
    cost = _builtins.max(7 * numel - 1, 1)
    with budget.deduct(
        "allclose",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, b.shape),
        dtypes=(a.dtype, b.dtype),
    ):
        result = _call_numpy(_np.allclose, a, b, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(
    allclose,
    _np.allclose,
    "counted_custom",
    "7*numel(broadcast) - 1 FLOPs (6/elem tolerance core + all-reduce)",
)
allclose.__signature__ = _inspect.signature(_np.allclose)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def array_equal(a: ArrayLike, b: ArrayLike, **kwargs: Any) -> bool:
    budget = require_budget()
    a = _np.asarray(a)
    b = _np.asarray(b)
    # array_equal does not broadcast; returns False on shape mismatch
    cost = _builtins.max(a.size, b.size, 1)
    with budget.deduct(
        "array_equal",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, b.shape),
        dtypes=(a.dtype, b.dtype),
    ):
        result = _call_numpy(_np.array_equal, a, b, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(array_equal, _np.array_equal, "counted_custom", "numel(a) FLOPs")
array_equal.__signature__ = _inspect.signature(_np.array_equal)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def array_equiv(a: ArrayLike, b: ArrayLike) -> bool:
    budget = require_budget()
    a = _np.asarray(a)
    b = _np.asarray(b)
    # array_equiv broadcasts; returns False if shapes are incompatible
    try:
        out_shape = _np.broadcast_shapes(a.shape, b.shape)
        numel = 1
        for d in out_shape:
            numel *= d
        cost = _builtins.max(numel, 1)
    except ValueError:
        cost = _builtins.max(a.size, b.size, 1)
    with budget.deduct(
        "array_equiv",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, b.shape),
        dtypes=(a.dtype, b.dtype),
    ):
        result = _call_numpy(_np.array_equiv, a, b)
    return result  # type: ignore[return-value]


attach_docstring(array_equiv, _np.array_equiv, "counted_custom", "numel(a) FLOPs")
array_equiv.__signature__ = _inspect.signature(_np.array_equiv)  # pyright: ignore[reportFunctionMemberAccess]


# ---------------------------------------------------------------------------
# Histogram & counting
# ---------------------------------------------------------------------------


@_counted_wrapper
def histogram(
    a: ArrayLike,
    bins: int | Sequence[int] | str = 10,
    **kwargs: Any,
) -> tuple[FlopscopeArray, FlopscopeArray]:
    budget = require_budget()
    a = _np.asarray(a)
    n = a.size
    if isinstance(bins, _builtins.int):
        cost = _builtins.max(n * _ceil_log2(bins), 1)
    elif isinstance(bins, _builtins.str):
        # Deferred cost: resolve nbins from returned edges, then charge
        # 2n (min/max scan) + estimator_cost*n + n*ceil_log2(nbins_resolved)
        with budget.deduct_after(
            "histogram", subscripts=None, shapes=(a.shape,), dtypes=(a.dtype,)
        ) as _op:
            result = _call_numpy(
                _np.histogram,
                a,
                bins=bins,
                **{k: _to_base_ndarray_tree(v) for k, v in kwargs.items()},
            )
            nbins = _builtins.max(_builtins.len(result[1]) - 1, 1)
            est = (
                _builtins.max(100, _math.isqrt(n))
                if bins == "stone"
                else _HIST_ESTIMATOR_COST.get(bins, 1)
            )
            _op.set_cost(_builtins.max(n * (2 + est + _ceil_log2(nbins)), 1))
        return result  # type: ignore[return-value]
    else:
        bins_arr = _np.asarray(bins)
        cost = _builtins.max(n * _ceil_log2(_builtins.len(bins_arr)), 1)
    # An array ``bins`` (edge values) is a genuine second operand — numpy
    # promotes across data and edges — so its dtype must join the billing tuple
    # or a complex/fp64 edge array bypasses the factor. An int ``bins`` is a
    # count, not data, so its dtype is irrelevant.
    _hist_dtypes = (
        (a.dtype,) if _np.isscalar(bins) else (a.dtype, _np.asarray(bins).dtype)
    )
    with budget.deduct(
        "histogram",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=_hist_dtypes,
    ):
        result = _call_numpy(
            _np.histogram,
            a,
            bins=_to_base_ndarray_tree(bins),
            **{k: _to_base_ndarray_tree(v) for k, v in kwargs.items()},
        )
    return result  # type: ignore[return-value]


attach_docstring(
    histogram,
    _np.histogram,
    "counted_custom",
    "n * ceil(log2(bins)) FLOPs when bins is int or edges; n*(2+estimator+ceil(log2(resolved bins))) when bins is a string estimator",
)
histogram.__signature__ = _inspect.signature(_np.histogram)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def histogram2d(
    x: ArrayLike,
    y: ArrayLike,
    bins: Any = 10,
    **kwargs: Any,
) -> tuple[FlopscopeArray, FlopscopeArray, FlopscopeArray]:
    budget = require_budget()
    x = _np.asarray(x)
    y = _np.asarray(y)
    n = x.size

    # Determine bx, by
    if isinstance(bins, _builtins.int):
        cost = _builtins.max(n * (_ceil_log2(bins) + _ceil_log2(bins)), 1)
    elif (
        isinstance(bins, (_builtins.list, tuple))
        and _builtins.len(bins) == 2
        and isinstance(bins[0], _builtins.int)
        and isinstance(bins[1], _builtins.int)
    ):
        bx, by = bins[0], bins[1]
        cost = _builtins.max(n * (_ceil_log2(bx) + _ceil_log2(by)), 1)
    elif isinstance(bins, (_builtins.list, tuple)) and _builtins.len(bins) == 2:
        b0 = _np.asarray(bins[0])
        b1 = _np.asarray(bins[1])
        if b0.ndim >= 1 and b1.ndim >= 1:
            cost = _builtins.max(
                n * (_ceil_log2(_builtins.len(b0)) + _ceil_log2(_builtins.len(b1))), 1
            )
        else:
            cost = _builtins.max(n, 1)
    else:
        cost = _builtins.max(n, 1)

    # Array-valued bin edges are genuine operands numpy promotes across; fold
    # their dtype in so complex/fp64 edges don't bypass the factor. Integer
    # bins are counts, not data -> skipped.
    _bin_dtypes = tuple(
        _np.asarray(b).dtype
        for b in (bins if isinstance(bins, (_builtins.list, tuple)) else ())
        if not _np.isscalar(b)
    )
    with budget.deduct(
        "histogram2d",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape, y.shape),
        dtypes=(x.dtype, y.dtype) + _bin_dtypes,
    ):
        result = _call_numpy(
            _np.histogram2d,
            x,
            y,
            bins=_to_base_ndarray_tree(bins),
            **{k: _to_base_ndarray_tree(v) for k, v in kwargs.items()},
        )
    return result  # type: ignore[return-value]


attach_docstring(
    histogram2d,
    _np.histogram2d,
    "counted_custom",
    "n * (ceil(log2(bx)) + ceil(log2(by))) FLOPs when bins is int pair; n FLOPs otherwise",
)
histogram2d.__signature__ = _inspect.signature(_np.histogram2d)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def histogramdd(
    sample: ArrayLike,
    bins: Any = 10,
    **kwargs: Any,
) -> tuple[FlopscopeArray, list[FlopscopeArray]]:
    budget = require_budget()
    sample = _np.asarray(sample)
    # sample shape: (n, d) or (n,) for 1-d
    if sample.ndim == 1:
        n = sample.shape[0]
        d = 1
    else:
        n, d = sample.shape[0], sample.shape[1]

    if isinstance(bins, _builtins.int):
        cost = _builtins.max(n * d * _ceil_log2(bins), 1)
    elif isinstance(bins, (_builtins.list, tuple)):
        total_log = 0
        for b in bins:
            if isinstance(b, _builtins.int):
                total_log += _ceil_log2(b)
            else:
                b_arr = _np.asarray(b)
                if b_arr.ndim >= 1 and b_arr.size > 0:
                    total_log += _ceil_log2(_builtins.len(b_arr))
                else:
                    total_log += 1
        cost = _builtins.max(n * total_log, 1)
    else:
        cost = _builtins.max(n, 1)

    # Array-valued bin edges are genuine operands numpy promotes across; fold
    # their dtype in so complex/fp64 edges don't bypass the factor. Integer
    # bins are counts, not data -> skipped.
    _bin_dtypes = tuple(
        _np.asarray(b).dtype
        for b in (bins if isinstance(bins, (_builtins.list, tuple)) else ())
        if not _np.isscalar(b)
    )
    with budget.deduct(
        "histogramdd",
        flop_cost=cost,
        subscripts=None,
        shapes=(sample.shape,),
        dtypes=(sample.dtype,) + _bin_dtypes,
    ):
        result = _call_numpy(
            _np.histogramdd,
            sample,
            bins=_to_base_ndarray_tree(bins),
            **{k: _to_base_ndarray_tree(v) for k, v in kwargs.items()},
        )
    return result  # type: ignore[return-value]


attach_docstring(
    histogramdd,
    _np.histogramdd,
    "counted_custom",
    "n * d * ceil(log2(bins)) FLOPs when bins is int; n FLOPs otherwise",
)
histogramdd.__signature__ = _inspect.signature(_np.histogramdd)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def histogram_bin_edges(
    a: ArrayLike,
    bins: int | Sequence[int] | str = 10,
    **kwargs: Any,
) -> FlopscopeArray:
    budget = require_budget()
    a = _np.asarray(a)
    n = a.size
    if isinstance(bins, _builtins.str):
        est = (
            _builtins.max(100, _math.isqrt(n))
            if bins == "stone"
            else _HIST_ESTIMATOR_COST.get(bins, 1)
        )
        cost = _builtins.max(n * (2 + est), 1)
    else:
        cost = _builtins.max(n, 1)
    with budget.deduct(
        "histogram_bin_edges",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(a.dtype,),
    ):
        result = _call_numpy(
            _np.histogram_bin_edges,
            a,
            bins=_to_base_ndarray_tree(bins),
            **{k: _to_base_ndarray_tree(v) for k, v in kwargs.items()},
        )
    return result  # type: ignore[return-value]


attach_docstring(
    histogram_bin_edges,
    _np.histogram_bin_edges,
    "counted_custom",
    "numel(a) FLOPs (int/edges); n*(2+estimator) FLOPs (string estimator)",
)
histogram_bin_edges.__signature__ = _inspect.signature(_np.histogram_bin_edges)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def bincount(x: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    budget = require_budget()
    x = _np.asarray(x)
    cost = _builtins.max(x.size, 1)
    with budget.deduct(
        "bincount",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape,),
        dtypes=(x.dtype,),
    ):
        result = _call_numpy(
            _np.bincount,
            x,
            **{k: _to_base_ndarray_tree(v) for k, v in kwargs.items()},
        )
    return result  # type: ignore[return-value]


attach_docstring(bincount, _np.bincount, "counted_custom", "numel(x) FLOPs")
try:
    bincount.__signature__ = _inspect.signature(_np.bincount)  # pyright: ignore[reportFunctionMemberAccess]
except (ValueError, TypeError):
    pass


# ---------------------------------------------------------------------------
# Array generation with arithmetic
# ---------------------------------------------------------------------------


@_counted_wrapper
def logspace(
    start: Any,
    stop: Any,
    num: int = 50,
    **kwargs: Any,
) -> FlopscopeArray:
    budget = require_budget()
    _dtype = kwargs.get("dtype")
    _billing_dtype = (
        _np.dtype(_dtype) if _dtype is not None else _np.result_type(start, stop)
    )
    with budget.deduct_after(
        "logspace", subscripts=None, shapes=(), dtypes=(_billing_dtype,)
    ) as _op:
        result = _call_numpy(
            _np.logspace,
            _to_base_ndarray(start) if hasattr(start, "__array__") else start,
            _to_base_ndarray(stop) if hasattr(stop, "__array__") else stop,
            num=num,
            **kwargs,
        )
        _op.set_cost(result.size if hasattr(result, "size") else 1)
    return result  # type: ignore[return-value]


attach_docstring(logspace, _np.logspace, "counted_custom", "num FLOPs")
logspace.__signature__ = _inspect.signature(_np.logspace)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def geomspace(
    start: Any,
    stop: Any,
    num: int = 50,
    **kwargs: Any,
) -> FlopscopeArray:
    budget = require_budget()
    _dtype = kwargs.get("dtype")
    _billing_dtype = (
        _np.dtype(_dtype) if _dtype is not None else _np.result_type(start, stop)
    )
    with budget.deduct_after(
        "geomspace", subscripts=None, shapes=(), dtypes=(_billing_dtype,)
    ) as _op:
        result = _call_numpy(
            _np.geomspace,
            _to_base_ndarray(start) if hasattr(start, "__array__") else start,
            _to_base_ndarray(stop) if hasattr(stop, "__array__") else stop,
            num=num,
            **kwargs,
        )
        _op.set_cost(result.size if hasattr(result, "size") else 1)
    return result  # type: ignore[return-value]


attach_docstring(geomspace, _np.geomspace, "counted_custom", "num FLOPs")
geomspace.__signature__ = _inspect.signature(_np.geomspace)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def vander(
    x: ArrayLike,
    N: int | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    budget = require_budget()
    x = _np.asarray(x)
    n = _builtins.len(x)
    if N is None:
        N = n
    cost = _builtins.max(n * (N - 2), 1)
    with budget.deduct(
        "vander", flop_cost=cost, subscripts=None, shapes=(x.shape,), dtypes=(x.dtype,)
    ):
        result = _call_numpy(_np.vander, x, N=N, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(vander, _np.vander, "counted_custom", "len(x) * (N-2) FLOPs")
vander.__signature__ = _inspect.signature(_np.vander)  # pyright: ignore[reportFunctionMemberAccess]

# ---------------------------------------------------------------------------
# Apply & piecewise (formerly blacklisted)
# ---------------------------------------------------------------------------


@_counted_wrapper
def apply_along_axis(
    func1d: Any,
    axis: int,
    arr: ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of np.apply_along_axis. Cost: numel(output)."""
    _warn_remote_callback("apply_along_axis")
    budget = require_budget()
    if not isinstance(arr, _np.ndarray):
        arr = _np.asarray(arr)
    result = _call_user_code(
        budget,
        _np.apply_along_axis,
        func1d,
        axis,
        _to_base_ndarray(arr),
        *args,
        **kwargs,
    )
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "apply_along_axis",
        flop_cost=cost,
        subscripts=None,
        shapes=(arr.shape,),
        dtypes=(arr.dtype,),
    ):
        pass
    return result  # type: ignore[return-value]


attach_docstring(
    apply_along_axis,
    _np.apply_along_axis,
    "counted_custom",
    "numel(output) FLOPs",
)


@_counted_wrapper
def apply_over_axes(
    func: Any,
    a: ArrayLike,
    axes: int | Sequence[int],
) -> FlopscopeArray:
    """Counted version of np.apply_over_axes. Cost: numel(output)."""
    _warn_remote_callback("apply_over_axes")
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    result = _call_user_code(
        budget, _np.apply_over_axes, func, _to_base_ndarray(a), axes
    )
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "apply_over_axes",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(a.dtype,),
    ):
        pass
    return result  # type: ignore[return-value]


attach_docstring(
    apply_over_axes,
    _np.apply_over_axes,
    "counted_custom",
    "numel(output) FLOPs",
)


@_counted_wrapper
def piecewise(
    x: ArrayLike,
    condlist: Any,
    funclist: Any,
    *args: Any,
    **kw: Any,
) -> FlopscopeArray:
    """Counted version of np.piecewise. Cost: numel(input)."""
    _warn_remote_callback("piecewise")
    budget = require_budget()
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    result = _call_user_code(
        budget,
        _np.piecewise,
        _to_base_ndarray(x),
        _to_base_ndarray_tree(condlist),
        funclist,
        *args,
        **kw,
    )
    cost = x.size
    with budget.deduct(
        "piecewise",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape,),
        dtypes=(x.dtype,),
    ):
        pass
    return result  # type: ignore[return-value]


attach_docstring(
    piecewise,
    _np.piecewise,
    "counted_custom",
    "numel(input) FLOPs",
)


import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
