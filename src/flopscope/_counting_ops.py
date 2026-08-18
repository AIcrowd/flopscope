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
from flopscope._dtype_billing import (
    heavier_billing_dtype,
    mean_compute_dtype,
    reduction_billing_dtype,
    sum_accumulator_dtype,
)
from flopscope._flops import _ceil_log2
from flopscope._ndarray import FlopscopeArray, _to_base_ndarray, _to_base_ndarray_tree
from flopscope._validation import _normalize_out, require_budget
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
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "trace")
    a = _np.asarray(a)
    # Normalise negative axes
    ndim = a.ndim
    ax1 = axis1 % ndim if ndim > 0 else 0
    ax2 = axis2 % ndim if ndim > 0 else 0
    if ndim >= 2 and a.shape[ax1] > 0 and a.shape[ax2] > 0:
        n1, n2 = a.shape[ax1], a.shape[ax2]
        # ``offset`` shifts the diagonal, shortening it exactly the way
        # numpy.linalg.trace already bills: a positive offset moves toward
        # ax2 (columns), a negative offset toward ax1 (rows). Ignoring it
        # over-bills every off-diagonal trace by the full main-diagonal
        # length (e.g. a 256x256 trace at offset=255 billed 256 instead of 1).
        k = _builtins.min(n1, n2)
        if offset > 0:
            k = _builtins.min(k, n2 - offset)
        elif offset < 0:
            k = _builtins.min(k, n1 + offset)
        k = _builtins.max(k, 1)
        # Number of independent trace evaluations = product of all other axes
        n_traces = a.size // (a.shape[ax1] * a.shape[ax2])
        cost = k * n_traces
    else:
        cost = 0
    out_stripped = _to_base_ndarray(out) if out is not None else None
    # numpy widens the trace accumulator like sum (trace(int32) runs int64).
    # An explicit dtype= bills as the accumulator numpy runs, in both
    # directions; out= widens it, and a narrower out only casts the final
    # store; the default never bills below the input rate. Mirrors the
    # reduction wrappers exactly.
    billing_dtypes: tuple = (
        reduction_billing_dtype(
            a.dtype,
            explicit_dtype=dtype,
            out_dtype=out_stripped.dtype if out_stripped is not None else None,
            default_dtype=sum_accumulator_dtype(a.dtype),
        ),
    )
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
    # np.histogram's counts output is always the platform default int
    # (bincount-style accumulation: int64 on 64-bit platforms) regardless of
    # a's dtype, while its bin-edges output follows the same mean-shaped
    # rule as linspace/mean (integer/bool a -> float64, float a preserved).
    # Both outputs must clear the billed rate, so the int floor joins the
    # edge-driven dtype(s) in the resolve tuple below (result_type, same
    # "append a sentinel dtype" pattern as cov/corrcoef/polyfit).
    _hist_int_floor = _np.dtype(_np.int_)
    if isinstance(bins, _builtins.int):
        cost = _builtins.max(n * _ceil_log2(bins), 1)
    elif isinstance(bins, _builtins.str):
        # Deferred cost: resolve nbins from returned edges, then charge
        # 2n (min/max scan) + estimator_cost*n + n*ceil_log2(nbins_resolved)
        with budget.deduct_after(
            "histogram",
            subscripts=None,
            shapes=(a.shape,),
            dtypes=(mean_compute_dtype(a.dtype), _hist_int_floor),
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
        (mean_compute_dtype(a.dtype), _hist_int_floor)
        if _np.isscalar(bins)
        else (mean_compute_dtype(a.dtype), _np.asarray(bins).dtype, _hist_int_floor)
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
    # bins are counts, not data -> skipped. histogram2d delegates to
    # histogramdd's weighted-accumulation counting, whose counts output is
    # always float64 regardless of x/y's dtype (verified: histogram2d(int32,
    # int32) counts -> float64, histogram2d(float32, float32) counts ->
    # float64 too), while the edges follow the mean-shaped rule (integer/
    # bool -> float64, float preserved) -- so both x and y route through
    # mean_compute_dtype and a float64 sentinel joins the resolve.
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
        dtypes=(
            (mean_compute_dtype(x.dtype), mean_compute_dtype(y.dtype))
            + _bin_dtypes
            + (_np.dtype(_np.float64),)
        ),
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
    # bins are counts, not data -> skipped. Same histogram2d/mean-shaped
    # rule: counts always float64, edges follow mean_compute_dtype(sample).
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
        dtypes=(mean_compute_dtype(sample.dtype),)
        + _bin_dtypes
        + (_np.dtype(_np.float64),),
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
    # Edges derived from a's own range follow the mean-shaped rule
    # (integer/bool a -> float64, float a preserved -- verified:
    # histogram_bin_edges(int32) -> float64, (float32) -> float32). An
    # array-valued ``bins`` instead echoes straight back unchanged (numpy
    # returns the same array), so its own dtype must join the resolve too or
    # a wider explicit edges array would be undercounted.
    _hbe_dtypes: tuple = (mean_compute_dtype(a.dtype),)
    if not _np.isscalar(bins) and not isinstance(bins, _builtins.str):
        _hbe_dtypes += (_np.asarray(bins).dtype,)
    with budget.deduct(
        "histogram_bin_edges",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=_hbe_dtypes,
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
    # np.bincount's output is always the platform default int (int64 on
    # 64-bit platforms), regardless of x's own integer width -- verified
    # across int8..64 AND uint8..64 (even bincount(uint64) -> int64, not
    # uint64). This is a fixed override, not a promotion (result_type(uint64,
    # int64) would incorrectly jump to float64), so heavier_billing_dtype
    # (max-by-rate) is used instead of folding int64 into the resolve tuple.
    with budget.deduct(
        "bincount",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape,),
        dtypes=(heavier_billing_dtype(x.dtype, _np.dtype(_np.int_)),),
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
    with budget.deduct_after("logspace", subscripts=None, shapes=(), dtypes=()) as _op:
        result = _call_numpy(
            _np.logspace,
            _to_base_ndarray(start) if hasattr(start, "__array__") else start,
            _to_base_ndarray(stop) if hasattr(stop, "__array__") else stop,
            num=num,
            **kwargs,
        )
        # Bill the dtype numpy actually PRODUCED: with ``dtype`` omitted,
        # integer endpoints yield float64 samples, so resolving from the
        # inputs would bill half the float64 rate (same fix as linspace).
        _op.set_dtypes((result.dtype,) if hasattr(result, "dtype") else ())
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
    with budget.deduct_after("geomspace", subscripts=None, shapes=(), dtypes=()) as _op:
        result = _call_numpy(
            _np.geomspace,
            _to_base_ndarray(start) if hasattr(start, "__array__") else start,
            _to_base_ndarray(stop) if hasattr(stop, "__array__") else stop,
            num=num,
            **kwargs,
        )
        # Bill the dtype numpy actually PRODUCED: geomspace promotes to at
        # least float64 (float32 endpoints included), so input-derived
        # resolution under-billed every omitted-dtype call (same fix as
        # linspace/logspace).
        _op.set_dtypes((result.dtype,) if hasattr(result, "dtype") else ())
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
    # np.vander allocates its output at ``promote_types(x.dtype, int)`` (its
    # own source: ``dtype=promote_types(x.dtype, int)``) -- ``int`` there is
    # numpy's platform default int (int64 on 64-bit platforms), so every
    # narrower integer/bool/uint promotes to it (verified: vander(int8..32)
    # AND vander(uint8..32) -> int64, matching promote_types exactly even
    # though uint's own same-kind default would be uint64), while float32/
    # complex64 still widen to float64/complex128 (promote_types(f32, i8)
    # can't stay float32). result_type(x.dtype, platform_int) reproduces
    # promote_types(x.dtype, int) exactly for every dtype kind.
    billing_dtypes = (x.dtype, _np.dtype(_np.int_))
    with budget.deduct(
        "vander",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape,),
        dtypes=billing_dtypes,
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
    # func1d is arbitrary user code (hence the RemoteCallbackWarning above):
    # numpy has no promotion rule to predict what it returns, so bill
    # whichever is pricier of the input array's dtype and the callback's
    # OWN demonstrated result dtype (already computed above) -- the same
    # "max of operand rates" pattern as astype, not a result_type combine
    # (there is no real promotion relationship between the two here).
    billing_dtypes: tuple = (arr.dtype,)
    if isinstance(result, _np.ndarray):
        billing_dtypes = (heavier_billing_dtype(arr.dtype, result.dtype),)
    with budget.deduct(
        "apply_along_axis",
        flop_cost=cost,
        subscripts=None,
        shapes=(arr.shape,),
        dtypes=billing_dtypes,
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
    # Same reasoning as apply_along_axis: func is arbitrary user code, so
    # bill whichever is pricier of the input dtype and the callback's own
    # demonstrated result dtype.
    billing_dtypes: tuple = (a.dtype,)
    if isinstance(result, _np.ndarray):
        billing_dtypes = (heavier_billing_dtype(a.dtype, result.dtype),)
    with budget.deduct(
        "apply_over_axes",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
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
    """Counted version of np.piecewise. Cost: numel(input) * len(condlist)."""
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
    # Count conditions exactly the way numpy's piecewise promotes condlist
    # (numpy.lib._function_base_impl.piecewise): a scalar condlist, or one
    # whose first element is neither a list nor an ndarray while x is not
    # 0-d, is a SINGLE condition (numpy wraps it as [condlist]); everything
    # else -- a list/tuple of per-piece conditions, or a 2-D+ bool ndarray
    # whose ROWS are independent conditions -- counts len(condlist). Each
    # condition is its own scan over x, so the bill scales with that count;
    # an isinstance(list/tuple) test alone would let a stacked 2-D condlist
    # bill as one condition. numpy has already accepted condlist above, so
    # condlist[0] here cannot raise on inputs numpy itself rejects.
    if _np.isscalar(condlist) or (
        not isinstance(condlist[0], (list, _np.ndarray)) and x.ndim != 0
    ):
        n_conds = 1
    else:
        n_conds = len(condlist)
    cost = x.size * max(n_conds, 1)
    # Same reasoning as apply_along_axis: funclist entries are arbitrary
    # user code, so bill whichever is pricier of the input dtype and the
    # callback's own demonstrated result dtype.
    billing_dtypes: tuple = (x.dtype,)
    if isinstance(result, _np.ndarray):
        billing_dtypes = (heavier_billing_dtype(x.dtype, result.dtype),)
    with budget.deduct(
        "piecewise",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape,),
        dtypes=billing_dtypes,
    ):
        pass
    return result  # type: ignore[return-value]


attach_docstring(
    piecewise,
    _np.piecewise,
    "counted_custom",
    "numel(input) * len(condlist) FLOPs",
)


import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
