"""Counted wrappers for sorting, search, and set operations."""

from __future__ import annotations

import inspect as _inspect
from collections.abc import Sequence
from typing import Any

import numpy as _np
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._flops import search_cost, sort_cost
from flopscope._ndarray import FlopscopeArray, _to_base_ndarray, _to_base_ndarray_tree
from flopscope._validation import require_budget
from flopscope.errors import UnsupportedFunctionError

# Numpy 2.3+ relaxed the sort guarantee for string / complex unique;
# this module shims the guarantee back for flopscope callers.
_NUMPY_GE_2_3 = tuple(int(x) for x in _np.__version__.split(".")[:2]) >= (2, 3)

# dtype kinds where numpy 2.3+ may drop the sort guarantee.
# Values are numpy dtype.kind codes:
#   U = unicode string, S = bytes string, O = object,
#   c = complex float (both complex64 and complex128).
_UNSORTED_IN_NP_2_3 = frozenset("USOc")


def _sort_cost_nd(a: _np.ndarray, axis: int) -> int:
    """Total sort cost for an n-d array sorted along *axis*.

    Total cost = num_slices * sort_cost(n) where
    num_slices = numel(a) / a.shape[axis].
    """
    n = a.shape[axis]
    numel = 1
    for d in a.shape:
        numel *= d
    num_slices = numel // n if n > 0 else 1
    return max(num_slices * sort_cost(n), 1)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


@_counted_wrapper
def sort(
    a: ArrayLike,
    axis: int | None = -1,
    kind: str | None = None,
    order: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of ``numpy.sort``. Cost: n*ceil(log2(n)) FLOPs per slice."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim == 0:
        cost = 1
    elif axis is None:
        cost = sort_cost(a.size)
    else:
        ax = axis % a.ndim
        cost = _sort_cost_nd(a, ax)
    with budget.deduct(
        "sort", flop_cost=cost, subscripts=None, shapes=(a.shape,), dtypes=(a.dtype,)
    ):
        result = _call_numpy(
            _np.sort, _to_base_ndarray(a), axis=axis, kind=kind, order=order, **kwargs
        )
    return result  # type: ignore[return-value]


attach_docstring(sort, _np.sort, "counted_custom", "n*ceil(log2(n)) FLOPs per slice")
sort.__signature__ = _inspect.signature(_np.sort)  # type: ignore[attr-defined]


@_counted_wrapper
def argsort(
    a: ArrayLike,
    axis: int | None = -1,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of ``numpy.argsort``. Cost: n*ceil(log2(n)) FLOPs per slice."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim == 0:
        cost = 1
    elif axis is None:
        cost = sort_cost(a.size)
    else:
        ax = axis % a.ndim
        cost = _sort_cost_nd(a, ax)
    with budget.deduct(
        "argsort", flop_cost=cost, subscripts=None, shapes=(a.shape,), dtypes=(a.dtype,)
    ):
        result = _call_numpy(_np.argsort, _to_base_ndarray(a), axis=axis, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(
    argsort, _np.argsort, "counted_custom", "n*ceil(log2(n)) FLOPs per slice"
)
argsort.__signature__ = _inspect.signature(_np.argsort)  # type: ignore[attr-defined]


@_counted_wrapper
def lexsort(keys: Sequence[ArrayLike], axis: int = -1) -> FlopscopeArray:
    """Counted version of ``numpy.lexsort``.

    Cost: k * num_slices * sort_cost(n) where k = len(keys),
    n = keys[0].shape[axis], and num_slices = keys[0].size // n
    (number of independent sort passes along axis).  For 1-D keys
    num_slices = 1, so the cost is unchanged vs. the previous formula.
    """
    budget = require_budget()
    # keys is a sequence of arrays; convert to list for inspection
    keys_list = list(keys)
    k = len(keys_list)
    if k == 0:
        cost = 1
    else:
        # numpy.lexsort uses the last key as primary; length is shape along axis
        first = _np.asarray(keys_list[0])
        n = first.shape[axis] if first.ndim > 0 else 1
        num_slices = (first.size // n) if (first.ndim > 0 and n > 0) else 1
        cost = max(k * num_slices * sort_cost(n), 1)
    shapes = tuple(_np.asarray(key).shape for key in keys_list)
    dtypes = tuple(_np.asarray(key).dtype for key in keys_list)
    with budget.deduct(
        "lexsort", flop_cost=cost, subscripts=None, shapes=shapes, dtypes=dtypes
    ):
        result = _call_numpy(_np.lexsort, _to_base_ndarray_tree(keys_list), axis=axis)
    return result


attach_docstring(
    lexsort,
    _np.lexsort,
    "counted_custom",
    "k keys × num_slices × n·ceil(log2 n) FLOPs",
)


@_counted_wrapper
def partition(
    a: ArrayLike,
    kth: int | Sequence[int],
    axis: int | None = -1,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of ``numpy.partition``. Cost: n * len(kth) per slice."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    # kth can be int, sequence of ints, or 0-d ndarray; np.size handles all forms
    kth_count = int(_np.size(kth))
    if a.ndim == 0:
        cost = 1
    else:
        ax = axis if axis is not None else -1
        ax = ax % a.ndim
        n = a.shape[ax]
        numel = 1
        for d in a.shape:
            numel *= d
        num_slices = numel // n if n > 0 else 1
        cost = max(num_slices * n * kth_count, 1)
    with budget.deduct(
        "partition",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(a.dtype,),
    ):
        result = _call_numpy(
            _np.partition, _to_base_ndarray(a), kth, axis=axis, **kwargs
        )
    return result  # type: ignore[return-value]


attach_docstring(
    partition, _np.partition, "counted_custom", "n FLOPs per slice (linear quickselect)"
)
partition.__signature__ = _inspect.signature(_np.partition)  # type: ignore[attr-defined]


@_counted_wrapper
def argpartition(
    a: ArrayLike,
    kth: int | Sequence[int],
    axis: int | None = -1,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of ``numpy.argpartition``. Cost: n * len(kth) per slice."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    # kth can be int, sequence of ints, or 0-d ndarray; np.size handles all forms
    kth_count = int(_np.size(kth))
    if a.ndim == 0:
        cost = 1
    else:
        ax = axis if axis is not None else -1
        ax = ax % a.ndim
        n = a.shape[ax]
        numel = 1
        for d in a.shape:
            numel *= d
        num_slices = numel // n if n > 0 else 1
        cost = max(num_slices * n * kth_count, 1)
    with budget.deduct(
        "argpartition",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(a.dtype,),
    ):
        result = _call_numpy(
            _np.argpartition, _to_base_ndarray(a), kth, axis=axis, **kwargs
        )
    return result  # type: ignore[return-value]


attach_docstring(
    argpartition,
    _np.argpartition,
    "counted_custom",
    "n FLOPs per slice (linear quickselect)",
)
argpartition.__signature__ = _inspect.signature(_np.argpartition)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@_counted_wrapper
def searchsorted(
    a: ArrayLike,
    v: ArrayLike,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of ``numpy.searchsorted``.

    Cost: m * ceil(log2(n)) where m = numel(v) and n = len(a).
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    v_arr = _np.asarray(v)
    n = a.shape[0] if a.ndim > 0 else 1
    m = max(v_arr.size, 1)
    cost = search_cost(m, n)
    with budget.deduct(
        "searchsorted",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, v_arr.shape),
        dtypes=(a.dtype, v_arr.dtype),
    ):
        result = _call_numpy(
            _np.searchsorted, _to_base_ndarray(a), _to_base_ndarray(v), **kwargs
        )
    return result  # type: ignore[return-value]


attach_docstring(
    searchsorted,
    _np.searchsorted,
    "counted_custom",
    "m*ceil(log2(n)) FLOPs (m queries into sorted array of size n)",
)
searchsorted.__signature__ = _inspect.signature(_np.searchsorted)  # type: ignore[attr-defined]


@_counted_wrapper
def digitize(
    x: ArrayLike,
    bins: ArrayLike,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of ``numpy.digitize``.

    Cost: n * ceil(log2(len(bins))) where n = numel(x).
    """
    budget = require_budget()
    x_arr = _np.asarray(x)
    bins_arr = _np.asarray(bins)
    n = max(x_arr.size, 1)
    cost = search_cost(n, max(len(bins_arr), 1))
    with budget.deduct(
        "digitize",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape, bins_arr.shape),
        dtypes=(x_arr.dtype, bins_arr.dtype),
    ):
        result = _call_numpy(
            _np.digitize, _to_base_ndarray(x), _to_base_ndarray(bins), **kwargs
        )  # type: ignore[arg-type]
    return result  # type: ignore[return-value]


attach_docstring(
    digitize,
    _np.digitize,
    "counted_custom",
    "n*ceil(log2(len(bins))) FLOPs",
)
digitize.__signature__ = _inspect.signature(_np.digitize)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------


def _unique_cost(ar, axis=None):
    """Compute sort-based cost for uniqueness.

    With ``axis=None`` (flat): sort_cost(ar.size).
    With ``axis=k``: num_slices * sort_cost(R) where R = shape[axis] (lexicographic
    row sort — each row of width W is compared as a unit, R rows, W words per key).
    This matches the Sort-and-select family rule declared in the sort-and-select
    section of docs/reference/cost-model.md.
    """
    if not isinstance(ar, _np.ndarray):
        ar = _np.asarray(ar)
    if axis is None or ar.ndim == 0:
        return sort_cost(max(ar.size, 1))
    return _sort_cost_nd(ar, int(axis) % ar.ndim)


@_counted_wrapper
def unique(ar: ArrayLike, **kwargs: Any) -> FlopscopeArray | tuple[FlopscopeArray, ...]:
    """Counted version of ``numpy.unique``. Cost: n*ceil(log2(n)) FLOPs.

    On numpy 2.3+ the sort guarantee is relaxed for string and complex dtypes.
    This wrapper re-sorts the values (only when no auxiliary-return kwargs are
    requested) to preserve pre-2.3 semantics.
    """
    budget = require_budget()
    ar_arr = _np.asarray(ar)
    cost = _unique_cost(ar_arr, axis=kwargs.get("axis", None))
    # The compat re-sort below only applies to the default signature (no
    # auxiliary-return kwargs); decide before opening the deduct block.
    _returns_tuple = any(
        kwargs.get(k, False)
        for k in ("return_index", "return_inverse", "return_counts")
    )
    with budget.deduct(
        "unique",
        flop_cost=cost,
        subscripts=None,
        shapes=(ar_arr.shape,),
        dtypes=(ar_arr.dtype,),
    ):
        result = _call_numpy(_np.unique, ar_arr, **kwargs)
        # Shim: restore the sort guarantee for string / complex dtypes on numpy
        # 2.3+. Done inside this deduct (via _call_numpy) so the re-sort's wall
        # time bills to the unique op's backend — without charging extra FLOPs
        # or recording a second op.
        if (
            _NUMPY_GE_2_3
            and not _returns_tuple
            and ar_arr.dtype.kind in _UNSORTED_IN_NP_2_3
        ):
            result = _call_numpy(_np.sort, _to_base_ndarray(result))
    return result  # type: ignore[return-value]


attach_docstring(
    unique,
    _np.unique,
    "counted_custom",
    "n*ceil(log2(n)) FLOPs; axis-aware: num_slices x n x ceil(log2 n), n = axis length",
)
unique.__signature__ = _inspect.signature(_np.unique)  # type: ignore[attr-defined]


@_counted_wrapper
def unique_all(x: ArrayLike, /) -> Any:
    """Counted version of ``numpy.unique_all``. Cost: n*ceil(log2(n)) FLOPs."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = _unique_cost(x_arr)
    with budget.deduct(
        "unique_all",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=(x_arr.dtype,),
    ):
        result = _call_numpy(_np.unique_all, x_arr)
    return result


attach_docstring(unique_all, _np.unique_all, "counted_custom", "n*ceil(log2(n)) FLOPs")


@_counted_wrapper
def unique_counts(x: ArrayLike, /) -> Any:
    """Counted version of ``numpy.unique_counts``. Cost: n*ceil(log2(n)) FLOPs."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = _unique_cost(x_arr)
    with budget.deduct(
        "unique_counts",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=(x_arr.dtype,),
    ):
        result = _call_numpy(_np.unique_counts, x_arr)
    return result


attach_docstring(
    unique_counts, _np.unique_counts, "counted_custom", "n*ceil(log2(n)) FLOPs"
)


@_counted_wrapper
def unique_inverse(x: ArrayLike, /) -> Any:
    """Counted version of ``numpy.unique_inverse``. Cost: n*ceil(log2(n)) FLOPs."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = _unique_cost(x_arr)
    with budget.deduct(
        "unique_inverse",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=(x_arr.dtype,),
    ):
        result = _call_numpy(_np.unique_inverse, x_arr)
    return result


attach_docstring(
    unique_inverse, _np.unique_inverse, "counted_custom", "n*ceil(log2(n)) FLOPs"
)


@_counted_wrapper
def unique_values(x: ArrayLike, /) -> FlopscopeArray:
    """Counted version of ``numpy.unique_values``. Cost: n*ceil(log2(n)) FLOPs."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = _unique_cost(x_arr)
    with budget.deduct(
        "unique_values",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=(x_arr.dtype,),
    ):
        result = _call_numpy(_np.unique_values, x_arr)
    return result  # type: ignore[return-value]


attach_docstring(
    unique_values, _np.unique_values, "counted_custom", "n*ceil(log2(n)) FLOPs"
)


# ---------------------------------------------------------------------------
# Set operations  (cost = (n+m) * ceil(log2(n+m)))
# ---------------------------------------------------------------------------


def _set_cost(ar1, ar2):
    """Compute cost for set operations on two arrays."""
    n = _np.asarray(ar1).size
    m = _np.asarray(ar2).size
    total = max(n + m, 1)
    return sort_cost(total)


def _membership_cost(a1: _np.ndarray, a2: _np.ndarray) -> int:
    """Cost for isin/in1d: replicates numpy 2.x _in1d branch selection.

    Returns max(sort_cost(n+m), 2*n*m) when numpy's masked-loop path triggers
    (m < 10*n**0.145 with non-integer dtypes, or object dtype), else sort_cost(n+m).
    The loop path runs n*m comparisons + n*m boolean accumulates = 2*n*m FLOPs.
    """
    n = a1.size
    m = a2.size
    base = sort_cost(max(n + m, 1))
    table_eligible = a1.dtype.kind in "uib" and a2.dtype.kind in "uib"
    contains_object = a1.dtype.hasobject or a2.dtype.hasobject
    if contains_object or (not table_eligible and m < 10 * max(n, 1) ** 0.145):
        return max(base, 2 * n * m)
    return base


if hasattr(_np, "in1d"):

    @_counted_wrapper
    def in1d(ar1: ArrayLike, ar2: ArrayLike, **kwargs: Any) -> FlopscopeArray:  # pyright: ignore[reportRedeclaration]
        """Counted version of ``numpy.in1d``.

        Cost: (n+m)*ceil(log2(n+m)) FLOPs (sort path) or max(sort_cost(n+m), 2*n*m)
        when numpy's masked-loop path triggers.
        """
        budget = require_budget()
        a1 = _np.asarray(ar1)
        a2 = _np.asarray(ar2)
        cost = _membership_cost(a1, a2)
        with budget.deduct(
            "in1d",
            flop_cost=cost,
            subscripts=None,
            shapes=(a1.shape, a2.shape),
            dtypes=(a1.dtype, a2.dtype),
        ):
            result = _call_numpy(
                _np.in1d, _to_base_ndarray(ar1), _to_base_ndarray(ar2), **kwargs
            )
        return result  # type: ignore[return-value]

    attach_docstring(
        in1d,
        _np.in1d,
        "counted_custom",
        "(n+m)*ceil(log2(n+m)) FLOPs; 2*n*m when numpy's masked-loop path triggers",
    )
    in1d.__signature__ = _inspect.signature(_np.in1d)  # type: ignore[attr-defined]

else:

    def in1d(*args: Any, **kwargs: Any) -> FlopscopeArray:
        raise UnsupportedFunctionError("in1d", max_version="2.4", replacement="isin")


@_counted_wrapper
def isin(
    element: ArrayLike,
    test_elements: ArrayLike,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of ``numpy.isin``.

    Cost: (n+m)*ceil(log2(n+m)) FLOPs (sort path) or max(sort_cost(n+m), 2*n*m)
    when numpy's masked-loop path triggers (m < 10*n**0.145 for non-integer dtypes,
    or object dtype).
    """
    budget = require_budget()
    el = _np.asarray(element)
    te = _np.asarray(test_elements)
    cost = _membership_cost(el, te)
    with budget.deduct(
        "isin",
        flop_cost=cost,
        subscripts=None,
        shapes=(el.shape, te.shape),
        dtypes=(el.dtype, te.dtype),
    ):
        result = _call_numpy(
            _np.isin,
            _to_base_ndarray(element),
            _to_base_ndarray(test_elements),
            **kwargs,
        )
    return result  # type: ignore[return-value]


attach_docstring(
    isin,
    _np.isin,
    "counted_custom",
    "(n+m)*ceil(log2(n+m)) FLOPs; 2*n*m when numpy's masked-loop path triggers",
)
isin.__signature__ = _inspect.signature(_np.isin)  # type: ignore[attr-defined]


@_counted_wrapper
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    **kwargs: Any,
) -> FlopscopeArray | tuple[FlopscopeArray, ...]:
    """Counted version of ``numpy.intersect1d``.

    When ``assume_unique`` is falsy (the default), numpy calls ``unique()``
    on both inputs before the concat-sort, so the honest cost is
    ``sort_cost(n) + sort_cost(m) + sort_cost(n+m)``.
    When ``assume_unique=True`` only the final concat-sort is needed:
    ``sort_cost(n+m)``.
    """
    budget = require_budget()
    a1 = _np.asarray(ar1)
    a2 = _np.asarray(ar2)
    assume_unique = bool(kwargs.get("assume_unique", False))
    n = max(a1.size, 1)
    m = max(a2.size, 1)
    if assume_unique:
        cost = sort_cost(n + m)
    else:
        cost = sort_cost(n) + sort_cost(m) + sort_cost(n + m)
    with budget.deduct(
        "intersect1d",
        flop_cost=cost,
        subscripts=None,
        shapes=(a1.shape, a2.shape),
        dtypes=(a1.dtype, a2.dtype),
    ):
        result = _call_numpy(
            _np.intersect1d, _to_base_ndarray(ar1), _to_base_ndarray(ar2), **kwargs
        )
    return result  # type: ignore[return-value]


attach_docstring(
    intersect1d,
    _np.intersect1d,
    "counted_custom",
    "sort_cost(n)+sort_cost(m)+sort_cost(n+m) FLOPs (default); sort_cost(n+m) when assume_unique=True",
)
intersect1d.__signature__ = _inspect.signature(_np.intersect1d)  # type: ignore[attr-defined]


@_counted_wrapper
def union1d(ar1: ArrayLike, ar2: ArrayLike) -> FlopscopeArray:
    """Counted version of ``numpy.union1d``. Cost: (n+m)*ceil(log2(n+m)) FLOPs."""
    budget = require_budget()
    a1 = _np.asarray(ar1)
    a2 = _np.asarray(ar2)
    cost = _set_cost(a1, a2)
    with budget.deduct(
        "union1d",
        flop_cost=cost,
        subscripts=None,
        shapes=(a1.shape, a2.shape),
        dtypes=(a1.dtype, a2.dtype),
    ):
        result = _call_numpy(_np.union1d, _to_base_ndarray(ar1), _to_base_ndarray(ar2))
    return result  # type: ignore[return-value]


attach_docstring(union1d, _np.union1d, "counted_custom", "(n+m)*ceil(log2(n+m)) FLOPs")


@_counted_wrapper
def setdiff1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of ``numpy.setdiff1d``. Cost: (n+m)*ceil(log2(n+m)) FLOPs."""
    budget = require_budget()
    a1 = _np.asarray(ar1)
    a2 = _np.asarray(ar2)
    cost = _set_cost(a1, a2)
    with budget.deduct(
        "setdiff1d",
        flop_cost=cost,
        subscripts=None,
        shapes=(a1.shape, a2.shape),
        dtypes=(a1.dtype, a2.dtype),
    ):
        result = _call_numpy(
            _np.setdiff1d, _to_base_ndarray(ar1), _to_base_ndarray(ar2), **kwargs
        )
    return result  # type: ignore[return-value]


attach_docstring(
    setdiff1d, _np.setdiff1d, "counted_custom", "(n+m)*ceil(log2(n+m)) FLOPs"
)
setdiff1d.__signature__ = _inspect.signature(_np.setdiff1d)  # type: ignore[attr-defined]


@_counted_wrapper
def setxor1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of ``numpy.setxor1d``. Cost: (n+m)*ceil(log2(n+m)) FLOPs."""
    budget = require_budget()
    a1 = _np.asarray(ar1)
    a2 = _np.asarray(ar2)
    cost = _set_cost(a1, a2)
    with budget.deduct(
        "setxor1d",
        flop_cost=cost,
        subscripts=None,
        shapes=(a1.shape, a2.shape),
        dtypes=(a1.dtype, a2.dtype),
    ):
        result = _call_numpy(
            _np.setxor1d, _to_base_ndarray(ar1), _to_base_ndarray(ar2), **kwargs
        )
    return result  # type: ignore[return-value]


attach_docstring(
    setxor1d, _np.setxor1d, "counted_custom", "(n+m)*ceil(log2(n+m)) FLOPs"
)
setxor1d.__signature__ = _inspect.signature(_np.setxor1d)  # type: ignore[attr-defined]

import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
