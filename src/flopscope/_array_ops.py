"""NumPy array creation, manipulation, and indexing wrappers.

Wraps NumPy's array-creation, shape-manipulation, and indexing routines.
Per-op FLOP cost is set by the registry / weights table, NOT by this module:
many ops here are billed (e.g. ``arange``, ``linspace``, ``nonzero``, ``isnan``),
while data-movement and constant-init ops are free (weight 0). Free ops still
route through ``budget.deduct(..., flop_cost=0)`` so their time is accounted.
"""

from __future__ import annotations

import inspect as _inspect
import math as _math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as _np
from numpy.typing import ArrayLike, DTypeLike

from flopscope import _symmetry_transport as _st
from flopscope._budget import _call_numpy, _call_user_code, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._dtype_billing import heavier_billing_dtype as _heavier_billing_dtype
from flopscope._ndarray import (
    FlopscopeArray,
    _asplainflopscope,
    _to_base_ndarray,
    _to_base_ndarray_tree,
)
from flopscope._perm_group import SymmetryGroup
from flopscope._symmetric import SymmetricTensor
from flopscope._symmetry_utils import (
    broadcast_group,
    validate_symmetry_group,
    wrap_with_inferred_symmetry,
    wrap_with_symmetry,
    wrap_with_trusted_symmetry,
)
from flopscope._validation import require_budget
from flopscope.errors import (
    SymmetryError,
    UnsupportedFunctionError,
    _warn_remote_callback,
    _warn_symmetry_loss,
)


def _warn_if_symmetric(arr, op_name: str) -> None:
    """Emit SymmetryLossWarning if `arr` is a SymmetricTensor with a group."""
    if isinstance(arr, SymmetricTensor) and arr.symmetry is not None:
        _warn_symmetry_loss(
            lost_dims=[arr.symmetry.axes or tuple(range(arr.symmetry.degree))],
            reason=f"op '{op_name}' is not symmetry-aware",
        )


@lru_cache(maxsize=1024)
def _infer_constant_shape_symmetry(shape):
    if len(shape) < 2:
        return None

    blocks_by_extent: dict[int, list[int]] = {}
    for axis, extent in enumerate(shape):
        blocks_by_extent.setdefault(int(extent), []).append(axis)

    blocks = tuple(tuple(axes) for axes in blocks_by_extent.values() if len(axes) >= 2)
    if not blocks:
        return None
    if len(blocks) == 1:
        return SymmetryGroup.symmetric(axes=blocks[0])
    return SymmetryGroup.young(blocks=blocks)


def _wrap_constant_fill(result: _np.ndarray) -> FlopscopeArray:
    symmetry = _infer_constant_shape_symmetry(result.shape)
    if symmetry is None:
        return result  # type: ignore[return-value]
    return wrap_with_inferred_symmetry(result, symmetry)  # type: ignore[return-value]


def _compatible_symmetry_for_shape(symmetry, shape):
    """Return ``symmetry`` only when ``shape`` still supports it exactly."""
    if symmetry is None:
        return None
    try:
        validate_symmetry_group(symmetry, ndim=len(shape), shape=shape)
    except (SymmetryError, ValueError):
        return None
    return symmetry


def _infer_structural_constructor_symmetry(*, kind, N=None, M=None, k=0, v_ndim=None):
    if kind == "eye":
        if k == 0 and (M is None or M == N):
            return SymmetryGroup.symmetric(axes=(0, 1))
        return None
    if kind == "identity":
        return SymmetryGroup.symmetric(axes=(0, 1))
    if kind == "diag":
        if v_ndim == 1 and k == 0:
            return SymmetryGroup.symmetric(axes=(0, 1))
        return None
    if kind == "diagflat":
        if k == 0:
            return SymmetryGroup.symmetric(axes=(0, 1))
        return None
    return None


def _eye_diagonal_length(N: int, M: int | None, k: int) -> int:
    """Number of ones an ``eye(N, M, k)``/``identity(n)`` call writes.

    A fully off-diagonal ``k`` (the requested diagonal misses the array
    entirely) writes nothing -- floors at 0, not 1: the zero background is
    free, and there is no value-writing work to bill when the diagonal is
    empty.
    """
    m = N if M is None else M
    return max(0, min(N, m - k)) if k >= 0 else max(0, min(N + k, m))


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------


@_counted_wrapper
def array(
    object: ArrayLike,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Create an array. Cost: numel(output)."""
    budget = require_budget()
    # Pre-compute cost from input to keep numpy call inside the timer
    _probe = _np.asarray(object)
    cost = max(_probe.size, 1)
    with budget.deduct(
        "array",
        flop_cost=cost,
        subscripts=None,
        shapes=(_probe.shape,),
        dtypes=(_np.dtype(dtype) if dtype is not None else _probe.dtype,),
    ):
        result = _call_numpy(_np.array, object, dtype=dtype, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(array, _np.array, "counted_custom", "numel(input) FLOPs")


@_counted_wrapper
def zeros(
    shape: int | Sequence[int],
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return array of zeros. Wraps ``numpy.zeros``. Cost: 0 FLOPs."""
    budget = require_budget()
    with budget.deduct("zeros", flop_cost=0, subscripts=None, shapes=(), dtypes=()):
        result = _call_numpy(_np.zeros, shape, dtype=dtype, **kwargs)
    return _wrap_constant_fill(result)


attach_docstring(zeros, _np.zeros, "free", "0 FLOPs")


@_counted_wrapper
def ones(
    shape: int | Sequence[int],
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return array of ones. Wraps ``numpy.ones``. Cost: numel(output)."""
    budget = require_budget()
    result = _call_numpy(_np.ones, shape, dtype=dtype, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "ones", flop_cost=cost, subscripts=None, shapes=(), dtypes=(result.dtype,)
    ):
        result = _wrap_constant_fill(result)
    return result


attach_docstring(ones, _np.ones, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def full(
    shape: int | Sequence[int],
    fill_value: ArrayLike,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return array filled with *fill_value*. Cost: numel(output)."""
    budget = require_budget()
    result = _np.full(shape, fill_value, dtype=dtype, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "full", flop_cost=cost, subscripts=None, shapes=(), dtypes=(result.dtype,)
    ):
        result = _wrap_constant_fill(result)
    return result


attach_docstring(full, _np.full, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return identity matrix. Wraps ``numpy.eye``. Cost: diagonal length written."""
    budget = require_budget()
    cost = _eye_diagonal_length(N, M, k)
    with budget.deduct(
        "eye",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(_np.dtype(dtype),),
    ):
        result = _call_numpy(_np.eye, N, M=M, k=k, dtype=dtype, **kwargs)
    symmetry = _infer_structural_constructor_symmetry(kind="eye", N=N, M=M, k=k)
    if symmetry is not None:
        return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]
    return result  # type: ignore[return-value]


attach_docstring(
    eye, _np.eye, "counted_custom", "diagonal length written (0 if k is off-diagonal)"
)


@_counted_wrapper
def diag(v: ArrayLike, k: int = 0) -> FlopscopeArray:
    """Extract diagonal or construct diagonal array.

    Cost (weight 1.0):
    - 2-D input (extract): ``min(m, n)`` — copies the diagonal elements.
    - 1-D input (construct): ``numel(output) = (n + |k|)^2`` — materialises the full
      output matrix.
    """
    budget = require_budget()
    v = _np.asarray(v)
    if v.ndim == 1:
        # Constructing diagonal matrix: output is (n+|k|) x (n+|k|)
        n = v.shape[0] + abs(k)
        cost = n * n
    else:
        # Extracting diagonal: copies min(m,n) elements
        m, n = v.shape[0], v.shape[1] if v.ndim > 1 else v.shape[0]
        cost = min(m, n)
    with budget.deduct(
        "diag", flop_cost=cost, subscripts=None, shapes=(v.shape,), dtypes=(v.dtype,)
    ):
        result = _call_numpy(_np.diag, v, k=k)
    symmetry = _infer_structural_constructor_symmetry(kind="diag", k=k, v_ndim=v.ndim)
    if symmetry is not None:
        return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]
    return result  # type: ignore[return-value]


attach_docstring(diag, _np.diag, "free", "0 FLOPs")


@_counted_wrapper
def arange(*args: Any, **kwargs: Any) -> FlopscopeArray:
    """Return evenly spaced values. Cost: 2*numel(output) FLOPs (start + i*step per element, FMA=2)."""
    budget = require_budget()
    _dtype = kwargs.get("dtype")
    _billing_dtype = (
        _np.dtype(_dtype)
        if _dtype is not None
        else _np.result_type(*args)
        if args
        else _np.dtype(float)
    )
    with budget.deduct_after(
        "arange", subscripts=None, shapes=(), dtypes=(_billing_dtype,)
    ) as _op:
        result = _call_numpy(_np.arange, *args, **kwargs)
        _op.set_cost(2 * (result.size if hasattr(result, "size") else 1))
    return result


attach_docstring(
    arange,
    _np.arange,
    "counted_custom",
    "2*numel(output) FLOPs (start + i*step per element, FMA=2)",
)


@_counted_wrapper
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return evenly spaced numbers. Cost: 2*numel(output) FLOPs (start + i*step per element, FMA=2)."""
    budget = require_budget()
    _dtype = kwargs.get("dtype")
    _billing_dtype = (
        _np.dtype(_dtype) if _dtype is not None else _np.result_type(start, stop)
    )
    with budget.deduct_after(
        "linspace", subscripts=None, shapes=(), dtypes=(_billing_dtype,)
    ) as _op:
        result = _call_numpy(  # type: ignore[arg-type, call-overload]
            _np.linspace,
            _to_base_ndarray(start) if hasattr(start, "__array__") else start,
            _to_base_ndarray(stop) if hasattr(stop, "__array__") else stop,
            num=num,
            **kwargs,
        )
        samples = result[0] if isinstance(result, tuple) else result
        _op.set_cost(2 * (samples.size if hasattr(samples, "size") else 1))
    return result  # pyright: ignore[reportReturnType]  # (samples, step) tuple when retstep=True


attach_docstring(
    linspace,
    _np.linspace,
    "counted_custom",
    "2*numel(output) FLOPs (start + i*step per element, FMA=2)",
)


@_counted_wrapper
def zeros_like(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return array of zeros with same shape. Wraps ``numpy.zeros_like``. Cost: 0 FLOPs."""
    budget = require_budget()
    base = _to_base_ndarray(a)
    with budget.deduct(
        "zeros_like",
        flop_cost=0,
        subscripts=None,
        shapes=(_np.shape(base),),
        dtypes=(),
    ):
        result = _call_numpy(_np.zeros_like, base, dtype=dtype, **kwargs)
    propagated_symmetry = None
    if isinstance(a, SymmetricTensor):
        propagated_symmetry = _compatible_symmetry_for_shape(a.symmetry, result.shape)
    if propagated_symmetry is not None:
        return wrap_with_trusted_symmetry(result, propagated_symmetry)  # type: ignore[return-value]
    inferred_symmetry = _infer_constant_shape_symmetry(result.shape)
    if inferred_symmetry is None:
        if isinstance(a, SymmetricTensor):
            return _np.array(result, copy=False, subok=False)  # type: ignore[return-value]
        return result  # type: ignore[return-value]
    return wrap_with_inferred_symmetry(result, inferred_symmetry)  # type: ignore[return-value]


attach_docstring(zeros_like, _np.zeros_like, "free", "0 FLOPs")


@_counted_wrapper
def ones_like(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return array of ones with same shape. Wraps ``numpy.ones_like``. Cost: numel(output)."""
    budget = require_budget()
    base = _to_base_ndarray(a)
    result = _call_numpy(_np.ones_like, base, dtype=dtype, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "ones_like",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(result.dtype,),
    ):
        propagated_symmetry = None
        if isinstance(a, SymmetricTensor):
            propagated_symmetry = _compatible_symmetry_for_shape(
                a.symmetry, result.shape
            )
        if propagated_symmetry is not None:
            return wrap_with_trusted_symmetry(result, propagated_symmetry)  # type: ignore[return-value]
        inferred_symmetry = _infer_constant_shape_symmetry(result.shape)
        if inferred_symmetry is None:
            if isinstance(a, SymmetricTensor):
                return _call_numpy(_np.array, result, copy=False, subok=False)  # type: ignore[return-value]
            return result  # type: ignore[return-value]
        return wrap_with_inferred_symmetry(result, inferred_symmetry)  # type: ignore[return-value]


attach_docstring(ones_like, _np.ones_like, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def full_like(
    a: ArrayLike,
    fill_value: ArrayLike,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return full array with same shape. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = max(a_arr.size, 1)
    _billing_dtype = _np.dtype(dtype) if dtype is not None else a_arr.dtype
    with budget.deduct(
        "full_like",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(_billing_dtype,),
    ):
        result = _call_numpy(
            _np.full_like, _to_base_ndarray(a), fill_value, dtype=dtype, **kwargs
        )
    propagated_symmetry = None
    if isinstance(a, SymmetricTensor):
        propagated_symmetry = _compatible_symmetry_for_shape(a.symmetry, result.shape)
    if propagated_symmetry is not None:
        return wrap_with_trusted_symmetry(result, propagated_symmetry)  # type: ignore[return-value]
    inferred_symmetry = _infer_constant_shape_symmetry(result.shape)
    if inferred_symmetry is None:
        if isinstance(a, SymmetricTensor):
            return _np.array(result, copy=False, subok=False)  # type: ignore[return-value]
        return result  # type: ignore[return-value]
    return wrap_with_inferred_symmetry(result, inferred_symmetry)  # type: ignore[return-value]


attach_docstring(full_like, _np.full_like, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def empty(
    shape: int | Sequence[int],
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return uninitialized array. Wraps ``numpy.empty``. Cost: 0 FLOPs."""
    budget = require_budget()
    with budget.deduct("empty", flop_cost=0, subscripts=None, shapes=(), dtypes=()):
        result = _call_numpy(_np.empty, shape, dtype=dtype, **kwargs)
    # Uninitialized memory is not a constant fill — do NOT infer symmetry.
    return _asplainflopscope(result)


attach_docstring(empty, _np.empty, "free", "0 FLOPs")


@_counted_wrapper
def empty_like(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return uninitialized array with same shape. Wraps ``numpy.empty_like``. Cost: 0 FLOPs."""
    budget = require_budget()
    base = _to_base_ndarray(a)
    with budget.deduct(
        "empty_like",
        flop_cost=0,
        subscripts=None,
        shapes=(_np.shape(base),),
        dtypes=(),
    ):
        result = _call_numpy(_np.empty_like, base, dtype=dtype, **kwargs)
    # Uninitialized memory is not a constant fill — do NOT infer symmetry.
    return _asplainflopscope(result)


attach_docstring(empty_like, _np.empty_like, "free", "0 FLOPs")


@_counted_wrapper
def identity(n: int, dtype: DTypeLike = float) -> FlopscopeArray:
    """Return identity matrix. Wraps ``numpy.identity``. Cost: diagonal length written (=n)."""
    budget = require_budget()
    cost = _eye_diagonal_length(n, n, 0)
    with budget.deduct(
        "identity",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(_np.dtype(dtype),),
    ):
        result = _call_numpy(_np.identity, n, dtype=dtype)
    symmetry = _infer_structural_constructor_symmetry(kind="identity")
    if symmetry is not None:
        return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]
    return result  # type: ignore[return-value]


attach_docstring(
    identity, _np.identity, "counted_custom", "diagonal length written (=n) FLOPs"
)

# ---------------------------------------------------------------------------
# Tensor manipulation
# ---------------------------------------------------------------------------


@_counted_wrapper
def reshape(a: ArrayLike, /, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Reshape an array. Wraps ``numpy.reshape``. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = max(a_arr.size, 1)
    in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
    with budget.deduct(
        "reshape",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_arr.shape,),
        dtypes=(a_arr.dtype,),
    ):
        result = _call_numpy(_np.reshape, a_arr, *args, **kwargs)
    out_group = _st.transport_reshape(
        in_group,
        input_shape=a_arr.shape,
        output_shape=result.shape,
    )
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason="reshape merges or splits axes inside the symmetric block",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(
    reshape, _np.reshape, "counted_custom", "numel(input) FLOPs (even as a view)"
)


@_counted_wrapper
def transpose(
    a: ArrayLike,
    axes: Sequence[int] | None = None,
) -> FlopscopeArray:
    """Permute array dimensions. Wraps ``numpy.transpose``. Cost: 0 FLOPs."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
    with budget.deduct(
        "transpose", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
    ):
        result = _call_numpy(_np.transpose, a_arr, axes=axes)
    out_group = _st.transport_transpose(in_group, ndim=a_arr.ndim, axes=axes)
    # transpose never genuinely drops (axis perm always preserves S_n etc.).
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(transpose, _np.transpose, "free", "0 FLOPs")


@_counted_wrapper
def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> FlopscopeArray:
    """Swap two axes. Wraps ``numpy.swapaxes``. Cost: 0 FLOPs."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
    with budget.deduct(
        "swapaxes", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
    ):
        result = _call_numpy(_np.swapaxes, a_arr, axis1, axis2)
    out_group = _st.transport_swapaxes(
        in_group,
        ndim=a_arr.ndim,
        axis1=axis1,
        axis2=axis2,
    )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(swapaxes, _np.swapaxes, "free", "0 FLOPs")


@_counted_wrapper
def moveaxis(
    a: ArrayLike,
    source,
    destination,
) -> FlopscopeArray:
    """Move axes to new positions. Wraps ``numpy.moveaxis``. Cost: 0 FLOPs."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
    with budget.deduct(
        "moveaxis", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
    ):
        result = _call_numpy(_np.moveaxis, a_arr, source, destination)
    out_group = _st.transport_moveaxis(
        in_group,
        ndim=a_arr.ndim,
        source=source,
        destination=destination,
    )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(moveaxis, _np.moveaxis, "free", "0 FLOPs")


@_counted_wrapper
def concatenate(
    arrays: Sequence[ArrayLike],
    axis: int | None = 0,
    **kwargs: Any,
) -> FlopscopeArray:
    """Join arrays along an axis. Cost: numel(output)."""
    budget = require_budget()
    arr_list = [_np.asarray(a) for a in arrays]
    cost = max(sum(a.size for a in arr_list), 1)
    groups = [(a.symmetry if isinstance(a, SymmetricTensor) else None) for a in arrays]
    raw_arrs = [_to_base_ndarray(a) for a in arrays]
    with budget.deduct(
        "concatenate",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=tuple(a.dtype for a in arr_list),
    ):
        result = _call_numpy(_np.concatenate, raw_arrs, axis=axis, **kwargs)
    out_group = _st.transport_concatenate(
        groups,
        output_ndim=result.ndim,
        axis=axis,
    )
    if any(g is not None for g in groups) and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                g.axes if g.axes is not None else tuple(range(g.degree))
                for g in groups
                if g is not None
            ],
            reason="concatenate breaks block symmetry or mixes with plain inputs",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(concatenate, _np.concatenate, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def stack(
    arrays: Sequence[ArrayLike],
    axis: int = 0,
    **kwargs: Any,
) -> FlopscopeArray:
    """Stack arrays along a new axis. Cost: numel(output)."""
    budget = require_budget()
    arr_list = [_np.asarray(a) for a in arrays]
    cost = max(sum(a.size for a in arr_list), 1)
    groups = [a.symmetry if isinstance(a, SymmetricTensor) else None for a in arrays]
    with budget.deduct(
        "stack",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=tuple(a.dtype for a in arr_list),
    ):
        result = _call_numpy(
            _np.stack, _to_base_ndarray_tree(arrays), axis=axis, **kwargs
        )
    out_group = _st.transport_stack(groups, output_ndim=result.ndim, axis=axis)
    if any(g is not None for g in groups) and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                g.axes if g.axes is not None else tuple(range(g.degree))
                for g in groups
                if g is not None
            ],
            reason="stack inputs disagree or include plain arrays",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(stack, _np.stack, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def vstack(tup: Sequence[ArrayLike]) -> FlopscopeArray:
    """Stack arrays vertically. Cost: numel(output)."""
    budget = require_budget()
    arr_list = [_np.asarray(a) for a in tup]
    cost = max(sum(a.size for a in arr_list), 1)
    groups = [a.symmetry if isinstance(a, SymmetricTensor) else None for a in tup]
    input_ndims = [a.ndim for a in arr_list]
    with budget.deduct(
        "vstack",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=tuple(a.dtype for a in arr_list),
    ):
        result = _call_numpy(_np.vstack, _to_base_ndarray_tree(tup))  # type: ignore[arg-type, call-overload]
    out_group = _st.transport_vstack(
        groups,
        output_ndim=result.ndim,
        input_ndims=input_ndims,
    )
    if any(g is not None for g in groups) and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                g.axes if g.axes is not None else tuple(range(g.degree))
                for g in groups
                if g is not None
            ],
            reason="vstack breaks block symmetry",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(vstack, _np.vstack, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def hstack(tup: Sequence[ArrayLike]) -> FlopscopeArray:
    """Stack arrays horizontally. Cost: numel(output)."""
    budget = require_budget()
    arr_list = [_np.asarray(a) for a in tup]
    cost = max(sum(a.size for a in arr_list), 1)
    groups = [a.symmetry if isinstance(a, SymmetricTensor) else None for a in tup]
    input_ndims = [a.ndim for a in arr_list]
    with budget.deduct(
        "hstack",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=tuple(a.dtype for a in arr_list),
    ):
        result = _call_numpy(_np.hstack, _to_base_ndarray_tree(tup))  # type: ignore[arg-type, call-overload]
    out_group = _st.transport_hstack(
        groups,
        output_ndim=result.ndim,
        input_ndims=input_ndims,
    )
    if any(g is not None for g in groups) and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                g.axes if g.axes is not None else tuple(range(g.degree))
                for g in groups
                if g is not None
            ],
            reason="hstack breaks block symmetry",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(hstack, _np.hstack, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def split(
    ary: ArrayLike,
    indices_or_sections: int | Sequence[int],
    axis: int = 0,
) -> list[FlopscopeArray]:
    """Split array. Cost: numel(input)."""
    budget = require_budget()
    ary_arr = _np.asarray(ary)
    cost = ary_arr.size
    in_group = ary.symmetry if isinstance(ary, SymmetricTensor) else None
    out_group = _st.transport_split(in_group, input_shape=ary_arr.shape, axis=axis)
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason=f"split along axis {axis} breaks block symmetry",
        )
    with budget.deduct(
        "split",
        flop_cost=cost,
        subscripts=None,
        shapes=(ary_arr.shape,),
        dtypes=(ary_arr.dtype,),
    ):
        raw_pieces = _call_numpy(
            _np.split,
            ary_arr,
            indices_or_sections,
            axis=axis,
        )
    if out_group is not None:
        return [wrap_with_symmetry(p, out_group) for p in raw_pieces]  # type: ignore[return-value]
    return [_asplainflopscope(p) for p in raw_pieces]  # type: ignore[return-value]


attach_docstring(split, _np.split, "free", "0 FLOPs")


@_counted_wrapper
def hsplit(
    ary: ArrayLike,
    indices_or_sections: int | Sequence[int],
) -> list[FlopscopeArray]:
    """Split array horizontally. Wraps ``numpy.hsplit``. Cost: 0 FLOPs."""
    budget = require_budget()
    ary_arr = _np.asarray(ary)
    in_group = ary.symmetry if isinstance(ary, SymmetricTensor) else None
    out_group = _st.transport_hsplit(in_group, input_shape=ary_arr.shape)
    with budget.deduct(
        "hsplit", flop_cost=0, subscripts=None, shapes=(ary_arr.shape,), dtypes=()
    ):
        raw_pieces = _call_numpy(_np.hsplit, ary_arr, indices_or_sections)
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason="hsplit breaks block symmetry",
        )
    if out_group is not None:
        return [wrap_with_symmetry(p, out_group) for p in raw_pieces]  # type: ignore[return-value]
    return [_asplainflopscope(p) for p in raw_pieces]  # type: ignore[return-value]


attach_docstring(hsplit, _np.hsplit, "free", "0 FLOPs")


@_counted_wrapper
def vsplit(
    ary: ArrayLike,
    indices_or_sections: int | Sequence[int],
) -> list[FlopscopeArray]:
    """Split array vertically. Cost: numel(input)."""
    budget = require_budget()
    ary_arr = _np.asarray(ary)
    cost = ary_arr.size
    in_group = ary.symmetry if isinstance(ary, SymmetricTensor) else None
    out_group = _st.transport_vsplit(in_group, input_shape=ary_arr.shape)
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason="vsplit breaks block symmetry",
        )
    with budget.deduct(
        "vsplit",
        flop_cost=cost,
        subscripts=None,
        shapes=(ary_arr.shape,),
        dtypes=(ary_arr.dtype,),
    ):
        raw_pieces = _call_numpy(_np.vsplit, ary_arr, indices_or_sections)
    if out_group is not None:
        return [wrap_with_symmetry(p, out_group) for p in raw_pieces]  # type: ignore[return-value]
    return [_asplainflopscope(p) for p in raw_pieces]  # type: ignore[return-value]


attach_docstring(vsplit, _np.vsplit, "free", "0 FLOPs")


@_counted_wrapper
def squeeze(
    a: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
) -> FlopscopeArray:
    """Remove length-1 axes. Wraps ``numpy.squeeze``. Cost: 0 FLOPs."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
    with budget.deduct(
        "squeeze", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
    ):
        result = _call_numpy(_np.squeeze, a_arr, axis=axis)
    out_group = _st.transport_squeeze(in_group, input_shape=a_arr.shape, axis=axis)
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason="squeeze removes an axis inside the symmetric block",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(squeeze, _np.squeeze, "free", "0 FLOPs")


@_counted_wrapper
def expand_dims(a: ArrayLike, axis) -> FlopscopeArray:
    """Insert a new axis. Wraps ``numpy.expand_dims``. Cost: 0 FLOPs."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
    with budget.deduct(
        "expand_dims", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
    ):
        result = _call_numpy(_np.expand_dims, a_arr, axis=axis)
    out_group = _st.transport_expand_dims(
        in_group,
        input_ndim=a_arr.ndim,
        axis=axis,
    )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(expand_dims, _np.expand_dims, "free", "0 FLOPs")


@_counted_wrapper
def ravel(a: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Flatten array. Cost: numel(input) (= numel(output); ravel does not change element count)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = max(a_arr.size, 1)
    in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
    out_group = _st.transport_ravel(in_group, input_shape=a_arr.shape)
    with budget.deduct(
        "ravel",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_arr.shape,),
        dtypes=(a_arr.dtype,),
    ):
        result = _call_numpy(_np.ravel, a_arr, **kwargs)
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason="ravel collapses to a single axis; block cannot fit",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(
    ravel, _np.ravel, "counted_custom", "numel(input) FLOPs (even for a view result)"
)


@_counted_wrapper
def copy(a: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Return copy of array. Wraps ``numpy.copy``. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = max(a_arr.size, 1)
    with budget.deduct(
        "copy",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_arr.shape,),
        dtypes=(a_arr.dtype,),
    ):
        result = _call_numpy(_np.copy, a_arr, **kwargs)
    if isinstance(a, SymmetricTensor):
        return wrap_with_symmetry(result, a.symmetry)  # type: ignore[return-value]
    return result  # type: ignore[return-value]


attach_docstring(copy, _np.copy, "counted_custom", "numel(input) FLOPs")


@_counted_wrapper
def where(
    condition: ArrayLike,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
) -> FlopscopeArray | tuple[FlopscopeArray, ...]:
    """Return elements chosen from *x*/*y*, or indices where *condition* holds.

    Cost: 3-arg select is free (selection by a given mask). 1-arg form
    (``where(condition)`` == ``nonzero``) derives indices by testing values,
    so it is charged ``numel`` at the comparison tier (weight 1.0).
    """
    budget = require_budget()
    cond_arr = _np.asarray(condition)
    if x is None and y is None:
        # 1-arg: equivalent to nonzero -> charged numel.
        with budget.deduct(
            "where",
            flop_cost=cond_arr.size,
            subscripts=None,
            shapes=(cond_arr.shape,),
            dtypes=(cond_arr.dtype,),
        ):
            result = _call_numpy(_np.where, _to_base_ndarray(condition))
    else:
        # 3-arg: pure selection by a given mask -> free (still time-accounted).
        with budget.deduct(
            "where", flop_cost=0, subscripts=None, shapes=(cond_arr.shape,), dtypes=()
        ):
            result = _call_numpy(
                _np.where,
                _to_base_ndarray(condition),
                _to_base_ndarray(x),  # type: ignore[arg-type, call-overload]
                _to_base_ndarray(y),  # type: ignore[arg-type, call-overload]
            )
    return result  # type: ignore[return-value]


attach_docstring(
    where, _np.where, "counted_custom", "numel(cond) FLOPs (1-arg); 0 FLOPs (3-arg)"
)


@_counted_wrapper
def tile(A: ArrayLike, reps: int | Sequence[int]) -> FlopscopeArray:
    """Construct array by repeating. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(A)
    in_group = A.symmetry if isinstance(A, SymmetricTensor) else None
    with budget.deduct_after(
        "tile", subscripts=None, shapes=(), dtypes=(a_arr.dtype,)
    ) as _op:
        result = _call_numpy(_np.tile, a_arr, reps)
        _op.set_cost(max(result.size, 1))
    out_group = _st.transport_tile(
        in_group,
        input_shape=a_arr.shape,
        output_shape=result.shape,
        reps=reps,
    )
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason="tile reps not constant on block orbit",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(tile, _np.tile, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def repeat(
    a: ArrayLike,
    repeats: int | ArrayLike,
    axis: int | None = None,
) -> FlopscopeArray:
    """Repeat elements. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
    out_group = _st.transport_repeat(in_group, input_shape=a_arr.shape, axis=axis)
    with budget.deduct_after(
        "repeat", subscripts=None, shapes=(a_arr.shape,), dtypes=(a_arr.dtype,)
    ) as _op:
        result = _call_numpy(_np.repeat, a_arr, repeats, axis=axis)  # type: ignore[arg-type]
        _op.set_cost(max(result.size, 1))
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason="repeat along a block axis breaks block symmetry",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(repeat, _np.repeat, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def flip(
    m: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
) -> FlopscopeArray:
    """Reverse order of elements. Wraps ``numpy.flip``. Cost: 0 FLOPs."""
    budget = require_budget()
    a_arr = _np.asarray(m)
    in_group = m.symmetry if isinstance(m, SymmetricTensor) else None
    with budget.deduct(
        "flip", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
    ):
        result = _call_numpy(_np.flip, a_arr, axis=axis)
    out_group = _st.transport_flip(
        in_group,
        ndim=a_arr.ndim,
        axes_flipped=axis,
    )
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason="flip on a proper subset of block axes breaks group action",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(flip, _np.flip, "free", "0 FLOPs")


@_counted_wrapper
def roll(
    a: ArrayLike,
    shift: int | Sequence[int],
    axis: int | Sequence[int] | None = None,
) -> FlopscopeArray:
    """Roll array elements along an axis. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
    with budget.deduct_after(
        "roll", subscripts=None, shapes=(a_arr.shape,), dtypes=(a_arr.dtype,)
    ) as _op:
        result = _call_numpy(_np.roll, a_arr, shift, axis=axis)
        _op.set_cost(max(result.size, 1))
    out_group = _st.transport_roll(in_group, input_shape=a_arr.shape, axis=axis)
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason="roll along a block axis breaks block symmetry",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(roll, _np.roll, "counted_custom", "numel(output) FLOPs")


_PAD_FREE_MODES = frozenset({"constant", "edge", "empty", "wrap"})
_PAD_STAT_MODES = frozenset({"maximum", "minimum", "mean", "median"})


def _pad_pairs(value, ndim):
    """Normalize int / (b, a) / ((b, a), ...) into a list of (before, after) per axis."""
    arr = _np.asarray(value)
    if arr.ndim == 0:
        v = int(arr)
        return [(v, v)] * ndim
    if arr.shape == (1,):
        v = int(arr[0])
        return [(v, v)] * ndim
    if arr.shape == (2,):
        return [(int(arr[0]), int(arr[1]))] * ndim
    if arr.shape == (1, 2):
        return [(int(arr[0, 0]), int(arr[0, 1]))] * ndim
    return [(int(arr[i, 0]), int(arr[i, 1])) for i in range(ndim)]


def _pad_flop_cost(in_shape, pad_width, mode, kwargs):
    """flop_cost for np.pad: 0 for movement modes; reduction/affine for value modes."""
    ndim = len(in_shape)
    # Movement modes short-circuit BEFORE normalizing pad_width, so a malformed
    # pad_width surfaces numpy's own clean ValueError (not an IndexError from
    # _pad_pairs) for these modes.
    if mode in _PAD_FREE_MODES:
        return 0
    if mode in ("reflect", "symmetric") and kwargs.get("reflect_type", "even") != "odd":
        return 0
    numel_in = _math.prod(in_shape) if ndim else 1
    pad_pairs = _pad_pairs(pad_width, ndim)
    numel_out = (
        _math.prod(s + b + a for s, (b, a) in zip(in_shape, pad_pairs, strict=False))
        if ndim
        else 1
    )
    if mode in ("reflect", "symmetric"):  # reflect_type == "odd" (even handled above)
        return 2 * (numel_out - numel_in)
    if mode == "linear_ramp":
        return 2 * (numel_out - numel_in)
    if mode in _PAD_STAT_MODES:
        stat_length = kwargs.get("stat_length", None)
        if stat_length is None:
            stat_pairs = [(in_shape[i], in_shape[i]) for i in range(ndim)]
        else:
            stat_pairs = _pad_pairs(stat_length, ndim)
        cost = 0
        for i in range(ndim):
            before, after = pad_pairs[i]
            axis_len = in_shape[i]
            if (before == 0 and after == 0) or axis_len == 0:
                continue
            cross = numel_in // axis_len
            sl_b = min(stat_pairs[i][0], axis_len)
            sl_a = min(stat_pairs[i][1], axis_len)
            # Charge only the PADDED sides (the stats actually placed in the
            # output). numpy also computes a stat for an unpadded side but discards
            # it (placed into a width-0 region, unreadable), so billing it would
            # over-charge for work the caller gets no value from.
            stats = []
            if before > 0:
                stats.append(sl_b)
            if after > 0:
                stats.append(sl_a)
            # A full-axis stat is identical for both sides -> numpy computes it once.
            if before > 0 and after > 0 and sl_b == axis_len and sl_a == axis_len:
                stats = [axis_len]
            cost += cross * sum(stats)
            if mode == "mean":
                cost += cross * len(stats)  # one divide per stat output cell
        return cost
    return 0  # unknown string mode: let numpy raise its own ValueError


@_counted_wrapper
def pad(
    array: ArrayLike, pad_width: Any, mode: Any = "constant", **kwargs: Any
) -> FlopscopeArray:
    """Pad an array. Cost: 0 for data-movement modes (constant/edge/empty/wrap/
    reflect/symmetric with reflect_type='even'); reduction cost for
    maximum/minimum/mean/median; 2*(numel_out-numel_in) for linear_ramp and for
    reflect/symmetric with reflect_type='odd'. mode=<callable> is unsupported."""
    if callable(mode):
        raise ValueError(
            "flopscope: pad(mode=<callable>) is not supported under FLOP metering "
            "(arbitrary uncounted compute). Use a string mode, or compute the padding "
            "values with counted ops and pad with mode='constant', constant_values=..."
        )
    budget = require_budget()
    _warn_if_symmetric(array, "pad")
    arr_probe = _np.asarray(array)
    cost = _pad_flop_cost(arr_probe.shape, pad_width, mode, kwargs)
    with budget.deduct(
        "pad", flop_cost=cost, subscripts=None, shapes=(), dtypes=(arr_probe.dtype,)
    ):
        result = _call_numpy(
            _np.pad, _to_base_ndarray(array), pad_width, mode=mode, **kwargs
        )
    return result  # type: ignore[return-value]


attach_docstring(
    pad,
    _np.pad,
    "counted_custom",
    "0 for movement modes; reduction/affine for value modes",
)


@_counted_wrapper
def triu(m: ArrayLike, k: int = 0) -> FlopscopeArray:
    """Upper triangle. Wraps ``numpy.triu``. Cost: numel(output)."""
    budget = require_budget()
    _warn_if_symmetric(m, "triu")
    m_arr = _np.asarray(m)
    with budget.deduct_after(
        "triu", subscripts=None, shapes=(), dtypes=(m_arr.dtype,)
    ) as _op:
        result = _call_numpy(_np.triu, _to_base_ndarray(m), k=k)
        _op.set_cost(result.size)
    return result  # type: ignore[return-value]


attach_docstring(triu, _np.triu, "free", "0 FLOPs")


@_counted_wrapper
def tril(m: ArrayLike, k: int = 0) -> FlopscopeArray:
    """Lower triangle. Wraps ``numpy.tril``. Cost: numel(output)."""
    budget = require_budget()
    _warn_if_symmetric(m, "tril")
    m_arr = _np.asarray(m)
    with budget.deduct_after(
        "tril", subscripts=None, shapes=(), dtypes=(m_arr.dtype,)
    ) as _op:
        result = _call_numpy(_np.tril, _to_base_ndarray(m), k=k)
        _op.set_cost(result.size)
    return result  # type: ignore[return-value]


attach_docstring(tril, _np.tril, "free", "0 FLOPs")


@_counted_wrapper
def diagonal(
    a: ArrayLike,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> FlopscopeArray:
    """Return diagonal view. Cost: 0 FLOPs.

    ``numpy.diagonal`` returns a read-only VIEW of the array data — no elements
    are copied or computed.  The budget deduction is zero.
    """
    budget = require_budget()
    _warn_if_symmetric(a, "diagonal")
    a_arr = _np.asarray(a)
    with budget.deduct(
        "diagonal", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
    ):
        result = _call_numpy(
            _np.diagonal, _to_base_ndarray(a), offset=offset, axis1=axis1, axis2=axis2
        )
    return result  # type: ignore[return-value]


attach_docstring(diagonal, _np.diagonal, "free", "0 FLOPs")


@_counted_wrapper
def broadcast_to(
    array: ArrayLike,
    shape: int | Sequence[int],
) -> FlopscopeArray:
    """Broadcast array to shape. Cost: numel(output)."""
    output_shape = (shape,) if isinstance(shape, int) else tuple(shape)
    arr = _np.asarray(array)
    budget = require_budget()
    cost = max(int(_np.prod(output_shape)), 1)
    with budget.deduct(
        "broadcast_to",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(arr.dtype,),
    ):
        result = _call_numpy(_np.broadcast_to, arr, output_shape)
    in_group = array.symmetry if isinstance(array, SymmetricTensor) else None
    out_group = _st.transport_broadcast_to(
        in_group,
        input_shape=arr.shape,
        output_shape=output_shape,
    )
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason="broadcast_to expands length-1 block axes",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(broadcast_to, _np.broadcast_to, "free", "0 FLOPs")


@_counted_wrapper
def meshgrid(*xi: ArrayLike, **kwargs: Any) -> tuple[FlopscopeArray, ...]:
    """Return coordinate matrices. Cost: numel of materialized output grids; sparse=True bills sum of input lengths; copy=False views cost 1."""
    budget = require_budget()
    sizes = [_np.asarray(x).size for x in xi]
    sparse = bool(kwargs.get("sparse", False))
    copy = bool(kwargs.get("copy", True))
    if not copy:
        cost = 1  # reshape/broadcast views; view convention, house floor of 1
    elif sparse:
        cost = max(sum(sizes), 1)  # len(xi) copies, each numel = own input length
    else:
        # dense: total output numel (unchanged)
        cost = max(int(_np.prod(sizes)) * len(sizes), 1) if sizes else 1
    _xi_dtypes = tuple(_np.asarray(x).dtype for x in xi) if xi else (_np.dtype(float),)
    with budget.deduct(
        "meshgrid", flop_cost=cost, subscripts=None, shapes=(), dtypes=_xi_dtypes
    ):
        result = _call_numpy(_np.meshgrid, *[_to_base_ndarray(x) for x in xi], **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(
    meshgrid,
    _np.meshgrid,
    "counted_custom",
    "numel of materialized output grids; sparse=True bills sum of input lengths; copy=False views cost 1",
)

# ---------------------------------------------------------------------------
# Type / info helpers
# ---------------------------------------------------------------------------


def _cast_changes_values(src_dtype: Any, dst_dtype: Any) -> bool:
    """True when casting src->dst alters element values (so it is charged).

    A lossless representation/width cast (e.g. float32->float64, int32->int64,
    bool->int) is ``can_cast(..., "safe")`` and stays free. To-bool (``!=0``),
    float->int (truncation), float-narrowing (round), and complex->real all
    fail the safe-cast test and are charged ``numel``.
    """
    return not _np.can_cast(_np.dtype(src_dtype), _np.dtype(dst_dtype), casting="safe")


@_counted_wrapper
def astype(
    x: ArrayLike,
    dtype: DTypeLike,
    /,
    *,
    copy: bool = True,
    device: Any = None,
) -> FlopscopeArray:
    """Cast array to *dtype*. Wraps ``np.astype(x, dtype)``.

    Cost: 0 -- a representation change (weight 0); every cast is free,
    lossy or lossless. The structural cost (``numel`` when the cast changes
    values: to-bool, float->int, narrowing, complex->real) is still tracked
    internally and stays available if the weight is ever raised.
    """
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size if _cast_changes_values(x_arr.dtype, dtype) else 0
    with budget.deduct(
        "astype",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=(_heavier_billing_dtype(x_arr.dtype, _np.dtype(dtype)),),
    ):
        result = _call_numpy(
            _np.astype, _to_base_ndarray(x), dtype, copy=copy, device=device
        )
    return result  # type: ignore[return-value]


@_counted_wrapper
def _astype_counted(
    arr: Any,
    dtype: DTypeLike,
    *,
    order: Any = "K",
    casting: Any = "unsafe",
    subok: bool = True,
    copy: bool = True,
) -> FlopscopeArray:
    """Counted backend for the ndarray.astype METHOD (honors all params).

    Unlike the array-api ``astype(x, dtype, *, copy, device)`` function above,
    this backend receives the full ndarray-method signature including ``order``,
    ``casting``, and ``subok``. In particular it passes ``casting`` through to
    ``np.ndarray.astype`` so unsafe casts raise ``TypeError`` just as they do
    on plain ndarrays.

    Cost: 0 -- a representation change (weight 0); every cast is free, lossy
    or lossless. The structural cost (``numel`` when the cast changes
    values) is still tracked internally and stays available if the weight
    is ever raised.
    """
    budget = require_budget()
    arr_np = _np.asarray(arr)
    cost = arr_np.size if _cast_changes_values(arr_np.dtype, dtype) else 0
    with budget.deduct(
        "astype",
        flop_cost=cost,
        subscripts=None,
        shapes=(arr_np.shape,),
        dtypes=(_heavier_billing_dtype(arr_np.dtype, _np.dtype(dtype)),),
    ):
        result = _call_numpy(
            _np.ndarray.astype,
            _to_base_ndarray(arr),
            dtype,
            order=order,
            casting=casting,
            subok=subok,
            copy=copy,
        )
    return _asplainflopscope(result)  # type: ignore[return-value]


@_counted_wrapper
def asarray(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Convert to array. Cost: 0 -- representation change (weight 0)."""
    budget = require_budget()
    _probe = _np.asarray(a)
    # asarray is a view/no-op unless an explicit dtype= forces a value-changing
    # cast; then it does the same structural work as astype (numel at the
    # heavier of source/target rate) tracked internally, but both are weight
    # 0 -- a value-changing asarray(dtype=) is free, same as astype.
    if dtype is not None and _cast_changes_values(_probe.dtype, dtype):
        cost = _probe.size
        _asarray_dtypes: tuple = (
            _heavier_billing_dtype(_probe.dtype, _np.dtype(dtype)),
        )
    else:
        cost = 0
        _asarray_dtypes = (_probe.dtype,)
    with budget.deduct(
        "asarray",
        flop_cost=cost,
        subscripts=None,
        shapes=(_probe.shape,),
        dtypes=_asarray_dtypes,
    ):
        result = _call_numpy(_np.asarray, a, dtype=dtype, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(
    asarray,
    _np.asarray,
    "free",
    "0 FLOPs (representation change; structural cost retained at weight 0)",
)


@_counted_wrapper
def isnan(x: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Test element-wise for NaN. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    with budget.deduct(
        "isnan",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=(x_arr.dtype,),
    ):
        # Strip flopscope subclasses so the raw NumPy ufunc does not
        # re-dispatch through __array_ufunc__ and recurse.
        result = _call_numpy(_np.isnan, _to_base_ndarray(x), **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(isnan, _np.isnan, "counted_custom", "numel(input) FLOPs")


@_counted_wrapper
def isfinite(x: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Test element-wise for finiteness. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    with budget.deduct(
        "isfinite",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=(x_arr.dtype,),
    ):
        result = _call_numpy(_np.isfinite, _to_base_ndarray(x), **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(isfinite, _np.isfinite, "counted_custom", "numel(input) FLOPs")


@_counted_wrapper
def isinf(x: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Test element-wise for Inf. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    with budget.deduct(
        "isinf",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=(x_arr.dtype,),
    ):
        result = _call_numpy(_np.isinf, _to_base_ndarray(x), **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(isinf, _np.isinf, "counted_custom", "numel(input) FLOPs")

# ---------------------------------------------------------------------------
# Additional array ops
# ---------------------------------------------------------------------------


@_counted_wrapper
def append(
    arr: ArrayLike,
    values: ArrayLike,
    axis: int | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Append values. Cost: numel(output) = arr.size + values.size."""
    budget = require_budget()
    _warn_if_symmetric(arr, "append")
    arr_arr = _np.asarray(arr)
    values_arr = _np.asarray(values)
    cost = max(
        arr_arr.size + values_arr.size, 1
    )  # numel(output): np.append = concatenate([arr, values])
    with budget.deduct(
        "append",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(arr_arr.dtype, values_arr.dtype),
    ):
        result = _call_numpy(
            _np.append,
            _to_base_ndarray(arr),
            _to_base_ndarray(values),
            axis=axis,
            **kwargs,
        )
    return result  # type: ignore[return-value]


attach_docstring(append, _np.append, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def argwhere(a: ArrayLike, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Find indices of non-zero elements. Cost: numel(input) at weight 1.0.

    Equivalent to ``transpose(nonzero(a))``; weight 1.0 matches ``nonzero``.
    """
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    with budget.deduct(
        "argwhere",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_arr.shape,),
        dtypes=(a_arr.dtype,),
    ):
        result = _call_numpy(_np.argwhere, _to_base_ndarray(a), *args, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(argwhere, _np.argwhere, "counted_custom", "numel(input) FLOPs")


@_counted_wrapper
def array_split(ary: ArrayLike, *args: Any, **kwargs: Any) -> list[FlopscopeArray]:
    """Split array into sub-arrays. Cost: numel(input)."""
    budget = require_budget()
    _warn_if_symmetric(ary, "array_split")
    ary_arr = _np.asarray(ary)
    cost = ary_arr.size
    with budget.deduct(
        "array_split",
        flop_cost=cost,
        subscripts=None,
        shapes=(ary_arr.shape,),
        dtypes=(ary_arr.dtype,),
    ):
        result = _call_numpy(_np.array_split, _to_base_ndarray(ary), *args, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(array_split, _np.array_split, "free", "0 FLOPs")


@_counted_wrapper
def asarray_chkfinite(a: ArrayLike, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Convert to array checking for NaN/Inf. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    with budget.deduct_after(
        "asarray_chkfinite", subscripts=None, shapes=(), dtypes=(a_arr.dtype,)
    ) as _op:
        result = _call_numpy(
            _np.asarray_chkfinite, _to_base_ndarray(a), *args, **kwargs
        )
        _op.set_cost(
            result.size
            if hasattr(result, "size")
            else len(result)
            if hasattr(result, "__len__")
            else 1
        )
    return result  # type: ignore[return-value]


attach_docstring(
    asarray_chkfinite, _np.asarray_chkfinite, "counted_custom", "numel(output) FLOPs"
)


@_counted_wrapper
def atleast_1d(
    *arys: ArrayLike,
) -> FlopscopeArray | tuple[FlopscopeArray, ...]:
    """Convert to 1-D or higher. Wraps ``numpy.atleast_1d``. Cost: 0 FLOPs."""
    budget = require_budget()

    def _one(a):
        a_arr = _np.asarray(a)
        in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
        with budget.deduct(
            "atleast_1d", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
        ):
            result = _call_numpy(_np.atleast_1d, a_arr)
        out_group = _st.transport_atleast_1d(in_group, input_shape=a_arr.shape)
        if in_group is not None and out_group is None:
            _warn_symmetry_loss(
                lost_dims=[
                    in_group.axes
                    if in_group.axes is not None
                    else tuple(range(in_group.degree))
                ],
                reason="atleast_1d incompatible with block structure",
            )
        if out_group is not None:
            return wrap_with_symmetry(result, out_group)
        return _asplainflopscope(result)

    if len(arys) == 1:
        return _one(arys[0])  # type: ignore[return-value]
    return tuple(_one(a) for a in arys)  # type: ignore[return-value]


attach_docstring(atleast_1d, _np.atleast_1d, "free", "0 FLOPs")


@_counted_wrapper
def atleast_2d(
    *arys: ArrayLike,
) -> FlopscopeArray | tuple[FlopscopeArray, ...]:
    """Convert to 2-D or higher. Wraps ``numpy.atleast_2d``. Cost: 0 FLOPs."""
    budget = require_budget()

    def _one(a):
        a_arr = _np.asarray(a)
        in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
        with budget.deduct(
            "atleast_2d", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
        ):
            result = _call_numpy(_np.atleast_2d, a_arr)
        out_group = _st.transport_atleast_2d(in_group, input_shape=a_arr.shape)
        if in_group is not None and out_group is None:
            _warn_symmetry_loss(
                lost_dims=[
                    in_group.axes
                    if in_group.axes is not None
                    else tuple(range(in_group.degree))
                ],
                reason="atleast_2d incompatible with block structure",
            )
        if out_group is not None:
            return wrap_with_symmetry(result, out_group)
        return _asplainflopscope(result)

    if len(arys) == 1:
        return _one(arys[0])  # type: ignore[return-value]
    return tuple(_one(a) for a in arys)  # type: ignore[return-value]


attach_docstring(atleast_2d, _np.atleast_2d, "free", "0 FLOPs")


@_counted_wrapper
def atleast_3d(
    *arys: ArrayLike,
) -> FlopscopeArray | tuple[FlopscopeArray, ...]:
    """Convert to 3-D or higher. Wraps ``numpy.atleast_3d``. Cost: 0 FLOPs."""
    budget = require_budget()

    def _one(a):
        a_arr = _np.asarray(a)
        in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
        with budget.deduct(
            "atleast_3d", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
        ):
            result = _call_numpy(_np.atleast_3d, a_arr)
        out_group = _st.transport_atleast_3d(in_group, input_shape=a_arr.shape)
        if in_group is not None and out_group is None:
            _warn_symmetry_loss(
                lost_dims=[
                    in_group.axes
                    if in_group.axes is not None
                    else tuple(range(in_group.degree))
                ],
                reason="atleast_3d incompatible with block structure",
            )
        if out_group is not None:
            return wrap_with_symmetry(result, out_group)
        return _asplainflopscope(result)

    if len(arys) == 1:
        return _one(arys[0])  # type: ignore[return-value]
    return tuple(_one(a) for a in arys)  # type: ignore[return-value]


attach_docstring(atleast_3d, _np.atleast_3d, "free", "0 FLOPs")


@_counted_wrapper
def base_repr(*args, **kwargs):
    """Return string representation of number. Cost: numel(output)."""
    budget = require_budget()
    result = _np.base_repr(*args, **kwargs)
    cost = len(result)
    with budget.deduct(
        "base_repr", flop_cost=cost, subscripts=None, shapes=(), dtypes=()
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(base_repr, _np.base_repr, "counted_custom", "len(result) FLOPs")


@_counted_wrapper
def binary_repr(*args, **kwargs):
    """Return binary representation of integer. Cost: numel(output)."""
    budget = require_budget()
    result = _np.binary_repr(*args, **kwargs)
    cost = len(result)
    with budget.deduct(
        "binary_repr", flop_cost=cost, subscripts=None, shapes=(), dtypes=()
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(binary_repr, _np.binary_repr, "counted_custom", "len(result) FLOPs")


@_counted_wrapper
def block(*args, **kwargs):
    """Assemble array from nested lists. Cost: numel(output)."""
    budget = require_budget()

    # Warn for any SymmetricTensor found in the nested structure.
    def _walk_warn(obj):
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _walk_warn(item)
        else:
            _warn_if_symmetric(obj, "block")

    for a in args:
        _walk_warn(a)
    with budget.deduct_after("block", subscripts=None, shapes=(), dtypes=()) as _op:
        result = _call_numpy(
            _np.block, *[_to_base_ndarray_tree(a) for a in args], **kwargs
        )
        _op.set_cost(result.size if hasattr(result, "size") else 1)
    return result


attach_docstring(block, _np.block, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def bmat(*args, **kwargs):
    """Build matrix from string/nested sequence. Cost: numel(output) at weight 1.0.

    Nested-block assembly is structurally identical to ``block``; both copy elements
    once, so weight 1.0 (matching ``block``).
    """
    budget = require_budget()
    # First arg may be a string OR a nested sequence of arrays
    stripped_args = []
    for arg in args:
        if isinstance(arg, (tuple, list)):
            stripped_args.append(_to_base_ndarray_tree(arg))
        else:
            stripped_args.append(arg)
    with budget.deduct_after("bmat", subscripts=None, shapes=(), dtypes=()) as _op:
        result = _call_numpy(_np.bmat, *stripped_args, **kwargs)
        _op.set_cost(result.size if hasattr(result, "size") else 1)
    return result


attach_docstring(bmat, _np.bmat, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def broadcast_arrays(*args: ArrayLike, **kwargs: Any) -> tuple[FlopscopeArray, ...]:
    """Broadcast any number of arrays. Cost: numel(output)."""
    arrays = tuple(_np.asarray(arg) for arg in args)
    budget = require_budget()
    result = _np.broadcast_arrays(*arrays, **kwargs)
    cost = sum(a.size for a in result)
    with budget.deduct(
        "broadcast_arrays",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=tuple(a.dtype for a in arrays),
    ):
        pass  # numpy call already executed above
    if not result:
        return result  # type: ignore[return-value]
    output_shape = result[0].shape
    wrapped = []
    for original, array, broadcasted in zip(args, arrays, result, strict=True):
        symmetry = broadcast_group(
            original.symmetry if isinstance(original, SymmetricTensor) else None,
            input_shape=array.shape,
            output_shape=output_shape,
        )
        wrapped.append(wrap_with_symmetry(broadcasted, symmetry))
    return tuple(wrapped)


attach_docstring(broadcast_arrays, _np.broadcast_arrays, "free", "0 FLOPs")


def broadcast_shapes(*args, **kwargs):
    """Broadcast shapes to a common shape. Wraps ``numpy.broadcast_shapes``. Cost: 0 FLOPs."""
    return _np.broadcast_shapes(*args, **kwargs)


attach_docstring(broadcast_shapes, _np.broadcast_shapes, "free", "0 FLOPs")


def can_cast(*args, **kwargs):
    """Returns True if cast between data types can occur. Wraps ``numpy.can_cast``. Cost: 0 FLOPs."""
    return _np.can_cast(*args, **kwargs)


attach_docstring(can_cast, _np.can_cast, "free", "0 FLOPs")


@_counted_wrapper
def choose(*args, **kwargs):
    """Construct array from index array. Cost: numel(output)."""
    budget = require_budget()
    # Warn if the first arg (index array) carries symmetry.
    if args:
        _warn_if_symmetric(args[0], "choose")
    # Args: (a, choices, ...) or just (a, choices) — strip arrays.
    stripped_args = []
    for arg in args:
        if isinstance(arg, _np.ndarray):
            stripped_args.append(_to_base_ndarray(arg))
        elif isinstance(arg, (tuple, list)):
            stripped_args.append(_to_base_ndarray_tree(arg))
        else:
            stripped_args.append(arg)
    with budget.deduct_after("choose", subscripts=None, shapes=(), dtypes=()) as _op:
        result = _call_numpy(_np.choose, *stripped_args, **kwargs)
        _op.set_cost(result.size if hasattr(result, "size") else 1)
    return result


attach_docstring(choose, _np.choose, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def column_stack(tup: Sequence[ArrayLike]) -> FlopscopeArray:
    """Stack 1-D arrays as columns into 2-D array. Cost: numel(output)."""
    budget = require_budget()
    arr_list = [_np.asarray(a) for a in tup]
    cost = max(sum(a.size for a in arr_list), 1)
    groups = [a.symmetry if isinstance(a, SymmetricTensor) else None for a in tup]
    input_ndims = [a.ndim for a in arr_list]
    with budget.deduct(
        "column_stack",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=tuple(a.dtype for a in arr_list),
    ):
        result = _call_numpy(_np.column_stack, _to_base_ndarray_tree(tup))  # type: ignore[arg-type, call-overload]
    out_group = _st.transport_column_stack(
        groups,
        output_ndim=result.ndim,
        input_ndims=input_ndims,
    )
    if any(g is not None for g in groups) and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                g.axes if g.axes is not None else tuple(range(g.degree))
                for g in groups
                if g is not None
            ],
            reason="column_stack breaks block symmetry",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(
    column_stack, _np.column_stack, "counted_custom", "numel(output) FLOPs"
)


def common_type(*args, **kwargs):
    """Return scalar type common to input arrays. Wraps ``numpy.common_type``. Cost: 0 FLOPs."""
    return _np.common_type(*[_to_base_ndarray(a) for a in args], **kwargs)


attach_docstring(common_type, _np.common_type, "free", "0 FLOPs")


@_counted_wrapper
def compress(
    condition: ArrayLike,
    a: ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return selected slices along an axis.

    Cost: ``len(condition) + 4 * numel(output)`` at weight 1.0.
    Mirrors ``extract``: scans ``len(condition)`` flags (1 FLOP each) then
    copies each selected element at gather-tier cost (4 FLOPs each).
    """
    budget = require_budget()
    _warn_if_symmetric(a, "compress")
    condition_arr = _np.asarray(condition)
    cond_len = condition_arr.size
    a_arr = _np.asarray(a)
    with budget.deduct_after(
        "compress", subscripts=None, shapes=(), dtypes=(a_arr.dtype,)
    ) as _op:
        result = _call_numpy(
            _np.compress,
            _to_base_ndarray(condition),  # type: ignore[arg-type]
            _to_base_ndarray(a),
            *args,
            **kwargs,
        )
        out_size = (
            result.size
            if hasattr(result, "size")
            else len(result)
            if hasattr(result, "__len__")
            else 1
        )
        _op.set_cost(cond_len + 4 * out_size)
    return result


attach_docstring(
    compress, _np.compress, "counted_custom", "len(condition) + 4*numel(output) FLOPs"
)


@_counted_wrapper
def concat(
    arrays: Sequence[ArrayLike],
    axis: int | None = 0,
    **kwargs: Any,
) -> FlopscopeArray:
    """Join arrays along an axis. Cost: numel(output)."""
    budget = require_budget()
    _concat_dtypes = tuple(_np.asarray(a).dtype for a in arrays)
    with budget.deduct_after(
        "concat", subscripts=None, shapes=(), dtypes=_concat_dtypes
    ) as _op:
        result = _call_numpy(
            _np.concat, _to_base_ndarray_tree(arrays), axis=axis, **kwargs
        )  # type: ignore[arg-type, call-overload]
        _op.set_cost(result.size if hasattr(result, "size") else 1)
    return result  # type: ignore[return-value]


attach_docstring(concat, _np.concat, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def copyto(dst, src, casting="same_kind", where=True):
    """Copy values from src to dst. Cost: one unit per element written (popcount of
    where= when masked)."""
    budget = require_budget()
    dst_arr = _np.asarray(dst)
    src_arr = _np.asarray(src)
    if where is True:
        cost = dst_arr.size
    else:
        where_arr = _np.asarray(where)
        try:
            cost = int(_np.count_nonzero(_np.broadcast_to(where_arr, dst_arr.shape)))
        except ValueError:
            cost = int(_np.count_nonzero(where_arr))
    with budget.deduct(
        "copyto",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(src_arr.dtype, dst_arr.dtype),
    ):
        result = _call_numpy(
            _np.copyto,
            _to_base_ndarray(dst),
            _to_base_ndarray(src),
            casting=casting,  # type: ignore[arg-type, call-overload]
            where=_to_base_ndarray(where) if where is not True else where,
        )
    return result


attach_docstring(
    copyto,
    _np.copyto,
    "counted_custom",
    "one unit per element written (popcount of where= when masked)",
)


@_counted_wrapper
def delete(
    arr: ArrayLike,
    obj: Any,
    axis: int | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return new array with sub-arrays deleted. Cost: numel(output) (np.delete copies the surviving elements)."""
    budget = require_budget()
    _warn_if_symmetric(arr, "delete")
    arr_arr = _np.asarray(arr)
    with budget.deduct_after(
        "delete", subscripts=None, shapes=(), dtypes=(arr_arr.dtype,)
    ) as _op:
        result = _call_numpy(
            _np.delete, _to_base_ndarray(arr), obj, axis=axis, **kwargs
        )
        _op.set_cost(int(result.size))
    return result  # type: ignore[return-value]


attach_docstring(delete, _np.delete, "counted_custom", "numel(output) FLOPs")


def diag_indices(*args, **kwargs):
    """Return indices to access main diagonal. Wraps ``numpy.diag_indices``. Cost: 0 FLOPs."""
    return _np.diag_indices(*args, **kwargs)


attach_docstring(diag_indices, _np.diag_indices, "free", "0 FLOPs")


def diag_indices_from(*args, **kwargs):
    """Return indices to access main diagonal of array. Wraps ``numpy.diag_indices_from``. Cost: 0 FLOPs."""
    return _np.diag_indices_from(*args, **kwargs)


attach_docstring(diag_indices_from, _np.diag_indices_from, "free", "0 FLOPs")


@_counted_wrapper
def diagflat(v: ArrayLike, k: int = 0) -> FlopscopeArray:
    """Create diagonal array from flattened input. Cost: numel(output)."""
    budget = require_budget()
    v_arr = _np.asarray(v)
    with budget.deduct_after(
        "diagflat", subscripts=None, shapes=(v_arr.shape,), dtypes=(v_arr.dtype,)
    ) as _op:
        result = _call_numpy(_np.diagflat, _to_base_ndarray(v), k=k)
        _op.set_cost(result.size)
    symmetry = _infer_structural_constructor_symmetry(
        kind="diagflat", k=k, v_ndim=v_arr.ndim
    )
    if symmetry is not None:
        return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]
    return result  # type: ignore[return-value]


attach_docstring(diagflat, _np.diagflat, "free", "0 FLOPs")


@_counted_wrapper
def dsplit(ary: ArrayLike, *args: Any, **kwargs: Any) -> list[FlopscopeArray]:
    """Split array along third axis. Cost: numel(input)."""
    budget = require_budget()
    ary_arr = _np.asarray(ary)
    cost = ary_arr.size
    in_group = ary.symmetry if isinstance(ary, SymmetricTensor) else None
    out_group = _st.transport_dsplit(in_group, input_shape=ary_arr.shape)
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason="dsplit breaks block symmetry",
        )
    with budget.deduct(
        "dsplit",
        flop_cost=cost,
        subscripts=None,
        shapes=(ary_arr.shape,),
        dtypes=(ary_arr.dtype,),
    ):
        raw_pieces = _call_numpy(_np.dsplit, ary_arr, *args, **kwargs)
    if out_group is not None:
        return [wrap_with_symmetry(p, out_group) for p in raw_pieces]  # type: ignore[return-value]
    return [_asplainflopscope(p) for p in raw_pieces]  # type: ignore[return-value]


attach_docstring(dsplit, _np.dsplit, "free", "0 FLOPs")


@_counted_wrapper
def dstack(tup: Sequence[ArrayLike]) -> FlopscopeArray:
    """Stack arrays along third axis. Cost: numel(output)."""
    budget = require_budget()
    for a in tup:
        _warn_if_symmetric(a, "dstack")
    _dstack_dtypes = tuple(_np.asarray(a).dtype for a in tup)
    with budget.deduct_after(
        "dstack", subscripts=None, shapes=(), dtypes=_dstack_dtypes
    ) as _op:
        result = _call_numpy(_np.dstack, _to_base_ndarray_tree(tup))  # type: ignore[arg-type]
        _op.set_cost(result.size if hasattr(result, "size") else 1)
    return result  # type: ignore[return-value]


attach_docstring(dstack, _np.dstack, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def extract(
    condition: ArrayLike,
    arr: ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return elements satisfying condition. Cost: numel(input)."""
    budget = require_budget()
    arr_np = _np.asarray(arr)
    cost = arr_np.size
    with budget.deduct(
        "extract",
        flop_cost=cost,
        subscripts=None,
        shapes=(arr_np.shape,),
        dtypes=(arr_np.dtype,),
    ):
        result = _call_numpy(
            _np.extract,
            _to_base_ndarray(condition),
            _to_base_ndarray(arr),
            *args,
            **kwargs,
        )
    return result  # type: ignore[return-value]


attach_docstring(extract, _np.extract, "counted_custom", "numel(input) FLOPs")


@_counted_wrapper
def fill_diagonal(
    a: ArrayLike,
    val: Any,
    wrap: bool = False,
    **kwargs: Any,
) -> None:
    """Fill main diagonal of array in-place. Cost: min(m,n)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = min(a_arr.shape[0], a_arr.shape[1]) if a_arr.ndim >= 2 else a_arr.size
    with budget.deduct(
        "fill_diagonal",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_arr.shape,),
        dtypes=(a_arr.dtype,),
    ):
        # ``np.fill_diagonal`` mutates ``a`` in-place; ``_to_base_ndarray``
        # is zero-copy so the mutation propagates to the user's array.
        result = _call_numpy(
            _np.fill_diagonal, _to_base_ndarray(a), val, wrap=wrap, **kwargs
        )  # type: ignore[arg-type, call-overload]
    return result


attach_docstring(fill_diagonal, _np.fill_diagonal, "counted_custom", "min(m,n) FLOPs")


@_counted_wrapper
def flatnonzero(a: ArrayLike, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Return indices of non-zero elements in flattened array. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    with budget.deduct(
        "flatnonzero",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_arr.shape,),
        dtypes=(a_arr.dtype,),
    ):
        result = _call_numpy(_np.flatnonzero, _to_base_ndarray(a), *args, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(flatnonzero, _np.flatnonzero, "counted_custom", "numel(input) FLOPs")


@_counted_wrapper
def fliplr(*args, **kwargs):
    """Reverse elements along axis 1. Wraps ``numpy.fliplr``. Cost: 0 FLOPs."""
    budget = require_budget()
    _warn_if_symmetric(args[0], "fliplr")
    a_arr = _np.asarray(args[0])
    with budget.deduct(
        "fliplr", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
    ):
        result = _call_numpy(_np.fliplr, *_to_base_ndarray_tree(args), **kwargs)
    return result


attach_docstring(fliplr, _np.fliplr, "free", "0 FLOPs")


@_counted_wrapper
def flipud(*args, **kwargs):
    """Reverse elements along axis 0. Wraps ``numpy.flipud``. Cost: 0 FLOPs."""
    budget = require_budget()
    _warn_if_symmetric(args[0], "flipud")
    a_arr = _np.asarray(args[0])
    with budget.deduct(
        "flipud", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
    ):
        result = _call_numpy(_np.flipud, *_to_base_ndarray_tree(args), **kwargs)
    return result


attach_docstring(flipud, _np.flipud, "free", "0 FLOPs")


@_counted_wrapper
def from_dlpack(*args, **kwargs):
    """Create array from DLPack capsule. Cost: numel(output)."""
    budget = require_budget()
    result = _np.from_dlpack(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "from_dlpack",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(result.dtype,) if hasattr(result, "dtype") else (),
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(from_dlpack, _np.from_dlpack, "free", "0 FLOPs")


@_counted_wrapper
def frombuffer(
    buffer: Any,
    dtype: DTypeLike = float,
    count: int = -1,
    offset: int = 0,
) -> FlopscopeArray:
    """Interpret buffer as 1-D array. Cost: numel(output)."""
    budget = require_budget()
    result = _np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "frombuffer",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(result.dtype,) if hasattr(result, "dtype") else (),
    ):
        pass  # numpy call already executed above
    return result  # type: ignore[return-value]


attach_docstring(frombuffer, _np.frombuffer, "free", "0 FLOPs")


@_counted_wrapper
def fromfile(*args, **kwargs):
    """Construct array from data in text or binary file. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromfile(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "fromfile",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(result.dtype,) if hasattr(result, "dtype") else (),
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(fromfile, _np.fromfile, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def fromfunction(*args, **kwargs):
    """Construct array by executing function over each coordinate. Cost: numel(output)."""
    _warn_remote_callback("fromfunction")
    budget = require_budget()
    result = _call_user_code(budget, _np.fromfunction, *args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "fromfunction",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(result.dtype,) if hasattr(result, "dtype") else (),
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(
    fromfunction, _np.fromfunction, "counted_custom", "numel(output) FLOPs"
)


@_counted_wrapper
def fromiter(*args, **kwargs):
    """Create array from iterable object. Cost: numel(output) at weight 1.0.

    Iterator materialisation with no libm calls; weight 1.0 matches other
    materialisation ops (``array``, ``concatenate``, etc.).
    """
    _warn_remote_callback("fromiter")
    budget = require_budget()
    result = _call_user_code(budget, _np.fromiter, *args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "fromiter",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(result.dtype,) if hasattr(result, "dtype") else (),
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(fromiter, _np.fromiter, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def fromregex(*args, **kwargs):
    """Construct array from text file using regex. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromregex(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "fromregex",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(result.dtype,) if hasattr(result, "dtype") else (),
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(fromregex, _np.fromregex, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def fromstring(*args, **kwargs):
    """Construct array from string. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromstring(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "fromstring",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(result.dtype,) if hasattr(result, "dtype") else (),
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(fromstring, _np.fromstring, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def indices(*args: Any, **kwargs: Any) -> FlopscopeArray:
    """Return array representing indices of a grid. Cost: numel of materialized output FLOPs (dense N*prod(dims); sparse sum(dims))."""
    budget = require_budget()
    _indices_dtype = _np.dtype(kwargs.get("dtype", int))
    with budget.deduct_after(
        "indices", subscripts=None, shapes=(), dtypes=(_indices_dtype,)
    ) as _op:
        result = _call_numpy(_np.indices, *args, **kwargs)
        _op.set_cost(
            sum(int(a.size) for a in result)
            if isinstance(result, tuple)
            else int(result.size)
        )
    return result  # pyright: ignore[reportReturnType]  # tuple of grids when sparse=True


attach_docstring(
    indices,
    _np.indices,
    "counted_custom",
    "numel of materialized output FLOPs (dense N*prod(dims); sparse sum(dims))",
)


@_counted_wrapper
def insert(
    arr: ArrayLike,
    obj: Any,
    values: ArrayLike,
    axis: int | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Insert values along axis before given indices. Cost: numel(output)."""
    budget = require_budget()
    _insert_dtypes = (_np.asarray(arr).dtype, _np.asarray(values).dtype)
    with budget.deduct_after(
        "insert", subscripts=None, shapes=(), dtypes=_insert_dtypes
    ) as _op:
        result = _call_numpy(
            _np.insert,
            _to_base_ndarray(arr),
            obj,
            _to_base_ndarray(values),
            axis=axis,
            **kwargs,
        )
        _op.set_cost(int(result.size))
    return result  # type: ignore[return-value]


attach_docstring(insert, _np.insert, "counted_custom", "numel(output) FLOPs")


def isdtype(*args, **kwargs):
    """Returns boolean indicating whether a provided dtype is of a specified kind. Wraps ``numpy.isdtype``. Cost: 0 FLOPs."""
    return _np.isdtype(*args, **kwargs)


attach_docstring(isdtype, _np.isdtype, "free", "0 FLOPs")


def isfortran(*args, **kwargs):
    """Returns True if array is Fortran contiguous. Wraps ``numpy.isfortran``. Cost: 0 FLOPs."""
    return _np.isfortran(*args, **kwargs)


attach_docstring(isfortran, _np.isfortran, "free", "0 FLOPs")


def isin(
    element: ArrayLike,
    test_elements: ArrayLike,
    assume_unique: bool = False,
    invert: bool = False,
) -> FlopscopeArray:
    """Test element-wise membership in a set. Wraps ``numpy.isin``. Cost: 0 FLOPs."""
    return _np.isin(  # type: ignore[return-value]
        _to_base_ndarray(element),
        _to_base_ndarray(test_elements),
        assume_unique=assume_unique,
        invert=invert,
    )


attach_docstring(isin, _np.isin, "counted_custom", "(n+m)*ceil(log2(n+m)) FLOPs")


def isscalar(*args, **kwargs):
    """Returns True if element is scalar type. Wraps ``numpy.isscalar``. Cost: 0 FLOPs."""
    return _np.isscalar(*args, **kwargs)


attach_docstring(isscalar, _np.isscalar, "free", "0 FLOPs")


def issubdtype(*args, **kwargs):
    """Returns True if first argument is a typecode lower/equal in type hierarchy. Wraps ``numpy.issubdtype``. Cost: 0 FLOPs."""
    return _np.issubdtype(*args, **kwargs)


attach_docstring(issubdtype, _np.issubdtype, "free", "0 FLOPs")


def iterable(*args, **kwargs):
    """Check whether or not object is iterable. Wraps ``numpy.iterable``. Cost: 0 FLOPs."""
    return _np.iterable(*args, **kwargs)


attach_docstring(iterable, _np.iterable, "free", "0 FLOPs")


@_counted_wrapper
def ix_(*args: ArrayLike, **kwargs: Any) -> tuple[FlopscopeArray, ...]:
    """Construct open mesh from multiple sequences. Cost: numel(output)."""
    budget = require_budget()
    stripped_args = _to_base_ndarray_tree(args)
    result = _np.ix_(*stripped_args, **kwargs)  # type: ignore[arg-type, call-overload]
    cost = sum(a.size for a in result)
    _ix_dtypes = tuple(a.dtype for a in result) if result else (_np.dtype(float),)
    with budget.deduct(
        "ix_", flop_cost=cost, subscripts=None, shapes=(), dtypes=_ix_dtypes
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(ix_, _np.ix_, "free", "0 FLOPs")


@_counted_wrapper
def mask_indices(*args, **kwargs):
    """Return indices to access main or off-diagonal of array.

    Cost: ``2*n^2 + 8*k`` at weight 1.0, where *n* is the matrix dimension and
    *k* is the number of selected index pairs (= len of each returned array).

    Formula breakdown:
    - ``2*n^2``: scan of the ``n×n`` boolean mask (mask_func allocates an ones
      matrix and applies the mask; 1 FLOP/cell × 2 for the boolean eval pass).
    - ``8*k``: gather of 2k index values at gather-tier cost (4 FLOPs each).
    """
    budget = require_budget()
    # n is first positional arg; extract before calling numpy
    n = args[0] if args else kwargs.get("n", 0)
    result = _np.mask_indices(*args, **kwargs)
    k = result[0].size if isinstance(result, tuple) and result else 0
    cost = 2 * int(n) * int(n) + 8 * int(k)
    _mask_dtypes = (
        tuple(a.dtype for a in result)
        if isinstance(result, tuple) and result
        else (_np.dtype(int),)
    )
    with budget.deduct(
        "mask_indices",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=_mask_dtypes,
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(mask_indices, _np.mask_indices, "counted_custom", "2*n^2 + 8*k FLOPs")


def matrix_transpose(x: ArrayLike) -> FlopscopeArray:
    """Swap last two axes. Wraps ``numpy.matrix_transpose``. Cost: 0 FLOPs."""
    x_arr = _np.asarray(x)
    result = _np.matrix_transpose(x_arr)
    in_group = x.symmetry if isinstance(x, SymmetricTensor) else None
    out_group = _st.transport_matrix_transpose(in_group, ndim=x_arr.ndim)
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[
                in_group.axes
                if in_group.axes is not None
                else tuple(range(in_group.degree))
            ],
            reason="matrix_transpose: rank too low for sym",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(matrix_transpose, _np.matrix_transpose, "free", "0 FLOPs")


def may_share_memory(*args, **kwargs):
    """Determine if two arrays might share memory. Wraps ``numpy.may_share_memory``. Cost: 0 FLOPs."""
    stripped_args = tuple(_to_base_ndarray(a) for a in args)
    return _np.may_share_memory(*stripped_args, **kwargs)  # type: ignore[arg-type, call-overload]


attach_docstring(may_share_memory, _np.may_share_memory, "free", "0 FLOPs")


def min_scalar_type(*args, **kwargs):
    """Return smallest scalar type. Wraps ``numpy.min_scalar_type``. Cost: 0 FLOPs."""
    return _np.min_scalar_type(*args, **kwargs)


attach_docstring(min_scalar_type, _np.min_scalar_type, "free", "0 FLOPs")


def mintypecode(*args, **kwargs):
    """Return minimum data type character. Wraps ``numpy.mintypecode``. Cost: 0 FLOPs."""
    return _np.mintypecode(*args, **kwargs)


attach_docstring(mintypecode, _np.mintypecode, "free", "0 FLOPs")


def ndim(*args, **kwargs):
    """Return number of dimensions. Wraps ``numpy.ndim``. Cost: 0 FLOPs."""
    return _np.ndim(*args, **kwargs)


attach_docstring(ndim, _np.ndim, "free", "0 FLOPs")


@_counted_wrapper
def nonzero(a: ArrayLike, *args: Any, **kwargs: Any) -> tuple[FlopscopeArray, ...]:
    """Return indices of non-zero elements. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    with budget.deduct(
        "nonzero",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_arr.shape,),
        dtypes=(a_arr.dtype,),
    ):
        result = _call_numpy(_np.nonzero, _to_base_ndarray(a), *args, **kwargs)  # type: ignore[arg-type, call-overload]
    return result  # type: ignore[return-value]


attach_docstring(nonzero, _np.nonzero, "counted_custom", "numel(input) FLOPs")


@_counted_wrapper
def packbits(a: ArrayLike, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Pack binary-valued array into bits. Cost: numel(input) at weight 1.0.

    Each input bit is tested and shifted, so cost is proportional to the number
    of input elements (symmetric with ``unpackbits`` which charges 8 × numel(output)).
    """
    budget = require_budget()
    a_arr = _np.asarray(a)
    in_size = a_arr.size
    with budget.deduct_after(
        "packbits", subscripts=None, shapes=(), dtypes=(a_arr.dtype,)
    ) as _op:
        result = _call_numpy(_np.packbits, _to_base_ndarray(a), *args, **kwargs)  # type: ignore[arg-type, call-overload]
        _op.set_cost(in_size)
    return result  # type: ignore[return-value]


attach_docstring(packbits, _np.packbits, "counted_custom", "numel(input) FLOPs")


def permute_dims(*args, **kwargs):
    """Permute dimensions of array. Wraps ``numpy.permute_dims``. Cost: 0 FLOPs."""
    stripped_args = _to_base_ndarray_tree(args)
    return _np.permute_dims(*stripped_args, **kwargs)


attach_docstring(permute_dims, _np.permute_dims, "free", "0 FLOPs")


@_counted_wrapper
def place(
    arr: ArrayLike,
    mask: ArrayLike,
    vals: ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Change elements of array based on conditional. Cost: numel(input)."""
    budget = require_budget()
    arr_np = _np.asarray(arr)
    cost = arr_np.size
    with budget.deduct(
        "place",
        flop_cost=cost,
        subscripts=None,
        shapes=(arr_np.shape,),
        dtypes=(arr_np.dtype,),
    ):
        # ``np.place`` mutates ``arr`` in-place; ``_to_base_ndarray`` is
        # zero-copy so the mutation propagates to the user's array.
        result = _call_numpy(
            _np.place,
            _to_base_ndarray(arr),  # type: ignore[arg-type, call-overload]
            _to_base_ndarray(mask),
            _to_base_ndarray(vals),
            *args,
            **kwargs,
        )
    return result


attach_docstring(place, _np.place, "counted_custom", "numel(input) FLOPs")


def promote_types(*args, **kwargs):
    """Return smallest size and least significant type. Wraps ``numpy.promote_types``. Cost: 0 FLOPs."""
    return _np.promote_types(*args, **kwargs)


attach_docstring(promote_types, _np.promote_types, "free", "0 FLOPs")


@_counted_wrapper
def put(
    a: ArrayLike,
    ind: ArrayLike,
    v: ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Replace elements at given flat indices. Cost: numel(indices)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    ind_arr = _np.asarray(ind)
    cost = ind_arr.size  # number of scatter writes; mode-independent
    with budget.deduct(
        "put",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_arr.shape,),
        dtypes=(a_arr.dtype,),
    ):
        # ``np.put`` mutates ``a`` in-place. ``_to_base_ndarray`` is a
        # zero-copy view, so the mutation propagates to the user's
        # original FlopscopeArray buffer.
        result = _call_numpy(
            _np.put,
            _to_base_ndarray(a),  # type: ignore[arg-type, call-overload]
            _to_base_ndarray(ind),  # type: ignore[arg-type, call-overload]
            _to_base_ndarray(v),
            *args,
            **kwargs,
        )
    return result


attach_docstring(put, _np.put, "counted_custom", "numel(indices) FLOPs")


@_counted_wrapper
def put_along_axis(
    arr: ArrayLike,
    indices: ArrayLike,
    values: ArrayLike,
    axis: int | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Put values into destination array along axis. Cost: elements scattered = (numel(arr) / arr.shape[axis]) x indices.shape[axis] (indices.size when axis=None)."""
    budget = require_budget()
    arr_np = _np.asarray(arr)
    idx_np = _np.asarray(indices)
    if axis is None:
        cost = int(idx_np.size)
    elif arr_np.size == 0:
        cost = 0
    else:
        cost = (arr_np.size // arr_np.shape[axis]) * int(idx_np.shape[axis])
    with budget.deduct(
        "put_along_axis",
        flop_cost=cost,
        subscripts=None,
        shapes=(arr_np.shape,),
        dtypes=(arr_np.dtype,),
    ):
        # ``np.put_along_axis`` mutates ``arr`` in-place; ``_to_base_ndarray``
        # is zero-copy so the mutation propagates to the user's array.
        result = _call_numpy(
            _np.put_along_axis,
            _to_base_ndarray(arr),  # type: ignore[arg-type, call-overload]
            _to_base_ndarray(indices),  # type: ignore[arg-type, call-overload]
            _to_base_ndarray(values),
            axis,
            *args,
            **kwargs,
        )
    return result


attach_docstring(
    put_along_axis, _np.put_along_axis, "counted_custom", "elements scattered FLOPs"
)


@_counted_wrapper
def putmask(
    a: ArrayLike,
    mask: ArrayLike,
    values: ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Change elements of array based on condition. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    with budget.deduct(
        "putmask",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_arr.shape,),
        dtypes=(a_arr.dtype,),
    ):
        result = _call_numpy(
            _np.putmask,
            _to_base_ndarray(a),  # type: ignore[arg-type, call-overload]
            _to_base_ndarray(mask),  # type: ignore[arg-type, call-overload]
            _to_base_ndarray(values),
            *args,
            **kwargs,
        )
    return result


attach_docstring(putmask, _np.putmask, "counted_custom", "numel(input) FLOPs")


@_counted_wrapper
def ravel_multi_index(multi_index, dims, mode="raise", order="C"):
    """Convert a multi-index to flat indices. Cost: 2*(ndim-1)*N (one stride is unity),
    plus N for mode in {'clip','wrap'} (one clamp/mod per element). N = #output indices."""
    budget = require_budget()
    stripped = _to_base_ndarray_tree(multi_index)
    idx_arrays = [_np.asarray(a) for a in stripped]
    n = int(_np.broadcast(*idx_arrays).size) if idx_arrays else 0
    ndim = len(dims) if hasattr(dims, "__len__") else 1
    cost = 2 * (ndim - 1) * n
    if mode != "raise":
        cost += n
    _rmi_dtypes = (
        tuple(a.dtype for a in idx_arrays) if idx_arrays else (_np.dtype(int),)
    )
    with budget.deduct(
        "ravel_multi_index",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=_rmi_dtypes,
    ):
        result = _call_numpy(  # type: ignore[arg-type, call-overload]
            _np.ravel_multi_index, stripped, dims, mode=mode, order=order
        )
    return result


attach_docstring(
    ravel_multi_index,
    _np.ravel_multi_index,
    "counted_custom",
    "2*(ndim-1)*N (+N for clip/wrap)",
)


@_counted_wrapper
def require(*args: Any, **kwargs: Any) -> FlopscopeArray:
    """Return array satisfying requirements. Wraps ``numpy.require``. Cost: numel(input)."""
    # Pass args through unstripped: ``_np.require`` is a thin Python
    # helper around ``np.asanyarray`` and does not enter the
    # ``__array_function__`` dispatch path, so passing a FlopscopeArray
    # cannot recurse. Stripping would break ``np.require(x).is(x)``
    # identity for already-conforming inputs. The cost peek below uses a
    # separate ``_np.asarray()`` call rather than the stripped value, so the
    # actual ``_np.require`` call still receives the original, unstripped args.
    budget = require_budget()
    a = args[0] if args else kwargs.get("a")
    a_arr = _np.asarray(a)
    cost = max(a_arr.size, 1)
    # numpy.require(a, dtype=...) casts/materializes at the REQUESTED dtype,
    # so bill that width when given (mirrors full_like), else the input's.
    # numpy signature: require(a, dtype=None, requirements=None, *, like=None)
    # -- dtype is the second positional or the ``dtype=`` kwarg.
    _dtype = args[1] if len(args) > 1 else kwargs.get("dtype")
    _billing_dtype = _np.dtype(_dtype) if _dtype is not None else a_arr.dtype
    with budget.deduct(
        "require",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(_billing_dtype,),
    ):
        result = _call_numpy(_np.require, *args, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(
    require, _np.require, "counted_custom", "numel(input) FLOPs (even if unchanged)"
)


@_counted_wrapper
def resize(*args, **kwargs):
    """Return new array with given shape. Cost: numel(output)."""
    budget = require_budget()
    stripped_args = _to_base_ndarray_tree(args)
    _resize_dtype = _np.asarray(args[0]).dtype if args else _np.dtype(float)
    with budget.deduct_after(
        "resize", subscripts=None, shapes=(), dtypes=(_resize_dtype,)
    ) as _op:
        result = _call_numpy(_np.resize, *stripped_args, **kwargs)
        _op.set_cost(result.size if hasattr(result, "size") else 1)
    return result


attach_docstring(resize, _np.resize, "counted_custom", "numel(output) FLOPs")


def result_type(*args, **kwargs):
    """Returns type that results from applying type promotion. Wraps ``numpy.result_type``. Cost: 0 FLOPs."""
    return _np.result_type(*args, **kwargs)


attach_docstring(result_type, _np.result_type, "free", "0 FLOPs")


@_counted_wrapper
def rollaxis(*args, **kwargs):
    """Roll specified axis backwards. Cost: numel(output)."""
    budget = require_budget()
    stripped_args = _to_base_ndarray_tree(args)
    result = _np.rollaxis(*stripped_args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        "rollaxis",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(result.dtype,) if hasattr(result, "dtype") else (),
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(rollaxis, _np.rollaxis, "free", "0 FLOPs")


@_counted_wrapper
def rot90(*args, **kwargs):
    """Rotate array 90 degrees. Wraps ``numpy.rot90``. Cost: 0 FLOPs."""
    budget = require_budget()
    a_arr = _np.asarray(args[0])
    with budget.deduct(
        "rot90", flop_cost=0, subscripts=None, shapes=(a_arr.shape,), dtypes=()
    ):
        result = _call_numpy(_np.rot90, *_to_base_ndarray_tree(args), **kwargs)
    return result


attach_docstring(rot90, _np.rot90, "free", "0 FLOPs")


def row_stack(tup: Sequence[ArrayLike]) -> FlopscopeArray:
    """Stack arrays vertically. Cost: numel(output) (alias for vstack; no
    ``deduct()`` of its own -- billed under vstack's op_name)."""
    return vstack(tup)


attach_docstring(
    row_stack,
    _np.row_stack,
    "counted_custom",
    "numel(output) FLOPs (billed under vstack)",
)


@_counted_wrapper
def select(
    condlist: Sequence[ArrayLike],
    choicelist: Sequence[ArrayLike],
    default: Any = 0,
) -> FlopscopeArray:
    """Return array drawn from elements depending on conditions.

    Cost: numel(output) — the true broadcast size of the result.
    Weight tier: gather (×4.0 from the packaged table).
    """
    budget = require_budget()
    _select_dtypes = tuple(_np.asarray(c).dtype for c in choicelist)
    with budget.deduct_after(
        "select", subscripts=None, shapes=(), dtypes=_select_dtypes
    ) as _op:
        result = _call_numpy(
            _np.select,
            _to_base_ndarray_tree(condlist),  # type: ignore[arg-type]
            _to_base_ndarray_tree(choicelist),  # type: ignore[arg-type]
            default=default,
        )
        _op.set_cost(result.size if hasattr(result, "size") else 1)
    return result  # type: ignore[return-value]


attach_docstring(
    select,
    _np.select,
    "counted_custom",
    "numel(output) FLOPs (Cost: numel(output), gather tier ×4)",
)


def shape(*args, **kwargs):
    """Return shape of array. Wraps ``numpy.shape``. Cost: 0 FLOPs."""
    return _np.shape(*args, **kwargs)


attach_docstring(shape, _np.shape, "free", "0 FLOPs")


def shares_memory(*args, **kwargs):
    """Determine if two arrays share memory. Wraps ``numpy.shares_memory``. Cost: 0 FLOPs."""
    return _np.shares_memory(*[_to_base_ndarray(a) for a in args], **kwargs)  # type: ignore[arg-type]


attach_docstring(shares_memory, _np.shares_memory, "free", "0 FLOPs")


def size(*args, **kwargs):
    """Return number of elements along a given axis. Wraps ``numpy.size``. Cost: 0 FLOPs."""
    return _np.size(*args, **kwargs)


attach_docstring(size, _np.size, "free", "0 FLOPs")


@_counted_wrapper
def take(
    a: ArrayLike,
    indices: ArrayLike,
    axis: int | None = None,
    out: FlopscopeArray | None = None,
    mode: str = "raise",
) -> FlopscopeArray:
    """Take elements from array along axis. Cost: numel(output)."""
    budget = require_budget()
    _a_arr = _np.asarray(a)
    with budget.deduct_after(
        "take", subscripts=None, shapes=(), dtypes=(_a_arr.dtype,)
    ) as _op:
        result = _call_numpy(
            _np.take,
            _to_base_ndarray(a),
            _to_base_ndarray(indices),  # type: ignore[arg-type]
            axis=axis,
            out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
        )
        _op.set_cost(result.size if hasattr(result, "size") else 1)
    return result  # type: ignore[return-value]


attach_docstring(take, _np.take, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def take_along_axis(
    arr: ArrayLike,
    indices: ArrayLike,
    axis: int | None,
) -> FlopscopeArray:
    """Take values from input array along axis using indices.

    Cost: numel(output) at weight 4.0 (gather tier; identical work to ``take``).
    Each output element requires an index dereference into the source array.
    """
    budget = require_budget()
    _arr_probe = _np.asarray(arr)
    with budget.deduct_after(
        "take_along_axis", subscripts=None, shapes=(), dtypes=(_arr_probe.dtype,)
    ) as _op:
        result = _call_numpy(
            _np.take_along_axis,
            _to_base_ndarray(arr),  # type: ignore[arg-type]
            _to_base_ndarray(indices),  # type: ignore[arg-type]
            axis=axis,
        )
        _op.set_cost(result.size if hasattr(result, "size") else 1)
    return result  # type: ignore[return-value]


attach_docstring(
    take_along_axis, _np.take_along_axis, "counted_custom", "numel(output) FLOPs"
)


@_counted_wrapper
def tri(*args, **kwargs):
    """Array with ones at and below the given diagonal. Wraps ``numpy.tri``. Cost: 0 FLOPs."""
    budget = require_budget()
    with budget.deduct("tri", flop_cost=0, subscripts=None, shapes=(), dtypes=()):
        result = _call_numpy(_np.tri, *args, **kwargs)
    # A triangular matrix is not symmetric — do NOT infer constant-fill symmetry.
    return _asplainflopscope(result)


attach_docstring(tri, _np.tri, "free", "0 FLOPs")


def tril_indices(*args, **kwargs):
    """Return indices for lower-triangle of array. Wraps ``numpy.tril_indices``. Cost: 0 FLOPs."""
    return _np.tril_indices(*args, **kwargs)


attach_docstring(tril_indices, _np.tril_indices, "free", "0 FLOPs")


def tril_indices_from(*args, **kwargs):
    """Return indices for lower-triangle of given array. Wraps ``numpy.tril_indices_from``. Cost: 0 FLOPs."""
    return _np.tril_indices_from(*args, **kwargs)


attach_docstring(tril_indices_from, _np.tril_indices_from, "free", "0 FLOPs")


@_counted_wrapper
def trim_zeros(filt: ArrayLike, trim: str = "fb", **kwargs: Any) -> FlopscopeArray:
    """Trim leading/trailing zeros. Cost: numel(input) (value scan for the nonzero
    boundary, same convention as nonzero/count_nonzero)."""
    budget = require_budget()
    filt_arr = _np.asarray(filt)
    cost = int(filt_arr.size)
    with budget.deduct(
        "trim_zeros",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(filt_arr.dtype,),
    ):
        result = _call_numpy(
            _np.trim_zeros, _to_base_ndarray(filt), trim=trim, **kwargs
        )  # type: ignore[arg-type]
    return result


attach_docstring(
    trim_zeros, _np.trim_zeros, "counted_custom", "numel(input) (value scan)"
)


def triu_indices(*args, **kwargs):
    """Return indices for upper-triangle of array. Wraps ``numpy.triu_indices``. Cost: 0 FLOPs."""
    return _np.triu_indices(*args, **kwargs)


attach_docstring(triu_indices, _np.triu_indices, "free", "0 FLOPs")


def triu_indices_from(*args, **kwargs):
    """Return indices for upper-triangle of given array. Wraps ``numpy.triu_indices_from``. Cost: 0 FLOPs."""
    return _np.triu_indices_from(*args, **kwargs)


attach_docstring(triu_indices_from, _np.triu_indices_from, "free", "0 FLOPs")


def typename(*args, **kwargs):
    """Return description for given data type code. Wraps ``numpy.typename``. Cost: 0 FLOPs."""
    return _np.typename(*args, **kwargs)


attach_docstring(typename, _np.typename, "free", "0 FLOPs")


@_counted_wrapper
def unpackbits(a: ArrayLike, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Unpack elements of uint8 array into binary-valued bit array. Cost: numel(output)."""
    budget = require_budget()
    _a_arr = _np.asarray(a)
    with budget.deduct_after(
        "unpackbits", subscripts=None, shapes=(), dtypes=(_a_arr.dtype,)
    ) as _op:
        result = _call_numpy(_np.unpackbits, _to_base_ndarray(a), *args, **kwargs)  # type: ignore[arg-type]
        _op.set_cost(
            result.size
            if hasattr(result, "size")
            else len(result)
            if hasattr(result, "__len__")
            else 1
        )
    return result  # type: ignore[return-value]


attach_docstring(unpackbits, _np.unpackbits, "counted_custom", "numel(output) FLOPs")


def unravel_index(*args, **kwargs):
    """Convert flat indices to multi-dimensional index. Wraps ``numpy.unravel_index``. Cost: 0 FLOPs."""
    return _np.unravel_index(*args, **kwargs)


attach_docstring(unravel_index, _np.unravel_index, "free", "0 FLOPs")


if hasattr(_np, "unstack"):

    @_counted_wrapper
    def unstack(x: ArrayLike, *args: Any, **kwargs: Any) -> tuple[FlopscopeArray, ...]:  # pyright: ignore[reportRedeclaration]
        """Split array into sequence of arrays along an axis. Cost: numel(input)."""
        budget = require_budget()
        x_arr = _np.asarray(x)
        cost = x_arr.size
        with budget.deduct(
            "unstack",
            flop_cost=cost,
            subscripts=None,
            shapes=(x_arr.shape,),
            dtypes=(x_arr.dtype,),
        ):
            result = _call_numpy(_np.unstack, _to_base_ndarray(x), *args, **kwargs)
        return result  # type: ignore[return-value]

    attach_docstring(unstack, _np.unstack, "free", "0 FLOPs")

else:

    def unstack(*args: Any, **kwargs: Any) -> tuple[FlopscopeArray, ...]:  # pyright: ignore[reportRedeclaration]
        raise UnsupportedFunctionError("unstack", min_version="2.1")


# ---------------------------------------------------------------------------
# Wrap all free op return values as FlopscopeArray
# ---------------------------------------------------------------------------

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_FREE_OPS_SKIP = {
    "shape",
    "size",
    "ndim",
    "isscalar",
    "isfortran",
    "isfinite",
    "isinf",
    "isnan",
    "isdtype",
    "issubdtype",
    "iscomplex",
    "iscomplexobj",
    "isnat",
    "isneginf",
    "isposinf",
    "isreal",
    "isrealobj",
    "iterable",
    "may_share_memory",
    "shares_memory",
    "can_cast",
    "common_type",
    "min_scalar_type",
    "promote_types",
    "result_type",
    "typename",
    "base_repr",
    "binary_repr",
    "broadcast_shapes",
    "fill_diagonal",
}

import sys as _sys  # noqa: E402

# ---------------------------------------------------------------------------
# Signature conformance: set __signature__ to match numpy exactly
# ---------------------------------------------------------------------------
_this_module = _sys.modules[__name__]


def _set_sig(func_name, np_func):
    """Set __signature__ of a module-level function to match numpy."""
    if isinstance(np_func, _np.ufunc):
        return
    fn = globals().get(func_name)
    if fn is not None and callable(np_func):
        try:
            fn.__signature__ = _inspect.signature(np_func)  # pyright: ignore[reportFunctionMemberAccess]
        except (ValueError, TypeError):
            pass


# Functions with *args/**kwargs that need numpy's signature
_set_sig("arange", _np.arange)
_set_sig("array", _np.array)
_set_sig("zeros", _np.zeros)
_set_sig("ones", _np.ones)
_set_sig("full", _np.full)
_set_sig("eye", _np.eye)
_set_sig("linspace", _np.linspace)
_set_sig("zeros_like", _np.zeros_like)
_set_sig("ones_like", _np.ones_like)
_set_sig("full_like", _np.full_like)
_set_sig("empty", _np.empty)
_set_sig("empty_like", _np.empty_like)
_set_sig("identity", _np.identity)
_set_sig("reshape", _np.reshape)
_set_sig("concatenate", _np.concatenate)
_set_sig("stack", _np.stack)
_set_sig("vstack", _np.vstack)
_set_sig("hstack", _np.hstack)
_set_sig("ravel", _np.ravel)
_set_sig("copy", _np.copy)
_set_sig("pad", _np.pad)
_set_sig("broadcast_to", _np.broadcast_to)
_set_sig("meshgrid", _np.meshgrid)
_set_sig("asarray", _np.asarray)
_set_sig("astype", _np.astype)
_set_sig("append", _np.append)
_set_sig("argwhere", _np.argwhere)
_set_sig("array_split", _np.array_split)
_set_sig("asarray_chkfinite", _np.asarray_chkfinite)
_set_sig("atleast_1d", _np.atleast_1d)
_set_sig("atleast_2d", _np.atleast_2d)
_set_sig("atleast_3d", _np.atleast_3d)
_set_sig("base_repr", _np.base_repr)
_set_sig("binary_repr", _np.binary_repr)
_set_sig("block", _np.block)
_set_sig("bmat", _np.bmat)
_set_sig("broadcast_arrays", _np.broadcast_arrays)
_set_sig("broadcast_shapes", _np.broadcast_shapes)
_set_sig("can_cast", _np.can_cast)
_set_sig("choose", _np.choose)
_set_sig("column_stack", _np.column_stack)
_set_sig("common_type", _np.common_type)
_set_sig("compress", _np.compress)
_set_sig("concat", _np.concat)
_set_sig("delete", _np.delete)
_set_sig("diag_indices", _np.diag_indices)
_set_sig("diag_indices_from", _np.diag_indices_from)
_set_sig("dsplit", _np.dsplit)
_set_sig("dstack", _np.dstack)
_set_sig("extract", _np.extract)
_set_sig("flatnonzero", _np.flatnonzero)
_set_sig("fliplr", _np.fliplr)
_set_sig("flipud", _np.flipud)
_set_sig("from_dlpack", _np.from_dlpack)
_set_sig("frombuffer", _np.frombuffer)
_set_sig("fromfile", _np.fromfile)
_set_sig("fromfunction", _np.fromfunction)
_set_sig("fromiter", _np.fromiter)
_set_sig("fromregex", _np.fromregex)
_set_sig("fromstring", _np.fromstring)
_set_sig("indices", _np.indices)
_set_sig("insert", _np.insert)
_set_sig("isdtype", _np.isdtype)
_set_sig("isfortran", _np.isfortran)
_set_sig("isin", _np.isin)
_set_sig("isnan", _np.isnan)
_set_sig("isfinite", _np.isfinite)
_set_sig("isinf", _np.isinf)
_set_sig("isscalar", _np.isscalar)
_set_sig("issubdtype", _np.issubdtype)
_set_sig("iterable", _np.iterable)
_set_sig("ix_", _np.ix_)
_set_sig("mask_indices", _np.mask_indices)
_set_sig("matrix_transpose", _np.matrix_transpose)
_set_sig("may_share_memory", _np.may_share_memory)
_set_sig("min_scalar_type", _np.min_scalar_type)
_set_sig("mintypecode", _np.mintypecode)
_set_sig("ndim", _np.ndim)
_set_sig("nonzero", _np.nonzero)
_set_sig("permute_dims", _np.permute_dims)
_set_sig("put", _np.put)
_set_sig("require", _np.require)
_set_sig("resize", _np.resize)
_set_sig("rollaxis", _np.rollaxis)
_set_sig("rot90", _np.rot90)
_set_sig("row_stack", _np.row_stack)
_set_sig("shape", _np.shape)
_set_sig("size", _np.size)
_set_sig("take", _np.take)
_set_sig("take_along_axis", _np.take_along_axis)
_set_sig("tri", _np.tri)
_set_sig("tril_indices", _np.tril_indices)
_set_sig("tril_indices_from", _np.tril_indices_from)
_set_sig("trim_zeros", _np.trim_zeros)
_set_sig("triu_indices", _np.triu_indices)
_set_sig("triu_indices_from", _np.triu_indices_from)
_set_sig("typename", _np.typename)
_set_sig("unravel_index", _np.unravel_index)
if hasattr(_np, "unstack"):
    _set_sig("unstack", _np.unstack)

del _set_sig, _this_module

_wrap_module_returns(_sys.modules[__name__], skip_names=_FREE_OPS_SKIP)
