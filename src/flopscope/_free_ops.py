"""Zero-FLOP wrappers around NumPy tensor creation and manipulation.

Every function in this module delegates directly to the corresponding
NumPy function and costs **0 FLOPs**, so they work both inside and
outside a :class:`~flopscope._budget.BudgetContext`.
"""

from __future__ import annotations

import inspect as _inspect
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as _np
from numpy.typing import ArrayLike, DTypeLike

from flopscope import _symmetry_transport as _st
from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._ndarray import FlopscopeArray, _asplainflopscope, _to_base_ndarray, _to_base_ndarray_tree
from flopscope._perm_group import SymmetryGroup
from flopscope._symmetric import SymmetricTensor
from flopscope._symmetry_utils import (
    broadcast_group,
    remap_group_axes,
    remap_group_for_expand_dims,
    validate_symmetry_group,
    wrap_with_symmetry,
    wrap_with_trusted_symmetry,
)
from flopscope._validation import require_budget
from flopscope.errors import SymmetryError, UnsupportedFunctionError, _warn_symmetry_loss


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
    return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]


def _compatible_symmetry_for_shape(symmetry, shape):
    """Return ``symmetry`` only when ``shape`` still supports it exactly."""
    if symmetry is None:
        return None
    try:
        validate_symmetry_group(symmetry, ndim=len(shape), shape=shape)
    except (SymmetryError, ValueError):
        return None
    return symmetry


def _normalize_axis_order(axes, ndim):
    return tuple(axis % ndim for axis in axes)


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
        "array", flop_cost=cost, subscripts=None, shapes=(_probe.shape,)
    ):
        result = _call_numpy(_np.array, object, dtype=dtype, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(array, _np.array, "counted_custom", "numel(input) FLOPs")


def zeros(
    shape: int | Sequence[int],
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return array of zeros. Wraps ``numpy.zeros``. Cost: 0 FLOPs."""
    return _wrap_constant_fill(_np.zeros(shape, dtype=dtype, **kwargs))


attach_docstring(zeros, _np.zeros, "free", "0 FLOPs")


def ones(
    shape: int | Sequence[int],
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return array of ones. Wraps ``numpy.ones``. Cost: 0 FLOPs."""
    return _wrap_constant_fill(_np.ones(shape, dtype=dtype, **kwargs))


attach_docstring(ones, _np.ones, "free", "0 FLOPs")


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
    with budget.deduct("full", flop_cost=cost, subscripts=None, shapes=()):
        result = _wrap_constant_fill(result)
    return result


attach_docstring(full, _np.full, "free", "0 FLOPs")


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return identity matrix. Wraps ``numpy.eye``. Cost: 0 FLOPs."""
    result = _np.eye(N, M=M, k=k, dtype=dtype, **kwargs)
    symmetry = _infer_structural_constructor_symmetry(kind="eye", N=N, M=M, k=k)
    if symmetry is not None:
        return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]
    return result  # type: ignore[return-value]


attach_docstring(eye, _np.eye, "free", "0 FLOPs")


@_counted_wrapper
def diag(v: ArrayLike, k: int = 0) -> FlopscopeArray:
    """Extract diagonal or construct diagonal array.

    Cost: numel(output) when constructing (1D→2D), min(m,n) when extracting (2D→1D).
    """
    budget = require_budget()
    v = _np.asarray(v)
    if v.ndim == 1:
        # Constructing diagonal matrix: output is (n+|k|) x (n+|k|)
        n = v.shape[0] + abs(k)
        cost = n * n
    else:
        # Extracting diagonal: reads min(m,n) elements
        m, n = v.shape[0], v.shape[1] if v.ndim > 1 else v.shape[0]
        cost = min(m, n)
    with budget.deduct("diag", flop_cost=cost, subscripts=None, shapes=(v.shape,)):
        result = _call_numpy(_np.diag, v, k=k)
    symmetry = _infer_structural_constructor_symmetry(kind="diag", k=k, v_ndim=v.ndim)
    if symmetry is not None:
        return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]
    return result  # type: ignore[return-value]


attach_docstring(diag, _np.diag, "free", "0 FLOPs")


@_counted_wrapper
def arange(*args: Any, **kwargs: Any) -> FlopscopeArray:
    """Return evenly spaced values. Cost: numel(output)."""
    budget = require_budget()
    # cost depends on result; duration is post-hoc
    # arange output size depends on start/stop/step parsing which is non-trivial
    result = _np.arange(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("arange", flop_cost=cost, subscripts=None, shapes=()):
        pass
    return result


attach_docstring(arange, _np.arange, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return evenly spaced numbers. Cost: numel(output)."""
    budget = require_budget()
    cost = max(int(num), 1)
    with budget.deduct("linspace", flop_cost=cost, subscripts=None, shapes=()):
        result = _call_numpy(_np.linspace, start, stop, num=num, **kwargs)  # type: ignore[arg-type, call-overload]
    return result


attach_docstring(linspace, _np.linspace, "counted_custom", "numel(output) FLOPs")


def zeros_like(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return array of zeros with same shape. Wraps ``numpy.zeros_like``. Cost: 0 FLOPs."""
    result = _np.zeros_like(_to_base_ndarray(a), dtype=dtype, **kwargs)
    symmetry = None
    if isinstance(a, SymmetricTensor):
        symmetry = _compatible_symmetry_for_shape(a.symmetry, result.shape)
    if symmetry is None:
        symmetry = _infer_constant_shape_symmetry(result.shape)
    if symmetry is None:
        if isinstance(a, SymmetricTensor):
            return _np.array(result, copy=False, subok=False)  # type: ignore[return-value]
        return result  # type: ignore[return-value]
    return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]


attach_docstring(zeros_like, _np.zeros_like, "free", "0 FLOPs")


def ones_like(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return array of ones with same shape. Wraps ``numpy.ones_like``. Cost: 0 FLOPs."""
    result = _np.ones_like(_to_base_ndarray(a), dtype=dtype, **kwargs)
    symmetry = None
    if isinstance(a, SymmetricTensor):
        symmetry = _compatible_symmetry_for_shape(a.symmetry, result.shape)
    if symmetry is None:
        symmetry = _infer_constant_shape_symmetry(result.shape)
    if symmetry is None:
        if isinstance(a, SymmetricTensor):
            return _np.array(result, copy=False, subok=False)  # type: ignore[return-value]
        return result  # type: ignore[return-value]
    return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]


attach_docstring(ones_like, _np.ones_like, "free", "0 FLOPs")


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
    with budget.deduct("full_like", flop_cost=cost, subscripts=None, shapes=()):
        result = _call_numpy(
            _np.full_like, _to_base_ndarray(a), fill_value, dtype=dtype, **kwargs
        )
    symmetry = None
    if isinstance(a, SymmetricTensor):
        symmetry = _compatible_symmetry_for_shape(a.symmetry, result.shape)
    if symmetry is None:
        symmetry = _infer_constant_shape_symmetry(result.shape)
    if symmetry is None:
        if isinstance(a, SymmetricTensor):
            return _np.array(result, copy=False, subok=False)  # type: ignore[return-value]
        return result  # type: ignore[return-value]
    return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]


attach_docstring(full_like, _np.full_like, "free", "0 FLOPs")


def empty(
    shape: int | Sequence[int],
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return uninitialized array. Wraps ``numpy.empty``. Cost: 0 FLOPs."""
    return _np.empty(shape, dtype=dtype, **kwargs)  # type: ignore[return-value]


attach_docstring(empty, _np.empty, "free", "0 FLOPs")


def empty_like(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return uninitialized array with same shape. Wraps ``numpy.empty_like``. Cost: 0 FLOPs."""
    return _np.empty_like(_to_base_ndarray(a), dtype=dtype, **kwargs)  # type: ignore[return-value]


attach_docstring(empty_like, _np.empty_like, "free", "0 FLOPs")


def identity(n: int, dtype: DTypeLike = float) -> FlopscopeArray:
    """Return identity matrix. Wraps ``numpy.identity``. Cost: 0 FLOPs."""
    result = _np.identity(n, dtype=dtype)
    symmetry = _infer_structural_constructor_symmetry(kind="identity")
    if symmetry is not None:
        return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]
    return result  # type: ignore[return-value]


attach_docstring(identity, _np.identity, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# Tensor manipulation
# ---------------------------------------------------------------------------


def reshape(a: ArrayLike, /, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Reshape an array. Wraps ``numpy.reshape``. Cost: 0 FLOPs."""
    return _np.reshape(_np.asarray(a), *args, **kwargs)


attach_docstring(reshape, _np.reshape, "free", "0 FLOPs")


def transpose(
    a: ArrayLike,
    axes: Sequence[int] | None = None,
) -> FlopscopeArray:
    """Permute array dimensions. Wraps ``numpy.transpose``. Cost: 0 FLOPs."""
    if not isinstance(a, SymmetricTensor):
        return _np.transpose(_to_base_ndarray(a), axes=axes)  # type: ignore[return-value]
    result = _np.transpose(_np.asarray(a), axes=axes)
    if axes is None:
        order = tuple(reversed(range(a.ndim)))
    else:
        order = _normalize_axis_order(tuple(axes), a.ndim)
    mapping = {old: new for new, old in enumerate(order)}
    return wrap_with_symmetry(result, remap_group_axes(a.symmetry, mapping))  # type: ignore[return-value]


attach_docstring(transpose, _np.transpose, "free", "0 FLOPs")


def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> FlopscopeArray:
    """Swap two axes. Wraps ``numpy.swapaxes``. Cost: 0 FLOPs."""
    if not isinstance(a, SymmetricTensor):
        return _np.swapaxes(_to_base_ndarray(a), axis1, axis2)  # type: ignore[return-value]
    result = _np.swapaxes(_np.asarray(a), axis1, axis2)
    order = list(range(a.ndim))
    axis1 %= a.ndim
    axis2 %= a.ndim
    order[axis1], order[axis2] = order[axis2], order[axis1]
    mapping = {old: new for new, old in enumerate(order)}
    return wrap_with_symmetry(result, remap_group_axes(a.symmetry, mapping))  # type: ignore[return-value]


attach_docstring(swapaxes, _np.swapaxes, "free", "0 FLOPs")


def moveaxis(
    a: ArrayLike,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> FlopscopeArray:
    """Move axes to new positions. Wraps ``numpy.moveaxis``. Cost: 0 FLOPs."""
    if not isinstance(a, SymmetricTensor):
        return _np.moveaxis(_to_base_ndarray(a), source, destination)  # type: ignore[return-value]
    result = _np.moveaxis(_np.asarray(a), source, destination)
    if _np.ndim(source) == 0:
        source_axes = (int(source),)  # type: ignore[arg-type, call-overload]
    else:
        source_axes = tuple(source)  # type: ignore[arg-type, call-overload]
    if _np.ndim(destination) == 0:
        destination_axes = (int(destination),)  # type: ignore[arg-type, call-overload]
    else:
        destination_axes = tuple(destination)  # type: ignore[arg-type, call-overload]
    source_axes = _normalize_axis_order(source_axes, a.ndim)
    destination_axes = _normalize_axis_order(destination_axes, a.ndim)
    order = [axis for axis in range(a.ndim) if axis not in source_axes]
    for dest, src in sorted(zip(destination_axes, source_axes, strict=True)):
        order.insert(dest, src)
    mapping = {old: new for new, old in enumerate(order)}
    return wrap_with_symmetry(result, remap_group_axes(a.symmetry, mapping))  # type: ignore[return-value]


attach_docstring(moveaxis, _np.moveaxis, "free", "0 FLOPs")


@_counted_wrapper
def concatenate(
    arrays: Sequence[ArrayLike],
    axis: int | None = 0,
    **kwargs: Any,
) -> FlopscopeArray:
    """Join arrays along an axis. Cost: numel(output)."""
    budget = require_budget()
    cost = max(sum(_np.asarray(a).size for a in arrays), 1)
    with budget.deduct("concatenate", flop_cost=cost, subscripts=None, shapes=()):
        result = _call_numpy(
            _np.concatenate, _to_base_ndarray_tree(arrays), axis=axis, **kwargs
        )  # type: ignore[arg-type, call-overload]
    return result  # type: ignore[return-value]


attach_docstring(concatenate, _np.concatenate, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def stack(
    arrays: Sequence[ArrayLike],
    axis: int = 0,
    **kwargs: Any,
) -> FlopscopeArray:
    """Stack arrays along a new axis. Cost: numel(output)."""
    budget = require_budget()
    cost = max(sum(_np.asarray(a).size for a in arrays), 1)
    with budget.deduct("stack", flop_cost=cost, subscripts=None, shapes=()):
        result = _call_numpy(
            _np.stack, _to_base_ndarray_tree(arrays), axis=axis, **kwargs
        )  # type: ignore[arg-type, call-overload]
    return result


attach_docstring(stack, _np.stack, "free", "0 FLOPs")


@_counted_wrapper
def vstack(tup: Sequence[ArrayLike]) -> FlopscopeArray:
    """Stack arrays vertically. Cost: numel(output)."""
    budget = require_budget()
    cost = max(sum(_np.asarray(a).size for a in tup), 1)
    with budget.deduct("vstack", flop_cost=cost, subscripts=None, shapes=()):
        result = _call_numpy(_np.vstack, _to_base_ndarray_tree(tup))  # type: ignore[arg-type, call-overload]
    return result


attach_docstring(vstack, _np.vstack, "free", "0 FLOPs")


def hstack(tup: Sequence[ArrayLike]) -> FlopscopeArray:
    """Stack arrays horizontally. Wraps ``numpy.hstack``. Cost: 0 FLOPs."""
    return _np.hstack(_to_base_ndarray_tree(tup))  # type: ignore[arg-type, call-overload]


attach_docstring(hstack, _np.hstack, "free", "0 FLOPs")


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
            lost_dims=[in_group.axes or tuple(range(in_group.degree))],
            reason=f"split along axis {axis} breaks block symmetry",
        )
    with budget.deduct(
        "split", flop_cost=cost, subscripts=None, shapes=(ary_arr.shape,)
    ):
        raw_pieces = _call_numpy(
            _np.split, ary_arr, indices_or_sections, axis=axis,
        )
    if out_group is not None:
        return [wrap_with_symmetry(p, out_group) for p in raw_pieces]  # type: ignore[return-value]
    return [_asplainflopscope(p) for p in raw_pieces]  # type: ignore[return-value]


attach_docstring(split, _np.split, "free", "0 FLOPs")


def hsplit(
    ary: ArrayLike,
    indices_or_sections: int | Sequence[int],
) -> list[FlopscopeArray]:
    """Split array horizontally. Wraps ``numpy.hsplit``. Cost: 0 FLOPs."""
    ary_arr = _np.asarray(ary)
    in_group = ary.symmetry if isinstance(ary, SymmetricTensor) else None
    out_group = _st.transport_hsplit(in_group, input_shape=ary_arr.shape)
    raw_pieces = _np.hsplit(ary_arr, indices_or_sections)
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[in_group.axes or tuple(range(in_group.degree))],
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
            lost_dims=[in_group.axes or tuple(range(in_group.degree))],
            reason="vsplit breaks block symmetry",
        )
    with budget.deduct(
        "vsplit", flop_cost=cost, subscripts=None, shapes=(ary_arr.shape,)
    ):
        raw_pieces = _call_numpy(_np.vsplit, ary_arr, indices_or_sections)
    if out_group is not None:
        return [wrap_with_symmetry(p, out_group) for p in raw_pieces]  # type: ignore[return-value]
    return [_asplainflopscope(p) for p in raw_pieces]  # type: ignore[return-value]


attach_docstring(vsplit, _np.vsplit, "free", "0 FLOPs")


def squeeze(
    a: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
) -> FlopscopeArray:
    """Remove length-1 axes. Wraps ``numpy.squeeze``. Cost: 0 FLOPs."""
    a_arr = _np.asarray(a)
    result = _np.squeeze(a_arr, axis=axis)
    in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
    out_group = _st.transport_squeeze(in_group, input_shape=a_arr.shape, axis=axis)
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[in_group.axes or tuple(range(in_group.degree))],
            reason="squeeze removes an axis inside the symmetric block",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(squeeze, _np.squeeze, "free", "0 FLOPs")


def expand_dims(
    a: ArrayLike,
    axis: int | tuple[int, ...],
) -> FlopscopeArray:
    """Insert a new axis. Wraps ``numpy.expand_dims``. Cost: 0 FLOPs."""
    a_arr = _np.asarray(a)
    result = _np.expand_dims(_np.asarray(a), axis=axis)
    symmetry = remap_group_for_expand_dims(
        a.symmetry if isinstance(a, SymmetricTensor) else None,
        ndim=a_arr.ndim,
        axis=axis,
    )
    return wrap_with_symmetry(result, symmetry) if symmetry is not None else result  # type: ignore[return-value]


attach_docstring(expand_dims, _np.expand_dims, "free", "0 FLOPs")


@_counted_wrapper
def ravel(a: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Flatten array. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = max(a_arr.size, 1)
    with budget.deduct("ravel", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)):
        result = _call_numpy(_np.ravel, a_arr, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(ravel, _np.ravel, "free", "0 FLOPs")


def copy(a: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Return copy of array. Wraps ``numpy.copy``. Cost: 0 FLOPs."""
    result = _np.copy(_np.asarray(a), **kwargs)
    if isinstance(a, SymmetricTensor):
        return wrap_with_symmetry(result, a.symmetry)  # type: ignore[return-value]
    return result  # type: ignore[return-value]


attach_docstring(copy, _np.copy, "free", "0 FLOPs")


@_counted_wrapper
def where(
    condition: ArrayLike,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
) -> FlopscopeArray | tuple[FlopscopeArray, ...]:
    """Return elements chosen from *x* or *y*. Cost: numel(input)."""
    budget = require_budget()
    cond_arr = _np.asarray(condition)
    cost = cond_arr.size
    with budget.deduct(
        "where", flop_cost=cost, subscripts=None, shapes=(cond_arr.shape,)
    ):
        if x is None and y is None:
            result = _call_numpy(_np.where, _to_base_ndarray(condition))
        else:
            result = _call_numpy(
                _np.where,
                _to_base_ndarray(condition),
                _to_base_ndarray(x),  # type: ignore[arg-type, call-overload]
                _to_base_ndarray(y),  # type: ignore[arg-type, call-overload]
            )
    return result  # type: ignore[return-value]


attach_docstring(where, _np.where, "free", "0 FLOPs")


@_counted_wrapper
def tile(A: ArrayLike, reps: int | Sequence[int]) -> FlopscopeArray:
    """Construct array by repeating. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(A)
    reps_tup = (reps,) if _np.ndim(reps) == 0 else tuple(reps)  # type: ignore[arg-type, call-overload]
    # Output size = input size * product of reps
    cost = max(a_arr.size * int(_np.prod(reps_tup)), 1)
    with budget.deduct("tile", flop_cost=cost, subscripts=None, shapes=()):
        result = _call_numpy(_np.tile, _to_base_ndarray(A), reps)
    return result  # type: ignore[return-value]


attach_docstring(tile, _np.tile, "free", "0 FLOPs")


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
    result = _np.repeat(a_arr, repeats, axis=axis)
    cost = max(result.size, 1)
    with budget.deduct("repeat", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)):
        pass  # cost deducted; result already computed
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[in_group.axes or tuple(range(in_group.degree))],
            reason="repeat along a block axis breaks block symmetry",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(repeat, _np.repeat, "free", "0 FLOPs")


def flip(
    m: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
) -> FlopscopeArray:
    """Reverse order of elements. Wraps ``numpy.flip``. Cost: 0 FLOPs."""
    return _np.flip(_to_base_ndarray(m), axis=axis)  # type: ignore[return-value]


attach_docstring(flip, _np.flip, "free", "0 FLOPs")


def roll(
    a: ArrayLike,
    shift: int | Sequence[int],
    axis: int | Sequence[int] | None = None,
) -> FlopscopeArray:
    """Roll array elements along an axis. Wraps ``numpy.roll``. Cost: 0 FLOPs."""
    a_arr = _np.asarray(a)
    result = _np.roll(a_arr, shift, axis=axis)
    in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
    out_group = _st.transport_roll(in_group, input_shape=a_arr.shape, axis=axis)
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[in_group.axes or tuple(range(in_group.degree))],
            reason="roll along a block axis breaks block symmetry",
        )
    if out_group is not None:
        return wrap_with_symmetry(result, out_group)  # type: ignore[return-value]
    return _asplainflopscope(result)  # type: ignore[return-value]


attach_docstring(roll, _np.roll, "free", "0 FLOPs")


@_counted_wrapper
def pad(array: ArrayLike, pad_width: Any, **kwargs: Any) -> FlopscopeArray:
    """Pad an array. Cost: numel(output)."""
    budget = require_budget()
    # cost depends on result; duration is post-hoc
    # pad_width parsing is complex (scalar, per-axis, per-side) — not worth replicating
    result = _np.pad(_to_base_ndarray(array), pad_width, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("pad", flop_cost=cost, subscripts=None, shapes=()):
        pass
    return result  # type: ignore[return-value]


attach_docstring(pad, _np.pad, "free", "0 FLOPs")


def triu(m: ArrayLike, k: int = 0) -> FlopscopeArray:
    """Upper triangle. Wraps ``numpy.triu``. Cost: 0 FLOPs."""
    return _np.triu(_to_base_ndarray(m), k=k)  # type: ignore[return-value]


attach_docstring(triu, _np.triu, "free", "0 FLOPs")


def tril(m: ArrayLike, k: int = 0) -> FlopscopeArray:
    """Lower triangle. Wraps ``numpy.tril``. Cost: 0 FLOPs."""
    return _np.tril(_to_base_ndarray(m), k=k)  # type: ignore[return-value]


attach_docstring(tril, _np.tril, "free", "0 FLOPs")


@_counted_wrapper
def diagonal(
    a: ArrayLike,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> FlopscopeArray:
    """Return diagonal. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    # Diagonal length along axis1/axis2
    m, n = a_arr.shape[axis1], a_arr.shape[axis2]
    if offset >= 0:
        diag_len = max(min(m, n - offset), 0)
    else:
        diag_len = max(min(m + offset, n), 0)
    cost = max(diag_len, 1)
    with budget.deduct(
        "diagonal", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
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
    input_array = _np.asarray(array)
    budget = require_budget()
    cost = max(int(_np.prod(output_shape)), 1)
    with budget.deduct("broadcast_to", flop_cost=cost, subscripts=None, shapes=()):
        result = _call_numpy(_np.broadcast_to, input_array, output_shape)
    symmetry = broadcast_group(
        array.symmetry if isinstance(array, SymmetricTensor) else None,
        input_shape=input_array.shape,
        output_shape=output_shape,
    )
    return wrap_with_symmetry(result, symmetry)  # type: ignore[return-value]


attach_docstring(broadcast_to, _np.broadcast_to, "free", "0 FLOPs")


@_counted_wrapper
def meshgrid(*xi: ArrayLike, **kwargs: Any) -> tuple[FlopscopeArray, ...]:
    """Return coordinate matrices. Cost: numel(output)."""
    budget = require_budget()
    # Each output grid has shape = product of all input lengths; there are len(xi) grids
    sizes = [_np.asarray(x).size for x in xi]
    grid_size = int(_np.prod(sizes)) if sizes else 0
    cost = max(grid_size * len(sizes), 1)
    with budget.deduct("meshgrid", flop_cost=cost, subscripts=None, shapes=()):
        result = _call_numpy(_np.meshgrid, *[_to_base_ndarray(x) for x in xi], **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(meshgrid, _np.meshgrid, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# Type / info helpers
# ---------------------------------------------------------------------------


def astype(
    x: ArrayLike,
    dtype: DTypeLike,
    /,
    *,
    copy: bool = True,
    device: Any = None,
) -> FlopscopeArray:
    """Cast array to *dtype*. Wraps ``np.astype(x, dtype)``. Cost: 0 FLOPs."""
    return _np.astype(_to_base_ndarray(x), dtype, copy=copy, device=device)  # type: ignore[arg-type, call-overload]


@_counted_wrapper
def asarray(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Convert to array. Cost: numel(output)."""
    budget = require_budget()
    # Pre-compute cost; asarray on an already-array is a no-op
    _probe = _np.asarray(a)
    cost = max(_probe.size, 1)
    with budget.deduct(
        "asarray", flop_cost=cost, subscripts=None, shapes=(_probe.shape,)
    ):
        result = _call_numpy(_np.asarray, a, dtype=dtype, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(asarray, _np.asarray, "free", "0 FLOPs")


@_counted_wrapper
def isnan(x: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Test element-wise for NaN. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    with budget.deduct("isnan", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,)):
        # Strip flopscope subclasses so the raw NumPy ufunc does not
        # re-dispatch through __array_ufunc__ and recurse.
        result = _call_numpy(_np.isnan, _to_base_ndarray(x), **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(isnan, _np.isnan, "free", "0 FLOPs")


@_counted_wrapper
def isfinite(x: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Test element-wise for finiteness. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    with budget.deduct(
        "isfinite", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,)
    ):
        result = _call_numpy(_np.isfinite, _to_base_ndarray(x), **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(isfinite, _np.isfinite, "free", "0 FLOPs")


@_counted_wrapper
def isinf(x: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Test element-wise for Inf. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    with budget.deduct("isinf", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,)):
        result = _call_numpy(_np.isinf, _to_base_ndarray(x), **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(isinf, _np.isinf, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# New free ops
# ---------------------------------------------------------------------------


@_counted_wrapper
def append(
    arr: ArrayLike,
    values: ArrayLike,
    axis: int | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Append values. Cost: numel(appended values)."""
    budget = require_budget()
    values_arr = _np.asarray(values)
    cost = values_arr.size  # num appended
    with budget.deduct("append", flop_cost=cost, subscripts=None, shapes=()):
        result = _call_numpy(
            _np.append,
            _to_base_ndarray(arr),
            _to_base_ndarray(values),
            axis=axis,
            **kwargs,
        )
    return result  # type: ignore[return-value]


attach_docstring(append, _np.append, "free", "0 FLOPs")


@_counted_wrapper
def argwhere(a: ArrayLike, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Find indices of non-zero elements. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    with budget.deduct(
        "argwhere", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
    ):
        result = _call_numpy(_np.argwhere, _to_base_ndarray(a), *args, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(argwhere, _np.argwhere, "free", "0 FLOPs")


@_counted_wrapper
def array_split(ary: ArrayLike, *args: Any, **kwargs: Any) -> list[FlopscopeArray]:
    """Split array into sub-arrays. Cost: numel(input)."""
    budget = require_budget()
    ary_arr = _np.asarray(ary)
    cost = ary_arr.size
    with budget.deduct(
        "array_split", flop_cost=cost, subscripts=None, shapes=(ary_arr.shape,)
    ):
        result = _call_numpy(_np.array_split, _to_base_ndarray(ary), *args, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(array_split, _np.array_split, "free", "0 FLOPs")


@_counted_wrapper
def asarray_chkfinite(a: ArrayLike, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Convert to array checking for NaN/Inf. Cost: numel(output)."""
    budget = require_budget()
    result = _np.asarray_chkfinite(_to_base_ndarray(a), *args, **kwargs)
    cost = (
        result.size
        if hasattr(result, "size")
        else len(result)
        if hasattr(result, "__len__")
        else 1
    )
    with budget.deduct(
        "asarray_chkfinite", flop_cost=cost, subscripts=None, shapes=(result.shape,)
    ):
        pass  # numpy call already executed above
    return result  # type: ignore[return-value]


attach_docstring(asarray_chkfinite, _np.asarray_chkfinite, "free", "0 FLOPs")


def atleast_1d(
    *arys: ArrayLike,
) -> FlopscopeArray | tuple[FlopscopeArray, ...]:
    """Convert to 1-D or higher. Wraps ``numpy.atleast_1d``. Cost: 0 FLOPs."""
    def _one(a):
        a_arr = _np.asarray(a)
        result = _np.atleast_1d(a_arr)
        in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
        out_group = _st.transport_atleast_1d(in_group, input_shape=a_arr.shape)
        if in_group is not None and out_group is None:
            _warn_symmetry_loss(
                lost_dims=[in_group.axes or tuple(range(in_group.degree))],
                reason="atleast_1d incompatible with block structure",
            )
        if out_group is not None:
            return wrap_with_symmetry(result, out_group)
        return _asplainflopscope(result)
    if len(arys) == 1:
        return _one(arys[0])
    return tuple(_one(a) for a in arys)


attach_docstring(atleast_1d, _np.atleast_1d, "free", "0 FLOPs")


def atleast_2d(
    *arys: ArrayLike,
) -> FlopscopeArray | tuple[FlopscopeArray, ...]:
    """Convert to 2-D or higher. Wraps ``numpy.atleast_2d``. Cost: 0 FLOPs."""
    def _one(a):
        a_arr = _np.asarray(a)
        result = _np.atleast_2d(a_arr)
        in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
        out_group = _st.transport_atleast_2d(in_group, input_shape=a_arr.shape)
        if in_group is not None and out_group is None:
            _warn_symmetry_loss(
                lost_dims=[in_group.axes or tuple(range(in_group.degree))],
                reason="atleast_2d incompatible with block structure",
            )
        if out_group is not None:
            return wrap_with_symmetry(result, out_group)
        return _asplainflopscope(result)
    if len(arys) == 1:
        return _one(arys[0])
    return tuple(_one(a) for a in arys)


attach_docstring(atleast_2d, _np.atleast_2d, "free", "0 FLOPs")


def atleast_3d(
    *arys: ArrayLike,
) -> FlopscopeArray | tuple[FlopscopeArray, ...]:
    """Convert to 3-D or higher. Wraps ``numpy.atleast_3d``. Cost: 0 FLOPs."""
    def _one(a):
        a_arr = _np.asarray(a)
        result = _np.atleast_3d(a_arr)
        in_group = a.symmetry if isinstance(a, SymmetricTensor) else None
        out_group = _st.transport_atleast_3d(in_group, input_shape=a_arr.shape)
        if in_group is not None and out_group is None:
            _warn_symmetry_loss(
                lost_dims=[in_group.axes or tuple(range(in_group.degree))],
                reason="atleast_3d incompatible with block structure",
            )
        if out_group is not None:
            return wrap_with_symmetry(result, out_group)
        return _asplainflopscope(result)
    if len(arys) == 1:
        return _one(arys[0])
    return tuple(_one(a) for a in arys)


attach_docstring(atleast_3d, _np.atleast_3d, "free", "0 FLOPs")


@_counted_wrapper
def base_repr(*args, **kwargs):
    """Return string representation of number. Cost: numel(output)."""
    budget = require_budget()
    result = _np.base_repr(*args, **kwargs)
    cost = len(result)
    with budget.deduct("base_repr", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(base_repr, _np.base_repr, "free", "0 FLOPs")


@_counted_wrapper
def binary_repr(*args, **kwargs):
    """Return binary representation of integer. Cost: numel(output)."""
    budget = require_budget()
    result = _np.binary_repr(*args, **kwargs)
    cost = len(result)
    with budget.deduct("binary_repr", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(binary_repr, _np.binary_repr, "free", "0 FLOPs")


@_counted_wrapper
def block(*args, **kwargs):
    """Assemble array from nested lists. Cost: numel(output)."""
    budget = require_budget()
    result = _np.block(*[_to_base_ndarray_tree(a) for a in args], **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("block", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(block, _np.block, "free", "0 FLOPs")


@_counted_wrapper
def bmat(*args, **kwargs):
    """Build matrix from string/nested sequence. Cost: numel(output)."""
    budget = require_budget()
    # First arg may be a string OR a nested sequence of arrays
    stripped_args = []
    for arg in args:
        if isinstance(arg, (tuple, list)):
            stripped_args.append(_to_base_ndarray_tree(arg))
        else:
            stripped_args.append(arg)
    result = _np.bmat(*stripped_args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("bmat", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(bmat, _np.bmat, "free", "0 FLOPs")


@_counted_wrapper
def broadcast_arrays(*args: ArrayLike, **kwargs: Any) -> tuple[FlopscopeArray, ...]:
    """Broadcast any number of arrays. Cost: numel(output)."""
    arrays = tuple(_np.asarray(arg) for arg in args)
    budget = require_budget()
    result = _np.broadcast_arrays(*arrays, **kwargs)
    cost = sum(a.size for a in result)
    with budget.deduct("broadcast_arrays", flop_cost=cost, subscripts=None, shapes=()):
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
    # Args: (a, choices, ...) or just (a, choices) — strip arrays.
    stripped_args = []
    for arg in args:
        if isinstance(arg, _np.ndarray):
            stripped_args.append(_to_base_ndarray(arg))
        elif isinstance(arg, (tuple, list)):
            stripped_args.append(_to_base_ndarray_tree(arg))
        else:
            stripped_args.append(arg)
    result = _np.choose(*stripped_args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("choose", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(choose, _np.choose, "free", "0 FLOPs")


def column_stack(tup: Sequence[ArrayLike]) -> FlopscopeArray:
    """Stack 1-D arrays as columns. Wraps ``numpy.column_stack``. Cost: 0 FLOPs."""
    # First positional arg is sequence of arrays
    return _np.column_stack(_to_base_ndarray_tree(tup))  # type: ignore[return-value]


attach_docstring(column_stack, _np.column_stack, "free", "0 FLOPs")


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
    """Return selected slices along an axis. Cost: numel(output)."""
    budget = require_budget()
    result = _np.compress(
        _to_base_ndarray(condition),  # type: ignore[arg-type]
        _to_base_ndarray(a),
        *args,
        **kwargs,
    )
    cost = (
        result.size
        if hasattr(result, "size")
        else len(result)
        if hasattr(result, "__len__")
        else 1
    )
    with budget.deduct(
        "compress", flop_cost=cost, subscripts=None, shapes=(result.shape,)
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(compress, _np.compress, "free", "0 FLOPs")


@_counted_wrapper
def concat(
    arrays: Sequence[ArrayLike],
    axis: int | None = 0,
    **kwargs: Any,
) -> FlopscopeArray:
    """Join arrays along an axis. Cost: numel(output)."""
    budget = require_budget()
    result = _np.concat(_to_base_ndarray_tree(arrays), axis=axis, **kwargs)  # type: ignore[arg-type, call-overload]
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("concat", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result  # type: ignore[return-value]


attach_docstring(concat, _np.concat, "free", "0 FLOPs")


@_counted_wrapper
def copyto(dst, src, casting="same_kind", where=True):
    """Copies values from one array to another. Cost: num elements copied."""
    budget = require_budget()
    src_arr = _np.asarray(src)
    if where is not True:
        where_arr = _np.asarray(where)
        cost = int(_np.count_nonzero(where_arr))
    else:
        cost = src_arr.size
    with budget.deduct("copyto", flop_cost=cost, subscripts=None, shapes=()):
        result = _call_numpy(
            _np.copyto,
            _to_base_ndarray(dst),
            _to_base_ndarray(src),
            casting=casting,  # type: ignore[arg-type, call-overload]
            where=_to_base_ndarray(where) if where is not True else where,
        )
    return result


attach_docstring(copyto, _np.copyto, "free", "0 FLOPs")


@_counted_wrapper
def delete(
    arr: ArrayLike,
    obj: Any,
    axis: int | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Return new array with sub-arrays deleted. Cost: num elements removed."""
    budget = require_budget()
    arr_np = _np.asarray(arr)
    result = _np.delete(_to_base_ndarray(arr), obj, axis=axis, **kwargs)
    cost = max(arr_np.size - result.size, 0)  # num deleted
    with budget.deduct("delete", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result  # type: ignore[return-value]


attach_docstring(delete, _np.delete, "free", "0 FLOPs")


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
    result = _np.diagflat(_to_base_ndarray(v), k=k)
    cost = result.size  # output is (n+|k|)×(n+|k|) matrix
    with budget.deduct(
        "diagflat", flop_cost=cost, subscripts=None, shapes=(v_arr.shape,)
    ):
        pass  # numpy call already executed above
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
            lost_dims=[in_group.axes or tuple(range(in_group.degree))],
            reason="dsplit breaks block symmetry",
        )
    with budget.deduct(
        "dsplit", flop_cost=cost, subscripts=None, shapes=(ary_arr.shape,)
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
    result = _np.dstack(_to_base_ndarray_tree(tup))  # type: ignore[arg-type]
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("dstack", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result  # type: ignore[return-value]


attach_docstring(dstack, _np.dstack, "free", "0 FLOPs")


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
        "extract", flop_cost=cost, subscripts=None, shapes=(arr_np.shape,)
    ):
        result = _call_numpy(
            _np.extract,
            _to_base_ndarray(condition),
            _to_base_ndarray(arr),
            *args,
            **kwargs,
        )
    return result  # type: ignore[return-value]


attach_docstring(extract, _np.extract, "free", "0 FLOPs")


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
        "fill_diagonal", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
    ):
        # ``np.fill_diagonal`` mutates ``a`` in-place; ``_to_base_ndarray``
        # is zero-copy so the mutation propagates to the user's array.
        result = _call_numpy(
            _np.fill_diagonal, _to_base_ndarray(a), val, wrap=wrap, **kwargs
        )  # type: ignore[arg-type, call-overload]
    return result


attach_docstring(fill_diagonal, _np.fill_diagonal, "free", "0 FLOPs")


@_counted_wrapper
def flatnonzero(a: ArrayLike, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Return indices of non-zero elements in flattened array. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    with budget.deduct(
        "flatnonzero", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
    ):
        result = _call_numpy(_np.flatnonzero, _to_base_ndarray(a), *args, **kwargs)
    return result  # type: ignore[return-value]


attach_docstring(flatnonzero, _np.flatnonzero, "free", "0 FLOPs")


def fliplr(*args, **kwargs):
    """Reverse elements along axis 1. Wraps ``numpy.fliplr``. Cost: 0 FLOPs."""
    stripped_args = _to_base_ndarray_tree(args)
    return _np.fliplr(*stripped_args, **kwargs)


attach_docstring(fliplr, _np.fliplr, "free", "0 FLOPs")


def flipud(*args, **kwargs):
    """Reverse elements along axis 0. Wraps ``numpy.flipud``. Cost: 0 FLOPs."""
    stripped_args = _to_base_ndarray_tree(args)
    return _np.flipud(*stripped_args, **kwargs)


attach_docstring(flipud, _np.flipud, "free", "0 FLOPs")


@_counted_wrapper
def from_dlpack(*args, **kwargs):
    """Create array from DLPack capsule. Cost: numel(output)."""
    budget = require_budget()
    result = _np.from_dlpack(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("from_dlpack", flop_cost=cost, subscripts=None, shapes=()):
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
    with budget.deduct("frombuffer", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result  # type: ignore[return-value]


attach_docstring(frombuffer, _np.frombuffer, "free", "0 FLOPs")


@_counted_wrapper
def fromfile(*args, **kwargs):
    """Construct array from data in text or binary file. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromfile(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("fromfile", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(fromfile, _np.fromfile, "free", "0 FLOPs")


@_counted_wrapper
def fromfunction(*args, **kwargs):
    """Construct array by executing function over each coordinate. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromfunction(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("fromfunction", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(fromfunction, _np.fromfunction, "free", "0 FLOPs")


@_counted_wrapper
def fromiter(*args, **kwargs):
    """Create array from iterable object. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromiter(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("fromiter", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(fromiter, _np.fromiter, "free", "0 FLOPs")


@_counted_wrapper
def fromregex(*args, **kwargs):
    """Construct array from text file using regex. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromregex(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("fromregex", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(fromregex, _np.fromregex, "free", "0 FLOPs")


@_counted_wrapper
def fromstring(*args, **kwargs):
    """Construct array from string. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromstring(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("fromstring", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(fromstring, _np.fromstring, "free", "0 FLOPs")


@_counted_wrapper
def indices(*args: Any, **kwargs: Any) -> FlopscopeArray:
    """Return array representing indices of a grid. Cost: numel(output)."""
    budget = require_budget()
    result = _np.indices(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("indices", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(indices, _np.indices, "free", "0 FLOPs")


@_counted_wrapper
def insert(
    arr: ArrayLike,
    obj: Any,
    values: ArrayLike,
    axis: int | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Insert values along axis before given indices. Cost: numel(inserted values)."""
    budget = require_budget()
    values_arr = _np.asarray(values)
    cost = values_arr.size  # num inserted
    with budget.deduct("insert", flop_cost=cost, subscripts=None, shapes=()):
        result = _call_numpy(
            _np.insert,
            _to_base_ndarray(arr),
            obj,
            _to_base_ndarray(values),
            axis=axis,
            **kwargs,
        )
    return result  # type: ignore[return-value]


attach_docstring(insert, _np.insert, "free", "0 FLOPs")


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


attach_docstring(isin, _np.isin, "free", "0 FLOPs")


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
    with budget.deduct("ix_", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(ix_, _np.ix_, "free", "0 FLOPs")


@_counted_wrapper
def mask_indices(*args, **kwargs):
    """Return indices to access main or off-diagonal of array. Cost: numel(output)."""
    budget = require_budget()
    result = _np.mask_indices(*args, **kwargs)
    cost = sum(a.size for a in result) if isinstance(result, tuple) else 1
    with budget.deduct("mask_indices", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(mask_indices, _np.mask_indices, "free", "0 FLOPs")


def matrix_transpose(x: ArrayLike) -> FlopscopeArray:
    """Swap last two axes. Wraps ``numpy.matrix_transpose``. Cost: 0 FLOPs."""
    x_arr = _np.asarray(x)
    result = _np.matrix_transpose(x_arr)
    in_group = x.symmetry if isinstance(x, SymmetricTensor) else None
    out_group = _st.transport_matrix_transpose(in_group, ndim=x_arr.ndim)
    if in_group is not None and out_group is None:
        _warn_symmetry_loss(
            lost_dims=[in_group.axes or tuple(range(in_group.degree))],
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
        "nonzero", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
    ):
        result = _call_numpy(_np.nonzero, _to_base_ndarray(a), *args, **kwargs)  # type: ignore[arg-type, call-overload]
    return result  # type: ignore[return-value]


attach_docstring(nonzero, _np.nonzero, "free", "0 FLOPs")


@_counted_wrapper
def packbits(a: ArrayLike, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Pack binary-valued array into bits. Cost: numel(output)."""
    budget = require_budget()
    result = _np.packbits(_to_base_ndarray(a), *args, **kwargs)  # type: ignore[arg-type, call-overload]
    cost = (
        result.size
        if hasattr(result, "size")
        else len(result)
        if hasattr(result, "__len__")
        else 1
    )
    with budget.deduct(
        "packbits", flop_cost=cost, subscripts=None, shapes=(result.shape,)
    ):
        pass  # numpy call already executed above
    return result  # type: ignore[return-value]


attach_docstring(packbits, _np.packbits, "free", "0 FLOPs")


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
        "place", flop_cost=cost, subscripts=None, shapes=(arr_np.shape,)
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


attach_docstring(place, _np.place, "free", "0 FLOPs")


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
    """Replace elements at given flat indices. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    with budget.deduct("put", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)):
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


attach_docstring(put, _np.put, "free", "0 FLOPs")


@_counted_wrapper
def put_along_axis(
    arr: ArrayLike,
    indices: ArrayLike,
    values: ArrayLike,
    axis: int | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Put values into destination array along axis. Cost: numel(input)."""
    budget = require_budget()
    arr_np = _np.asarray(arr)
    cost = arr_np.size
    with budget.deduct(
        "put_along_axis", flop_cost=cost, subscripts=None, shapes=(arr_np.shape,)
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


attach_docstring(put_along_axis, _np.put_along_axis, "free", "0 FLOPs")


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
        "putmask", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
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


attach_docstring(putmask, _np.putmask, "free", "0 FLOPs")


def ravel_multi_index(*args, **kwargs):
    """Convert multi-index to flat index. Wraps ``numpy.ravel_multi_index``. Cost: 0 FLOPs."""
    stripped_args = _to_base_ndarray_tree(args)
    return _np.ravel_multi_index(*stripped_args, **kwargs)


attach_docstring(ravel_multi_index, _np.ravel_multi_index, "free", "0 FLOPs")


def require(*args, **kwargs):
    """Return array satisfying requirements. Wraps ``numpy.require``. Cost: 0 FLOPs."""
    # Pass args through unstripped: ``_np.require`` is a thin Python
    # helper around ``np.asanyarray`` and does not enter the
    # ``__array_function__`` dispatch path, so passing a FlopscopeArray
    # cannot recurse. Stripping would break ``np.require(x).is(x)``
    # identity for already-conforming inputs.
    return _np.require(*args, **kwargs)


attach_docstring(require, _np.require, "free", "0 FLOPs")


@_counted_wrapper
def resize(*args, **kwargs):
    """Return new array with given shape. Cost: numel(output)."""
    budget = require_budget()
    stripped_args = _to_base_ndarray_tree(args)
    result = _np.resize(*stripped_args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("resize", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(resize, _np.resize, "free", "0 FLOPs")


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
    with budget.deduct("rollaxis", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(rollaxis, _np.rollaxis, "free", "0 FLOPs")


def rot90(*args, **kwargs):
    """Rotate array 90 degrees. Wraps ``numpy.rot90``. Cost: 0 FLOPs."""
    stripped_args = _to_base_ndarray_tree(args)
    return _np.rot90(*stripped_args, **kwargs)


attach_docstring(rot90, _np.rot90, "free", "0 FLOPs")


def row_stack(*args, **kwargs):
    """Stack arrays vertically (alias for vstack). Wraps ``numpy.row_stack``. Cost: 0 FLOPs."""
    stripped_args = _to_base_ndarray_tree(args)
    return _np.row_stack(*stripped_args, **kwargs)


attach_docstring(row_stack, _np.row_stack, "free", "0 FLOPs")


@_counted_wrapper
def select(
    condlist: Sequence[ArrayLike],
    choicelist: Sequence[ArrayLike],
    default: Any = 0,
) -> FlopscopeArray:
    """Return array drawn from elements depending on conditions. Cost: numel(input)."""
    budget = require_budget()
    # Cost based on the size of the choice arrays
    cost = max((_np.asarray(c).size for c in choicelist), default=1)
    with budget.deduct("select", flop_cost=cost, subscripts=None, shapes=()):
        result = _call_numpy(
            _np.select,
            _to_base_ndarray_tree(condlist),  # type: ignore[arg-type]
            _to_base_ndarray_tree(choicelist),  # type: ignore[arg-type]
            default=default,
        )
    return result  # type: ignore[return-value]


attach_docstring(select, _np.select, "free", "0 FLOPs")


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
    result = _np.take(
        _to_base_ndarray(a),
        _to_base_ndarray(indices),  # type: ignore[arg-type]
        axis=axis,
        out=_to_base_ndarray(out) if out is not None else None,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
    )
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("take", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result  # type: ignore[return-value]


attach_docstring(take, _np.take, "free", "0 FLOPs")


@_counted_wrapper
def take_along_axis(
    arr: ArrayLike,
    indices: ArrayLike,
    axis: int | None,
) -> FlopscopeArray:
    """Take values from input array along axis using indices. Cost: numel(output)."""
    budget = require_budget()
    result = _np.take_along_axis(
        _to_base_ndarray(arr),  # type: ignore[arg-type]
        _to_base_ndarray(indices),  # type: ignore[arg-type]
        axis=axis,
    )
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("take_along_axis", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result  # type: ignore[return-value]


attach_docstring(take_along_axis, _np.take_along_axis, "free", "0 FLOPs")


def tri(*args, **kwargs):
    """Array with ones at and below the given diagonal. Wraps ``numpy.tri``. Cost: 0 FLOPs."""
    return _np.tri(*args, **kwargs)


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
    """Trim leading and/or trailing zeros from 1-D array. Cost: num elements trimmed."""
    budget = require_budget()
    filt_arr = _np.asarray(filt)
    result = _np.trim_zeros(_to_base_ndarray(filt), trim=trim, **kwargs)  # type: ignore[arg-type]
    result_arr = _np.asarray(result)
    cost = max(filt_arr.size - result_arr.size, 0)  # num trimmed
    with budget.deduct("trim_zeros", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(trim_zeros, _np.trim_zeros, "free", "0 FLOPs")


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
    result = _np.unpackbits(_to_base_ndarray(a), *args, **kwargs)  # type: ignore[arg-type]
    cost = (
        result.size
        if hasattr(result, "size")
        else len(result)
        if hasattr(result, "__len__")
        else 1
    )
    with budget.deduct(
        "unpackbits", flop_cost=cost, subscripts=None, shapes=(result.shape,)
    ):
        pass  # numpy call already executed above
    return result  # type: ignore[return-value]


attach_docstring(unpackbits, _np.unpackbits, "free", "0 FLOPs")


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
            "unstack", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,)
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
