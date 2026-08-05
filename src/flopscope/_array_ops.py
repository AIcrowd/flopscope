"""NumPy array creation, manipulation, and indexing wrappers.

Wraps NumPy's array-creation, shape-manipulation, and indexing routines.
Per-op FLOP cost is set by the registry / weights table, NOT by this module:
many ops here are billed (e.g. ``arange``, ``linspace``, ``nonzero``, ``isnan``),
while data-movement and constant-init ops are free (weight 0). Free ops still
route through ``budget.deduct(..., flop_cost=0)`` so their time is accounted.
"""

from __future__ import annotations

import inspect as _inspect
import itertools as _itertools
import math as _math
import operator as _operator
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as _np
from numpy.typing import ArrayLike, DTypeLike

from flopscope import _symmetry_transport as _st
from flopscope._budget import (
    _DTYPE_SCAN_MAX_DEPTH,
    _call_numpy,
    _call_user_code,
    _counted_wrapper,
)
from flopscope._docstrings import attach_docstring
from flopscope._dtype_billing import (
    billing_operand,
    refuse_non_numeric_dtype,
    store_billing_dtypes,
)
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
from flopscope._validation import _normalize_out, require_budget
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

# Plain object dtype -- the one dtype safe to forward into array()'s cost
# probe (see array() below). Deliberately an equality check, not
# `hasobject`: a structured dtype can carry `hasobject` via one field while
# another is numeric, and constructing that from a sequence would coerce
# the numeric field's source values too.
_OBJECT_DTYPE = _np.dtype(object)


@_counted_wrapper
def array(
    object: ArrayLike,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Create an array. Cost: numel(output)."""
    budget = require_budget()
    # Pre-compute cost from input to keep numpy call inside the timer. The
    # probe must never perform an object->numeric cast (arbitrary participant
    # code, and by the time it returned there would be nothing object-shaped
    # left for `refuse_non_numeric_dtype` below to catch), so `dtype` is only
    # forwarded into the probe when it resolves to plain ``object`` -- a
    # reference store, never element coercion. Every other case probes with
    # no dtype at all, so `_probe.dtype` reflects the source's true dtype.
    # This also lets a ragged `dtype=object` request succeed here, as plain
    # numpy allows, so the ban's own message fires instead of numpy's raw
    # "inhomogeneous shape" error (see test_object_dtype_ban.py's ragged/
    # duck-typed-dtype tests). `_np.dtype(dtype)` itself only runs numpy's
    # documented dtype protocol, not speculative probing of an untrusted
    # property.
    _requested_dtype = _np.dtype(dtype) if dtype is not None else None
    _probe_dtype = _requested_dtype if _requested_dtype == _OBJECT_DTYPE else None
    _probe = (
        _np.asarray(object, dtype=_probe_dtype)
        if _probe_dtype is not None
        else _np.asarray(object)
    )
    # An explicit dtype= drops the source dtype from the billing tuple below,
    # so check the source independently of what feeds the rate -- converting
    # it to a numeric dtype would otherwise slip an object source past the
    # ban. Pass the already-resolved `_requested_dtype`, not the raw `dtype`
    # arg, so a duck-typed dtype-like object's `.dtype` property is read only
    # once per call (see `_plain_dtype_like`'s docstring).
    refuse_non_numeric_dtype("array", _probe.dtype, _requested_dtype)
    cost = max(_probe.size, 1)
    with budget.deduct(
        "array",
        flop_cost=cost,
        subscripts=None,
        shapes=(_probe.shape,),
        dtypes=(_requested_dtype if _requested_dtype is not None else _probe.dtype,),
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
    # Deduct BEFORE allocating: cost and dtype are fully determined by the
    # request, so an over-budget ``fnp.ones((10**12,))`` is rejected by FLOP
    # accounting before numpy allocates the buffer. Negative dims clamp the
    # cost to 0 and numpy raises inside the deduct as before.
    dims = (shape,) if isinstance(shape, (int, _np.integer)) else tuple(shape)
    cost = max(int(_math.prod(int(d) for d in dims)), 0)
    with budget.deduct(
        "ones", flop_cost=cost, subscripts=None, shapes=(), dtypes=(_np.dtype(dtype),)
    ):
        result = _call_numpy(_np.ones, shape, dtype=dtype, **kwargs)
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
    # Deduct BEFORE allocating (see ``ones``): cost from the requested shape,
    # dtype from the argument or numpy's fill_value inference.
    dims = (shape,) if isinstance(shape, (int, _np.integer)) else tuple(shape)
    cost = max(int(_math.prod(int(d) for d in dims)), 0)
    # Probe fill_value's OWN dtype with no dtype forced (stores pointers,
    # never casts) independently of what feeds the rate below -- an explicit
    # numeric dtype= would otherwise drop fill_value's true dtype from the
    # billing tuple entirely, the same gap array()'s probe closes for its
    # source argument.
    _fill_probe_dtype = _np.asarray(fill_value).dtype
    refuse_non_numeric_dtype("full", _fill_probe_dtype)
    _billing_dtype = _np.dtype(dtype) if dtype is not None else _fill_probe_dtype
    with budget.deduct(
        "full", flop_cost=cost, subscripts=None, shapes=(), dtypes=(_billing_dtype,)
    ):
        result = _call_numpy(_np.full, shape, fill_value, dtype=dtype, **kwargs)
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
    - 2-D input (extract): 0 — ``numpy.diag`` returns a VIEW of the input
      (unconditionally, confirmed in ``tests/test_view_semantics_lock.py``),
      so no elements are copied.
    - 1-D input (construct): ``v.shape[0]`` — one write per input value; the
      zero background is free.
    """
    budget = require_budget()
    v = _np.asarray(v)
    if v.ndim == 1:
        # Constructing diagonal matrix: one write per input value.
        cost = v.shape[0]
    else:
        # Extracting diagonal: a view, nothing copied.
        cost = 0
    with budget.deduct(
        "diag", flop_cost=cost, subscripts=None, shapes=(v.shape,), dtypes=(v.dtype,)
    ):
        result = _call_numpy(_np.diag, v, k=k)
    symmetry = _infer_structural_constructor_symmetry(kind="diag", k=k, v_ndim=v.ndim)
    if symmetry is not None:
        return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]
    return result  # type: ignore[return-value]


attach_docstring(diag, _np.diag, "counted_custom", "0 (2-D view); v.shape[0] (1-D)")


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
    with budget.deduct_after("linspace", subscripts=None, shapes=(), dtypes=()) as _op:
        result = _call_numpy(  # type: ignore[arg-type, call-overload]
            _np.linspace,
            _to_base_ndarray(start) if hasattr(start, "__array__") else start,
            _to_base_ndarray(stop) if hasattr(stop, "__array__") else stop,
            num=num,
            **kwargs,
        )
        samples: Any = result[0] if isinstance(result, tuple) else result
        # Bill the dtype numpy actually PRODUCED: with ``dtype`` omitted,
        # integer endpoints still yield float64 samples, so resolving from
        # the inputs (result_type(start, stop) -> int rate 1) would bill
        # half the float64 rate. Reading it off the result mirrors numpy's
        # inference exactly, on every numpy version.
        _op.set_dtypes((samples.dtype,) if hasattr(samples, "dtype") else ())
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
    base = _np.asarray(_to_base_ndarray(a))
    # Deduct BEFORE allocating (see ``ones``): cost from the base's size (or
    # the ``shape=`` override), dtype from the argument or the base.
    _shape_override = kwargs.get("shape")
    if _shape_override is None:
        cost = int(base.size)
    else:
        dims = (
            (_shape_override,)
            if isinstance(_shape_override, (int, _np.integer))
            else tuple(_shape_override)
        )
        cost = max(int(_math.prod(int(d) for d in dims)), 0)
    _billing_dtype = _np.dtype(dtype) if dtype is not None else base.dtype
    with budget.deduct(
        "ones_like",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(_billing_dtype,),
    ):
        result = _call_numpy(_np.ones_like, base, dtype=dtype, **kwargs)
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
    # Unlike full(), the default billing dtype here is `a`'s (the shape
    # template), never fill_value's -- so fill_value must be checked on its
    # own regardless of whether dtype= was given at all. Probed with no
    # dtype forced, so this never casts.
    refuse_non_numeric_dtype("full_like", _np.asarray(fill_value).dtype)
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
    # ``out`` arrives through **kwargs here, so it reached neither the
    # normalization every declared-parameter sibling gets nor the billing
    # fold below. See the note on the destination dtype further down.
    out = _normalize_out(kwargs.pop("out", None), "concatenate")
    arr_list = [_np.asarray(a) for a in arrays]
    cost = max(sum(a.size for a in arr_list), 1)
    groups = [(a.symmetry if isinstance(a, SymmetricTensor) else None) for a in arrays]
    raw_arrs = [_to_base_ndarray(a) for a in arrays]
    # numpy's default casting for concatenate is ``same_kind``, which admits
    # float -> complex, so the copy loop that actually runs is the
    # DESTINATION's: joining float32 blocks into a complex128 buffer converts
    # every element on the way in (measured 1.100 ms against 0.225 ms for a
    # float32 destination, and 0.886 ms for the same join with complex128
    # inputs). Folding the destination into the rate prices that loop --
    # 80,000 for the case that used to bill 20,000, exactly what the
    # all-complex128 join bills.
    billing_dtypes = tuple(a.dtype for a in arr_list) + store_billing_dtypes(out)
    if out is not None:
        # Unstripped, a FlopscopeArray destination reaches numpy still wrapped
        # and trips the internal "reached numpy.concatenate from inside an fnp
        # wrapper" guard -- which used to happen AFTER the deduct, at full
        # price. Passing it as ``out=`` also lets _call_numpy record the write,
        # voiding any symmetry tag that observed the destination's buffer.
        kwargs["out"] = _to_base_ndarray(out)
    with budget.deduct(
        "concatenate",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=billing_dtypes,
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
    if out is not None:
        # numpy.concatenate returns the destination, so hand back the caller's
        # own object rather than a fresh view of it. A destination never
        # carries a symmetry tag out: a tag is a billing discount, and this op
        # cannot verify one against a buffer the caller already owns and may
        # keep writing to.
        return out  # type: ignore[return-value]
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
    # See ``concatenate`` above: ``out`` arrives through **kwargs, so it
    # reached neither the normalization nor the destination-dtype fold, and an
    # unstripped FlopscopeArray destination tripped the internal strip guard
    # after the deduct rather than before it.
    out = _normalize_out(kwargs.pop("out", None), "stack")
    arr_list = [_np.asarray(a) for a in arrays]
    cost = max(sum(a.size for a in arr_list), 1)
    groups = [a.symmetry if isinstance(a, SymmetricTensor) else None for a in arrays]
    billing_dtypes = tuple(a.dtype for a in arr_list) + store_billing_dtypes(out)
    if out is not None:
        kwargs["out"] = _to_base_ndarray(out)
    with budget.deduct(
        "stack",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=billing_dtypes,
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
    if out is not None:
        return out  # type: ignore[return-value]
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
def ravel(a: ArrayLike, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Flatten array. Cost: numel(input) (= numel(output); ravel does not change element count).

    Accepts ``order`` either positionally or by keyword (``np.ravel(a, 'F')``
    and ``np.ravel(a, order='F')`` both work), matching ``numpy.ravel``'s own
    signature -- required so the ``.ravel()`` ndarray-method override can
    forward its args unchanged regardless of call style.
    """
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
        result = _call_numpy(_np.ravel, a_arr, *args, **kwargs)
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
def copy(a: ArrayLike, *args: Any, **kwargs: Any) -> FlopscopeArray:
    """Return copy of array. Wraps ``numpy.copy``. Cost: numel(input).

    Accepts ``order`` either positionally or by keyword (``np.copy(a, 'F')``
    and ``np.copy(a, order='F')`` both work), matching ``numpy.copy``'s own
    signature -- required so the ``.copy()`` ndarray-method override can
    forward its args unchanged regardless of call style.
    """
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
        result = _call_numpy(_np.copy, a_arr, *args, **kwargs)
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

    Cost: 3-arg form scans and writes the whole broadcast result, charged
    ``4 * numel(broadcast(cond, x, y))`` at the select tier (weight 4.0). 1-arg
    form (``where(condition)`` == ``nonzero``) derives indices by testing
    values, so it bills identically to ``nonzero`` (numel, weight 1.0).
    """
    budget = require_budget()
    cond_arr = _np.asarray(condition)
    if x is None and y is None:
        # 1-arg where IS nonzero (same values-derive-indices computation), so
        # it deducts under nonzero's op name -- alias parity, not a separate
        # price: this call bills identically to fnp.nonzero(condition).
        with budget.deduct(
            "nonzero",
            flop_cost=cond_arr.size,
            subscripts=None,
            shapes=(cond_arr.shape,),
            dtypes=(cond_arr.dtype,),
        ):
            result = _call_numpy(_np.where, _to_base_ndarray(condition))
    else:
        # 3-arg: selection by a given mask still tests every output element
        # and writes the full broadcast result -- charged at the select tier.
        x_arr = _np.asarray(x)
        y_arr = _np.asarray(y)
        out_numel = max(
            int(
                _np.prod(_np.broadcast_shapes(cond_arr.shape, x_arr.shape, y_arr.shape))
            ),
            1,
        )
        with budget.deduct(
            "where",
            flop_cost=out_numel,
            subscripts=None,
            shapes=(cond_arr.shape, x_arr.shape, y_arr.shape),
            dtypes=(x_arr.dtype, y_arr.dtype),
        ):
            result = _call_numpy(
                _np.where,
                _to_base_ndarray(condition),
                _to_base_ndarray(x),  # type: ignore[arg-type, call-overload]
                _to_base_ndarray(y),  # type: ignore[arg-type, call-overload]
            )
    return result  # type: ignore[return-value]


attach_docstring(
    where,
    _np.where,
    "counted_custom",
    "numel(cond) FLOPs at nonzero's weight (1-arg); 4 * numel(output) FLOPs (3-arg)",
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
    """Normalize a ``pad_width``/``stat_length`` argument into a list of
    ``(before, after)`` pairs, one per axis.

    Deliberately mirrors ``numpy.pad``'s own ``_as_pairs(..., as_index=True)``
    broadcasting -- round to integer, then broadcast to shape ``(ndim, 2)`` --
    including its scalar fast path, its single-``(before, after)``-pair fast
    path, and its ``(2, 1)`` exclusion (a per-axis column like ``[[1], [2]]``
    broadcasts to ``[[1, 1], [2, 2]]``, NOT to a single ``(1, 2)`` pair). Any
    argument ``numpy.pad`` accepts must normalize here to the SAME widths, and
    any argument it rejects must raise here too: a normalizer narrower than the
    real ``np.pad`` that runs afterward would let a genuinely padded call --
    one that returns a full-size output -- fall through to a 0 FLOP bill.
    """
    x = _np.round(_np.asarray(value)).astype(_np.intp, copy=False)
    if x.ndim < 3:
        if x.size == 1:
            flat = x.ravel()
            return [(int(flat[0]), int(flat[0]))] * ndim
        if x.size == 2 and x.shape != (2, 1):
            flat = x.ravel()
            return [(int(flat[0]), int(flat[1]))] * ndim
    pairs = _np.broadcast_to(x, (ndim, 2))
    return [(int(pairs[i, 0]), int(pairs[i, 1])) for i in range(ndim)]


def _pad_flop_cost(in_shape, pad_width, mode, kwargs):
    """flop_cost for np.pad: numpy allocates a fresh output and writes EVERY
    cell (interior copy + border fill), so every mode bills a numel(output)
    base; movement modes add nothing on top, linear_ramp/odd-reflect add the
    border's second write pass, and the stat modes add their reduction cost.
    """
    ndim = len(in_shape)
    numel_in = _math.prod(in_shape) if ndim else 1
    # _pad_pairs mirrors np.pad's own normalization exactly, so it accepts every
    # pad_width numpy accepts (yielding the true output size, hence a correct
    # nonzero cost) and raises on every pad_width numpy rejects (surfacing
    # numpy's own broadcasting error before any budget deduction). No fallback
    # swallow here: returning 0 for a pad_width numpy would go on to pad in full
    # was a budget bypass (real output, zero bill).
    pad_pairs = _pad_pairs(pad_width, ndim)
    numel_out = (
        _math.prod(s + b + a for s, (b, a) in zip(in_shape, pad_pairs, strict=False))
        if ndim
        else 1
    )
    if mode in _PAD_FREE_MODES:
        return max(numel_out, 1)
    if mode in ("reflect", "symmetric") and kwargs.get("reflect_type", "even") != "odd":
        return max(numel_out, 1)
    if mode in ("reflect", "symmetric"):  # reflect_type == "odd" (even handled above)
        return max(numel_out + (numel_out - numel_in), 1)
    if mode == "linear_ramp":
        return max(numel_out + (numel_out - numel_in), 1)
    if mode in _PAD_STAT_MODES:
        if numel_in == 0:
            # numpy.pad takes an entirely different code path when the input
            # has 0 elements (any axis length 0): it only checks that every
            # zero-length axis keeps pad_width (0, 0) -- raising if not --
            # then returns the allocated buffer as-is. The per-axis
            # stat_func loop below is never reached in that case, so there
            # is no reduction cost at all, regardless of other axes' pad or
            # stat_length widths.
            return max(numel_out, 1)
        stat_length = kwargs.get("stat_length", None)
        if stat_length is None:
            stat_pairs = [(in_shape[i], in_shape[i]) for i in range(ndim)]
        else:
            stat_pairs = _pad_pairs(stat_length, ndim)
        cost = 0
        # numpy pads axes in order 0..ndim-1, mutating the SAME output buffer
        # in place: by the time axis i's stat is computed, every earlier axis
        # is already grown to its final padded size (and filled with valid
        # data), while later axes are still at their original size (their
        # border is uninitialized until their own turn). So the cross-section
        # each axis reduces over is prod(grown axes before it) * prod(original
        # axes after it) -- NOT the static numel_in // axis_len this used to
        # assume.
        grown = list(in_shape)
        for i in range(ndim):
            before, after = pad_pairs[i]
            axis_len = in_shape[i]
            cross = _math.prod(grown[:i]) * _math.prod(in_shape[i + 1 :])
            sl_b = min(stat_pairs[i][0], axis_len)
            sl_a = min(stat_pairs[i][1], axis_len)
            # numpy ALWAYS computes the left-side reduction for every axis,
            # and a separate right-side reduction unless both sides read the
            # identical full axis (in which case it reuses the left result) --
            # this happens regardless of before/after being 0. A (0, 0) axis
            # still gets reduced; the result is just discarded into a
            # width-0 output region instead of being written anywhere.
            if sl_b == sl_a == axis_len:
                stats = [axis_len]
            else:
                stats = [sl_b, sl_a]
            cost += cross * sum(stats)
            if mode == "mean":
                cost += cross * len(stats)  # one divide per stat output cell
            grown[i] = axis_len + before + after
        return max(numel_out + cost, 1)
    return max(numel_out, 1)  # unknown string mode: let numpy raise its own ValueError


@_counted_wrapper
def pad(
    array: ArrayLike, pad_width: Any, mode: Any = "constant", **kwargs: Any
) -> FlopscopeArray:
    """Pad an array. Cost: numel(output) + mode extras (movement 0;
    linear_ramp/odd +(out-in); stat modes +stat cost); mode=<callable> raises."""
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
    "numel(output) + mode extras (movement 0; linear_ramp/odd +(out-in); "
    "stat modes +stat cost); mode=<callable> raises",
)


def _triangle_kept(m: int, n: int, k: int, upper: bool) -> int:
    """Count of an ``m x n`` matrix's elements kept by ``triu``/``tril`` at offset ``k``.

    ``upper=True`` mirrors ``triu``: row ``i`` keeps columns
    ``[max(i+k, 0), n)``. ``upper=False`` mirrors ``tril``: row ``i`` keeps
    columns ``[0, min(i+k+1, n))``. Matches ``numpy.tri``'s own
    diagonal-offset convention (the mask both wrappers are built from)
    exactly, so it reproduces numpy's actual kept-element count for any
    ``m``, ``n``, ``k`` -- including ``k`` fully off either edge, where it
    correctly returns 0.
    """
    kept = 0
    for i in range(m):
        if upper:
            kept += n - min(max(i + k, 0), n)
        else:
            kept += min(max(i + k + 1, 0), n)
    return kept


@_counted_wrapper
def triu(m: ArrayLike, k: int = 0) -> FlopscopeArray:
    """Upper triangle. Wraps ``numpy.triu``.

    Cost (weight 1.0): elements at/above the kth diagonal. A 1-D (or lower)
    input is promoted to a square 2-D output by numpy itself; any leading
    batch dimensions on a >=2-D input multiply the per-matrix count in.
    Floored at 1 (an out-of-range ``k`` keeps zero elements but the op still
    ran).
    """
    budget = require_budget()
    _warn_if_symmetric(m, "triu")
    m_arr = _np.asarray(m)
    with budget.deduct_after(
        "triu", subscripts=None, shapes=(), dtypes=(m_arr.dtype,)
    ) as _op:
        result = _call_numpy(_np.triu, _to_base_ndarray(m), k=k)
        rows, cols = result.shape[-2], result.shape[-1]
        lead = result.size // (rows * cols) if rows and cols else 0
        _op.set_cost(max(lead * _triangle_kept(rows, cols, k, upper=True), 1))
    return result  # type: ignore[return-value]


attach_docstring(triu, _np.triu, "counted_custom", "elements at/above kth diagonal")


@_counted_wrapper
def tril(m: ArrayLike, k: int = 0) -> FlopscopeArray:
    """Lower triangle. Wraps ``numpy.tril``.

    Cost (weight 1.0): elements at/below the kth diagonal. A 1-D (or lower)
    input is promoted to a square 2-D output by numpy itself; any leading
    batch dimensions on a >=2-D input multiply the per-matrix count in.
    Floored at 1 (an out-of-range ``k`` keeps zero elements but the op still
    ran).
    """
    budget = require_budget()
    _warn_if_symmetric(m, "tril")
    m_arr = _np.asarray(m)
    with budget.deduct_after(
        "tril", subscripts=None, shapes=(), dtypes=(m_arr.dtype,)
    ) as _op:
        result = _call_numpy(_np.tril, _to_base_ndarray(m), k=k)
        rows, cols = result.shape[-2], result.shape[-1]
        lead = result.size // (rows * cols) if rows and cols else 0
        _op.set_cost(max(lead * _triangle_kept(rows, cols, k, upper=False), 1))
    return result  # type: ignore[return-value]


attach_docstring(tril, _np.tril, "counted_custom", "elements at/below kth diagonal")


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

# ``np.astype`` only grew its ``device=`` keyword in numpy 2.1; flopscope
# supports numpy >=2.0. When numpy cannot accept the keyword, reproduce
# numpy>=2.1's device handling (None/"cpu" pass, anything else ValueError,
# raised inside the deduct block -- which charges on entry -- so behavior
# and billing are identical across the supported range).
_NP_ASTYPE_HAS_DEVICE = "device" in _inspect.signature(_np.astype).parameters


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

    Cost: ``numel(input)`` at the heavier of the source/destination dtype
    rate (:func:`_heavier_billing_dtype`) -- the same formula ``copy``
    bills. Charged for every call that performs real work: any dtype
    change, or a same-dtype request with the default ``copy=True``. Free
    only for the genuine no-op -- ``copy=False`` with ``dtype`` already
    equal to ``x``'s dtype -- where NumPy returns the identical object and
    performs no work at all.
    """
    budget = require_budget()
    x_arr = _np.asarray(x)
    resolved_dtype = _np.dtype(dtype)
    # _heavier_billing_dtype below folds source and destination into a SINGLE
    # winning dtype by rate, which silently drops whichever side loses --
    # object always rates 1.0, so any destination with a higher rate (e.g.
    # float64) would erase it from the billing tuple and let an object
    # source escape the ban. Check both sides directly first.
    refuse_non_numeric_dtype("astype", x_arr.dtype, resolved_dtype)
    is_noop = copy is False and resolved_dtype == x_arr.dtype
    cost = 0 if is_noop else x_arr.size
    with budget.deduct(
        "astype",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=(_heavier_billing_dtype(x_arr.dtype, resolved_dtype),),
    ):
        if _NP_ASTYPE_HAS_DEVICE:
            result = _call_numpy(
                _np.astype, _to_base_ndarray(x), dtype, copy=copy, device=device
            )
        else:
            if device is not None and device != "cpu":
                raise ValueError(
                    'Device not understood. Only "cpu" is allowed, '
                    f"but received: {device}"
                )
            result = _call_numpy(_np.astype, _to_base_ndarray(x), dtype, copy=copy)
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

    Cost: ``numel(input)`` at the heavier of the source/destination dtype
    rate -- the same formula and ``copy=False`` no-op carve-out as the
    array-API ``astype`` above (see its docstring).
    """
    budget = require_budget()
    arr_np = _np.asarray(arr)
    resolved_dtype = _np.dtype(dtype)
    # See the array-API astype() above: _heavier_billing_dtype picks a single
    # winner by rate and would silently drop an object source that loses to
    # a higher-rate numeric destination.
    refuse_non_numeric_dtype("astype", arr_np.dtype, resolved_dtype)
    is_noop = copy is False and resolved_dtype == arr_np.dtype
    cost = 0 if is_noop else arr_np.size
    with budget.deduct(
        "astype",
        flop_cost=cost,
        subscripts=None,
        shapes=(arr_np.shape,),
        dtypes=(_heavier_billing_dtype(arr_np.dtype, resolved_dtype),),
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
    """Convert to array.

    Cost: ``numel(output)`` at the heavier of source/destination dtype rate
    when the call materializes a fresh buffer; ``0`` when NumPy returns a
    view of the existing one.
    """
    budget = require_budget()
    _probe = _np.asarray(a)
    # Whether asarray copies depends on more than dtype=: copy=True forces
    # one regardless of dtype, and order="C"/"F" forces one when the input
    # doesn't already conform to that layout. Ground truth is only knowable
    # after the call, so bill from what NumPy actually produced (does the
    # result share memory with the input?) rather than gating on dtype=
    # alone, which misses those copy=/order= materializations.
    with budget.deduct_after(
        "asarray", subscripts=None, shapes=(_probe.shape,), dtypes=(_probe.dtype,)
    ) as _op:
        result = _call_numpy(_np.asarray, a, dtype=dtype, **kwargs)
        base = result  # np.asarray always returns a base ndarray, never a subclass
        materialized = not _np.may_share_memory(base, _probe)
        if materialized:
            _op.set_cost(base.size)
            _op.set_dtypes((_heavier_billing_dtype(_probe.dtype, base.dtype),))
        else:
            _op.set_cost(0)
    return result  # type: ignore[return-value]


attach_docstring(
    asarray,
    _np.asarray,
    "counted_custom",
    "numel(output) FLOPs at the heavier of source/destination dtype rate -- "
    "same formula as copy -- charged whenever the call materializes a fresh "
    "buffer (dtype conversion, copy=True, or an order= that forces a copy); "
    "0 when NumPy returns a view of the existing buffer.",
)


def _counted_predicate(op_name, ufunc, x, kwargs):
    """Shared body of ``isnan`` / ``isinf`` / ``isfinite``.

    These three take ``out=`` through **kwargs, which is how the destination
    escaped normalization, escaped the billing rate, and reached numpy still
    wrapped when a caller passed a FlopscopeArray -- tripping the internal
    strip guard *after* the deduct, at full price.

    On the rate, the destination is folded in, but NOT because numpy runs a
    wider predicate loop -- it does not. Every loop these ufuncs publish ends
    in ``?`` (``np.isnan.types`` is ``'e->?', 'f->?', 'd->?', ...``), and
    ``np.isnan.resolve_dtypes((float32, float64))`` reports ``(float32,
    bool)``: the ``f->?`` loop runs and the bool answer is then cast into the
    caller's buffer. The destination is a cast target, nothing more.

    It is folded in because that cast is real, measurable work, not a
    bookkeeping detail. Over 4e6 float32 values: into a bool destination
    0.179 ms, into a float64 destination 1.314 ms, and the explicit two-step
    -- predicate into bool (0.179 ms) then ``bool.astype(float64)``
    (1.090 ms) -- 1.269 ms. The fused spelling IS the two-step, to within 4%,
    so leaving the destination out of the rate handed back a free ``astype``:
    under the shipped weights the written-out cast bills 20,000 where the
    fused form billed 10,000.

    Folding is the widest-participating-buffer rule from ``_dtype_billing``
    applied unchanged, and what it buys is stated without reference to any
    weights profile: a destination now widens the rate exactly as an equally
    wide OPERAND does, so ``isnan(f32, out=f64)`` bills what ``isnan(f64)``
    bills. A bool destination -- the natural one -- resolves with the input
    and changes nothing.
    """
    budget = require_budget()
    out = _normalize_out(kwargs.pop("out", None), op_name)
    x_arr = _np.asarray(x)
    cost = x_arr.size
    if out is not None:
        kwargs["out"] = _to_base_ndarray(out)
    with budget.deduct(
        op_name,
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=(x_arr.dtype,) + store_billing_dtypes(out),
    ):
        # Strip flopscope subclasses so the raw NumPy ufunc does not
        # re-dispatch through __array_ufunc__ and recurse.
        result = _call_numpy(ufunc, _to_base_ndarray(x), **kwargs)
    # numpy hands the destination back; so do we, as the caller's own object.
    return out if out is not None else result


@_counted_wrapper
def isnan(x: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Test element-wise for NaN. Cost: numel(input)."""
    return _counted_predicate("isnan", _np.isnan, x, kwargs)


attach_docstring(isnan, _np.isnan, "counted_custom", "numel(input) FLOPs")


@_counted_wrapper
def isfinite(x: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Test element-wise for finiteness. Cost: numel(input)."""
    return _counted_predicate("isfinite", _np.isfinite, x, kwargs)


attach_docstring(isfinite, _np.isfinite, "counted_custom", "numel(input) FLOPs")


@_counted_wrapper
def isinf(x: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Test element-wise for Inf. Cost: numel(input)."""
    return _counted_predicate("isinf", _np.isinf, x, kwargs)


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
        # The billed dtype is the promoted dtype of the (possibly deeply
        # nested) leaf blocks -- exactly what ``result.dtype`` already is, so
        # read it off the output instead of re-walking the nested structure.
        # Leaving dtypes=() above would resolve to the dtype-neutral rate 1.0
        # / complex factor 1.0, discounting float64/complex blocks.
        _op.set_dtypes((result.dtype,) if hasattr(result, "dtype") else ())
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
        # See block() above: bill the promoted dtype read off the output --
        # cheaper and more robust than re-deriving it from the raw arguments,
        # which for bmat may be a string referencing named matrices rather
        # than arrays at all.
        _op.set_dtypes((result.dtype,) if hasattr(result, "dtype") else ())
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


@_counted_wrapper
def broadcast_shapes(*args, **kwargs):
    """Broadcast shapes to a common shape. Cost: sum of len(shape) across the
    input shape tuples (floor 1); a bare int argument counts as one axis."""
    budget = require_budget()
    cost = max(sum(len(s) if hasattr(s, "__len__") else 1 for s in args), 1)
    with budget.deduct(
        # dtype-neutral (dtypes=()): pure shape arithmetic, no array operands.
        "broadcast_shapes",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(),
    ):
        result = _call_numpy(_np.broadcast_shapes, *args, **kwargs)
    return result


attach_docstring(
    broadcast_shapes,
    _np.broadcast_shapes,
    "counted_custom",
    "sum of len(shape) across inputs FLOPs",
)


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
    # Args: (a, choices, ...) or just (a, choices) — strip arrays. Kwargs too:
    # ``out=`` arrives as a keyword (e.g. ``ndarray.choose(..., out=arr)``) and
    # an unstripped FlopscopeArray there trips the numpy-entry guard; stripping
    # to the base view writes into the same buffer.
    stripped_args = []
    for arg in args:
        if isinstance(arg, _np.ndarray):
            stripped_args.append(_to_base_ndarray(arg))
        elif isinstance(arg, (tuple, list)):
            stripped_args.append(_to_base_ndarray_tree(arg))
        else:
            stripped_args.append(arg)
    stripped_kwargs = {
        key: (
            _to_base_ndarray(val)
            if isinstance(val, _np.ndarray)
            else _to_base_ndarray_tree(val)
            if isinstance(val, (tuple, list))
            else val
        )
        for key, val in kwargs.items()
    }
    with budget.deduct_after("choose", subscripts=None, shapes=(), dtypes=()) as _op:
        result = _call_numpy(_np.choose, *stripped_args, **stripped_kwargs)
        # Bill the promoted dtype of the choices, read off the output rather
        # than re-derived from *args/**kwargs (choices can arrive positional
        # or keyword, as a list/tuple/array). Leaving dtypes=() above would
        # resolve to the dtype-neutral rate 1.0 / complex factor 1.0,
        # discounting float64/complex choices -- e.g. a complex128 choose
        # would bill 1/4 of an equivalent take.
        _op.set_dtypes((result.dtype,) if hasattr(result, "dtype") else ())
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
    # ``out`` arrives through **kwargs. deduct_after already made a refusal
    # free (__exit__ does not charge on an exception), so what was actually
    # broken here is the strip: a FlopscopeArray destination reached numpy
    # still wrapped and tripped the internal guard. The destination's dtype
    # deliberately does NOT join the rate -- numpy's take/compress require
    # ``can_cast(out.dtype, a.dtype, "safe")``, so a WIDER destination is
    # unreachable (float32 input with a float64 out is a TypeError); only
    # same-or-narrower destinations exist, and those never move the rate.
    out = _normalize_out(kwargs.pop("out", None), "compress")
    _warn_if_symmetric(a, "compress")
    condition_arr = _np.asarray(condition)
    cond_len = condition_arr.size
    a_arr = _np.asarray(a)
    if out is not None:
        kwargs["out"] = _to_base_ndarray(out)
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
    # numpy.compress returns the destination when given one.
    return out if out is not None else result


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
    # concat IS concatenate under another name, and had the same two defects:
    # the destination's dtype never reached the rate (see the measurement in
    # ``concatenate``), and an unstripped FlopscopeArray destination tripped
    # the internal guard. deduct_after already made a refusal free.
    out = _normalize_out(kwargs.pop("out", None), "concat")
    _concat_dtypes = tuple(_np.asarray(a).dtype for a in arrays)
    _concat_dtypes += store_billing_dtypes(out)
    if out is not None:
        kwargs["out"] = _to_base_ndarray(out)
    with budget.deduct_after(
        "concat", subscripts=None, shapes=(), dtypes=_concat_dtypes
    ) as _op:
        result = _call_numpy(
            _np.concat, _to_base_ndarray_tree(arrays), axis=axis, **kwargs
        )  # type: ignore[arg-type, call-overload]
        _op.set_cost(result.size if hasattr(result, "size") else 1)
    return out if out is not None else result  # type: ignore[return-value]


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
        dtypes=(src_arr.dtype,) + store_billing_dtypes(dst_arr),
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


@_counted_wrapper
def diag_indices(*args, **kwargs):
    """Return indices to access main diagonal. Cost: numel of the returned index arrays."""
    budget = require_budget()
    result = _np.diag_indices(*args, **kwargs)
    cost = max(sum(int(r.size) for r in result), 1)
    with budget.deduct(
        # dtype-neutral (dtypes=()): index bookkeeping, same convention as
        # random.permutation / _counted_classes. Passing the int64 index dtype
        # would double every bill via the 2.0 int64 rate.
        "diag_indices",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(),
    ):
        pass
    return result


attach_docstring(
    diag_indices, _np.diag_indices, "counted_custom", "numel(output) FLOPs"
)


@_counted_wrapper
def diag_indices_from(*args, **kwargs):
    """Return indices to access main diagonal of array. Cost: numel of the returned index arrays."""
    budget = require_budget()
    stripped_args = _to_base_ndarray_tree(args)
    result = _np.diag_indices_from(*stripped_args, **kwargs)
    cost = max(sum(int(r.size) for r in result), 1)
    with budget.deduct(
        "diag_indices_from", flop_cost=cost, subscripts=None, shapes=(), dtypes=()
    ):
        pass
    return result


attach_docstring(
    diag_indices_from,
    _np.diag_indices_from,
    "counted_custom",
    "numel(output) FLOPs",
)


@_counted_wrapper
def diagflat(v: ArrayLike, k: int = 0) -> FlopscopeArray:
    """Create diagonal array from flattened input.

    Cost (weight 1.0): ``numel(v)`` — one write per input value; the zero
    background is free.
    """
    budget = require_budget()
    v_arr = _np.asarray(v)
    with budget.deduct_after(
        "diagflat", subscripts=None, shapes=(v_arr.shape,), dtypes=(v_arr.dtype,)
    ) as _op:
        result = _call_numpy(_np.diagflat, _to_base_ndarray(v), k=k)
        _op.set_cost(v_arr.size)
    symmetry = _infer_structural_constructor_symmetry(
        kind="diagflat", k=k, v_ndim=v_arr.ndim
    )
    if symmetry is not None:
        return wrap_with_trusted_symmetry(result, symmetry)  # type: ignore[return-value]
    return result  # type: ignore[return-value]


attach_docstring(diagflat, _np.diagflat, "counted_custom", "numel(v)")


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
    # numpy signature: fromiter(iter, dtype, count=-1, *, like=None) -- dtype
    # has no default. A malformed call (missing dtype, a 4th positional arg,
    # a non-int count) falls straight through to the original raw call so
    # numpy's own arity/type error surfaces unchanged; only a well-formed
    # call takes the probe path below.
    _bound: dict[str, Any] = (
        dict(zip(("iter", "dtype", "count"), args, strict=False))
        if len(args) <= 3
        else {}
    )
    _bound.update(kwargs)
    _count = _bound.get("count", -1)
    _well_formed = (
        len(args) <= 3
        and "dtype" in _bound
        and (_count == -1 or isinstance(_count, (int, _np.integer)))
    )
    if _well_formed:
        _iterable = _bound["iter"]
        _dtype = _bound["dtype"]
        # -1 (or omitted) means "read the whole iterable", matching
        # np.fromiter's own sentinel exactly -- every other value, including
        # every other negative one, bounds how much of the (possibly
        # unbounded) source this materializes.
        _read_all = _count == -1
        _materialized = _call_user_code(
            budget,
            (lambda: list(_iterable))
            if _read_all
            else (lambda: list(_itertools.islice(_iterable, max(int(_count), 0)))),
        )
        # Probe with no dtype forced -- stores object pointers rather than
        # casting, so the probe's dtype reveals the source's true dtype
        # before anything downstream coerces a single element.
        _probe = _np.asarray(_materialized)
        refuse_non_numeric_dtype("fromiter", _probe.dtype, _np.dtype(_dtype))
        _extra = {
            k: v for k, v in _bound.items() if k not in ("iter", "dtype", "count")
        }
        result = _call_user_code(
            budget, _np.fromiter, _materialized, _dtype, count=_count, **_extra
        )
    else:
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

    Cost: numel of the mask that ``mask_func`` produces (the array numpy's
    own ``nonzero(a != 0)`` scans internally), matching the nonzero /
    flatnonzero / argwhere / count_nonzero convention of billing
    numel(input), priced at the mask's own dtype -- floored at n*n, the
    numel of the ``ones((n, n), int)`` probe ``mask_func`` receives as an
    argument, so a ``mask_func`` that captures the probe and returns
    something smaller cannot buy a cheaper scan than that probe's own
    allocation would cost. The index arrays this op returns are not charged
    for -- exactly as ``nonzero`` does not charge for the indices it
    returns. ``numpy`` runs ``mask_func`` internally on its own plain
    (non-flopscope) probe matrix: a plain-numpy callable (e.g. ``np.triu``)
    runs unbilled, while an fnp callable (e.g. ``fnp.triu``) bills its own
    cost separately through its own wrapper, on top of this op's mask-scan
    cost.

    numpy's ``mask_indices`` body ends with a bare top-level ``nonzero(a != 0)``
    on ``a = mask_func(m, k)``. An fnp ``mask_func`` returns a FlopscopeArray,
    which would leak into that internal ``nonzero`` and trip the wrapper-depth
    guard. We coerce the mask_func's return value to a base ndarray before
    numpy continues, capturing its size/shape/dtype for billing along the
    way -- the fnp mask_func still runs and bills its own cost; only its
    result is stripped so numpy's own ``nonzero`` sees a plain array.
    """
    _warn_remote_callback("mask_indices")
    budget = require_budget()

    # numpy calls ``nonzero`` on whatever mask_func returns, so that array is
    # what this op scans. Capture its size/dtype here -- the wrapper already
    # intercepts the return value, so this costs no extra numpy work.
    scanned: dict[str, Any] = {}

    def _strip_mask_func(fn: Any) -> Any:
        if not callable(fn):
            return fn

        def _wrapped(*a: Any, **kw: Any) -> Any:
            out = _to_base_ndarray(fn(*a, **kw))
            mask = _np.asarray(out)
            scanned["size"] = int(mask.size)
            scanned["shape"] = mask.shape
            scanned["dtype"] = mask.dtype
            # Return ``mask``, NOT ``out``: numpy's own body immediately
            # does ``a != 0`` on whatever this callback returns, then
            # ``nonzero()``s the result -- that comparison is where the
            # scan actually happens. ``_to_base_ndarray`` only strips
            # flopscope's own array subclasses; an arbitrary OTHER ndarray
            # subclass (or non-ndarray) passes through untouched, so
            # ``out`` can carry an overridden ``__ne__``/``__eq__`` (or any
            # other dunder numpy's comparison dispatches through) that
            # returns something completely unrelated to what we measured
            # above. ``np.asarray`` always returns a genuine, subclass-free
            # ``np.ndarray`` (a no-op *view* when ``out`` is already a
            # plain, correctly-typed ndarray -- see the parity tests this
            # guards), so forwarding ``mask`` instead guarantees the object
            # numpy's comparison runs against is *exactly* the array whose
            # ``.size`` we just billed -- no dunder of the caller's
            # original ``out`` can run again afterward.
            return mask

        return _wrapped

    # numpy signature: mask_indices(n, mask_func, k=0) -- mask_func is the 2nd
    # positional arg or the `mask_func=` keyword.
    args = list(args)
    # Resolve ``n`` through the integer-index protocol EXACTLY ONCE, before
    # numpy ever sees it, and substitute the plain resolved int back into
    # ``args``/``kwargs``. numpy's own body builds the probe via
    # ``ones((n, n), int)``, which resolves ``n`` through ``__index__``; the
    # billing floor below used to re-read ``n`` a SECOND time via ``int(n)``
    # -- a different protocol -- after numpy had already run. A caller-
    # supplied ``n`` exposing both could report one size to numpy's probe
    # (built and handed to ``mask_func``, so it's exposed either way) and a
    # smaller one to the floor. Resolving once here and handing numpy the
    # same plain int both times it reads the shape closes that gap.
    if len(args) >= 1:
        args[0] = _operator.index(args[0])
    elif "n" in kwargs:
        kwargs["n"] = _operator.index(kwargs["n"])
    # ``n`` not being supplied at all (neither positionally nor by keyword)
    # is left alone here -- the real call below raises numpy's own
    # ``TypeError`` for the missing required argument.
    if len(args) >= 2:
        args[1] = _strip_mask_func(args[1])
    elif "mask_func" in kwargs:
        kwargs["mask_func"] = _strip_mask_func(kwargs["mask_func"])
    # ``mask_func`` is participant code: route it through ``_call_user_code`` so
    # its runtime books to residual, exactly as ``fromfunction`` / ``fromiter`` /
    # ``apply_along_axis`` / ``apply_over_axes`` / ``piecewise`` already do.
    # Calling ``_np.mask_indices`` bare left the callback's entire wall time in
    # ``flopscope_overhead_time_s``, which is excluded from effective compute --
    # i.e. arbitrary user computation ran free of charge on both axes.
    result = _call_user_code(budget, _np.mask_indices, *args, **kwargs)
    # numpy's body is ``m = ones((n, n), int); a = mask_func(m, k); return
    # nonzero(a != 0)`` -- ``mask_func`` receives ``m`` AS AN ARGUMENT, so it
    # can capture that reference and return something arbitrarily small (or
    # unrelated) while still having had the full probe handed to it. The
    # scanned size ``mask_func``'s return value reports is therefore only a
    # ceiling on what got returned, never a floor on what got allocated and
    # exposed -- floor the bill at the probe's own numel (n*n) so capturing
    # it and returning a token-sized result cannot buy a cheaper scan than
    # `fnp.ones((n, n), int)` itself would cost. ``n`` here is the SAME
    # resolved int substituted into ``args``/``kwargs`` above -- not a fresh
    # read of the caller's original object.
    n = args[0] if args else kwargs["n"]
    probe_size = scanned.get("size", n * n)
    if probe_size >= n * n:
        shapes: tuple = (scanned.get("shape", (n, n)),)
        dtypes: tuple = (scanned.get("dtype", _np.dtype(int)),)
    else:
        shapes = ((n, n),)
        dtypes = (_np.dtype(int),)
    cost = max(probe_size, n * n, 1)
    with budget.deduct(
        # Priced as a full value scan of the mask numpy calls ``nonzero`` on,
        # matching the nonzero/flatnonzero/argwhere/count_nonzero convention of
        # billing numel(input), floored at the internal ``ones((n, n), int)``
        # probe's own numel -- that probe is handed to ``mask_func`` as an
        # argument, so it is exposed to (and harvestable by) participant code
        # even though it is never itself returned. The ``a != 0`` compare is
        # not charged on top (``nonzero(a != 0)`` is semantically
        # ``nonzero(a)``). The index arrays this op returns are not charged
        # either -- exactly as ``nonzero`` itself does not charge for the
        # indices it returns.
        "mask_indices",
        flop_cost=cost,
        subscripts=None,
        shapes=shapes,
        dtypes=dtypes,
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(
    mask_indices, _np.mask_indices, "counted_custom", "max(numel(mask), n*n) FLOPs"
)


def matrix_transpose(x: ArrayLike) -> FlopscopeArray:
    """Swap last two axes. Wraps ``numpy.matrix_transpose``. Cost: 0 FLOPs."""
    x_arr = _np.asarray(x)
    # Unlike its siblings (transpose, swapaxes, ...), this function is not
    # decorated with @_counted_wrapper, so the non-numeric-dtype backstop
    # there never runs for it -- check directly. linalg.matrix_transpose
    # delegates here, so this one check covers both registry entries.
    refuse_non_numeric_dtype("matrix_transpose", x_arr.dtype)
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


def _refuse_non_numeric_dtype_tree(
    op_name: str,
    value: Any,
    _depth: int = 0,
    _scanned: set[int] | None = None,
    _active: set[int] | None = None,
) -> None:
    """Recursively refuse a non-numeric-dtype array anywhere inside *value*
    (a bare array, or a list/tuple of arrays).

    Depth-bounded by ``_DTYPE_SCAN_MAX_DEPTH`` for the same reason as
    ``_refuse_non_numeric_operands.check`` in ``_budget.py``, and guarded by
    the identical pair of ``_scanned``/``_active`` id sets -- see that
    function's docstring for the full rationale: ``_scanned`` keeps a
    container reachable through more than one non-cyclic path from being
    walked (and, for a list of arrays, checked) more than once, and
    ``_active`` turns a genuine self-reference into a prompt ``ValueError``
    instead of NumPy re-deriving the same conclusion at exponential cost.
    """
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        refuse_non_numeric_dtype(op_name, dtype)
    elif isinstance(value, (list, tuple)) and _depth < _DTYPE_SCAN_MAX_DEPTH:
        if _scanned is None:
            _scanned = set()
        if _active is None:
            _active = set()
        marker = id(value)
        if marker in _active:
            raise ValueError(
                f"{op_name}: cannot construct an array from a self-referential sequence"
            )
        if marker in _scanned:
            return
        _scanned.add(marker)
        _active.add(marker)
        try:
            for item in value:
                _refuse_non_numeric_dtype_tree(
                    op_name, item, _depth + 1, _scanned, _active
                )
        finally:
            _active.discard(marker)


def permute_dims(*args, **kwargs):
    """Permute dimensions of array. Wraps ``numpy.permute_dims``. Cost: 0 FLOPs."""
    stripped_args = _to_base_ndarray_tree(args)
    # Not decorated with @_counted_wrapper (unlike its sibling movement ops),
    # so the non-numeric-dtype backstop there never runs for it -- check
    # directly.
    for _a in stripped_args:
        _refuse_non_numeric_dtype_tree("permute_dims", _a)
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
    """Convert a multi-index to flat indices. Cost: numel(output) (= N, the
    number of index tuples), dtype-neutral."""
    budget = require_budget()
    stripped = _to_base_ndarray_tree(multi_index)
    idx_arrays = [_np.asarray(a) for a in stripped]
    n = int(_np.broadcast(*idx_arrays).size) if idx_arrays else 0
    with budget.deduct(
        # dtype-neutral (dtypes=()): index bookkeeping, same convention as
        # the tri*_indices family -- passing the int64 index dtype would
        # double the bill.
        "ravel_multi_index",
        flop_cost=n,
        subscripts=None,
        shapes=(),
        dtypes=(),
    ):
        result = _call_numpy(  # type: ignore[arg-type, call-overload]
            _np.ravel_multi_index, stripped, dims, mode=mode, order=order
        )
    return result


attach_docstring(
    ravel_multi_index,
    _np.ravel_multi_index,
    "counted_custom",
    "numel(output) FLOPs",
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
    # a_arr is already the safe, no-dtype probe of `a`'s true source dtype
    # (see array()) -- it was just never checked. Without this, an explicit
    # dtype= drops that probe from the billing tuple, and np.require(a,
    # dtype=...) then casts `a` for real, running any object payload's
    # __float__/__index__/__complex__ per element before anything refuses.
    refuse_non_numeric_dtype("require", a_arr.dtype)
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

    Cost: ``numel(output) * len(condlist)`` — each condition is its own scan
    over the output, at weight 1.0. Billing dtype resolves over the
    choicelist AND default -- default is part of the output wherever every
    condition is False, so it must participate in dtype resolution the same
    as a choice (a plain Python scalar default still promotes weakly, per
    NEP 50).
    """
    budget = require_budget()
    _select_dtypes = tuple(_np.asarray(c).dtype for c in choicelist) + (
        billing_operand(default, _np.asarray(default)),
    )
    with budget.deduct_after(
        "select", subscripts=None, shapes=(), dtypes=_select_dtypes
    ) as _op:
        result = _call_numpy(
            _np.select,
            _to_base_ndarray_tree(condlist),  # type: ignore[arg-type]
            _to_base_ndarray_tree(choicelist),  # type: ignore[arg-type]
            default=default,
        )
        out_numel = result.size if hasattr(result, "size") else 1
        _op.set_cost(out_numel * max(len(list(condlist)), 1))
    return result  # type: ignore[return-value]


attach_docstring(
    select,
    _np.select,
    "counted_custom",
    "numel(output) * len(condlist) FLOPs",
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
    # take already declared ``out`` and already stripped it, but it was the one
    # op in the codebase that accepted ``out=dest`` and refused ``out=(dest,)``
    # -- every sibling unwraps the one-tuple numpy's own ufunc protocol
    # unwraps. Normalizing here makes the two spellings the same call.
    # No destination-dtype fold: numpy requires
    # ``can_cast(out.dtype, a.dtype, "safe")``, so a wider destination cannot
    # be reached (float32 input, float64 out -> TypeError), and a narrower one
    # never moves the rate.
    out = _normalize_out(out, "take")
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
    # numpy.take returns the destination; hand back the caller's own object
    # rather than the stripped view we passed down.
    return out if out is not None else result  # type: ignore[return-value]


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
    """Array with ones at and below the given diagonal. Cost: numel(output)."""
    budget = require_budget()
    result = _np.tri(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct(
        # tri constructs a real float (or requested-dtype) matrix -- NOT
        # dtype-neutral, unlike the sibling index-generator ops below. Bill
        # its actual output dtype, mirroring full/ones/eye/identity.
        "tri",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(result.dtype,),
    ):
        # A triangular matrix is not symmetric — do NOT infer constant-fill symmetry.
        result = _asplainflopscope(result)
    return result


attach_docstring(tri, _np.tri, "counted_custom", "numel(output) FLOPs")


@_counted_wrapper
def tril_indices(*args, **kwargs):
    """Return indices for lower-triangle of array. Cost: numel of the returned index arrays."""
    budget = require_budget()
    result = _np.tril_indices(*args, **kwargs)
    cost = max(sum(int(r.size) for r in result), 1)
    with budget.deduct(
        # dtype-neutral (dtypes=()): index bookkeeping, same convention as
        # random.permutation / _counted_classes. Passing the int64 index dtype
        # would double every bill via the 2.0 int64 rate.
        "tril_indices",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(),
    ):
        pass
    return result


attach_docstring(
    tril_indices, _np.tril_indices, "counted_custom", "numel(output) FLOPs"
)


@_counted_wrapper
def tril_indices_from(*args, **kwargs):
    """Return indices for lower-triangle of given array. Cost: numel of the returned index arrays."""
    budget = require_budget()
    stripped_args = _to_base_ndarray_tree(args)
    result = _np.tril_indices_from(*stripped_args, **kwargs)
    cost = max(sum(int(r.size) for r in result), 1)
    with budget.deduct(
        "tril_indices_from", flop_cost=cost, subscripts=None, shapes=(), dtypes=()
    ):
        pass
    return result


attach_docstring(
    tril_indices_from,
    _np.tril_indices_from,
    "counted_custom",
    "numel(output) FLOPs",
)


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


@_counted_wrapper
def triu_indices(*args, **kwargs):
    """Return indices for upper-triangle of array. Cost: numel of the returned index arrays."""
    budget = require_budget()
    result = _np.triu_indices(*args, **kwargs)
    cost = max(sum(int(r.size) for r in result), 1)
    with budget.deduct(
        "triu_indices", flop_cost=cost, subscripts=None, shapes=(), dtypes=()
    ):
        pass
    return result


attach_docstring(
    triu_indices, _np.triu_indices, "counted_custom", "numel(output) FLOPs"
)


@_counted_wrapper
def triu_indices_from(*args, **kwargs):
    """Return indices for upper-triangle of given array. Cost: numel of the returned index arrays."""
    budget = require_budget()
    stripped_args = _to_base_ndarray_tree(args)
    result = _np.triu_indices_from(*stripped_args, **kwargs)
    cost = max(sum(int(r.size) for r in result), 1)
    with budget.deduct(
        "triu_indices_from", flop_cost=cost, subscripts=None, shapes=(), dtypes=()
    ):
        pass
    return result


attach_docstring(
    triu_indices_from,
    _np.triu_indices_from,
    "counted_custom",
    "numel(output) FLOPs",
)


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


@_counted_wrapper
def unravel_index(*args, **kwargs):
    """Convert flat indices to multi-dimensional index. Cost: numel of the returned index arrays."""
    budget = require_budget()
    stripped_args = _to_base_ndarray_tree(args)
    result = _np.unravel_index(*stripped_args, **kwargs)
    cost = max(sum(int(r.size) for r in result), 1)
    with budget.deduct(
        # dtype-neutral (dtypes=()): index bookkeeping, same convention as
        # random.permutation / _counted_classes. Passing the int64 index dtype
        # would double every bill via the 2.0 int64 rate.
        "unravel_index",
        flop_cost=cost,
        subscripts=None,
        shapes=(),
        dtypes=(),
    ):
        pass
    return result


attach_docstring(
    unravel_index, _np.unravel_index, "counted_custom", "numel(output) FLOPs"
)


if hasattr(_np, "unstack"):

    @_counted_wrapper
    def unstack(x: ArrayLike, *args: Any, **kwargs: Any) -> tuple[FlopscopeArray, ...]:  # pyright: ignore[reportRedeclaration]
        """Split array into sequence of arrays along an axis. Wraps ``numpy.unstack``. Cost: 0 FLOPs."""
        budget = require_budget()
        x_arr = _np.asarray(x)
        with budget.deduct(
            "unstack", flop_cost=0, subscripts=None, shapes=(x_arr.shape,), dtypes=()
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
