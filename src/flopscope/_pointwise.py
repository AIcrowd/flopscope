"""Counted pointwise operations and reductions for flopscope."""

from __future__ import annotations

import builtins as _builtins
import functools as _functools
import inspect as _inspect
import warnings as _warnings
from math import prod as _math_prod
from typing import Any

import numpy as _np
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._config import get_setting as _get_setting
from flopscope._docstrings import attach_docstring
from flopscope._flops import _ceil_log2
from flopscope._flops import (
    analytical_pointwise_cost as pointwise_cost,
)
from flopscope._flops import (
    analytical_reduction_cost as reduction_cost,
)
from flopscope._ndarray import (
    FlopscopeArray,
    _asflopscope,
    _to_base_ndarray,
    _to_base_ndarray_tree,
)
from flopscope._perm_group import _DiminoBudgetExceeded
from flopscope._symmetric import SymmetricTensor
from flopscope._symmetric import is_symmetric as _is_symmetric
from flopscope._symmetry_utils import (
    broadcast_group,
    direct_product_groups,
    intersect_groups,
    reduce_group,
    remap_group_axes,
    restrict_group_to_axes,
    unique_elements_for_shape,
)
from flopscope._validation import maybe_check_nan_inf, require_budget
from flopscope.errors import (
    CostFallbackWarning,
    SymmetryError,
    UnsupportedFunctionError,
    _warn_symmetry_loss,
)

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _symmetry_of(value):
    return value.symmetry if isinstance(value, SymmetricTensor) else None


def _supports_out_argument(np_func) -> bool:
    if isinstance(np_func, _np.ufunc):
        return True
    try:
        return "out" in _inspect.signature(np_func).parameters
    except (TypeError, ValueError):
        return False


def _prepare_symmetric_out(out, target_symmetry):
    if not isinstance(out, SymmetricTensor):
        return target_symmetry
    carried_symmetry = out.symmetry
    if target_symmetry is None:
        raise ValueError("out symmetry does not match result symmetry")
    if carried_symmetry is not None and carried_symmetry != target_symmetry:
        raise ValueError("out symmetry does not match result symmetry")

    if not _is_symmetric(_np.asarray(out), symmetry=target_symmetry):
        axes = target_symmetry.axes
        if axes is None:
            axes = tuple(range(target_symmetry.degree))
        raise SymmetryError(axes=tuple(axes), max_deviation=float("inf"))
    return target_symmetry


def _validate_result_symmetry(result, symmetry):
    if symmetry is None:
        return
    result_arr = _np.asarray(result)
    # Skip numerical validation when the result has non-finite entries:
    # np.allclose treats inf-inf=nan as not-close, which would raise a
    # false SymmetryError. The symmetry was already enforced structurally
    # by the (symmetric) inputs; numerical checks on inf/nan are meaningless.
    if not _np.all(_np.isfinite(result_arr)):
        return
    if not _is_symmetric(result_arr, symmetry=symmetry):
        axes = symmetry.axes
        if axes is None:
            axes = tuple(range(symmetry.degree))
        raise SymmetryError(axes=tuple(axes), max_deviation=float("inf"))


def _is_oversized_for_cost_model(group):
    """``True`` if walking ``group``'s elements would be prohibitively slow.

    Uses ``group.order()`` against the configured ``dimino_budget``.
    For known-kind groups, ``order()`` is O(1) closed form (#71) — the
    check is cheap. For unknown-kind groups, ``order()`` runs ``_dimino``;
    if it exceeds the budget mid-enumeration, ``_DiminoBudgetExceeded``
    raises and we treat the group as oversized.
    """
    if group is None:
        return False
    budget = int(_get_setting("dimino_budget"))  # type: ignore[arg-type]
    try:
        return group.order() > budget
    except _DiminoBudgetExceeded:
        return True


@_functools.cache
def _seen_oversized(op_name: str, group_order: int) -> bool:
    """Return ``True`` once per ``(op, |G|)`` pair, ``False`` thereafter.

    Used by :func:`_warn_oversized_once` to dedup warnings per
    process. The ``lru_cache`` does the deduplication; we use the
    miss-vs-hit discipline at the call site (see that function).
    """
    return True


def _warn_oversized_once(op_name: str, group_order: int) -> None:
    """Emit :class:`CostFallbackWarning` once per ``(op_name, |G|)``.

    Hot paths (e.g. numpy compat tests doing thousands of ufunc calls
    on the same auto-inferred ``S_n`` symmetry) would otherwise spam
    one warning per call. The warning fires once per process for each
    ``(op, |G|)`` pair so users get the diagnostic without log
    flooding.

    Honours ``flops.configure(symmetry_warnings=False)`` — shares the
    flag with :class:`SymmetryLossWarning` since both are
    symmetry-related diagnostics.
    """
    if not _get_setting("symmetry_warnings"):
        return
    info_before = _seen_oversized.cache_info()
    _seen_oversized(op_name, group_order)
    if _seen_oversized.cache_info().hits > info_before.hits:
        return  # already warned for this (op, |G|) pair
    budget = int(_get_setting("dimino_budget"))  # type: ignore[arg-type]
    _warnings.warn(
        f"{op_name}: skipping symmetry-aware cost adjustment for a "
        f"SymmetryGroup of order {group_order} (budget {budget}); "
        f"charging dense cost. Group enumeration would exceed the budget. "
        f"Suppress with flops.configure(symmetry_warnings=False).",
        CostFallbackWarning,
        stacklevel=4,
    )


def _symmetry_adjusted_cost(dense_cost, output_shape, output_symmetry):
    """Scale a dense FLOP cost by the output's symmetry-savings ratio.

    Placeholder model: for an output of shape ``output_shape`` with
    permutation symmetry ``output_symmetry``, the number of *unique*
    elements is at most ``unique_elements_for_shape(output_symmetry,
    output_shape)``. We scale the dense cost by ``unique / dense`` so
    the budget reflects the symmetry savings a symmetry-aware
    implementation could realise.

    For non-symmetric outputs, the ratio is ``1.0`` and ``cost ==
    dense_cost`` (no behaviour change for users without
    SymmetricTensor inputs). For symmetric outputs, the ratio drops
    below 1 and captures redundant-element savings.

    TODO: this is a placeholder. The real algorithmic cost depends on
    whether the underlying NumPy call (or the flopscope wrapper) actually
    skips redundant work — today, our wrappers compute the dense
    output and discard the duplicates. Replace with a per-op
    algorithmic-cost model when one is available.
    """
    if output_symmetry is None:
        return int(dense_cost)
    # Use the Python builtins to avoid the module-level ``max`` /
    # ``prod`` reduction wrappers that shadow them in this module.
    dense_output = _builtins.max(_math_prod(output_shape), 1)
    if dense_output <= 1:
        return int(dense_cost)
    unique = unique_elements_for_shape(output_symmetry, output_shape)
    if unique >= dense_output:
        return int(dense_cost)
    # Integer-division form avoids float drift on large arrays.
    return _builtins.max(int(dense_cost) * int(unique) // dense_output, 1)


def _call_with_optional_out(np_func, *args, out=None, supports_out=False, **kwargs):
    # Strip flopscope subclasses (FlopscopeArray / SymmetricTensor) from arrays so
    # the raw NumPy call does not re-dispatch through ``__array_ufunc__`` /
    # ``__array_function__`` and recurse infinitely. Python scalars and
    # other non-array values pass through unchanged so NEP 50 weak-typing
    # rules continue to apply at the NumPy boundary.
    args = tuple(_to_base_ndarray(a) for a in args)
    # ``where=`` kwarg may be a FlopscopeArray bool mask; strip it. Other
    # array-valued kwargs (e.g. ``axes`` lists for matmul / einsum
    # tensor-axis specs) typically aren't ndarrays, but tree-strip is
    # cheap and safe for nested arg containers.
    for k, v in list(kwargs.items()):
        if isinstance(v, _np.ndarray):
            kwargs[k] = _to_base_ndarray(v)
        elif isinstance(v, (tuple, list)):
            kwargs[k] = _to_base_ndarray_tree(v)
    out_stripped = _to_base_ndarray(out) if out is not None else None
    if out is None:
        return _call_numpy(np_func, *args, **kwargs)
    if supports_out:
        return _call_numpy(np_func, *args, out=out_stripped, **kwargs)
    result = _call_numpy(np_func, *args, **kwargs)
    # Fallback copy when np_func doesn't natively support out=. This is
    # flopscope's overhead, NOT routed through _call_numpy.
    _np.copyto(out_stripped, _np.asarray(result), casting="unsafe")  # type: ignore[arg-type]
    return out


def _call_with_optional_multi_out(np_func, *args, out=None, nout, **kwargs):
    """Multi-output sibling of :func:`_call_with_optional_out`.

    ``out`` is either ``None`` (numpy allocates all outputs) or a tuple of
    length ``nout``. Each slot is either an ndarray write-target or
    ``None`` (let numpy allocate that one slot).

    Returns a tuple of length ``nout``. Identity is preserved per-slot:
    if the caller supplied a non-``None`` array at slot *i*, the
    returned tuple's *i*-th element is exactly the same object. ``None``
    slots are filled with the freshly-allocated plain ndarray that numpy
    returned.
    """
    args = tuple(_to_base_ndarray(a) for a in args)
    for k, v in list(kwargs.items()):
        if isinstance(v, _np.ndarray):
            kwargs[k] = _to_base_ndarray(v)
        elif isinstance(v, (tuple, list)):
            kwargs[k] = _to_base_ndarray_tree(v)
    if out is None:
        return _call_numpy(np_func, *args, **kwargs)
    if not isinstance(out, tuple) or len(out) != nout:
        length_repr = len(out) if hasattr(out, "__len__") else "?"
        raise TypeError(
            f"multi-output {getattr(np_func, '__name__', '?')} requires "
            f"out= to be a tuple of length {nout}; got "
            f"{type(out).__name__} of length {length_repr}"
        )
    stripped = tuple(_to_base_ndarray(o) if o is not None else None for o in out)
    result = _call_numpy(np_func, *args, out=stripped, **kwargs)
    # Numpy returns a tuple of the stripped buffers (or fresh allocations
    # for None slots). Replace each non-None slot with the caller's
    # original to preserve object identity.
    return tuple(
        orig if orig is not None else r for orig, r in zip(out, result, strict=True)
    )


def _wrap_result(result, *, out=None, symmetry=None):
    if out is not None:
        if not isinstance(out, SymmetricTensor):
            _validate_result_symmetry(result, symmetry)
            return out
        effective_symmetry = _prepare_symmetric_out(out, symmetry)
        _validate_result_symmetry(result, effective_symmetry)
        _np.copyto(_np.asarray(out), _np.asarray(result), casting="unsafe")
        return out
    if symmetry is not None:
        return SymmetricTensor(_np.asarray(result), symmetry=symmetry)
    return _asflopscope(result)


def _wrap_multi_result(result, *, out=None, symmetry=None):
    """Wrap each element of a multi-output result tuple.

    For elementwise multi-output ufuncs (``divmod`` / ``frexp`` /
    ``modf``), every output inherits the same ``symmetry`` as the
    (broadcast) input. ``out`` is an optional tuple of caller-provided
    write targets matching ``result`` 1:1; ``None`` slots get fresh
    wrappers, non-``None`` slots get identity + symmetry validation
    routed through :func:`_wrap_result`.
    """
    if not isinstance(result, tuple):
        return _wrap_result(result, out=out, symmetry=symmetry)
    if out is None:
        return tuple(_wrap_result(part, symmetry=symmetry) for part in result)
    return tuple(
        _wrap_result(part, out=o, symmetry=symmetry)
        for part, o in zip(result, out, strict=True)
    )


def _pointwise_symmetry(operands, output_shape):
    aligned_groups = []
    dense_operand_present = False

    for operand, symmetry in operands:
        if operand.ndim == 0:
            continue
        if symmetry is None:
            dense_operand_present = True
            continue
        aligned = broadcast_group(
            symmetry,
            input_shape=operand.shape,
            output_shape=output_shape,
        )
        if aligned is not None:
            aligned_groups.append(aligned)

    if not aligned_groups:
        return None, []
    if dense_operand_present:
        return None, aligned_groups

    output_symmetry = aligned_groups[0]
    for aligned in aligned_groups[1:]:
        output_symmetry = intersect_groups(
            output_symmetry,
            aligned,
            ndim=len(output_shape),
        )
        if output_symmetry is None:
            break
    return output_symmetry, aligned_groups


@_counted_wrapper
def _counted_unary(np_func, op_name: str):
    supports_out = _supports_out_argument(np_func)

    @_counted_wrapper
    def wrapper(
        x: ArrayLike, out: FlopscopeArray | None = None, **kwargs: Any
    ) -> FlopscopeArray:
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        symmetry = _symmetry_of(x)
        symmetry = _prepare_symmetric_out(out, symmetry)
        cost = pointwise_cost(x.shape, symmetry=symmetry)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,)):
            result = _call_with_optional_out(
                np_func,
                x,
                out=None if isinstance(out, SymmetricTensor) else out,
                supports_out=supports_out,
                **kwargs,
            )
        maybe_check_nan_inf(result, op_name)
        return _wrap_result(result, out=out, symmetry=symmetry)  # type: ignore[return-value]

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(output) FLOPs")
    try:
        wrapper.__signature__ = _inspect.signature(np_func)  # pyright: ignore[reportFunctionMemberAccess]
    except (ValueError, TypeError):
        pass
    return wrapper


@_counted_wrapper
def _counted_unary_multi(np_func, op_name: str):
    """Factory for unary functions that return multiple arrays (modf, frexp).

    Supports ``out=(out1, out2)`` (or with ``None`` slots for partial
    allocation) — per-slot stripping and identity preservation are routed
    through :func:`_call_with_optional_multi_out`. Symmetry of the input
    is inherited by every output (elementwise ufuncs).
    """
    nout = getattr(np_func, "nout", 2)

    @_counted_wrapper
    def wrapper(
        x: ArrayLike,
        out: tuple[FlopscopeArray, FlopscopeArray] | None = None,
        **kwargs: Any,
    ) -> tuple[FlopscopeArray, FlopscopeArray]:
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        symmetry = _symmetry_of(x)
        cost = pointwise_cost(x.shape, symmetry=symmetry)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,)):
            result = _call_with_optional_multi_out(
                np_func,
                x,
                out=out,
                nout=nout,
                **kwargs,
            )
        return _wrap_multi_result(result, out=out, symmetry=symmetry)  # type: ignore[return-value]

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(input) FLOPs")
    try:
        wrapper.__signature__ = _inspect.signature(np_func)  # pyright: ignore[reportFunctionMemberAccess]
    except (ValueError, TypeError):
        pass
    return wrapper


@_counted_wrapper
def _counted_binary(np_func, op_name: str):
    supports_out = _supports_out_argument(np_func)

    @_counted_wrapper
    def wrapper(
        x: ArrayLike, y: ArrayLike, out: FlopscopeArray | None = None, **kwargs: Any
    ) -> FlopscopeArray:
        budget = require_budget()
        # Preserve original (possibly Python-scalar) values for the actual
        # numpy call so that NEP 50 weak-typing rules apply correctly. We
        # only need ndarray views for shape and symmetry inspection below.
        x_orig, y_orig = x, y
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        output_shape = _np.broadcast_shapes(x.shape, y.shape)
        x_sym = _symmetry_of(x)
        y_sym = _symmetry_of(y)
        x_is_scalar = x.ndim == 0
        y_is_scalar = y.ndim == 0
        if x_is_scalar ^ y_is_scalar:
            out_symmetry = y_sym if x_is_scalar else x_sym
            aligned_inputs = [out_symmetry] if out_symmetry is not None else []
        else:
            out_symmetry, aligned_inputs = _pointwise_symmetry(
                ((x, x_sym), (y, y_sym)),
                output_shape,
            )
        out_symmetry = _prepare_symmetric_out(out, out_symmetry)

        cost = pointwise_cost(output_shape, symmetry=out_symmetry)
        with budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        ):
            # Call the underlying ufunc with the ORIGINAL inputs so that
            # Python-scalar dtype promotion (NEP 50) and FloatingPointError
            # propagation (np.errstate) work exactly as in plain numpy.
            result = _call_with_optional_out(
                np_func,
                x_orig,
                y_orig,
                out=None if isinstance(out, SymmetricTensor) else out,
                supports_out=supports_out,
                **kwargs,
            )
        maybe_check_nan_inf(result, op_name)
        if out_symmetry is not None:
            lost = []
            for group in aligned_inputs:
                if group != out_symmetry and group.axes is not None:
                    lost.append(group.axes)
            if lost:
                _warn_symmetry_loss(
                    list(dict.fromkeys(lost)),
                    f"{op_name} — groups not shared by both operands",
                )
        else:
            lost = [group.axes for group in aligned_inputs if group.axes is not None]
            if lost:
                _warn_symmetry_loss(
                    list(dict.fromkeys(lost)),
                    f"{op_name} — no symmetry groups shared by both operands",
                )
        return _wrap_result(result, out=out, symmetry=out_symmetry)  # type: ignore[return-value]

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_binary", "numel(output) FLOPs")
    try:
        wrapper.__signature__ = _inspect.signature(np_func)  # pyright: ignore[reportFunctionMemberAccess]
    except (ValueError, TypeError):
        pass
    return wrapper


@_counted_wrapper
def _counted_binary_multi(np_func, op_name: str):
    """Factory for binary functions that return multiple arrays (divmod).

    Mirrors :func:`_counted_binary` for the multi-output case: scalar
    operand special-case, symmetry-loss warning on unshared input
    groups, per-slot ``out=`` identity preservation. Cost is charged
    once (the underlying numpy ufunc produces all outputs in a single
    pass).
    """
    nout = getattr(np_func, "nout", 2)

    @_counted_wrapper
    def wrapper(
        x: ArrayLike,
        y: ArrayLike,
        out: tuple[FlopscopeArray, FlopscopeArray] | None = None,
        **kwargs: Any,
    ) -> tuple[FlopscopeArray, FlopscopeArray]:
        budget = require_budget()
        # Preserve original (possibly Python-scalar) values for the actual
        # numpy call so that NEP 50 weak-typing rules apply correctly. We
        # only need ndarray views for shape and symmetry inspection below.
        x_orig, y_orig = x, y
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        output_shape = _np.broadcast_shapes(x.shape, y.shape)
        x_sym = _symmetry_of(x)
        y_sym = _symmetry_of(y)
        x_is_scalar = x.ndim == 0
        y_is_scalar = y.ndim == 0
        if x_is_scalar ^ y_is_scalar:
            out_symmetry = y_sym if x_is_scalar else x_sym
            aligned_inputs = [out_symmetry] if out_symmetry is not None else []
        else:
            out_symmetry, aligned_inputs = _pointwise_symmetry(
                ((x, x_sym), (y, y_sym)),
                output_shape,
            )
        cost = pointwise_cost(output_shape, symmetry=out_symmetry)
        with budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        ):
            # Pass the ORIGINAL inputs so NEP 50 dtype-promotion rules
            # apply at the NumPy boundary. Stripping happens inside the
            # helper for ndarray-typed values only.
            result = _call_with_optional_multi_out(
                np_func,
                x_orig,
                y_orig,
                out=out,
                nout=nout,
                **kwargs,
            )
        # Symmetry-loss warnings (parity with _counted_binary).
        if out_symmetry is not None:
            lost = []
            for group in aligned_inputs:
                if group != out_symmetry and group.axes is not None:
                    lost.append(group.axes)
            if lost:
                _warn_symmetry_loss(
                    list(dict.fromkeys(lost)),
                    f"{op_name} — groups not shared by both operands",
                )
        else:
            lost = [group.axes for group in aligned_inputs if group.axes is not None]
            if lost:
                _warn_symmetry_loss(
                    list(dict.fromkeys(lost)),
                    f"{op_name} — no symmetry groups shared by both operands",
                )
        return _wrap_multi_result(result, out=out, symmetry=out_symmetry)  # type: ignore[return-value]

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_binary", "numel(output) FLOPs")
    try:
        wrapper.__signature__ = _inspect.signature(np_func)  # pyright: ignore[reportFunctionMemberAccess]
    except (ValueError, TypeError):
        pass
    return wrapper


# ---------------------------------------------------------------------------
# Generic ufunc-method helpers (outer, reduceat, at, generic reduce/accumulate)
# ---------------------------------------------------------------------------


@_counted_wrapper
def _counted_ufunc_outer(ufunc, a, b, *, out=None, **kwargs):
    """Cost-tracked ``ufunc.outer(a, b)`` for any binary ufunc.

    Output shape is ``a.shape + b.shape``; output symmetry is the direct
    product of the input symmetries (with ``b``'s axes lifted by
    ``a.ndim`` so they refer to the correct slots in the combined
    output). Cost is symmetry-adjusted: dense ``a.size * b.size``
    scaled by ``unique / dense`` of the output (see
    :func:`_symmetry_adjusted_cost`).
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    a_sym = _symmetry_of(a)
    b_sym = _symmetry_of(b)
    output_shape = tuple(a.shape) + tuple(b.shape)
    dense = _builtins.max(a.size * b.size, 1)
    # The cost-model branch below enumerates |output_symmetry| group
    # elements; if either input group's |G| exceeds dimino_budget the
    # enumeration would be infeasible (np.ones((1,)*33) → S_33 with
    # 33! ≈ 8.7e36 elements). The cost adjustment is irrelevant when
    # the output is trivially small anyway.
    if _is_oversized_for_cost_model(a_sym) or _is_oversized_for_cost_model(b_sym):
        try:
            oversized_order = (
                a_sym.order() if _is_oversized_for_cost_model(a_sym) else b_sym.order()  # type: ignore[union-attr]
            )
        except _DiminoBudgetExceeded:
            # Unknown-kind group exceeds budget mid-enumeration; can't
            # compute exact |G|. Use sentinel so all such groups share
            # one dedup slot for the warning.
            oversized_order = -1
        _warn_oversized_once(f"{ufunc.__name__}.outer", oversized_order)
        out_sym = None
        cost = dense
    else:
        # Lift ``b``'s symmetry axes into the combined output's slot range.
        b_sym_lifted = b_sym
        if b_sym is not None and b_sym.axes is not None:
            axis_map = {ax: ax + a.ndim for ax in b_sym.axes}
            b_sym_lifted = remap_group_axes(b_sym, axis_map)
        out_sym = direct_product_groups(a_sym, b_sym_lifted)
        cost = _symmetry_adjusted_cost(dense, output_shape, out_sym)
    out_stripped = _to_base_ndarray(out) if out is not None else None
    with budget.deduct(
        f"{ufunc.__name__}.outer",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, b.shape),
    ):
        result = ufunc.outer(
            _to_base_ndarray(a),
            _to_base_ndarray(b),
            out=out_stripped,
            **kwargs,
        )
    return _wrap_result(result, out=out, symmetry=out_sym)


@_counted_wrapper
def _counted_ufunc_reduce_generic(
    ufunc, a, *, axis=0, out=None, keepdims=False, **kwargs
):
    """Cost-tracked fallback for ``ufunc.reduce`` of arbitrary binary ufuncs.

    Used for ufuncs not in :class:`FlopscopeArray._REDUCE_TO_WHEST` (e.g.
    ``subtract``, ``logical_xor``, ``bitwise_or``). Cost equals
    :func:`reduction_cost` (numel of input, or the symmetry-aware
    unique count); output symmetry follows
    :func:`reduce_group(symmetry, ndim, axis, keepdims)`.
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym = _symmetry_of(a)
    cost = reduction_cost(a.shape, axis=axis, symmetry=sym)
    out_sym = (
        reduce_group(sym, ndim=a.ndim, axis=axis, keepdims=keepdims)
        if sym is not None
        else None
    )
    out_stripped = _to_base_ndarray(out) if out is not None else None
    with budget.deduct(
        f"{ufunc.__name__}.reduce",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
    ):
        result = ufunc.reduce(
            _to_base_ndarray(a),
            axis=axis,
            out=out_stripped,
            keepdims=keepdims,
            **kwargs,
        )
    return _wrap_result(result, out=out, symmetry=out_sym)


@_counted_wrapper
def _counted_ufunc_accumulate_generic(ufunc, a, *, axis=0, out=None, **kwargs):
    """Cost-tracked fallback for ``ufunc.accumulate`` of arbitrary binary ufuncs.

    Used for ufuncs not in :class:`FlopscopeArray._ACCUMULATE_TO_WHEST`.
    Cost equals :func:`reduction_cost` (cumulative ops touch every
    element). Output shape matches input shape, but accumulation along
    ``axis`` breaks any permutation symmetry that includes that axis.
    Output symmetry: surviving stabilizer with ``keepdims=True`` (drops
    symmetry on the accumulate axis only).
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym = _symmetry_of(a)
    cost = reduction_cost(a.shape, axis=axis, symmetry=sym)
    out_sym = (
        reduce_group(sym, ndim=a.ndim, axis=axis, keepdims=True)
        if sym is not None
        else None
    )
    out_stripped = _to_base_ndarray(out) if out is not None else None
    with budget.deduct(
        f"{ufunc.__name__}.accumulate",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
    ):
        result = ufunc.accumulate(
            _to_base_ndarray(a),
            axis=axis,
            out=out_stripped,
            **kwargs,
        )
    return _wrap_result(result, out=out, symmetry=out_sym)


@_counted_wrapper
def _counted_ufunc_reduceat(ufunc, a, indices, *, axis=0, out=None, **kwargs):
    """Cost-tracked ``ufunc.reduceat(a, indices, axis=...)``.

    Cost is dense ``numel(input)`` — every element is touched by
    exactly one segment. Output symmetry is ``None``: arbitrary segment
    boundaries don't respect any axis-permutation group action.
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    cost = _builtins.max(int(a.size), 1)
    out_stripped = _to_base_ndarray(out) if out is not None else None
    # Strip ``indices`` only when it's already a flopscope-typed ndarray —
    # otherwise let numpy handle the dtype coercion (e.g. an empty
    # Python list must reach numpy as-is so it doesn't get the float64
    # default that ``np.asarray([])`` would assign).
    indices_stripped = (
        _to_base_ndarray(indices) if isinstance(indices, _np.ndarray) else indices
    )
    with budget.deduct(
        f"{ufunc.__name__}.reduceat",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
    ):
        result = ufunc.reduceat(
            _to_base_ndarray(a),
            indices_stripped,
            axis=axis,
            out=out_stripped,
            **kwargs,
        )
    return _wrap_result(result, out=out, symmetry=None)


@_counted_wrapper
def _counted_ufunc_at(ufunc, a, indices, *args, **kwargs):
    """Cost-tracked ``ufunc.at(a, indices[, values])`` (in-place fancy index).

    ``ufunc.at`` is the in-place unbuffered counterpart to fancy
    indexing — for repeated indices, each application is performed
    rather than deduplicated. The mutation propagates back through
    :func:`_to_base_ndarray`'s zero-copy view-cast.

    **Refusal on SymmetricTensor**: the asymmetric-index write almost
    certainly breaks the tagged symmetry, so we refuse rather than
    silently corrupt metadata. Users can downgrade with
    ``_asplainflopscope(a)`` first if they really want the unbuffered
    update on a view.
    """
    if isinstance(a, SymmetricTensor):
        sym = a.symmetry
        sym_axes = sym.axes if sym is not None else None
        raise ValueError(
            f"in-place ufunc.{ufunc.__name__}.at on a SymmetricTensor would "
            f"break symmetry on axes {sym_axes}; downgrade to plain FlopscopeArray "
            f"(e.g. via ``_asplainflopscope(a)``) before calling "
            f"np.{ufunc.__name__}.at(...)."
        )
    budget = require_budget()
    # ``indices`` can be many things: int, list of ints, ndarray, slice,
    # Ellipsis, or a tuple thereof (for multi-axis fancy indexing).
    # ``ufunc.at`` accepts all of these. Only convert to ndarray when
    # it's already array-like; let scalars / slices / Ellipsis through
    # unchanged so numpy's own semantics apply.
    indices_stripped = (
        _to_base_ndarray(indices) if isinstance(indices, _np.ndarray) else indices
    )
    if isinstance(indices, _np.ndarray):
        n_ops = _builtins.max(int(_np.size(indices)), 1)
    elif hasattr(a, "size"):
        # Conservative for non-array index forms (slice / Ellipsis): use
        # the input size as an upper bound on the touched cells.
        n_ops = _builtins.max(int(a.size), 1)
    else:
        n_ops = 1
    # Strip any flopscope-typed positional values too.
    stripped_args = tuple(
        _to_base_ndarray(v) if isinstance(v, _np.ndarray) else v for v in args
    )
    with budget.deduct(
        f"{ufunc.__name__}.at",
        flop_cost=n_ops,
        subscripts=None,
        shapes=(a.shape,) if hasattr(a, "shape") else (),
    ):
        ufunc.at(
            _to_base_ndarray(a),
            indices_stripped,
            *stripped_args,
            **kwargs,
        )
    return None  # numpy's ufunc.at returns None (mutation is the side effect)


@_counted_wrapper
def _counted_reduction(
    np_func, op_name: str, cost_multiplier: int = 1, extra_output: bool = False
):
    supports_out = _supports_out_argument(np_func)

    # Per-factory signature introspection for positional `out`.
    # NumPy reductions place `out` at different positional slots;
    # method overrides forwarding through ``*args`` need to find it
    # correctly for each underlying function. ``_axis_is_second_positional``
    # tracks whether `axis` is at slot 1 AND positional-acceptable (true for
    # sum/prod/argmax) or otherwise (false for cumulative_sum where axis is
    # KEYWORD_ONLY, and for percentile/quantile whose slot 1 is `q`).
    try:
        _sig_params = _inspect.signature(np_func).parameters
        _params = list(_sig_params)
    except (ValueError, TypeError):
        _sig_params = {}
        _params = []
    _axis_is_second_positional = (
        len(_params) >= 2
        and _params[1] == "axis"
        and _sig_params["axis"].kind
        in (
            _inspect.Parameter.POSITIONAL_ONLY,
            _inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )
    _args_offset = 2 if _axis_is_second_positional else 1
    _out_args_idx = (
        _params.index("out") - _args_offset
        if "out" in _params and _params.index("out") >= _args_offset
        else None
    )

    @_counted_wrapper
    def wrapper(
        a: ArrayLike, axis: int | None = None, *args: Any, **kwargs: Any
    ) -> FlopscopeArray:
        budget = require_budget()
        if not isinstance(a, _np.ndarray):
            a = _np.asarray(a)
        symmetry = a.symmetry if isinstance(a, SymmetricTensor) else None
        keepdims = kwargs.get("keepdims", False)

        # Resolve `out` from either kwargs OR a positional slot in args
        # (per-function — see _out_args_idx computed at factory build time).
        args_list = list(args)
        out = kwargs.pop("out", None)
        out_came_from_args = False
        if (
            out is None
            and _out_args_idx is not None
            and 0 <= _out_args_idx < len(args_list)
            and isinstance(args_list[_out_args_idx], _np.ndarray)
        ):
            out = args_list[_out_args_idx]
            out_came_from_args = True

        new_symmetry = (
            reduce_group(symmetry, ndim=len(a.shape), axis=axis, keepdims=keepdims)
            if symmetry is not None
            else None
        )
        _prepare_symmetric_out(out, new_symmetry)
        cost = reduction_cost(a.shape, axis, symmetry=symmetry) * cost_multiplier
        if extra_output:
            # Pre-compute extra cost from output shape without running numpy yet
            if axis is None:
                extra_cost = 1  # scalar output
            else:
                ax = axis if axis >= 0 else axis + a.ndim
                if keepdims:
                    out_shape = a.shape[:ax] + (1,) + a.shape[ax + 1 :]
                else:
                    out_shape = a.shape[:ax] + a.shape[ax + 1 :]
                extra_cost = pointwise_cost(out_shape)
            cost += extra_cost
        out_for_np = None if isinstance(out, SymmetricTensor) else out
        if out_came_from_args:
            # Stripped out goes back into the same positional slot.
            # _out_args_idx is not None here (out_came_from_args requires it)
            args_list[_out_args_idx] = out_for_np  # type: ignore[index]
            np_out_kwarg = None
            np_supports_out_for_call = False
        else:
            np_out_kwarg = out_for_np
            np_supports_out_for_call = supports_out

        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,)):
            if _axis_is_second_positional:
                result = _call_with_optional_out(
                    np_func,
                    a,
                    axis,
                    *args_list,
                    out=np_out_kwarg,
                    supports_out=np_supports_out_for_call,
                    **kwargs,
                )
            else:
                # axis is keyword-only or at slot 3+; pass via kwargs.
                result = _call_with_optional_out(
                    np_func,
                    a,
                    *args_list,
                    axis=axis,
                    out=np_out_kwarg,
                    supports_out=np_supports_out_for_call,
                    **kwargs,
                )

        # Propagate symmetry through reduction.
        if out is not None:
            return _wrap_result(result, out=out, symmetry=new_symmetry)  # type: ignore[return-value]

        if symmetry is not None:
            if new_symmetry is not None:
                reduced_axes = (
                    set(range(a.ndim))
                    if axis is None
                    else (
                        {axis % a.ndim}
                        if isinstance(axis, int)
                        else {ax % a.ndim for ax in axis}
                    )
                )
                symmetry_axes = (
                    set(symmetry.axes)
                    if symmetry.axes is not None
                    else set(range(symmetry.degree))
                )
                if reduced_axes & symmetry_axes and new_symmetry != symmetry:
                    if symmetry.axes is not None:
                        _warn_symmetry_loss([symmetry.axes], f"{op_name} reduced dims")
            else:
                if symmetry is not None and symmetry.axes is not None:
                    _warn_symmetry_loss(
                        [symmetry.axes],
                        f"{op_name} removed all symmetric dim groups",
                    )
        return _wrap_result(result, symmetry=new_symmetry)  # type: ignore[return-value]

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    cost_desc = (
        f"numel(input) * {cost_multiplier} FLOPs"
        if cost_multiplier > 1
        else "numel(input) FLOPs"
    )
    if extra_output:
        cost_desc += " + numel(output)"
    attach_docstring(wrapper, np_func, "counted_reduction", cost_desc)
    try:
        wrapper.__signature__ = _inspect.signature(np_func)  # pyright: ignore[reportFunctionMemberAccess]
    except (ValueError, TypeError):
        pass
    return wrapper


# ---------------------------------------------------------------------------
# Unary ops (original)
# ---------------------------------------------------------------------------

exp = _counted_unary(_np.exp, "exp")
log = _counted_unary(_np.log, "log")
log2 = _counted_unary(_np.log2, "log2")
log10 = _counted_unary(_np.log10, "log10")
abs = _counted_unary(_np.abs, "abs")
negative = _counted_unary(_np.negative, "negative")
sqrt = _counted_unary(_np.sqrt, "sqrt")
square = _counted_unary(_np.square, "square")
sin = _counted_unary(_np.sin, "sin")
cos = _counted_unary(_np.cos, "cos")
tanh = _counted_unary(_np.tanh, "tanh")
sign = _counted_unary(_np.sign, "sign")
ceil = _counted_unary(_np.ceil, "ceil")
floor = _counted_unary(_np.floor, "floor")

# ---------------------------------------------------------------------------
# Unary ops (new)
# ---------------------------------------------------------------------------

absolute = _counted_unary(_np.absolute, "absolute")
acos = _counted_unary(_np.acos, "acos")
acosh = _counted_unary(_np.acosh, "acosh")
angle = _counted_unary(_np.angle, "angle")
angle.__signature__ = _inspect.signature(_np.angle)  # pyright: ignore[reportFunctionMemberAccess]
arccos = _counted_unary(_np.arccos, "arccos")
arccosh = _counted_unary(_np.arccosh, "arccosh")
arcsin = _counted_unary(_np.arcsin, "arcsin")
arcsinh = _counted_unary(_np.arcsinh, "arcsinh")
arctan = _counted_unary(_np.arctan, "arctan")
arctanh = _counted_unary(_np.arctanh, "arctanh")


@_counted_wrapper
def around(
    a: ArrayLike, decimals: int = 0, out: FlopscopeArray | None = None
) -> FlopscopeArray | Any:
    """Counted version of np.around. Cost = numel(output) FLOPs."""
    budget = require_budget()
    a_is_scalar = not isinstance(a, _np.ndarray) and _np.ndim(a) == 0
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    symmetry = _symmetry_of(a)
    _prepare_symmetric_out(out, symmetry)
    cost = pointwise_cost(a.shape, symmetry=symmetry)
    with budget.deduct("around", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _call_with_optional_out(
            _np.around,
            a,
            decimals=decimals,
            out=None if isinstance(out, SymmetricTensor) else out,
            supports_out=True,
        )
    maybe_check_nan_inf(result, "around")
    if (
        a_is_scalar
        and out is None
        and _np.ndim(result) == 0
        and hasattr(result, "item")
    ):
        return result.item()
    return _wrap_result(result, out=out, symmetry=symmetry)


attach_docstring(around, _np.around, "counted_unary", "numel(output) FLOPs")
asin = _counted_unary(_np.asin, "asin")
asinh = _counted_unary(_np.asinh, "asinh")
atan = _counted_unary(_np.atan, "atan")
atanh = _counted_unary(_np.atanh, "atanh")
if hasattr(_np, "bitwise_count"):
    bitwise_count = _counted_unary(_np.bitwise_count, "bitwise_count")
else:

    def bitwise_count(*args: Any, **kwargs: Any) -> FlopscopeArray:
        raise UnsupportedFunctionError("bitwise_count", min_version="2.1")


bitwise_invert = _counted_unary(_np.bitwise_invert, "bitwise_invert")
bitwise_not = _counted_unary(_np.bitwise_not, "bitwise_not")
cbrt = _counted_unary(_np.cbrt, "cbrt")
conj = _counted_unary(_np.conj, "conj")
conjugate = _counted_unary(_np.conjugate, "conjugate")
cosh = _counted_unary(_np.cosh, "cosh")
deg2rad = _counted_unary(_np.deg2rad, "deg2rad")
degrees = _counted_unary(_np.degrees, "degrees")
exp2 = _counted_unary(_np.exp2, "exp2")
expm1 = _counted_unary(_np.expm1, "expm1")
fabs = _counted_unary(_np.fabs, "fabs")
fix = _counted_unary(_np.fix, "fix")
fix.__signature__ = _inspect.signature(_np.fix)  # pyright: ignore[reportFunctionMemberAccess]
i0 = _counted_unary(_np.i0, "i0")
imag = _counted_unary(_np.imag, "imag")
imag.__signature__ = _inspect.signature(_np.imag)  # pyright: ignore[reportFunctionMemberAccess]
invert = _counted_unary(_np.invert, "invert")
iscomplex = _counted_unary(_np.iscomplex, "iscomplex")
iscomplexobj = _counted_unary(_np.iscomplexobj, "iscomplexobj")
isnat = _counted_unary(_np.isnat, "isnat")
isneginf = _counted_unary(_np.isneginf, "isneginf")
isneginf.__signature__ = _inspect.signature(_np.isneginf)  # pyright: ignore[reportFunctionMemberAccess]
isposinf = _counted_unary(_np.isposinf, "isposinf")
isposinf.__signature__ = _inspect.signature(_np.isposinf)  # pyright: ignore[reportFunctionMemberAccess]
isreal = _counted_unary(_np.isreal, "isreal")
isrealobj = _counted_unary(_np.isrealobj, "isrealobj")
log1p = _counted_unary(_np.log1p, "log1p")
logical_not = _counted_unary(_np.logical_not, "logical_not")
nan_to_num = _counted_unary(_np.nan_to_num, "nan_to_num")
nan_to_num.__signature__ = _inspect.signature(_np.nan_to_num)  # pyright: ignore[reportFunctionMemberAccess]
positive = _counted_unary(_np.positive, "positive")
rad2deg = _counted_unary(_np.rad2deg, "rad2deg")
radians = _counted_unary(_np.radians, "radians")
real = _counted_unary(_np.real, "real")
real.__signature__ = _inspect.signature(_np.real)  # pyright: ignore[reportFunctionMemberAccess]
real_if_close = _counted_unary(_np.real_if_close, "real_if_close")
real_if_close.__signature__ = _inspect.signature(_np.real_if_close)  # pyright: ignore[reportFunctionMemberAccess]
reciprocal = _counted_unary(_np.reciprocal, "reciprocal")
rint = _counted_unary(_np.rint, "rint")


@_counted_wrapper
def round(
    a: ArrayLike, decimals: int = 0, out: FlopscopeArray | None = None
) -> FlopscopeArray | Any:
    """Counted version of np.round. Cost = numel(output) FLOPs."""
    budget = require_budget()
    a_is_scalar = not isinstance(a, _np.ndarray) and _np.ndim(a) == 0
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    symmetry = _symmetry_of(a)
    _prepare_symmetric_out(out, symmetry)
    cost = pointwise_cost(a.shape, symmetry=symmetry)
    with budget.deduct("round", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _call_with_optional_out(
            _np.round,
            a,
            decimals=decimals,
            out=None if isinstance(out, SymmetricTensor) else out,
            supports_out=True,
        )
    maybe_check_nan_inf(result, "round")
    if (
        a_is_scalar
        and out is None
        and _np.ndim(result) == 0
        and hasattr(result, "item")
    ):
        return result.item()
    return _wrap_result(result, out=out, symmetry=symmetry)


attach_docstring(round, _np.round, "counted_unary", "numel(output) FLOPs")
signbit = _counted_unary(_np.signbit, "signbit")
sinc = _counted_unary(_np.sinc, "sinc")
sinh = _counted_unary(_np.sinh, "sinh")


@_counted_wrapper
def sort_complex(a: ArrayLike) -> FlopscopeArray:
    """Counted version of np.sort_complex. Cost: n*ceil(log2(n))."""
    import math

    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.size
    log2n = math.ceil(math.log2(n)) if n > 1 else 1
    cost = n * log2n
    with budget.deduct(
        "sort_complex", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _call_numpy(_np.sort_complex, _to_base_ndarray(a))
    return result  # type: ignore[return-value]  # wrapped at fnp.sort_complex import time


spacing = _counted_unary(_np.spacing, "spacing")
tan = _counted_unary(_np.tan, "tan")
trunc = _counted_unary(_np.trunc, "trunc")

# Multi-output unary ops
modf = _counted_unary_multi(_np.modf, "modf")
frexp = _counted_unary_multi(_np.frexp, "frexp")


# isclose is binary (takes 2 args) but classified as unary in registry
@_counted_wrapper
def isclose(a: ArrayLike, b: ArrayLike, **kwargs: Any) -> FlopscopeArray | bool:
    """Counted version of np.isclose. Cost = numel(output) FLOPs."""
    budget = require_budget()
    a_is_scalar = not isinstance(a, _np.ndarray) and _np.ndim(a) == 0
    b_is_scalar = not isinstance(b, _np.ndarray) and _np.ndim(b) == 0
    # Keep Python scalars as-is so NEP 50 type promotion works correctly
    # (converting them to np.asarray before passing would coerce to float64
    # and break float32 vs Python-float comparisons).
    a_arr = a if isinstance(a, _np.ndarray) else _np.asarray(a)
    b_arr = b if isinstance(b, _np.ndarray) else _np.asarray(b)
    output_shape = _np.broadcast_shapes(a_arr.shape, b_arr.shape)
    out_symmetry, _ = _pointwise_symmetry(
        ((a_arr, _symmetry_of(a_arr)), (b_arr, _symmetry_of(b_arr))),
        output_shape,
    )
    cost = pointwise_cost(output_shape, symmetry=out_symmetry)
    with budget.deduct(
        "isclose", flop_cost=cost, subscripts=None, shapes=(a_arr.shape, b_arr.shape)
    ):
        result = _call_numpy(
            _np.isclose, _to_base_ndarray(a), _to_base_ndarray(b), **kwargs
        )
    if a_is_scalar and b_is_scalar and _np.ndim(result) == 0:
        return bool(result)
    return _wrap_result(result, symmetry=out_symmetry)  # type: ignore[return-value]


attach_docstring(isclose, _np.isclose, "counted_unary", "numel(output) FLOPs")
isclose.__signature__ = _inspect.signature(_np.isclose)  # pyright: ignore[reportFunctionMemberAccess]


# ---------------------------------------------------------------------------
# Binary ops (original)
# ---------------------------------------------------------------------------

add = _counted_binary(_np.add, "add")
subtract = _counted_binary(_np.subtract, "subtract")
multiply = _counted_binary(_np.multiply, "multiply")
divide = _counted_binary(_np.divide, "divide")
maximum = _counted_binary(_np.maximum, "maximum")
minimum = _counted_binary(_np.minimum, "minimum")
power = _counted_binary(_np.power, "power")
mod = _counted_binary(_np.mod, "mod")

# ---------------------------------------------------------------------------
# Binary ops (new)
# ---------------------------------------------------------------------------

arctan2 = _counted_binary(_np.arctan2, "arctan2")
atan2 = _counted_binary(_np.atan2, "atan2")
bitwise_and = _counted_binary(_np.bitwise_and, "bitwise_and")
bitwise_left_shift = _counted_binary(_np.bitwise_left_shift, "bitwise_left_shift")
bitwise_or = _counted_binary(_np.bitwise_or, "bitwise_or")
bitwise_right_shift = _counted_binary(_np.bitwise_right_shift, "bitwise_right_shift")
bitwise_xor = _counted_binary(_np.bitwise_xor, "bitwise_xor")
copysign = _counted_binary(_np.copysign, "copysign")
equal = _counted_binary(_np.equal, "equal")
float_power = _counted_binary(_np.float_power, "float_power")
floor_divide = _counted_binary(_np.floor_divide, "floor_divide")
fmax = _counted_binary(_np.fmax, "fmax")
fmin = _counted_binary(_np.fmin, "fmin")
fmod = _counted_binary(_np.fmod, "fmod")
gcd = _counted_binary(_np.gcd, "gcd")
greater = _counted_binary(_np.greater, "greater")
greater_equal = _counted_binary(_np.greater_equal, "greater_equal")
heaviside = _counted_binary(_np.heaviside, "heaviside")
hypot = _counted_binary(_np.hypot, "hypot")
lcm = _counted_binary(_np.lcm, "lcm")
ldexp = _counted_binary(_np.ldexp, "ldexp")
left_shift = _counted_binary(_np.left_shift, "left_shift")
less = _counted_binary(_np.less, "less")
less_equal = _counted_binary(_np.less_equal, "less_equal")
logaddexp = _counted_binary(_np.logaddexp, "logaddexp")
logaddexp2 = _counted_binary(_np.logaddexp2, "logaddexp2")
logical_and = _counted_binary(_np.logical_and, "logical_and")
logical_or = _counted_binary(_np.logical_or, "logical_or")
logical_xor = _counted_binary(_np.logical_xor, "logical_xor")
nextafter = _counted_binary(_np.nextafter, "nextafter")
not_equal = _counted_binary(_np.not_equal, "not_equal")
pow = _counted_binary(_np.pow, "pow")
remainder = _counted_binary(_np.remainder, "remainder")
right_shift = _counted_binary(_np.right_shift, "right_shift")
true_divide = _counted_binary(_np.true_divide, "true_divide")


if hasattr(_np, "vecdot"):

    @_counted_wrapper
    def vecdot(a: ArrayLike, b: ArrayLike, **kwargs: Any) -> FlopscopeArray:  # pyright: ignore[reportRedeclaration]
        """Counted version of np.vecdot.

        Vector dot product along last axis. Each output element is the dot
        product of two vectors of length K (the last axis), costing K FLOPs.
        Total cost = batch_size * K = numel(a) when a and b have the same shape.
        """
        budget = require_budget()
        if not isinstance(a, _np.ndarray):
            a = _np.asarray(a)
        if not isinstance(b, _np.ndarray):
            b = _np.asarray(b)
        # Cost = output_elements * contracted_axis_size
        # For vecdot, the last axis is contracted.
        contracted = a.shape[-1] if a.ndim > 0 else 1
        out_shape = (
            _np.broadcast_shapes(a.shape[:-1], b.shape[:-1]) if a.ndim > 0 else ()
        )
        cost = (
            _builtins.max(int(_np.prod(out_shape)) * contracted, 1)
            if out_shape
            else contracted
        )
        out = kwargs.pop("out", None)
        out_stripped = _to_base_ndarray(out) if out is not None else None
        with budget.deduct(
            "vecdot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
        ):
            result = _call_numpy(
                _np.vecdot,
                _to_base_ndarray(a),
                _to_base_ndarray(b),
                out=out_stripped,
                **kwargs,
            )
        return out if out is not None else result  # type: ignore[return-value]

else:

    def vecdot(*args: Any, **kwargs: Any) -> FlopscopeArray:  # pyright: ignore[reportRedeclaration]
        raise UnsupportedFunctionError("vecdot", min_version="2.1")


if hasattr(_np, "matvec"):

    @_counted_wrapper
    def matvec(a: ArrayLike, b: ArrayLike, **kwargs: Any) -> FlopscopeArray:  # pyright: ignore[reportRedeclaration]
        """Counted version of np.matvec.

        Matrix-vector product. A is (..., m, n), v is (..., n), result is (..., m).
        Cost = output_size * contracted_axis (A's last axis).
        """
        budget = require_budget()
        if not isinstance(a, _np.ndarray):
            a = _np.asarray(a)
        if not isinstance(b, _np.ndarray):
            b = _np.asarray(b)
        contracted = a.shape[-1] if a.ndim > 0 else 1
        # output shape: (..., m) where m = a.shape[-2]
        out_m = a.shape[-2] if a.ndim >= 2 else 1
        batch = a.shape[:-2] if a.ndim > 2 else ()
        cost = _builtins.max(
            int(_np.prod(batch)) * out_m * contracted if batch else out_m * contracted,
            1,
        )
        out = kwargs.pop("out", None)
        out_stripped = _to_base_ndarray(out) if out is not None else None
        with budget.deduct(
            "matvec", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
        ):
            result = _call_numpy(
                _np.matvec,
                _to_base_ndarray(a),
                _to_base_ndarray(b),
                out=out_stripped,
                **kwargs,
            )
        return out if out is not None else result  # type: ignore[return-value]

else:

    def matvec(*args: Any, **kwargs: Any) -> FlopscopeArray:  # pyright: ignore[reportRedeclaration]
        raise UnsupportedFunctionError("matvec", min_version="2.2")


if hasattr(_np, "vecmat"):

    @_counted_wrapper
    def vecmat(a: ArrayLike, b: ArrayLike, **kwargs: Any) -> FlopscopeArray:  # pyright: ignore[reportRedeclaration]
        """Counted version of np.vecmat.

        Vector-matrix product. v is (..., n), A is (..., n, m), result is (..., m).
        Cost = output_size * contracted_axis (v's last axis).
        """
        budget = require_budget()
        if not isinstance(a, _np.ndarray):
            a = _np.asarray(a)
        if not isinstance(b, _np.ndarray):
            b = _np.asarray(b)
        contracted = a.shape[-1] if a.ndim > 0 else 1
        # output shape: (..., m) where m = b.shape[-1]
        out_m = b.shape[-1] if b.ndim >= 2 else 1
        batch = b.shape[:-2] if b.ndim > 2 else ()
        cost = _builtins.max(
            int(_np.prod(batch)) * out_m * contracted if batch else out_m * contracted,
            1,
        )
        out = kwargs.pop("out", None)
        out_stripped = _to_base_ndarray(out) if out is not None else None
        with budget.deduct(
            "vecmat", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
        ):
            result = _call_numpy(
                _np.vecmat,
                _to_base_ndarray(a),
                _to_base_ndarray(b),
                out=out_stripped,
                **kwargs,
            )
        return out if out is not None else result  # type: ignore[return-value]

else:

    def vecmat(*args: Any, **kwargs: Any) -> FlopscopeArray:  # pyright: ignore[reportRedeclaration]
        raise UnsupportedFunctionError("vecmat", min_version="2.2")


# Multi-output binary ops
divmod = _counted_binary_multi(_np.divmod, "divmod")


# ---------------------------------------------------------------------------
# Special: clip
# ---------------------------------------------------------------------------


@_counted_wrapper
def clip(
    a: ArrayLike, *args: Any, out: FlopscopeArray | None = None, **kwargs: Any
) -> FlopscopeArray:
    """Counted version of np.clip. Cost = numel(input) or unique_elements if symmetric."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    operand_arrays = [(a, _symmetry_of(a))]
    for value in args:
        if value is None:
            continue
        arr = value if isinstance(value, _np.ndarray) else _np.asarray(value)
        operand_arrays.append((arr, _symmetry_of(arr)))
    for key in ("a_min", "a_max", "min", "max"):
        value = kwargs.get(key)
        if value is None:
            continue
        arr = value if isinstance(value, _np.ndarray) else _np.asarray(value)
        operand_arrays.append((arr, _symmetry_of(arr)))
    symmetry, _ = _pointwise_symmetry(operand_arrays, a.shape)
    _prepare_symmetric_out(out, symmetry)
    cost = pointwise_cost(a.shape, symmetry=symmetry)
    with budget.deduct("clip", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        # Delegate all argument handling (validation, min/max/a_min/a_max) to numpy
        result = _call_with_optional_out(
            _np.clip,
            a,
            *args,
            out=None if isinstance(out, SymmetricTensor) else out,
            supports_out=True,
            **kwargs,
        )
    if a.dtype.kind in ("f", "c"):
        maybe_check_nan_inf(result, "clip")
    return _wrap_result(result, out=out, symmetry=symmetry)  # type: ignore[return-value]


attach_docstring(clip, _np.clip, "counted_custom", "numel(input) FLOPs")
clip.__signature__ = _inspect.signature(_np.clip)  # pyright: ignore[reportFunctionMemberAccess]


# ---------------------------------------------------------------------------
# Reductions (original)
# ---------------------------------------------------------------------------

sum = _counted_reduction(_np.sum, "sum")
max = _counted_reduction(_np.max, "max")
min = _counted_reduction(_np.min, "min")
prod = _counted_reduction(_np.prod, "prod")


@_counted_wrapper
def mean(
    a: ArrayLike,
    axis: int | None = None,
    dtype=None,
    out: FlopscopeArray | None = None,
    keepdims: bool = False,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of np.mean.

    Cost = sum_cost (orbit-mapping FLOPs via Tier-1 model)
           + num_output_orbits (one divide per output orbit).
    """
    from flopscope._accumulation._reduction import (
        _normalize_axis,
        _num_output_orbits,
        compute_reduction_accumulation_cost,
    )

    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    symmetry = _symmetry_of(a)
    keepdims = bool(keepdims)

    axes_summed = _normalize_axis(axis, a.ndim)
    num_orbits = _num_output_orbits(tuple(a.shape), axes_summed, symmetry)
    cost = compute_reduction_accumulation_cost(
        input_shape=tuple(a.shape),
        axes_summed=axes_summed,
        symmetry=symmetry,
        op_factor=1,
        extra_ops=num_orbits,  # one divide per output orbit
    ).total

    new_symmetry = (
        reduce_group(symmetry, ndim=a.ndim, axis=axis, keepdims=keepdims)
        if symmetry is not None
        else None
    )
    _prepare_symmetric_out(out, new_symmetry)
    out_for_np = None if isinstance(out, SymmetricTensor) else out

    with budget.deduct("mean", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _call_with_optional_out(
            _np.mean,
            a,
            axis=axis,
            out=out_for_np,
            keepdims=keepdims,
            dtype=dtype,
            supports_out=True,
            **kwargs,
        )

    if out is not None:
        return _wrap_result(result, out=out, symmetry=new_symmetry)  # type: ignore[return-value]
    return _wrap_result(result, symmetry=new_symmetry)  # type: ignore[return-value]


mean.__signature__ = _inspect.signature(_np.mean)  # pyright: ignore[reportFunctionMemberAccess]

std = _counted_reduction(_np.std, "std")
var = _counted_reduction(_np.var, "var")
argmax = _counted_reduction(_np.argmax, "argmax")
argmin = _counted_reduction(_np.argmin, "argmin")
cumsum = _counted_reduction(_np.cumsum, "cumsum")
cumprod = _counted_reduction(_np.cumprod, "cumprod")

# ---------------------------------------------------------------------------
# Reductions (new)
# ---------------------------------------------------------------------------

all = _counted_reduction(_np.all, "all")
amax = _counted_reduction(_np.amax, "amax")
amin = _counted_reduction(_np.amin, "amin")
any = _counted_reduction(_np.any, "any")
average = _counted_reduction(_np.average, "average")
_count_nonzero_counted = _counted_reduction(_np.count_nonzero, "count_nonzero")


def count_nonzero(
    a: ArrayLike, axis: int | tuple[int, ...] | None = None, *, keepdims: bool = False
) -> FlopscopeArray | int:
    """Counted version of ``numpy.count_nonzero``. Cost: numel(input) FLOPs.

    When ``axis is None`` (and not ``keepdims``) the result is always
    coerced to a Python ``int``. This is unconditional because flopscope's
    ``_counted_reduction`` factory wraps scalar results via ``_asflopscope``
    on every numpy version, so without this coercion users would see a
    ``FlopscopeArray`` rather than the plain ``int`` that ``numpy.count_nonzero``
    documents. The coercion also normalizes the numpy 2.3+ change where
    the raw numpy return type became a numpy scalar.
    """
    result = _count_nonzero_counted(a, axis=axis, keepdims=keepdims)  # type: ignore[arg-type]
    if axis is None and not keepdims:
        return int(result)
    return result


attach_docstring(
    count_nonzero, _np.count_nonzero, "counted_reduction", "numel(input) FLOPs"
)
if hasattr(_np, "cumulative_prod"):
    cumulative_prod = _counted_reduction(_np.cumulative_prod, "cumulative_prod")
else:

    def cumulative_prod(*args: Any, **kwargs: Any) -> FlopscopeArray:
        raise UnsupportedFunctionError("cumulative_prod", min_version="2.1")


if hasattr(_np, "cumulative_sum"):
    cumulative_sum = _counted_reduction(_np.cumulative_sum, "cumulative_sum")
else:

    def cumulative_sum(*args: Any, **kwargs: Any) -> FlopscopeArray:
        raise UnsupportedFunctionError("cumulative_sum", min_version="2.1")


def _tier2_reduction_cost(a, axis, dense_per_output_cost: int) -> int:
    """Tier-2 reduction cost for non-ufunc reductions.

    Returns num_output_orbits × dense_per_output_cost. When *a* has no
    symmetry, num_output_orbits == num_output_elems and the cost equals
    the dense baseline.
    """
    from flopscope._accumulation._reduction import (
        _normalize_axis,
        output_discounted_reduction_cost,
    )

    sym = _symmetry_of(a)
    axes_summed = _normalize_axis(axis, a.ndim)
    return output_discounted_reduction_cost(
        input_shape=tuple(a.shape),
        axes_summed=axes_summed,
        symmetry=sym,
        dense_per_output_cost=dense_per_output_cost,
    )


@_counted_wrapper
def median(
    a: ArrayLike,
    axis: int | None = None,
    out: FlopscopeArray | None = None,
    keepdims: bool = False,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of np.median.

    Cost = num_output_orbits × axis_dim (Tier-2 partition-based model).
    """
    import math as _math

    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym = _symmetry_of(a)

    # Dense per-output cost for partition-based median: axis_dim (one pass).
    if axis is None:
        axis_dim = _math.prod(a.shape) if a.shape else 1
    elif isinstance(axis, int):
        axis_dim = a.shape[axis]
    else:
        axis_dim = _math.prod(a.shape[ax] for ax in axis)

    cost = _tier2_reduction_cost(a, axis, dense_per_output_cost=axis_dim)

    out_sym = (
        reduce_group(sym, ndim=a.ndim, axis=axis, keepdims=keepdims)
        if sym is not None
        else None
    )
    out_stripped = _to_base_ndarray(out) if out is not None else None
    with budget.deduct(
        "median",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
    ):
        result = _call_numpy(
            _np.median,
            _to_base_ndarray(a),
            axis=axis,
            out=out_stripped,
            keepdims=keepdims,
            **kwargs,
        )
    return _wrap_result(result, out=out, symmetry=out_sym)  # type: ignore[return-value]


median.__signature__ = _inspect.signature(_np.median)  # pyright: ignore[reportFunctionMemberAccess]

nanargmax = _counted_reduction(_np.nanargmax, "nanargmax")
nanargmin = _counted_reduction(_np.nanargmin, "nanargmin")
nancumprod = _counted_reduction(_np.nancumprod, "nancumprod")
nancumsum = _counted_reduction(_np.nancumsum, "nancumsum")
nanmax = _counted_reduction(_np.nanmax, "nanmax")
nanmean = _counted_reduction(_np.nanmean, "nanmean")
nanmedian = _counted_reduction(_np.nanmedian, "nanmedian")
nanmin = _counted_reduction(_np.nanmin, "nanmin")
nanpercentile = _counted_reduction(_np.nanpercentile, "nanpercentile")
nanprod = _counted_reduction(_np.nanprod, "nanprod")
nanquantile = _counted_reduction(_np.nanquantile, "nanquantile")
nanstd = _counted_reduction(_np.nanstd, "nanstd")
nansum = _counted_reduction(_np.nansum, "nansum")
nanvar = _counted_reduction(_np.nanvar, "nanvar")


@_counted_wrapper
def percentile(
    a: ArrayLike,
    q: float | ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    out: FlopscopeArray | None = None,
    keepdims: bool = False,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of np.percentile.

    Cost = num_output_orbits × axis_dim (Tier-2 partition-based model).
    """
    import math as _math

    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym = _symmetry_of(a)

    # Dense per-output cost for partition-based percentile: axis_dim (one pass).
    if axis is None:
        axis_dim = _math.prod(a.shape) if a.shape else 1
    elif isinstance(axis, int):
        axis_dim = a.shape[axis]
    else:
        axis_dim = _math.prod(a.shape[ax] for ax in axis)

    cost = _tier2_reduction_cost(a, axis, dense_per_output_cost=axis_dim)

    out_sym = (
        reduce_group(sym, ndim=a.ndim, axis=axis, keepdims=keepdims)
        if sym is not None
        else None
    )
    out_stripped = _to_base_ndarray(out) if out is not None else None
    with budget.deduct(
        "percentile",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
    ):
        result = _call_numpy(
            _np.percentile,
            _to_base_ndarray(a),
            q,
            axis=axis,
            out=out_stripped,
            keepdims=keepdims,
            **kwargs,
        )
    return _wrap_result(result, out=out, symmetry=out_sym)  # type: ignore[return-value]


percentile.__signature__ = _inspect.signature(_np.percentile)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def quantile(
    a: ArrayLike,
    q: float | ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    out: FlopscopeArray | None = None,
    keepdims: bool = False,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of np.quantile.

    Cost = num_output_orbits × axis_dim (Tier-2 partition-based model).
    """
    import math as _math

    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym = _symmetry_of(a)

    # Dense per-output cost for partition-based quantile: axis_dim (one pass).
    if axis is None:
        axis_dim = _math.prod(a.shape) if a.shape else 1
    elif isinstance(axis, int):
        axis_dim = a.shape[axis]
    else:
        axis_dim = _math.prod(a.shape[ax] for ax in axis)

    cost = _tier2_reduction_cost(a, axis, dense_per_output_cost=axis_dim)

    out_sym = (
        reduce_group(sym, ndim=a.ndim, axis=axis, keepdims=keepdims)
        if sym is not None
        else None
    )
    out_stripped = _to_base_ndarray(out) if out is not None else None
    with budget.deduct(
        "quantile",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
    ):
        result = _call_numpy(
            _np.quantile,
            _to_base_ndarray(a),
            q,
            axis=axis,
            out=out_stripped,
            keepdims=keepdims,
            **kwargs,
        )
    return _wrap_result(result, out=out, symmetry=out_sym)  # type: ignore[return-value]


quantile.__signature__ = _inspect.signature(_np.quantile)  # pyright: ignore[reportFunctionMemberAccess]

# ptp: numpy 2.0 removed it from ndarray but np.ptp still exists
if hasattr(_np, "ptp"):
    ptp = _counted_reduction(_np.ptp, "ptp")
else:

    @_counted_wrapper
    def ptp(a: ArrayLike, axis: int | None = None, **kwargs: Any) -> FlopscopeArray:
        """Peak-to-peak range. Cost = numel(input) FLOPs."""
        budget = require_budget()
        if not isinstance(a, _np.ndarray):
            a = _np.asarray(a)
        cost = reduction_cost(a.shape, axis)
        with budget.deduct("ptp", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
            stripped = _to_base_ndarray(a)
            result = _call_numpy(_np.max, stripped, axis=axis, **kwargs) - _call_numpy(
                _np.min, stripped, axis=axis, **kwargs
            )
        return result  # type: ignore[return-value]  # wrapped at fnp.ptp import time

    attach_docstring(ptp, _np.max, "counted_reduction", "numel(input) FLOPs")


# ---------------------------------------------------------------------------
# dot and matmul
# ---------------------------------------------------------------------------


@_counted_wrapper
def dot(a: ArrayLike, b: ArrayLike) -> FlopscopeArray:
    """Counted version of np.dot."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    subs: str | None
    if a.ndim == 2 and b.ndim == 2:
        subs = "ij,jk->ik"
    elif a.ndim == 1 and b.ndim == 1:
        subs = "i,i->"
    else:
        subs = None
    if subs is None:
        cost: int = a.size * b.size
        output_sym = None
        canonical_subs: str | None = None
    else:
        from flopscope._einsum import _resolve_cost_and_output_symmetry

        info = _resolve_cost_and_output_symmetry(subs, a, b)
        cost = info.accumulation.total
        output_sym = info.output_symmetry
        canonical_subs = info.canonical_subscripts
    inputs_were_whest = isinstance(a, _np.ndarray) and (
        type(a) is not _np.ndarray or type(b) is not _np.ndarray
    )
    with budget.deduct(
        "dot", flop_cost=cost, subscripts=canonical_subs, shapes=(a.shape, b.shape)
    ):
        result = _call_numpy(_np.dot, _to_base_ndarray(a), _to_base_ndarray(b))
    maybe_check_nan_inf(result, "dot")
    if output_sym is not None:
        _validate_result_symmetry(result, output_sym)
        return SymmetricTensor(_np.asarray(result), symmetry=output_sym)  # type: ignore[return-value]
    return _asflopscope(result) if inputs_were_whest else result  # type: ignore[return-value]


attach_docstring(dot, _np.dot, "counted_custom", "depends on operand dimensions")


@_counted_wrapper
def matmul(a: ArrayLike, b: ArrayLike) -> FlopscopeArray:
    """Counted version of np.matmul."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    # Subscripts for the 2D×2D and 1D×1D cases; everything else falls
    # back to a dense size×size estimate (batched matmul, mixed ndim).
    subs: str | None
    if a.ndim == 2 and b.ndim == 2:
        subs = "ij,jk->ik"
    elif a.ndim == 1 and b.ndim == 1:
        subs = "i,i->"
    else:
        subs = None
    if subs is None:
        cost: int = a.size * b.size
        output_sym = None
        canonical_subs: str | None = None
    else:
        from flopscope._einsum import _resolve_cost_and_output_symmetry

        info = _resolve_cost_and_output_symmetry(subs, a, b)
        cost = info.accumulation.total
        output_sym = info.output_symmetry
        canonical_subs = info.canonical_subscripts
    inputs_were_whest = isinstance(a, _np.ndarray) and (
        type(a) is not _np.ndarray or type(b) is not _np.ndarray
    )
    with budget.deduct(
        "matmul", flop_cost=cost, subscripts=canonical_subs, shapes=(a.shape, b.shape)
    ):
        with _np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            result = _call_numpy(_np.matmul, _to_base_ndarray(a), _to_base_ndarray(b))
    maybe_check_nan_inf(result, "matmul")
    if output_sym is not None:
        _validate_result_symmetry(result, output_sym)
        return SymmetricTensor(_np.asarray(result), symmetry=output_sym)  # type: ignore[return-value]
    return _asflopscope(result) if inputs_were_whest else result  # type: ignore[return-value]


attach_docstring(matmul, _np.matmul, "counted_custom", "depends on operand dimensions")


# ---------------------------------------------------------------------------
# Custom ops (new)
# ---------------------------------------------------------------------------


@_counted_wrapper
def inner(a: ArrayLike, b: ArrayLike) -> FlopscopeArray:
    """Counted version of np.inner."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    subs: str | None
    if a.ndim == 1 and b.ndim == 1:
        subs = "i,i->"
    elif a.ndim == 2 and b.ndim == 2:
        subs = "ij,kj->ik"
    else:
        subs = None
    if subs is None:
        cost: int = (
            a.size
            if (a.ndim <= 1 and b.ndim <= 1)
            else a.size * (b.shape[-1] if b.ndim > 1 else 1)
        )
        output_sym = None
        canonical_subs: str | None = None
    else:
        from flopscope._einsum import _resolve_cost_and_output_symmetry

        info = _resolve_cost_and_output_symmetry(subs, a, b)
        cost = info.accumulation.total
        output_sym = info.output_symmetry
        canonical_subs = info.canonical_subscripts
    with budget.deduct(
        "inner", flop_cost=cost, subscripts=canonical_subs, shapes=(a.shape, b.shape)
    ):
        result = _call_numpy(_np.inner, _to_base_ndarray(a), _to_base_ndarray(b))
    if output_sym is not None:
        _validate_result_symmetry(result, output_sym)
        return SymmetricTensor(_np.asarray(result), symmetry=output_sym)  # type: ignore[return-value]
    return result  # type: ignore[return-value]


attach_docstring(inner, _np.inner, "counted_custom", "product of matching dims")


@_counted_wrapper
def outer(
    a: ArrayLike, b: ArrayLike, out: FlopscopeArray | None = None
) -> FlopscopeArray:
    """Counted version of np.outer."""
    budget = require_budget()
    # Capture aliasing BEFORE asarray conversion so outer(v, v) is detected
    # even when v is a list or other non-ndarray type.
    a_orig_is_b_orig = a is b
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    # outer flattens its inputs to 1-D, so "i,j->ij" is always the right
    # einsum subscripts regardless of original ndim.
    from flopscope._einsum import _resolve_cost_and_output_symmetry

    a_flat = a.ravel() if a.ndim != 1 else a
    b_flat = b.ravel() if b.ndim != 1 else b
    # Preserve operand-aliasing through the asarray boundary: if the user
    # passed the same Python object for both operands, treat them as one
    # array for the helper's identity-pattern detection.
    if a_orig_is_b_orig:
        b_flat = a_flat
    info = _resolve_cost_and_output_symmetry("i,j->ij", a_flat, b_flat)
    cost = info.accumulation.total
    output_sym = info.output_symmetry
    canonical_subs = info.canonical_subscripts
    if output_sym is not None:
        output_sym = _prepare_symmetric_out(out, output_sym)
    with budget.deduct(
        "outer", flop_cost=cost, subscripts=canonical_subs, shapes=(a.shape, b.shape)
    ):
        result = _call_numpy(
            _np.outer,
            _to_base_ndarray(a),
            _to_base_ndarray(b),
            out=None if isinstance(out, SymmetricTensor) else out,
        )
    if output_sym is None:
        if out is not None:
            return out
        return result  # type: ignore[return-value]
    return _wrap_result(result, out=out, symmetry=output_sym)  # type: ignore[return-value]


attach_docstring(outer, _np.outer, "counted_custom", "n(n+1)/2 FLOPs when v outer v")


def _tensordot_parse_axes(a_ndim, b_ndim, axes):
    """Parse ``np.tensordot``'s ``axes`` argument into ``(a_axes, b_axes)``.

    Accepts the same forms as numpy: ``int N`` (contract last N of ``a``
    with first N of ``b``), ``(int, int)`` (single-axis pair), or
    ``(iterable, iterable)`` (per-axis pairing). Returns a pair of
    tuples of contracted axis indices.
    """
    if isinstance(axes, int):
        return (
            tuple(range(a_ndim - axes, a_ndim)),
            tuple(range(axes)),
        )
    a_spec, b_spec = axes
    a_axes = (a_spec,) if isinstance(a_spec, int) else tuple(a_spec)
    b_axes = (b_spec,) if isinstance(b_spec, int) else tuple(b_spec)
    return a_axes, b_axes


def _surviving_symmetry_after_contraction(group, surviving_axes):
    """Restrict ``group`` to the axes that remain after contraction.

    Returns ``None`` if the surviving axes don't carry any of the
    group's permutations (e.g. the contraction broke a 2-axis S₂).
    The returned group is still indexed in the *original* tensor's
    axis space — call :func:`remap_group_axes` afterwards to relabel.
    """
    if group is None:
        return None
    group_axes = group.axes if group.axes is not None else tuple(range(group.degree))
    wanted = tuple(ax for ax in surviving_axes if ax in group_axes)
    if len(wanted) < 2:
        return None
    return restrict_group_to_axes(group, axes=wanted)


@_counted_wrapper
def tensordot(a: ArrayLike, b: ArrayLike, axes: Any = 2) -> FlopscopeArray:
    """Counted version of ``np.tensordot``.

    The dense FLOP cost is ``a.size * b.size / contracted_size``. When
    either operand carries a :class:`SymmetricTensor` symmetry, flopscope
    composes the surviving (post-contraction) symmetry on the output
    axes via :func:`flopscope._symmetry_utils.direct_product_groups` and
    scales the cost by the unique-element fraction of the output (see
    :func:`_symmetry_adjusted_cost`). Above degree 12 the adjustment is
    skipped and :class:`flopscope.errors.CostFallbackWarning` fires.
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    a_contract_axes, b_contract_axes = _tensordot_parse_axes(a.ndim, b.ndim, axes)
    # Fast path: a full inner contraction over all axes maps cleanly to
    # einsum and benefits from joint-operand savings when a is b.
    is_full_inner = (
        a.ndim == b.ndim
        and a_contract_axes == tuple(range(a.ndim))
        and b_contract_axes == tuple(range(b.ndim))
        and a.ndim >= 1
    )
    if is_full_inner:
        # Build matching einsum subscripts (e.g. ndim=2 -> "ij,ij->").
        letters = "abcdefghijklmnopqrstuvwxyz"[: a.ndim]
        subs = f"{letters},{letters}->"
        from flopscope._einsum import _resolve_cost_and_output_symmetry

        info = _resolve_cost_and_output_symmetry(subs, a, b)
        cost = info.accumulation.total
        canonical_subs = info.canonical_subscripts
        out_sym = info.output_symmetry  # scalar output — always None
        with budget.deduct(
            "tensordot",
            flop_cost=cost,
            subscripts=canonical_subs,
            shapes=(a.shape, b.shape),
        ):
            result = _call_numpy(
                _np.tensordot, _to_base_ndarray(a), _to_base_ndarray(b), axes=axes
            )
        if out_sym is not None:
            return _wrap_result(result, symmetry=out_sym)  # type: ignore[return-value]
        return result  # type: ignore[return-value]
    # Fallback: keep the existing sophisticated direct_product_groups path
    # for partial contractions and unusual axes specs.
    contracted = 1
    for ax in a_contract_axes:
        if 0 <= ax < a.ndim:
            contracted *= a.shape[ax]
    # Surviving (non-contracted) axes for each operand.
    a_surviving = tuple(i for i in range(a.ndim) if i not in a_contract_axes)
    b_surviving = tuple(i for i in range(b.ndim) if i not in b_contract_axes)
    output_shape = tuple(a.shape[i] for i in a_surviving) + tuple(
        b.shape[j] for j in b_surviving
    )
    # output_size * contracted = (a.size / contracted) * (b.size / contracted) * contracted
    # = a.size * b.size / contracted
    dense = _builtins.max(a.size * b.size // contracted, 1) if contracted > 0 else 1
    # Compose output symmetry from each input's surviving symmetry, with
    # b's axes lifted past a's surviving count so they refer to their
    # final slots in the combined output. Bail on the composition when
    # either group's |G| exceeds dimino_budget (see
    # ``_is_oversized_for_cost_model``).
    a_sym = _symmetry_of(a)
    b_sym = _symmetry_of(b)
    if _is_oversized_for_cost_model(a_sym) or _is_oversized_for_cost_model(b_sym):
        try:
            oversized_order = (
                a_sym.order() if _is_oversized_for_cost_model(a_sym) else b_sym.order()  # type: ignore[union-attr]
            )
        except _DiminoBudgetExceeded:
            # Unknown-kind group exceeds budget mid-enumeration; can't
            # compute exact |G|. Use sentinel so all such groups share
            # one dedup slot for the warning.
            oversized_order = -1
        _warn_oversized_once("tensordot", oversized_order)
        out_sym = None
        cost = dense
    else:
        a_sym_kept = _surviving_symmetry_after_contraction(a_sym, a_surviving)
        b_sym_kept = _surviving_symmetry_after_contraction(b_sym, b_surviving)
        a_sym_remapped = (
            remap_group_axes(
                a_sym_kept, {ax: new for new, ax in enumerate(a_surviving)}
            )
            if a_sym_kept is not None
            else None
        )
        b_offset = len(a_surviving)
        b_sym_remapped = (
            remap_group_axes(
                b_sym_kept,
                {ax: new + b_offset for new, ax in enumerate(b_surviving)},
            )
            if b_sym_kept is not None
            else None
        )
        out_sym = direct_product_groups(a_sym_remapped, b_sym_remapped)
        cost = _symmetry_adjusted_cost(dense, output_shape, out_sym)
    with budget.deduct(
        "tensordot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    ):
        result = _call_numpy(
            _np.tensordot, _to_base_ndarray(a), _to_base_ndarray(b), axes=axes
        )
    if out_sym is not None:
        return _wrap_result(result, symmetry=out_sym)  # type: ignore[return-value]
    return result  # type: ignore[return-value]  # wrapped at fnp.tensordot import time


attach_docstring(tensordot, _np.tensordot, "counted_custom", "product of all dims")


@_counted_wrapper
def vdot(a: ArrayLike, b: ArrayLike) -> FlopscopeArray:
    """Counted version of np.vdot."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    from flopscope._einsum import _resolve_cost_and_output_symmetry

    a_flat = a.ravel() if a.ndim != 1 else a
    b_flat = b.ravel() if b.ndim != 1 else b
    info = _resolve_cost_and_output_symmetry("i,i->", a_flat, b_flat)
    cost = info.accumulation.total
    canonical_subs = info.canonical_subscripts
    with budget.deduct(
        "vdot", flop_cost=cost, subscripts=canonical_subs, shapes=(a.shape, b.shape)
    ):
        result = _call_numpy(_np.vdot, _to_base_ndarray(a), _to_base_ndarray(b))
    # vdot returns a scalar, never a SymmetricTensor.
    return result  # type: ignore[return-value]


attach_docstring(vdot, _np.vdot, "counted_custom", "size of input FLOPs")


@_counted_wrapper
def kron(a: ArrayLike, b: ArrayLike) -> FlopscopeArray:
    """Counted version of np.kron."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    # kron output size = a.size * b.size
    cost = _builtins.max(a.size * b.size, 1)
    with budget.deduct(
        "kron", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    ):
        result = _call_numpy(_np.kron, _to_base_ndarray(a), _to_base_ndarray(b))
    return result  # type: ignore[return-value]  # wrapped at fnp.kron import time


attach_docstring(kron, _np.kron, "counted_custom", "output size FLOPs")


@_counted_wrapper
def cross(a: ArrayLike, b: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Counted version of np.cross."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    # np.cross supports axisa/axisb/axisc kwargs that change output shape,
    # so we compute the result first, then deduct based on actual output size.
    result = _np.cross(_to_base_ndarray(a), _to_base_ndarray(b), **kwargs)
    cost = _builtins.max(_np.asarray(result).size * 3, 1)
    with budget.deduct(
        "cross",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, b.shape),
    ):
        pass  # numpy call already done; timer records near-zero duration
    return result  # type: ignore[return-value]  # wrapped at fnp.cross import time


attach_docstring(cross, _np.cross, "counted_custom", "output_size * 3 FLOPs")
cross.__signature__ = _inspect.signature(_np.cross)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def diff(a: ArrayLike, n: int = 1, axis: int = -1, **kwargs: Any) -> FlopscopeArray:
    """Counted version of np.diff."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    # Pre-compute output size: along the diff axis, size decreases by n
    ax = axis if axis >= 0 else axis + a.ndim
    out_axis_len = a.shape[ax] - n
    cost = _builtins.max(
        int(_np.prod(a.shape[:ax])) * out_axis_len * int(_np.prod(a.shape[ax + 1 :])), 1
    )
    with budget.deduct(
        "diff",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
    ):
        result = _call_numpy(_np.diff, _to_base_ndarray(a), n=n, axis=axis, **kwargs)
    return result  # type: ignore[return-value]  # wrapped at fnp.diff import time


attach_docstring(diff, _np.diff, "counted_custom", "numel(output) FLOPs")
diff.__signature__ = _inspect.signature(_np.diff)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def gradient(
    f: ArrayLike, *varargs: ArrayLike, **kwargs: Any
) -> FlopscopeArray | list[FlopscopeArray]:
    """Counted version of np.gradient."""
    budget = require_budget()
    if not isinstance(f, _np.ndarray):
        f = _np.asarray(f)
    with budget.deduct(
        "gradient", flop_cost=f.size, subscripts=None, shapes=(f.shape,)
    ):
        result = _call_numpy(
            _np.gradient,
            _to_base_ndarray(f),
            *[_to_base_ndarray(v) for v in varargs],
            **kwargs,
        )
    return result  # type: ignore[return-value]  # wrapped at fnp.gradient import time


attach_docstring(gradient, _np.gradient, "counted_custom", "numel(input) FLOPs")
gradient.__signature__ = _inspect.signature(_np.gradient)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def ediff1d(ary: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Counted version of np.ediff1d."""
    budget = require_budget()
    if not isinstance(ary, _np.ndarray):
        ary = _np.asarray(ary)
    # Output size = ary.size - 1 (plus any to_begin/to_end extras)
    to_begin = kwargs.get("to_begin", None)
    to_end = kwargs.get("to_end", None)
    extra = 0
    if to_begin is not None:
        extra += _np.asarray(to_begin).size
    if to_end is not None:
        extra += _np.asarray(to_end).size
    cost = _builtins.max(ary.size - 1 + extra, 1)
    with budget.deduct(
        "ediff1d",
        flop_cost=cost,
        subscripts=None,
        shapes=(ary.shape,),
    ):
        # ``to_begin`` / ``to_end`` kwargs may be FlopscopeArrays — strip via tree.
        stripped_kwargs = {
            k: _to_base_ndarray(v) if isinstance(v, _np.ndarray) else v
            for k, v in kwargs.items()
        }
        result = _call_numpy(_np.ediff1d, _to_base_ndarray(ary), **stripped_kwargs)
    return result  # type: ignore[return-value]  # wrapped at fnp.ediff1d import time


attach_docstring(ediff1d, _np.ediff1d, "counted_custom", "numel(output) FLOPs")
ediff1d.__signature__ = _inspect.signature(_np.ediff1d)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def convolve(a: ArrayLike, v: ArrayLike, mode: str = "full") -> FlopscopeArray:
    """Counted version of np.convolve."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(v, _np.ndarray):
        v = _np.asarray(v)
    cost = _builtins.max(a.size * v.size, 1)
    with budget.deduct(
        "convolve",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, v.shape),
    ):
        result = _call_numpy(
            _np.convolve, _to_base_ndarray(a), _to_base_ndarray(v), mode=mode
        )  # type: ignore[arg-type]
    return result  # type: ignore[return-value]  # wrapped at fnp.convolve import time


attach_docstring(convolve, _np.convolve, "counted_custom", "n * m FLOPs")


@_counted_wrapper
def correlate(a: ArrayLike, v: ArrayLike, mode: str = "valid") -> FlopscopeArray:
    """Counted version of np.correlate."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(v, _np.ndarray):
        v = _np.asarray(v)
    cost = _builtins.max(a.size * v.size, 1)
    with budget.deduct(
        "correlate",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, v.shape),
    ):
        result = _call_numpy(
            _np.correlate, _to_base_ndarray(a), _to_base_ndarray(v), mode=mode
        )  # type: ignore[arg-type]
    return result  # type: ignore[return-value]  # wrapped at fnp.correlate import time


attach_docstring(correlate, _np.correlate, "counted_custom", "n * m FLOPs")


def _cov_cost(x, y=None):
    """Cost for corrcoef/cov: 2 * f^2 * s.

    For a (f, s) input: f features, s samples.
    Covariance requires f^2 dot products of length s, plus mean subtraction.
    """
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    if x.ndim == 1:
        f, s = 1, x.shape[0]
    else:
        f, s = x.shape[0], x.shape[1]
    if y is not None:
        y_arr = _np.asarray(y)
        f2 = 1 if y_arr.ndim == 1 else y_arr.shape[0]
        f += f2
    return _builtins.max(2 * f * f * s, 1)


@_counted_wrapper
def corrcoef(x: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> FlopscopeArray:
    """Counted version of np.corrcoef. Cost: 2 * f^2 * s FLOPs."""
    budget = require_budget()
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    cost = _cov_cost(x, y)
    with budget.deduct("corrcoef", flop_cost=cost, subscripts=None, shapes=(x.shape,)):
        result = _call_numpy(
            _np.corrcoef,
            _to_base_ndarray(x),
            y=_to_base_ndarray(y) if y is not None else None,  # type: ignore[arg-type]
            **kwargs,
        )
    return result  # type: ignore[return-value]  # wrapped at fnp.corrcoef import time


attach_docstring(corrcoef, _np.corrcoef, "counted_custom", r"$2 f^2 s$ FLOPs")
corrcoef.__signature__ = _inspect.signature(_np.corrcoef)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def cov(m: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> FlopscopeArray:
    """Counted version of np.cov. Cost: 2 * f^2 * s FLOPs."""
    budget = require_budget()
    if not isinstance(m, _np.ndarray):
        m = _np.asarray(m)
    cost = _cov_cost(m, y)
    with budget.deduct("cov", flop_cost=cost, subscripts=None, shapes=(m.shape,)):
        result = _call_numpy(
            _np.cov,
            _to_base_ndarray(m),
            y=_to_base_ndarray(y) if y is not None else None,  # type: ignore[arg-type]
            **kwargs,
        )
    return result  # type: ignore[return-value]  # wrapped at fnp.cov import time


attach_docstring(cov, _np.cov, "counted_custom", r"$2 f^2 s$ FLOPs")
cov.__signature__ = _inspect.signature(_np.cov)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def trapezoid(
    y: ArrayLike, x: ArrayLike | None = None, dx: float = 1.0, axis: int = -1
) -> FlopscopeArray:
    """Counted version of np.trapezoid."""
    budget = require_budget()
    if not isinstance(y, _np.ndarray):
        y = _np.asarray(y)
    with budget.deduct(
        "trapezoid", flop_cost=y.size, subscripts=None, shapes=(y.shape,)
    ):
        result = _call_numpy(
            _np.trapezoid,
            _to_base_ndarray(y),
            x=_to_base_ndarray(x) if x is not None else None,  # type: ignore[arg-type]
            dx=dx,
            axis=axis,
        )
    return result  # type: ignore[return-value]  # wrapped at fnp.trapezoid import time


attach_docstring(trapezoid, _np.trapezoid, "counted_custom", "numel(input) FLOPs")


if hasattr(_np, "trapz"):

    @_counted_wrapper
    def trapz(  # pyright: ignore[reportRedeclaration]
        y: ArrayLike, x: ArrayLike | None = None, dx: float = 1.0, axis: int = -1
    ) -> FlopscopeArray:
        """Counted version of np.trapz (deprecated alias for trapezoid)."""
        budget = require_budget()
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        with budget.deduct(
            "trapz", flop_cost=y.size, subscripts=None, shapes=(y.shape,)
        ):
            result = _call_numpy(
                _np.trapz,
                _to_base_ndarray(y),
                x=_to_base_ndarray(x) if x is not None else None,
                dx=dx,
                axis=axis,
            )
        return result  # type: ignore[return-value]  # wrapped at fnp.trapz import time

    attach_docstring(trapz, _np.trapz, "counted_custom", "numel(input) FLOPs")

else:

    def trapz(*args, **kwargs):
        raise UnsupportedFunctionError(
            "trapz", max_version="2.4", replacement="trapezoid"
        )


@_counted_wrapper
def interp(x: ArrayLike, xp: ArrayLike, fp: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Counted version of np.interp. Cost: n * ceil(log2(len(xp))) FLOPs."""
    budget = require_budget()
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    xp_arr = _np.asarray(xp)
    n = _builtins.max(x.size, 1)
    xp_len = _builtins.max(xp_arr.size, 1)
    cost = _builtins.max(n * _ceil_log2(xp_len), 1)
    with budget.deduct(
        "interp", flop_cost=cost, subscripts=None, shapes=(x.shape, xp_arr.shape)
    ):
        result = _call_numpy(
            _np.interp,
            _to_base_ndarray(x),
            _to_base_ndarray(xp),  # type: ignore[arg-type]
            _to_base_ndarray(fp),  # type: ignore[arg-type]
            **kwargs,
        )
    return result  # type: ignore[return-value]  # wrapped at fnp.interp import time


attach_docstring(interp, _np.interp, "counted_custom", "n * ceil(log2(xp)) FLOPs")
interp.__signature__ = _inspect.signature(_np.interp)  # pyright: ignore[reportFunctionMemberAccess]
