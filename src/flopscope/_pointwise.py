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

from flopscope._accumulation._cost import contraction_complex_override
from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._config import get_setting as _get_setting
from flopscope._docstrings import attach_docstring
from flopscope._dtype_billing import (
    billing_operand,
    binary_float_loop_dtype,
    heavier_billing_dtype,
    mean_compute_dtype,
    reduction_billing_dtype,
    resolve_billing_dtype,
    store_billing_dtypes,
    sum_accumulator_dtype,
    unary_float_loop_dtype,
)
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
from flopscope._symmetric import SymmetricTensor, _check_generators
from flopscope._symmetry_utils import (
    broadcast_group,
    direct_product_groups,
    intersect_groups,
    reduce_group,
    remap_group_axes,
    restrict_group_to_axes,
    unique_elements_for_shape,
)
from flopscope._validation import _normalize_out, maybe_check_nan_inf, require_budget
from flopscope._write_epoch import note_write
from flopscope.errors import (
    CostFallbackWarning,
    SymmetryError,
    UnsupportedFunctionError,
    _warn_symmetry_loss,
)

# Reductions that accumulate values (add/multiply family) and so inherit
# numpy's integer-widening accumulator dtype when called with dtype=None.
# Extremum (max/min), index (argmax/argmin) and boolean (all/any) reductions
# are excluded: they do not widen, and an index output is int64 regardless of
# the arithmetic precision.
_INTEGER_ACCUMULATING_REDUCTIONS = frozenset(
    {
        "sum",
        "prod",
        "cumsum",
        "cumprod",
        "cumulative_sum",
        "cumulative_prod",
        "nansum",
        "nanprod",
        "nancumsum",
        "nancumprod",
    }
)

# Float-only ufuncs: numpy has no integer loops for these, so integer/bool
# inputs promote to a float compute dtype (same-size float for unary ops,
# float64 for binary ops). Billing the raw input dtype would undercharge the
# actual arithmetic width. Membership is derived from numpy loop resolution
# (an op belongs iff an int32 input yields a float result); the compute-dtype
# conformance sweep enforces it stays complete.
#
# Includes NumPy 2.x array-API spelling aliases (acos/acosh/asin/asinh/
# atan/atanh/atan2 -- literally the same ufunc object as arccos/arccosh/
# arcsin/arcsinh/arctan/arctanh/arctan2, `np.acos is np.arccos` etc.) since
# flopscope wraps each spelling under its own op_name; omitting the alias
# would leave it undercharged while its canonical twin was fixed. Also
# includes ``rint`` (no integer loop at all, same size-mapped promotion as
# the rest of the family) and ``ldexp`` (no all-integer loop; an int32/int32
# call promotes its first operand the same size-mapped way -- a mixed
# float32/int-exponent call is a separate, pre-existing, out-of-scope
# overcount unrelated to this undercount fix; see task-6-report.md).
#
# Also includes ``angle`` (arctan2(0, x) internally -- same size-mapped
# promotion for every integer/unsigned width; a python-int-vs-bool-array NEP
# 50 quirk means angle(bool_) actually computes float64 rather than the
# float16 this mapping would predict, a narrow pre-existing gap this fix does
# not chase) and the ``_counted_unary_multi`` pair ``modf``/``frexp`` (real
# multi-output ufuncs with the identical same-size float loop for their
# primary output; ``frexp``'s exponent output is handled separately, see
# ``_counted_unary_multi``).
_UNARY_FLOAT_LOOP_OPS = frozenset(
    {
        "acos",
        "acosh",
        "angle",
        "arccos",
        "arccosh",
        "arcsin",
        "arcsinh",
        "arctan",
        "arctanh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "cbrt",
        "cos",
        "cosh",
        "deg2rad",
        "degrees",
        "exp",
        "exp2",
        "expm1",
        "fabs",
        "frexp",
        "log",
        "log1p",
        "log2",
        "log10",
        "modf",
        "rad2deg",
        "radians",
        "rint",
        "sin",
        "sinh",
        "spacing",
        "sqrt",
        "tan",
        "tanh",
    }
)
# Unary ops with NO size-mapped integer loop: numpy always computes them in
# (at least) float64 for integer/bool input regardless of the input's own
# width -- unlike the size-mapped _UNARY_FLOAT_LOOP_OPS family (i0(int8) ->
# float64, not float16, matching numpy's Chebyshev-polynomial/sin-ratio
# implementations that don't have narrower loops). Float/complex inputs keep
# their own width. binary_float_loop_dtype's "any int kind -> float64,
# float/complex unchanged" mapping already expresses exactly this rule (it
# doesn't care about arity), so it is reused here instead of a new helper.
_UNARY_FLOAT64_MIN_OPS = frozenset({"i0", "sinc"})
_BINARY_FLOAT_LOOP_OPS = frozenset(
    {
        "arctan2",
        "atan2",
        "copysign",
        "divide",
        "heaviside",
        "hypot",
        "ldexp",
        "logaddexp",
        "logaddexp2",
        "nextafter",
        "true_divide",
    }
)
# float_power computes at DOUBLE PRECISION minimum: it has no
# single-precision loops at all (dd->d / DD->D are its narrowest), so
# float32 inputs compute float64 and complex64 inputs compute complex128.
# It is deliberately NOT a member of _BINARY_FLOAT_LOOP_OPS: that set's
# resolver (binary_float_loop_dtype) leaves float/complex inputs unchanged,
# which would bill float_power(f32, f32) at float32 and
# float_power(c64, c64) at complex64 -- both below its true compute width.
_BINARY_FLOAT64_MIN_OPS = frozenset({"float_power"})

# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------


def _apply_numpy_signature(wrapper, np_func) -> None:
    """Copy np_func's signature onto wrapper, EXCEPT for ufuncs.

    On current numpy, ``inspect.signature(<ufunc>)`` returns the opaque
    ``(*args, **kwargs)``, which would clobber the wrapper's rich typed
    signature that the API-docs generator emits. Keep the wrapper's own
    signature for ufuncs; adopt numpy's for everything else.
    """
    if isinstance(np_func, _np.ufunc):
        return
    try:
        wrapper.__signature__ = _inspect.signature(np_func)  # pyright: ignore[reportFunctionMemberAccess]
    except (ValueError, TypeError):
        pass


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


def _is_out_like(value) -> bool:
    """Is *value*, sitting in a positional ``out`` slot, meant to be ``out``?

    Anything that is not ``None`` is. The slot index comes from numpy's own
    parameter list for the op (``_params.index("out")``), so a value there is
    the caller's ``out`` whatever its type -- and numpy's ``out`` accepts only
    an array or ``None``. Recognising it is what lets :func:`_normalize_out`
    refuse a bad one for free; leaving it unrecognised would pass it through
    to numpy, which raises only after the reduction has been billed.
    """
    return value is not None


def _require_ndarray_out(out, op_name):
    """Internal tripwire: ``out`` must already have been normalized.

    Every public entry point rebinds ``out`` through :func:`_normalize_out` at
    the top of its wrapper, so by the time control reaches here a container is
    unreachable by construction. It is kept because the failure it catches is
    silent otherwise: a wrapper added later without the normalize call would
    not raise, it would quietly bill the contraction at the wrong rate.
    """
    if out is None or isinstance(out, _np.ndarray):
        return
    raise TypeError(
        f"{op_name}(): out= must be an array, got {type(out).__name__}. "
        f"Pass the destination array itself, not a container holding it."
    )


def _prepare_symmetric_out(out, target_symmetry):
    if not isinstance(out, SymmetricTensor):
        return target_symmetry
    carried_symmetry = out.symmetry
    inferred = getattr(out, "_symmetry_inferred", False)
    if target_symmetry is None:
        if inferred:
            return None
        raise ValueError("out symmetry does not match result symmetry")
    if carried_symmetry is not None and carried_symmetry != target_symmetry:
        if inferred:
            return None
        raise ValueError("out symmetry does not match result symmetry")

    if not _check_generators(_np.asarray(out), target_symmetry):
        if inferred:
            return None
        axes = target_symmetry.axes
        if axes is None:
            axes = tuple(range(target_symmetry.degree))
        raise SymmetryError(axes=tuple(axes), max_deviation=float("inf"))
    return target_symmetry


def _validate_result_symmetry(result, symmetry) -> bool:
    """Raise if ``result`` contradicts ``symmetry``; report whether it was checked.

    The return value is the caller's licence to stamp ``symmetry`` onto a
    buffer: it is ``True`` only when the data was actually verified against the
    group. Callers that cannot verify must not re-stamp, otherwise an
    unverifiable result launders a tag onto asymmetric data.
    """
    if symmetry is None:
        return False
    result_arr = _np.asarray(result)
    # Skip numerical validation when the result has non-finite entries:
    # np.allclose treats inf-inf=nan as not-close, which would raise a
    # false SymmetryError. The symmetry was already enforced structurally
    # by the (symmetric) inputs; numerical checks on inf/nan are meaningless.
    if not _np.all(_np.isfinite(result_arr)):
        return False
    if not _check_generators(result_arr, symmetry):
        axes = symmetry.axes
        if axes is None:
            axes = tuple(range(symmetry.degree))
        raise SymmetryError(axes=tuple(axes), max_deviation=float("inf"))
    return True


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
    _require_ndarray_out(out, getattr(np_func, "__name__", "op"))
    out_stripped = _to_base_ndarray(out) if out is not None else None
    if out is None:
        return _call_numpy(np_func, *args, **kwargs)
    if supports_out:
        return _call_numpy(np_func, *args, out=out_stripped, **kwargs)
    result = _call_numpy(np_func, *args, **kwargs)
    # Fallback copy when np_func doesn't natively support out=. This is
    # flopscope's overhead, NOT routed through _call_numpy -- so the write has
    # to be recorded here, or a tag on `out`'s buffer would outlive its data.
    _np.copyto(out_stripped, _np.asarray(result), casting="unsafe")  # type: ignore[arg-type]
    note_write(out)
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
        verified = _validate_result_symmetry(result, effective_symmetry)
        _np.copyto(_np.asarray(out), _np.asarray(result), casting="unsafe")
        # This copy is flopscope-internal and so bypasses _call_numpy's hook.
        # Whatever ``out`` claimed before describes data that is now gone; the
        # claim only comes back if this result was verified against the group.
        note_write(out)
        if verified:
            out._symmetry = effective_symmetry
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
        # Above every later read of ``out`` -- the billing dtype, the
        # symmetry check, and what gets returned -- and above the deduct,
        # so a refused form costs nothing.
        out = _normalize_out(out, op_name)
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        symmetry = _symmetry_of(x)
        symmetry = _prepare_symmetric_out(out, symmetry)
        cost = pointwise_cost(x.shape, symmetry=symmetry)
        # An explicit dtype= forces the ufunc loop: numpy casts operands on
        # read and computes at that width, so it replaces the operand
        # promotion for billing. out= alone does not narrow the loop. A bool
        # dtype= is excluded: it names the output of a value-testing loop,
        # which still reads full-width operands -- bill the operands.
        explicit_dtype = kwargs.get("dtype")
        if explicit_dtype is not None and _np.dtype(explicit_dtype).kind != "b":
            billing_dtypes: tuple = (_np.dtype(explicit_dtype),)
        else:
            billing_dtypes = (x.dtype,)
            if isinstance(out, _np.ndarray):
                billing_dtypes += store_billing_dtypes(out)
        if op_name in _UNARY_FLOAT_LOOP_OPS:
            resolved = resolve_billing_dtype(billing_dtypes)
            if resolved is not None:
                billing_dtypes = (unary_float_loop_dtype(resolved),)
        elif op_name in _UNARY_FLOAT64_MIN_OPS:
            resolved = resolve_billing_dtype(billing_dtypes)
            if resolved is not None:
                billing_dtypes = (binary_float_loop_dtype(resolved),)
        with budget.deduct(
            op_name,
            flop_cost=cost,
            subscripts=None,
            shapes=(x.shape,),
            dtypes=billing_dtypes,
        ):
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
    _apply_numpy_signature(wrapper, np_func)
    return wrapper


@_counted_wrapper
def _free_unary(np_func, op_name: str):
    """Factory for unary ops that are pure component/data extraction.

    Structurally identical to :func:`_counted_unary` (same symmetry / ``out=``
    / NaN-Inf handling) but always bills ``flop_cost=0``: the op returns a
    view or a constant-filled array and performs no floating-point
    arithmetic, so charging ``numel`` would overcount. Still routes through
    ``budget.deduct`` so wall time is accounted.
    """
    supports_out = _supports_out_argument(np_func)

    @_counted_wrapper
    def wrapper(
        x: ArrayLike, out: FlopscopeArray | None = None, **kwargs: Any
    ) -> FlopscopeArray:
        budget = require_budget()
        # Above every later read of ``out`` -- the billing dtype, the
        # symmetry check, and what gets returned -- and above the deduct,
        # so a refused form costs nothing.
        out = _normalize_out(out, op_name)
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        symmetry = _symmetry_of(x)
        symmetry = _prepare_symmetric_out(out, symmetry)
        billing_dtypes: tuple = (x.dtype,)
        if kwargs.get("dtype") is not None:
            billing_dtypes += (_np.dtype(kwargs["dtype"]),)
        if isinstance(out, _np.ndarray):
            billing_dtypes += store_billing_dtypes(out)
        with budget.deduct(
            op_name,
            flop_cost=0,
            subscripts=None,
            shapes=(x.shape,),
            dtypes=billing_dtypes,
        ):
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
    attach_docstring(wrapper, np_func, "free", "0 FLOPs")
    _apply_numpy_signature(wrapper, np_func)
    return wrapper


@_counted_wrapper
def _counted_unary_multi(np_func, op_name: str):
    """Factory for unary functions that return multiple arrays (modf, frexp).

    Supports ``out=(out1, out2)`` (or with ``None`` slots for partial
    allocation) — per-slot stripping and identity preservation are routed
    through :func:`_call_with_optional_multi_out`. Symmetry of the input
    is inherited by every output (elementwise ufuncs).

    Both ``modf`` and ``frexp`` are true ufuncs with the same same-size float
    loop as the ``_UNARY_FLOAT_LOOP_OPS`` family (``modf``/``frexp`` on int8
    input compute float16, same as ``sin``), so they share that membership
    set and resolver. ``frexp``'s second output (the exponent) is always
    int32 regardless of the mantissa's size-mapped precision -- a narrow
    mantissa (float16, from an int8 input) would otherwise undercount the
    fixed-width exponent, so its billed dtype is floored at int32's rate too.
    """
    nout = getattr(np_func, "nout", 2)

    @_counted_wrapper
    def wrapper(
        x: ArrayLike,
        out: tuple[FlopscopeArray, FlopscopeArray] | None = None,
        **kwargs: Any,
    ) -> tuple[FlopscopeArray, FlopscopeArray]:
        budget = require_budget()
        # Above every later read of ``out`` -- the billing dtype, the
        # symmetry check, and what gets returned -- and above the deduct,
        # so a refused form costs nothing.
        out = _normalize_out(out, op_name, nout=nout)
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        symmetry = _symmetry_of(x)
        cost = pointwise_cost(x.shape, symmetry=symmetry)
        # An explicit dtype= forces the ufunc loop: numpy casts operands on
        # read and computes at that width, so it replaces the operand
        # promotion for billing. out= alone does not narrow the loop. A bool
        # dtype= is excluded: it names the output of a value-testing loop,
        # which still reads full-width operands -- bill the operands.
        explicit_dtype = kwargs.get("dtype")
        if explicit_dtype is not None and _np.dtype(explicit_dtype).kind != "b":
            billing_dtypes: tuple = (_np.dtype(explicit_dtype),)
        else:
            billing_dtypes = (x.dtype,)
            if out is not None:
                for o in out:
                    billing_dtypes += store_billing_dtypes(o)
        if op_name in _UNARY_FLOAT_LOOP_OPS:
            resolved = resolve_billing_dtype(billing_dtypes)
            if resolved is not None:
                mapped = unary_float_loop_dtype(resolved)
                if op_name == "frexp":
                    mapped = heavier_billing_dtype(mapped, _np.dtype(_np.int32))
                billing_dtypes = (mapped,)
        with budget.deduct(
            op_name,
            flop_cost=cost,
            subscripts=None,
            shapes=(x.shape,),
            dtypes=billing_dtypes,
        ):
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
    _apply_numpy_signature(wrapper, np_func)
    return wrapper


@_counted_wrapper
def _counted_binary(np_func, op_name: str):
    supports_out = _supports_out_argument(np_func)

    @_counted_wrapper
    def wrapper(
        x: ArrayLike, y: ArrayLike, out: FlopscopeArray | None = None, **kwargs: Any
    ) -> FlopscopeArray:
        budget = require_budget()
        # Above every later read of ``out`` -- the billing dtype, the
        # symmetry check, and what gets returned -- and above the deduct,
        # so a refused form costs nothing.
        out = _normalize_out(out, op_name)
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
        # An explicit dtype= forces the ufunc loop: numpy casts operands on
        # read and computes at that width, so it replaces the operand
        # promotion for billing. out= alone does not narrow the loop. A bool
        # dtype= is excluded: it names the output of a value-testing loop,
        # which still reads full-width operands -- bill the operands.
        explicit_dtype = kwargs.get("dtype")
        if explicit_dtype is not None and _np.dtype(explicit_dtype).kind != "b":
            billing_dtypes = (_np.dtype(explicit_dtype),)
        else:
            billing_dtypes = (billing_operand(x_orig, x), billing_operand(y_orig, y))
            if isinstance(out, _np.ndarray):
                billing_dtypes += store_billing_dtypes(out)
        if op_name in _BINARY_FLOAT_LOOP_OPS:
            resolved = resolve_billing_dtype(billing_dtypes)
            if resolved is not None:
                billing_dtypes = (binary_float_loop_dtype(resolved),)
        elif op_name in _BINARY_FLOAT64_MIN_OPS:
            resolved = resolve_billing_dtype(billing_dtypes)
            if resolved is not None:
                # Double-precision minimum in BOTH kinds: float_power has no
                # single-precision loops at all (dd->d / DD->D are its
                # narrowest), so real inputs floor at float64 and complex
                # inputs floor at complex128. The complex minimum stays
                # complex-KIND so the registry complex_factor still applies
                # -- folding complex into a bare real float64 would silently
                # discard the complex structure premium.
                minimum = (
                    _np.dtype(_np.complex128)
                    if resolved.kind == "c"
                    else _np.dtype(_np.float64)
                )
                billing_dtypes = (heavier_billing_dtype(resolved, minimum),)
        with budget.deduct(
            op_name,
            flop_cost=cost,
            subscripts=None,
            shapes=(x.shape, y.shape),
            dtypes=billing_dtypes,
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
    _apply_numpy_signature(wrapper, np_func)
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
        # Above every later read of ``out`` -- the billing dtype, the
        # symmetry check, and what gets returned -- and above the deduct,
        # so a refused form costs nothing.
        out = _normalize_out(out, op_name, nout=nout)
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
        # An explicit dtype= forces the ufunc loop: numpy casts operands on
        # read and computes at that width, so it replaces the operand
        # promotion for billing. out= alone does not narrow the loop. A bool
        # dtype= is excluded: it names the output of a value-testing loop,
        # which still reads full-width operands -- bill the operands.
        explicit_dtype = kwargs.get("dtype")
        if explicit_dtype is not None and _np.dtype(explicit_dtype).kind != "b":
            billing_dtypes = (_np.dtype(explicit_dtype),)
        else:
            billing_dtypes = (billing_operand(x_orig, x), billing_operand(y_orig, y))
            if out is not None:
                for o in out:
                    billing_dtypes += store_billing_dtypes(o)
        with budget.deduct(
            op_name,
            flop_cost=cost,
            subscripts=None,
            shapes=(x.shape, y.shape),
            dtypes=billing_dtypes,
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
    _apply_numpy_signature(wrapper, np_func)
    return wrapper


# ---------------------------------------------------------------------------
# Generic ufunc-method helpers (outer, reduceat, at, generic reduce/accumulate)
# ---------------------------------------------------------------------------


def _ufunc_loop_dtype(ufunc, *operand_dtypes: _np.dtype | type) -> _np.dtype:
    """The physical loop/output dtype numpy resolves for ``ufunc`` on ``operand_dtypes``.

    Shared by the five generic ufunc-method paths below (``outer`` /
    ``reduce`` / ``accumulate`` / ``reduceat`` / ``at``) to bill a float-only
    ufunc's actual compute dtype instead of its integer input dtype --
    ``true_divide(int32, int32)`` runs entirely in float64; billing the raw
    int32 label undercharges it 2x under production rates.

    ``ufunc.resolve_dtypes`` asks numpy directly which loop it will select.
    Its input tuple is sized from ``ufunc.nin``: a missing second operand of
    a binary ufunc repeats the first (matching how reduce / accumulate /
    reduceat feed a single array through a binary loop), and an unspecified
    (``None``) output slot is appended. An operand may also be a bare
    Python type (``int`` / ``float`` / ``complex``), which
    ``resolve_dtypes`` treats as a NEP 50 weak scalar -- ``ufunc.at`` uses
    this for Python-scalar ``vals``. On failure (``TypeError`` /
    ``ValueError`` -- the operand combination has no valid loop for this
    ufunc, or a multi-output ufunc needs a wider tuple) falls back to
    ``np.result_type(*operand_dtypes)``, which recovers each call site's
    pre-existing behavior exactly: a dtype with itself resolves to itself,
    and two distinct operand dtypes resolve to their common promoted dtype
    (the same resolution ``resolve_billing_dtype`` would have performed on
    the undifferentiated tuple).
    """
    first, *rest = operand_dtypes
    if ufunc.nin == 1:
        inputs: tuple = (first,)
    else:
        inputs = (first, rest[0] if rest else first)
    try:
        return ufunc.resolve_dtypes((*inputs, None))[len(inputs)]
    except (TypeError, ValueError):
        return _np.result_type(*operand_dtypes)


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
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, f"{ufunc.__name__}.outer", nout=ufunc.nout)
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
    # An explicit dtype= forces the loop numpy actually runs (both
    # directions), the same as the plain pointwise factories -- it replaces
    # the operand-promoted default rather than discounting it. A bool
    # dtype= is excluded: it names the output of a value-testing loop
    # (comparison/logical), which still reads full-width operands, so it
    # falls through to the default path below instead of billing the
    # (lighter) bool rate.
    explicit_dtype = kwargs.get("dtype")
    if explicit_dtype is not None and _np.dtype(explicit_dtype).kind != "b":
        billing_dtypes: tuple = (_np.dtype(explicit_dtype),)
    else:
        # This default path shares the operand-width behavior of the
        # reduce/accumulate/reduceat/at siblings: a comparison/logical
        # ufunc's loop OUTPUT is bool, which for wide-int inputs would bill
        # NARROWER than the input -- never charge below it.
        # heavier_billing_dtype keeps the loop dtype on a rate tie, so float
        # widening (float64 >= int rate) is unaffected.
        billing_dtypes = (
            heavier_billing_dtype(
                _ufunc_loop_dtype(ufunc, a.dtype, b.dtype), a.dtype, b.dtype
            ),
        )
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    with budget.deduct(
        f"{ufunc.__name__}.outer",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, b.shape),
        dtypes=billing_dtypes,
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
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, f"{ufunc.__name__}.reduce", nout=ufunc.nout)
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
    # The reduce/accumulate loop runs at the ufunc's own resolved loop dtype
    # (true_divide(int32) -> float64, subtract(int32) -> int32, logical_* ->
    # bool). add/multiply's extra integer widening never matters here: they
    # are routed to sum/prod, not this generic path.
    default_dtype = _ufunc_loop_dtype(ufunc, a.dtype, a.dtype)
    billing_dtypes: tuple = (
        reduction_billing_dtype(
            a.dtype,
            explicit_dtype=kwargs.get("dtype"),
            out_dtype=out.dtype if isinstance(out, _np.ndarray) else None,
            default_dtype=default_dtype,
        ),
    )
    with budget.deduct(
        f"{ufunc.__name__}.reduce",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
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
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, f"{ufunc.__name__}.accumulate", nout=ufunc.nout)
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
    # Same loop resolution as the generic reduce path above: the accumulate
    # loop runs at the ufunc's own resolved loop dtype (true_divide(int32)
    # -> float64, subtract(int32) -> int32).
    default_dtype = _ufunc_loop_dtype(ufunc, a.dtype, a.dtype)
    billing_dtypes: tuple = (
        reduction_billing_dtype(
            a.dtype,
            explicit_dtype=kwargs.get("dtype"),
            out_dtype=out.dtype if isinstance(out, _np.ndarray) else None,
            default_dtype=default_dtype,
        ),
    )
    with budget.deduct(
        f"{ufunc.__name__}.accumulate",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
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
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, f"{ufunc.__name__}.reduceat", nout=ufunc.nout)
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
    # This default path (no explicit dtype=) shares the operand-width
    # behavior of the generic reduce/accumulate paths: reduceat runs the
    # ufunc's own resolved loop dtype by default (true_divide(int32) ->
    # float64, subtract(int32) -> int32) -- except add/multiply, which numpy
    # runs through the same integer-widening sum/prod accumulator, regardless
    # of the segment indices. reduction_billing_dtype supplies the
    # input-rate floor -- a comparison/logical ufunc's bool loop OUTPUT must
    # never bill narrower than a wide-int input. An explicit dtype= now
    # resolves per numpy's accumulator semantics instead (billed exactly as
    # requested, wider or narrower).
    default_dtype = (
        sum_accumulator_dtype(a.dtype)
        if ufunc.__name__ in ("add", "multiply")
        else _ufunc_loop_dtype(ufunc, a.dtype, a.dtype)
    )
    billing_dtypes: tuple = (
        reduction_billing_dtype(
            a.dtype,
            explicit_dtype=kwargs.get("dtype"),
            default_dtype=default_dtype,
        ),
    )
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    with budget.deduct(
        f"{ufunc.__name__}.reduceat",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
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
    # Same loop resolution as the other generic ufunc-method paths:
    # ``ufunc.at`` applies the ufunc's own resolved loop and casts the
    # result back in place with unsafe casting, so a float-only loop runs
    # on integer arrays WITHOUT raising (exp.at(int32) computes float64).
    # Bill that loop, floored at the array's own rate (the established
    # reduction_billing_dtype semantics -- logical_and.at(f64) keeps the
    # f64 rate). Binary ufuncs contribute their ``vals`` operand: NEP 50
    # weak Python scalars pass as their bare type (bool never widens a
    # loop, so it just repeats the array dtype via the nin-padding);
    # everything else contributes its (coerced) array dtype.
    if hasattr(a, "dtype"):
        operands: list = [a.dtype]
        if args:
            vals = args[0]
            if isinstance(vals, (bool, int, float, complex)) and not isinstance(
                vals, _np.generic
            ):
                if not isinstance(vals, bool):
                    operands.append(type(vals))
            elif isinstance(vals, _np.ndarray):
                operands.append(vals.dtype)
            else:
                operands.append(_np.asarray(vals).dtype)
        billing_dtypes: tuple = (
            reduction_billing_dtype(
                a.dtype,
                default_dtype=_ufunc_loop_dtype(ufunc, *operands),
            ),
        )
    else:
        billing_dtypes = ()
    with budget.deduct(
        f"{ufunc.__name__}.at",
        flop_cost=n_ops,
        subscripts=None,
        shapes=(a.shape,) if hasattr(a, "shape") else (),
        dtypes=billing_dtypes,
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
    _dtype_args_idx = (
        _params.index("dtype") - _args_offset
        if "dtype" in _params and _params.index("dtype") >= _args_offset
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
        # Normalized before anything reads it: unlike the other wrappers this
        # one cannot normalize at the top, because ``out`` is not bound yet --
        # it arrives either as a keyword or in a positional slot. Both routes
        # go through the same helper here, ahead of _prepare_symmetric_out,
        # the out_dtype billing fold below, and the deduct.
        out = _normalize_out(kwargs.pop("out", None), op_name)
        out_came_from_args = False
        if (
            out is None
            and _out_args_idx is not None
            and 0 <= _out_args_idx < len(args_list)
            and _is_out_like(args_list[_out_args_idx])
        ):
            out = _normalize_out(args_list[_out_args_idx], op_name)
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

        # numpy computes the reduction in the explicit dtype= (positional or
        # keyword) if given, else in out's dtype, else in the family default
        # (integer-widening for the accumulating reductions). An explicit
        # dtype= bills as the accumulator numpy runs, in both directions;
        # out= widens the accumulator, and a narrower out only casts the
        # final store; the default never bills below the operand width.
        explicit_dtype = kwargs.get("dtype")
        if (
            explicit_dtype is None
            and _dtype_args_idx is not None
            and 0 <= _dtype_args_idx < len(args_list)
        ):
            explicit_dtype = args_list[_dtype_args_idx]
        default_dtype = (
            sum_accumulator_dtype(a.dtype)
            if op_name in _INTEGER_ACCUMULATING_REDUCTIONS
            else a.dtype
        )
        billing_dtypes: tuple = (
            reduction_billing_dtype(
                a.dtype,
                explicit_dtype=explicit_dtype,
                out_dtype=out.dtype if isinstance(out, _np.ndarray) else None,
                default_dtype=default_dtype,
            ),
        )
        with budget.deduct(
            op_name,
            flop_cost=cost,
            subscripts=None,
            shapes=(a.shape,),
            dtypes=billing_dtypes,
        ):
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
    _apply_numpy_signature(wrapper, np_func)
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
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "around")
    a_is_scalar = not isinstance(a, _np.ndarray) and _np.ndim(a) == 0
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    symmetry = _symmetry_of(a)
    _prepare_symmetric_out(out, symmetry)
    cost = pointwise_cost(a.shape, symmetry=symmetry)
    billing_dtypes: tuple = (a.dtype,)
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    with budget.deduct(
        "around",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
    ):
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
imag = _free_unary(_np.imag, "imag")
imag.__signature__ = _inspect.signature(_np.imag)  # pyright: ignore[reportFunctionMemberAccess]
invert = _counted_unary(_np.invert, "invert")
iscomplex = _counted_unary(_np.iscomplex, "iscomplex")


def iscomplexobj(x: Any) -> bool:
    """Returns True if x is a complex type or an array of complex type.
    This is a dtype/metadata predicate (O(1)) — Cost: 0 FLOPs."""
    return _np.iscomplexobj(_to_base_ndarray(x) if isinstance(x, _np.ndarray) else x)


attach_docstring(iscomplexobj, _np.iscomplexobj, "free", "0 FLOPs")
isnat = _counted_unary(_np.isnat, "isnat")
isneginf = _counted_unary(_np.isneginf, "isneginf")
isneginf.__signature__ = _inspect.signature(_np.isneginf)  # pyright: ignore[reportFunctionMemberAccess]
isposinf = _counted_unary(_np.isposinf, "isposinf")
isposinf.__signature__ = _inspect.signature(_np.isposinf)  # pyright: ignore[reportFunctionMemberAccess]
isreal = _counted_unary(_np.isreal, "isreal")


def isrealobj(x: Any) -> bool:
    """Returns True if x is not a complex type or an array of complex type.
    This is a dtype/metadata predicate (O(1)) — Cost: 0 FLOPs."""
    return _np.isrealobj(_to_base_ndarray(x) if isinstance(x, _np.ndarray) else x)


attach_docstring(isrealobj, _np.isrealobj, "free", "0 FLOPs")
log1p = _counted_unary(_np.log1p, "log1p")
logical_not = _counted_unary(_np.logical_not, "logical_not")
nan_to_num = _counted_unary(_np.nan_to_num, "nan_to_num")
nan_to_num.__signature__ = _inspect.signature(_np.nan_to_num)  # pyright: ignore[reportFunctionMemberAccess]
positive = _counted_unary(_np.positive, "positive")
rad2deg = _counted_unary(_np.rad2deg, "rad2deg")
radians = _counted_unary(_np.radians, "radians")
real = _free_unary(_np.real, "real")
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
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "round")
    a_is_scalar = not isinstance(a, _np.ndarray) and _np.ndim(a) == 0
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    symmetry = _symmetry_of(a)
    _prepare_symmetric_out(out, symmetry)
    cost = pointwise_cost(a.shape, symmetry=symmetry)
    billing_dtypes: tuple = (a.dtype,)
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    with budget.deduct(
        "round",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
    ):
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


def _sort_complex_billing_dtype(dtype: _np.dtype) -> _np.dtype:
    """np.sort_complex's exact output dtype -- a hardcoded 3-way table in
    numpy's own source, not a `result_type`-style promotion:

        if b.dtype.char in 'bhBH': return b.astype('F')   # int8/16, uint8/16 -> complex64
        elif b.dtype.char == 'g':  return b.astype('G')   # longdouble -> clongdouble
        else:                      return b.astype('D')   # everything else -> complex128

    So bool/int32/int64/uint32/uint64/float16/float32/float64 all compute in
    complex128 (verified: sort_complex(float32) -> complex128, NOT
    complex64), only the narrow bhBH integer kinds get the complex64 loop.
    Already-complex input is returned unchanged.
    """
    if dtype.kind == "c":
        return dtype
    if dtype.char in "bhBH":
        return _np.dtype(_np.complex64)
    if dtype.char == "g":
        if hasattr(_np, "complex256"):
            return _np.dtype(_np.complex256)
        return _np.dtype(_np.complex128)
    return _np.dtype(_np.complex128)


@_counted_wrapper
def sort_complex(a: ArrayLike) -> FlopscopeArray:
    """Counted version of np.sort_complex.

    Cost: n*ceil(log2(n)) per last-axis slice (n = a.shape[-1]).
    numpy.sort_complex sorts each 1-D slice along the last axis, so the
    total cost is num_slices * sort_cost(n).  For 1-D input this equals
    the previous n*ceil(log2(n)) formula.
    """
    from flopscope._sorting_ops import _sort_cost_nd

    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    cost = 1 if a.ndim == 0 else _sort_cost_nd(a, a.ndim - 1)
    with budget.deduct(
        "sort_complex",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(_sort_complex_billing_dtype(a.dtype),),
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
    # 6 FLOPs/elem: sub + 2*abs + mul + add + cmp (tolerance core; floor per documented policy)
    cost = 6 * pointwise_cost(output_shape, symmetry=out_symmetry)
    billing_dtypes = (billing_operand(a, a_arr), billing_operand(b, b_arr))
    with budget.deduct(
        "isclose",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_arr.shape, b_arr.shape),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(
            _np.isclose, _to_base_ndarray(a), _to_base_ndarray(b), **kwargs
        )
    if a_is_scalar and b_is_scalar and _np.ndim(result) == 0:
        return bool(result)
    return _wrap_result(result, symmetry=out_symmetry)  # type: ignore[return-value]


attach_docstring(
    isclose,
    _np.isclose,
    "counted_unary",
    "6*numel(output) FLOPs (tolerance core: sub+2*abs+mul+add+cmp)",
)
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
        """Counted version of np.vecdot (vector dot product along last axis)."""
        return _einsum_routed_binary(
            "vecdot", _np.vecdot, "...n,...n->...", a, b, **kwargs
        )

else:

    def vecdot(*args: Any, **kwargs: Any) -> FlopscopeArray:  # pyright: ignore[reportRedeclaration]
        raise UnsupportedFunctionError("vecdot", min_version="2.1")


if hasattr(_np, "matvec"):

    @_counted_wrapper
    def matvec(a: ArrayLike, b: ArrayLike, **kwargs: Any) -> FlopscopeArray:  # pyright: ignore[reportRedeclaration]
        """Counted version of np.matvec (matrix-vector product).

        A is (..., m, n), v is (..., n), result is (..., m). Cost is the exact
        einsum accumulation cost, counting batch/broadcast on either operand.
        """
        return _einsum_routed_binary(
            "matvec", _np.matvec, "...mn,...n->...m", a, b, **kwargs
        )

else:

    def matvec(*args: Any, **kwargs: Any) -> FlopscopeArray:  # pyright: ignore[reportRedeclaration]
        raise UnsupportedFunctionError("matvec", min_version="2.2")


if hasattr(_np, "vecmat"):

    @_counted_wrapper
    def vecmat(a: ArrayLike, b: ArrayLike, **kwargs: Any) -> FlopscopeArray:  # pyright: ignore[reportRedeclaration]
        """Counted version of np.vecmat (vector-matrix product).

        v is (..., n), A is (..., n, m), result is (..., m). Cost is the exact
        einsum accumulation cost, counting batch/broadcast on either operand.
        """
        return _einsum_routed_binary(
            "vecmat", _np.vecmat, "...n,...nm->...m", a, b, **kwargs
        )

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
    """Counted version of np.clip.

    Cost = n_bounds * numel(output) FLOPs (1 compare-select per bound, per output elem).
    n_bounds = number of non-None bounds (0, 1, or 2); max(n_bounds, 1) ensures a
    materialising-copy floor when no bounds are given (numpy>=2.1 no-bound clip).
    Output shape is the broadcast of a with all bound arrays.
    """
    budget = require_budget()
    # ``out`` is keyword-only in this signature but numpy's is
    # ``clip(a, a_min, a_max, out=None, ...)``, and clip.__signature__ is
    # overwritten with numpy's below -- so a caller following the advertised
    # signature passes the destination as the third positional. It would
    # otherwise land in *args and be counted as a third BOUND: measured 12,000
    # FLOPs against 8,000 for the identical keyword call, because every bound
    # costs numel, and the destination's dtype never reached the rate at all.
    if len(args) >= 3:
        args, out_positional = args[:2], args[2]
        if out is not None and out_positional is not None:
            raise TypeError("clip(): out= given both positionally and by keyword")
        if out is None:
            out = out_positional
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "clip")
    a_orig = a
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    operand_arrays = [(a, _symmetry_of(a))]
    billing_dtypes: tuple = (billing_operand(a_orig, a),)
    for value in args:
        if value is None:
            continue
        arr = value if isinstance(value, _np.ndarray) else _np.asarray(value)
        operand_arrays.append((arr, _symmetry_of(arr)))
        billing_dtypes += (billing_operand(value, arr),)
    for key in ("a_min", "a_max", "min", "max"):
        value = kwargs.get(key)
        if value is None:
            continue
        arr = value if isinstance(value, _np.ndarray) else _np.asarray(value)
        operand_arrays.append((arr, _symmetry_of(arr)))
        billing_dtypes += (billing_operand(value, arr),)
    n_bounds = len(operand_arrays) - 1
    output_shape = _np.broadcast_shapes(*(arr.shape for arr, _ in operand_arrays))
    symmetry, _ = _pointwise_symmetry(operand_arrays, output_shape)
    _prepare_symmetric_out(out, symmetry)
    cost = _builtins.max(n_bounds, 1) * pointwise_cost(output_shape, symmetry=symmetry)
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    with budget.deduct(
        "clip",
        flop_cost=cost,
        subscripts=None,
        shapes=(output_shape,),
        dtypes=billing_dtypes,
    ):
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


attach_docstring(
    clip,
    _np.clip,
    "counted_custom",
    "n_bounds x numel(output) FLOPs (1 compare-select per bound; numel copy floor with no bounds)",
)
clip.__signature__ = _inspect.signature(_np.clip)  # pyright: ignore[reportFunctionMemberAccess]


# ---------------------------------------------------------------------------
# Reductions (original)
# ---------------------------------------------------------------------------

sum = _counted_reduction(_np.sum, "sum")
max = _counted_reduction(_np.max, "max")
min = _counted_reduction(_np.min, "min")
prod = _counted_reduction(_np.prod, "prod")


def _counted_mean(np_func, op_name: str):
    """Factory for mean-family wrappers (mean, nanmean).

    Cost = sum_cost (orbit-mapping FLOPs via Tier-1 model)
           + num_output_orbits (one divide per output orbit).

    np.nanmean shares the (a, axis, dtype, out, keepdims, where) kwargs and
    supports out — body is identical to mean modulo op_name and np_func.
    """

    @_counted_wrapper
    def wrapper(
        a: ArrayLike,
        axis: int | None = None,
        dtype=None,
        out: FlopscopeArray | None = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> FlopscopeArray:
        from flopscope._accumulation._reduction import (
            _normalize_axis,
            _num_output_orbits,
            compute_reduction_accumulation_cost,
        )

        budget = require_budget()
        # Above every later read of ``out`` -- the billing dtype, the
        # symmetry check, and what gets returned -- and above the deduct,
        # so a refused form costs nothing.
        out = _normalize_out(out, op_name)
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

        # numpy computes an integer mean in float64 unless an explicit dtype
        # or out= overrides the accumulator. An explicit dtype= bills as the
        # accumulator numpy runs, in both directions; out= widens it, and a
        # narrower out only casts the final store; the family default never
        # bills below the operand width.
        billing_dtypes: tuple = (
            reduction_billing_dtype(
                a.dtype,
                explicit_dtype=dtype,
                out_dtype=out.dtype if isinstance(out, _np.ndarray) else None,
                default_dtype=mean_compute_dtype(a.dtype),
            ),
        )
        with budget.deduct(
            op_name,
            flop_cost=cost,
            subscripts=None,
            shapes=(a.shape,),
            dtypes=billing_dtypes,
        ):
            result = _call_with_optional_out(
                np_func,
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

    _apply_numpy_signature(wrapper, np_func)
    return wrapper


mean = _counted_mean(_np.mean, "mean")


def _variance_family_cost(a, axis, symmetry, *, with_sqrt: bool) -> int:
    """Honest FMA=2 cost: 2 pointwise passes (center, square) + 2 reductions
    (mean-sum, var-sum) + per-output divides (+ sqrt for std) = 4*numel (+M)."""
    from flopscope._accumulation._reduction import (
        _normalize_axis,
        _num_output_orbits,
        compute_reduction_accumulation_cost,
    )

    axes_summed = _normalize_axis(axis, a.ndim)
    m = _num_output_orbits(tuple(a.shape), axes_summed, symmetry)
    reduce_cost = compute_reduction_accumulation_cost(
        input_shape=tuple(a.shape),
        axes_summed=axes_summed,
        symmetry=symmetry,
        op_factor=2,
        extra_ops=2 * m,
    ).total
    cost = 2 * pointwise_cost(tuple(a.shape), symmetry) + reduce_cost
    return cost + m if with_sqrt else cost


def _counted_variance(np_func, op_name: str, *, with_sqrt: bool):
    @_counted_wrapper
    def wrapper(
        a: ArrayLike,
        axis: int | None = None,
        dtype=None,
        out: FlopscopeArray | None = None,
        ddof: int = 0,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> FlopscopeArray:
        budget = require_budget()
        # Above every later read of ``out`` -- the billing dtype, the
        # symmetry check, and what gets returned -- and above the deduct,
        # so a refused form costs nothing.
        out = _normalize_out(out, op_name)
        if not isinstance(a, _np.ndarray):
            a = _np.asarray(a)
        symmetry = _symmetry_of(a)
        keepdims = bool(keepdims)
        cost = _variance_family_cost(a, axis, symmetry, with_sqrt=with_sqrt)
        new_symmetry = (
            reduce_group(symmetry, ndim=a.ndim, axis=axis, keepdims=keepdims)
            if symmetry is not None
            else None
        )
        _prepare_symmetric_out(out, new_symmetry)
        out_for_np = None if isinstance(out, SymmetricTensor) else out
        # numpy computes an integer var/std in float64 unless an explicit
        # dtype or out= overrides the accumulator. An explicit dtype= bills
        # as the accumulator numpy runs, in both directions; out= widens it,
        # and a narrower out only casts the final store; the family default
        # never bills below the operand width.
        billing_dtypes: tuple = (
            reduction_billing_dtype(
                a.dtype,
                explicit_dtype=dtype,
                out_dtype=out.dtype if isinstance(out, _np.ndarray) else None,
                default_dtype=mean_compute_dtype(a.dtype),
            ),
        )
        with budget.deduct(
            op_name,
            flop_cost=cost,
            subscripts=None,
            shapes=(a.shape,),
            dtypes=billing_dtypes,
        ):
            result = _call_with_optional_out(
                np_func,
                a,
                axis=axis,
                out=out_for_np,
                ddof=ddof,
                keepdims=keepdims,
                dtype=dtype,
                supports_out=True,
                **kwargs,
            )
        if out is not None:
            return _wrap_result(result, out=out, symmetry=new_symmetry)  # type: ignore[return-value]
        return _wrap_result(result, symmetry=new_symmetry)  # type: ignore[return-value]

    _apply_numpy_signature(wrapper, np_func)
    return wrapper


std = _counted_variance(_np.std, "std", with_sqrt=True)
var = _counted_variance(_np.var, "var", with_sqrt=False)
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


@_counted_wrapper
def average(
    a: ArrayLike,
    axis: int | None = None,
    weights=None,
    returned: bool = False,
    *,
    keepdims: bool = False,
    **kwargs: Any,
):
    """Counted np.average.

    Cost (no weights) = reduction_cost(input) + num_output_orbits
                      = same as np.mean (sum pass + one divide per output).
    Cost (weights)    = reduction_cost(input)        # a*w sum pass
                      + pointwise_cost(input)        # a*w multiply pass
                      + reduction_cost(input, axis)  # w.sum() pass (full-shape conservative)
                      + num_output_orbits            # per-output divides
    """
    from flopscope._accumulation._reduction import (
        _normalize_axis,
        _num_output_orbits,
    )

    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    symmetry = _symmetry_of(a)
    axes_summed = _normalize_axis(axis, a.ndim)
    m = _num_output_orbits(tuple(a.shape), axes_summed, symmetry)
    cost = reduction_cost(a.shape, axis, symmetry=symmetry) + m
    if weights is not None:
        cost += pointwise_cost(tuple(a.shape), symmetry)  # the a*w multiply pass
        cost += reduction_cost(a.shape, axis, symmetry=symmetry)  # w.sum() conservative
    new_symmetry = (
        reduce_group(symmetry, ndim=a.ndim, axis=axis, keepdims=keepdims)
        if symmetry is not None
        else None
    )
    a_raw = _to_base_ndarray(a)
    weights_raw = (
        _to_base_ndarray(weights) if isinstance(weights, _np.ndarray) else weights
    )
    # np.average is mean-shaped: integer/bool `a` always computes in float64
    # (average(int32) -> float64), float `a` keeps its own precision unless
    # explicit weights widen it (average(f32, weights=f64) -> float64) --
    # the same rule as mean/median/percentile's compute dtype.
    billing_dtypes: tuple = (mean_compute_dtype(a.dtype),)
    if weights is not None:
        weights_arr = (
            weights if isinstance(weights, _np.ndarray) else _np.asarray(weights)
        )
        billing_dtypes += (billing_operand(weights, weights_arr),)
    with budget.deduct(
        "average",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(
            _np.average,
            a_raw,
            axis=axis,
            weights=weights_raw,
            returned=returned,
            keepdims=keepdims,
            **kwargs,
        )
    return _wrap_result(result, symmetry=new_symmetry)


_apply_numpy_signature(average, _np.average)


@_counted_wrapper
def count_nonzero(
    a: ArrayLike, axis: int | tuple[int, ...] | None = None, *, keepdims: bool = False
) -> FlopscopeArray | int:
    """Counted version of ``numpy.count_nonzero``.

    Cost: numel(input) FLOPs (one comparison per element, axis-independent).

    The boolean-sum accumulation (integer adds over the non-zero mask) is
    intentionally not charged — numel(input) is the documented conservative
    floor per policy. This is axis-independent: every element is tested
    regardless of which axis is reduced.

    When ``axis is None`` (and not ``keepdims``) the result is always
    coerced to a Python ``int``. This is unconditional because flopscope's
    wrapping machinery wraps scalar results via ``_asflopscope`` on every
    numpy version, so without this coercion users would see a
    ``FlopscopeArray`` rather than the plain ``int`` that ``numpy.count_nonzero``
    documents. The coercion also normalizes the numpy 2.3+ change where
    the raw numpy return type became a numpy scalar.
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    symmetry = _symmetry_of(a)
    # numel(input) comparisons, orbit-aware; axis-independent cost
    cost = pointwise_cost(tuple(a.shape), symmetry)
    new_symmetry = (
        reduce_group(symmetry, ndim=a.ndim, axis=axis, keepdims=keepdims)
        if symmetry is not None
        else None
    )
    with budget.deduct(
        "count_nonzero",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(a.dtype,),
    ):
        result = _call_numpy(
            _np.count_nonzero, _to_base_ndarray(a), axis=axis, keepdims=keepdims
        )
    if axis is None and not keepdims:
        return int(result)
    return _wrap_result(result, symmetry=new_symmetry)  # type: ignore[return-value]


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


def _quantile_dense_cost(axis_dim: int, q_count: int, *, weighted: bool) -> int:
    """Per-output cost for one quantile/percentile reduction along an axis.

    ``n = axis_dim``, ``k = q_count``, ``L = ceil(log2(n))`` (0 for ``n<=1``).

    Weighted (``weights=`` given): the interpolated-quantile method sorts (or
    argsorts) the ``n`` values, builds a cumulative-weight array, then does a
    per-``q`` cumulative-weight lookup and interpolation: ``4*n*L`` (argsort
    at sort's own op-weight) + ``3*n`` (gather values/weights and cumulative
    sum) + ``k*(L + 4)`` (a lookup plus a fixed interpolation cost per
    quantile).

    Unweighted: the cheaper of ``k`` independent partition passes
    (``n`` each) versus one shared sort-parity pass amortized across all
    ``k`` outputs, plus a fixed per-quantile gather-and-interpolate term:
    ``n * min(k, 1 + 4*L') + 4*k``, where ``L' = ceil(log2(min(k, n)))``
    (0 for ``min(k, n)<=1``).
    """
    import math as _math

    n = _builtins.max(int(axis_dim), 0)
    k = _builtins.max(int(q_count), 1)
    log_n = _math.ceil(_math.log2(n)) if n > 1 else 0
    if weighted:
        return 4 * n * log_n + 3 * n + k * (log_n + 4)
    log_min = (
        _math.ceil(_math.log2(_builtins.min(k, n))) if _builtins.min(k, n) > 1 else 0
    )
    return n * _builtins.min(k, 1 + 4 * log_min) + 4 * k


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
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "median")
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
    # numpy's median always computes in float64 for integer/bool input (the
    # partition-based selection still runs the interpolating mean-of-two for
    # even-length axes) and preserves the input's own float precision
    # otherwise -- the same rule as mean/var/std's compute dtype.
    billing_dtypes: tuple = (mean_compute_dtype(a.dtype),)
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    with budget.deduct(
        "median",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
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
# nanmean: billed identically to mean (reduction + per-output divide; nan-masking not
# charged, consistent with nansum/nanstd convention).
nanmean = _counted_mean(_np.nanmean, "nanmean")


@_counted_wrapper
def nanmedian(
    a: ArrayLike,
    axis: int | None = None,
    out: FlopscopeArray | None = None,
    keepdims: bool = False,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of np.nanmedian.

    Cost = num_output_orbits × axis_dim (Tier-2 partition-based model).
    Billed identically to median; nan-masking overhead is not charged
    (consistent with nanpercentile/nanquantile convention).
    """
    import math as _math

    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "nanmedian")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym = _symmetry_of(a)

    # Dense per-output cost for partition-based nanmedian: axis_dim (one pass).
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
    # Same compute-dtype rule as median: float64 for integer/bool input,
    # the input's own float precision otherwise.
    billing_dtypes: tuple = (mean_compute_dtype(a.dtype),)
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    with budget.deduct(
        "nanmedian",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(
            _np.nanmedian,
            _to_base_ndarray(a),
            axis=axis,
            out=out_stripped,
            keepdims=keepdims,
            **kwargs,
        )
    return _wrap_result(result, out=out, symmetry=out_sym)  # type: ignore[return-value]


nanmedian.__signature__ = _inspect.signature(_np.nanmedian)  # pyright: ignore[reportFunctionMemberAccess]
nanmin = _counted_reduction(_np.nanmin, "nanmin")
nanprod = _counted_reduction(_np.nanprod, "nanprod")


@_counted_wrapper
def nanpercentile(
    a: ArrayLike,
    q: float | ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    out: FlopscopeArray | None = None,
    keepdims: bool = False,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of np.nanpercentile.

    Cost = num_output_orbits × per-output cost (Tier-2 partition-based model).
    Per-output cost is piecewise in axis_dim (n) and q.size (k), and also
    depends on whether weights= is given -- see _quantile_dense_cost.
    """
    import math as _math

    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "nanpercentile")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym = _symmetry_of(a)

    # Reduced axis length (n), fed into _quantile_dense_cost below.
    if axis is None:
        axis_dim = _math.prod(a.shape) if a.shape else 1
    elif isinstance(axis, int):
        axis_dim = a.shape[axis]
    else:
        axis_dim = _math.prod(a.shape[ax] for ax in axis)

    q_arr = q if isinstance(q, _np.ndarray) else _np.asarray(q)
    # numpy prepends q.shape to the output and computes each requested
    # quantile independently, so the per-output cost scales with the number
    # of quantiles requested, not just the reduced axis length; a weights=
    # array additionally routes through the sort-parity weighted formula
    # instead of the cheaper partition-based one.
    q_count = _builtins.max(int(q_arr.size), 1)
    weighted = kwargs.get("weights") is not None
    cost = _tier2_reduction_cost(
        a,
        axis,
        dense_per_output_cost=_quantile_dense_cost(
            axis_dim, q_count, weighted=weighted
        ),
    )

    out_sym = (
        reduce_group(sym, ndim=a.ndim, axis=axis, keepdims=keepdims)
        if sym is not None
        else None
    )
    out_stripped = _to_base_ndarray(out) if out is not None else None
    # numpy's percentile family always divides q by 100 first (an implicit
    # true_divide, "q = np.true_divide(q, a.dtype.type(100) if a.dtype.kind
    # == 'f' else 100)" in numpy's own source), so an integer/bool `a` always
    # computes in float64 -- even an exact-integer q like 50 -- unlike
    # quantile, whose q is used directly with no such division and so can
    # stay at `a`'s own dtype for an exact q of 0 or 1. mean_compute_dtype
    # captures that "int/bool -> float64, float preserved" half; q still
    # participates so an explicitly wider q array (e.g. float64 q against a
    # float32 `a`) still raises the bill to match numpy's actual widening.
    billing_dtypes: tuple = (mean_compute_dtype(a.dtype), billing_operand(q, q_arr))
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    with budget.deduct(
        "nanpercentile",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(
            _np.nanpercentile,
            _to_base_ndarray(a),
            q,
            axis=axis,
            out=out_stripped,
            keepdims=keepdims,
            **kwargs,
        )
    return _wrap_result(result, out=out, symmetry=out_sym)  # type: ignore[return-value]


nanpercentile.__signature__ = _inspect.signature(_np.nanpercentile)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def nanquantile(
    a: ArrayLike,
    q: float | ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    out: FlopscopeArray | None = None,
    keepdims: bool = False,
    **kwargs: Any,
) -> FlopscopeArray:
    """Counted version of np.nanquantile.

    Cost = num_output_orbits × per-output cost (Tier-2 partition-based model).
    Per-output cost is piecewise in axis_dim (n) and q.size (k), and also
    depends on whether weights= is given -- see _quantile_dense_cost.
    """
    import math as _math

    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "nanquantile")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym = _symmetry_of(a)

    # Reduced axis length (n), fed into _quantile_dense_cost below.
    if axis is None:
        axis_dim = _math.prod(a.shape) if a.shape else 1
    elif isinstance(axis, int):
        axis_dim = a.shape[axis]
    else:
        axis_dim = _math.prod(a.shape[ax] for ax in axis)

    q_arr = q if isinstance(q, _np.ndarray) else _np.asarray(q)
    # numpy prepends q.shape to the output and computes each requested
    # quantile independently, so the per-output cost scales with the number
    # of quantiles requested, not just the reduced axis length; a weights=
    # array additionally routes through the sort-parity weighted formula
    # instead of the cheaper partition-based one.
    q_count = _builtins.max(int(q_arr.size), 1)
    weighted = kwargs.get("weights") is not None
    cost = _tier2_reduction_cost(
        a,
        axis,
        dense_per_output_cost=_quantile_dense_cost(
            axis_dim, q_count, weighted=weighted
        ),
    )

    out_sym = (
        reduce_group(sym, ndim=a.ndim, axis=axis, keepdims=keepdims)
        if sym is not None
        else None
    )
    out_stripped = _to_base_ndarray(out) if out is not None else None
    billing_dtypes: tuple = (a.dtype, billing_operand(q, q_arr))
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    with budget.deduct(
        "nanquantile",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(
            _np.nanquantile,
            _to_base_ndarray(a),
            q,
            axis=axis,
            out=out_stripped,
            keepdims=keepdims,
            **kwargs,
        )
    return _wrap_result(result, out=out, symmetry=out_sym)  # type: ignore[return-value]


nanquantile.__signature__ = _inspect.signature(_np.nanquantile)  # pyright: ignore[reportFunctionMemberAccess]
nanstd = _counted_variance(_np.nanstd, "nanstd", with_sqrt=True)
nansum = _counted_reduction(_np.nansum, "nansum")
nanvar = _counted_variance(_np.nanvar, "nanvar", with_sqrt=False)


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

    Cost = num_output_orbits × per-output cost (Tier-2 partition-based model).
    Per-output cost is piecewise in axis_dim (n) and q.size (k), and also
    depends on whether weights= is given -- see _quantile_dense_cost.
    """
    import math as _math

    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "percentile")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym = _symmetry_of(a)

    # Reduced axis length (n), fed into _quantile_dense_cost below.
    if axis is None:
        axis_dim = _math.prod(a.shape) if a.shape else 1
    elif isinstance(axis, int):
        axis_dim = a.shape[axis]
    else:
        axis_dim = _math.prod(a.shape[ax] for ax in axis)

    q_arr = q if isinstance(q, _np.ndarray) else _np.asarray(q)
    # numpy prepends q.shape to the output and computes each requested
    # quantile independently, so the per-output cost scales with the number
    # of quantiles requested, not just the reduced axis length; a weights=
    # array additionally routes through the sort-parity weighted formula
    # instead of the cheaper partition-based one.
    q_count = _builtins.max(int(q_arr.size), 1)
    weighted = kwargs.get("weights") is not None
    cost = _tier2_reduction_cost(
        a,
        axis,
        dense_per_output_cost=_quantile_dense_cost(
            axis_dim, q_count, weighted=weighted
        ),
    )

    out_sym = (
        reduce_group(sym, ndim=a.ndim, axis=axis, keepdims=keepdims)
        if sym is not None
        else None
    )
    out_stripped = _to_base_ndarray(out) if out is not None else None
    # Same "always divides q by 100 first" rule as nanpercentile: integer/
    # bool `a` always computes in float64, regardless of q's own dtype.
    billing_dtypes: tuple = (mean_compute_dtype(a.dtype), billing_operand(q, q_arr))
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    with budget.deduct(
        "percentile",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
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

    Cost = num_output_orbits × per-output cost (Tier-2 partition-based model).
    Per-output cost is piecewise in axis_dim (n) and q.size (k), and also
    depends on whether weights= is given -- see _quantile_dense_cost.
    """
    import math as _math

    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "quantile")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym = _symmetry_of(a)

    # Reduced axis length (n), fed into _quantile_dense_cost below.
    if axis is None:
        axis_dim = _math.prod(a.shape) if a.shape else 1
    elif isinstance(axis, int):
        axis_dim = a.shape[axis]
    else:
        axis_dim = _math.prod(a.shape[ax] for ax in axis)

    q_arr = q if isinstance(q, _np.ndarray) else _np.asarray(q)
    # numpy prepends q.shape to the output and computes each requested
    # quantile independently, so the per-output cost scales with the number
    # of quantiles requested, not just the reduced axis length; a weights=
    # array additionally routes through the sort-parity weighted formula
    # instead of the cheaper partition-based one.
    q_count = _builtins.max(int(q_arr.size), 1)
    weighted = kwargs.get("weights") is not None
    cost = _tier2_reduction_cost(
        a,
        axis,
        dense_per_output_cost=_quantile_dense_cost(
            axis_dim, q_count, weighted=weighted
        ),
    )

    out_sym = (
        reduce_group(sym, ndim=a.ndim, axis=axis, keepdims=keepdims)
        if sym is not None
        else None
    )
    out_stripped = _to_base_ndarray(out) if out is not None else None
    billing_dtypes: tuple = (a.dtype, billing_operand(q, q_arr))
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    with budget.deduct(
        "quantile",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
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

# ptp: numpy 2.0 removed it from ndarray but np.ptp still exists.
# Honest cost: two full reductions (max-pass + min-pass) + M subtracts
# = 2*(numel − M) + M = 2*numel − M
# where M = num_output_orbits (1 for full reduction, or output numel for axis).


@_counted_wrapper
def ptp(
    a: ArrayLike, axis: int | tuple[int, ...] | None = None, **kwargs: Any
) -> FlopscopeArray:
    """Peak-to-peak range. Cost = 2*numel(input) − numel(output) FLOPs
    (max pass + min pass + subtract)."""
    from flopscope._accumulation._reduction import (
        _normalize_axis,
        _num_output_orbits,
    )

    budget = require_budget()
    # ``out`` arrives inside **kwargs here rather than as a named parameter,
    # which is how it escaped both the normalization every sibling reduction
    # gets and the destination-dtype fold below. Left alone, a wider
    # destination was free -- ptp into a complex128 buffer billed the same as
    # no destination at all, where max/min bill four times as much -- a
    # refused form was charged in full before numpy rejected it, and an
    # unstripped FlopscopeArray reached numpy and tripped an internal guard.
    out = _normalize_out(kwargs.pop("out", None), "ptp")
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    axes_summed = _normalize_axis(axis, a.ndim)
    symmetry = a.symmetry if isinstance(a, SymmetricTensor) else None
    m = _num_output_orbits(tuple(a.shape), axes_summed, symmetry)
    cost = 2 * reduction_cost(a.shape, axis, symmetry=symmetry) + m
    billing_dtype = reduction_billing_dtype(
        a.dtype,
        explicit_dtype=kwargs.get("dtype"),
        out_dtype=out.dtype if isinstance(out, _np.ndarray) else None,
        default_dtype=a.dtype,
    )
    if out is not None:
        kwargs["out"] = _to_base_ndarray(out)
    with budget.deduct(
        "ptp",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(billing_dtype,),
    ):
        stripped = _to_base_ndarray(a)
        if hasattr(_np, "ptp"):
            result = _call_numpy(_np.ptp, stripped, axis=axis, **kwargs)
        else:
            result = _call_numpy(_np.max, stripped, axis=axis, **kwargs) - _call_numpy(
                _np.min, stripped, axis=axis, **kwargs
            )
    return _wrap_result(result, out=out, symmetry=None)  # type: ignore[return-value]


attach_docstring(
    ptp,
    _np.ptp if hasattr(_np, "ptp") else _np.max,
    "counted_reduction",
    "2*numel(input) − numel(output) FLOPs (max pass + min pass + subtract)",
)

if hasattr(_np, "ptp"):
    ptp.__signature__ = _inspect.signature(_np.ptp)  # pyright: ignore[reportFunctionMemberAccess]


# ---------------------------------------------------------------------------
# dot and matmul
# ---------------------------------------------------------------------------


def _einsum_routed_binary(
    op_name: str,
    np_fn: Any,
    subs: str,
    a: Any,
    b: Any,
    *,
    errstate: bool = False,
    nan_check: bool = False,
    out: Any = None,
    **call_kwargs: Any,
) -> Any:
    """Route a binary contraction op's cost + output-symmetry through the einsum
    accumulation model (FMA=2) and run its native numpy op.

    `subs` is the einsum subscript string for this call's operand layout
    (built by the per-op subscript helper). Charges `op_name` exactly once
    (so each op keeps its own weight), preserves operand symmetry/aliasing via
    `_resolve_cost_and_output_symmetry`, and wraps a symmetric result as
    `SymmetricTensor` — mirroring the existing matmul/dot 2-D behavior.
    """
    from flopscope._einsum import _resolve_cost_and_output_symmetry

    budget = require_budget()
    # Above every later read of ``out``: the billing gate below, the forward
    # to numpy, and ``result = out``. A tuple reaching those bills the
    # contraction without the destination's dtype and hands the tuple back.
    out = _normalize_out(out, op_name)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    info = _resolve_cost_and_output_symmetry(subs, a, b)
    inputs_were_whest = isinstance(a, _np.ndarray) and (
        type(a) is not _np.ndarray or type(b) is not _np.ndarray
    )
    billing_dtypes = (a.dtype, b.dtype)
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    resolved = resolve_billing_dtype(billing_dtypes)
    complex_override = contraction_complex_override(info.accumulation, resolved)
    if out is not None:
        call_kwargs = {**call_kwargs, "out": _to_base_ndarray(out)}
    with budget.deduct(
        op_name,
        flop_cost=info.accumulation.total,
        subscripts=info.canonical_subscripts,
        shapes=(a.shape, b.shape),
        dtypes=billing_dtypes,
        complex_factor_override=complex_override,
    ):
        if errstate:
            with _np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                result = _call_numpy(
                    np_fn, _to_base_ndarray(a), _to_base_ndarray(b), **call_kwargs
                )
        else:
            result = _call_numpy(
                np_fn, _to_base_ndarray(a), _to_base_ndarray(b), **call_kwargs
            )
    if nan_check:
        maybe_check_nan_inf(result, op_name)
    if out is not None:
        result = out
    if info.output_symmetry is not None and _validate_result_symmetry(
        result, info.output_symmetry
    ):
        return SymmetricTensor(_np.asarray(result), symmetry=info.output_symmetry)
    return _asflopscope(result) if inputs_were_whest else result


def _outer_contract_subscripts(
    a_ndim: int, b_ndim: int, *, b_contract_axis: int
) -> str:
    """Distinct-label einsum subscripts for an outer-product-style contraction
    (np.dot / np.inner, ndim >= 2): contract a's last axis with b's
    `b_contract_axis` (e.g. -1 for inner, -2 for dot). Output = a's free axes
    then b's free axes.
    """
    import string as _string

    letters = iter(_string.ascii_lowercase + _string.ascii_uppercase)
    a_labels = [next(letters) for _ in range(a_ndim)]
    b_labels = [next(letters) for _ in range(b_ndim)]
    b_ax = b_contract_axis % b_ndim
    b_labels[b_ax] = a_labels[-1]  # tie the contracted axes
    out = "".join(a_labels[:-1]) + "".join(
        lab for ax, lab in enumerate(b_labels) if ax != b_ax
    )
    return f"{''.join(a_labels)},{''.join(b_labels)}->{out}"


@_counted_wrapper
def dot(a: ArrayLike, b: ArrayLike) -> FlopscopeArray:
    """Counted version of np.dot."""
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    if a.ndim == 2 and b.ndim == 2:
        subs = "ij,jk->ik"
    elif a.ndim == 1 and b.ndim == 1:
        subs = "i,i->"
    elif b.ndim == 1:
        subs = _outer_contract_subscripts(a.ndim, 1, b_contract_axis=-1)
    else:
        subs = _outer_contract_subscripts(a.ndim, b.ndim, b_contract_axis=-2)
    return _einsum_routed_binary(  # type: ignore[return-value]
        "dot", _np.dot, subs, a, b, errstate=False, nan_check=True
    )


attach_docstring(dot, _np.dot, "counted_custom", "depends on operand dimensions")


@_counted_wrapper
def matmul(a: ArrayLike, b: ArrayLike) -> FlopscopeArray:
    """Counted version of np.matmul."""
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    if a.ndim == 1 and b.ndim == 1:
        subs = "i,i->"
    elif a.ndim == 1:
        subs = "k,...kn->...n"  # (k,) @ (...,k,n) -> (...,n)
    elif b.ndim == 1:
        subs = "...mk,k->...m"  # (...,m,k) @ (k,) -> (...,m)
    else:
        subs = "...ij,...jk->...ik"  # 2-D and batched/broadcast N-D
    return _einsum_routed_binary(
        "matmul", _np.matmul, subs, a, b, errstate=True, nan_check=True
    )


attach_docstring(matmul, _np.matmul, "counted_custom", "depends on operand dimensions")


# ---------------------------------------------------------------------------
# Custom ops (new)
# ---------------------------------------------------------------------------


@_counted_wrapper
def inner(a: ArrayLike, b: ArrayLike) -> FlopscopeArray:
    """Counted version of np.inner.

    # routes through the shared helper -> wraps tracked inputs like dot/matmul
    """
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    if a.ndim == 1 and b.ndim == 1:
        subs = "i,i->"
    elif a.ndim == 2 and b.ndim == 2:
        subs = "ij,kj->ik"
    else:
        subs = _outer_contract_subscripts(a.ndim, b.ndim, b_contract_axis=-1)
    return _einsum_routed_binary(  # type: ignore[return-value]
        "inner", _np.inner, subs, a, b, errstate=False, nan_check=False
    )


attach_docstring(inner, _np.inner, "counted_custom", "product of matching dims")


@_counted_wrapper
def outer(
    a: ArrayLike, b: ArrayLike, out: FlopscopeArray | None = None
) -> FlopscopeArray:
    """Counted version of np.outer."""
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    out = _normalize_out(out, "outer")
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

    # Strip the FlopscopeArray subclass BEFORE the internal ravel: this flatten
    # is a private cost-model helper (its result only feeds the einsum cost
    # resolver, never the user), so it must stay numpy's free C view. Since
    # FlopscopeArray.ravel() now bills numel@w1, a bare a.ravel() here would
    # over-bill the operand's size on top of the real outer cost (and on a
    # SymmetricTensor would also emit a spurious symmetry-loss warning). The
    # ndim==1 branch never ravels, so it keeps the operand as-is.
    a_flat = _to_base_ndarray(a).ravel() if a.ndim != 1 else a
    b_flat = _to_base_ndarray(b).ravel() if b.ndim != 1 else b
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
    elif isinstance(out, SymmetricTensor):
        # A SymmetricTensor destination is not exotic: a square constant fill
        # picks up an INFERRED symmetry tag, so ``fnp.zeros((n, n))`` -- the
        # obvious way to build a destination -- already is one. numpy is not
        # allowed to write it directly (that would leave the tag standing over
        # data it never saw), so the write is done by ``_wrap_result`` below.
        # Validating the tag HERE, above the deduct, keeps a destination whose
        # tag cannot survive the result free to refuse rather than charged.
        _prepare_symmetric_out(out, None)
    billing_dtypes: tuple = (a.dtype, b.dtype)
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    with budget.deduct(
        "outer",
        flop_cost=cost,
        subscripts=canonical_subs,
        shapes=(a.shape, b.shape),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(
            _np.outer,
            _to_base_ndarray(a),
            _to_base_ndarray(b),
            # The operands are stripped; the destination was not, so a
            # FlopscopeArray ``out`` reached numpy still wrapped and tripped
            # the internal "reached numpy.outer from inside an fnp wrapper"
            # guard -- after the deduct, so the caller paid the whole
            # contraction and then got a RuntimeError. The strip is a
            # zero-copy view, so numpy still writes the caller's buffer.
            out=None if isinstance(out, SymmetricTensor) else _to_base_ndarray(out),
        )
    if output_sym is None and not isinstance(out, SymmetricTensor):
        if out is not None:
            return out
        return result  # type: ignore[return-value]
    # A SymmetricTensor destination reaches _wrap_result even when the result
    # carries no symmetry. It used to take the branch above and return itself
    # UNWRITTEN: numpy had been handed out=None, nothing ever copied the answer
    # across, and the caller got their untouched destination back having paid
    # the whole contraction, with no exception. _wrap_result copies the result
    # in, records the write so no tag outlives the data it described, and drops
    # the inferred tag the destination no longer earns.
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


def _tensordot_einsum_subscripts(a_ndim, b_ndim, a_axes, b_axes):
    """Build einsum subscripts equivalent to a tensordot contraction.

    Returns None if operand rank exceeds the 52-letter budget (caller falls
    back to the dense estimate then).
    """
    import string as _string

    if a_ndim + b_ndim > 52:
        return None
    letters = _string.ascii_letters
    a_labels = list(letters[:a_ndim])
    b_labels = list(letters[a_ndim : a_ndim + b_ndim])
    a_ax = [ax % a_ndim for ax in a_axes]
    b_ax = [ax % b_ndim for ax in b_axes]
    for ai, bi in zip(a_ax, b_ax, strict=False):
        b_labels[bi] = a_labels[ai]  # tie contracted pairs
    out = [a_labels[i] for i in range(a_ndim) if i not in a_ax]
    out += [b_labels[i] for i in range(b_ndim) if i not in b_ax]
    return f"{''.join(a_labels)},{''.join(b_labels)}->{''.join(out)}"


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
        billing_dtypes = (a.dtype, b.dtype)
        resolved = resolve_billing_dtype(billing_dtypes)
        complex_override = contraction_complex_override(info.accumulation, resolved)
        with budget.deduct(
            "tensordot",
            flop_cost=cost,
            subscripts=canonical_subs,
            shapes=(a.shape, b.shape),
            dtypes=billing_dtypes,
            complex_factor_override=complex_override,
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
    # Route cost through einsum when possible (FMA=2 correct); fall back to
    # the old multiply-only dense formula only for rank >52 operands.
    _subs = _tensordot_einsum_subscripts(
        a.ndim, b.ndim, a_contract_axes, b_contract_axes
    )
    # Compose output symmetry from each input's surviving symmetry, with
    # b's axes lifted past a's surviving count so they refer to their
    # final slots in the combined output. Bail on the composition when
    # either group's |G| exceeds dimino_budget (see
    # ``_is_oversized_for_cost_model``).
    a_sym = _symmetry_of(a)
    b_sym = _symmetry_of(b)
    # Only the branches below that reach `_resolve_cost_and_output_symmetry`
    # (i.e. keep an AccumulationCost around, not just its .total) can supply
    # an exact complex_factor_override; the einsum_cost/_symmetry_adjusted_cost
    # branches (oversized symmetry, or rank>52 with no subscripts) compute a
    # plain int with no mu/adds decomposition, so they stay None and rely on
    # complex_factor_for's fail-closed raise if a complex dtype ever reaches
    # them (registry classification "exact").
    accumulation_for_billing = None
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
        if _subs is not None:
            # Oversized symmetry: route cost through the shape-only einsum
            # formula (FMA=2) WITHOUT _resolve_cost_and_output_symmetry, which
            # would re-trigger the dimino enumeration this branch exists to
            # avoid (and raise _DiminoBudgetExceeded).
            from flopscope._flops import einsum_cost

            cost = einsum_cost(_subs, [tuple(a.shape), tuple(b.shape)])
            canonical_subs = _subs
        else:
            dense = (
                _builtins.max(a.size * b.size // contracted, 1) if contracted > 0 else 1
            )
            cost = _symmetry_adjusted_cost(dense, output_shape, out_sym)
            canonical_subs = None
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
        if _subs is not None:
            from flopscope._einsum import _resolve_cost_and_output_symmetry

            _info = _resolve_cost_and_output_symmetry(_subs, a, b)
            cost = _info.accumulation.total
            canonical_subs = _subs
            accumulation_for_billing = _info.accumulation
        else:
            dense = (
                _builtins.max(a.size * b.size // contracted, 1) if contracted > 0 else 1
            )
            cost = _symmetry_adjusted_cost(dense, output_shape, out_sym)
            canonical_subs = None
    billing_dtypes = (a.dtype, b.dtype)
    resolved = resolve_billing_dtype(billing_dtypes)
    # accumulation_for_billing is None for the oversized-symmetry / rank>52
    # branches above, which never reach _resolve_cost_and_output_symmetry and
    # so have no AccumulationCost to draw an exact override from; leave
    # complex_override None there so complex_factor_for's fail-closed "exact"
    # raise still fires if a complex dtype reaches those branches.
    complex_override = (
        contraction_complex_override(accumulation_for_billing, resolved)
        if accumulation_for_billing is not None
        else None
    )
    with budget.deduct(
        "tensordot",
        flop_cost=cost,
        subscripts=canonical_subs,
        shapes=(a.shape, b.shape),
        dtypes=billing_dtypes,
        complex_factor_override=complex_override,
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

    # Strip the subclass before this internal ravel so it stays a free numpy
    # view: its result only feeds the einsum cost resolver below, and a bare
    # a.ravel() would now bill numel@w1 (FlopscopeArray.ravel is counted),
    # over-charging vdot-of-matrices (Frobenius inner product) by both
    # operands' sizes. ndim==1 never ravels, so it is left untouched.
    a_flat = _to_base_ndarray(a).ravel() if a.ndim != 1 else a
    b_flat = _to_base_ndarray(b).ravel() if b.ndim != 1 else b
    info = _resolve_cost_and_output_symmetry("i,i->", a_flat, b_flat)
    cost = info.accumulation.total
    canonical_subs = info.canonical_subscripts
    billing_dtypes = (a.dtype, b.dtype)
    resolved = resolve_billing_dtype(billing_dtypes)
    complex_override = contraction_complex_override(info.accumulation, resolved)
    with budget.deduct(
        "vdot",
        flop_cost=cost,
        subscripts=canonical_subs,
        shapes=(a.shape, b.shape),
        dtypes=billing_dtypes,
        complex_factor_override=complex_override,
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
        "kron",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, b.shape),
        dtypes=(a.dtype, b.dtype),
    ):
        result = _call_numpy(_np.kron, _to_base_ndarray(a), _to_base_ndarray(b))
    return result  # type: ignore[return-value]  # wrapped at fnp.kron import time


attach_docstring(kron, _np.kron, "counted_custom", "output size FLOPs")


@_counted_wrapper
def cross(a: ArrayLike, b: ArrayLike, **kwargs: Any) -> FlopscopeArray:
    """Counted version of np.cross.

    Cost model: 5 ops per output element (3 mults + 1 mult + 1 sub per output
    triple component, which is 5 element-wise ops per output scalar). Issue #69.
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    # Cost is 3 FLOPs per output element (2 mul + 1 sub per component): the
    # 3-vector path keeps the vector dim, the 2-D z-only path drops it (one
    # scalar z per pair); both reduce to 3 × numel(output). Bill from the ACTUAL
    # result so axisa/axisb/axisc and any broadcast shape are exact — inferring
    # the output shape from input dims undercounts when the vector axis is moved
    # by an axis kwarg. Issue #69 (original), audit-completion Task 4 (numel out).
    stripped_a = _to_base_ndarray(a)
    stripped_b = _to_base_ndarray(b)
    with budget.deduct_after(
        "cross", subscripts=None, shapes=(a.shape, b.shape), dtypes=(a.dtype, b.dtype)
    ) as _op:
        result = _call_numpy(_np.cross, stripped_a, stripped_b, **kwargs)
        _op.set_cost(
            _builtins.max(3 * (result.size if hasattr(result, "size") else 1), 1)
        )
    return result  # type: ignore[return-value]


attach_docstring(
    cross,
    _np.cross,
    "counted_custom",
    "3 * numel(output) FLOPs (2 mul + 1 sub per output element)",
)
cross.__signature__ = _inspect.signature(_np.cross)  # pyright: ignore[reportFunctionMemberAccess]


# Use numpy's own stable _NoValue singleton as the "not provided" sentinel for
# diff's prepend/append.  A plain `object()` would break when _pointwise is
# reloaded (test_numpy_version_support does this): the function's compiled
# default would reference the OLD sentinel while the module-level name would
# resolve to the NEW one after reload, causing a false "is not sentinel" check
# that then passes the old sentinel object into numpy.diff as a prepend/append
# value.  np._NoValue survives reloads because it lives in numpy's own module.
_DIFF_NO_VALUE = _np._NoValue  # type: ignore[attr-defined]


@_counted_wrapper
def diff(
    a: ArrayLike,
    n: int = 1,
    axis: int = -1,
    prepend: Any = _DIFF_NO_VALUE,
    append: Any = _DIFF_NO_VALUE,
) -> FlopscopeArray:
    """Counted version of np.diff."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    # numpy.diff implements `for _ in range(n): a = subtract(a[1:], a[:-1])`,
    # so the cost is the SUM over n iterations of (numel_along_axis - k) for
    # k = 1..n. Closed form: n*L - n*(n+1)//2, scaled by the product of
    # the other axes' sizes. Issue #69.
    ax = axis if axis >= 0 else axis + a.ndim
    L = a.shape[ax]
    # numpy concatenates prepend/append along the diff axis before differencing,
    # so the effective axis length L grows by their contribution.
    billing_dtypes: tuple = (a.dtype,)
    if prepend is not _np._NoValue:  # type: ignore[attr-defined]
        p = _np.asanyarray(_to_base_ndarray(prepend))
        L += 1 if p.ndim == 0 else p.shape[ax] if p.ndim > ax else 1
        # np.diff concatenates prepend/append via np.concatenate, which does
        # NOT follow NEP 50 weak-scalar promotion (unlike ufunc arithmetic);
        # bill the coerced array's dtype directly rather than billing_operand.
        billing_dtypes += (p.dtype,)
    if append is not _np._NoValue:  # type: ignore[attr-defined]
        p = _np.asanyarray(_to_base_ndarray(append))
        L += 1 if p.ndim == 0 else p.shape[ax] if p.ndim > ax else 1
        billing_dtypes += (p.dtype,)
    prod_outside = int(_np.prod(a.shape[:ax]))
    prod_inside = int(_np.prod(a.shape[ax + 1 :]))
    per_iter_sum = n * L - n * (n + 1) // 2
    cost = _builtins.max(prod_outside * per_iter_sum * prod_inside, 1)
    # Forward prepend/append to numpy only when provided; strip FlopscopeArrays
    # so numpy's internals don't receive counted subclass instances.
    np_kwargs: dict[str, Any] = {}
    if prepend is not _np._NoValue:  # type: ignore[attr-defined]
        np_kwargs["prepend"] = _to_base_ndarray(prepend)
    if append is not _np._NoValue:  # type: ignore[attr-defined]
        np_kwargs["append"] = _to_base_ndarray(append)
    with budget.deduct(
        "diff",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(_np.diff, _to_base_ndarray(a), n=n, axis=axis, **np_kwargs)
    return result  # type: ignore[return-value]  # wrapped at fnp.diff import time


attach_docstring(
    diff, _np.diff, "counted_custom", "n*L - n*(n+1)/2 FLOPs along the diff axis"
)
diff.__signature__ = _inspect.signature(_np.diff)  # pyright: ignore[reportFunctionMemberAccess]


def _gradient_spacing_surcharge(f_shape, f_size, varargs, axes) -> int:
    """Compute the spacing surcharge for np.gradient coord-array arguments.

    This is a pure helper (not a counted wrapper) so that _np.diff can be
    called outside the @_counted_wrapper guard.  The np.diff call here is
    part of cost accounting only (not routed through the budget), so it must
    live outside the decorated function body.

    Returns the integer surcharge to add to the base gradient cost.
    """
    surcharge = 0
    for ax, v in zip(axes, varargs, strict=False):
        v_arr = _np.asarray(v)
        if v_arr.ndim == 1 and v_arr.size == f_shape[ax]:
            L = f_shape[ax]
            S = f_size
            # Convert to float64 as numpy does for integer-typed coord arrays
            if _np.issubdtype(v_arr.dtype, _np.integer):
                v_arr = v_arr.astype(_np.float64)
            d = _np.diff(v_arr)
            uniform = bool((d == d[0]).all())
            if uniform:
                # numpy still executes diff + equal + all-reduce to detect uniformity
                surcharge += 3 * (L - 1)
            else:
                surcharge += (
                    3
                    * S
                    * _builtins.max(L - 2, 0)
                    // L  # blend pass: +3 per interior elem
                    + 10 * _builtins.max(L - 2, 0)  # coefficient arrays a,b,c
                    + 3 * (L - 1)  # diff + uniformity check
                    + 4 * S // L  # two boundary hyperplanes
                )
    return surcharge


@_counted_wrapper
def gradient(
    f: ArrayLike, *varargs: ArrayLike, **kwargs: Any
) -> FlopscopeArray | list[FlopscopeArray]:
    """Counted version of np.gradient.

    Base cost (no coord arrays, or uniform scalar spacing):
      sum over axes of 2 * f.size * max(shape[ax]-2, 0) // shape[ax]

    Spacing surcharge per coord-array axis (1-D array length == shape[ax]):
      - If coord diffs are bit-exactly uniform (e.g. np.arange): +3*(L-1)
        (numpy still runs diff + equal + all-reduce to detect uniformity)
      - Otherwise (non-uniform floats): + 3*S*(L-2)//L + 10*(L-2) + 3*(L-1) + 4*S//L
        (blend coeff arrays a,b,c; diff+uniformity check; two edge hyperplanes)

    The non-uniform formula is a conservative floor at edge_order=1 boundaries.
    Scaling: S = f.size, L = f.shape[ax].
    """
    budget = require_budget()
    if not isinstance(f, _np.ndarray):
        f = _np.asarray(f)
    if f.ndim == 0:
        cost = 1
    else:
        base = _builtins.max(
            _builtins.sum(
                2 * f.size * _builtins.max(f.shape[ax] - 2, 0) // f.shape[ax]
                for ax in range(f.ndim)
            ),
            1,
        )

        # --- spacing surcharge ---
        # Normalise axes as numpy does
        ax_kw = kwargs.get("axis")
        if ax_kw is None:
            axes = range(f.ndim)
        elif isinstance(ax_kw, int):
            axes = (ax_kw % f.ndim,)
        else:
            axes = tuple(a % f.ndim for a in ax_kw)
        axes = tuple(axes)

        n_varargs = len(varargs)
        if n_varargs == 0 or (n_varargs == 1 and _np.ndim(varargs[0]) == 0):
            surcharge = 0  # no coord arrays or scalar spacing
        elif n_varargs == len(axes):
            # _gradient_spacing_surcharge is a non-decorated helper so _np.diff
            # is called outside the @_counted_wrapper guard
            surcharge = _gradient_spacing_surcharge(f.shape, f.size, varargs, axes)
        else:
            surcharge = 0  # mismatched varargs; let numpy raise; charge base only

        cost = base + surcharge

    # varargs (spacing/coordinate arrays) never change np.gradient's output
    # dtype (verified: gradient(f32, spacing=np.float64(...)) stays float32),
    # so only f's dtype prices the call. np.gradient is mean-shaped on that
    # dtype: integer/bool input always computes in float64 (the central-
    # difference division needs float precision), float input keeps its own
    # width (gradient(float32) -> float32).
    with budget.deduct(
        "gradient",
        flop_cost=cost,
        subscripts=None,
        shapes=(f.shape,),
        dtypes=(mean_compute_dtype(f.dtype),),
    ):
        result = _call_numpy(
            _np.gradient,
            _to_base_ndarray(f),
            *[_to_base_ndarray(v) for v in varargs],
            **kwargs,
        )
    return result  # type: ignore[return-value]  # wrapped at fnp.gradient import time


attach_docstring(
    gradient,
    _np.gradient,
    "counted_custom",
    "uniform: sum_ax 2*S*(L-2)/L; non-uniform axis adds 3*S*(L-2)/L + 10*(L-2) + 3*(L-1) + 4*S/L FLOPs",
)
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
    # to_begin/to_end must be same_kind-compatible with ary's dtype (numpy
    # raises TypeError otherwise) and never change the output dtype, so only
    # ary's dtype prices the call.
    with budget.deduct(
        "ediff1d",
        flop_cost=cost,
        subscripts=None,
        shapes=(ary.shape,),
        dtypes=(ary.dtype,),
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
    """Counted version of np.convolve.

    Per-mode FLOPs (FMA=2); reuses :func:`_correlate_cost` per-mode formula:

    full  (default): ``2*n*m - n - m`` FLOPs
    valid:           ``(2*min(n,m) - 1) * (max(n,m) - min(n,m) + 1)`` FLOPs
    same:            exact dot-length sum per numpy C layout

    Previously charged mode-blind ``2*n*m - n - m`` for all modes, which
    over-counted valid and same by a factor of ~2–10×.
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(v, _np.ndarray):
        v = _np.asarray(v)
    # _correlate_cost gives exact per-mode FLOPs.  np.convolve(a, v, mode) is
    # mathematically equivalent to np.correlate(a, v[::-1], mode) up to a
    # time-reversal, so the per-mode output length (and hence FLOP count) is
    # identical.  The one difference is that correlate "full" uses the +1
    # constant (off-by-one fix from PR #123); convolve "full" uses 2nm-n-m
    # (matching the prior formula) so we keep backwards-compatibility here.
    _mode_str = str(mode).lower()[:1] if not isinstance(mode, int) else mode
    if _mode_str == "f" or _mode_str == 2:
        # full: traditional convolve formula (no +1) for backwards compat
        flop_cost = _builtins.max(2 * a.size * v.size - a.size - v.size, 1)
    else:
        flop_cost = _correlate_cost(a.size, v.size, mode)
    with budget.deduct(
        "convolve",
        flop_cost=flop_cost,
        subscripts=None,
        shapes=(a.shape, v.shape),
        dtypes=(a.dtype, v.dtype),
    ):
        result = _call_numpy(
            _np.convolve, _to_base_ndarray(a), _to_base_ndarray(v), mode=mode
        )  # type: ignore[arg-type]
    return result  # type: ignore[return-value]  # wrapped at fnp.convolve import time


attach_docstring(
    convolve,
    _np.convolve,
    "counted_custom",
    "per-mode FLOPs (FMA=2): full 2nm-n-m; valid (2*min-1)*(max-min+1); same exact dot-length sum",
)


def _correlate_cost(n: int, m: int, mode) -> int:
    """Per-mode FLOPs for np.correlate(a, v) with len(a)=n, len(v)=m.

    Normalise mode: numpy accepts int 0/1/2 = valid/same/full and
    case-insensitive strings matched on first letter.  Unknown strings fall
    back to "f" (conservative max; numpy will raise on truly invalid modes).

    Exact closed forms validated against ground-truth FLOPs for all (n,m) in
    {1..10, 17, 100, 101}²:

      full  (f): 2*n*m - n - m + 1
      valid (v): (2*mn - 1) * (mx - mn + 1)   where mn=min(n,m), mx=max(n,m)
      same  (s): exact dot-length sum via numpy C layout (see below)
    """
    _mode_map = {0: "v", 1: "s", 2: "f"}
    if isinstance(mode, int):
        _m = _mode_map.get(mode, "f")
    else:
        _m = str(mode).lower()[:1]
        if _m not in ("v", "s", "f"):
            _m = "f"  # conservative fallback; numpy will raise for truly invalid modes

    mn = _builtins.min(n, m)
    mx = _builtins.max(n, m)

    if _m == "f":
        # full: 2*n*m - n - m + 1 (exact; fixes prior off-by-one)
        return _builtins.max(2 * n * m - n - m + 1, 1)
    elif _m == "v":
        # valid: each of (mx - mn + 1) output positions uses a dot of length mn;
        # FMA=2: 2*mn - 1 FLOPs per position
        return _builtins.max((2 * mn - 1) * (mx - mn + 1), 1)
    else:
        # same: numpy C implementation uses a variable-length dot per output position.
        # The output has mx elements.  The interior (mx - mn + 1) positions each use
        # a full-length dot (mn muls + mn-1 adds = 2*mn - 1 FLOPs).  The n_left left-
        # edge and n_right right-edge positions use progressively shorter dots.
        # n_left = mn // 2  (left padding)
        # n_right = mn - n_left - 1  (right padding)
        # Edge FLOPs = sum_{k=1}^{n_left} (2k - 1) + sum_{k=1}^{n_right} (2k - 1)
        #            = n_left^2 + n_right^2  (sum of odd = k^2)
        # Simplify: k*mn - k*(k+1)//2 for each edge group where each position i has
        # dot-length starting from 1 and stepping by 1 → FLOPs = 2*i - 1 per position
        n_left = mn // 2
        n_right = mn - n_left - 1
        edge_flops = (n_left * mn - n_left * (n_left + 1) // 2) + (
            n_right * mn - n_right * (n_right + 1) // 2
        )
        interior = mn * (mx - mn + 1)
        return _builtins.max(2 * (interior + edge_flops) - mx, 1)


@_counted_wrapper
def correlate(a: ArrayLike, v: ArrayLike, mode: str = "valid") -> FlopscopeArray:
    """Counted version of np.correlate.

    Per-mode FLOPs (FMA=2):
      full  (default for convolve): 2*n*m - n - m + 1
      valid (numpy default):        (2*min-1) * (max-min+1)
      same:                         exact dot-length sum per numpy C layout
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(v, _np.ndarray):
        v = _np.asarray(v)
    cost = _correlate_cost(a.size, v.size, mode)
    with budget.deduct(
        "correlate",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, v.shape),
        dtypes=(a.dtype, v.dtype),
    ):
        result = _call_numpy(
            _np.correlate, _to_base_ndarray(a), _to_base_ndarray(v), mode=mode
        )  # type: ignore[arg-type]
    return result  # type: ignore[return-value]  # wrapped at fnp.correlate import time


attach_docstring(
    correlate,
    _np.correlate,
    "counted_custom",
    "per-mode FLOPs (FMA=2): full 2nm-n-m+1; valid (2*min-1)*(max-min+1); same exact dot-length sum",
)


def _cov_cost(x, y=None, rowvar=True):
    """Cost for cov: 2 * f^2 * s + 2 * f * s FLOPs.

    For a (f, s) input: f features, s samples. ``rowvar=True`` (numpy's
    default) reads x as (features, samples) -- f=shape[0], s=shape[1].
    ``rowvar=False`` transposes that reading: each COLUMN is a variable and
    each ROW an observation -- f=shape[1], s=shape[0]. Same for ``y``: numpy
    transposes it under ``rowvar=False`` before stacking, so its variable
    count is read off ``shape[1]`` there too.

    - Gram term:      f^2 dot products of length s → ``2 * f^2 * s`` FLOPs (FMA=2)
    - Centering pass: subtract per-feature mean from each sample → ``2 * f * s``
      FLOPs (1 mean divide + 1 subtract per element, 2 * f * s total).

    Previously only the Gram term was counted, under-counting by ``2 * f * s``.
    Previously ``rowvar`` was also ignored entirely, pinning every call to the
    ``rowvar=True`` binding regardless of what was passed -- an unbounded
    under-bill under ``rowvar=False`` whenever the sample axis is larger than
    the feature axis (the common observations-as-rows layout).
    """
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    if x.ndim == 1:
        f, s = 1, x.shape[0]
    elif rowvar:
        f, s = x.shape[0], x.shape[1]
    else:
        f, s = x.shape[1], x.shape[0]
    if y is not None:
        y_arr = _np.asarray(y)
        if y_arr.ndim == 1:
            f2 = 1
        elif rowvar:
            f2 = y_arr.shape[0]
        else:
            f2 = y_arr.shape[1]  # numpy transposes y under rowvar=False
        f += f2
    return _builtins.max(2 * f * f * s + 2 * f * s, 1)


def _corrcoef_cost(x, y=None, rowvar=True):
    """Cost for corrcoef: cov_cost(x, y) + 2 * f^2 + f FLOPs.

    ``f`` here is the same rowvar-dependent feature count as ``_cov_cost``
    (the output correlation matrix is f x f, so the normalization term must
    track the same orientation as the Gram/centering term it wraps).

    Normalization step: f^2 divides (divide cov[i,j] by std_i * std_j) plus
    f sqrt calls (one per feature) → ``2 * f^2 + f`` additional FLOPs.
    The ``2 * f^2`` counts the f^2 element-wise divides (FMA=2 convention gives
    each divide as 2 FLOPs) and ``f`` counts the sqrt calls (1 FLOP each at the
    transcendental-as-1 convention used for simple elementwise ops here).
    """
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    if x.ndim == 1:
        f = 1
    elif rowvar:
        f = x.shape[0]
    else:
        f = x.shape[1]
    if y is not None:
        y_arr = _np.asarray(y)
        if y_arr.ndim == 1:
            f2 = 1
        elif rowvar:
            f2 = y_arr.shape[0]
        else:
            f2 = y_arr.shape[1]
        f += f2
    return _builtins.max(_cov_cost(x, y, rowvar) + 2 * f * f + f, 1)


@_counted_wrapper
def corrcoef(x: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> FlopscopeArray:
    """Counted version of np.corrcoef. Cost: (2*f^2*s + 2*f*s) + 2*f^2 + f FLOPs."""
    budget = require_budget()
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    cost = _corrcoef_cost(x, y, rowvar=kwargs.get("rowvar", True))
    # np.corrcoef always computes at least in float64 (numpy internally uses
    # np.result_type(x, y, np.float64)), regardless of the input dtype.
    billing_dtypes: tuple = (x.dtype, _np.dtype("float64"))
    if y is not None:
        y_arr = y if isinstance(y, _np.ndarray) else _np.asarray(y)
        billing_dtypes += (y_arr.dtype,)
    with budget.deduct(
        "corrcoef",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape,),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(
            _np.corrcoef,
            _to_base_ndarray(x),
            y=_to_base_ndarray(y) if y is not None else None,  # type: ignore[arg-type]
            **kwargs,
        )
    return result  # type: ignore[return-value]  # wrapped at fnp.corrcoef import time


attach_docstring(
    corrcoef,
    _np.corrcoef,
    "counted_custom",
    r"$(2 f^2 s + 2 f s) + 2 f^2 + f$ FLOPs (cov + normalization)",
)
corrcoef.__signature__ = _inspect.signature(_np.corrcoef)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def cov(m: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> FlopscopeArray:
    """Counted version of np.cov. Cost: 2 * f^2 * s + 2 * f * s FLOPs (Gram + centering)."""
    budget = require_budget()
    if not isinstance(m, _np.ndarray):
        m = _np.asarray(m)
    cost = _cov_cost(m, y, rowvar=kwargs.get("rowvar", True))
    # np.cov always computes at least in float64 (numpy internally uses
    # np.result_type(m, y, np.float64)), regardless of the input dtype.
    billing_dtypes: tuple = (m.dtype, _np.dtype("float64"))
    if y is not None:
        y_arr = y if isinstance(y, _np.ndarray) else _np.asarray(y)
        billing_dtypes += (y_arr.dtype,)
    with budget.deduct(
        "cov",
        flop_cost=cost,
        subscripts=None,
        shapes=(m.shape,),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(
            _np.cov,
            _to_base_ndarray(m),
            y=_to_base_ndarray(y) if y is not None else None,  # type: ignore[arg-type]
            **kwargs,
        )
    return result  # type: ignore[return-value]  # wrapped at fnp.cov import time


attach_docstring(
    cov, _np.cov, "counted_custom", r"$2 f^2 s + 2 f s$ FLOPs (Gram + centering)"
)
cov.__signature__ = _inspect.signature(_np.cov)  # pyright: ignore[reportFunctionMemberAccess]


@_counted_wrapper
def trapezoid(
    y: ArrayLike, x: ArrayLike | None = None, dx: float = 1.0, axis: int = -1
) -> FlopscopeArray:
    """Counted version of np.trapezoid."""
    budget = require_budget()
    if not isinstance(y, _np.ndarray):
        y = _np.asarray(y)
    billing_dtypes: tuple = (y.dtype, billing_operand(dx, _np.asarray(dx)))
    if x is not None:
        x_arr = x if isinstance(x, _np.ndarray) else _np.asarray(x)
        billing_dtypes += (billing_operand(x, x_arr),)
    with budget.deduct(
        "trapezoid",
        flop_cost=4 * y.size,
        subscripts=None,
        shapes=(y.shape,),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(
            _np.trapezoid,
            _to_base_ndarray(y),
            x=_to_base_ndarray(x) if x is not None else None,  # type: ignore[arg-type]
            dx=dx,
            axis=axis,
        )
    return result  # type: ignore[return-value]  # wrapped at fnp.trapezoid import time


attach_docstring(
    trapezoid, _np.trapezoid, "counted_custom", "4 * numel(input) FLOPs (FMA=2)"
)


if hasattr(_np, "trapz"):

    @_counted_wrapper
    def trapz(  # pyright: ignore[reportRedeclaration]
        y: ArrayLike, x: ArrayLike | None = None, dx: float = 1.0, axis: int = -1
    ) -> FlopscopeArray:
        """Counted version of np.trapz (deprecated alias for trapezoid)."""
        budget = require_budget()
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        billing_dtypes: tuple = (y.dtype, billing_operand(dx, _np.asarray(dx)))
        if x is not None:
            x_arr = x if isinstance(x, _np.ndarray) else _np.asarray(x)
            billing_dtypes += (billing_operand(x, x_arr),)
        with budget.deduct(
            "trapz",
            flop_cost=4 * y.size,
            subscripts=None,
            shapes=(y.shape,),
            dtypes=billing_dtypes,
        ):
            result = _call_numpy(
                _np.trapz,
                _to_base_ndarray(y),
                x=_to_base_ndarray(x) if x is not None else None,
                dx=dx,
                axis=axis,
            )
        return result  # type: ignore[return-value]  # wrapped at fnp.trapz import time

    attach_docstring(
        trapz, _np.trapz, "counted_custom", "4 * numel(input) FLOPs (FMA=2)"
    )

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
    fp_arr = _np.asarray(fp)
    n = _builtins.max(x.size, 1)
    xp_len = _builtins.max(xp_arr.size, 1)
    cost = _builtins.max(3 * n + n * _ceil_log2(xp_len), 1)
    # np.interp always computes the x/xp search+blend in float64 regardless
    # of x/xp's own precision, but the interpolated VALUE keeps fp's dtype
    # (e.g. complex fp -> complex128 output); bill both contributions.
    billing_dtypes = (_np.dtype("float64"), fp_arr.dtype)
    with budget.deduct(
        "interp",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape, xp_arr.shape),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(
            _np.interp,
            _to_base_ndarray(x),
            _to_base_ndarray(xp),  # type: ignore[arg-type]
            _to_base_ndarray(fp),  # type: ignore[arg-type]
            **kwargs,
        )
    return result  # type: ignore[return-value]  # wrapped at fnp.interp import time


attach_docstring(
    interp,
    _np.interp,
    "counted_custom",
    "3*n + n*ceil(log2(xp)) FLOPs (arithmetic + search)",
)
interp.__signature__ = _inspect.signature(_np.interp)  # pyright: ignore[reportFunctionMemberAccess]
