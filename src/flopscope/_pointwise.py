"""Counted pointwise operations and reductions for flopscope."""

from __future__ import annotations

import builtins as _builtins
import functools as _functools
import inspect as _inspect
import operator as _operator
import string as _string
import warnings as _warnings
from math import prod as _math_prod
from typing import Any

import numpy as _np
from numpy.exceptions import AxisError as _AxisError
from numpy.lib.array_utils import normalize_axis_tuple as _normalize_axis_tuple
from numpy.typing import ArrayLike

from flopscope._accumulation._cost import AccumulationCost, contraction_complex_override
from flopscope._budget import (
    _call_numpy,
    _call_numpy_with_python_callbacks,
    _counted_wrapper,
)
from flopscope._config import get_setting as _get_setting
from flopscope._docstrings import attach_docstring
from flopscope._dtype_billing import (
    billing_operand,
    heavier_billing_dtype,
    integer_to_float64_min_dtype,
    mean_compute_dtype,
    multi_store_billing_dtypes,
    natural_output_dtypes,
    reduction_billing_dtype,
    resolve_billing_dtype,
    store_billing_dtypes,
    sum_accumulator_dtype,
    ufunc_resolver_operand,
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
    _warn_remote_callback,
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

# Reductions whose result is an INDEX rather than a value. numpy fixes their
# output dtype at ``np.intp`` regardless of the input's precision, and
# constrains ``out=`` by KIND rather than by width: any integer or boolean
# buffer is accepted at any width, and every float or complex one is refused
# outright, before the reduction runs. Two consequences, both handled in
# ``_counted_reduction``:
#
#   * The refusal is decidable from the destination's dtype alone, so it
#     belongs ABOVE ``budget.deduct`` and must cost zero. It used to sit
#     below: ``argmin`` on 10,000 float32 with a float64 destination charged
#     19,998 FLOPs and only then raised ``ValueError`` from numpy.
#   * The destination can never be the accumulator. An index buffer holds
#     positions, not the values being compared, so its width says nothing
#     about the arithmetic -- ``out`` is kept out of
#     ``reduction_billing_dtype`` for these ops entirely, which is what makes
#     supplying the ``intp`` buffer numpy would have allocated anyway
#     price-neutral against the bare call (both 9,999 on the case above,
#     where the destination previously doubled the rate).
#
# Membership is not hand-maintained: ``test_index_reduction_out_billing``
# re-derives it from the registry by probing numpy itself, so a reduction
# that starts returning indices cannot be missed.
_INDEX_RETURNING_REDUCTIONS = frozenset(
    {
        "argmax",
        "argmin",
        "nanargmax",
        "nanargmin",
    }
)


def _refuse_non_index_destination(op_name: str, out: object, axis: object) -> None:
    """Refuse an ``out=`` numpy cannot use as an index buffer -- before charging.

    numpy's own rule, measured on 2.2.6 across every scalar dtype and both
    the ``axis=None`` and ``axis=`` forms of all four ops, is exactly
    ``can_cast(out.dtype, intp, casting="safe")``: bool and every signed or
    unsigned integer narrower than ``intp`` are accepted (numpy casts the
    index down on store), ``uint64``, every float, every complex and every
    non-numeric dtype are refused. Reproducing the predicate rather than
    inventing a stricter one is what keeps this guard from failing a call
    plain numpy would allow.

    Class and wording mirror numpy's, for the same reason
    :func:`_normalize_out` mirrors the ufunc parser: intercepting the
    argument earlier than numpy does should not change what the failure looks
    like to a caller. numpy says "scalar" for the whole-array reduction and
    "array data" for a per-axis one.
    """
    if not isinstance(out, _np.ndarray):
        return
    if _np.can_cast(out.dtype, _np.intp, casting="safe"):
        return
    noun = "scalar" if axis is None else "array data"
    raise TypeError(
        f"{op_name}(): Cannot cast {noun} from {out.dtype!r} to "
        f"{_np.dtype(_np.intp)!r} according to the rule 'safe'"
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
# numpy < 2.1 has no integer loops for ceil/floor/trunc (nor fix, their
# composite): integer input is promoted and computed in the size-mapped float
# loop (int8 -> float16, ..., int32 -> float64), exactly like sin. numpy 2.1
# added identity integer loops, making them integer-preserving. Membership is
# probed from the running numpy -- never a version table -- so billing always
# tracks what this backend actually computes.
_UNARY_FLOAT_LOOP_OPS |= frozenset(
    op_name
    for op_name, fn in (
        ("ceil", _np.ceil),
        ("floor", _np.floor),
        ("trunc", _np.trunc),
        ("fix", _np.fix),
    )
    if fn(_np.ones(1, dtype=_np.int32)).dtype.kind == "f"
)
# Unary ops with NO size-mapped integer loop: numpy always computes them in
# (at least) float64 for integer/bool input regardless of the input's own
# width -- unlike the size-mapped _UNARY_FLOAT_LOOP_OPS family (i0(int8) ->
# float64, not float16, matching numpy's Chebyshev-polynomial/sin-ratio
# implementations that don't have narrower loops). Float/complex inputs keep
# their own width. integer_to_float64_min_dtype's "any int kind -> float64,
# float/complex unchanged" mapping already expresses exactly this rule (it
# doesn't care about arity), so it is reused here instead of a new helper.
_UNARY_FLOAT64_MIN_OPS = frozenset({"i0", "sinc"})

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


@_functools.cache
def _seen_label_budget(op_name: str) -> bool:
    """Return ``True`` once per op, ``False`` thereafter.

    Deduplicates :func:`_warn_label_budget_once` per process, using the same
    miss-vs-hit discipline as :func:`_seen_oversized`.
    """
    return True


def _warn_label_budget_once(op_name: str) -> None:
    """Emit :class:`CostFallbackWarning` once per op for a label-budget fallback.

    Called when symmetry savings or repeated-operand savings *could* be
    forfeited, which is broader than when they actually are: the charge is
    often the same either way. When neither could apply the arithmetic
    fallback is exact, so this does not fire on the common path.

    The warning is a diagnostic, not a control: it is suppressible via
    ``flops.configure(symmetry_warnings=False)``, and correct billing must
    not depend on anyone seeing it.
    """
    if not _get_setting("symmetry_warnings"):
        return
    info_before = _seen_label_budget.cache_info()
    _seen_label_budget(op_name)
    if _seen_label_budget.cache_info().hits > info_before.hits:
        return  # already warned for this op
    _warnings.warn(
        f"{op_name}: combined operand rank exceeds the {_SUBSCRIPT_BUDGET}-"
        f"letter einsum subscript budget, so this contraction is priced "
        f"without a subscript string. The charge may be higher than what the "
        f"einsum path would compute, and is never lower.",
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

    A dense cost of 0 stays 0. The floor of 1 below exists so *real* work
    whose symmetry-scaled charge rounds down to nothing still costs
    something; a zero-sized (empty) contraction performed no multiplies and
    no accumulations, so there is nothing for the ratio to scale and nothing
    to floor. Charging it 1 would also break the zero-domain invariant
    ``_dense_accumulation_cost`` and ``aggregate_einsum`` both hold, and
    would desynchronise ``cost`` from ``accumulation.total`` at the
    contraction call sites — withholding the exact complex factor they
    derive from that accumulation and tripping ``complex_factor_for``'s
    fail-closed raise on an otherwise valid complex call.

    TODO: this is a placeholder. The real algorithmic cost depends on
    whether the underlying NumPy call (or the flopscope wrapper) actually
    skips redundant work — today, our wrappers compute the dense
    output and discard the duplicates. Replace with a per-op
    algorithmic-cost model when one is available.
    """
    dense_cost = int(dense_cost)
    if dense_cost == 0:
        return 0
    if output_symmetry is None:
        return dense_cost
    # Use the Python builtins to avoid the module-level ``max`` /
    # ``prod`` reduction wrappers that shadow them in this module.
    dense_output = _builtins.max(_math_prod(output_shape), 1)
    if dense_output <= 1:
        return dense_cost
    unique = unique_elements_for_shape(output_symmetry, output_shape)
    if unique >= dense_output:
        return dense_cost
    # Integer-division form avoids float drift on large arrays.
    return _builtins.max(dense_cost * int(unique) // dense_output, 1)


_ARRAY_UFUNC_MISSING = object()


class _ForeignUfuncResult:
    """Raw result of a call whose dispatch involved foreign NEP 13 code."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _static_array_ufunc_implementation(value):
    """Read a type's NEP 13 implementation without invoking its metaclass."""
    return _inspect.getattr_static(type(value), "__array_ufunc__", _ARRAY_UFUNC_MISSING)


def _has_foreign_array_ufunc(value) -> bool:
    """Whether *value* can dispatch a raw ufunc call to foreign Python code."""
    if isinstance(value, FlopscopeArray):
        return False
    implementation = _static_array_ufunc_implementation(value)
    return (
        implementation is not _ARRAY_UFUNC_MISSING
        and implementation is not None
        and implementation is not _np.ndarray.__array_ufunc__
    )


def _flatten_ufunc_out_slots(out) -> tuple:
    """Flatten only true ufunc output slots, never arbitrary sequences."""
    if type(out) is tuple:
        return out
    return (out,)


def _preflight_ufunc_out_arity(out, nout: int) -> None:
    """Match NumPy's tuple-arity error before inspecting ufunc participants."""
    if type(out) is tuple and len(out) != nout:
        raise ValueError("The 'out' tuple must have exactly one entry per ufunc output")


def _call_ufunc_with_protocol_timing(
    op_name, fn, *args, protocol_operands=(), **kwargs
):
    """Call a ufunc, attributing foreign protocol callbacks to residual time."""
    if _builtins.any(_has_foreign_array_ufunc(value) for value in protocol_operands):
        _warn_remote_callback(op_name)
        return _ForeignUfuncResult(
            _call_numpy_with_python_callbacks(fn, *args, **kwargs)
        )
    return _call_numpy(fn, *args, **kwargs)


def _restore_foreign_ufunc_out_identity(result, out, stripped):
    """Map only NumPy's canonical stripped-``out`` result back to the caller."""
    if isinstance(result, _ForeignUfuncResult):
        value = result.value
        if (
            type(value) is tuple
            and isinstance(out, tuple)
            and isinstance(stripped, tuple)
            and len(value) == len(stripped)
            and _builtins.all(
                original is None or returned is stripped_value
                for original, stripped_value, returned in zip(
                    out, stripped, value, strict=True
                )
            )
        ):
            value = tuple(
                original
                if original is not None and returned is stripped_value
                else returned
                for original, stripped_value, returned in zip(
                    out, stripped, value, strict=True
                )
            )
        elif (
            type(value) is tuple
            and not isinstance(out, tuple)
            and not isinstance(stripped, tuple)
            and len(value) == 1
            and value[0] is stripped
        ):
            value = (out,)
        elif out is not None and value is stripped:
            value = out
        return _ForeignUfuncResult(value)
    return result


def _call_with_optional_out(
    np_func,
    *args,
    out=None,
    supports_out=False,
    callback_op_name=None,
    defer_out_write_tracking=False,
    **kwargs,
):
    # Strip flopscope subclasses (FlopscopeArray / SymmetricTensor) from arrays so
    # the raw NumPy call does not re-dispatch through ``__array_ufunc__`` /
    # ``__array_function__`` and recurse infinitely. Python scalars and
    # other non-array values pass through unchanged so NEP 50 weak-typing
    # rules continue to apply at the NumPy boundary.
    args = tuple(_to_base_ndarray(a) for a in args)
    protocol_args = args
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
        if isinstance(np_func, _np.ufunc):
            return _call_ufunc_with_protocol_timing(
                callback_op_name or np_func.__name__,
                np_func,
                *args,
                protocol_operands=protocol_args,
                **kwargs,
            )
        return _call_numpy(np_func, *args, **kwargs)
    if supports_out:
        if isinstance(np_func, _np.ufunc):
            if defer_out_write_tracking:
                call_result = _call_ufunc_with_protocol_timing(
                    callback_op_name or np_func.__name__,
                    np_func,
                    *args,
                    out_stripped,
                    protocol_operands=(*protocol_args, out),
                    **kwargs,
                )
            else:
                call_result = _call_ufunc_with_protocol_timing(
                    callback_op_name or np_func.__name__,
                    np_func,
                    *args,
                    out=out_stripped,
                    protocol_operands=(*protocol_args, out),
                    **kwargs,
                )
            return _restore_foreign_ufunc_out_identity(
                call_result,
                out,
                out_stripped,
            )
        return _call_numpy(np_func, *args, out=out_stripped, **kwargs)
    result = _call_numpy(np_func, *args, **kwargs)
    # Fallback copy when np_func doesn't natively support out=. This is
    # flopscope's overhead, NOT routed through _call_numpy -- so the write has
    # to be recorded here, or a tag on `out`'s buffer would outlive its data.
    _np.copyto(out_stripped, _np.asarray(result), casting="unsafe")  # type: ignore[arg-type]
    note_write(out)
    return out


def _symmetric_out_scratch(out: SymmetricTensor) -> _np.ndarray:
    """Copy ``out`` into isolated storage while preserving its exact layout."""
    source = _to_base_ndarray(out)
    if source.size == 0:
        min_offset = 0
        storage_nbytes = 0
    else:
        byte_extents = tuple(
            (dimension - 1) * stride
            for dimension, stride in zip(source.shape, source.strides, strict=True)
        )
        min_offset = _builtins.sum(_builtins.min(0, extent) for extent in byte_extents)
        max_offset = _builtins.sum(_builtins.max(0, extent) for extent in byte_extents)
        storage_nbytes = max_offset - min_offset + source.dtype.itemsize
    backing = _np.empty(storage_nbytes, dtype=_np.uint8)
    scratch = _np.ndarray(
        shape=source.shape,
        dtype=source.dtype,
        buffer=backing,
        offset=-min_offset,
        strides=source.strides,
    )
    _np.copyto(scratch, source, casting="no")
    if not source.flags.writeable:
        scratch.flags.writeable = False
    return scratch


def _logical_array_bytes(array: _np.ndarray) -> bytes:
    """Snapshot logical element bytes independent of strides and aliasing."""
    return array.tobytes(order="C")


def _snapshot_symmetric_out(
    out: SymmetricTensor,
) -> tuple[_np.ndarray, bytes, _np.ndarray, object, bool]:
    """Capture transaction state before exposing a real output to a callback."""
    base = _to_base_ndarray(out)
    return (
        base,
        _logical_array_bytes(base),
        _np.array(base, copy=True, order="C", subok=False),
        out.symmetry,
        out._symmetry_inferred,
    )


def _finish_foreign_symmetric_out(
    out: SymmetricTensor,
    *,
    base: _np.ndarray,
    before: bytes,
    saved_data: _np.ndarray,
    target_symmetry,
    previous_symmetry,
    previous_inferred: bool,
) -> None:
    """Validate a changed callback output, rolling back invalid mutations."""
    if _logical_array_bytes(base) == before:
        return
    try:
        verified = _validate_result_symmetry(base, target_symmetry)
    except SymmetryError:
        _np.copyto(base, saved_data, casting="no")
        note_write(out)
        out._symmetry = previous_symmetry
        out._symmetry_inferred = previous_inferred
        raise
    note_write(out)
    if verified:
        out._symmetry = target_symmetry


def _call_with_optional_multi_out(
    np_func, *args, out=None, nout, callback_op_name=None, **kwargs
):
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
    protocol_args = args
    for k, v in list(kwargs.items()):
        if isinstance(v, _np.ndarray):
            kwargs[k] = _to_base_ndarray(v)
        elif isinstance(v, (tuple, list)):
            kwargs[k] = _to_base_ndarray_tree(v)
    if out is None:
        if isinstance(np_func, _np.ufunc):
            return _call_ufunc_with_protocol_timing(
                callback_op_name or np_func.__name__,
                np_func,
                *args,
                protocol_operands=protocol_args,
                **kwargs,
            )
        return _call_numpy(np_func, *args, **kwargs)
    if not isinstance(out, tuple) or len(out) != nout:
        length_repr = len(out) if hasattr(out, "__len__") else "?"
        raise TypeError(
            f"multi-output {getattr(np_func, '__name__', '?')} requires "
            f"out= to be a tuple of length {nout}; got "
            f"{type(out).__name__} of length {length_repr}"
        )
    protocol_operands = (*protocol_args, *out)
    stripped = tuple(_to_base_ndarray(o) if o is not None else None for o in out)
    if isinstance(np_func, _np.ufunc):
        result = _restore_foreign_ufunc_out_identity(
            _call_ufunc_with_protocol_timing(
                callback_op_name or np_func.__name__,
                np_func,
                *args,
                out=stripped,
                protocol_operands=protocol_operands,
                **kwargs,
            ),
            out,
            stripped,
        )
    else:
        result = _call_numpy(np_func, *args, out=stripped, **kwargs)
    if isinstance(result, _ForeignUfuncResult):
        return result
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


def _wrap_metered_result(result):
    """Wrap an ndarray result; hand a numpy scalar back exactly as numpy made it.

    The two branches mirror the server's own ``_pack_result``, which tests
    ``isinstance(result, np.ndarray)`` BEFORE ``isinstance(result, np.generic)``:

    * an ndarray is stored as a handle and reaches the participant as a
      ``RemoteArray``, whose arithmetic is dispatched to the server and billed;
    * a numpy scalar is *usually* packed by value and reaches them as a
      ``RemoteScalar``, whose arithmetic runs locally in Python and is billed
      nothing. "Usually", because ``_pack_result`` packs by value only when
      ``.item()`` is msgpack-native: a ``complex128`` scalar is not, so it
      falls back to ``store_array`` and reaches them as a handle after all.
      See :func:`test_complex_scalar_results_are_a_recorded_residual_gap`.

    So for an ndarray result the in-process path was the one that was wrong --
    it handed back a raw ``numpy.ndarray`` and billed 0 for downstream
    arithmetic the grader was charging (#193) -- and wrapping it makes the two
    agree. For a by-value scalar result the two already agree at 0, and
    wrapping it would *break* that agreement: a 0-d ``FlopscopeArray`` is an
    ``ndarray``, so the server would store it as a handle and start charging
    downstream arithmetic that costs nothing today. That is a repricing, not a
    fix, so scalars are passed through untouched.

    That leaves the handle-packed scalars above as a residual #193 gap rather
    than a resolved case: locally free, billed on the grader. Closing it is not
    just "wrap them too" -- it is wire-neutral only for the dtypes that already
    fall back to a handle, so it needs its own dtype-by-dtype measurement.

    Tuple results are handled element by element, matching ``_pack_result``'s
    own per-element branch order for a tuple return.
    """
    if isinstance(result, tuple):
        return tuple(_wrap_metered_result(part) for part in result)
    if isinstance(result, _np.ndarray):
        return _wrap_result(result)
    return result


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
    is_ufunc = isinstance(np_func, _np.ufunc)

    @_counted_wrapper
    def wrapper(
        x: ArrayLike, out: FlopscopeArray | None = None, **kwargs: Any
    ) -> FlopscopeArray:
        budget = require_budget()
        # Ufuncs must reject an opt-out before inspecting out or billing. The
        # non-ufunc NumPy dispatch helpers materialize such operands instead.
        if is_ufunc:
            _preflight_ufunc_out_arity(out, np_func.nout)
            _preflight_ufunc_opt_out(x, *_flatten_ufunc_out_slots(out))
        out = _normalize_out(out, op_name)
        symmetry = _symmetry_of(x)
        if is_ufunc:
            x, x_fwd = _resolve_ufunc_data_operand(x)
        else:
            if not isinstance(x, _np.ndarray):
                x = _np.asarray(x)
            x_fwd = x
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
                billing_dtypes = (integer_to_float64_min_dtype(resolved),)
        with budget.deduct(
            op_name,
            flop_cost=cost,
            subscripts=None,
            shapes=(x.shape,),
            dtypes=billing_dtypes,
        ):
            foreign_symmetric_out = (
                is_ufunc
                and isinstance(out, SymmetricTensor)
                and _has_foreign_array_ufunc(x_fwd)
            )
            transaction = None
            if foreign_symmetric_out:
                assert isinstance(out, SymmetricTensor)
                transaction = _snapshot_symmetric_out(out)
                out_for_np = out
            elif isinstance(out, SymmetricTensor):
                out_for_np = _symmetric_out_scratch(out)
            else:
                out_for_np = out
            try:
                result = _call_with_optional_out(
                    np_func,
                    x_fwd,
                    out=out_for_np,
                    supports_out=supports_out,
                    callback_op_name=op_name,
                    defer_out_write_tracking=foreign_symmetric_out,
                    **kwargs,
                )
            except BaseException:
                if transaction is not None:
                    base, before, *_ = transaction
                    if _logical_array_bytes(base) != before:
                        note_write(out)
                raise
        if is_ufunc and isinstance(result, _ForeignUfuncResult):
            if transaction is not None:
                assert isinstance(out, SymmetricTensor)
                base, before, saved_data, previous_symmetry, previous_inferred = (
                    transaction
                )
                _finish_foreign_symmetric_out(
                    out,
                    base=base,
                    before=before,
                    saved_data=saved_data,
                    target_symmetry=symmetry,
                    previous_symmetry=previous_symmetry,
                    previous_inferred=previous_inferred,
                )
            return result.value
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
    is_ufunc = isinstance(np_func, _np.ufunc)

    @_counted_wrapper
    def wrapper(
        x: ArrayLike, out: FlopscopeArray | None = None, **kwargs: Any
    ) -> FlopscopeArray:
        budget = require_budget()
        # Ufuncs must reject an opt-out before inspecting out or billing. The
        # non-ufunc NumPy dispatch helpers materialize such operands instead.
        if is_ufunc:
            _preflight_ufunc_out_arity(out, np_func.nout)
            _preflight_ufunc_opt_out(x, *_flatten_ufunc_out_slots(out))
        out = _normalize_out(out, op_name)
        symmetry = _symmetry_of(x)
        if is_ufunc:
            x, x_fwd = _resolve_ufunc_data_operand(x)
        else:
            if not isinstance(x, _np.ndarray):
                x = _np.asarray(x)
            x_fwd = x
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
                x_fwd,
                out=None if isinstance(out, SymmetricTensor) else out,
                supports_out=supports_out,
                callback_op_name=op_name,
                **kwargs,
            )
        if is_ufunc and isinstance(result, _ForeignUfuncResult):
            return result.value
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

    That floor is the ONLY thing that prices the exponent. A caller-supplied
    exponent buffer must not price it a second time: an ``int32`` destination
    is what numpy allocates for that slot anyway, so folding it into
    ``np.result_type`` alongside a float32 mantissa promoted the whole
    resolution to float64 and doubled the bill for naming a destination the
    op was always going to produce. ``multi_store_billing_dtypes`` keeps a
    slot out of the resolution unless it is genuinely pricier than numpy's
    own choice for that slot, so widening either output still widens the
    rate while the natural pair stays free.
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
        _preflight_ufunc_out_arity(out, nout)
        _preflight_ufunc_opt_out(x, *_flatten_ufunc_out_slots(out))
        out = _normalize_out(out, op_name, nout=nout)
        symmetry = _symmetry_of(x)
        x, x_fwd = _resolve_ufunc_data_operand(x)
        # A multi-output ufunc is priced as nout independent applications of
        # the reference algorithm (modf's fractional AND integral part;
        # frexp's mantissa AND exponent count as two one-output-unary
        # calls) -- the same "standard reference algorithm, no discount for
        # backend compute-sharing" rule behind FMA=2 and linalg.inv's 2n^3
        # (docs/reference/cost-model.md), NOT the write-metering doctrine
        # (arithmetic weight, not buffer traffic, dominates these ops).
        # Billing only one output priced modf/frexp at a flat single-output
        # rate (half their honest cost against a same-shape two-output
        # call). This is the cell-count (flop_cost) axis; it is applied
        # before billing_dtypes is resolved below and is therefore
        # orthogonal to the out= dtype-rate axis (see
        # multi_store_billing_dtypes) -- supplying out= does not multiply
        # this again.
        cost = nout * pointwise_cost(x.shape, symmetry=symmetry)
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
            # Only what a destination adds ON TOP of the buffer numpy would
            # have allocated for that slot: ``frexp``'s exponent is int32 by
            # signature, not because anything asked for 32-bit integer
            # arithmetic, so naming it must not promote the resolution.
            billing_dtypes += multi_store_billing_dtypes(
                out, natural_output_dtypes(np_func, x.dtype)
            )
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
                x_fwd,
                out=out,
                nout=nout,
                callback_op_name=op_name,
                **kwargs,
            )
        if isinstance(result, _ForeignUfuncResult):
            return result.value
        return _wrap_multi_result(result, out=out, symmetry=symmetry)  # type: ignore[return-value]

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(input) FLOPs")
    _apply_numpy_signature(wrapper, np_func)
    return wrapper


def _pointwise_complex_factor_override(
    loop_dtypes: tuple[_np.dtype, ...], out: object
) -> float | None:
    """Derive complex structure from the binary ufunc loop, not its rate floor.

    Any complex loop slot retains the operation's registry factor. A complex
    destination after a non-complex loop moves two real components, while a
    non-complex loop with no such destination is structurally neutral even if
    the promoted raw inputs raise its billing rate to a complex dtype.
    """
    if _builtins.any(dtype.kind == "c" for dtype in loop_dtypes):
        return None
    if isinstance(out, _np.ndarray) and out.dtype.kind == "c":
        return 2.0
    return 1.0


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
        _preflight_ufunc_out_arity(out, np_func.nout)
        _preflight_ufunc_opt_out(x, y, *_flatten_ufunc_out_slots(out))
        out = _normalize_out(out, op_name)
        # Resolve and freeze a caller-supplied dtype before observing either
        # operand. A dtype-like object's property can be stateful or mutate an
        # owning ndarray, so this one immutable value must govern both billing
        # snapshots and the real numpy call.
        explicit_dtype, explicit_signature, explicit_casting = (
            _freeze_binary_ufunc_type_kwargs(kwargs)
        )
        # Preserve original (possibly Python-scalar) values for the actual
        # numpy call so that NEP 50 weak-typing rules apply correctly. We
        # only need ndarray views for shape and symmetry inspection below.
        x_orig, y_orig = x, y
        x_sym = _symmetry_of(x_orig)
        y_sym = _symmetry_of(y_orig)
        x_view, x_fwd = _resolve_ufunc_data_operand(x_orig)
        y_view, y_fwd = _resolve_ufunc_data_operand(y_orig)
        x_view, x_fwd = _refresh_ufunc_data_operand(x_orig, (x_view, x_fwd))
        output_shape = _np.broadcast_shapes(x_view.shape, y_view.shape)
        x_is_scalar = x_view.ndim == 0
        y_is_scalar = y_view.ndim == 0
        if x_is_scalar ^ y_is_scalar:
            out_symmetry = y_sym if x_is_scalar else x_sym
            aligned_inputs = [out_symmetry] if out_symmetry is not None else []
        else:
            out_symmetry, aligned_inputs = _pointwise_symmetry(
                ((x_view, x_sym), (y_view, y_sym)),
                output_shape,
            )
        out_symmetry = _prepare_symmetric_out(out, out_symmetry)

        cost = pointwise_cost(output_shape, symmetry=out_symmetry)
        # dtype= constrains the output DType, while signature=/sig= can
        # constrain any loop slot. Resolve the complete constrained signature
        # so asymmetric participants (ldexp's integer exponent) and forced
        # complex loops remain visible to both the rate and factor models.
        loop_constraint = explicit_signature
        if loop_constraint is _UFUNC_SIGNATURE_MISSING and explicit_dtype is not None:
            loop_constraint = (
                *([None] * np_func.nin),
                *([explicit_dtype] * np_func.nout),
            )
        if loop_constraint is not _UFUNC_SIGNATURE_MISSING:
            loop_dtypes = _ufunc_loop_signature(
                np_func,
                ufunc_resolver_operand(x_orig, x_view),
                ufunc_resolver_operand(y_orig, y_view),
                signature=loop_constraint,
                casting=explicit_casting,
            )
            billing_dtypes = (heavier_billing_dtype(*loop_dtypes),)
        else:
            floor_operands = (
                billing_operand(x_orig, x_view),
                billing_operand(y_orig, y_view),
            )
            loop_dtypes = _ufunc_loop_signature(
                np_func,
                ufunc_resolver_operand(x_orig, x_view),
                ufunc_resolver_operand(y_orig, y_view),
                casting=explicit_casting,
            )
            loop_billing_dtype = heavier_billing_dtype(*loop_dtypes)
            input_floor = resolve_billing_dtype(floor_operands)
            billing_dtype = (
                loop_billing_dtype
                if input_floor is None
                else heavier_billing_dtype(loop_billing_dtype, input_floor)
            )
            billing_dtypes = (billing_dtype,)
        # Billing must inspect the destination's real ndarray descriptor,
        # never an overridable Python-level ``.dtype`` on a foreign subclass.
        # Derive this view only after every input dtype/loop participant has
        # resolved, since any of those caller-controlled reads can mutate an
        # owning destination. The original ``out`` remains authoritative for
        # symmetry, numpy forwarding, and return identity.
        out_view = (
            _np.asarray(_to_base_ndarray(out)) if isinstance(out, _np.ndarray) else None
        )
        if isinstance(out_view, _np.ndarray):
            billing_dtypes += store_billing_dtypes(out_view)
        with budget.deduct(
            op_name,
            flop_cost=cost,
            subscripts=None,
            shapes=(x_view.shape, y_view.shape),
            dtypes=billing_dtypes,
            complex_factor_override=_pointwise_complex_factor_override(
                loop_dtypes, out_view
            ),
        ):
            # Forward originals when their NumPy protocol semantics matter,
            # while retaining exact Python-scalar weak promotion (NEP 50).
            foreign_symmetric_out = isinstance(out, SymmetricTensor) and _builtins.any(
                _has_foreign_array_ufunc(value) for value in (x_fwd, y_fwd)
            )
            transaction = None
            if foreign_symmetric_out:
                assert isinstance(out, SymmetricTensor)
                transaction = _snapshot_symmetric_out(out)
                out_for_np = out
            elif isinstance(out, SymmetricTensor):
                out_for_np = _symmetric_out_scratch(out)
            else:
                out_for_np = out
            try:
                result = _call_with_optional_out(
                    np_func,
                    x_fwd,
                    y_fwd,
                    out=out_for_np,
                    supports_out=supports_out,
                    callback_op_name=op_name,
                    defer_out_write_tracking=foreign_symmetric_out,
                    **kwargs,
                )
            except BaseException:
                if transaction is not None:
                    base, before, *_ = transaction
                    if _logical_array_bytes(base) != before:
                        note_write(out)
                raise
        if isinstance(result, _ForeignUfuncResult):
            if transaction is not None:
                assert isinstance(out, SymmetricTensor)
                base, before, saved_data, previous_symmetry, previous_inferred = (
                    transaction
                )
                _finish_foreign_symmetric_out(
                    out,
                    base=base,
                    before=before,
                    saved_data=saved_data,
                    target_symmetry=out_symmetry,
                    previous_symmetry=previous_symmetry,
                    previous_inferred=previous_inferred,
                )
            return result.value
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
        _preflight_ufunc_out_arity(out, nout)
        _preflight_ufunc_opt_out(x, y, *_flatten_ufunc_out_slots(out))
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
        # divmod is priced as running floor_divide and mod separately --
        # nout=2 applications of the reference algorithm, not one -- per
        # the same standard-algorithm/no-compute-sharing-discount rule (see
        # the matching note in _counted_unary_multi and
        # docs/reference/cost-model.md). NumPy's own divmod shares the
        # division and derives the remainder almost for free (measured
        # ~1x floor_divide, not ~2x), but flopscope prices the reference
        # algorithm, not this backend's implementation. Independent of the
        # out= dtype-rate axis.
        cost = nout * pointwise_cost(output_shape, symmetry=out_symmetry)
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
            # Resolve the operands FIRST so a NEP 50 weak scalar cannot make
            # the natural destination look wider than the loop really is --
            # see ``natural_output_dtypes``.
            billing_dtypes += multi_store_billing_dtypes(
                out,
                natural_output_dtypes(np_func, resolve_billing_dtype(billing_dtypes)),
            )
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
                callback_op_name=op_name,
                **kwargs,
            )
        if isinstance(result, _ForeignUfuncResult):
            return result.value
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


_UFUNC_SIGNATURE_MISSING = object()
_UFUNC_CASTING_MISSING = object()


def _ufunc_loop_signature(
    ufunc,
    *operand_dtypes: _np.dtype | type,
    signature: object = _UFUNC_SIGNATURE_MISSING,
    casting: object = _UFUNC_CASTING_MISSING,
) -> tuple[_np.dtype, ...]:
    """The complete input/output dtype signature NumPy resolves for ``ufunc``.

    The returned tuple includes every resolved input-loop slot followed by
    every output slot. Direct binary and ``outer`` billing inspect all slots:
    predicates may return bool while still reading complex inputs. Reduction/
    accumulator consumers use the first output slot through
    :func:`_ufunc_loop_dtype`.

    The input tuple retains the prior operand-repeat rule: a missing second
    operand of a binary ufunc repeats the first, matching reduce/accumulate/
    reduceat. Bare Python ``int``/``float``/``complex`` types remain valid
    NEP 50 weak-scalar inputs to ``resolve_dtypes``. If NumPy refuses loop
    resolution, every slot falls back to the same ``np.result_type`` used by
    the previous output-only helper, preserving its conservative behavior.
    """
    first, *rest = operand_dtypes
    if ufunc.nin == 1:
        inputs: tuple = (first,)
    else:
        inputs = (first, rest[0] if rest else first)
    resolve_kwargs = {}
    if signature is not _UFUNC_SIGNATURE_MISSING:
        resolve_kwargs["signature"] = signature
    if casting is not _UFUNC_CASTING_MISSING:
        resolve_kwargs["casting"] = casting
    try:
        if not resolve_kwargs:
            return tuple(ufunc.resolve_dtypes((*inputs, *([None] * ufunc.nout))))
        with _warnings.catch_warnings():
            # NumPy <=2.2 warns for legacy length-one signature forms. The
            # real call below remains responsible for emitting that warning
            # once; this internal mirror must not duplicate it.
            _warnings.simplefilter("ignore", DeprecationWarning)
            return tuple(
                ufunc.resolve_dtypes(
                    (*inputs, *([None] * ufunc.nout)), **resolve_kwargs
                )
            )
    except (TypeError, ValueError):
        if signature is not _UFUNC_SIGNATURE_MISSING:
            raise
        fallback = _np.result_type(*operand_dtypes)
        return (fallback,) * (len(inputs) + ufunc.nout)


def _ufunc_loop_dtype(ufunc, *operand_dtypes: _np.dtype | type) -> _np.dtype:
    """Return the first output/accumulator slot of NumPy's resolved loop."""
    return _ufunc_loop_signature(ufunc, *operand_dtypes)[ufunc.nin]


def _resolve_explicit_dtype_kwarg(kwargs: dict) -> _np.dtype | None:
    """Resolve a ``dtype=`` kwarg to a concrete ``np.dtype`` exactly once.

    Shared by direct single-output binary wrappers and the four generic
    ufunc-method paths below (``outer`` / ``reduce`` / ``accumulate`` /
    ``reduceat``). ``kwargs["dtype"]`` (when present) is read here to compute
    the billing dtype, AND is the same object later forwarded to the real
    numpy call via ``**kwargs`` -- a caller-supplied dtype-like object (e.g.
    one exposing a stateful ``.dtype`` property, which ``np.dtype()``
    honours) would otherwise get independently re-resolved a second time
    inside numpy's own call, letting it report a cheap dtype to us while
    handing numpy something pricier. Overwriting ``kwargs["dtype"]`` in place
    with the resolved, immutable ``np.dtype`` closes that gap: numpy then
    resolves the exact same value we billed.

    ``dtype=None`` (the default, meaning "no explicit dtype") is left
    alone -- resolving it here would turn "use the family default" into an
    explicit float64 request.
    """
    explicit = kwargs.get("dtype")
    if explicit is None:
        return None
    resolved = _np.dtype(explicit)
    kwargs["dtype"] = resolved
    return resolved


def _freeze_binary_ufunc_type_kwargs(
    kwargs: dict,
) -> tuple[_np.dtype | None, object, object]:
    """Freeze dtype/signature constraints before binary operand snapshots.

    NumPy treats ``sig`` as an alias of ``signature`` and rejects conflicts
    based on keyword presence, even when a value is ``None``. A tuple
    signature may contain caller-controlled dtype-like objects; resolve each
    such slot once and forward the frozen tuple to NumPy so billing and
    execution cannot observe different values.
    """
    has_signature = "signature" in kwargs
    has_sig = "sig" in kwargs
    if has_signature and has_sig:
        raise TypeError("cannot specify both 'sig' and 'signature'")
    signature_key = "signature" if has_signature else "sig" if has_sig else None
    if signature_key is not None and "dtype" in kwargs:
        raise TypeError("cannot specify both 'signature' and 'dtype'")

    explicit_dtype = _resolve_explicit_dtype_kwarg(kwargs)
    casting: object = kwargs.get("casting", _UFUNC_CASTING_MISSING)
    if isinstance(casting, str):
        casting = str.__str__(casting)
        kwargs["casting"] = casting
    if signature_key is None:
        return explicit_dtype, _UFUNC_SIGNATURE_MISSING, casting

    signature = kwargs[signature_key]
    if isinstance(signature, str):
        frozen_signature: object = str.__str__(signature)
    elif isinstance(signature, tuple):
        slots = (
            tuple.__getitem__(signature, index)
            for index in range(tuple.__len__(signature))
        )
        frozen_signature = tuple(
            slot
            if slot is None or (isinstance(slot, type) and issubclass(slot, _np.dtype))
            else _np.dtype(slot)
            for slot in slots
        )
    else:
        raise TypeError("the signature object to ufunc must be a string or a tuple")
    kwargs[signature_key] = frozen_signature
    return explicit_dtype, frozen_signature, casting


def _implements_array_ufunc(x) -> bool:
    """Does ``type(x)`` define NumPy's ufunc-dispatch protocol (NEP 13)?

    Checked on the TYPE, matching how numpy itself looks this up for
    protocol dispatch (an instance attribute would not count). The explicit
    ``__array_ufunc__ = None`` opt-out spelling is excluded: callers that
    resolve data operands reject it before this predicate can consider a
    protocol implementation, matching NumPy's immediate ``TypeError``.
    """
    implementation = _static_array_ufunc_implementation(x)
    return implementation is not _ARRAY_UFUNC_MISSING and implementation is not None


def _raise_ufunc_opt_out(x) -> None:
    """Raise NumPy's error for a type declaring ``__array_ufunc__ = None``."""
    type_name = type.__getattribute__(type(x), "__name__")
    raise TypeError(
        f"operand '{type_name}' does not support ufuncs (__array_ufunc__=None)"
    )


def _preflight_ufunc_opt_out(*operands) -> None:
    """Reject top-level ufunc operands that explicitly opt out of NEP 13.

    NumPy raises before it materializes any operand or executes a ufunc loop
    when a participating type defines ``__array_ufunc__ = None``.  Callers
    pass data operands directly and flatten only real ``out=`` slots;
    ``where=`` and unrelated kwargs or index sequences are deliberately
    excluded.
    """
    for operand in operands:
        if _static_array_ufunc_implementation(operand) is None:
            _raise_ufunc_opt_out(operand)


def _resolve_ufunc_data_operand(x):
    """Bill-and-forward pair for a ufunc-method data operand.

    Used by direct binary wrappers and :func:`_counted_ufunc_outer` for
    ``a``/``b``/``out``, and by
    :func:`_counted_ufunc_reduce_generic` / :func:`_counted_ufunc_accumulate_generic`
    / :func:`_counted_ufunc_reduceat` for ``a`` -- every generic ufunc-method
    path whose ``a``/``b`` position can legitimately carry an operand that is
    not already known to be an owned, freshly-made array. (``at``'s
    ``values`` keeps its own copy, ``_resolve_at_operand``, because it
    additionally must not force a bare Python scalar through ``_np.asarray``
    for the *billing* read the way this helper does -- see that function's
    docstring -- but mirrors this same three-way rule.)

    Four operand kinds, four different requirements:

    - An explicit type-level ``__array_ufunc__ = None`` opt-out: reject it
      immediately, before either billing or ``__array__`` materialization,
      because NumPy does exactly that for every ufunc operand.

    - Already an ``ndarray`` (including a flopscope subclass, or a foreign
      one like ``np.ma.MaskedArray``): forward the ORIGINAL object to numpy
      so its subclass semantics (e.g. a mask) survive execution. Bill from a
      SEPARATE ``_np.asarray`` view -- stripping any flopscope wrapper
      first, to avoid recursing back through our own protocol handlers -- a
      no-op VIEW, same buffer, for an operand that is already a genuine
      plain ndarray, so an overridden ``.dtype``/``.shape`` property cannot
      misreport what numpy actually computes. Reading ``.shape``/``.dtype``
      off an ndarray never invokes ``__array__``, so there is no
      double-resolution risk to guard against here.
    - NOT an ndarray, but its type implements ``__array_ufunc__``
      (:func:`_implements_array_ufunc`): forward the ORIGINAL object.
      NumPy's own dispatch (NEP 13) hands a ufunc call to THAT protocol
      instead of independently reconverting the operand through
      ``__array__`` -- confirmed directly: a duck object exposing both a
      stateful ``__array__`` and an ``__array_ufunc__`` that returns
      ``NotImplemented`` sees its ``__array_ufunc__`` invoked by the real
      call below, and its ``__array__`` is never touched by that call at
      all. So forwarding the original here does NOT reopen the
      stateful-``__array__`` double-materialization hole the next branch
      exists to close -- there is no second, independent resolution for a
      stateful callback to race. Materializing this operand instead (the
      prior, two-way behaviour) would bypass the protocol entirely and let
      flopscope silently compute from ``__array__`` where plain numpy would
      have dispatched to ``__array_ufunc__`` (and very possibly raised).
      Billing still needs a shape/dtype to size the cost, so it reads them
      off a single ``_np.asarray(x)`` view -- ONE call, same accounting as
      the ndarray branch above, and never a second one, since the real call
      below never touches ``__array__`` on this object.
    - Anything else (a duck type implementing only ``__array__`, or a
      list/tuple/other sequence):
      materialize it EXACTLY ONCE via ``_np.asarray``, and use that SAME
      materialized array as both the billing view and the forwarded object.
      There is no legitimate subclass semantics to lose here -- it was never
      an ndarray subclass -- and using one materialization for both closes a
      hole where a stateful ``__array__`` (one that returns a bigger array
      on its second call than its first) would otherwise be billed against
      the small first result while numpy, receiving the ORIGINAL unresolved
      object, independently re-resolved it a second time and executed the
      larger one.

    A bare Python scalar (``bool``/``int``/``float``/``complex``, not a
    numpy scalar) is a special case of the third bullet: it carries no
    mutable state for ``_np.asarray`` to observe differently across two
    reads, but forwarding the MATERIALIZED 0-d array instead of the scalar
    itself would turn a NEP 50 weak scalar into a strong-typed operand and
    change numpy's own promotion result against the other side. So the
    scalar is forwarded unchanged; only the billing view is materialized.

    Returns ``(billing_view, forward_obj)``.
    """
    implementation = _static_array_ufunc_implementation(x)
    if implementation is None:
        _raise_ufunc_opt_out(x)
    if isinstance(x, _np.ndarray):
        stripped = _to_base_ndarray(x)
        return _np.asarray(stripped), stripped
    if implementation is not _ARRAY_UFUNC_MISSING:
        return _np.asarray(x), x
    if isinstance(x, (bool, int, float, complex)) and not isinstance(x, _np.generic):
        return _np.asarray(x), x
    resolved = _np.asarray(x)
    return resolved, resolved


def _refresh_ufunc_data_operand(x, cached: tuple):
    """Re-read the ``(billing_view, forward_obj)`` pair for ``x`` after a
    LATER resolution step that might have mutated it in place.

    Two callers, both for the same underlying reason -- a structurally
    necessary first read of ``x`` has to happen before some OTHER
    participant-controlled argument can be resolved, and resolving that
    other argument can mutate ``x`` in place:

    - :func:`_counted_ufunc_reduceat`: ``axis`` resolution there
      (``_resolve_reduceat_axis``) genuinely needs ``x``'s ndim before it
      can even decide which accepted ``axis`` form applies (unlike
      :func:`_resolve_generic_reduce_axis`, whose reduce/accumulate
      siblings need no such read and so can resolve ``axis`` before
      touching ``a`` at all -- see its docstring). That first read of
      ``x`` happens BEFORE axis is resolved, and resolving axis runs
      participant code (``operator.index``) that -- per the confirmed
      reproduction this whole module's reordering guards against -- can
      mutate ``x`` in place via an owning ndarray subclass's
      ``resize(n, refcheck=False)``.
    - :func:`_counted_ufunc_outer`: ``a``'s billing view has to be read
      before ``b`` can even be resolved (the wrapper takes two positional
      data operands, not one), and resolving ``b`` runs the same kind of
      participant code (an ``__array__`` call) that can mutate ``a`` the
      same way, from inside the caller's own closure for ``b``.

    This function re-derives the billing view afterwards so cost is
    computed from what numpy will actually touch, not from the
    pre-mutation snapshot.

    Only an already-``ndarray`` operand can have been mutated this way in
    the first place -- ``resize`` is an ``ndarray`` method, so a
    non-ndarray duck has nothing for it to exploit. Re-deriving that
    operand's billing view is a second ``_np.asarray`` VIEW read, never a
    second ``__array__``/``__array_ufunc__`` call (see
    :func:`_resolve_ufunc_data_operand`'s ndarray branch), so refreshing it
    costs nothing extra and cannot double-materialize a stateful duck. A
    non-ndarray operand is returned via ``cached`` UNCHANGED -- reusing the
    honest, single-materialization result from the first call rather than
    touching its protocol a second time, which is exactly the guarantee
    ``_resolve_ufunc_data_operand`` exists to give a stateful ``__array__``
    duck.
    """
    if isinstance(x, _np.ndarray):
        return _resolve_ufunc_data_operand(x)
    return cached


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
    _preflight_ufunc_out_arity(out, ufunc.nout)
    _preflight_ufunc_opt_out(a, b, *_flatten_ufunc_out_slots(out))
    out = _normalize_out(out, f"{ufunc.__name__}.outer", nout=ufunc.nout)
    # Resolved here, BEFORE ``a``/``b`` are read for billing below, not
    # down where it is USED. ``np.dtype()`` accepts any object exposing a
    # ``.dtype`` attribute, and that attribute can be a Python property --
    # one that, as a side effect of returning a valid dtype, mutates ``a``
    # or ``b`` in place (e.g. an owning ndarray subclass calling
    # ``resize(n, refcheck=False)``, the confirmed reproduction this
    # module's reordering guards against). Resolving it before either
    # operand's billing view exists closes that window structurally: there
    # is no stale pre-mutation snapshot left for it to race against.
    explicit_dtype, explicit_signature, explicit_casting = (
        _freeze_binary_ufunc_type_kwargs(kwargs)
    )
    # Symmetry tags live on the ORIGINAL ``SymmetricTensor`` instances --
    # read them BEFORE the stripping below, which (deliberately) discards
    # that subclass along with everything else.
    a_sym = _symmetry_of(a)
    b_sym = _symmetry_of(b)
    # ``outer`` is a binary op: numpy dispatches here as soon as EITHER
    # operand is flopscope-aware, so the OTHER operand can be an arbitrary
    # caller-supplied ndarray subclass -- one overriding ``.dtype`` as a
    # Python property is not a hypothetical: numpy's own ufunc dispatch
    # reads the true underlying descriptor at the C level regardless of
    # what a subclass's ``.dtype`` reports, so a bare ``isinstance(a,
    # ndarray)`` check (which leaves an already-ndarray operand
    # untouched) would let that property lie to the billing dtype read
    # below while numpy executes at the real, unspoofed rate. The OTHER
    # operand can just as easily be a non-ndarray array-like implementing
    # only ``__array__`` -- and that ``__array__`` can be STATEFUL, e.g.
    # returning a small array on its first call (what gets billed) and a
    # much larger one on a second, independent call (what numpy would
    # execute) if the original, unresolved object were forwarded to numpy
    # below while a separately-materialized copy was billed here.
    # ``_resolve_ufunc_data_operand`` closes both gaps -- plus a third --
    # in one place: for an already-ndarray operand it bills from a stripped
    # ``_np.asarray`` view (immune to a lying property) while forwarding the
    # caller's ORIGINAL object below, so a legitimate foreign subclass
    # (``np.ma.MaskedArray``, a units array, anything with meaningful
    # ``__array_wrap__`` behaviour) still reaches numpy and keeps its
    # semantics (e.g. the mask); for a non-ndarray operand whose TYPE
    # implements ``__array_ufunc__`` it likewise forwards the ORIGINAL, so
    # numpy's own dispatch protocol decides what happens (and can raise
    # exactly as it would without flopscope in the loop) instead of
    # flopscope silently computing from ``__array__`` on its behalf; for
    # anything else it materializes via ``_np.asarray`` EXACTLY ONCE and
    # bills from AND forwards that same materialized array, so a stateful
    # ``__array__`` only ever gets called once total.
    a_view, a_fwd = _resolve_ufunc_data_operand(a)
    b_view, b_fwd = _resolve_ufunc_data_operand(b)
    # ``b``'s resolution just above runs participant code (an ``__array__``
    # call, for a duck ``b``) that -- per the confirmed reproduction this
    # whole module's reordering guards against -- can mutate ``a`` in place
    # via an owning ndarray subclass's ``resize(n, refcheck=False)``, called
    # from inside the very closure the caller constructed for ``b``.
    # ``a_view``, captured ABOVE this line (before ``b`` was ever touched),
    # would otherwise be billed from a stale, pre-mutation snapshot while
    # the real call below forwards ``a_fwd`` -- the SAME (already-mutated)
    # object, not a copy -- to numpy. Refresh it now that ``b`` has been
    # read, mirroring the treatment ``_counted_ufunc_reduceat`` gives ``a``
    # after ``axis`` resolves. See ``_refresh_ufunc_data_operand`` for why
    # this cannot double-invoke a stateful ``__array__`` duck: only an
    # already-``ndarray`` ``a`` can have been mutated this way in the first
    # place, and re-deriving ITS billing view is a second, free
    # ``_np.asarray`` VIEW read, never a second protocol call. A non-ndarray
    # ``a`` is returned via the cache unchanged.
    a_view, a_fwd = _refresh_ufunc_data_operand(a, (a_view, a_fwd))
    output_shape = tuple(a_view.shape) + tuple(b_view.shape)
    dense = _builtins.max(a_view.size * b_view.size, 1)
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
            axis_map = {ax: ax + a_view.ndim for ax in b_sym.axes}
            b_sym_lifted = remap_group_axes(b_sym, axis_map)
        out_sym = direct_product_groups(a_sym, b_sym_lifted)
        cost = _symmetry_adjusted_cost(dense, output_shape, out_sym)
    # Same lying-subclass exposure as ``a``/``b`` above, but for ``out=``:
    # its dtype participates in the billing rate (``store_billing_dtypes``
    # below) and must be read off the real buffer, not a Python-level
    # override. ``_normalize_out`` above already restricts ``out`` to
    # ``None`` or a genuine ``ndarray`` (never an arbitrary array-like), so
    # only the first branch of ``_resolve_ufunc_data_operand`` can ever
    # fire here; ``out_view`` is billing-only -- the real call further down
    # forwards ``out_fwd`` (the caller's original object) instead.
    out_view, out_fwd = (
        _resolve_ufunc_data_operand(out) if out is not None else (None, None)
    )
    loop_constraint = explicit_signature
    if loop_constraint is _UFUNC_SIGNATURE_MISSING and explicit_dtype is not None:
        loop_constraint = (
            *([None] * ufunc.nin),
            *([explicit_dtype] * ufunc.nout),
        )
    if loop_constraint is not _UFUNC_SIGNATURE_MISSING:
        loop_dtypes = _ufunc_loop_signature(
            ufunc,
            a_view.dtype,
            b_view.dtype,
            signature=loop_constraint,
            casting=explicit_casting,
        )
        billing_dtypes: tuple = (heavier_billing_dtype(*loop_dtypes),)
    else:
        # This default path shares the operand-width behavior of the
        # reduce/accumulate/reduceat/at siblings: a comparison/logical
        # ufunc's loop OUTPUT is bool, which for wide-int inputs would bill
        # NARROWER than the input -- never charge below it.
        # The full resolved signature keeps NumPy's actual loop kind first on
        # rate ties (for example, complex inputs for a complex predicate
        # loop). The jointly promoted input dtype is only a rate floor, so it
        # can raise the rate but cannot replace that loop kind on a tie.
        loop_dtypes = _ufunc_loop_signature(
            ufunc, a_view.dtype, b_view.dtype, casting=explicit_casting
        )
        loop_billing_dtype = heavier_billing_dtype(*loop_dtypes)
        input_floor = resolve_billing_dtype((a_view.dtype, b_view.dtype))
        billing_dtype = (
            loop_billing_dtype
            if input_floor is None
            else heavier_billing_dtype(loop_billing_dtype, input_floor)
        )
        billing_dtypes = (billing_dtype,)
    if isinstance(out_view, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out_view)
    with budget.deduct(
        f"{ufunc.__name__}.outer",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_view.shape, b_view.shape),
        dtypes=billing_dtypes,
        complex_factor_override=_pointwise_complex_factor_override(
            loop_dtypes, out_view
        ),
    ):
        result = _restore_foreign_ufunc_out_identity(
            _call_ufunc_with_protocol_timing(
                f"{ufunc.__name__}.outer",
                ufunc.outer,
                a_fwd,
                b_fwd,
                out=out_fwd,
                protocol_operands=(a_fwd, b_fwd, out_fwd),
                **kwargs,
            ),
            out,
            out_fwd,
        )
    if isinstance(result, _ForeignUfuncResult):
        return result.value
    return _wrap_result(result, out=out, symmetry=out_sym)


def _resolve_generic_reduce_axis(axis) -> int | tuple[int, ...] | None:
    """Resolve ``axis`` for the generic ``ufunc.reduce``/``ufunc.accumulate``
    fallback paths -- the SOLE resolution of ``axis``, authoritative for
    both the bill and the real call below.

    ``None`` passes through unchanged (numpy's own "reduce every axis" /
    "accumulate does not allow this" semantics apply downstream, in the
    real call). A bare axis or each element of a TUPLE of axes is read
    through ``operator.index`` EXACTLY ONCE and returned as a plain
    ``int`` -- ``bool``/``np.bool_`` are rejected first, matching numpy's
    own ``TypeError: an integer is required`` (``bool`` implements
    ``__index__`` but numpy's axis parser special-cases it out).

    A LIST is deliberately not given the tuple's element-by-element
    treatment: real ``ufunc.reduce``/``ufunc.accumulate`` accept only a
    bare integer or a tuple of integers for ``axis`` and reject a list
    outright (``TypeError: 'list' object cannot be interpreted as an
    integer``) -- regardless of what it contains, even a single in-range
    int. A list falls through to the same ``operator.index`` call the
    bare-axis case uses below, which raises that exact message for a
    list (lists have no ``__index__``) without ever inspecting its
    elements, matching numpy's own accept/reject boundary instead of
    silently normalizing a form numpy itself refuses.

    This function does not itself replicate every numpy-side semantic
    rule (duplicate axes, an ufunc that is not "reorderable" restricting
    reduce to a single axis, accumulate rejecting more than one axis,
    range checks) -- those still run natively, downstream, inside numpy's
    own call, exactly as before. What changes is that they now run
    against the PLAIN, already-resolved value returned here rather than
    the caller's original object: a caller-supplied axis exposing
    ``__index__`` (or, for a tuple, each element's) that behaves
    differently across two invocations -- succeeding with one axis on a
    first read and a different one on a second -- could previously be
    billed for the first read while flopscope's own cost math forwarded
    the ORIGINAL object to numpy, which resolved it again independently
    and executed along whatever the second read produced.

    Deliberately takes no ``ndim`` -- unlike ``_resolve_reduceat_axis``,
    this resolver never range-checks against it (that's the "range checks
    ... run natively, downstream" deferral described above), so it has no
    structural need to read anything off ``a`` before resolving ``axis``.
    That makes it callable BEFORE ``a`` is touched at all: a caller-
    supplied ``axis`` exposing ``__index__`` that mutates ``a`` in place
    (e.g. an owning ndarray subclass calling ``a.resize(n,
    refcheck=False)``) can only run here, before ``a``'s billing view is
    ever read, so there is no stale-shape snapshot for it to race against.
    """
    if axis is None:
        return None
    if isinstance(axis, tuple):
        resolved = []
        for entry in axis:
            if isinstance(entry, (bool, _np.bool_)):
                raise TypeError("an integer is required")
            resolved.append(_operator.index(entry))
        return tuple(resolved)
    if isinstance(axis, (bool, _np.bool_)):
        raise TypeError("an integer is required")
    return _operator.index(axis)


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
    _preflight_ufunc_out_arity(out, ufunc.nout)
    _preflight_ufunc_opt_out(a, *_flatten_ufunc_out_slots(out))
    out = _normalize_out(out, f"{ufunc.__name__}.reduce", nout=ufunc.nout)
    # Resolved here, BEFORE ``a`` is read for billing below -- see
    # ``_resolve_generic_reduce_axis``'s docstring for why it can run this
    # early (it never needs anything off ``a``). A caller-supplied ``axis``
    # exposing a stateful ``__index__`` can only mutate ``a`` here, before
    # any billing view of ``a`` exists, closing the confirmed
    # stale-billing-snapshot reproduction structurally rather than via a
    # refresh.
    axis = _resolve_generic_reduce_axis(axis)
    # Same reasoning as ``axis`` above, and for the same reason: ``dtype=``
    # is participant-controlled (``np.dtype()`` honours an arbitrary
    # object's ``.dtype`` PROPERTY, which can mutate ``a`` as a side
    # effect), so it is resolved before ``a``'s billing view exists too.
    explicit_dtype = _resolve_explicit_dtype_kwarg(kwargs)
    # Symmetry tags live on the ORIGINAL flopscope instance -- read them
    # BEFORE ``_resolve_ufunc_data_operand``'s stripping, which discards that
    # subclass along with everything else.
    sym = _symmetry_of(a)
    # ``a`` can already be a foreign ndarray subclass, a non-ndarray duck
    # array, or a bare sequence here: numpy dispatches to this fallback as
    # soon as EITHER ``a`` or ``out`` is flopscope-aware (e.g.
    # ``ufunc.reduce(foreign_array, out=fnp.zeros(...))``), so a bare
    # ``isinstance(a, ndarray)`` check would leave a non-flopscope operand
    # untouched. ``_resolve_ufunc_data_operand`` is what makes every one of
    # those cases safe: an ndarray subclass overriding
    # ``.dtype``/``.shape``/``.ndim`` as Python properties cannot misreport
    # the billing view (a genuine ``_np.asarray`` read -- numpy's own ufunc
    # dispatch reads the true descriptor at the C level regardless of what
    # those properties report, see ``_counted_ufunc_outer``), a legitimate
    # foreign subclass (e.g. ``np.ma.MaskedArray``) still reaches numpy below
    # with its semantics intact, and a non-ndarray operand whose type
    # implements ``__array_ufunc__`` is forwarded ORIGINAL rather than
    # materialized, so numpy's own dispatch protocol decides what happens
    # instead of flopscope silently consuming ``__array__`` on its behalf.
    # This is the ONLY read of ``a``'s billing view -- it happens LAST,
    # after ``axis`` and ``dtype=`` have both already been resolved above,
    # so it reflects whatever either of them may have mutated ``a`` into.
    a_view, a_fwd = _resolve_ufunc_data_operand(a)
    cost = reduction_cost(a_view.shape, axis=axis, symmetry=sym)
    out_sym = (
        reduce_group(sym, ndim=a_view.ndim, axis=axis, keepdims=keepdims)
        if sym is not None
        else None
    )
    # Same lying-subclass exposure as ``a`` above, but for ``out=``: its
    # dtype participates in the billing floor (``out_dtype=`` below) and
    # must be read off the real buffer, not a Python-level override. Only a
    # bare ndarray gets this treatment -- a tuple ``out`` (a multi-output
    # ufunc form ``.reduce`` never actually supports) is left as
    # ``_to_base_ndarray`` already handled it, matching prior behavior.
    # ``out_view`` is billing-only; the real call further down forwards the
    # caller's original ``out`` object instead.
    out_view = _to_base_ndarray(out) if out is not None else None
    if isinstance(out_view, _np.ndarray):
        out_view = _np.asarray(out_view)
    # The reduce/accumulate loop runs at the ufunc's own resolved loop dtype
    # (true_divide(int32) -> float64, subtract(int32) -> int32, logical_* ->
    # bool). add/multiply's extra integer widening never matters here: they
    # are routed to sum/prod, not this generic path.
    default_dtype = _ufunc_loop_dtype(ufunc, a_view.dtype, a_view.dtype)
    billing_dtypes: tuple = (
        reduction_billing_dtype(
            a_view.dtype,
            explicit_dtype=explicit_dtype,
            out_dtype=out_view.dtype if isinstance(out_view, _np.ndarray) else None,
            default_dtype=default_dtype,
        ),
    )
    with budget.deduct(
        f"{ufunc.__name__}.reduce",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_view.shape,),
        dtypes=billing_dtypes,
    ):
        out_fwd = _to_base_ndarray(out) if out is not None else None
        result = _restore_foreign_ufunc_out_identity(
            _call_ufunc_with_protocol_timing(
                f"{ufunc.__name__}.reduce",
                ufunc.reduce,
                a_fwd,
                axis=axis,
                out=out_fwd,
                keepdims=keepdims,
                protocol_operands=(a_fwd, out),
                **kwargs,
            ),
            out,
            out_fwd,
        )
    if isinstance(result, _ForeignUfuncResult):
        return result.value
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
    _preflight_ufunc_out_arity(out, ufunc.nout)
    _preflight_ufunc_opt_out(a, *_flatten_ufunc_out_slots(out))
    out = _normalize_out(out, f"{ufunc.__name__}.accumulate", nout=ufunc.nout)
    # Resolved here, BEFORE ``a`` is read for billing below -- see
    # ``_resolve_generic_reduce_axis``'s docstring for why it can run this
    # early (it never needs anything off ``a``). A caller-supplied ``axis``
    # exposing a stateful ``__index__`` can only mutate ``a`` here, before
    # any billing view of ``a`` exists, closing the confirmed
    # stale-billing-snapshot reproduction structurally rather than via a
    # refresh.
    axis = _resolve_generic_reduce_axis(axis)
    # Same reasoning as ``axis`` above, and for the same reason: ``dtype=``
    # is participant-controlled (``np.dtype()`` honours an arbitrary
    # object's ``.dtype`` PROPERTY, which can mutate ``a`` as a side
    # effect), so it is resolved before ``a``'s billing view exists too.
    explicit_dtype = _resolve_explicit_dtype_kwarg(kwargs)
    # Symmetry tags live on the ORIGINAL flopscope instance -- read them
    # BEFORE ``_resolve_ufunc_data_operand``'s stripping, which discards that
    # subclass along with everything else.
    sym = _symmetry_of(a)
    # ``a`` can already be a foreign ndarray subclass, a non-ndarray duck
    # array, or a bare sequence here: numpy dispatches to this fallback as
    # soon as EITHER ``a`` or ``out`` is flopscope-aware, so a bare
    # ``isinstance(a, ndarray)`` check would leave a non-flopscope operand
    # untouched. ``_resolve_ufunc_data_operand`` is what makes every one of
    # those cases safe: an ndarray subclass overriding
    # ``.dtype``/``.shape``/``.ndim`` as Python properties cannot misreport
    # the billing view (a genuine ``_np.asarray`` read -- numpy's own ufunc
    # dispatch reads the true descriptor at the C level regardless of what
    # those properties report, see ``_counted_ufunc_outer``), a legitimate
    # foreign subclass (e.g. ``np.ma.MaskedArray``) still reaches numpy below
    # with its semantics intact, and a non-ndarray operand whose type
    # implements ``__array_ufunc__`` is forwarded ORIGINAL rather than
    # materialized, so numpy's own dispatch protocol decides what happens
    # instead of flopscope silently consuming ``__array__`` on its behalf.
    # This is the ONLY read of ``a``'s billing view -- it happens LAST,
    # after ``axis`` and ``dtype=`` have both already been resolved above,
    # so it reflects whatever either of them may have mutated ``a`` into.
    a_view, a_fwd = _resolve_ufunc_data_operand(a)
    cost = reduction_cost(a_view.shape, axis=axis, symmetry=sym)
    out_sym = (
        reduce_group(sym, ndim=a_view.ndim, axis=axis, keepdims=True)
        if sym is not None
        else None
    )
    # Same lying-subclass exposure as ``a`` above, but for ``out=``: its
    # dtype participates in the billing floor (``out_dtype=`` below) and
    # must be read off the real buffer, not a Python-level override.
    # ``out_view`` is billing-only; the real call further down forwards the
    # caller's original ``out`` object instead.
    out_view = _to_base_ndarray(out) if out is not None else None
    if isinstance(out_view, _np.ndarray):
        out_view = _np.asarray(out_view)
    # Same loop resolution as the generic reduce path above: the accumulate
    # loop runs at the ufunc's own resolved loop dtype (true_divide(int32)
    # -> float64, subtract(int32) -> int32).
    default_dtype = _ufunc_loop_dtype(ufunc, a_view.dtype, a_view.dtype)
    billing_dtypes: tuple = (
        reduction_billing_dtype(
            a_view.dtype,
            explicit_dtype=explicit_dtype,
            out_dtype=out_view.dtype if isinstance(out_view, _np.ndarray) else None,
            default_dtype=default_dtype,
        ),
    )
    with budget.deduct(
        f"{ufunc.__name__}.accumulate",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_view.shape,),
        dtypes=billing_dtypes,
    ):
        out_fwd = _to_base_ndarray(out) if out is not None else None
        result = _restore_foreign_ufunc_out_identity(
            _call_ufunc_with_protocol_timing(
                f"{ufunc.__name__}.accumulate",
                ufunc.accumulate,
                a_fwd,
                axis=axis,
                out=out_fwd,
                protocol_operands=(a_fwd, out),
                **kwargs,
            ),
            out,
            out_fwd,
        )
    if isinstance(result, _ForeignUfuncResult):
        return result.value
    return _wrap_result(result, out=out, symmetry=out_sym)


def _resolve_reduceat_axis(axis, ndim: int) -> int:
    """Resolve ``axis`` to the concrete axis ``ufunc.reduceat`` will reduce
    along -- the SOLE resolution of ``axis``, authoritative for both the
    bill and the real call below.

    A prior version of this function returned ``None`` for any axis it
    could not pin down, floored the cost accordingly, and let the caller
    forward the ORIGINAL ``axis`` object to numpy so numpy could resolve it
    a second time and raise its own error. That is exactly the gap this
    rewrite closes: an ``axis`` exposing ``__index__`` (or ``__len__``, for
    the tuple-length check) that behaves differently across two calls --
    raising or reporting an out-of-range value on the first, succeeding
    with an in-range one on the second -- would be billed at the floor
    while numpy silently executed the real, second-call axis. This
    function now either returns a concrete, valid axis or raises directly
    -- callers must never fall back to re-resolving ``axis`` themselves.

    Real ``ufunc.reduceat`` accepts:

    - a bare integer -- anything ``operator.index`` accepts, EXCEPT
      ``bool``/``np.bool_``: numpy's own axis parser rejects both with
      ``TypeError: an integer is required``, even though ``bool.__index__``
      would otherwise happily return 0 or 1;
    - that same integer wrapped in a length-1 tuple, which numpy unwraps
      before resolving (``axis=(0,)`` behaves exactly like ``axis=0``) --
      any OTHER tuple length raises ``ValueError: reduceat does not allow
      multiple axes``;
    - ``axis=None`` meaning axis 0, but ONLY when ``a`` is 1-D, and ONLY
      when ``None`` is the axis argument ITSELF -- a length-1 tuple
      unwrapping to ``None`` (``axis=(None,)``) is NOT given this special
      treatment by real numpy; it falls through to the generic integer
      conversion below and raises ``TypeError`` same as any other
      non-integer. ``axis=None`` (bare) raises on ``ndim > 1``
      (``ValueError: reduceat does not allow multiple axes``) and on
      ``ndim == 0`` (``TypeError: cannot reduceat on a scalar``).
    - a 0-d array is never a valid target: a bare out-of-range/scalar axis
      raises ``TypeError: cannot reduceat on a scalar``, while the SAME
      axis wrapped in a 1-tuple raises ``AxisError`` instead -- a genuine
      quirk of numpy's own C parser (the 0-d special case only fires for
      an axis that arrived un-wrapped), reproduced here via ``was_tuple``.

    Anything outside those forms raises the same exception type and (where
    practical) message real numpy raises for it, matching numpy's own
    accept/reject boundary without ever handing the caller-supplied
    ``axis`` object back to numpy for a second, independent resolution.
    """
    was_tuple = isinstance(axis, tuple)
    if was_tuple:
        if len(axis) != 1:
            raise ValueError("reduceat does not allow multiple axes")
        (axis,) = axis
    if axis is None and not was_tuple:
        if ndim == 1:
            return 0
        if ndim == 0:
            raise TypeError("cannot reduceat on a scalar")
        raise ValueError("reduceat does not allow multiple axes")
    if isinstance(axis, (bool, _np.bool_)):
        raise TypeError("an integer is required")
    # ``operator.index`` is the ONLY invocation of whatever protocol
    # ``axis`` exposes -- its TypeError (for a non-integer type, or a
    # custom ``__index__`` that raises) propagates verbatim rather than
    # being caught and re-attempted, matching numpy's own message for the
    # standard non-integer cases exactly (numpy's C parser calls the same
    # ``PyNumber_Index`` machinery under the hood). ``axis`` CAN still be
    # ``None`` here -- a length-1 tuple unwrapping to ``None`` deliberately
    # skips the special-case branch above (see the docstring) and falls
    # through to this generic conversion, where ``operator.index(None)``
    # raises the same ``TypeError`` numpy itself raises for that form.
    axis = _operator.index(axis)  # type: ignore[arg-type]
    if ndim == 0 and not was_tuple:
        raise TypeError("cannot reduceat on a scalar")
    if not (-ndim <= axis < ndim):
        raise _AxisError(axis, ndim)
    return axis


def _reduceat_work_per_lane(indices, n: int) -> tuple[int, int]:
    """Per-lane arithmetic applications and produced cells for ``reduceat``.

    ``indices`` MUST already be the frozen, owned snapshot built by
    :func:`_counted_ufunc_reduceat` -- reading it here must be the only
    read of it, matching the single-read discipline
    :func:`_ufunc_at_touched_cells` follows for ``ufunc.at``'s index.

    NumPy's semantics per segment ``i``: if ``indices[i] < indices[i+1]``,
    ``result[i] = reduce(a[indices[i]:indices[i+1]])`` (``L-1`` applications
    for a length-``L`` segment); otherwise ``result[i] = a[indices[i]]``, a
    plain copy with no arithmetic (0 applications). Every accepted segment
    still produces one output cell, so safely castable index snapshots also
    return ``k`` as a produced-cell billing floor. Rejected ndarray index
    dtypes retain their existing arithmetic bill without gaining that floor.
    The final segment always runs to the end of the axis (length ``n``).
    Segments may overlap or run backwards, so the true application count is
    bounded by the INDEX VALUES, not by the array's own size.

    ``n`` is the resolved axis's length -- ``_resolve_reduceat_axis`` raises
    directly (before this function is ever called) for any axis it cannot
    pin down, so by the time ``n`` reaches here it is always the length of
    a real, validated axis; ``n == 0`` only happens for a genuinely
    zero-length axis, handled the same as an empty index list. The
    remaining guard is on the index VALUES themselves: real
    ``ufunc.reduceat`` requires every index in ``[0, n)`` and raises
    ``IndexError`` otherwise -- including for negative indices, which
    (unlike plain array indexing) it does NOT wrap around. An index outside
    that range would make ``ends - idx64`` below swing arbitrarily far
    below zero (a huge, unbounded phantom segment length) for a call that
    real numpy is about to reject anyway, so any out-of-range index floors
    the whole lane to 0 rather than billing that phantom length.
    """
    k = indices.shape[0] if indices.ndim == 1 else 0
    if k == 0 or n == 0:
        return (0, 0)
    idx64 = indices.astype(_np.int64)
    if bool(_np.any((idx64 < 0) | (idx64 >= n))):
        return (0, 0)
    ends = _np.empty(k, dtype=_np.int64)
    ends[:-1] = idx64[1:]
    ends[-1] = n  # the final segment always runs to the axis end
    lengths = ends - idx64
    # A real reduce segment of length L costs L-1 applications; the
    # non-monotonic (indices[i] >= indices[i+1]) branch is a plain element
    # copy and costs 0 -- np.maximum(lengths - 1, 0) covers both uniformly.
    segment_applications = _np.maximum(lengths - 1, 0)
    # The range guard above makes every contribution at most n-1. Keep the
    # vectorized hot path whenever its Python-int upper bound fits int64;
    # only potentially overflowing totals need per-element Python integers.
    if k * (n - 1) <= _np.iinfo(_np.int64).max:
        applications = int(_np.sum(segment_applications))
    else:
        applications = _builtins.sum(int(value) for value in segment_applications)
    output_cells = (
        k if _np.can_cast(indices.dtype, _np.dtype(_np.intp), casting="safe") else 0
    )
    return (applications, output_cells)


@_counted_wrapper
def _counted_ufunc_reduceat(ufunc, a, indices, *, axis=0, out=None, **kwargs):
    """Cost-tracked ``ufunc.reduceat(a, indices, axis=...)``.

    Cost is the larger of the honest per-segment application count and the
    valid produced-cell floor from :func:`_reduceat_work_per_lane`, each
    times the number of lanes (every combination of indices in the axes
    other than ``axis``).
    Output symmetry is ``None``: arbitrary segment boundaries don't respect
    any axis-permutation group action.
    """
    budget = require_budget()
    # Above every later read of ``out`` -- the billing dtype, the
    # symmetry check, and what gets returned -- and above the deduct,
    # so a refused form costs nothing.
    _preflight_ufunc_out_arity(out, ufunc.nout)
    _preflight_ufunc_opt_out(a, *_flatten_ufunc_out_slots(out))
    out = _normalize_out(out, f"{ufunc.__name__}.reduceat", nout=ufunc.nout)
    # Resolved here, BEFORE ``a`` is read for billing below, for the same
    # reason as ``_counted_ufunc_reduce_generic``: ``np.dtype()`` honours an
    # arbitrary object's ``.dtype`` PROPERTY, which can mutate ``a`` as a
    # side effect. Unlike ``axis`` below, dtype= resolution never needs
    # anything off ``a``, so it moves all the way up here rather than
    # needing a refresh afterwards.
    explicit_dtype = _resolve_explicit_dtype_kwarg(kwargs)
    # Snapshot ``indices`` into ONE owned, frozen copy -- the cost formula
    # below reads the index VALUES (not just ``.size``), which opens a
    # time-of-check/time-of-use gap a live view can't close: numpy re-reads
    # the index buffer from inside ``ufunc.reduceat`` itself, and
    # ``ndarray.resize(n, refcheck=False)`` mutates ``.size`` in place even
    # through a read-only view (see ``_canon_entry`` for the identical gap
    # in ``ufunc.at``). Only an owned copy, unreachable from the caller,
    # closes it. This same snapshot is what gets costed AND what gets
    # handed to ``ufunc.reduceat`` below -- ``indices`` itself must not be
    # read again after this point. This resolution has no dependency on
    # ``a`` either, so -- like ``dtype=`` and ``out=`` above -- it runs
    # before ``a`` is touched.
    #
    # An already-ndarray ``indices`` keeps its own dtype: numpy's ndarray
    # path only safe-casts (an int32 array widens fine; a float64 array
    # still raises downstream exactly as it would have unconverted -- we
    # must not turn a rejected input into an accepted one). A non-ndarray
    # ``indices`` (list, tuple, range, ...) goes through the same lenient
    # sequence-to-intp coercion numpy's own converter applies for that
    # form -- an unsafe cast where floats truncate, and an empty sequence
    # stays intp instead of the float64 ``np.asarray([])`` would infer
    # (which ``ufunc.reduceat`` would then reject).
    if isinstance(indices, _np.ndarray):
        # ``_np.asarray`` -- not just ``_to_base_ndarray``, which only strips
        # flopscope's own wrapper types -- BEFORE reading ``.dtype``: an
        # arbitrary OTHER ndarray subclass can override ``.dtype`` as a
        # Python property that reports a narrower/cheaper kind than its real
        # buffer (e.g. claiming ``intp`` over an actual float64 buffer).
        # numpy's own C-level dispatch reads the true underlying descriptor
        # regardless of what that property claims, so classifying off the
        # property here would accept (and silently truncate) a float-backed
        # index real numpy rejects outright. ``_np.asarray`` always returns a
        # genuine, subclass-free view (a no-op view for an already-plain
        # buffer -- no extra copy), so its ``.dtype`` cannot lie.
        indices_base = _np.asarray(_to_base_ndarray(indices))
        indices_snapshot = _np.array(indices_base, dtype=indices_base.dtype, copy=True)
    else:
        indices_snapshot = _np.array(indices, dtype=_np.intp, copy=True)
    indices_snapshot.flags.writeable = False
    # ``a`` can already be a foreign ndarray subclass, a non-ndarray duck
    # array, or a bare sequence here: numpy dispatches to this wrapper as
    # soon as EITHER ``a`` or ``out`` is flopscope-aware, so a bare
    # ``isinstance(a, ndarray)`` check would leave a non-flopscope operand
    # untouched. ``_resolve_ufunc_data_operand`` is what makes every one of
    # those cases safe: an ndarray subclass overriding
    # ``.dtype``/``.shape``/``.ndim``/``.size`` as Python properties cannot
    # misreport the billing view -- used below for ``n``, ``lanes``, and the
    # billing dtype -- (a genuine ``_np.asarray`` read -- numpy's own ufunc
    # dispatch reads the true descriptor at the C level regardless of what
    # those properties report, see ``_counted_ufunc_outer``), a legitimate
    # foreign subclass (e.g. ``np.ma.MaskedArray``) still reaches numpy below
    # with its semantics intact, and a non-ndarray operand whose type
    # implements ``__array_ufunc__`` is forwarded ORIGINAL rather than
    # materialized, so numpy's own dispatch protocol decides what happens
    # instead of flopscope silently consuming ``__array__`` on its behalf.
    #
    # This FIRST read exists only to hand ``_resolve_reduceat_axis`` an
    # ndim: unlike the generic reduce/accumulate resolver, real
    # ``ufunc.reduceat`` accepts ``axis=None`` only on a 1-D array and
    # range-checks the resolved axis against ndim, so the resolver
    # genuinely needs SOME ndim before it can decide which accepted form
    # applies. It is NOT the billing read.
    a_view, a_fwd = _resolve_ufunc_data_operand(a)
    # Resolve ``axis`` through the same accept/reject boundary real
    # ``ufunc.reduceat`` uses (see ``_resolve_reduceat_axis`` for exactly
    # which forms that covers, including ``axis=None`` on a 1-D array,
    # which numpy fully resolves to axis 0 rather than rejecting). This is
    # the ONLY resolution of ``axis`` -- a form the resolver can't pin down
    # raises HERE, before any budget is deducted, instead of falling
    # through to the real call below with the caller's original object
    # (which numpy would then resolve a second time, possibly differently).
    # ``operator.index`` inside it is participant code and (per the
    # confirmed reproduction this whole module's reordering guards
    # against) can mutate ``a`` in place via an owning ndarray subclass's
    # ``resize(n, refcheck=False)`` -- which is exactly why the billing
    # view is refreshed immediately below rather than trusted as-is.
    resolved_axis = _resolve_reduceat_axis(axis, a_view.ndim)
    # THE billing read: re-derived now that ``axis`` (the last remaining
    # participant-controlled argument) has fully resolved, so it reflects
    # whatever that resolution may have mutated ``a`` into. See
    # ``_refresh_ufunc_data_operand`` for why this cannot double-invoke a
    # stateful ``__array__`` duck.
    a_view, a_fwd = _refresh_ufunc_data_operand(a, (a_view, a_fwd))
    # ``resolved_axis`` was validated against the ndim read BEFORE the
    # refresh; if axis resolution's own mutation also changed ``a``'s
    # ndim, that bound may no longer hold. Re-check it against the FRESH
    # ndim -- never via a second ``_resolve_reduceat_axis``/``operator.index``
    # call, which would invoke the caller's ``__index__`` a second time --
    # before any cost is computed from it, so an axis invalidated by the
    # mutation it caused is refused rather than silently indexed with.
    if not (-a_view.ndim <= resolved_axis < a_view.ndim):
        raise _AxisError(resolved_axis, a_view.ndim)
    n = a_view.shape[resolved_axis]
    applications_per_lane, output_cells_per_lane = _reduceat_work_per_lane(
        indices_snapshot, n
    )
    lanes = a_view.size // n if n else 0
    cost = _builtins.max(
        lanes * applications_per_lane,
        lanes * output_cells_per_lane,
        1,
    )
    # ``out=``'s billing view is captured here -- LAST, after every other
    # participant-controlled argument (``dtype=``, ``indices`` via
    # ``indices_snapshot``, and ``axis`` via ``resolved_axis``/the
    # ``a_view`` refresh above) has already fully resolved. Each of those
    # resolutions runs participant code (a ``.dtype`` property,
    # ``__array__``, ``__index__``) that -- per the confirmed reproduction
    # this whole module's reordering guards against -- can mutate ``out`` in
    # place (e.g. an owning ndarray subclass widening its own dtype as a
    # side effect of ``indices`` being converted). A previous version of
    # this function captured ``out``'s billing view FIRST, before any of
    # that ran, on the reasoning that ``out`` has no shape/dtype dependency
    # on ``a`` -- true, but irrelevant: the mutation isn't ``a`` depending on
    # ``out``, it's participant code reachable from ANY later resolution
    # step reaching into ``out`` and changing it out from under an
    # already-taken snapshot. Capturing it only now, mirroring the
    # treatment ``a`` already gets above, is what makes it reflect whatever
    # numpy is actually about to write into. ``_normalize_out`` already
    # restricts ``out`` to ``None`` or a genuine ``ndarray``, so
    # ``_resolve_ufunc_data_operand`` here can only ever take its ndarray
    # branch: a fresh, no-op ``_np.asarray`` view, never a second
    # ``__array__``/``__array_ufunc__`` call.
    out_view, out_fwd = (
        _resolve_ufunc_data_operand(out) if out is not None else (None, None)
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
        sum_accumulator_dtype(a_view.dtype)
        if ufunc.__name__ in ("add", "multiply")
        else _ufunc_loop_dtype(ufunc, a_view.dtype, a_view.dtype)
    )
    # ``out=`` is threaded through ``reduction_billing_dtype`` as
    # ``out_dtype=`` -- the SAME accumulator model the generic reduce and
    # accumulate paths already use -- rather than folded in with
    # ``store_billing_dtypes``. reduceat's accumulator obeys numpy's reduce
    # semantics exactly like ``reduce``/``accumulate``: ``out=`` is not an
    # accumulator selector, it can only WIDEN. A wider ``out`` genuinely
    # widens the loop (``add.reduceat(float32, out=float64)`` accumulates in
    # float64 -- bit-verified -- so it is billed at the float64 rate), while a
    # narrower ``out`` merely casts the final store and never lowers the bill.
    # Folding the raw ``out`` dtype in with ``store_billing_dtypes`` instead
    # let ``result_type`` promote a real accumulator against a narrower
    # complex store up to the wide complex dtype -- e.g.
    # ``add.reduceat(float64, out=complex64)`` priced at complex128 (4x) when
    # numpy actually runs a complex64 loop (2x, the real part narrowed to
    # float32) -- the over-bill this alignment removes. A non-numeric ``out=``
    # is still refused: ``reduction_billing_dtype`` returns it unchanged into
    # ``billing_dtypes`` for ``deduct`` to reject, exactly as the sibling
    # paths rely on.
    billing_dtypes: tuple = (
        reduction_billing_dtype(
            a_view.dtype,
            explicit_dtype=explicit_dtype,
            out_dtype=out_view.dtype if isinstance(out_view, _np.ndarray) else None,
            default_dtype=default_dtype,
        ),
    )
    with budget.deduct(
        f"{ufunc.__name__}.reduceat",
        flop_cost=cost,
        subscripts=None,
        shapes=(a_view.shape,),
        dtypes=billing_dtypes,
    ):
        # ``resolved_axis`` is what the cost above was billed against --
        # forward THAT to the real call, not the caller's original ``axis``
        # object, so numpy can't re-resolve a different axis a second time
        # (e.g. via a stateful ``__index__``) than the one we billed for. A
        # form ``_resolve_reduceat_axis`` cannot pin down already raised,
        # above, before reaching this point -- there is no fallback to the
        # original ``axis`` object left to guard here.
        result = _restore_foreign_ufunc_out_identity(
            _call_ufunc_with_protocol_timing(
                f"{ufunc.__name__}.reduceat",
                ufunc.reduceat,
                a_fwd,
                indices_snapshot,
                axis=resolved_axis,
                out=out_fwd,
                protocol_operands=(a_fwd, out_fwd),
                **kwargs,
            ),
            out,
            out_fwd,
        )
    if isinstance(result, _ForeignUfuncResult):
        return result.value
    return _wrap_result(result, out=out, symmetry=None)


def _canon_entry(entry):
    """Resolve one index entry to an immutable canonical form.

    Branch order is load-bearing. ``bool`` must be tested before ``int``
    (Python ``bool`` implements ``__index__`` but numpy treats it as a 0-d
    mask that ADDS an axis); ``ndarray`` must be tested before ``__index__``
    (a 0-d integer array implements it).

    Both boolean masks and integer index arrays are snapshotted into an
    owned, read-only copy. Boolean masks need it because their cost depends
    on their VALUES (``count_nonzero``): a live view would let a caller
    mutate the mask between costing and writing. Integer arrays need it for
    a subtler reason: their cost depends only on ``.size``, and ``.size`` is
    NOT immutable while anyone else holds a reference to the same buffer --
    ``ndarray.resize(n, refcheck=False)`` mutates it in place on the array
    it is called on. A read-only VIEW of that array is not enough either:
    resizing the base still succeeds and leaves the view dangling over
    reallocated memory. Only an owned copy, unreachable from the caller,
    closes this. Ordering the count late (see ``_counted_ufunc_at``) is not
    sufficient on its own: numpy re-reads the index buffer from inside
    ``ufunc.at`` itself, AFTER invoking ``__array__`` on the ``values``
    operand, so a participant's ``__array__`` callback can resize the index
    on numpy's own internal re-read -- a step no amount of reordering in
    this module runs before. We only ever freeze copies WE made --
    ``_np.asarray`` can hand back the caller's own array, and freezing that
    would leave a participant's array permanently read-only.
    """
    if entry is None or entry is Ellipsis:
        return entry
    if isinstance(entry, slice):

        def _ix(part):
            return None if part is None else _operator.index(part)

        return slice(_ix(entry.start), _ix(entry.stop), _ix(entry.step))
    if isinstance(entry, (bool, _np.bool_)) and not isinstance(entry, _np.ndarray):
        return _np.bool_(entry)
    if isinstance(entry, _np.ndarray):
        # ``_np.asarray`` -- not just ``_to_base_ndarray`` -- BEFORE reading
        # ``.dtype``: an arbitrary OTHER ndarray subclass can override
        # ``.dtype`` as a Python property, reporting an accepted integer (or
        # boolean) kind while the real buffer underneath is float -- numpy's
        # own C-level index parser reads the true descriptor regardless of
        # what that property claims, so classifying off the property here
        # would accept (and silently truncate) a float-backed index real
        # numpy rejects outright. ``_np.asarray`` always returns a genuine,
        # subclass-free view (a no-op view for an already-plain buffer -- no
        # extra copy), so its ``.dtype`` cannot lie.
        base = _np.asarray(_to_base_ndarray(entry))
        if base.dtype == _np.bool_:
            snapshot = _np.array(base, dtype=bool, copy=True)
            snapshot.flags.writeable = False
            return snapshot
        if base.dtype.kind not in "biu":
            raise IndexError(
                "arrays used as indices must be of integer (or boolean) type"
            )
        snapshot = _np.array(base, dtype=base.dtype, copy=True)
        snapshot.flags.writeable = False
        return snapshot
    if hasattr(type(entry), "__index__"):
        return _operator.index(entry)
    arr = _np.asarray(entry)
    if arr.size == 0 and arr.dtype.kind not in "biu":
        # numpy accepts an empty non-ndarray sequence as an index but rejects a
        # bare empty float ndarray; the ndarray branch above preserves that
        # asymmetry, so this must NOT be gated on the entry's python type.
        arr = arr.astype(_np.intp)
    if arr.dtype.kind not in "biu":
        raise IndexError("arrays used as indices must be of integer (or boolean) type")
    if arr.dtype == _np.bool_:
        snapshot = _np.array(arr, dtype=bool, copy=True)
        snapshot.flags.writeable = False
        return snapshot
    snapshot = _np.array(arr, dtype=arr.dtype, copy=True)
    snapshot.flags.writeable = False
    return snapshot


def _canonical_index(indices):
    """Resolve ``indices`` exactly once into a form safe to bill from AND execute with.

    A top-level tuple is the only multi-axis index form; anything else is a
    single entry (numpy does not treat a bare list as multi-axis).
    """
    if isinstance(indices, tuple):
        return tuple(_canon_entry(e) for e in indices)
    return _canon_entry(indices)


def _ufunc_at_touched_cells(a, indices) -> int:
    """Number of cells ``ufunc.at(a, indices, ...)`` operates on.

    ``indices`` MUST already be canonical (see :func:`_canonical_index`).

    ``ufunc.at`` applies the ufunc once per selected cell and does not
    deduplicate repeated indices, so this equals the size of the indexing
    result::

        prod(broadcast(advanced)) * prod(slice lengths) * prod(trailing axes)

    Advanced operands (integer arrays, boolean masks, integer scalars)
    broadcast together; basic (slice) axes are independent and MULTIPLY, both
    with each other and with the advanced result. Any axis no entry consumes
    is swept in full.
    """
    shape = getattr(a, "shape", None)
    if shape is None:
        return 1
    ndim = len(shape)
    entries = indices if isinstance(indices, tuple) else (indices,)

    def _consumes(entry) -> int:
        if entry is None or entry is Ellipsis:
            return 0
        if isinstance(entry, _np.ndarray) and entry.dtype == _np.bool_:
            return entry.ndim
        if isinstance(entry, (bool, _np.bool_)):
            return 0
        return 1

    filler = ndim - _builtins.sum(_consumes(e) for e in entries)
    expanded: list = []
    for entry in entries:
        if entry is Ellipsis:
            expanded.extend([slice(None)] * _builtins.max(filler, 0))
        else:
            expanded.append(entry)

    adv_shapes: list = []
    basic = 1
    axis = 0
    for entry in expanded:
        if entry is None:
            continue
        if isinstance(entry, _np.ndarray) and entry.dtype == _np.bool_:
            adv_shapes.append((int(_np.count_nonzero(entry)),))
            axis += entry.ndim
            continue
        if isinstance(entry, (bool, _np.bool_)):
            adv_shapes.append((int(bool(entry)),))
            continue
        if isinstance(entry, slice):
            if axis < ndim:
                basic *= len(range(*entry.indices(shape[axis])))
            axis += 1
            continue
        if isinstance(entry, _np.ndarray):
            adv_shapes.append(entry.shape)
            axis += 1
            continue
        adv_shapes.append(())
        axis += 1

    selected = int(_np.prod(_np.broadcast_shapes(*adv_shapes))) if adv_shapes else 1
    trailing = int(_math_prod(shape[axis:])) if axis < ndim else 1
    return _builtins.max(selected * basic * trailing, 1)


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
    _preflight_ufunc_opt_out(a, *args)
    # ``indices`` can be many things: int, list of ints, ndarray, slice,
    # Ellipsis, or a tuple thereof (for multi-axis fancy indexing).
    # ``ufunc.at`` accepts all of these. ``_canonical_index`` resolves every
    # entry -- via ``_canon_entry`` -- into a form safe to both cost and
    # execute with (bool/int arrays snapshotted-or-view-cast, scalars
    # normalized through ``__index__``, slices normalized). Resolve the
    # index ONCE. Everything downstream -- the cost and the write itself --
    # must use ``canonical``; re-reading ``indices`` below would let the
    # billed index and the written index differ.
    canonical = _canonical_index(indices)

    # Resolve the ``values``/``vals`` positional operand (``args[0]``, if
    # present) through the array protocol AT MOST ONCE. Mirrors the
    # three-way rule in :func:`_resolve_ufunc_data_operand` -- see that
    # function's docstring for the full reasoning -- with one narrower
    # exception: a plain Python scalar (bool/int/float/complex, not a numpy
    # scalar) is left alone for BOTH forward and billing, not materialized
    # even once, because ``ufunc.at``'s billing dtype resolution below reads
    # ``type(vals)`` to preserve NEP 50 weak typing against ``a``'s dtype;
    # ``_resolve_ufunc_data_operand`` cannot make that same trade (its
    # ``outer`` caller needs an actual 0-d array's ``.shape`` for the output
    # size, which a bare scalar doesn't carry on its own). This function
    # returns ``(forward, billing)`` -- reversed from
    # ``_resolve_ufunc_data_operand``'s ``(billing_view, forward_obj)`` --
    # matching this call site's pre-existing tuple order.
    #
    # An object that's already an ndarray yields TWO things: ``forward`` --
    # the caller's original object, stripped only of a flopscope wrapper,
    # which is what ``ufunc.at`` actually gets called with below, so a
    # legitimate foreign subclass (e.g. ``np.ma.MaskedArray``) keeps its
    # semantics -- and ``billing`` -- a SEPARATE, genuine ``np.ndarray`` view
    # for reading ``.dtype``: that Python-level attribute is overridable on
    # an arbitrary OTHER ndarray subclass, and numpy's own ufunc dispatch
    # reads the TRUE underlying descriptor at the C level regardless of what
    # it reports -- a subclass instance can report a cheap dtype to whatever
    # reads ``v.dtype`` here while numpy computes at its real (pricier) one.
    # ``_np.asarray`` always returns a genuine ``np.ndarray`` (a no-op VIEW,
    # same buffer, for an already-plain ndarray -- see the parity tests this
    # guards), so its ``.dtype`` cannot lie.
    #
    # A non-ndarray whose TYPE implements ``__array_ufunc__`` is forwarded
    # ORIGINAL: numpy's own dispatch (NEP 13) hands the call to THAT
    # protocol instead of independently reconverting the operand through
    # ``__array__`` a second time (confirmed directly -- see
    # ``_resolve_ufunc_data_operand``), so billing from a single
    # ``_np.asarray(v)`` view here does not reopen the
    # stateful-``__array__`` double-materialization hole the next branch
    # exists to close. Forwarding the ORIGINAL rather than a materialized
    # copy is what lets ``Duck().__array_ufunc__`` fire on ``ufunc.at`` (and
    # very possibly raise ``TypeError``, matching plain numpy) instead of
    # flopscope silently computing from ``__array__`` behind the protocol's
    # back.
    #
    # Anything else (e.g. something exposing only ``__array__``) is
    # materialized via ``_np.asarray`` ONCE right here and used as both
    # ``forward`` and ``billing`` -- there is no legitimate subclass to
    # preserve for a non-ndarray input, and re-deriving it a second time
    # (once for billing, once inside numpy's own conversion) would let a
    # participant report a cheap dtype to us while handing numpy something
    # pricier.
    def _resolve_at_operand(v):
        if _static_array_ufunc_implementation(v) is None:
            _raise_ufunc_opt_out(v)
        if isinstance(v, _np.ndarray):
            forward = _to_base_ndarray(v)
            return forward, _np.asarray(forward)
        if _implements_array_ufunc(v):
            return v, _np.asarray(v)
        if isinstance(v, (bool, int, float, complex)) and not isinstance(
            v, _np.generic
        ):
            return v, v
        resolved = _np.asarray(v)
        return resolved, resolved

    resolved_args = tuple(_resolve_at_operand(v) for v in args)
    forward_args = tuple(forward for forward, _billing in resolved_args)
    billing_args = tuple(billing for _forward, billing in resolved_args)
    # ``a`` (the in-place destination) is read for billing LAST, only now
    # that both ``indices`` (``canonical``, above) and ``values``
    # (``resolved_args``, just above) have been fully resolved -- both of
    # those resolutions run participant code (``__index__`` inside
    # ``_canon_entry``, ``__array__`` inside ``_resolve_at_operand``), and
    # (per the confirmed reproduction this whole module's reordering
    # guards against) that code can mutate ``a`` in place via an owning
    # ndarray subclass's ``resize(n, refcheck=False)``. Reading ``a``'s
    # billing view only now is what makes it reflect whatever numpy is
    # actually about to touch below, rather than a pre-mutation snapshot
    # taken before either resolution ran.
    #
    # ``a`` can already be a foreign ndarray subclass here: numpy
    # dispatches to this wrapper as soon as ``a`` is flopscope-aware, but
    # nothing stops a caller-supplied subclass from ALSO overriding
    # ``.dtype``/``.shape`` as Python properties -- numpy's own ufunc
    # dispatch reads the true descriptor at the C level regardless of what
    # those properties report (the same exposure ``_resolve_at_operand``
    # above closes for the ``values`` operand), so billing off ``a``
    # directly (its dtype and shape are both read further down) would let
    # a lying subclass under-report what numpy actually computes and
    # writes. Read the billing dtype/shape off a SEPARATE ``_np.asarray``
    # view (stripping any flopscope wrapper first, to avoid recursing back
    # through our own protocol handlers) -- a no-op VIEW, same buffer, for
    # an operand that is already a genuine plain ndarray. ``a`` itself
    # stays bound to the caller's original object -- see
    # ``_counted_ufunc_outer`` for why a legitimate foreign subclass (e.g.
    # ``np.ma.MaskedArray``) must still reach numpy below, so the in-place
    # mutation keeps its subclass semantics (not just its raw buffer)
    # rather than silently degrading to a plain ndarray write.
    a_view = _np.asarray(_to_base_ndarray(a)) if isinstance(a, _np.ndarray) else a
    # Same loop resolution as the other generic ufunc-method paths:
    # ``ufunc.at`` applies the ufunc's own resolved loop and casts the
    # result back in place with unsafe casting, so a float-only loop runs
    # on integer arrays WITHOUT raising (exp.at(int32) computes float64).
    # Bill that loop, floored at the array's own rate (the established
    # reduction_billing_dtype semantics -- logical_and.at(f64) keeps the
    # f64 rate). Binary ufuncs contribute their ``vals`` operand: NEP 50
    # weak Python scalars pass as their bare type (bool never widens a
    # loop, so it just repeats the array dtype via the nin-padding);
    # everything else contributes its (already-resolved) billing dtype.
    if hasattr(a_view, "dtype"):
        operands: list = [a_view.dtype]
        if billing_args:
            vals = billing_args[0]
            if isinstance(vals, (bool, int, float, complex)) and not isinstance(
                vals, _np.generic
            ):
                if not isinstance(vals, bool):
                    operands.append(type(vals))
            else:
                operands.append(vals.dtype)
        billing_dtypes: tuple = (
            reduction_billing_dtype(
                a_view.dtype,
                default_dtype=_ufunc_loop_dtype(ufunc, *operands),
            ),
        )
    else:
        billing_dtypes = ()
    # Count touched cells LAST, immediately before the deduct -- after every
    # step above that can run participant code (in particular the
    # ``_np.asarray(vals)`` fallback just above, which can invoke a
    # participant's ``__array__``). ``canonical``'s integer (and boolean)
    # index arrays are owned, read-only copies made by ``_canon_entry`` --
    # not views into a participant's live buffer -- so nothing reachable
    # from participant code, including numpy's OWN re-read of the index
    # buffer from inside ``ufunc.at`` below, can change what this counts
    # or what gets applied out from under it. Counting here anyway is
    # defense in depth: it keeps the count adjacent to the write with
    # nothing but ``budget.deduct`` itself (pure bookkeeping -- see
    # ``_charge_op``) in between. ``a_view`` is used here (rather than
    # ``a``) so a lying ``.shape`` property still can't under-report the
    # touched-cell count.
    n_ops = _ufunc_at_touched_cells(a_view, canonical)
    with budget.deduct(
        f"{ufunc.__name__}.at",
        flop_cost=n_ops,
        subscripts=None,
        shapes=(a_view.shape,) if hasattr(a_view, "shape") else (),
        dtypes=billing_dtypes,
    ):
        # A foreign NEP 13 callback may mutate and then raise, so the epoch
        # must be advanced before the invocation rather than after it returns.
        note_write(_to_base_ndarray(a) if isinstance(a, _np.ndarray) else a)
        result = _call_ufunc_with_protocol_timing(
            f"{ufunc.__name__}.at",
            ufunc.at,
            _to_base_ndarray(a),
            canonical,
            *forward_args,
            protocol_operands=(a, *forward_args),
            **kwargs,
        )
    if isinstance(result, _ForeignUfuncResult):
        return result.value
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

        # Still above the deduct, and above every read of ``out`` below it:
        # an index reduction refuses a non-index destination outright, and a
        # refused form must cost zero.
        if op_name in _INDEX_RETURNING_REDUCTIONS:
            _refuse_non_index_destination(op_name, out, axis)

        new_symmetry = (
            reduce_group(symmetry, ndim=len(a.shape), axis=axis, keepdims=keepdims)
            if symmetry is not None
            else None
        )
        _prepare_symmetric_out(out, new_symmetry)
        cost = reduction_cost(a.shape, axis, symmetry=symmetry) * cost_multiplier
        # A nan-prefixed reduction tests every input element for NaN before
        # reducing -- one extra full pass its plain sibling does not run. The
        # model charges every other value test (count_nonzero, 1-arg where,
        # isclose), so charge this one too, at the input rate.
        if op_name.startswith("nan"):
            cost += pointwise_cost(a.shape)
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
        # An index destination is not an accumulator: it holds positions, not
        # the values being compared, so its width says nothing about the
        # arithmetic and it stays out of the resolution entirely. Supplying
        # the intp buffer numpy would have allocated anyway then prices
        # exactly like the bare call.
        out_dtype = (
            out.dtype
            if isinstance(out, _np.ndarray)
            and op_name not in _INDEX_RETURNING_REDUCTIONS
            else None
        )
        billing_dtypes: tuple = (
            reduction_billing_dtype(
                a.dtype,
                explicit_dtype=explicit_dtype,
                out_dtype=out_dtype,
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
            if out_came_from_args:
                # The destination travels in a positional slot of ``args_list``,
                # so ``np_out_kwarg`` is None and _call_numpy's out= hook never
                # sees it -- the same write that the keyword spelling records
                # would otherwise go unnoticed, leaving a symmetry tag standing
                # over data it no longer describes. Recorded here rather than
                # above the deduct so a refused op does not void a tag on a
                # buffer numpy was never given the chance to write.
                note_write(out_for_np)
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
        result: Any = _call_with_optional_out(
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
        result: Any = _call_with_optional_out(
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
    return _wrap_metered_result(result)  # type: ignore[return-value]


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
        a, b, a_axis, b_axis = _core_contraction_axes(a, b, 1, kwargs)
        return _einsum_routed_binary(
            "vecdot",
            _np.vecdot,
            "...n,...n->...",
            a,
            b,
            a_contract_axis=a_axis,
            b_contract_axis=b_axis,
            **kwargs,
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
        a, b, a_axis, b_axis = _core_contraction_axes(a, b, 1, kwargs)
        return _einsum_routed_binary(
            "matvec",
            _np.matvec,
            "...mn,...n->...m",
            a,
            b,
            a_contract_axis=a_axis,
            b_contract_axis=b_axis,
            **kwargs,
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
        a, b, a_axis, b_axis = _core_contraction_axes(a, b, 2, kwargs)
        return _einsum_routed_binary(
            "vecmat",
            _np.vecmat,
            "...n,...nm->...m",
            a,
            b,
            a_contract_axis=a_axis,
            b_contract_axis=b_axis,
            **kwargs,
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
    # Exactly three, never "three or more": numpy's clip has four positional
    # slots and raises for a fifth, and truncating args here would swallow the
    # extras instead. That is worse than a lenient parse -- ``where`` is
    # keyword-only in numpy's clip, so a caller passing it positionally gets a
    # TypeError from numpy, while a silent truncation would hand back an
    # UNMASKED clip and no warning. Anything past the fourth slot is left for
    # numpy to reject in its own words.
    if len(args) > 3:
        # numpy has exactly four positional slots (a, a_min, a_max, out) and
        # raises for a fifth. Extras must not be silently absorbed as further
        # BOUNDS: ``where`` is keyword-only in numpy's clip, so a caller
        # passing it positionally would otherwise get an UNMASKED clip back
        # and no warning. Raise in numpy's own words rather than inventing one.
        raise TypeError(
            f"clip() takes from 1 to 4 positional arguments but "
            f"{len(args) + 1} were given"
        )
    if len(args) == 3:
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
        # A nan-prefixed reduction tests every input element for NaN before
        # reducing -- one extra full pass its plain sibling does not run. The
        # model charges every other value test (count_nonzero, 1-arg where,
        # isclose), so charge this one too, at the input rate.
        if op_name.startswith("nan"):
            cost += pointwise_cost(tuple(a.shape))

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

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
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
        # A nan-prefixed reduction tests every input element for NaN before
        # reducing -- one extra full pass its plain sibling does not run. The
        # model charges every other value test (count_nonzero, 1-arg where,
        # isclose), so charge this one too, at the input rate.
        if op_name.startswith("nan"):
            cost += pointwise_cost(tuple(a.shape))
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

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
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
# nanmean: mean's reduction + per-output divide, plus the isnan pass surcharge
# _counted_mean applies for any op_name that starts with "nan".
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

    Cost = num_output_orbits × axis_dim (Tier-2 partition-based model),
    plus one full isnan pass over the input (see the surcharge below).
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
    # nanmedian tests every input element for NaN before partitioning -- one
    # extra full pass median does not run. The model charges every other
    # value test (count_nonzero, 1-arg where, isclose), so charge this one
    # too, at the input rate.
    cost += pointwise_cost(tuple(a.shape))

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

    Cost = num_output_orbits × per-output cost (Tier-2 partition-based model),
    plus one full isnan pass over the input (see the surcharge below).
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
    # nanpercentile tests every input element for NaN before partitioning --
    # one extra full pass percentile does not run. The model charges every
    # other value test (count_nonzero, 1-arg where, isclose), so charge this
    # one too, at the input rate.
    cost += pointwise_cost(tuple(a.shape))

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
            _to_base_ndarray(q),
            axis=axis,
            out=out_stripped,
            keepdims=keepdims,
            # q (an array of percentiles) and weights= are secondary operands;
            # strip both or an fnp-built q/weights trips the in-wrapper tripwire.
            **{k: _to_base_ndarray_tree(v) for k, v in kwargs.items()},
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

    Cost = num_output_orbits × per-output cost (Tier-2 partition-based model),
    plus one full isnan pass over the input (see the surcharge below).
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
    # nanquantile tests every input element for NaN before partitioning --
    # one extra full pass quantile does not run. The model charges every
    # other value test (count_nonzero, 1-arg where, isclose), so charge this
    # one too, at the input rate.
    cost += pointwise_cost(tuple(a.shape))

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
            _to_base_ndarray(q),
            axis=axis,
            out=out_stripped,
            keepdims=keepdims,
            # q (an array of probabilities) and weights= are secondary operands;
            # strip both or an fnp-built q/weights trips the in-wrapper tripwire.
            **{k: _to_base_ndarray_tree(v) for k, v in kwargs.items()},
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
            _to_base_ndarray(q),
            axis=axis,
            out=out_stripped,
            keepdims=keepdims,
            # q (an array of percentiles) and weights= are secondary operands;
            # strip both or an fnp-built q/weights trips the in-wrapper tripwire.
            **{k: _to_base_ndarray_tree(v) for k, v in kwargs.items()},
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
            _to_base_ndarray(q),
            axis=axis,
            out=out_stripped,
            keepdims=keepdims,
            # q (an array of probabilities) and weights= are secondary operands;
            # strip both or an fnp-built q/weights trips the in-wrapper tripwire.
            **{k: _to_base_ndarray_tree(v) for k, v in kwargs.items()},
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


def _ensure_contraction_out_written(dest, result) -> None:
    """Write ``result`` into ``dest`` when numpy left the destination alone.

    The contraction wrappers hand ``out`` back, so a numpy that accepted the
    argument and quietly ignored it would return the caller an untouched buffer
    AS the contraction result, at full price. That is not hypothetical for the
    project as a whole: ``np.fft.hfft`` / ``ifft2`` / ``irfft2`` hardcode
    ``out=None`` on numpy 2.0 through 2.4, which is what
    :func:`flopscope.numpy.fft._transforms._ensure_out_written` exists to
    repair. No contraction entry point does it on any numpy in the support
    matrix -- this is a behavioural check, not a version table, so it keeps
    holding on numpy releases that do not exist yet.

    ``dest`` must be the stripped base ndarray handed to numpy: numpy returns
    the very array it was given when it honours ``out=``, and it is never shown
    the caller's ``FlopscopeArray``, so an identity test against that would
    never match. ``may_share_memory`` covers a version returning a distinct
    view onto the destination; an ignored ``out=`` is always a fresh
    allocation, which cannot overlap it.
    """
    if not isinstance(dest, _np.ndarray) or not isinstance(result, _np.ndarray):
        return
    if result is dest or _np.may_share_memory(result, dest):
        return
    _call_numpy(_np.copyto, dest, result, casting="same_kind")


def _validate_contracted_extents(
    op_name: str,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    a_axes: tuple[int, ...],
    b_axes: tuple[int, ...],
) -> None:
    """Refuse a contraction whose paired axes have different extents.

    ``np.dot``, ``np.inner`` and ``np.tensordot`` all require each contracted
    pair to match *exactly*: measured against numpy 2.x, none of the three
    broadcasts a size-1 contracted axis against a size-n one, and every
    unequal pair -- including ``0`` against ``n`` -- is a ``ValueError``.
    Equal extents are always accepted, ``0`` against ``0`` included (the
    contraction is empty and the result is a zero fill), so this check is
    exactly numpy's predicate and cannot refuse a call numpy would have run.

    It has to run *before* the cost is computed and before ``budget.deduct``.
    ``deduct`` charges on entry and does not roll back when the wrapped numpy
    call raises, so pricing first would make an impossible contraction consume
    budget -- or raise ``BudgetExhaustedError`` in place of numpy's shape
    error. Refuse before charging, never charge for a call you are about to
    fail; same rule as the out-of-range axis rejection in
    :func:`_tensordot_axis_index`.

    Neither of the two paths that reach here validates on its own. Above the
    52-letter subscript budget the arithmetic fallback prices from shapes and
    never inspects the pairing at all. Below it the einsum route only *looks*
    like it validates: ``_build_size_map`` rejects two different label sizes,
    but einsum broadcasts an extent of 1, so ``ij,jk->ik`` happily prices
    ``j=1`` against ``j=7`` and leaves numpy to reject the call afterwards.
    Checking here covers both, which is also what keeps the refusal identical
    either side of the letter budget.

    ``a_axes``/``b_axes`` index the two shapes directly, so a negative axis
    resolves the way ``shape[ax]`` does; anything out of range is the caller's
    to reject first. :func:`_tensordot_pair_axes` has already range-checked
    and normalised its pair by the time it calls here, and `dot`/`inner` pass
    axes they derived from the ranks themselves.
    """
    if len(a_axes) != len(b_axes):
        raise ValueError(
            f"{op_name}: contraction pairs {len(a_axes)} axes of the first "
            f"operand with {len(b_axes)} of the second; the counts must match"
        )
    for ax, bx in zip(a_axes, b_axes, strict=True):
        if a_shape[ax] != b_shape[bx]:
            raise ValueError(
                f"{op_name}: contracted axis {ax} of the first operand has "
                f"extent {a_shape[ax]}, but axis {bx} of the second operand "
                f"has extent {b_shape[bx]}; contracted extents must be equal "
                f"(shapes {a_shape} and {b_shape})"
            )


def _core_contraction_axes(
    a: Any, b: Any, b_core_from_end: int, call_kwargs: dict[str, Any]
) -> tuple[Any, Any, int | None, int | None]:
    """Coerce a gufunc contraction's operands and locate its contracted pair.

    ``matmul``'s broadcasting siblings -- ``vecdot`` (``(n),(n)->()``),
    ``matvec`` (``(m,n),(n)->(m)``) and ``vecmat`` (``(n),(n,m)->(m)``) --
    each contract a's LAST axis against b's ``b_core_from_end``-th from the
    end. Their pairing is implicit in the ``...``-broadcasting subscript they
    are priced with, which is not the same as being checked by it: einsum
    broadcasts an extent of 1, numpy's gufunc core does not, so a size-1
    contracted axis was priced and charged against a size-n one before numpy
    refused the call. Returning the pair gets it validated ahead of any cost
    -- see :func:`_validate_contracted_extents`.

    Both axes come back ``None``, skipping validation, in the two cases where
    the default pair is not the pair numpy will contract:

    * An operand with too few dimensions to have that axis. numpy rejects the
      call for the missing core dimension, which is not this check's to
      pre-empt (and it already charged nothing for it).
    * An explicit ``axis=``/``axes=``, which relocates the core dimension off
      the position the subscript assumes. Validating the default pair there
      would refuse calls numpy runs -- a worse failure than the
      charge-then-fail this closes. Those spellings also price from the
      default layout, a separate pre-existing gap this does not touch.
    """
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    if "axis" in call_kwargs or "axes" in call_kwargs:
        return a, b, None, None
    a_axis = a.ndim - 1 if a.ndim >= 1 else None
    b_axis = b.ndim - b_core_from_end if b.ndim >= b_core_from_end else None
    return a, b, a_axis, b_axis


def _fallback_contraction_output_symmetry(
    op_name: str,
    a: Any,
    b: Any,
    a_contract_axis: int | None,
    b_contract_axis: int | None,
):
    """Output symmetry surviving a single contracted axis pair, or ``None``.

    The label-budget fallback's counterpart to what
    ``_resolve_cost_and_output_symmetry`` derives from a subscript string,
    and a direct mirror of ``tensordot``'s non-oversized arm restricted to
    one contracted axis per operand: restrict each operand's group to its
    surviving axes, relabel those axes to their slots in the concatenated
    output (b's offset past a's surviving count), then compose.

    ``None`` -- the dense price -- is returned whenever the composition
    cannot be completed: no symmetry to carry, a caller that supplied no
    axis pair, an oversized group, or an enumeration that blows
    ``dimino_budget`` mid-composition. Every group operation below can raise
    ``_DiminoBudgetExceeded``; ``dot`` and ``inner`` reach this function on
    exactly the high-rank operands that provoke it, and this branch exists
    to stop those two crashing, so the exception must not escape. Falling
    back to ``None`` charges the full dense accumulation, which is the
    never-under-bill direction.
    """
    if a_contract_axis is None or b_contract_axis is None:
        return None
    a_sym = _symmetry_of(a)
    b_sym = _symmetry_of(b)
    if a_sym is None and b_sym is None:
        return None
    if _is_oversized_for_cost_model(a_sym) or _is_oversized_for_cost_model(b_sym):
        try:
            oversized_order = (
                a_sym.order() if _is_oversized_for_cost_model(a_sym) else b_sym.order()  # type: ignore[union-attr]
            )
        except _DiminoBudgetExceeded:
            # Unknown-kind group exceeds budget mid-enumeration; can't compute
            # exact |G|. Sentinel keeps all such groups in one dedup slot.
            oversized_order = -1
        _warn_oversized_once(op_name, oversized_order)
        return None
    a_surviving = tuple(i for i in range(a.ndim) if i != a_contract_axis)
    b_surviving = tuple(j for j in range(b.ndim) if j != b_contract_axis)
    try:
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
        return direct_product_groups(a_sym_remapped, b_sym_remapped)
    except _DiminoBudgetExceeded:
        return None


def _einsum_routed_binary(
    op_name: str,
    np_fn: Any,
    subs: str | None,
    a: Any,
    b: Any,
    *,
    errstate: bool = False,
    nan_check: bool = False,
    out: Any = None,
    fallback_contracted: int | None = None,
    fallback_output_shape: tuple[int, ...] | None = None,
    a_contract_axis: int | None = None,
    b_contract_axis: int | None = None,
    **call_kwargs: Any,
) -> Any:
    """Route a binary contraction op's cost + output-symmetry through the einsum
    accumulation model (FMA=2) and run its native numpy op.

    `subs` is the einsum subscript string for this call's operand layout
    (built by the per-op subscript helper). Charges `op_name` exactly once
    (so each op keeps its own weight), preserves operand symmetry/aliasing via
    `_resolve_cost_and_output_symmetry`, and wraps a symmetric result as
    `SymmetricTensor` — mirroring the existing matmul/dot 2-D behavior.

    `subs=None` means the operands' combined rank exceeded the 52-letter
    subscript budget. That path needs `fallback_contracted` and
    `fallback_output_shape` for the arithmetic price, and takes the
    `*_contract_axis` pair to compose the surviving output symmetry (see
    `_fallback_contraction_output_symmetry`). Callers that pass a literal
    `subs` need neither `fallback_` keyword: both are read only under
    `subs is None`.

    `a_contract_axis`/`b_contract_axis` are NOT fallback-only. When a caller
    supplies the pair it is validated against the operand shapes on every
    path, before any cost is computed and before `budget.deduct` -- see
    `_validate_contracted_extents` for why neither path validates on its own.
    Every caller supplies it, the broadcasting ones (`matmul`, `vecdot`,
    `matvec`, `vecmat`) included: a `...`-broadcasting `subs` makes the
    pairing implicit, not checked, and einsum broadcasts an extent of 1 that
    the gufunc core rejects. `None` means only that this call has no such pair
    to check -- a 0-d operand, or an `axis=`/`axes=` that relocated the core
    dimension (see `_core_contraction_axes`).
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
    # Above every pricing branch below and above ``budget.deduct``: an
    # impossible contraction must cost nothing, whichever side of the letter
    # budget it lands on.
    if a_contract_axis is not None and b_contract_axis is not None:
        _validate_contracted_extents(
            op_name, a.shape, b.shape, (a_contract_axis,), (b_contract_axis,)
        )
    if subs is not None:
        info = _resolve_cost_and_output_symmetry(subs, a, b)
        accumulation = info.accumulation
        canonical_subs = info.canonical_subscripts
        output_symmetry = info.output_symmetry
        cost = accumulation.total
        accumulation_for_billing = accumulation
    else:
        # Out of subscript letters. Price it arithmetically from the same
        # FMA=2 model; repeated-operand savings are forfeited, which errs
        # expensive. Operand symmetry is not: composing the surviving group
        # keeps a symmetric contraction priced the same either side of the
        # letter budget, which is what stops rank from being a surcharge.
        if fallback_contracted is None or fallback_output_shape is None:
            raise ValueError(
                f"{op_name}: subs=None requires fallback_contracted and "
                f"fallback_output_shape"
            )
        if _symmetry_of(a) is not None or _symmetry_of(b) is not None or a is b:
            _warn_label_budget_once(op_name)
        accumulation = _dense_accumulation_cost(
            a.size, b.size, fallback_contracted, fallback_output_shape
        )
        canonical_subs = None
        output_symmetry = _fallback_contraction_output_symmetry(
            op_name, a, b, a_contract_axis, b_contract_axis
        )
        cost = _symmetry_adjusted_cost(
            accumulation.total, fallback_output_shape, output_symmetry
        )
        # Only claim an exact complex override when the symmetry adjustment
        # was a no-op. When it scales `cost` down, the accumulation's mu/total
        # split no longer describes what is being charged, so leave the
        # override None and let complex_factor_for's fail-closed guard fire
        # rather than bill a ratio derived from the wrong total. Same
        # reasoning as tensordot's `accumulation_for_billing`.
        accumulation_for_billing = accumulation if cost == accumulation.total else None
    # The gate the pointwise factories already apply: a tag the caller paid
    # ``as_symmetric`` to validate is a claim about this buffer, so a result
    # that contradicts it is an error; a tag merely inferred from a constant
    # fill is dropped quietly, which keeps a scratch arena usable.
    if out is not None:
        _prepare_symmetric_out(out, output_symmetry)
    inputs_were_whest = isinstance(a, _np.ndarray) and (
        type(a) is not _np.ndarray or type(b) is not _np.ndarray
    )
    billing_dtypes = (a.dtype, b.dtype)
    if isinstance(out, _np.ndarray):
        billing_dtypes += store_billing_dtypes(out)
    resolved = resolve_billing_dtype(billing_dtypes)
    complex_override = (
        contraction_complex_override(accumulation_for_billing, resolved)
        if accumulation_for_billing is not None
        else None
    )
    if out is not None:
        call_kwargs = {**call_kwargs, "out": _to_base_ndarray(out)}
    with budget.deduct(
        op_name,
        flop_cost=cost,
        subscripts=canonical_subs,
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
        _ensure_contraction_out_written(_to_base_ndarray(out), result)
        result = out
    if output_symmetry is not None and _validate_result_symmetry(
        result, output_symmetry
    ):
        # Only when the caller supplied no destination. Wrapping ``out`` would
        # hand back a second object of a different type over the caller's own
        # buffer; numpy and fnp.einsum both return the destination itself.
        if out is None:
            return SymmetricTensor(_np.asarray(result), symmetry=output_symmetry)
    if out is not None:
        return out
    # Two branches, not one, and the asymmetry is deliberate.
    #
    # A flopscope operand has always left here through ``_asflopscope``, which
    # converts even numpy's SCALAR (the 1-D x 1-D shape of dot/inner/matmul/
    # vecdot) into a 0-d FlopscopeArray. The server packs that as an array
    # handle, the participant gets a RemoteArray, and downstream arithmetic is
    # billed. Collapsing this branch into ``_wrap_metered_result`` -- which
    # passes scalars through untouched -- would hand the scalar back, the
    # server would pack it by value, and the grader's charge for that
    # arithmetic would drop from billed to 0. That is a repricing of shipped
    # behaviour, so the branch stays exactly as it was.
    #
    # A plain-numpy operand took ``return result``, handing back the raw
    # ndarray numpy allocated: arithmetic on it billed 0 in-process while the
    # grader billed it through the client's RemoteArray (#193). Only the
    # ndarray half of that is wrong -- a numpy scalar of an msgpack-native
    # dtype reaches the grader by value and is free on both sides -- which is
    # exactly the line ``_wrap_metered_result`` draws. A complex128 scalar is
    # the documented exception: it packs as a handle, so it stays a residual
    # #193 gap here rather than a case this change closes.
    if inputs_were_whest:
        return _asflopscope(result)
    return _wrap_metered_result(result)


@_counted_wrapper
def dot(a: ArrayLike, b: ArrayLike) -> FlopscopeArray:
    """Counted version of np.dot."""
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    fallback_contracted = None
    fallback_output_shape = None
    # ``np.dot`` always contracts a's last axis with b's last-but-one (b's
    # only axis when b is 1-D), so the pair is known on every branch, not just
    # the two that need it for the fallback's symmetry composition. Supplying
    # it everywhere is what gets the contraction validated before it is priced
    # on BOTH sides of the letter budget -- see `_validate_contracted_extents`.
    a_axis = a.ndim - 1 if a.ndim else None
    b_axis = (0 if b.ndim == 1 else b.ndim - 2) if b.ndim else None
    if a.ndim == 2 and b.ndim == 2:
        subs = "ij,jk->ik"
    elif a.ndim == 1 and b.ndim == 1:
        subs = "i,i->"
    elif b.ndim == 1:
        subs = _contraction_subscripts(a.ndim, 1, (a.ndim - 1,), (0,))
        fallback_contracted = a.shape[-1]
        fallback_output_shape = a.shape[:-1]
    else:
        subs = _contraction_subscripts(a.ndim, b.ndim, (a.ndim - 1,), (b.ndim - 2,))
        fallback_contracted = a.shape[-1]
        fallback_output_shape = a.shape[:-1] + b.shape[:-2] + b.shape[-1:]
    return _einsum_routed_binary(  # type: ignore[return-value]
        "dot",
        _np.dot,
        subs,
        a,
        b,
        errstate=False,
        nan_check=True,
        fallback_contracted=fallback_contracted,
        fallback_output_shape=fallback_output_shape,
        a_contract_axis=a_axis,
        b_contract_axis=b_axis,
    )


attach_docstring(dot, _np.dot, "counted_custom", "depends on operand dimensions")


@_counted_wrapper
def matmul(
    a: ArrayLike, b: ArrayLike, out: FlopscopeArray | None = None
) -> FlopscopeArray:
    """Counted version of np.matmul."""
    # Declared rather than accepted through **kwargs, which is how the derived
    # out= coverage discovers an op: it reads the wrapper's parameters, so a
    # destination arriving as varkw is invisible to the sweep. The helper
    # normalizes and prices it -- see _einsum_routed_binary.
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
    # ``np.matmul`` pairs the same two axes ``np.dot`` does -- a's last against
    # b's only axis when b is 1-D, b's last-but-one otherwise -- on every
    # branch above. The pairing being implicit in a broadcasting subscript is
    # not the same as it being CHECKED: einsum broadcasts an extent of 1 and
    # the matmul gufunc's core does not, so ``...ij,...jk->...ik`` happily
    # priced ``j=1`` against ``j=7``, charged it, and only then hit numpy's
    # ValueError. Supplying the pair validates it before any cost is computed
    # -- see `_validate_contracted_extents`. A 0-d operand has no contracted
    # axis at all; numpy rejects it for the missing core dimension, which is
    # not this check's to pre-empt.
    a_axis = a.ndim - 1 if a.ndim else None
    b_axis = (0 if b.ndim == 1 else b.ndim - 2) if b.ndim else None
    return _einsum_routed_binary(
        "matmul",
        _np.matmul,
        subs,
        a,
        b,
        errstate=True,
        nan_check=True,
        out=out,
        a_contract_axis=a_axis,
        b_contract_axis=b_axis,
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
    fallback_contracted = None
    fallback_output_shape = None
    # ``np.inner`` always contracts the last axis of both operands, so the
    # pair is known on every branch. Passing it everywhere -- not only where
    # the fallback needs it for symmetry -- is what gets the contraction
    # validated before it is priced on both sides of the letter budget.
    a_axis = a.ndim - 1 if a.ndim else None
    b_axis = b.ndim - 1 if b.ndim else None
    if a.ndim == 1 and b.ndim == 1:
        subs = "i,i->"
    elif a.ndim == 2 and b.ndim == 2:
        subs = "ij,kj->ik"
    else:
        subs = _contraction_subscripts(a.ndim, b.ndim, (a.ndim - 1,), (b.ndim - 1,))
        fallback_contracted = a.shape[-1]
        fallback_output_shape = a.shape[:-1] + b.shape[:-1]
    return _einsum_routed_binary(  # type: ignore[return-value]
        "inner",
        _np.inner,
        subs,
        a,
        b,
        errstate=False,
        nan_check=False,
        fallback_contracted=fallback_contracted,
        fallback_output_shape=fallback_output_shape,
        a_contract_axis=a_axis,
        b_contract_axis=b_axis,
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
            # numpy's out= contract is identity: hand back the caller's own
            # object, unwrapped. Only the allocated result below may be wrapped.
            return out
        return _wrap_metered_result(result)  # type: ignore[return-value]
    # A SymmetricTensor destination reaches _wrap_result even when the result
    # carries no symmetry. It used to take the branch above and return itself
    # UNWRITTEN: numpy had been handed out=None, nothing ever copied the answer
    # across, and the caller got their untouched destination back having paid
    # the whole contraction, with no exception. _wrap_result copies the result
    # in, records the write so no tag outlives the data it described, and drops
    # the inferred tag the destination no longer earns.
    return _wrap_result(result, out=out, symmetry=output_sym)  # type: ignore[return-value]


attach_docstring(outer, _np.outer, "counted_custom", "n(n+1)/2 FLOPs when v outer v")


def _tensordot_spec_axes(spec):
    """One operand's contracted-axis tuple, by ``np.tensordot``'s own rule.

    numpy decides "one axis or many" with a ``len()`` probe, not a type test::

        try:
            na = len(axes_a); axes_a = list(axes_a)
        except TypeError:
            axes_a = [axes_a]; na = 1

    Mirroring that is what makes every per-operand spelling numpy takes --
    ``int``, any numpy integer scalar, a list, tuple, set, range or ndarray --
    land here with the meaning it has there, and it needs no integer test of
    its own. The predicate this replaces was ``isinstance(spec, int)``, which
    is ``False`` for every numpy integer scalar and so rejected, with
    ``TypeError``, calls numpy runs.

    It also preserves numpy's treatment of a spec with no ``__len__`` that is
    *not* an integer, such as a drained iterator: numpy wraps it in a
    one-element list and then uses it as an index, which fails. Returning
    ``(spec,)`` keeps that failure and moves it into
    :func:`_tensordot_axis_index`, which runs before any budget is
    charged. Draining the iterator into a tuple instead -- what this used to
    do -- both hid the error and left the exhausted object to be forwarded to
    numpy, so the call was priced and charged and only then refused.
    """
    try:
        len(spec)
    except TypeError:
        return (spec,)
    return tuple(spec)


def _tensordot_parse_axes(axes):
    """Split ``np.tensordot``'s ``axes`` argument into two per-operand specs.

    Accepts the same forms as numpy: an integer ``N`` (contract the last
    ``N`` axes of ``a`` with the first ``N`` of ``b``), or a pair of
    per-operand specs. Returns a pair of tuples of raw -- possibly negative,
    possibly not even integers -- contracted axis indices;
    :func:`_tensordot_pair_axes` vets and normalises them.

    The arm is chosen with numpy's own ``try: iter(axes)`` probe rather than
    a type test. Any type test is wrong in one direction or the other: an
    ``isinstance(axes, int)`` test rejects every numpy integer scalar numpy
    accepts, and widening it to ``np.integer`` still sends a 0-d array --
    which numpy reads as a count, since a 0-d array is not iterable -- down
    the unpacking arm to fail with the wrong exception. Probing for iteration
    reproduces numpy's split exactly and needs no list of accepted types.

    The integer arm expands to numpy's own ``range(-axes, 0)`` /
    ``range(0, axes)`` rather than the equivalent-looking
    ``range(a_ndim - axes, a_ndim)``. For every ``axes`` numpy actually runs
    the two agree once normalised, but they part company on an *unsigned*
    scalar: numpy negates it, the value wraps, and the a-side list comes out
    empty, so the pairing count never matches and the call is a
    ``ValueError``. Deriving the axes from ``a_ndim`` instead would invent a
    plausible axis, price the contraction, and only then let numpy refuse it
    -- a charge for work that never ran. Taking numpy's expression verbatim
    needs no unsigned special case.
    """
    try:
        iter(axes)
    except Exception:  # noqa: BLE001 -- numpy's own predicate, verbatim
        return tuple(range(-axes, 0)), tuple(range(axes))
    a_spec, b_spec = axes
    return _tensordot_spec_axes(a_spec), _tensordot_spec_axes(b_spec)


def _tensordot_axis_index(ax, ndim: int, operand: str) -> int:
    """One contracted axis, canonicalised to a plain non-negative ``int``.

    Refuses exactly what ``np.tensordot`` refuses for a per-operand axis, and
    with numpy's exception types:

    * ``TypeError`` for anything that is not an index at all -- a float, a
      string, a ``None``, a nested list, a drained iterator. numpy reaches
      this when it writes ``as_[axes_a[k]]`` and Python's tuple indexing
      fails.
    * ``TypeError`` for a **boolean**. ``bool`` is an ``int`` subclass and
      ``np.bool_`` implements ``__index__``, so a boolean indexes a shape
      tuple happily and survives as far as numpy's internal ``transpose``,
      which refuses it outright. (numpy's whole-``axes`` integer arm still
      takes a ``bool`` -- ``-True`` is ``-1`` -- and must keep taking it;
      ``range`` hands this function plain ints there, so it never sees one.)
    * ``IndexError`` for an integer outside the operand's rank. A 0-d operand
      needs no special case: every index is out of range at rank 0, which is
      what ``np.tensordot`` reports too.

    The plain ``int`` it returns is what keeps the rest of this module -- and
    the pairing eventually handed back to ``np.tensordot`` -- independent of
    how the caller spelled the axis. numpy accepts several spellings of the
    same axis (``int``, ``np.int64``, a 0-d ``ndarray``) but does not treat
    them alike in every release: numpy 2.4 added a duplicate check that puts
    the axis objects in a ``set``, so a 0-d ``ndarray`` -- a perfectly valid
    index, but unhashable -- is refused there and accepted by 2.0 through
    2.3. Canonicalising here means flopscope prices, bills and runs every
    spelling identically on every supported numpy rather than inheriting that
    split (see ``test_zero_d_array_axis_runs_on_every_supported_numpy``).
    """
    if isinstance(ax, (bool, _np.bool_)):
        raise TypeError(
            f"tensordot: {ax!r} is not a valid contracted axis for the "
            f"{operand} operand; numpy refuses a boolean axis (pass an integer)"
        )
    try:
        index = _operator.index(ax)
    except TypeError:
        raise TypeError(
            f"tensordot: contracted axes must be integers, got {type(ax).__name__}"
        ) from None
    if not -ndim <= index < ndim:
        raise IndexError(
            f"tensordot: axis {index} is out of range for the {operand} operand, "
            f"which has {ndim} axes"
        )
    return index + ndim if index < 0 else index


def _tensordot_pair_axes(
    op_name: str,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    a_axes,
    b_axes,
):
    """Vet and normalise a ``tensordot`` pairing, before anything is priced.

    Four checks, in one pass: the two axis counts must match, every axis must
    be an in-range integer, neither operand may name the same axis twice, and
    each pair's extents must be equal. Returns both axis tuples as plain
    non-negative ``int``s, which is what everything downstream (``contracted``,
    ``a_surviving``/``b_surviving``, ``output_shape``, the symmetry
    restriction, and the pairing forwarded to numpy) assumes.

    A spec that fails any of them fails *here*, above every pricing arm and
    every ``budget.deduct`` in :func:`tensordot`. That is what this function
    exists for -- refuse before charging, never charge for a call you are
    about to fail -- and it holds on every numpy in the support range.

    The order is numpy 2.4's: an explicit duplicate check that raises
    ``ValueError`` ahead of the shape work, rather than a repeat falling
    through to the ``transpose`` that refuses it (which is how 2.0 through 2.3
    reach the same ``ValueError``, with no duplicate check of their own).
    numpy reorders these checks between its own releases, so the exception
    *type* for a spec that is wrong in more than one way is numpy's to change
    and is deliberately not something flopscope pins: on
    ``tensordot(zeros((2,3,4)), zeros((3,)), axes=([0,0],[0,1]))`` numpy 2.2
    raises ``IndexError`` and numpy 2.4 ``ValueError``. *Which* specs are
    refused is pinned; which of ``ValueError`` / ``IndexError`` / ``TypeError``
    a given bad one gets is not.
    """
    if len(a_axes) != len(b_axes):
        # Raised through the shared helper so the wording matches what
        # `dot`/`inner` give for the same mis-pairing. numpy compares the two
        # counts before it indexes either shape, and so does this.
        _validate_contracted_extents(op_name, a_shape, b_shape, a_axes, b_axes)
    a_norm = tuple(_tensordot_axis_index(ax, len(a_shape), "first") for ax in a_axes)
    b_norm = tuple(_tensordot_axis_index(bx, len(b_shape), "second") for bx in b_axes)
    for operand, axes in (("first", a_norm), ("second", b_norm)):
        # Normalised, so `([0, -2], ...)` on a rank-2 operand is caught as the
        # duplicate it is rather than slipping through as two distinct labels.
        # numpy's own 2.4 check runs on the raw axes and misses that spelling,
        # leaving its `transpose` to raise the same ValueError one step later.
        # Without the check the duplicate was priced as though the axis really
        # were contracted twice -- its extent multiplied into `contracted`
        # twice, the axis dropped from `a_surviving` once -- and
        # `budget.deduct` charged that on entry, on both sides of the
        # 52-letter budget, for a call numpy then refused.
        if len(set(axes)) != len(axes):
            raise ValueError(
                f"{op_name}: the {operand} operand contracts the same axis "
                f"more than once (axes {axes}); each contracted axis may "
                "appear only once"
            )
    _validate_contracted_extents(op_name, a_shape, b_shape, a_norm, b_norm)
    return a_norm, b_norm


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


def _dense_accumulation_cost(
    a_size: int, b_size: int, contracted: int, output_shape: tuple[int, ...]
) -> AccumulationCost:
    """FLOP cost of a two-operand contraction, without needing subscripts.

    Mirrors ``aggregate_einsum`` for ``k=2`` with no symmetry: ``alpha``
    multiplies plus ``alpha - M`` accumulations, where ``alpha`` is the
    multiply count and ``M`` the number of output cells. Used when the
    52-letter subscript budget is exceeded and no einsum string can be
    built — the price is never below what the einsum path would have
    charged for the same operands, so running out of letters costs
    precision, never a discount.

    Returns a real ``AccumulationCost`` rather than an int so the caller can
    hand it to ``contraction_complex_override`` exactly as it would the
    einsum path's, keeping complex billing exact on this branch too.
    """
    alpha = (a_size * b_size // contracted) if contracted > 0 else 0
    m = _math_prod(output_shape)
    # Zero-sized arithmetic domain: no multiply or accumulation events occur,
    # and there is no first term to receive the free initial-copy correction.
    # Without this guard ``2 * alpha - m`` goes NEGATIVE and refunds budget.
    total = 0 if alpha == 0 else 2 * alpha - m
    return AccumulationCost(
        total=total,
        mu=alpha,
        alpha=alpha,
        m_total=alpha,
        dense_baseline=alpha,
        num_terms=2,
        per_component=(),
        fallback_used=False,
    )


_SUBSCRIPT_LETTERS = _string.ascii_letters
_SUBSCRIPT_BUDGET = len(_SUBSCRIPT_LETTERS)  # 52


def _contraction_subscripts(
    a_ndim: int,
    b_ndim: int,
    a_axes: tuple[int, ...],
    b_axes: tuple[int, ...],
) -> str | None:
    """Einsum subscripts for a two-operand contraction, or ``None``.

    ``a_axes``/``b_axes`` are the paired contracted axis indices; negatives
    are normalised. Returns ``None`` when ``a_ndim + b_ndim`` exceeds the
    52-letter budget.

    ``None`` is the *only* out-of-letters signal in this module: callers
    price the contraction with :func:`_dense_accumulation_cost` instead.
    Every contraction wrapper that needs generated labels routes through
    here, so they cannot drift apart on what "out of letters" means or on
    what it costs.
    """
    if a_ndim + b_ndim > _SUBSCRIPT_BUDGET:
        return None
    a_labels = list(_SUBSCRIPT_LETTERS[:a_ndim])
    b_labels = list(_SUBSCRIPT_LETTERS[a_ndim : a_ndim + b_ndim])
    a_ax = [ax % a_ndim for ax in a_axes]
    b_ax = [ax % b_ndim for ax in b_axes]
    for ai, bi in zip(a_ax, b_ax, strict=False):
        b_labels[bi] = a_labels[ai]  # tie contracted pairs
    out = [a_labels[i] for i in range(a_ndim) if i not in a_ax]
    out += [b_labels[j] for j in range(b_ndim) if j not in b_ax]
    return f"{''.join(a_labels)},{''.join(b_labels)}->{''.join(out)}"


@_counted_wrapper
def tensordot(a: ArrayLike, b: ArrayLike, axes: Any = 2) -> FlopscopeArray:
    """Counted version of ``np.tensordot``.

    The FLOP cost is ``(2K - 1) x M`` for contracted extent ``K`` and ``M``
    output cells — ``K`` multiplies and ``K - 1`` accumulations per cell
    (FMA = 2). A zero-sized contraction costs 0, never a negative amount.

    A :class:`SymmetricTensor` symmetry on either operand, or the same array
    passed as both operands, can bring the charge below that figure. When
    the surviving (post-contraction) symmetry can be composed on the output
    axes via :func:`flopscope._symmetry_utils.direct_product_groups`, the
    cost follows the unique-element fraction of the output (see
    :func:`_symmetry_adjusted_cost`). That composition is skipped, and
    :class:`flopscope.errors.CostFallbackWarning` fires, when a group's
    order exceeds the configured ``dimino_budget``.

    Above the 52-letter einsum subscript budget the cost is computed
    arithmetically instead of through a subscript string. The charge is
    then never lower than what the einsum path would compute for the same
    operands, and can be higher.

    An ``axes`` spec ``np.tensordot`` would refuse is refused here too, and
    before any cost is computed or any budget is charged, so a call that was
    never going to run leaves ``flops_used`` untouched. The exception is
    :class:`ValueError` for unequal axis counts, a pair whose extents differ,
    or the same axis contracted twice on one operand; :class:`IndexError` for
    an axis outside the operand's rank; :class:`TypeError` for something that
    is not an axis at all (a float, a one-shot iterator, or a boolean in the
    per-operand form).

    Which of the three a *particular* bad spec gets is not promised to match
    ``np.tensordot``. numpy reorders these checks between its own releases --
    2.4 added an explicit duplicate check ahead of the shape indexing that
    2.0 through 2.3 reach first -- so no fixed order can agree with the whole
    support range. What is promised is the refusal itself, and that it costs
    nothing.

    ``axes`` accepts what ``np.tensordot`` accepts, numpy integer scalars
    (``np.int64``, ``np.int32``, ...) included, in both the whole-``axes``
    form and the per-operand form. One spelling is deliberately more
    permissive than the newest numpy: a per-operand axis given as a 0-d
    ``ndarray`` runs here on every supported numpy, where numpy itself accepts
    it up to 2.3 and refuses it from 2.4 (its duplicate check hashes the axis
    objects, and a 0-d ``ndarray`` is unhashable). Accepting it uniformly is
    what keeps the same program billed the same amount on every numpy in the
    support range.
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    a_contract_axes, b_contract_axes = _tensordot_parse_axes(axes)
    # Counts, axis validity, duplicates, extents -- all of it above all three
    # pricing arms (full-inner, oversized-symmetry and the ordinary partial
    # contraction) and above every `budget.deduct` in this function, so a
    # contraction numpy is going to refuse costs nothing on either side of the
    # 52-letter budget.
    a_contract_axes, b_contract_axes = _tensordot_pair_axes(
        "tensordot", a.shape, b.shape, a_contract_axes, b_contract_axes
    )
    # Everything below hands numpy the normalised pairing rather than the
    # caller's `axes` object, so the spec is parsed exactly once. Re-parsing it
    # was its own defect: a one-shot spec (an iterator, a generator) is already
    # drained by the time numpy sees it, so a call flopscope had just priced
    # and charged was then refused for running out of values. The two forms are
    # interchangeable for every spec numpy accepts -- `_tensordot_parse_axes`
    # reproduces numpy's own expansion, normalisation preserves each pair's
    # position, and numpy treats `axes=((), ())` exactly as `axes=0`.
    numpy_axes = (a_contract_axes, b_contract_axes)
    # Fast path: a full inner contraction over all axes maps cleanly to
    # einsum and benefits from joint-operand savings when a is b.
    is_full_inner = (
        a.ndim == b.ndim
        and a_contract_axes == tuple(range(a.ndim))
        and b_contract_axes == tuple(range(b.ndim))
        and a.ndim >= 1
    )
    if is_full_inner:
        # Every axis contracted on both sides (e.g. ndim=2 -> "ij,ij->").
        subs = _contraction_subscripts(
            a.ndim, b.ndim, tuple(range(a.ndim)), tuple(range(b.ndim))
        )
        if subs is not None:
            from flopscope._einsum import _resolve_cost_and_output_symmetry

            info = _resolve_cost_and_output_symmetry(subs, a, b)
            accumulation = info.accumulation
            canonical_subs = info.canonical_subscripts
            out_sym = info.output_symmetry  # scalar output — always None
        else:
            # Out of letters. Contracting every axis means K = a.size and a
            # scalar output, so the honest price is 2 * numel - 1.
            if _symmetry_of(a) is not None or _symmetry_of(b) is not None or a is b:
                _warn_label_budget_once("tensordot")
            accumulation = _dense_accumulation_cost(a.size, b.size, a.size, ())
            canonical_subs = None
            out_sym = None
        cost = accumulation.total
        billing_dtypes = (a.dtype, b.dtype)
        resolved = resolve_billing_dtype(billing_dtypes)
        complex_override = contraction_complex_override(accumulation, resolved)
        with budget.deduct(
            "tensordot",
            flop_cost=cost,
            subscripts=canonical_subs,
            shapes=(a.shape, b.shape),
            dtypes=billing_dtypes,
            complex_factor_override=complex_override,
        ):
            result = _call_numpy(
                _np.tensordot, _to_base_ndarray(a), _to_base_ndarray(b), axes=numpy_axes
            )
        if out_sym is not None:
            return _wrap_result(result, symmetry=out_sym)  # type: ignore[return-value]
        return _wrap_metered_result(result)  # type: ignore[return-value]
    # Fallback: keep the existing sophisticated direct_product_groups path
    # for partial contractions and unusual axes specs.
    # `a_contract_axes` is already normalised to `[0, a.ndim)` above, so
    # every entry indexes `a.shape` validly here.
    contracted = 1
    for ax in a_contract_axes:
        contracted *= a.shape[ax]
    # Surviving (non-contracted) axes for each operand.
    a_surviving = tuple(i for i in range(a.ndim) if i not in a_contract_axes)
    b_surviving = tuple(i for i in range(b.ndim) if i not in b_contract_axes)
    output_shape = tuple(a.shape[i] for i in a_surviving) + tuple(
        b.shape[j] for j in b_surviving
    )
    # Route cost through einsum when possible (FMA=2 correct); above the
    # 52-letter budget, price the same FMA=2 model without subscripts.
    _subs = _contraction_subscripts(
        a.ndim, b.ndim, tuple(a_contract_axes), tuple(b_contract_axes)
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
            if a_sym is not None or b_sym is not None or a is b:
                _warn_label_budget_once("tensordot")
            accumulation = _dense_accumulation_cost(
                a.size, b.size, contracted, output_shape
            )
            cost = _symmetry_adjusted_cost(accumulation.total, output_shape, out_sym)
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
            if a_sym is not None or b_sym is not None or a is b:
                _warn_label_budget_once("tensordot")
            accumulation = _dense_accumulation_cost(
                a.size, b.size, contracted, output_shape
            )
            cost = _symmetry_adjusted_cost(accumulation.total, output_shape, out_sym)
            canonical_subs = None
            # Only claim an exact complex override when the symmetry adjustment
            # was a no-op. When it scales `cost` down, the accumulation's
            # mu/total split no longer describes what is being charged, so leave
            # the override None and let complex_factor_for's fail-closed guard
            # fire rather than bill a ratio derived from the wrong total.
            accumulation_for_billing = (
                accumulation if cost == accumulation.total else None
            )
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
            _np.tensordot, _to_base_ndarray(a), _to_base_ndarray(b), axes=numpy_axes
        )
    if out_sym is not None:
        return _wrap_result(result, symmetry=out_sym)  # type: ignore[return-value]
    return _wrap_metered_result(result)  # type: ignore[return-value]


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
    # vdot returns a numpy scalar, never a SymmetricTensor. It stays a numpy
    # scalar: the server packs it by value, so downstream arithmetic on it is
    # free on the grader too, and wrapping it would start charging for that.
    return _wrap_metered_result(result)  # type: ignore[return-value]


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
    return _wrap_metered_result(result)  # type: ignore[return-value]


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
    # Safe at every vector width, including the 2-vector form that contracts the
    # vector axis away: numpy.cross returns a 0-d ndarray there, not a numpy
    # scalar, so the server already stored a handle and the wrap cannot flip the
    # wire form. See test_cross_results_were_already_array_handles.
    return _wrap_metered_result(result)  # type: ignore[return-value]


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
    return _wrap_metered_result(result)  # type: ignore[return-value]


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

    Base cost (no coord arrays, or uniform scalar spacing), summed over the
    axes ``axis=`` actually selects (every axis when ``axis`` is None):
      sum over requested axes of 2 * f.size            (edge_order == 1)
      sum over requested axes of 2 * f.size + 6 * (f.size // L)   (otherwise)

    np.gradient emits one output value per input element along each
    differentiated axis: the interior central difference costs 2 FLOPs
    (sub + div) and, with edge_order == 1, so does each one-sided boundary
    ``(f[1]-f[0])/dx``, giving a flat 2 * f.size per axis independent of its
    length L. For any other accepted edge_order (0, 2, or a non-integer in
    (1, 2]; > 2 numpy rejects) numpy runs the second-order 3-term boundary
    stencil ``a*f[0] + b*f[1] + c*f[2]`` (5 FLOPs), +3 per boundary element;
    the two boundary hyperplanes hold f.size // L elements each, so that adds
    6 * (f.size // L) per axis (L = f.shape[axis]). A single-axis gradient is
    thus exactly 1/ndim of the all-axes gradient and the per-axis costs add
    back up to it exactly.

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
        # Normalise ``axis`` with numpy's own helper -- the very call
        # ``np.gradient`` itself makes -- so the billed axis set is exactly
        # the differentiated one for every accepted spelling (bare int or
        # any ``__index__`` object, tuple/list of ints, negative indices,
        # None meaning every axis) and every rejected one raises numpy's own
        # error, before anything is charged, rather than being silently
        # folded into range with ``%``.
        ax_kw = kwargs.get("axis")
        if ax_kw is None:
            axes = tuple(range(f.ndim))
        else:
            axes = _normalize_axis_tuple(ax_kw, f.ndim)

        # np.gradient runs one independent central-difference pass per
        # REQUESTED axis, so the base cost sums over ``axes`` -- not over
        # every axis of ``f``. Summing over range(f.ndim) charged a
        # single-axis gradient the full n-dimensional price (an ndim-fold
        # over-bill).
        #
        # Each axis produces one output value per input element -- interior
        # AND both boundary hyperplanes. The interior central difference is
        # 2 FLOPs/element (sub + div), so an axis of length L costs
        # 2*(L-2) interior + (2 boundaries) per line, times S/L lines.
        #
        # The boundary stencil depends on ``edge_order``. numpy runs the
        # cheap one-sided difference ``(f[1]-f[0])/dx`` (2 FLOPs) ONLY when
        # edge_order == 1; for any other accepted value (0, 2, or a
        # non-integer in (1, 2]; > 2 numpy rejects) it runs the second-order
        # 3-term boundary stencil ``a*f[0] + b*f[1] + c*f[2]`` (3 mul + 2 add
        # = 5 FLOPs), +3 FLOPs per boundary element. With edge_order == 1
        # every element costs a flat 2 FLOPs so an axis is exactly 2*S
        # regardless of L; otherwise the two boundary hyperplanes (S/L
        # elements each) each cost 3 FLOPs more, adding 6*(S//L) per axis.
        edge_order = kwargs.get("edge_order", 1)
        per_axis_costs = []
        for ax in axes:
            per = 2 * f.size
            if edge_order != 1 and f.shape[ax] > 0:
                # 2 boundary hyperplanes * (S//L lines) * +3 FLOPs each
                per += 6 * (f.size // f.shape[ax])
            per_axis_costs.append(per)
        base = _builtins.max(_builtins.sum(per_axis_costs), 1)

        # --- spacing surcharge ---
        # ``varargs`` pair up positionally with ``axes``, so the surcharge is
        # already attributed to the requested axes only.
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
    # np.gradient returns a tuple for multi-axis input and a single array
    # otherwise; _wrap_metered_result covers both, element by element.
    return _wrap_metered_result(result)  # type: ignore[return-value]


attach_docstring(
    gradient,
    _np.gradient,
    "counted_custom",
    "uniform: sum over requested axes of 2*S (edge_order=1), +6*S/L per axis for edge_order!=1; non-uniform axis adds 3*S*(L-2)/L + 10*(L-2) + 3*(L-1) + 4*S/L FLOPs",
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
    return _wrap_metered_result(result)  # type: ignore[return-value]


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
    return _wrap_metered_result(result)  # type: ignore[return-value]


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
    return _wrap_metered_result(result)  # type: ignore[return-value]


attach_docstring(
    correlate,
    _np.correlate,
    "counted_custom",
    "per-mode FLOPs (FMA=2): full 2nm-n-m+1; valid (2*min-1)*(max-min+1); same exact dot-length sum",
)


def _reject_zero_d_y(y_arr, op_name: str) -> None:
    """Refuse a 0-d ``y`` with ``ValueError``, before any shape indexing.

    The feature-count reads below are guarded only by ``ndim == 1``, so a 0-d
    ``y`` fell through to ``y_arr.shape[0]`` and raised flopscope's own
    IndexError. This runs inside the cost helper, which completes before
    ``budget.deduct``, so nothing is billed before or after and the set of
    calls flopscope refuses is unchanged -- only the exception class and the
    message move.

    numpy refuses the same call whenever ``x`` carries more than one
    observation. It *accepts* a 0-d ``y`` in the single-observation case,
    broadcasting it to ``(1, 1)``: ``np.cov([1.0], np.float64(2.0))`` returns a
    2x2 array. flopscope refused that before this guard (with the IndexError)
    and still refuses it. Accepting it would change accept/refuse on a billable
    call, which is a pricing decision, so the divergence is deliberate and
    pinned by ``test_numpy_accepts_zero_d_y_for_a_single_observation``.
    """
    if y_arr.ndim == 0:
        raise ValueError(
            f"{op_name}: y must be at least 1-dimensional, got a 0-d operand "
            f"(shape {y_arr.shape}, dtype {y_arr.dtype})"
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
        _reject_zero_d_y(y_arr, "cov")
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
        _reject_zero_d_y(y_arr, "corrcoef")
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
    return _wrap_metered_result(result)  # type: ignore[return-value]


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
            # fweights=/aweights= are secondary array operands; strip them or an
            # fnp-built weights array trips the in-wrapper tripwire.
            **{k: _to_base_ndarray_tree(v) for k, v in kwargs.items()},
        )
    return _wrap_metered_result(result)  # type: ignore[return-value]


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
    return _wrap_metered_result(result)  # type: ignore[return-value]


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
        return _wrap_metered_result(result)  # type: ignore[return-value]

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
    return _wrap_metered_result(result)  # type: ignore[return-value]


attach_docstring(
    interp,
    _np.interp,
    "counted_custom",
    "3*n + n*ceil(log2(xp)) FLOPs (arithmetic + search)",
)
interp.__signature__ = _inspect.signature(_np.interp)  # pyright: ignore[reportFunctionMemberAccess]
