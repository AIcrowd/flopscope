"""Dtype-aware billing helpers: resolved calculation dtype and complex factors.

The billed dtype for a call is ``np.result_type`` over the declared operands
(input dtypes, python scalars for NEP 50 weak promotion, plus any explicit
``dtype=``/``out=`` dtype). ``dtypes=()`` declares a dtype-neutral op
(rate 1.0, factor 1.0).

Folding ``out=`` into that resolution prices the widest participating
buffer -- ``max(compute width, store width)`` -- not just the compute width.
A wider ``out=`` forces a real materialization: writing the result into a
wider buffer is itself chargeable, the same as any other copy into a wider
dtype. Billing it at the wider rate holds ``out=``-casting at exact
``astype`` parity in both directions -- a wider ``out=`` costs what the
equivalent ``astype`` costs, and a narrower ``out=`` never discounts the
loop that actually runs. Reductions separately fold ``out=`` into
``reduction_billing_dtype`` for a different, genuine reason: there, a wider
``out=`` changes the accumulator numpy actually runs, not merely the final
store, so that path is unaffected by this note.
"""

from __future__ import annotations

import numpy as _np

from flopscope._weights import get_dtype_rate
from flopscope.errors import UnsupportedDtypeError


def billing_operand(orig, coerced):
    """Result-type operand for billing.

    Python scalars are passed through so numpy's weak-promotion rules apply
    (``f32_array * 2.0`` bills float32); everything else contributes the
    coerced array's dtype.
    """
    if isinstance(orig, (bool, int, float, complex)) and not isinstance(
        orig, _np.generic
    ):
        return orig
    return coerced.dtype


def _drop_non_numeric(dtypes: tuple) -> tuple:
    """Ignore non-numeric contributions when anything numeric participates.

    ``result_type`` resolves any mix containing a non-numeric kind *to* that
    kind, which then bills at the neutral rate 1.0 and, for contractions, also
    drops the complex factor. That is the intended answer only when the
    arithmetic itself is non-numeric. It is the wrong answer when the
    non-numeric dtype is merely where the result is stored -- an ``out=`` of
    object or string dtype -- because the loop still runs at the operands'
    precision. Dropping those contributions can only raise the resolved rate
    (they carry the minimum rate), so this never discounts a call.
    """
    numeric = tuple(d for d in dtypes if _resolved_kind(d) not in _NON_NUMERIC_KINDS)
    return numeric or dtypes


def _resolved_kind(dtype_like) -> str:
    try:
        return _np.result_type(dtype_like).kind
    except Exception:
        return ""


def resolve_billing_dtype(dtypes: tuple) -> _np.dtype | None:
    """Resolved calculation dtype, or None for a declared dtype-neutral call.

    Some numpy loops accept operand mixes that have NO common promoted dtype
    (logical ufuncs accept anything; ``timedelta64 / float64`` divides without
    promoting), so ``result_type`` can refuse a combination the op itself
    supports. In that case bill conservatively at the heaviest individual
    operand's rate instead of failing a call plain numpy would allow.
    """
    if not dtypes:
        return None
    dtypes = _drop_non_numeric(dtypes)
    try:
        return _np.result_type(*dtypes)
    except _np.exceptions.DTypePromotionError:
        resolved = [_np.result_type(d) for d in dtypes]  # each alone resolves
        return max(resolved, key=rate_for)


# Non-numeric dtype kinds: object (O), str (U), bytes (S), void/structured (V),
# datetime64 (M), timedelta64 (m). As OPERANDS these are not floating-point
# arithmetic, so no precision-packing exploit is possible through them -- they
# bill at the neutral rate 1.0 instead of failing closed, and their wall time is
# covered by the residual-time penalty.
#
# That reasoning covers operands only. A non-numeric dtype arriving as ``out=``
# describes where the result is stored, not how it was computed: the loop still
# runs at the operands' precision, at native speed, so neither the neutral rate
# nor the residual penalty applies to it. ``_drop_non_numeric`` therefore keeps
# such a dtype out of the resolution -- otherwise it would drag the rate (and,
# for contractions, the complex factor) down to 1.0.
#
# The fail-closed rule targets NUMERIC dtypes absent from the supported table --
# future types numpy or an extension package might introduce; every known numpy
# numeric dtype (including the extended-precision longdouble family) carries an
# explicit rate.
_NON_NUMERIC_KINDS = frozenset("OUSVMm")


def rate_for(resolved: _np.dtype) -> float:
    if resolved.kind in _NON_NUMERIC_KINDS:
        return 1.0
    return get_dtype_rate(resolved.name)


def heavier_billing_dtype(*dtypes: _np.dtype) -> _np.dtype:
    """Return the operand dtype with the highest billing rate (ties -> first).

    Unlike ``np.result_type``, this never promotes to a *third* dtype. Use it
    where the billed cost is the MAX of the operand rates rather than the
    promoted-loop rate -- e.g. ``astype``, which reads the source and writes
    the destination, so it should bill at whichever is pricier. ``result_type``
    would be wrong twice over there: ``result_type(float32, int32) == float64``
    over-charges a narrowing cross-kind cast, while billing the source alone
    under-charges when the destination is pricier (``complex64 -> float64``).
    """
    dts = [_np.dtype(d) for d in dtypes]
    return max(dts, key=rate_for)


_DEFAULT_INT = _np.dtype(_np.int_)
_DEFAULT_UINT = _np.dtype(_np.uint)


def sum_accumulator_dtype(a_dtype: _np.dtype) -> _np.dtype:
    """numpy's accumulator dtype for a ``sum``/``prod``-style reduction with
    ``dtype=None``.

    numpy widens a narrow-integer or boolean reduction to the default platform
    integer to avoid overflow (bool/int8-32 -> int64, uint8-32 -> uint64 on a
    64-bit platform); floats and complex -- including float16 -- keep their own
    dtype. Billing on the input dtype would charge such a reduction at the
    narrow-integer rate even though numpy accumulates at 64-bit, a width
    discount; billing on this accumulator dtype closes it. Keying the threshold
    off the platform's own ``int_`` size keeps it correct where the default
    integer is 32-bit.
    """
    kind, size = a_dtype.kind, a_dtype.itemsize
    if kind == "b" or (kind == "i" and size < _DEFAULT_INT.itemsize):
        return _DEFAULT_INT
    if kind == "u" and size < _DEFAULT_UINT.itemsize:
        return _DEFAULT_UINT
    return a_dtype


def mean_compute_dtype(a_dtype: _np.dtype) -> _np.dtype:
    """numpy's compute dtype for ``mean``/``var``/``std`` with ``dtype=None``.

    numpy evaluates the mean/variance of a boolean or integer array in
    ``float64``; float and complex inputs keep their own dtype (so complex
    variance still carries its complex factor). Billing on the input dtype
    would charge an integer mean/variance at the integer rate even though the
    arithmetic runs in float64.
    """
    if a_dtype.kind in "biu":
        return _np.dtype(_np.float64)
    return a_dtype


def reduction_billing_dtype(
    a_dtype: _np.dtype,
    *,
    explicit_dtype=None,
    out_dtype: _np.dtype | None = None,
    default_dtype: _np.dtype | None = None,
) -> _np.dtype:
    """Billed dtype for a reduction: the accumulator numpy actually runs.

    An explicit ``dtype=`` (positional or keyword) IS numpy's accumulator --
    billed as requested, wider or narrower. ``out=`` without ``dtype=``
    widens the intermediate when it is wider than the input; a narrower
    ``out`` only casts the final store, so the loop keeps the input's own
    width. With neither, the family default applies (integer widening for
    sum/prod, float64 for integer mean/var, the ufunc loop dtype for
    generic methods), never billed below the operand width: a value-testing
    loop (comparison, logical) reads full-width operands even though its
    output is bool.
    """
    if explicit_dtype is not None:
        return _np.dtype(explicit_dtype)
    accumulator = (
        out_dtype
        if out_dtype is not None
        else (default_dtype if default_dtype is not None else a_dtype)
    )
    return heavier_billing_dtype(accumulator, a_dtype)


def unary_float_loop_dtype(resolved: _np.dtype) -> _np.dtype:
    """Compute dtype of a float-only UNARY ufunc (exp/sin/sqrt family).

    numpy selects the same-size float loop for integer/bool inputs
    (exp(int8) -> float16, exp(int16) -> float32, exp(int32/64) -> float64);
    float and complex inputs keep their own loop. Billing the raw integer
    dtype would charge int32 transcendentals at the 32-bit rate while the
    arithmetic runs in float64.
    """
    if resolved.kind not in "biu":
        return resolved
    if resolved.itemsize <= 1:
        return _np.dtype(_np.float16)
    if resolved.itemsize == 2:
        return _np.dtype(_np.float32)
    return _np.dtype(_np.float64)


def binary_float_loop_dtype(resolved: _np.dtype) -> _np.dtype:
    """Compute dtype of a float-only BINARY ufunc (divide/arctan2/hypot...).

    Unlike the unary case, numpy resolves integer/bool operand pairs of a
    binary float-only ufunc to the default float: divide(int8, int8) ->
    float64. Float and complex inputs keep their own loop.
    """
    if resolved.kind in "biu":
        return _np.dtype(_np.float64)
    return resolved


def fft_billing_dtype(input_dtype: _np.dtype) -> _np.dtype:
    """Compute dtype of an FFT: the complex working precision.

    Half-width inputs (float16/float32/complex64) run the complex64 path;
    everything else — float64, complex128, and ALL integer inputs — runs
    complex128. The complex-arithmetic structure is already priced into the
    5N*log2(N) formulas (complex_factor 1.0), so the rate axis carries the
    component width: complex64 bills 1.0, complex128 bills 2.0.
    """
    kind, size = input_dtype.kind, input_dtype.itemsize
    if (kind == "f" and size <= 4) or (kind == "c" and size <= 8):
        return _np.dtype(_np.complex64)
    return _np.dtype(_np.complex128)


def linalg_compute_dtype(resolved: _np.dtype) -> _np.dtype:
    """Compute dtype of a LAPACK-backed linalg op.

    numpy.linalg maps integer/bool inputs to float64 (its _commonType);
    float32/float64/complex inputs keep their own LAPACK driver precision.
    """
    if resolved.kind in "biu":
        return _np.dtype(_np.float64)
    return resolved


def linalg_billing_dtypes(*dtypes) -> tuple:
    """dtypes= tuple for a LAPACK-backed deduct site: one resolved compute dtype.

    Mirrors ``numpy.linalg._commonType``: the moment ANY operand is
    non-inexact (integer/bool), the driver runs in double precision --
    regardless of the other operand's width -- so ``solve(bool_matrix,
    float32_vector)`` computes float64 and ``solve(int8_matrix,
    complex64_vector)`` computes complex128. Only all-inexact operand sets
    keep their promoted single-precision drivers.
    """
    dts = [_np.dtype(d) for d in dtypes]
    if any(dt.kind in "biu" for dt in dts):
        if any(dt.kind == "c" for dt in dts):
            return (_np.dtype(_np.complex128),)
        return (_np.dtype(_np.float64),)
    return (linalg_compute_dtype(_np.result_type(*dts)),)


# Generic ufunc method names are "<ufunc>.<method>"; their per-element
# arithmetic is the base ufunc's, so they inherit its complex factor.
_UFUNC_METHOD_SUFFIXES = (".reduce", ".accumulate", ".reduceat", ".outer", ".at")


def complex_factor_for(op_name: str, resolved: _np.dtype) -> float:
    """Complex structure factor for one billed unit of ``op_name``.

    1.0 for real dtypes. For complex dtypes the factor comes from the op's
    registry classification. Ops explicitly marked ``"illegal"`` (numpy raises
    on complex) or ``"exact"`` (contraction family — the call site must supply
    an override) fail closed. Ops with NO classification are free /
    data-movement / blacklisted / unknown: they relocate or allocate whole
    complex values, and a complex value is two real components, so their
    factor is 2.0 (one unit per component). Charged ops are guaranteed an
    explicit factor by ``test_complex_factor_completeness``, so a missing
    classification here is never a charged op silently defaulting.

    A generic ufunc-method name (``"<ufunc>.<method>"``, e.g.
    ``"multiply.reduce"``) that is not itself a registry key falls back to
    its base ufunc's factor: the per-element arithmetic of ``reduce`` /
    ``accumulate`` / ``reduceat`` / ``outer`` / ``at`` IS the base ufunc's.
    Real registry keys that happen to contain dots (``fft.fft``,
    ``linalg.svd``, ``linalg.outer``, ``stats.norm.pdf``) always resolve on
    the direct lookup first, so they are never mistaken for a ufunc method.
    """
    if resolved.kind != "c":
        return 1.0
    from flopscope._registry import REGISTRY

    entry = REGISTRY.get(op_name)
    if entry is None:
        for suffix in _UFUNC_METHOD_SUFFIXES:
            if op_name.endswith(suffix):
                entry = REGISTRY.get(op_name[: -len(suffix)])
                break
    factor = None if entry is None else entry.get("complex_factor")
    if factor == "illegal":
        raise UnsupportedDtypeError(
            f"operation {op_name!r} is not defined for complex dtypes "
            f"(resolved dtype {resolved.name!r})"
        )
    if factor == "exact":
        raise RuntimeError(
            f"operation {op_name!r} computes its complex cost exactly; the call "
            "site must pass complex_factor_override to deduct()/deduct_after()"
        )
    if factor is None:
        # Free / data-movement / blacklisted / unknown op: relocates or
        # allocates whole complex values -- two real components per value.
        return 2.0
    return float(factor)
