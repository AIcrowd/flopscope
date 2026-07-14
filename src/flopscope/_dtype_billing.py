"""Dtype-aware billing helpers: resolved calculation dtype and complex factors.

The billed dtype for a call is ``np.result_type`` over the declared operands
(input dtypes, python scalars for NEP 50 weak promotion, plus any explicit
``dtype=``/``out=`` dtype). ``dtypes=()`` declares a dtype-neutral op
(rate 1.0, factor 1.0).
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


def resolve_billing_dtype(dtypes: tuple) -> _np.dtype | None:
    """Resolved calculation dtype, or None for a declared dtype-neutral call."""
    if not dtypes:
        return None
    return _np.result_type(*dtypes)


def rate_for(resolved: _np.dtype) -> float:
    return get_dtype_rate(resolved.name)


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
    complex values without doing complex arithmetic, so their factor is 1.0
    (one relocation per value). Charged ops are guaranteed an explicit factor
    by ``test_complex_factor_completeness``, so a missing classification here
    is never a charged op silently defaulting.

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
        # allocates whole complex values, no complex arithmetic -> factor 1.0.
        return 1.0
    return float(factor)
