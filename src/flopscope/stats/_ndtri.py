"""Inverse standard normal CDF (ndtri) via a rational approximation + Newton
refinement.

Accuracy: ~1e-12 against scipy.special.ndtri.
"""

from __future__ import annotations

import numpy as np

from flopscope.stats._erf import _erf

# ---------------------------------------------------------------------------
# Rational approximation coefficients
# ---------------------------------------------------------------------------

_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)

_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW

_SQRT2 = np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_pdf_internal(x):
    """Standard normal PDF (internal helper, no budget deduction)."""
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _norm_cdf_internal(x):
    """Standard normal CDF (internal helper, no budget deduction)."""
    return 0.5 * (1.0 + _erf(x / _SQRT2))


def _ndtri(p):
    """Inverse standard normal CDF.

    Maps probability *p* in (0, 1) to the quantile *x* such that
    Phi(x) = p, where Phi is the standard normal CDF.

    Uses a rational approximation with one Newton-Raphson refinement step
    for ~1e-12 accuracy.

    Edge cases: p=0 -> -inf, p=1 -> +inf, p<0 or p>1 -> nan.
    """
    p = np.asarray(p, dtype=np.float64)
    scalar = p.ndim == 0
    p = np.atleast_1d(p)

    out = np.empty_like(p)

    # Edge cases
    out[p == 0.0] = -np.inf
    out[p == 1.0] = np.inf
    out[(p < 0.0) | (p > 1.0)] = np.nan

    # Lower region: 0 < p < P_LOW
    m_low = (p > 0.0) & (p < _P_LOW)
    if np.any(m_low):
        q = np.sqrt(-2.0 * np.log(p[m_low]))
        num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        out[m_low] = num / den

    # Central region: P_LOW <= p <= P_HIGH
    m_mid = (p >= _P_LOW) & (p <= _P_HIGH)
    if np.any(m_mid):
        q = p[m_mid] - 0.5
        r = q * q
        num = ((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]
        den = ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        out[m_mid] = q * num / den

    # Upper region: P_HIGH < p < 1
    m_high = (p > _P_HIGH) & (p < 1.0)
    if np.any(m_high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[m_high]))
        num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        out[m_high] = -(num / den)

    # Newton refinement (one step) for the interior
    m_interior = (p > 0.0) & (p < 1.0)
    if np.any(m_interior):
        x0 = out[m_interior]
        phi = _norm_cdf_internal(x0)
        pdf = _norm_pdf_internal(x0)
        # Avoid division by zero in pdf
        safe = pdf > 1e-300
        correction = np.where(safe, (phi - p[m_interior]) / pdf, 0.0)
        out[m_interior] = x0 - correction

    if scalar:
        return float(out[0])
    return out
