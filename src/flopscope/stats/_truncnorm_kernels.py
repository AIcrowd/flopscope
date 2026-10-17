"""Stable pure-NumPy kernels for the truncated normal distribution.

Every iterative path has four fixed Newton steps; narrow intervals use eight
fixed Gauss-Legendre nodes. These constants are part of the analytical numerical
cost bound in docs/reference/cost-model.md and must be reviewed if changed.

The complementary-tail rational approximation and coefficients derive from
Sun fdlibm s_erf.c: https://www.netlib.org/fdlibm/s_erf.c

Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.

The inverse-normal rational seed is reused from flopscope.stats._ndtri.
SciPy is not a runtime dependency.
"""

from __future__ import annotations

import numpy as np

from flopscope.stats import _erf as _e
from flopscope.stats import _ndtri as _n

# Exact primary fdlibm values where the current flopscope transcription differs.
# Keep these local so this distribution does not change other erf consumers.
_TAIL_COEFF_CORRECTIONS = {
    "_ra4": -1.62396669462573470355e02,
    "_sa2": 1.37657754143519042600e02,
    "_rb3": -1.60636384855821916062e02,
    "_rb4": -6.37566443368389627722e02,
    "_sb1": 3.03380607434824582924e01,
}
LOG2 = np.log(2.0)
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)
SQRT2 = np.sqrt(2.0)
LOG_NDTR_MODERATE_BOUND = -1.25 * SQRT2
LOG_NDTR_SEED_BOUND = np.log(_n._P_LOW)
# Fixed 8-node Gauss-Legendre integration on [0,1]. Only used when the density
# varies by less than approximately 11% across the whole interval.
NODES = np.array(
    [
        0.019855071751231912,
        0.10166676129318664,
        0.2372337950418355,
        0.4082826787521751,
        0.5917173212478248,
        0.7627662049581645,
        0.8983332387068134,
        0.9801449282487681,
    ],
    dtype=np.float64,
)
WEIGHTS = np.array(
    [
        0.05061426814518853,
        0.11119051722668721,
        0.15685332293894344,
        0.18134189168918083,
        0.18134189168918083,
        0.15685332293894344,
        0.11119051722668721,
        0.05061426814518853,
    ],
    dtype=np.float64,
)


def _scalar(out):
    return float(out) if out.ndim == 0 else out


def _horner(x, coefficients):
    y = np.zeros_like(x) + coefficients[-1]
    for c in coefficients[-2::-1]:
        y = y * x + c
    return y


def _coefficient(name):
    return _TAIL_COEFF_CORRECTIONS.get(name, getattr(_e, name))


def log_ndtr(x):
    """Log Phi(x), retaining the complementary-tail rational form directly."""
    x = np.asarray(x, dtype=float)
    y = -np.abs(x)
    out = np.full_like(x, np.nan)
    moderate = np.isfinite(y) & (y > LOG_NDTR_MODERATE_BOUND)
    out[moderate] = np.log(0.5 * (1.0 + _e._erf(y[moderate] / SQRT2)))
    tail = np.isfinite(y) & ~moderate
    for region, prefix, nr, ns in (
        (tail & (-y / SQRT2 < 1.0 / 0.35), "a", 8, 8),
        (tail & (-y / SQRT2 >= 1.0 / 0.35), "b", 7, 7),
    ):
        t = y[region]
        s = 2.0 / (t * t)
        r = _horner(s, tuple(_coefficient(f"_r{prefix}{i}") for i in range(nr)))
        d = _horner(
            s, (1.0,) + tuple(_coefficient(f"_s{prefix}{i}") for i in range(1, ns + 1))
        )
        out[region] = -0.5 * t * t - 0.5625 + r / d - np.log(-t / SQRT2) - LOG2
    out[np.isinf(y)] = -np.inf
    positive = x > 0
    out[positive] = np.log1p(-np.exp(out[positive]))
    return _scalar(out)


def _narrow(a, b):
    with np.errstate(invalid="ignore", over="ignore"):
        return (
            np.isfinite(a)
            & np.isfinite(b)
            & ((b - a) * np.maximum(1, np.maximum(abs(a), abs(b))) < 0.1)
        )


def _local_integral(a, width):
    """Integral of exp(-a*t-t*t/2) from 0 to width, near-unit integrand."""
    t = width[..., None] * NODES
    return width * np.sum(WEIGHTS * np.exp(-a[..., None] * t - 0.5 * t * t), axis=-1)


def log_mass(a, b):
    """Log normal probability on [a,b], including very narrow intervals."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    out = np.full_like(a, np.nan)
    valid = a < b
    narrow = valid & _narrow(a, b)
    out[narrow] = (
        -0.5 * a[narrow] ** 2
        - LOG_SQRT_2PI
        + np.log(_local_integral(a[narrow], b[narrow] - a[narrow]))
    )
    side = valid & ~narrow & ((b <= 0) | (a >= 0))
    lo = np.where(a[side] >= 0, -b[side], a[side])
    hi = np.where(a[side] >= 0, -a[side], b[side])
    lp_hi, lp_lo = log_ndtr(hi), log_ndtr(lo)
    out[side] = lp_hi + np.log(-np.expm1(lp_lo - lp_hi))
    center = valid & ~narrow & ~side
    out[center] = np.log1p(-np.exp(log_ndtr(a[center])) - np.exp(log_ndtr(-b[center])))
    out[a == b] = -np.inf
    return _scalar(out)


def ndtri_log_lower(logp):
    """Inverse Phi from log probability <= log(.5); four fixed Newton steps."""
    logp = np.asarray(logp, float)
    out = np.full_like(logp, np.nan)
    valid = np.isfinite(logp) & (logp <= -LOG2)
    middle = valid & (logp >= LOG_NDTR_SEED_BOUND)
    out[middle] = _n._ndtri(np.exp(logp[middle]))
    tail = valid & ~middle
    q = np.sqrt(-2.0 * logp[tail])
    out[tail] = _horner(q, _n._C[::-1]) / _horner(q, (1.0,) + _n._D[::-1])
    for _ in range(4):
        x = out[valid]
        lp = log_ndtr(x)
        derivative = np.exp(-0.5 * x * x - LOG_SQRT_2PI - lp)
        out[valid] -= (lp - logp[valid]) / derivative
    out[np.isneginf(logp)] = -np.inf
    return _scalar(out)


def _inputs(x, a, b, loc, scale):
    x, a, b, loc, scale = np.broadcast_arrays(
        *(np.asarray(v, float) for v in (x, a, b, loc, scale))
    )
    valid = (a < b) & ~np.isnan(loc) & ~np.isnan(scale) & (scale > 0) & ~np.isnan(x)
    return x, a, b, loc, scale, valid


def pdf(x, a, b, loc=0, scale=1):
    x, a, b, loc, scale, valid = _inputs(x, a, b, loc, scale)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        z = (x - loc) / scale
    valid &= ~np.isnan(z)
    out = np.full_like(x, np.nan)
    out[valid] = 0
    active = valid & (z >= a) & (z <= b) & np.isfinite(z)
    narrow = active & _narrow(a, b)
    t = z[narrow] - a[narrow]
    out[narrow] = (
        np.exp(-a[narrow] * t - 0.5 * t * t)
        / _local_integral(a[narrow], b[narrow] - a[narrow])
        / scale[narrow]
    )
    regular = active & ~narrow
    out[regular] = (
        np.exp(-0.5 * z[regular] ** 2 - LOG_SQRT_2PI - log_mass(a[regular], b[regular]))
        / scale[regular]
    )
    return _scalar(out)


def cdf(x, a, b, loc=0, scale=1):
    x, a, b, loc, scale, valid = _inputs(x, a, b, loc, scale)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        z = (x - loc) / scale
    valid &= ~np.isnan(z)
    out = np.full_like(x, np.nan)
    out[valid & (z <= a)] = 0
    out[valid & (z >= b)] = 1
    active = valid & (z > a) & (z < b)
    narrow = active & _narrow(a, b)
    out[narrow] = _local_integral(a[narrow], z[narrow] - a[narrow]) / _local_integral(
        a[narrow], b[narrow] - a[narrow]
    )
    regular = active & ~narrow
    mass = log_mass(a[regular], b[regular])
    lower = log_mass(a[regular], z[regular]) - mass
    upper = log_mass(z[regular], b[regular]) - mass
    out[regular] = np.where(lower <= -LOG2, np.exp(lower), -np.expm1(upper))
    return _scalar(np.clip(out, 0, 1))


def ppf(q, a, b, loc=0, scale=1):
    q, a, b, loc, scale, valid = _inputs(q, a, b, loc, scale)
    valid &= (q >= 0) & (q <= 1)
    out = np.full_like(q, np.nan)
    out[valid & (q == 0)] = a[valid & (q == 0)]
    out[valid & (q == 1)] = b[valid & (q == 1)]
    active = valid & (q > 0) & (q < 1)
    narrow = active & _narrow(a, b)
    aa, width, qq = a[narrow], b[narrow] - a[narrow], q[narrow]
    t = qq * width
    total = _local_integral(aa, width)
    for _ in range(4):
        t -= (_local_integral(aa, t) - qq * total) / np.exp(-aa * t - 0.5 * t * t)
        t = np.clip(t, 0, width)
    out[narrow] = aa + t
    regular = active & ~narrow
    qq, aa, bb = q[regular], a[regular], b[regular]
    lq, l1q = np.log(qq), np.log1p(-qq)
    lp = np.logaddexp(l1q + log_ndtr(aa), lq + log_ndtr(bb))
    ls = np.logaddexp(l1q + log_ndtr(-aa), lq + log_ndtr(-bb))
    left = lp <= ls
    z = np.empty_like(lp)
    z[left] = ndtri_log_lower(lp[left])
    z[~left] = -ndtri_log_lower(ls[~left])
    out[regular] = z
    out = np.clip(out, a, b)
    return _scalar(loc + scale * out)
