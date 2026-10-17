"""Vectorized error function using the fdlibm/Sun rational approximation.

Uses the same algorithm and coefficients as the C99 math.erf / glibc / fdlibm
``s_erf.c`` implementation, which is the gold-standard for double-precision erf.

Four regions:
  |x| < 0.84375 : erf(x) = x + x*P(x^2)/Q(x^2)
  0.84375<=|x|<1.25 : erf(x) = erx + P(|x|-1)/Q(|x|-1)
  1.25 <= |x| < ~6 : erfc(x) = exp(-x^2-0.5625+R(1/x^2)/S(1/x^2)) / |x|
  |x| >= 6        : erf(x) = sign(x) * 1.0

Accuracy: matches C99 math.erf / scipy.special.erf to ~1 ULP.

References
----------
.. [1] Sun Microsystems, "Freely Distributable LIBM (fdlibm) s_erf.c",
   https://www.netlib.org/fdlibm/s_erf.c
   Algorithm and coefficients by Sun Microsystems (see copyright below).

.. [2] glibc implementation (derived from fdlibm):
   https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/ieee754/dbl-64/s_erf.c

Copyright (original fdlibm source)
-----------------------------------
Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Coefficients from fdlibm s_erf.c (Sun / glibc / OpenBSD)
# ---------------------------------------------------------------------------

# erx = erf(1) to extended precision
_erx = 8.45062911510467529297e-01

# Region 1: |x| < 0.84375  --  erf(x) = x + x * pp(x^2)/qq(x^2)
_pp0 = 1.28379167095512558561e-01
_pp1 = -3.25042107247001499370e-01
_pp2 = -2.84817495755985104766e-02
_pp3 = -5.77027029648944159157e-03
_pp4 = -2.37630166566501626084e-05

_qq1 = 3.97917223959155352819e-01
_qq2 = 6.50222499887672944485e-02
_qq3 = 5.08130628187576562776e-03
_qq4 = 1.32494738004321644526e-04
_qq5 = -3.96022827877536812320e-06

# Region 2: 0.84375 <= |x| < 1.25  --  erf(x) = erx + P(s)/Q(s), s = |x| - 1
_pa0 = -2.36211856075265944077e-03
_pa1 = 4.14856118683748331666e-01
_pa2 = -3.72207876035701323847e-01
_pa3 = 3.18346619901161753674e-01
_pa4 = -1.10894694282396677476e-01
_pa5 = 3.54783043195201877747e-02
_pa6 = -2.16637559983254089680e-03

_qa1 = 1.06420880400844228286e-01
_qa2 = 5.40397917702171048937e-01
_qa3 = 7.18286544141962539399e-02
_qa4 = 1.26171219808761642112e-01
_qa5 = 1.36370839120290507362e-02
_qa6 = 1.19844998467991074170e-02

# Region 3a: 1.25 <= |x| < 1/0.35 (~2.857)
_ra0 = -9.86494403484714822705e-03
_ra1 = -6.93858572707181764372e-01
_ra2 = -1.05586262253232909814e01
_ra3 = -6.23753324503260060396e01
_ra4 = -1.62396669462573071767e02
_ra5 = -1.84605092906711035994e02
_ra6 = -8.12874355063065934246e01
_ra7 = -9.81432934416914548592e00

_sa1 = 1.96512716674392571292e01
_sa2 = 1.37657754143519702237e02
_sa3 = 4.34565877475229228608e02
_sa4 = 6.45387271733267880594e02
_sa5 = 4.29008140027567833386e02
_sa6 = 1.08635005541779435134e02
_sa7 = 6.57024977031928170135e00
_sa8 = -6.04244152148580987438e-02

# Region 3b: 1/0.35 (~2.857) <= |x| < 6
_rb0 = -9.86494292470009928597e-03
_rb1 = -7.99283237680523006574e-01
_rb2 = -1.77579549177547519889e01
_rb3 = -1.60636384855557935030e02
_rb4 = -6.37566443368389085394e02
_rb5 = -1.02509513161107724954e03
_rb6 = -4.83519191608651397019e02

_sb1 = 3.03380607875625778203e01
_sb2 = 3.25792512996573918826e02
_sb3 = 1.53672958608443695994e03
_sb4 = 3.19985821950859553908e03
_sb5 = 2.55305040643316442583e03
_sb6 = 4.74528541206955367215e02
_sb7 = -2.24409524465858183362e01


def _erf(x):
    """Vectorized error function matching scipy.special.erf to ~1 ULP.

    Algorithm and coefficients from fdlibm ``s_erf.c`` [1]_.
    """
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)

    # NaN matches none of the region masks below, so initialize its result.
    out = np.full_like(x, np.nan)
    ax = np.abs(x)
    sign = np.sign(x)

    # --- Region 1: |x| < 0.84375 ---
    m1 = ax < 0.84375
    if np.any(m1):
        xm = x[m1]
        s = xm * xm
        r = _pp0 + s * (_pp1 + s * (_pp2 + s * (_pp3 + s * _pp4)))
        S = 1.0 + s * (_qq1 + s * (_qq2 + s * (_qq3 + s * (_qq4 + s * _qq5))))
        out[m1] = xm + xm * (r / S)

    # --- Region 2: 0.84375 <= |x| < 1.25 ---
    m2 = (ax >= 0.84375) & (ax < 1.25)
    if np.any(m2):
        s = ax[m2] - 1.0
        P = _pa0 + s * (
            _pa1 + s * (_pa2 + s * (_pa3 + s * (_pa4 + s * (_pa5 + s * _pa6))))
        )
        Q = 1.0 + s * (
            _qa1 + s * (_qa2 + s * (_qa3 + s * (_qa4 + s * (_qa5 + s * _qa6))))
        )
        out[m2] = sign[m2] * (_erx + P / Q)

    # --- Region 3a: 1.25 <= |x| < 1/0.35 (~2.857) ---
    m3a = (ax >= 1.25) & (ax < (1.0 / 0.35))
    if np.any(m3a):
        axm = ax[m3a]
        s = 1.0 / (axm * axm)
        R = _ra0 + s * (
            _ra1
            + s * (_ra2 + s * (_ra3 + s * (_ra4 + s * (_ra5 + s * (_ra6 + s * _ra7)))))
        )
        S = 1.0 + s * (
            _sa1
            + s
            * (
                _sa2
                + s
                * (_sa3 + s * (_sa4 + s * (_sa5 + s * (_sa6 + s * (_sa7 + s * _sa8)))))
            )
        )
        erfc_val = np.exp(-axm * axm - 0.5625 + R / S) / axm
        out[m3a] = sign[m3a] * (1.0 - erfc_val)

    # --- Region 3b: 1/0.35 <= |x| < 6 ---
    m3b = (ax >= (1.0 / 0.35)) & (ax < 6.0)
    if np.any(m3b):
        axm = ax[m3b]
        s = 1.0 / (axm * axm)
        R = _rb0 + s * (
            _rb1 + s * (_rb2 + s * (_rb3 + s * (_rb4 + s * (_rb5 + s * _rb6))))
        )
        S = 1.0 + s * (
            _sb1
            + s * (_sb2 + s * (_sb3 + s * (_sb4 + s * (_sb5 + s * (_sb6 + s * _sb7)))))
        )
        erfc_val = np.exp(-axm * axm - 0.5625 + R / S) / axm
        out[m3b] = sign[m3b] * (1.0 - erfc_val)

    # --- Region 4: |x| >= 6 ---
    m4 = ax >= 6.0
    if np.any(m4):
        out[m4] = sign[m4] * 1.0

    if scalar:
        return float(out[0])
    return out


def _erfc(x):
    """Vectorized complementary error function: erfc(x) = 1 - erf(x)."""
    return 1.0 - _erf(x)
