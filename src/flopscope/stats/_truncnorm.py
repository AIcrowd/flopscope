"""Truncated normal distribution for :mod:`flopscope.stats`."""

from __future__ import annotations

import numpy as _np

from flopscope.stats._base import ContinuousDistribution
from flopscope.stats._erf import _erf
from flopscope.stats._ndtri import _ndtri

_SQRT2 = _np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / _np.sqrt(2.0 * _np.pi)


def _std_norm_cdf(x):
    """Standard normal CDF (no budget deduction)."""
    return 0.5 * (1.0 + _erf(x / _SQRT2))


def _std_norm_pdf(x):
    """Standard normal PDF (no budget deduction)."""
    return _INV_SQRT_2PI * _np.exp(-0.5 * x * x)


class TruncnormDistribution(ContinuousDistribution):
    """Truncated normal continuous random variable.

    This object mirrors ``scipy.stats.truncnorm``.

    Methods
    -------
    pdf(x, a, b, loc=0, scale=1)
        Evaluate the probability density function.
    cdf(x, a, b, loc=0, scale=1)
        Evaluate the cumulative distribution function.
    ppf(q, a, b, loc=0, scale=1)
        Evaluate the percent-point function.

    Notes
    -----
    ``a`` and ``b`` are standardized lower and upper bounds. The truncated
    support is ``[a * scale + loc, b * scale + loc]``, and both bounds appear
    before ``loc`` and ``scale`` to match SciPy's signature. pdf deducts
    ``28 * numel(broadcast(input, a, b, loc, scale))`` FLOPs (composite:
    z(2)+std_norm_pdf(20)+div(1)+bounds(5), FMA=2, weight 1.0; calibrated
    alpha 28.0). cdf deducts ``51 * numel(broadcast(input, a, b, loc, scale))``
    FLOPs (composite: z(2)+std_norm_cdf(46)+result(3)+2 where(4), FMA=2,
    weight 1.0; calibrated alpha 50.6). ppf deducts
    ``81 * numel(broadcast(input, a, b, loc, scale))`` FLOPs (composite:
    erf + ndtri rational approx + arithmetic, weight 1.0; audit-2 verified).
    """

    def __init__(self):
        super().__init__("truncnorm")

    def pdf(self, x, a, b, loc=0, scale=1):
        """Evaluate the probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the density.
        a : float
            Lower standardized bound.
        b : float
            Upper standardized bound.
        loc : float, optional
            Mean of the underlying normal distribution. Defaults to ``0``.
        scale : float, optional
            Standard deviation of the underlying normal distribution.
            Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Probability density evaluated elementwise at ``x``.

        Notes
        -----
        Equivalent to ``scipy.stats.truncnorm.pdf(x, a, b, loc, scale)``.
        FLOP cost: ``28 * numel(broadcast(x, a, b, loc, scale))`` (composite:
        z(2)+std_norm_pdf(20)+div(1)+bounds(5), FMA=2, weight 1.0; calibrated
        alpha 28.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-0.5, 0.0, 0.5])
        >>> np.round(flops.stats.truncnorm.pdf(x, a=-1.0, b=1.0), 3)
        array([0.516, 0.584, 0.516])
        """
        return self._deduct_and_call("pdf", 28, x, a, b, loc=loc, scale=scale)

    def cdf(self, x, a, b, loc=0, scale=1):
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative probability.
        a : float
            Lower standardized bound.
        b : float
            Upper standardized bound.
        loc : float, optional
            Mean of the underlying normal distribution. Defaults to ``0``.
        scale : float, optional
            Standard deviation of the underlying normal distribution.
            Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Cumulative probability evaluated elementwise at ``x``.

        Notes
        -----
        Equivalent to ``scipy.stats.truncnorm.cdf(x, a, b, loc, scale)``.
        FLOP cost: ``51 * numel(broadcast(x, a, b, loc, scale))`` (composite:
        z(2)+std_norm_cdf(46)+result(3)+2 where(4), FMA=2, weight 1.0;
        calibrated alpha 50.6).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-0.5, 0.0, 0.5])
        >>> np.round(flops.stats.truncnorm.cdf(x, a=-1.0, b=1.0), 3)
        array([0.22, 0.5 , 0.78])
        """
        return self._deduct_and_call("cdf", 51, x, a, b, loc=loc, scale=scale)

    def ppf(self, q, a, b, loc=0, scale=1):
        """Evaluate the percent-point function.

        Parameters
        ----------
        q : array_like
            Probabilities in ``[0, 1]``.
        a : float
            Lower standardized bound.
        b : float
            Upper standardized bound.
        loc : float, optional
            Mean of the underlying normal distribution. Defaults to ``0``.
        scale : float, optional
            Standard deviation of the underlying normal distribution.
            Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Quantiles corresponding to ``q``.

        Notes
        -----
        Equivalent to ``scipy.stats.truncnorm.ppf(q, a, b, loc, scale)``.
        FLOP cost: ``81 * numel(broadcast(q, a, b, loc, scale))`` (composite:
        erf + ndtri rational approx + arithmetic, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> q = np.array([0.25, 0.5, 0.75])
        >>> np.round(flops.stats.truncnorm.ppf(q, a=-1.0, b=1.0), 3)
        array([-0.442,  0.   ,  0.442])
        """
        return self._deduct_and_call("ppf", 81, q, a, b, loc=loc, scale=scale)

    def _compute_pdf(self, x, a, b, loc=0, scale=1):
        z = (x - loc) / scale
        phi_a = _std_norm_cdf(a)
        phi_b = _std_norm_cdf(b)
        denom = scale * (phi_b - phi_a)
        result = _std_norm_pdf(z) / denom
        return _np.where((z >= a) & (z <= b), result, 0.0)

    def _compute_cdf(self, x, a, b, loc=0, scale=1):
        z = (x - loc) / scale
        phi_a = _std_norm_cdf(a)
        phi_b = _std_norm_cdf(b)
        result = (_std_norm_cdf(z) - phi_a) / (phi_b - phi_a)
        result = _np.where(z < a, 0.0, result)
        result = _np.where(z > b, 1.0, result)
        return result

    def _compute_ppf(self, q, a, b, loc=0, scale=1):
        phi_a = _std_norm_cdf(a)
        phi_b = _std_norm_cdf(b)
        inner = phi_a + q * (phi_b - phi_a)
        z = _ndtri(inner)
        return loc + scale * z


truncnorm = TruncnormDistribution()
