"""Exponential distribution for :mod:`flopscope.stats`."""

from __future__ import annotations

import numpy as _np

from flopscope.stats._base import ContinuousDistribution


class ExponDistribution(ContinuousDistribution):
    """Exponential continuous random variable.

    This object mirrors ``scipy.stats.expon``.

    Methods
    -------
    pdf(x, loc=0, scale=1)
        Evaluate the probability density function.
    cdf(x, loc=0, scale=1)
        Evaluate the cumulative distribution function.
    ppf(q, loc=0, scale=1)
        Evaluate the percent-point function.

    Notes
    -----
    ``loc`` shifts the origin and ``scale`` is the reciprocal of the rate.
    Per-method FLOP costs (weight 1.0): pdf deducts ``22 * numel(broadcast(input, loc, scale))``
    (composite: z=(x-loc)/scale(2) + exp(-z)(17) + /scale(1) + where(2),
    FMA=2), cdf deducts ``22 * numel(broadcast(input, loc, scale))`` (composite: z(2) + exp(-z)(17)
    + 1-exp(1) + where(2), FMA=2), ppf deducts ``27 * numel(broadcast(input, loc, scale))``
    (composite: log1p(-q)(19) + 3 where/cmp/and(8), FMA=2).
    """

    def __init__(self):
        super().__init__("expon")

    def pdf(self, x, loc=0, scale=1):
        """Evaluate the probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the density.
        loc : float, optional
            Location parameter that shifts the support. Defaults to ``0``.
        scale : float, optional
            Scale parameter of the distribution. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Probability density evaluated elementwise at ``x``.

        Notes
        -----
        Equivalent to ``scipy.stats.expon.pdf(x, loc, scale)``.
        FLOP cost: ``22 * numel(broadcast(x, loc, scale))`` (composite: z=(x-loc)/scale(2) +
        exp(-z)(17) + /scale(1) + where(2), FMA=2, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([0.0, 1.0, 2.0])
        >>> np.round(flops.stats.expon.pdf(x), 3)
        array([1.   , 0.368, 0.135])
        """
        return self._deduct_and_call("pdf", 22, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative probability.
        loc : float, optional
            Location parameter that shifts the support. Defaults to ``0``.
        scale : float, optional
            Scale parameter of the distribution. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Cumulative probability evaluated elementwise at ``x``.

        Notes
        -----
        Equivalent to ``scipy.stats.expon.cdf(x, loc, scale)``.
        FLOP cost: ``22 * numel(broadcast(x, loc, scale))`` (composite: z(2) + exp(-z)(17) +
        1-exp(1) + where(2), FMA=2, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([0.0, 1.0, 2.0])
        >>> np.round(flops.stats.expon.cdf(x), 3)
        array([0.   , 0.632, 0.865])
        """
        return self._deduct_and_call("cdf", 22, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        """Evaluate the percent-point function.

        Parameters
        ----------
        q : array_like
            Probabilities in ``[0, 1]``.
        loc : float, optional
            Location parameter that shifts the support. Defaults to ``0``.
        scale : float, optional
            Scale parameter of the distribution. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Quantiles corresponding to ``q``.

        Notes
        -----
        Equivalent to ``scipy.stats.expon.ppf(q, loc, scale)``.
        FLOP cost: ``27 * numel(broadcast(q, loc, scale))`` (composite: loc-scale*log1p(-q)(19) +
        3 where/cmp/and(8), FMA=2, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> q = np.array([0.25, 0.5, 0.75])
        >>> np.round(flops.stats.expon.ppf(q), 3)
        array([0.288, 0.693, 1.386])
        """
        return self._deduct_and_call("ppf", 27, q, loc=loc, scale=scale)

    def _compute_pdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return _np.where(x >= loc, _np.exp(-z) / scale, 0.0)

    def _compute_cdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return _np.where(x >= loc, 1.0 - _np.exp(-z), 0.0)

    def _compute_ppf(self, q, loc=0, scale=1):
        result = loc - scale * _np.log1p(-q)
        result = _np.where((q >= 0) & (q <= 1), result, _np.nan)
        result = _np.where(q == 0, loc, result)
        result = _np.where(q == 1, _np.inf, result)
        return result


expon = ExponDistribution()
