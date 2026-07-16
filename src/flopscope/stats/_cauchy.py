"""Cauchy distribution for :mod:`flopscope.stats`."""

from __future__ import annotations

import numpy as _np

from flopscope.stats._base import ContinuousDistribution


class CauchyDistribution(ContinuousDistribution):
    """Cauchy (Lorentz) continuous random variable.

    This object mirrors ``scipy.stats.cauchy``.

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
    ``loc`` is the location parameter and ``scale`` is the half-width at
    half-maximum. pdf deducts ``6 * numel(broadcast(input, loc, scale))`` FLOPs (pure-arithmetic
    composite: z=(x-loc)/scale; 1/(pi*scale*(1+z^2)) = sub+div+mul+add+mul+div
    = 6 FLOPs/elem, weight 1.0; calibrated alpha 6.0). cdf deducts
    ``20 * numel(broadcast(input, loc, scale))`` FLOPs (composite: z(2) + arctan(16) + /pi(1) +
    0.5+(1), FMA=2, weight 1.0). ppf deducts ``28 * numel(broadcast(input, loc, scale))`` FLOPs
    (composite: q-0.5(1) + pi*(1) + tan(16) + loc+scale*(2) + 3 where(8),
    FMA=2, weight 1.0).
    """

    def __init__(self):
        super().__init__("cauchy")

    def pdf(self, x, loc=0, scale=1):
        """Evaluate the probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the density.
        loc : float, optional
            Location parameter of the distribution. Defaults to ``0``.
        scale : float, optional
            Scale parameter of the distribution. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Probability density evaluated elementwise at ``x``.

        Notes
        -----
        Equivalent to ``scipy.stats.cauchy.pdf(x, loc, scale)``.
        FLOP cost: ``6 * numel(broadcast(x, loc, scale))`` (pure-arithmetic composite:
        z=(x-loc)/scale; 1/(pi*scale*(1+z^2)) = 6 FLOPs/elem, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> np.round(flops.stats.cauchy.pdf(x), 3)
        array([0.159, 0.318, 0.159])
        """
        return self._deduct_and_call("pdf", 6, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative probability.
        loc : float, optional
            Location parameter of the distribution. Defaults to ``0``.
        scale : float, optional
            Scale parameter of the distribution. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Cumulative probability evaluated elementwise at ``x``.

        Notes
        -----
        Equivalent to ``scipy.stats.cauchy.cdf(x, loc, scale)``.
        FLOP cost: ``20 * numel(broadcast(x, loc, scale))`` (composite: z(2) + arctan(16) +
        /pi(1) + 0.5+(1), FMA=2, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> np.round(flops.stats.cauchy.cdf(x), 3)
        array([0.25, 0.5 , 0.75])
        """
        return self._deduct_and_call("cdf", 20, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        """Evaluate the percent-point function.

        Parameters
        ----------
        q : array_like
            Probabilities in ``[0, 1]``.
        loc : float, optional
            Location parameter of the distribution. Defaults to ``0``.
        scale : float, optional
            Scale parameter of the distribution. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Quantiles corresponding to ``q``.

        Notes
        -----
        Equivalent to ``scipy.stats.cauchy.ppf(q, loc, scale)``.
        FLOP cost: ``28 * numel(broadcast(q, loc, scale))`` (composite: q-0.5(1) + pi*(1) +
        tan(16) + loc+scale*(2) + 3 where(8), FMA=2, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> q = np.array([0.25, 0.5, 0.75])
        >>> np.round(flops.stats.cauchy.ppf(q), 3)
        array([-1.,  0.,  1.])
        """
        return self._deduct_and_call("ppf", 28, q, loc=loc, scale=scale)

    def _compute_pdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return 1.0 / (_np.pi * scale * (1.0 + z * z))

    def _compute_cdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return 0.5 + _np.arctan(z) / _np.pi

    def _compute_ppf(self, q, loc=0, scale=1):
        result = loc + scale * _np.tan(_np.pi * (q - 0.5))
        result = _np.where((q > 0) & (q < 1), result, _np.nan)
        result = _np.where(q == 0, -_np.inf, result)
        result = _np.where(q == 1, _np.inf, result)
        return result


cauchy = CauchyDistribution()
