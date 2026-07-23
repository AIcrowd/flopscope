"""Logistic distribution for :mod:`flopscope.stats`."""

from __future__ import annotations

import numpy as _np

from flopscope.stats._base import ContinuousDistribution


class LogisticDistribution(ContinuousDistribution):
    """Logistic continuous random variable.

    This object mirrors ``scipy.stats.logistic``.

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
    ``loc`` is the center of the distribution and ``scale`` controls the
    spread. The CDF is the sigmoid function and the PPF is the logit
    function. Per-method FLOP costs (weight 1.0): pdf deducts
    ``23 * numel(broadcast(input, loc, scale))`` (composite: z(2)+exp(-z)(17)+(1+ez)(1)+sq(1)+
    scale*(1)+div(1), FMA=2), cdf deducts ``21 * numel(broadcast(input, loc, scale))`` (composite:
    z(2)+exp(-z)(17)+1+ez(1)+1/denom(1), FMA=2), ppf deducts
    ``28 * numel(broadcast(input, loc, scale))`` (composite: 1-q(1)+q/(1-q)(1)+log(16)+
    loc+scale*(2)+3 where(8), FMA=2).
    """

    def __init__(self):
        super().__init__("logistic")

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
        Equivalent to ``scipy.stats.logistic.pdf(x, loc, scale)``.
        FLOP cost: ``23 * numel(broadcast(x, loc, scale))`` (composite: z(2)+exp(-z)(17)+
        (1+ez)(1)+sq(1)+scale*(1)+div(1), FMA=2, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> np.round(flops.stats.logistic.pdf(x), 3)
        array([0.197, 0.25 , 0.197])
        """
        return self._deduct_and_call("pdf", 23, x, loc=loc, scale=scale)

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
        Equivalent to ``scipy.stats.logistic.cdf(x, loc, scale)``.
        FLOP cost: ``21 * numel(broadcast(x, loc, scale))`` (composite: z(2)+exp(-z)(17)+
        1+ez(1)+1/denom(1), FMA=2, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> np.round(flops.stats.logistic.cdf(x), 3)
        array([0.269, 0.5  , 0.731])
        """
        return self._deduct_and_call("cdf", 21, x, loc=loc, scale=scale)

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
        Equivalent to ``scipy.stats.logistic.ppf(q, loc, scale)``.
        FLOP cost: ``28 * numel(broadcast(q, loc, scale))`` (composite: 1-q(1)+q/(1-q)(1)+log(16)+
        loc+scale*(2)+3 where(8), FMA=2, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> q = np.array([0.25, 0.5, 0.75])
        >>> np.round(flops.stats.logistic.ppf(q), 3)
        array([-1.099,  0.   ,  1.099])
        """
        return self._deduct_and_call("ppf", 28, q, loc=loc, scale=scale)

    def _compute_pdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        ez = _np.exp(-z)
        return ez / (scale * (1.0 + ez) ** 2)

    def _compute_cdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return 1.0 / (1.0 + _np.exp(-z))

    def _compute_ppf(self, q, loc=0, scale=1):
        result = loc + scale * _np.log(q / (1.0 - q))
        result = _np.where((q > 0) & (q < 1), result, _np.nan)
        result = _np.where(q == 0, -_np.inf, result)
        result = _np.where(q == 1, _np.inf, result)
        return result


logistic = LogisticDistribution()
