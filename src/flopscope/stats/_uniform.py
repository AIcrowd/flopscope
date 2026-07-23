"""Uniform distribution for :mod:`flopscope.stats`."""

from __future__ import annotations

import numpy as _np

from flopscope.stats._base import ContinuousDistribution


class UniformDistribution(ContinuousDistribution):
    """Continuous uniform random variable on ``[loc, loc + scale]``.

    This object mirrors ``scipy.stats.uniform``.

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
    ``loc`` is the lower bound and ``scale`` is the interval width. pdf and
    ppf deduct ``1 * numel(broadcast(input, loc, scale))`` FLOPs. cdf deducts
    ``4 * numel(broadcast(input, loc, scale))`` FLOPs (composite: sub + div +
    2 clip compare/selects, weight 1.0; calibrated alpha 4.0).
    """

    def __init__(self):
        super().__init__("uniform")

    def pdf(self, x, loc=0, scale=1):
        """Evaluate the probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the density.
        loc : float, optional
            Lower bound of the support. Defaults to ``0``.
        scale : float, optional
            Width of the support interval. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Probability density evaluated elementwise at ``x``.

        Notes
        -----
        Equivalent to ``scipy.stats.uniform.pdf(x, loc, scale)``.
        FLOP cost: ``1 * numel(broadcast(x, loc, scale))``.

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-0.5, 0.5, 1.5])
        >>> np.round(flops.stats.uniform.pdf(x), 3)
        array([0., 1., 0.])
        """
        return self._deduct_and_call("pdf", 1, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative probability.
        loc : float, optional
            Lower bound of the support. Defaults to ``0``.
        scale : float, optional
            Width of the support interval. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Cumulative probability evaluated elementwise at ``x``.

        Notes
        -----
        Equivalent to ``scipy.stats.uniform.cdf(x, loc, scale)``.
        FLOP cost: ``4 * numel(broadcast(x, loc, scale))`` (composite: sub + div + 2 clip
        compare/selects, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([0.0, 0.5, 1.0])
        >>> np.round(flops.stats.uniform.cdf(x), 3)
        array([0. , 0.5, 1. ])
        """
        return self._deduct_and_call("cdf", 4, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        """Evaluate the percent-point function.

        Parameters
        ----------
        q : array_like
            Probabilities in ``[0, 1]``.
        loc : float, optional
            Lower bound of the support. Defaults to ``0``.
        scale : float, optional
            Width of the support interval. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Quantiles corresponding to ``q``.

        Notes
        -----
        Equivalent to ``scipy.stats.uniform.ppf(q, loc, scale)``.
        FLOP cost: ``1 * numel(broadcast(q, loc, scale))``.

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> q = np.array([0.25, 0.5, 0.75])
        >>> np.round(flops.stats.uniform.ppf(q), 3)
        array([0.25, 0.5 , 0.75])
        """
        return self._deduct_and_call("ppf", 1, q, loc=loc, scale=scale)

    def _compute_pdf(self, x, loc=0, scale=1):
        return _np.where((x >= loc) & (x <= loc + scale), 1.0 / scale, 0.0)

    def _compute_cdf(self, x, loc=0, scale=1):
        return _np.clip((x - loc) / scale, 0.0, 1.0)

    def _compute_ppf(self, q, loc=0, scale=1):
        result = loc + q * scale
        result = _np.where((q >= 0) & (q <= 1), result, _np.nan)
        return result


uniform = UniformDistribution()
