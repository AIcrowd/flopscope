"""Laplace distribution for :mod:`flopscope.stats`."""

from __future__ import annotations

import numpy as _np

from flopscope.stats._base import ContinuousDistribution


class LaplaceDistribution(ContinuousDistribution):
    """Laplace (double-exponential) continuous random variable.

    This object mirrors ``scipy.stats.laplace``.

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
    ``loc`` is the center and ``scale`` controls the exponential decay away
    from that center. Per-method FLOP costs (weight 1.0): pdf deducts
    ``22 * numel(broadcast(input, loc, scale))`` (composite: |x-loc|(3) + exp(-z)(17) +
    /(2*scale)(2), FMA=2), cdf deducts ``40 * numel(broadcast(input, loc, scale))`` (composite:
    two eager exp branches + 8 arith/cmp/select; audit-2 verified), ppf
    deducts ``51 * numel(broadcast(input, loc, scale))`` (composite: two eager log branches + edge
    selects; audit-2 verified).
    """

    def __init__(self):
        super().__init__("laplace")

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
        Equivalent to ``scipy.stats.laplace.pdf(x, loc, scale)``.
        FLOP cost: ``22 * numel(broadcast(x, loc, scale))`` (composite: |x-loc|(3) + exp(-z)(17) +
        /(2*scale)(2), FMA=2, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> np.round(flops.stats.laplace.pdf(x), 3)
        array([0.184, 0.5  , 0.184])
        """
        return self._deduct_and_call("pdf", 22, x, loc=loc, scale=scale)

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
        Equivalent to ``scipy.stats.laplace.cdf(x, loc, scale)``.
        FLOP cost: ``40 * numel(broadcast(x, loc, scale))`` (composite: two eager exp branches +
        arithmetic/select, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> np.round(flops.stats.laplace.cdf(x), 3)
        array([0.184, 0.5  , 0.816])
        """
        return self._deduct_and_call("cdf", 40, x, loc=loc, scale=scale)

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
        Equivalent to ``scipy.stats.laplace.ppf(q, loc, scale)``.
        FLOP cost: ``51 * numel(broadcast(q, loc, scale))`` (composite: two eager log branches +
        edge selects, weight 1.0). Derivation (FMA=2): 2 eager log branches
        (2×16=32) + 19 arith/cmp/select (branch A: maximum+2 mul+add=4;
        branch B: sub+maximum+2 mul+sub=5; where#1 cmp+select=2; where#2
        2 cmp+and+select=4; where#3,#4 cmp+select=2 each).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> q = np.array([0.25, 0.5, 0.75])
        >>> np.round(flops.stats.laplace.ppf(q), 3)
        array([-0.693,  0.   ,  0.693])
        """
        return self._deduct_and_call("ppf", 51, q, loc=loc, scale=scale)

    def _compute_pdf(self, x, loc=0, scale=1):
        z = _np.abs(x - loc) / scale
        return _np.exp(-z) / (2.0 * scale)

    def _compute_cdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return _np.where(z <= 0, 0.5 * _np.exp(z), 1.0 - 0.5 * _np.exp(-z))

    def _compute_ppf(self, q, loc=0, scale=1):
        result = _np.where(
            q <= 0.5,
            loc + scale * _np.log(2.0 * _np.maximum(q, 1e-300)),
            loc - scale * _np.log(2.0 * _np.maximum(1.0 - q, 1e-300)),
        )
        result = _np.where((q >= 0) & (q <= 1), result, _np.nan)
        result = _np.where(q == 0, -_np.inf, result)
        result = _np.where(q == 1, _np.inf, result)
        return result


laplace = LaplaceDistribution()
