"""Truncated normal distribution for :mod:`flopscope.stats`."""

from __future__ import annotations

from flopscope.stats import _truncnorm_kernels
from flopscope.stats._base import ContinuousDistribution

# Fixed analytical bounds for the numerical kernels, before dtype pricing.
# See docs/reference/cost-model.md. Four Newton steps and eight quadrature nodes
# are part of this derivation; changing them requires revisiting these bounds.
_PDF_COST = 315
_CDF_COST = 844
_PPF_COST = 1392


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
    before ``loc`` and ``scale`` to match SciPy's signature. Invalid bounds
    (``a >= b``), nonpositive scales, and NaN inputs return NaN; ppf also
    returns NaN for probabilities outside ``[0, 1]``.

    Base composite costs are ``315*n`` for pdf, ``844*n`` for cdf, and
    ``1392*n`` for ppf, where
    ``n = max(numel(broadcast(input, a, b, loc, scale)), 1)``. These are
    analytical numerical upper bounds (FMA=2, stats weight 1.0), not hardware
    calibrations. Configured dtype pricing applies to the float64 output.
    The kernels use stable log-tail probabilities, eight fixed quadrature
    nodes for narrow intervals, and four fixed inverse-refinement steps.
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
        Base FLOP cost: ``315 * max(numel(broadcast(x, a, b, loc, scale)), 1)``.
        Analytical numerical upper bound, FMA=2, weight 1.0, before the
        configured float64 dtype multiplier.

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-0.5, 0.0, 0.5])
        >>> np.round(flops.stats.truncnorm.pdf(x, a=-1.0, b=1.0), 3)
        array([0.516, 0.584, 0.516])
        """
        return self._deduct_and_call("pdf", _PDF_COST, x, a, b, loc=loc, scale=scale)

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
        Base FLOP cost: ``844 * max(numel(broadcast(x, a, b, loc, scale)), 1)``.
        Analytical numerical upper bound, FMA=2, weight 1.0, before the
        configured float64 dtype multiplier.

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-0.5, 0.0, 0.5])
        >>> np.round(flops.stats.truncnorm.cdf(x, a=-1.0, b=1.0), 3)
        array([0.22, 0.5 , 0.78])
        """
        return self._deduct_and_call("cdf", _CDF_COST, x, a, b, loc=loc, scale=scale)

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
        Base FLOP cost: ``1392 * max(numel(broadcast(q, a, b, loc, scale)), 1)``.
        Analytical numerical upper bound, FMA=2, weight 1.0, before the
        configured float64 dtype multiplier.

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> q = np.array([0.25, 0.5, 0.75])
        >>> np.round(flops.stats.truncnorm.ppf(q, a=-1.0, b=1.0), 3)
        array([-0.442,  0.   ,  0.442])
        """
        return self._deduct_and_call("ppf", _PPF_COST, q, a, b, loc=loc, scale=scale)

    def _compute_pdf(self, x, a, b, loc=0, scale=1):
        return _truncnorm_kernels.pdf(x, a, b, loc=loc, scale=scale)

    def _compute_cdf(self, x, a, b, loc=0, scale=1):
        return _truncnorm_kernels.cdf(x, a, b, loc=loc, scale=scale)

    def _compute_ppf(self, q, a, b, loc=0, scale=1):
        return _truncnorm_kernels.ppf(q, a, b, loc=loc, scale=scale)


truncnorm = TruncnormDistribution()
