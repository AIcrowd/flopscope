"""Normal distribution for :mod:`flopscope.stats`."""

from __future__ import annotations

import numpy as np

from flopscope.stats._base import ContinuousDistribution
from flopscope.stats._erf import _erf
from flopscope.stats._ndtri import _ndtri

_SQRT2 = np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


class NormDistribution(ContinuousDistribution):
    """Normal (Gaussian) continuous random variable.

    This object mirrors ``scipy.stats.norm`` and uses the standard
    ``loc``/``scale`` parameterization.

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
    ``loc`` is the mean and ``scale`` is the standard deviation. Cost constants
    are composite (rational-approximation + transcendental kernels), weight 1.0:
    pdf=27 FLOPs/elem, cdf=48 FLOPs/elem, ppf=83 FLOPs/elem.
    """

    def __init__(self):
        super().__init__("norm")

    def pdf(self, x, loc=0, scale=1):
        """Evaluate the probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the density.
        loc : float, optional
            Mean of the distribution. Defaults to ``0``.
        scale : float, optional
            Standard deviation of the distribution. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Probability density evaluated elementwise at ``x``.

        Notes
        -----
        Equivalent to ``scipy.stats.norm.pdf(x, loc, scale)``.
        FLOP cost: ``27 * numel(broadcast(x, loc, scale))`` (composite:
        exp + arithmetic, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> np.round(flops.stats.norm.pdf(x), 3)
        array([0.242, 0.399, 0.242])
        """
        return self._deduct_and_call("pdf", 27, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative probability.
        loc : float, optional
            Mean of the distribution. Defaults to ``0``.
        scale : float, optional
            Standard deviation of the distribution. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Cumulative probability evaluated elementwise at ``x``.

        Notes
        -----
        Equivalent to ``scipy.stats.norm.cdf(x, loc, scale)``.
        FLOP cost: ``48 * numel(broadcast(x, loc, scale))`` (composite:
        erf rational approx + arithmetic, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> np.round(flops.stats.norm.cdf(x), 3)
        array([0.159, 0.5  , 0.841])
        """
        return self._deduct_and_call("cdf", 48, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        """Evaluate the percent-point function.

        Parameters
        ----------
        q : array_like
            Probabilities in ``[0, 1]``.
        loc : float, optional
            Mean of the distribution. Defaults to ``0``.
        scale : float, optional
            Standard deviation of the distribution. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Quantiles corresponding to ``q``.

        Notes
        -----
        Equivalent to ``scipy.stats.norm.ppf(q, loc, scale)``.
        FLOP cost: ``83 * numel(broadcast(q, loc, scale))`` (composite:
        Acklam rational ndtri + Newton polish with erf+exp, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> q = np.array([0.25, 0.5, 0.75])
        >>> np.round(flops.stats.norm.ppf(q), 3)
        array([-0.674,  0.   ,  0.674])
        """
        return self._deduct_and_call("ppf", 83, q, loc=loc, scale=scale)

    # --- Pure-NumPy implementations (no budget deduction) ---

    def _compute_pdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return _INV_SQRT_2PI / scale * np.exp(-0.5 * z * z)

    def _compute_cdf(self, x, loc=0, scale=1):
        z = (x - loc) / (scale * _SQRT2)
        return 0.5 * (1.0 + _erf(z))

    def _compute_ppf(self, q, loc=0, scale=1):
        return loc + scale * _ndtri(q)


norm = NormDistribution()
