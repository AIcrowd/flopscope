"""Log-normal distribution for :mod:`flopscope.stats`."""

from __future__ import annotations

import numpy as _np

from flopscope.stats._base import ContinuousDistribution
from flopscope.stats._erf import _erf
from flopscope.stats._ndtri import _ndtri

_SQRT2 = _np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / _np.sqrt(2.0 * _np.pi)


class LognormDistribution(ContinuousDistribution):
    """Log-normal continuous random variable.

    This object mirrors ``scipy.stats.lognorm``.

    Methods
    -------
    pdf(x, s, loc=0, scale=1)
        Evaluate the probability density function.
    cdf(x, s, loc=0, scale=1)
        Evaluate the cumulative distribution function.
    ppf(q, s, loc=0, scale=1)
        Evaluate the percent-point function.

    Notes
    -----
    ``s`` is the shape parameter: the standard deviation of the underlying
    normal distribution. It is the first positional argument, ahead of
    ``loc`` and ``scale``, matching SciPy's ``lognorm`` signature. pdf
    deducts ``62 * numel(broadcast(input, s, loc, scale))`` FLOPs (composite:
    log + exp + arithmetic, weight 1.0; audit-2 verified; calibration alpha
    62.30). cdf deducts ``70 * numel(broadcast(input, s, loc, scale))`` FLOPs
    (composite: log + erf rational approx + arithmetic, weight 1.0; audit-2
    verified; calibration alpha 69.98). ppf deducts
    ``106 * numel(broadcast(input, s, loc, scale))`` FLOPs (composite:
    ndtri + exp, weight 1.0; audit-2 verified).
    """

    def __init__(self):
        super().__init__("lognorm")

    def pdf(self, x, s, loc=0, scale=1):
        """Evaluate the probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the density.
        s : float
            Shape parameter of the distribution.
        loc : float, optional
            Location parameter. Defaults to ``0``.
        scale : float, optional
            Scale parameter. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Probability density evaluated elementwise at ``x``.

        Notes
        -----
        Equivalent to ``scipy.stats.lognorm.pdf(x, s, loc, scale)``.
        FLOP cost: ``62 * numel(broadcast(x, s, loc, scale))`` (composite:
        log + exp + arithmetic, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([0.5, 1.0, 2.0])
        >>> np.round(flops.stats.lognorm.pdf(x, s=0.5), 3)
        array([0.61 , 0.798, 0.153])
        """
        return self._deduct_and_call("pdf", 62, x, s, loc=loc, scale=scale)

    def cdf(self, x, s, loc=0, scale=1):
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative probability.
        s : float
            Shape parameter of the distribution.
        loc : float, optional
            Location parameter. Defaults to ``0``.
        scale : float, optional
            Scale parameter. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Cumulative probability evaluated elementwise at ``x``.

        Notes
        -----
        Equivalent to ``scipy.stats.lognorm.cdf(x, s, loc, scale)``.
        FLOP cost: ``70 * numel(broadcast(x, s, loc, scale))`` (composite:
        log + erf rational approx + arithmetic, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> x = np.array([0.5, 1.0, 2.0])
        >>> np.round(flops.stats.lognorm.cdf(x, s=0.5), 3)
        array([0.083, 0.5  , 0.917])
        """
        return self._deduct_and_call("cdf", 70, x, s, loc=loc, scale=scale)

    def ppf(self, q, s, loc=0, scale=1):
        """Evaluate the percent-point function.

        Parameters
        ----------
        q : array_like
            Probabilities in ``[0, 1]``.
        s : float
            Shape parameter of the distribution.
        loc : float, optional
            Location parameter. Defaults to ``0``.
        scale : float, optional
            Scale parameter. Defaults to ``1``.

        Returns
        -------
        FlopscopeArray
            Quantiles corresponding to ``q``.

        Notes
        -----
        Equivalent to ``scipy.stats.lognorm.ppf(q, s, loc, scale)``.
        FLOP cost: ``106 * numel(broadcast(q, s, loc, scale))`` (composite:
        ndtri + exp, weight 1.0).

        Examples
        --------
        >>> import numpy as np
        >>> import flopscope as flops
        >>> q = np.array([0.25, 0.5, 0.75])
        >>> np.round(flops.stats.lognorm.ppf(q, s=0.5), 3)
        array([0.714, 1.   , 1.401])
        """
        return self._deduct_and_call("ppf", 106, q, s, loc=loc, scale=scale)

    def _compute_pdf(self, x, s, loc=0, scale=1):
        y = (x - loc) / scale
        safe_y = _np.where(y > 0, y, 1.0)  # avoid log(0)
        lny = _np.log(safe_y)
        result = _INV_SQRT_2PI / (s * safe_y * scale) * _np.exp(-0.5 * (lny / s) ** 2)
        return _np.where(y > 0, result, 0.0)

    def _compute_cdf(self, x, s, loc=0, scale=1):
        y = (x - loc) / scale
        safe_y = _np.where(y > 0, y, 1.0)
        z = _np.log(safe_y) / (s * _SQRT2)
        result = 0.5 * (1.0 + _erf(z))
        return _np.where(y > 0, result, 0.0)

    def _compute_ppf(self, q, s, loc=0, scale=1):
        return loc + scale * _np.exp(s * _ndtri(q))


lognorm = LognormDistribution()
