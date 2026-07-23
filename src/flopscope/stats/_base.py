"""Base support for SciPy-compatible continuous distributions."""

from __future__ import annotations

import math as _math

import numpy as _np

from flopscope._budget import _counted_wrapper
from flopscope._ndarray import _asflopscope
from flopscope._validation import require_budget


class ContinuousDistribution:
    """Base class for FLOP-counted continuous distributions.

    Parameters
    ----------
    name : str
        Distribution name used to construct operation labels such as
        ``"stats.norm.pdf"`` in the budget log.

    Notes
    -----
    Subclasses implement ``_compute_pdf``, ``_compute_cdf``, and
    ``_compute_ppf`` as pure NumPy kernels. Public ``pdf``, ``cdf``, and
    ``ppf`` methods should delegate through :meth:`_deduct_and_call` so that
    budget deduction and ``FlopscopeArray`` wrapping stay consistent across
    the stats surface.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @_counted_wrapper
    def _deduct_and_call(self, method: str, cost_per_elem: int, x, *args, **kwargs):
        """Deduct FLOPs then call the pure-numpy implementation.

        Parameters
        ----------
        method : str
            Method name for budget logging, e.g. ``"pdf"``.
        cost_per_elem : int
            Flat FLOP cost per output element.
        x : array_like
            Primary input array (broadcast with the distribution parameters,
            it determines the output size).
        *args, **kwargs
            Distribution parameters (numeric array_likes), forwarded to
            ``_compute_{method}``.

        Returns
        -------
        FlopscopeArray
            Result returned by the matching ``_compute_{method}``
            implementation after budget deduction.

        Notes
        -----
        The deducted FLOP charge is ``cost_per_elem * max(numel(out), 1)``
        where ``out`` is the broadcast of ``x`` with every array-valued
        distribution parameter. Array ``loc``/``scale`` (or ``a``/``b``/``s``)
        broadcast the output larger than ``x``, so charging on ``x`` alone
        would undercount.
        """
        budget = require_budget()
        x = _np.asarray(x, dtype=_np.float64)
        param_shapes = tuple(
            _np.shape(v) for v in (*args, *kwargs.values()) if v is not None
        )
        try:
            out_shape = _np.broadcast_shapes(x.shape, *param_shapes)
        except ValueError:
            # Incompatible shapes: keep the old x-based charge and let the
            # compute call below raise numpy's own broadcast error -- billing
            # must not change the exception surface for invalid calls.
            out_shape = x.shape
        n = max(_math.prod(out_shape), 1)
        op_name = f"stats.{self._name}.{method}"
        compute_fn = getattr(self, f"_compute_{method}")
        with budget.deduct(
            op_name,
            flop_cost=cost_per_elem * n,
            subscripts=None,
            shapes=(out_shape,),
            dtypes=(x.dtype,),
        ):
            result = compute_fn(x, *args, **kwargs)
        return _asflopscope(result)

    def __repr__(self) -> str:
        return f"<flopscope.stats.{self._name}>"
