"""Base support for SciPy-compatible continuous distributions."""

from __future__ import annotations

import math as _math
import warnings as _warnings

import numpy as _np

from flopscope._budget import _counted_wrapper, refuse_non_numeric_source
from flopscope._ndarray import _asflopscope
from flopscope._validation import require_budget
from flopscope.errors import FlopscopeWarning, _user_stacklevel


def _warn_float64_promotion(op_name: str, source_dtype: _np.dtype) -> None:
    """Warn that a narrow real floating input will be promoted to float64."""
    dtype_name = source_dtype.name
    _warnings.warn(
        f"{op_name} promoted its {dtype_name} input to float64 to match "
        "scipy.stats. If float64 output was not intended, cast the result back "
        f"with result.astype(np.{dtype_name}) before downstream operations; "
        f"float64 operations are billed at twice the {dtype_name} dtype rate.",
        FlopscopeWarning,
        stacklevel=_user_stacklevel(),
    )


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
        op_name = f"stats.{self._name}.{method}"
        # x is about to be cast to float64 below, and every distribution
        # parameter (loc/scale/a/b/s/...) is broadcast against it by the
        # pure-numpy _compute_* kernel -- both cast/coerce a payload's
        # __float__ per element with nothing billed for it. Probe each one
        # first; a probe never casts.
        refuse_non_numeric_source(op_name, x)
        for _v in args:
            refuse_non_numeric_source(op_name, _v)
        for _v in kwargs.values():
            refuse_non_numeric_source(op_name, _v)
        source_x = _np.asarray(x)
        if source_x.dtype.kind == "f" and source_x.dtype.itemsize < _np.dtype(
            _np.float64
        ).itemsize:
            _warn_float64_promotion(op_name, source_x.dtype)
        x = _np.asarray(source_x, dtype=_np.float64)
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
