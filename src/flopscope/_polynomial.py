"""Counted polynomial operations for flopscope."""

from __future__ import annotations

import inspect as _inspect
from typing import Any

import numpy as _np
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._ndarray import FlopscopeArray, _to_base_ndarray
from flopscope._validation import require_budget

# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------


def polyval_cost(deg: int, m: int) -> int:
    """Cost for polyval: Horner's method under FMA=2 textbook convention.

    Per coefficient: 1 multiply + 1 add (FMA=2). m output cells, deg coefficients.
    Returns 2 * m * deg FLOPs.
    """
    return max(2 * m * deg, 1)


def polyadd_cost(n1: int, n2: int) -> int:
    """Cost for polyadd: max(n1, n2) FLOPs."""
    return max(n1, n2, 1)


def polysub_cost(n1: int, n2: int) -> int:
    """Cost for polysub: max(n1, n2) FLOPs."""
    return max(n1, n2, 1)


def polyder_cost(n: int) -> int:
    """Cost for polyder: n FLOPs (n = len of coeffs)."""
    return max(n, 1)


def polyint_cost(n: int) -> int:
    """Cost for polyint: n FLOPs (n = len of coeffs)."""
    return max(n, 1)


def polymul_cost(n1: int, n2: int) -> int:
    """Cost for polymul: n1 * n2 FLOPs."""
    return max(n1 * n2, 1)


def polydiv_cost(n1: int, n2: int) -> int:
    """Cost for polydiv: n1 * n2 FLOPs."""
    return max(n1 * n2, 1)


def polyfit_cost(m: int, deg: int) -> int:
    """Cost for polyfit: 2 * m * (deg+1)^2 FLOPs."""
    return max(2 * m * (deg + 1) ** 2, 1)


def poly_cost(n: int) -> int:
    """Cost for poly: $n^2$ FLOPs."""
    return max(n * n, 1)


def roots_cost(n: int) -> int:
    """Cost for roots: $n^3$ FLOPs (companion matrix eigendecomposition, simplified)."""
    return max(n**3, 1)


# ---------------------------------------------------------------------------
# Wrapped operations
# ---------------------------------------------------------------------------


@_counted_wrapper
def polyval(p: ArrayLike, x: ArrayLike) -> FlopscopeArray:
    """Evaluate a polynomial at given points. Wraps ``numpy.polyval``.

    Both ``p`` and ``x`` are converted to plain ``np.ndarray`` (via
    ``_to_base_ndarray`` after ``np.asarray``) before being passed to
    ``_np.polyval``, because numpy's polyval implementation internally calls
    ``np.zeros_like(x)`` and other ops that do not handle
    ``FlopscopeArray`` subclasses (they are not in the
    ``__array_function__`` allowlist).
    """
    budget = require_budget()
    p_arr = _to_base_ndarray(_np.asarray(p))
    x_arr = _to_base_ndarray(_np.asarray(x))
    deg = len(p_arr) - 1
    m = x_arr.size
    cost = polyval_cost(deg, m)
    with budget.deduct(
        "polyval", flop_cost=cost, subscripts=None, shapes=(p_arr.shape, x_arr.shape)
    ):
        result = _call_numpy(_np.polyval, p_arr, x_arr)
    return result  # type: ignore[return-value]


attach_docstring(
    polyval, _np.polyval, "counted_custom", "m * deg FLOPs (Horner's method, FMA=1)"
)


@_counted_wrapper
def polyadd(a1: ArrayLike, a2: ArrayLike) -> FlopscopeArray:
    """Add two polynomials. Wraps ``numpy.polyadd``."""
    budget = require_budget()
    a1 = _np.asarray(a1)
    a2 = _np.asarray(a2)
    n1 = len(a1)
    n2 = len(a2)
    cost = polyadd_cost(n1, n2)
    with budget.deduct(
        "polyadd", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape)
    ):
        result = _call_numpy(_np.polyadd, a1, a2)
    return result  # type: ignore[return-value]


attach_docstring(polyadd, _np.polyadd, "counted_custom", "max(n1, n2) FLOPs")


@_counted_wrapper
def polysub(a1: ArrayLike, a2: ArrayLike) -> FlopscopeArray:
    """Subtract two polynomials. Wraps ``numpy.polysub``."""
    budget = require_budget()
    a1 = _np.asarray(a1)
    a2 = _np.asarray(a2)
    n1 = len(a1)
    n2 = len(a2)
    cost = polysub_cost(n1, n2)
    with budget.deduct(
        "polysub", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape)
    ):
        result = _call_numpy(_np.polysub, a1, a2)
    return result  # type: ignore[return-value]


attach_docstring(polysub, _np.polysub, "counted_custom", "max(n1, n2) FLOPs")


@_counted_wrapper
def polyder(p: ArrayLike, m: int = 1) -> FlopscopeArray:
    """Differentiate a polynomial. Wraps ``numpy.polyder``."""
    budget = require_budget()
    p = _np.asarray(p)
    n = len(p)
    cost = polyder_cost(n)
    with budget.deduct("polyder", flop_cost=cost, subscripts=None, shapes=(p.shape,)):
        result = _call_numpy(_np.polyder, p, m=m)
    return result  # type: ignore[return-value]


attach_docstring(polyder, _np.polyder, "counted_custom", "n FLOPs (n = len(coeffs))")


@_counted_wrapper
def polyint(p: ArrayLike, m: int = 1, k: ArrayLike | None = None) -> FlopscopeArray:
    """Integrate a polynomial. Wraps ``numpy.polyint``."""
    budget = require_budget()
    p = _np.asarray(p)
    n = len(p)
    cost = polyint_cost(n)
    with budget.deduct("polyint", flop_cost=cost, subscripts=None, shapes=(p.shape,)):
        if k is None:
            result = _call_numpy(_np.polyint, p, m=m)
        else:
            result = _call_numpy(_np.polyint, p, m=m, k=k)  # type: ignore[arg-type]
    return result  # type: ignore[return-value]


attach_docstring(polyint, _np.polyint, "counted_custom", "n FLOPs (n = len(coeffs))")


@_counted_wrapper
def polymul(a1: ArrayLike, a2: ArrayLike) -> FlopscopeArray:
    """Multiply polynomials. Wraps ``numpy.polymul``."""
    budget = require_budget()
    a1 = _np.asarray(a1)
    a2 = _np.asarray(a2)
    n1 = len(a1)
    n2 = len(a2)
    cost = polymul_cost(n1, n2)
    with budget.deduct(
        "polymul", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape)
    ):
        result = _call_numpy(_np.polymul, a1, a2)
    return result  # type: ignore[return-value]


attach_docstring(polymul, _np.polymul, "counted_custom", "n1 * n2 FLOPs")


@_counted_wrapper
def polydiv(u: ArrayLike, v: ArrayLike) -> tuple[FlopscopeArray, FlopscopeArray]:
    """Divide one polynomial by another. Wraps ``numpy.polydiv``."""
    budget = require_budget()
    u = _np.atleast_1d(_np.asarray(u))
    v = _np.atleast_1d(_np.asarray(v))
    n1 = len(u)
    n2 = len(v)
    cost = polydiv_cost(n1, n2)
    with budget.deduct(
        "polydiv", flop_cost=cost, subscripts=None, shapes=(u.shape, v.shape)
    ):
        result = _call_numpy(_np.polydiv, u, v)
    return result  # type: ignore[return-value]


attach_docstring(polydiv, _np.polydiv, "counted_custom", "n1 * n2 FLOPs")


@_counted_wrapper
def polyfit(
    x: ArrayLike,
    y: ArrayLike,
    deg: int,
    **kwargs: Any,
) -> FlopscopeArray:
    """Least-squares polynomial fit. Wraps ``numpy.polyfit``."""
    budget = require_budget()
    x = _np.asarray(x)
    m = len(x)
    cost = polyfit_cost(m, deg)
    with budget.deduct("polyfit", flop_cost=cost, subscripts=None, shapes=(x.shape,)):
        result = _call_numpy(_np.polyfit, x, y, deg, **kwargs)  # type: ignore[arg-type]
    return result  # type: ignore[return-value]


attach_docstring(polyfit, _np.polyfit, "counted_custom", "2 * m * (deg+1)^2 FLOPs")
polyfit.__signature__ = _inspect.signature(_np.polyfit)  # type: ignore[attr-defined]


@_counted_wrapper
def poly(seq_of_zeros: ArrayLike) -> FlopscopeArray:
    """Return polynomial coefficients from roots. Wraps ``numpy.poly``."""
    budget = require_budget()
    seq = _np.asarray(seq_of_zeros)
    # If 2D (square matrix), n = shape[0]; if 1D, n = len(seq)
    if seq.ndim == 2:
        n = seq.shape[0]
    else:
        n = len(seq)
    cost = poly_cost(n)
    with budget.deduct("poly", flop_cost=cost, subscripts=None, shapes=(seq.shape,)):
        result = _call_numpy(_np.poly, seq_of_zeros)
    return result  # type: ignore[return-value]


attach_docstring(poly, _np.poly, "counted_custom", "n^2 FLOPs")


@_counted_wrapper
def roots(p: ArrayLike) -> FlopscopeArray:
    """Return the roots of a polynomial with given coefficients. Wraps ``numpy.roots``."""
    budget = require_budget()
    p = _np.asarray(p)
    n = len(p) - 1  # degree = number of roots
    cost = roots_cost(n)
    with budget.deduct("roots", flop_cost=cost, subscripts=None, shapes=(p.shape,)):
        result = _call_numpy(_np.roots, p)
    return result  # type: ignore[return-value]


attach_docstring(
    roots, _np.roots, "counted_custom", "n^3 FLOPs (companion matrix eig, simplified)"
)

import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
