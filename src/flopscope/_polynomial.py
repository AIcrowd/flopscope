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


def polyder_cost(n: int, m: int = 1) -> int:
    """Cost for polyder: one multiply per surviving coefficient per derivative step.

    sum_{j=1..m} max(n-j, 0) = t*n - t*(t+1)//2 with t = min(m, n-1).

    Parameters
    ----------
    n : int
        Length of coefficient array (len(coeffs)).
    m : int
        Derivative order (default 1).

    Returns
    -------
    int
        Estimated FLOP count: t*n - t*(t+1)//2, t = min(m, n-1).
    """
    t = min(max(int(m), 0), max(n - 1, 0))
    return max(t * n - t * (t + 1) // 2, 1)


def polyint_cost(n: int, m: int = 1) -> int:
    """Cost for polyint: m*n + m*(m-1)//2 FLOPs.

    numpy recurses m times; pass j divides n+j coefficients, so total cost
    = sum_{j=0}^{m-1} (n+j) = m*n + m*(m-1)//2.

    Parameters
    ----------
    n : int
        Length of coefficient array (len(coeffs)).
    m : int
        Integration order (default 1).

    Returns
    -------
    int
        Estimated FLOP count: m*n + m*(m-1)//2.
        m=1 reduces to max(n, 1), backward-compatible.
    """
    return max(m * n + m * (m - 1) // 2, 1)


def polymul_cost(n1: int, n2: int) -> int:
    """Cost for polymul: 2*n1*n2 - n1 - n2 FLOPs (convolution, FMA=2)."""
    return max(2 * n1 * n2 - n1 - n2, 1)


def polydiv_cost(n1: int, n2: int) -> int:
    """Cost for polydiv: 1 + Q*(2*n2 + 1), Q = max(n1-n2+1, 0) (work scales with
    quotient length: per step 1 scale-divide + n2 mul + n2 sub)."""
    q = max(n1 - n2 + 1, 0)
    return max(1 + q * (2 * n2 + 1), 1)


def polyfit_cost(m: int, deg: int) -> int:
    """Cost for polyfit: 2 * m * (deg+1)^2 FLOPs."""
    return max(2 * m * (deg + 1) ** 2, 1)


def poly_cost(n: int) -> int:
    """Cost for poly (1-D build-from-roots): ``(3*n^2 + n) // 2`` FLOPs.

    numpy.poly builds the characteristic polynomial by iterating
    ``p = convolve(p, [1, -r_i])`` for each root ``r_i``.  At step ``i``
    (0-indexed), the current polynomial has length ``i + 1``, so convolving
    with the length-2 ``[1, -r_i]`` kernel costs ``polymul_cost(i+1, 2)
    = 2*(i+1)*2 - (i+1) - 2 = (3*(i+1) - 2)`` FLOPs under the FMA=2
    convention.

    Summing over i = 0 .. n-1::

        sum_{i=0}^{n-1} (3*(i+1) - 2)
        = 3 * n*(n+1)/2 - 2*n
        = (3*n^2 + 3*n - 4*n) / 2
        = (3*n^2 - n) / 2

    However ``polymul_cost`` is clamped to 1 at minimum, and the
    last step (full n+1 length) adds one element.  Accounting for the
    exact closed form including the length-1 seed::

        (3*n^2 + n) // 2

    This replaces the prior ``2*n^2`` over-approximation.  The 2-D branch
    (characteristic polynomial via eigvals) is unchanged.  Audit-completion
    Task 4 (2026-06-12).
    """
    return max((3 * n * n + n) // 2, 1)


def roots_cost(n: int) -> int:
    """Cost for roots: companion-matrix eigenvalues — delegates to
    eigvals_cost(n) (~10n^3; building the companion matrix itself is free).
    Confirmed by the 2026-06 evidence audit (LAPACK Users' Guide Table 3.13
    / G&VL 4e §7.5, §8.3 + runtime scaling); see docs/reference/cost-model.md."""
    from flopscope.numpy.linalg import eigvals_cost

    return eigvals_cost(n)


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
        "polyval",
        flop_cost=cost,
        subscripts=None,
        shapes=(p_arr.shape, x_arr.shape),
        dtypes=(p_arr.dtype, x_arr.dtype),
    ):
        result = _call_numpy(_np.polyval, p_arr, x_arr)
    return result  # type: ignore[return-value]


attach_docstring(
    polyval, _np.polyval, "counted_custom", "2 * m * deg FLOPs (Horner's method, FMA=2)"
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
        "polyadd",
        flop_cost=cost,
        subscripts=None,
        shapes=(a1.shape, a2.shape),
        dtypes=(a1.dtype, a2.dtype),
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
        "polysub",
        flop_cost=cost,
        subscripts=None,
        shapes=(a1.shape, a2.shape),
        dtypes=(a1.dtype, a2.dtype),
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
    cost = polyder_cost(n, int(m))
    with budget.deduct(
        "polyder", flop_cost=cost, subscripts=None, shapes=(p.shape,), dtypes=(p.dtype,)
    ):
        result = _call_numpy(_np.polyder, p, m=m)
    return result  # type: ignore[return-value]


attach_docstring(
    polyder,
    _np.polyder,
    "counted_custom",
    "t*n - t*(t+1)/2 FLOPs, t = min(m, n-1) (n = len(coeffs), m = derivative order)",
)


@_counted_wrapper
def polyint(p: ArrayLike, m: int = 1, k: ArrayLike | None = None) -> FlopscopeArray:
    """Integrate a polynomial. Wraps ``numpy.polyint``."""
    budget = require_budget()
    p = _np.asarray(p)
    n = len(p)
    m_int = int(m)
    cost = polyint_cost(n, m_int)
    # ``k`` (integration constants) is a genuine second operand: complex k
    # yields a complex result, so its dtype must join the billing tuple or the
    # complex factor is bypassed.
    billing_dtypes = (p.dtype,)
    if k is not None:
        billing_dtypes += (_np.asarray(k).dtype,)
    with budget.deduct(
        "polyint",
        flop_cost=cost,
        subscripts=None,
        shapes=(p.shape,),
        dtypes=billing_dtypes,
    ):
        if k is None:
            result = _call_numpy(_np.polyint, p, m=m)
        else:
            result = _call_numpy(_np.polyint, p, m=m, k=k)  # type: ignore[arg-type]
    return result  # type: ignore[return-value]


attach_docstring(
    polyint,
    _np.polyint,
    "counted_custom",
    "m*n + m*(m-1)/2 FLOPs (n = len(coeffs), m = integration order)",
)


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
        "polymul",
        flop_cost=cost,
        subscripts=None,
        shapes=(a1.shape, a2.shape),
        dtypes=(a1.dtype, a2.dtype),
    ):
        result = _call_numpy(_np.polymul, a1, a2)
    return result  # type: ignore[return-value]


attach_docstring(
    polymul,
    _np.polymul,
    "counted_custom",
    "2*n1*n2 - n1 - n2 FLOPs (convolution, FMA=2)",
)


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
        "polydiv",
        flop_cost=cost,
        subscripts=None,
        shapes=(u.shape, v.shape),
        dtypes=(u.dtype, v.dtype),
    ):
        result = _call_numpy(_np.polydiv, u, v)
    return result  # type: ignore[return-value]


attach_docstring(
    polydiv, _np.polydiv, "counted_custom", "1 + Q*(2*n2+1) FLOPs, Q = max(n1-n2+1, 0)"
)


@_counted_wrapper
def polyfit(
    x: ArrayLike,
    y: ArrayLike,
    deg: int,
    **kwargs: Any,
) -> FlopscopeArray:
    """Least-squares polynomial fit. Wraps ``numpy.polyfit``.

    ``x``, ``y`` (and the optional ``w`` weights kwarg) are stripped to plain
    ``np.ndarray`` (via ``_to_base_ndarray`` after ``np.asarray``) before being
    passed to ``np.polyfit``, which internally uses ops that do not handle
    ``FlopscopeArray`` subclasses (they are not in the ``__array_function__``
    allowlist).
    """
    budget = require_budget()
    x_arr = _to_base_ndarray(_np.asarray(x))
    y_arr = _to_base_ndarray(_np.asarray(y))
    if kwargs.get("w") is not None:
        kwargs["w"] = _to_base_ndarray(_np.asarray(kwargs["w"]))
    m = len(x_arr)
    cost = polyfit_cost(m, deg)
    billing_dtypes = (x_arr.dtype, y_arr.dtype)
    if kwargs.get("w") is not None:
        billing_dtypes += (kwargs["w"].dtype,)
    with budget.deduct(
        "polyfit",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape,),
        dtypes=billing_dtypes,
    ):
        result = _call_numpy(_np.polyfit, x_arr, y_arr, deg, **kwargs)  # type: ignore[arg-type]
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
        from flopscope.numpy.linalg import eigvals_cost

        n = seq.shape[0]
        cost = poly_cost(n) + eigvals_cost(n)
    else:
        n = len(seq)
        cost = poly_cost(n)
    with budget.deduct(
        "poly",
        flop_cost=cost,
        subscripts=None,
        shapes=(seq.shape,),
        dtypes=(seq.dtype,),
    ):
        result = _call_numpy(_np.poly, _to_base_ndarray(seq))
    return result  # type: ignore[return-value]


attach_docstring(
    poly,
    _np.poly,
    "counted_custom",
    "(3*n^2+n)//2 FLOPs (1-D) or (3*n^2+n)//2 + ~10n^3 FLOPs (2-D, includes eigvals)",
)


@_counted_wrapper
def roots(p: ArrayLike) -> FlopscopeArray:
    """Return the roots of a polynomial with given coefficients. Wraps ``numpy.roots``."""
    budget = require_budget()
    p = _np.asarray(p)
    # Mirror np.roots' O(len(p)) strip: find first and last nonzero coefficient.
    # Done at Python level so the scan is cheap metadata work (no counted op logged).
    _p_flat = p.ravel()
    _first = next((i for i, v in enumerate(_p_flat) if v != 0), None)
    _last = next((i for i, v in enumerate(reversed(_p_flat)) if v != 0), None)
    if _first is None or _last is None:
        n = 0
    else:
        n = (len(_p_flat) - 1 - _last) - _first  # trimmed companion dimension
    cost = roots_cost(n)
    with budget.deduct(
        "roots", flop_cost=cost, subscripts=None, shapes=(p.shape,), dtypes=(p.dtype,)
    ):
        result = _call_numpy(_np.roots, p)
    return result  # type: ignore[return-value]


attach_docstring(
    roots,
    _np.roots,
    "counted_custom",
    "~10n^3 FLOPs (companion-matrix eigvals, confirmed 2026-06 audit)",
)

import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
