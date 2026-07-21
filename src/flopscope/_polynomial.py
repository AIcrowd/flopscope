"""Counted polynomial operations for flopscope."""

from __future__ import annotations

import inspect as _inspect
import math as _math
from typing import Any

import numpy as _np
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._dtype_billing import linalg_compute_dtype, resolve_billing_dtype
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
    """Cost for polyadd: max(n1, n2) FLOPs.

    Legacy length-based formula, kept for the public ``flops.polyadd_cost``
    analytical-cost API (1-D coefficient arrays, where axis-0 length *is*
    the element count). The ``polyadd`` wrapper itself bills via
    ``_polyaddsub_cost``, which also covers higher-rank/broadcasting inputs
    -- see that function's docstring for why axis-0 length alone is not
    enough there.
    """
    return max(n1, n2, 1)


def polysub_cost(n1: int, n2: int) -> int:
    """Cost for polysub: max(n1, n2) FLOPs.

    Legacy length-based formula, kept for the public ``flops.polysub_cost``
    analytical-cost API (1-D coefficient arrays, where axis-0 length *is*
    the element count). The ``polysub`` wrapper itself bills via
    ``_polyaddsub_cost``, which also covers higher-rank/broadcasting inputs
    -- see that function's docstring for why axis-0 length alone is not
    enough there.
    """
    return max(n1, n2, 1)


def _polyaddsub_cost(a1: ArrayLike, a2: ArrayLike) -> int:
    """Exact FLOP cost for polyadd/polysub: the true size of numpy's result.

    ``numpy.polyadd``/``numpy.polysub`` share this recipe (see
    ``numpy/lib/_polynomial_impl.py``)::

        a1 = atleast_1d(a1); a2 = atleast_1d(a2)
        diff = len(a2) - len(a1)                      # len() reads axis-0 size
        if diff == 0:
            val = a1 + a2                              # plain broadcast, ANY rank
        elif diff > 0:
            val = concatenate((zeros(diff), a1)) + a2  # requires a1 to be 1-D
        else:
            val = a1 + concatenate((zeros(-diff), a2)) # requires a2 to be 1-D

    A naive "zero-pad axis 0 to max(len), then broadcast the trailing axes"
    model gets two things wrong here:

    * When axis-0 lengths already match (``diff == 0``), numpy does a
      *plain* ``a1 + a2`` over the FULL shapes. Broadcasting aligns from the
      trailing axis, not from axis 0 -- if the two inputs have different
      rank this is not "axis-0 aligned, trailing broadcast" at all (e.g. a
      1-D array of the same axis-0 length as a 2-D array's axis 0 does not
      broadcast against it the way trailing-axis alignment would suggest).
    * When axis-0 lengths differ, the zero-pad array numpy builds is always
      1-D, so ``concatenate`` only succeeds if the shorter-by-axis-0 operand
      is itself exactly 1-D (a 0-d/scalar operand qualifies once promoted by
      ``atleast_1d``) -- anything higher-rank raises. When it *does*
      succeed, the padded 1-D array is broadcast (from the trailing axis)
      against the other operand's full shape, which can blow the result
      size up far past ``max(len(a1), len(a2))``: e.g.
      ``polyadd(5.0, ones((1000, 1)))`` pads the scalar to a length-1000
      vector and broadcasts it against a (1000, 1) array, yielding a
      (1000, 1000) result (1e6 elements) for a call a naive axis-0-only
      formula would bill at 1000 -- a 1000x under-bill.

    This mirrors that exact algorithm on shapes alone (no data is
    allocated), so the billed cost always equals
    ``numpy.polyadd(a1, a2).size`` for every input numpy accepts, and raises
    ``ValueError`` in precisely the cases numpy itself rejects (verified by
    fuzzing both against real numpy; see tests/test_batch_underbill_pins.py).
    """
    a1 = _np.asarray(a1)
    a2 = _np.asarray(a2)
    s1 = a1.shape if a1.ndim >= 1 else (1,)
    s2 = a2.shape if a2.ndim >= 1 else (1,)
    diff = s2[0] - s1[0]
    if diff == 0:
        result_shape = _np.broadcast_shapes(s1, s2)
    elif diff > 0:
        if len(s1) != 1:
            raise ValueError(
                f"polyadd/polysub: a1 with shape {a1.shape} cannot be "
                "zero-padded against a longer a2 (numpy's concatenate "
                "requires the shorter-by-axis-0 operand to be 1-D)"
            )
        result_shape = _np.broadcast_shapes((s2[0],), s2)
    else:
        if len(s2) != 1:
            raise ValueError(
                f"polyadd/polysub: a2 with shape {a2.shape} cannot be "
                "zero-padded against a longer a1 (numpy's concatenate "
                "requires the shorter-by-axis-0 operand to be 1-D)"
            )
        result_shape = _np.broadcast_shapes(s1, (s1[0],))
    return max(int(_math.prod(result_shape)), 1)


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


def polyfit_cost(m: int, deg: int, ncols: int = 1) -> int:
    """Cost for polyfit: 2 * m * (deg+1)^2 * ncols FLOPs.

    ``ncols`` is the number of right-hand-side columns being fit: 1 for a
    1-D ``y`` (the usual case), or ``y.shape[1]`` when ``y`` is 2-D (numpy
    solves one independent least-squares system per column, so the work
    scales linearly with it). Defaults to 1 so every pre-existing 1-D-``y``
    caller is bit-identical to the old ``2 * m * (deg+1)^2`` formula.
    """
    return max(2 * m * (deg + 1) ** 2 * ncols, 1)


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
    Includes runtime scaling; see docs/reference/cost-model.md."""
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
    cost = _polyaddsub_cost(a1, a2)
    with budget.deduct(
        "polyadd",
        flop_cost=cost,
        subscripts=None,
        shapes=(a1.shape, a2.shape),
        dtypes=(a1.dtype, a2.dtype),
    ):
        result = _call_numpy(_np.polyadd, a1, a2)
    return result  # type: ignore[return-value]


attach_docstring(
    polyadd,
    _np.polyadd,
    "counted_custom",
    "size of the axis-0 zero-padded, broadcast-aligned add result "
    "(== numpy.polyadd(a1, a2).size; reduces to max(n1, n2) for 1-D inputs)",
)


@_counted_wrapper
def polysub(a1: ArrayLike, a2: ArrayLike) -> FlopscopeArray:
    """Subtract two polynomials. Wraps ``numpy.polysub``."""
    budget = require_budget()
    a1 = _np.asarray(a1)
    a2 = _np.asarray(a2)
    cost = _polyaddsub_cost(a1, a2)
    with budget.deduct(
        "polysub",
        flop_cost=cost,
        subscripts=None,
        shapes=(a1.shape, a2.shape),
        dtypes=(a1.dtype, a2.dtype),
    ):
        result = _call_numpy(_np.polysub, a1, a2)
    return result  # type: ignore[return-value]


attach_docstring(
    polysub,
    _np.polysub,
    "counted_custom",
    "size of the axis-0 zero-padded, broadcast-aligned subtract result "
    "(== numpy.polysub(a1, a2).size; reduces to max(n1, n2) for 1-D inputs)",
)


@_counted_wrapper
def polyder(p: ArrayLike, m: int = 1) -> FlopscopeArray:
    """Differentiate a polynomial. Wraps ``numpy.polyder``."""
    budget = require_budget()
    p = _np.asarray(p)
    n = len(p)
    cost = polyder_cost(n, int(m))
    # np.polyder multiplies each coefficient by an arange-derived exponent
    # array, whose dtype is the platform default int (int64 on 64-bit
    # platforms) -- so the billed dtype is p's dtype promoted against that
    # default int, exactly like numpy's own arithmetic (verified:
    # polyder(int8/16/32/64) -> int64, polyder(float32) -> float64,
    # polyder(complex64) -> complex128; result_type(X, platform_int)
    # reproduces every case).
    billing_dtypes = (p.dtype, _np.dtype(_np.int_))
    with budget.deduct(
        "polyder",
        flop_cost=cost,
        subscripts=None,
        shapes=(p.shape,),
        dtypes=billing_dtypes,
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
    # complex factor is bypassed. np.polyint's coefficient division also
    # always runs in (at least) float64 -- even from float32 p (verified:
    # polyint(float32).dtype == float64) and complex64 -> complex128 -- so a
    # float64 sentinel joins the resolve too (kind-preserving, see polyfit).
    billing_dtypes = (p.dtype, _np.dtype(_np.float64))
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
    # np.polydiv's per-step scale-divide always runs in (at least) float64
    # for integer/bool operands (verified: polydiv(int, int) -> float64) but
    # preserves float32/complex64 precision otherwise (polydiv(f32, f32) ->
    # float32, polydiv(c64, c64) -> c64) -- combine u/v's own promotion
    # first, then apply the same kind-conditional float64 floor as an
    # eigvals-backed linalg op (int/bool -> float64, float/complex kept).
    combined = resolve_billing_dtype((u.dtype, v.dtype))
    billing_dtypes = (
        linalg_compute_dtype(combined) if combined is not None else u.dtype,
    )
    with budget.deduct(
        "polydiv",
        flop_cost=cost,
        subscripts=None,
        shapes=(u.shape, v.shape),
        dtypes=billing_dtypes,
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
    # numpy.polyfit accepts only 1-D or 2-D y (anything else is rejected by
    # numpy itself downstream); a 2-D y fits one independent least-squares
    # system per column, so the solve cost scales with ncols, not just m.
    ncols = y_arr.shape[1] if y_arr.ndim == 2 else 1
    cost = polyfit_cost(m, deg, ncols)
    # np.polyfit's least-squares solve always runs in (at least) float64 --
    # even from float32 x/y (verified: polyfit(f32, f32, deg).dtype ==
    # float64) -- and complex64 y computes complex128, so the float64
    # sentinel must join the resolve rather than replace it (result_type
    # preserves the complex/real kind: result_type(complex64, float64) ==
    # complex128, not float64).
    billing_dtypes = (x_arr.dtype, y_arr.dtype, _np.dtype(_np.float64))
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


attach_docstring(
    polyfit,
    _np.polyfit,
    "counted_custom",
    "2 * m * (deg+1)^2 * ncols FLOPs (ncols = y.shape[1] for 2-D y, else 1)",
)
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
    # The 2-D branch delegates to eigvals (LAPACK _commonType: int/bool ->
    # float64, float/complex kept), same as roots. The 1-D branch instead
    # casts through np.mintypecode, whose *default* typeset is only
    # {float32, float64, complex64, complex128} -- anything else, including
    # integers AND float16 (verified: poly(float16_roots) -> float64, NOT
    # float16), falls back to float64. linalg_compute_dtype already covers
    # every case except float16, which it would otherwise leave unchanged.
    poly_dtype = linalg_compute_dtype(seq.dtype)
    if poly_dtype.kind == "f" and poly_dtype.itemsize < 4:
        poly_dtype = _np.dtype(_np.float64)
    with budget.deduct(
        "poly",
        flop_cost=cost,
        subscripts=None,
        shapes=(seq.shape,),
        dtypes=(poly_dtype,),
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
    # np.roots delegates to np.linalg.eigvals on the companion matrix, so its
    # compute dtype follows the exact same LAPACK _commonType rule: integer/
    # bool input always computes in float64, float/complex input keeps its
    # own precision (verified: roots(int32) -> float64, roots(float32) ->
    # float32, roots(complex64) -> complex64 -- NOT widened to complex128).
    with budget.deduct(
        "roots",
        flop_cost=cost,
        subscripts=None,
        shapes=(p.shape,),
        dtypes=(linalg_compute_dtype(p.dtype),),
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
