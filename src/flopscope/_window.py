# src/flopscope/_window.py
"""Window function wrappers with FLOP counting."""

from __future__ import annotations

import numpy as _np

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._ndarray import FlopscopeArray
from flopscope._validation import require_budget


def bartlett_cost(n: int) -> int:
    """FLOP cost of Bartlett window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: 4n (compare + divide + add + select per sample, FMA=2).

    Notes
    -----
    Four ops per sample: compare, divide, add/subtract, select (single branch of
    numpy where-based evaluation). Weight 1.0 (no transcendental; constant in flop_cost).
    """
    return max(4 * n, 1)


@_counted_wrapper
def bartlett(M: int) -> FlopscopeArray:
    budget = require_budget()
    cost = bartlett_cost(M)
    with budget.deduct(
        "bartlett",
        flop_cost=cost,
        subscripts=None,
        shapes=((M,),),
        # numpy window functions always return float64; bill that width.
        dtypes=(_np.dtype(_np.float64),),
    ):
        result = _call_numpy(_np.bartlett, M)
    return result  # type: ignore[return-value]


attach_docstring(bartlett, _np.bartlett, "counted_custom", "4n FLOPs (FMA=2)")


def blackman_cost(n: int) -> int:
    """FLOP cost of Blackman window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: 40n (composite: 2 cosine evals at transcendental rate 16
        + 8 mul/div/add per sample; the 0.42 term is a constant, not a cosine).

    Notes
    -----
    numpy's blackman: 0.42 + 0.5*cos(pi*n/(M-1)) + 0.08*cos(2*pi*n/(M-1)) — exactly
    two cosine evals per element. Weight 1.0 (constant in flop_cost per composite-kernel tier policy).
    """
    return max(40 * n, 1)


@_counted_wrapper
def blackman(M: int) -> FlopscopeArray:
    budget = require_budget()
    cost = blackman_cost(M)
    with budget.deduct(
        "blackman",
        flop_cost=cost,
        subscripts=None,
        shapes=((M,),),
        # numpy window functions always return float64; bill that width.
        dtypes=(_np.dtype(_np.float64),),
    ):
        result = _call_numpy(_np.blackman, M)
    return result  # type: ignore[return-value]


attach_docstring(
    blackman, _np.blackman, "counted_custom", "40n FLOPs (2 cos + 8 arith per sample)"
)


def hamming_cost(n: int) -> int:
    """FLOP cost of Hamming window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: 18n (cosine at the transcendental tier + multiply
        + subtract per sample).

    Notes
    -----
    Per sample: 1 cosine at the transcendental tier (16) + 1 multiply + 1
    subtract = 18. Weight 1.0 (constant in flop_cost).
    """
    return max(18 * n, 1)


@_counted_wrapper
def hamming(M: int) -> FlopscopeArray:
    budget = require_budget()
    cost = hamming_cost(M)
    with budget.deduct(
        "hamming",
        flop_cost=cost,
        subscripts=None,
        shapes=((M,),),
        # numpy window functions always return float64; bill that width.
        dtypes=(_np.dtype(_np.float64),),
    ):
        result = _call_numpy(_np.hamming, M)
    return result  # type: ignore[return-value]


attach_docstring(hamming, _np.hamming, "counted_custom", "18n FLOPs")


def hanning_cost(n: int) -> int:
    """FLOP cost of Hanning window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: 18n (cosine at the transcendental tier + multiply
        + subtract per sample).

    Notes
    -----
    Per sample: 1 cosine at the transcendental tier (16) + 1 multiply + 1
    subtract = 18. Weight 1.0 (constant in flop_cost).
    """
    return max(18 * n, 1)


@_counted_wrapper
def hanning(M: int) -> FlopscopeArray:
    budget = require_budget()
    cost = hanning_cost(M)
    with budget.deduct(
        "hanning",
        flop_cost=cost,
        subscripts=None,
        shapes=((M,),),
        # numpy window functions always return float64; bill that width.
        dtypes=(_np.dtype(_np.float64),),
    ):
        result = _call_numpy(_np.hanning, M)
    return result  # type: ignore[return-value]


attach_docstring(hanning, _np.hanning, "counted_custom", "18n FLOPs")


def kaiser_cost(n: int) -> int:
    """FLOP cost of Kaiser window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: 23n.

    Notes
    -----
    Per sample: 1 Bessel I0 eval at the transcendental tier (16) + 7 scalar FLOPs
    (sub, div, square, rsub, sqrt, mul-by-beta, final div) under FMA=2. Tracks the
    in-system price of i0 (numel x 16.0); revisit jointly with the pointwise family
    if i0 is ever re-derived to its Cephes cost. Weight 1.0 (constant in flop_cost).
    """
    return max(23 * n, 1)


@_counted_wrapper
def kaiser(M: int, beta: float) -> FlopscopeArray:
    budget = require_budget()
    cost = kaiser_cost(M)
    with budget.deduct(
        "kaiser",
        flop_cost=cost,
        subscripts=None,
        shapes=((M,),),
        # numpy window functions always return float64; bill that width.
        dtypes=(_np.dtype(_np.float64),),
    ):
        result = _call_numpy(_np.kaiser, M, beta)
    return result  # type: ignore[return-value]


attach_docstring(kaiser, _np.kaiser, "counted_custom", "23n FLOPs")

import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
