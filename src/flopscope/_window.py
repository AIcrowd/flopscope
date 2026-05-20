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
        Estimated FLOP count: n.

    Notes
    -----
    One linear evaluation per sample.
    """
    return max(n, 1)


@_counted_wrapper
def bartlett(M: int) -> FlopscopeArray:
    budget = require_budget()
    cost = bartlett_cost(M)
    with budget.deduct("bartlett", flop_cost=cost, subscripts=None, shapes=((M,),)):
        result = _call_numpy(_np.bartlett, M)
    return result  # type: ignore[return-value]


attach_docstring(bartlett, _np.bartlett, "counted_custom", "n FLOPs")


def blackman_cost(n: int) -> int:
    """FLOP cost of Blackman window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: 3n.

    Notes
    -----
    Three cosine terms per sample.
    """
    return max(3 * n, 1)


@_counted_wrapper
def blackman(M: int) -> FlopscopeArray:
    budget = require_budget()
    cost = blackman_cost(M)
    with budget.deduct("blackman", flop_cost=cost, subscripts=None, shapes=((M,),)):
        result = _call_numpy(_np.blackman, M)
    return result  # type: ignore[return-value]


attach_docstring(blackman, _np.blackman, "counted_custom", "3n FLOPs")


def hamming_cost(n: int) -> int:
    """FLOP cost of Hamming window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: 2n (FMA=2 textbook: 1 multiply + 1 add per sample).

    Notes
    -----
    Two ops per sample under FMA=2 convention (1 multiply + 1 add).
    """
    return max(2 * n, 1)


@_counted_wrapper
def hamming(M: int) -> FlopscopeArray:
    budget = require_budget()
    cost = hamming_cost(M)
    with budget.deduct("hamming", flop_cost=cost, subscripts=None, shapes=((M,),)):
        result = _call_numpy(_np.hamming, M)
    return result  # type: ignore[return-value]


attach_docstring(hamming, _np.hamming, "counted_custom", "2n FLOPs (FMA=2)")


def hanning_cost(n: int) -> int:
    """FLOP cost of Hanning window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: 2n (FMA=2 textbook: 1 multiply + 1 add per sample).

    Notes
    -----
    Two ops per sample under FMA=2 convention (1 multiply + 1 add).
    """
    return max(2 * n, 1)


@_counted_wrapper
def hanning(M: int) -> FlopscopeArray:
    budget = require_budget()
    cost = hanning_cost(M)
    with budget.deduct("hanning", flop_cost=cost, subscripts=None, shapes=((M,),)):
        result = _call_numpy(_np.hanning, M)
    return result  # type: ignore[return-value]


attach_docstring(hanning, _np.hanning, "counted_custom", "2n FLOPs (FMA=2)")


def kaiser_cost(n: int) -> int:
    """FLOP cost of Kaiser window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: 3n.

    Notes
    -----
    Bessel function evaluation per sample.
    """
    return max(3 * n, 1)


@_counted_wrapper
def kaiser(M: int, beta: float) -> FlopscopeArray:
    budget = require_budget()
    cost = kaiser_cost(M)
    with budget.deduct("kaiser", flop_cost=cost, subscripts=None, shapes=((M,),)):
        result = _call_numpy(_np.kaiser, M, beta)
    return result  # type: ignore[return-value]


attach_docstring(kaiser, _np.kaiser, "counted_custom", "3n FLOPs")

import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
