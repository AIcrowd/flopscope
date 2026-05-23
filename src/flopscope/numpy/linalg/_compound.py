"""Compound linalg operations with FLOP counting."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as _np
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._ndarray import FlopscopeArray, _asflopscope, _to_base_ndarray
from flopscope._validation import require_budget
from flopscope.numpy.linalg._solvers import _batch_size, _has_zero_dim


def _popcount(n: int) -> int:
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def multi_dot_cost(shapes: Sequence[Sequence[int]]) -> int:
    """FLOP cost of optimal matrix chain multiplication.

    Parameters
    ----------
    shapes : list of tuple of int
        Shapes of the matrices to be multiplied.

    Returns
    -------
    int
        Estimated FLOP count using optimal parenthesization.

    Notes
    -----
    Uses dynamic programming for optimal parenthesization.
    Source: Cormen et al., *Introduction to Algorithms* (CLRS), §15.2.

    Each binary matmul step (m x k) @ (k x n) costs ``2 * m * k * n`` FLOPs
    under the FMA=2 textbook convention (m*k*n multiplies + m*k*n adds,
    ignoring the −m*n off-by-one for accumulation at this level of
    approximation).  FMA=2 unification 2026-05-20.
    """
    n = len(shapes)
    if n < 2:
        return 0
    dims = [s[0] for s in shapes] + [shapes[-1][-1]]
    if n == 2:
        return 2 * dims[0] * dims[1] * dims[2]
    cost_table = [[0] * n for _ in range(n)]
    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            cost_table[i][j] = float("inf")  # type: ignore[reportAssignmentType]
            for k in range(i, j):
                cost = (
                    cost_table[i][k]
                    + cost_table[k + 1][j]
                    + 2 * dims[i] * dims[k + 1] * dims[j + 1]
                )
                if cost < cost_table[i][j]:
                    cost_table[i][j] = cost
    return max(int(cost_table[0][n - 1]), 1)


@_counted_wrapper
def multi_dot(
    arrays: Sequence[ArrayLike], *, out: ArrayLike | None = None
) -> FlopscopeArray:
    """Efficient multi-matrix dot product with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = any(isinstance(a, FlopscopeArray) for a in arrays)
    arrays = [a if isinstance(a, _np.ndarray) else _np.asarray(a) for a in arrays]
    shapes = [arr.shape for arr in arrays]
    cost = multi_dot_cost(shapes)
    out_stripped = _to_base_ndarray(out) if out is not None else None
    with budget.deduct(
        "linalg.multi_dot", flop_cost=cost, subscripts=None, shapes=tuple(shapes)
    ):
        result = _call_numpy(
            _np.linalg.multi_dot,
            [_to_base_ndarray(a) for a in arrays],
            out=out_stripped,  # type: ignore[reportArgumentType]
        )
    if out is not None:
        return out  # type: ignore[reportReturnType]
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    multi_dot, _np.linalg.multi_dot, "linalg", "Optimal chain multiplication cost"
)


def matrix_power_cost(n: int, k: int) -> int:
    """FLOP cost of matrix power ``A**k``.

    Parameters
    ----------
    n : int
        Matrix dimension.
    k : int
        Exponent.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    Uses exponentiation by repeated squaring. For ``k < 0``, adds
    ``matmul_cost(n, n, n)`` for the initial matrix inversion (LU-based).
    Per-matmul cost is delegated to ``matmul_cost(n, n, n)`` so this formula
    tracks ``fnp.matmul``'s convention automatically (issue #69; the FMA=2
    unification of 2026-05-20 previously left this wrapper undercounting
    by ~2x).
    """
    from flopscope._flops import matmul_cost

    if k == 0 or k == 1:
        return 0
    if k < 0:
        # Inversion: LU-based, ~2*n^3 - n^2, modelled as one matmul-equivalent.
        return matmul_cost(n, n, n) + matrix_power_cost(n, abs(k))
    num_ops = math.floor(math.log2(k)) + _popcount(k) - 1
    return max(num_ops * matmul_cost(n, n, n), 1)


@_counted_wrapper
def matrix_power(a: ArrayLike, n: int) -> FlopscopeArray:
    """Matrix power with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    size = a.shape[-1]
    batch = _batch_size(a.shape)
    cost = matrix_power_cost(size, n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.matrix_power", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _call_numpy(_np.linalg.matrix_power, _to_base_ndarray(a), n)
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    matrix_power,
    _np.linalg.matrix_power,
    "linalg",
    r"$n^3 \times \text{exponent}$ FLOPs (repeated squaring)",
)
