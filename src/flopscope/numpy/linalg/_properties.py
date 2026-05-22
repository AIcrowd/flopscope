# src/flopscope/linalg/_properties.py
"""Matrix property wrappers with FLOP counting."""

from __future__ import annotations

from typing import Any

import numpy as _np
from numpy.linalg._linalg import SlogdetResult
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._ndarray import FlopscopeArray, _asflopscope, _to_base_ndarray
from flopscope._symmetric import SymmetricTensor
from flopscope._validation import require_budget
from flopscope.numpy.linalg._solvers import _batch_size, _has_zero_dim


def trace_cost(n: int) -> int:
    """FLOP cost of matrix trace.

    Parameters
    ----------
    n : int
        Number of diagonal elements to sum.

    Returns
    -------
    int
        Estimated FLOP count: n.

    Notes
    -----
    Simply sums n diagonal elements.
    """
    return max(n, 1)


@_counted_wrapper
def trace(x: ArrayLike, /, *, offset: int = 0, dtype: Any = None) -> FlopscopeArray:
    """Matrix trace with FLOP counting (numpy 2.0 linalg.trace signature)."""
    budget = require_budget()
    inputs_were_whest = isinstance(x, FlopscopeArray)
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    n = min(x.shape[-2], x.shape[-1])
    if offset > 0:
        n = min(n, x.shape[-1] - offset)
    elif offset < 0:
        n = min(n, x.shape[-2] + offset)
    n = max(n, 0)
    cost = trace_cost(n)
    with budget.deduct(
        "linalg.trace", flop_cost=cost, subscripts=None, shapes=(x.shape,)
    ):
        result = _call_numpy(
            _np.linalg.trace, _to_base_ndarray(x), offset=offset, dtype=dtype
        )
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(trace, _np.linalg.trace, "linalg", r"$n$ FLOPs")


def det_cost(n: int, symmetric: bool = False) -> int:
    """FLOP cost of determinant.

    Parameters
    ----------
    n : int
        Matrix dimension.
    symmetric : bool, optional
        Ignored (kept for API compatibility). Default is False.

    Returns
    -------
    int
        Estimated FLOP count: $n^3$.

    Notes
    -----
    Simplified cubic cost model for determinant computation.
    """
    return max(n**3, 1)


@_counted_wrapper
def det(a: ArrayLike) -> FlopscopeArray:
    """Determinant with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    is_symmetric = isinstance(a, SymmetricTensor)
    cost = (
        det_cost(n, symmetric=is_symmetric) * batch if not _has_zero_dim(a.shape) else 0
    )
    with budget.deduct(
        "linalg.det", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _call_numpy(_np.linalg.det, _to_base_ndarray(a))
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(det, _np.linalg.det, "linalg", r"$n^3$ FLOPs")


def slogdet_cost(n: int, symmetric: bool = False) -> int:
    """FLOP cost of sign and log-determinant.

    Parameters
    ----------
    n : int
        Matrix dimension.
    symmetric : bool, optional
        Ignored (kept for API compatibility). Default is False.

    Returns
    -------
    int
        Estimated FLOP count: $n^3$.

    Notes
    -----
    Simplified cubic cost model for sign and log-determinant computation.
    """
    return max(n**3, 1)


@_counted_wrapper
def slogdet(a: ArrayLike) -> SlogdetResult:
    """Sign and log-determinant with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    is_symmetric = isinstance(a, SymmetricTensor)
    cost = (
        slogdet_cost(n, symmetric=is_symmetric) * batch
        if not _has_zero_dim(a.shape)
        else 0
    )
    with budget.deduct(
        "linalg.slogdet", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _call_numpy(_np.linalg.slogdet, _to_base_ndarray(a))
    if inputs_were_whest:
        return SlogdetResult(
            _asflopscope(result.sign)
            if isinstance(result.sign, _np.ndarray)
            else result.sign,
            _asflopscope(result.logabsdet)
            if isinstance(result.logabsdet, _np.ndarray)
            else result.logabsdet,
        )
    return result


attach_docstring(slogdet, _np.linalg.slogdet, "linalg", r"$n^3$ FLOPs")


def norm_cost(shape: tuple, ord=None) -> int:
    """FLOP cost of matrix or vector norm.

    Parameters
    ----------
    shape : tuple of int
        Shape of the input array (or effective shape along norm axes).
    ord : {None, 'fro', 'nuc', inf, -inf, int}, optional
        Order of the norm.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    Cost depends on the ``ord`` parameter and input dimensionality.

    - Elementwise norms (Frobenius, L1, Linf, etc.): ``2 * numel`` (FMA=2, weight=1 baked in).
    - SVD-based norms (2-norm, nuclear norm): ``4 * m * n * min(m, n)``
      (weight=4 baked in, consistent with linalg.svd weight=4).
    """
    numel = 1
    for d in shape:
        numel *= d
    numel = max(numel, 1)
    if len(shape) == 1:
        # FMA=2: all vector norms cost 2*numel (one multiply + accumulate per element)
        return 2 * numel
    else:
        m, n = shape[-2], shape[-1]
        if ord is None or ord == "fro":
            return 2 * numel  # FMA=2
        elif ord in (1, -1, _np.inf, -_np.inf):
            return 2 * numel  # FMA=2
        elif ord == 2 or ord == -2:
            return 4 * m * n * min(m, n)  # SVD-based, weight=4 baked in
        elif ord == "nuc":
            return 4 * m * n * min(m, n)  # SVD-based, weight=4 baked in
        return 2 * numel  # FMA=2


@_counted_wrapper
def norm(
    x: ArrayLike,
    ord: Any = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> FlopscopeArray:
    """Matrix or vector norm with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(x, FlopscopeArray)
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    # Compute effective shape for FLOP cost, guarding against invalid axis.
    # If axis is out of bounds or ord is invalid, numpy will raise the correct
    # error (AxisError / ValueError); we skip budget deduction in that case.
    try:
        if axis is None:
            effective_shape = x.shape
        elif isinstance(axis, int):
            ndim = x.ndim
            norm_axis = axis + ndim if axis < 0 else axis
            if norm_axis < 0 or norm_axis >= max(ndim, 1):
                return _np.linalg.norm(  # type: ignore[reportReturnType]
                    _to_base_ndarray(x), ord=ord, axis=axis, keepdims=keepdims
                )
            effective_shape = (x.shape[norm_axis],) if ndim > 0 else ()
        else:
            effective_shape = tuple(x.shape[ax] for ax in axis)
        cost = norm_cost(effective_shape, ord=ord)
    except (IndexError, ValueError):
        # Let numpy raise the proper error with the right type/message
        return _np.linalg.norm(  # type: ignore[reportReturnType]
            _to_base_ndarray(x), ord=ord, axis=axis, keepdims=keepdims
        )
    with budget.deduct(
        "linalg.norm", flop_cost=cost, subscripts=None, shapes=(x.shape,)
    ):
        result = _call_numpy(
            _np.linalg.norm, _to_base_ndarray(x), ord=ord, axis=axis, keepdims=keepdims
        )
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    norm, _np.linalg.norm, "linalg", "depends on ord parameter -- see docstring"
)


def vector_norm_cost(shape: tuple, ord=None) -> int:
    """FLOP cost of vector norm.

    Parameters
    ----------
    shape : tuple of int
        Shape of the input array (or effective shape along norm axes).
    ord : {None, inf, -inf, int, float}, optional
        Order of the norm.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    All norms cost 2*numel FLOPs (FMA=2: one multiply + accumulate per element).
    """
    numel = 1
    for d in shape:
        numel *= d
    numel = max(numel, 1)
    # FMA=2: all norms cost 2*numel (one multiply + accumulate per element).
    return 2 * numel


@_counted_wrapper
def vector_norm(
    x: ArrayLike,
    ord: Any = 2,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> FlopscopeArray:
    """Vector norm with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(x, FlopscopeArray)
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    if axis is not None:
        if isinstance(axis, int):
            effective_shape = (x.shape[axis],)
        else:
            effective_shape = tuple(x.shape[ax] for ax in axis)
    else:
        effective_shape = x.shape
    cost = vector_norm_cost(effective_shape, ord=ord)
    with budget.deduct(
        "linalg.vector_norm", flop_cost=cost, subscripts=None, shapes=(x.shape,)
    ):
        result = _call_numpy(
            _np.linalg.vector_norm,
            _to_base_ndarray(x),
            ord=ord,
            axis=axis,
            keepdims=keepdims,
        )
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    vector_norm, _np.linalg.vector_norm, "linalg", "depends on ord parameter"
)


def matrix_norm_cost(shape: tuple, ord=None) -> int:
    """FLOP cost of matrix norm.

    Parameters
    ----------
    shape : tuple of int
        Shape of the input array (last two dims are the matrix).
    ord : {None, 'fro', 'nuc', inf, -inf, 1, -1, 2, -2}, optional
        Order of the norm.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    - Elementwise norms (Frobenius, L1, Linf): ``2 * numel`` (FMA=2, weight=1 baked in).
    - SVD-based norms (2-norm, nuclear): ``4 * m * n * min(m, n)``
      (weight=4 baked in, consistent with linalg.svd weight=4).
    """
    m, n = shape[-2], shape[-1]
    numel = m * n
    if ord is None or ord == "fro":
        return 2 * numel  # FMA=2
    elif ord in (1, -1, _np.inf, -_np.inf):
        return 2 * numel  # FMA=2
    elif ord == 2 or ord == -2:
        return 4 * m * n * min(m, n)  # SVD-based, weight=4 baked in
    elif ord == "nuc":
        return 4 * m * n * min(m, n)  # SVD-based, weight=4 baked in
    return 2 * numel  # FMA=2


@_counted_wrapper
def matrix_norm(
    x: ArrayLike, ord: Any = "fro", keepdims: bool = False
) -> FlopscopeArray:
    """Matrix norm with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(x, FlopscopeArray)
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    cost = matrix_norm_cost(x.shape, ord=ord)
    with budget.deduct(
        "linalg.matrix_norm", flop_cost=cost, subscripts=None, shapes=(x.shape,)
    ):
        result = _call_numpy(
            _np.linalg.matrix_norm, _to_base_ndarray(x), ord=ord, keepdims=keepdims
        )  # type: ignore[reportCallIssue]
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    matrix_norm, _np.linalg.matrix_norm, "linalg", "depends on ord parameter"
)


def cond_cost(m: int, n: int, p=None) -> int:
    """FLOP cost of condition number.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    p : {None, 2, -2, 1, -1, inf, -inf}, optional
        Norm type. ``None`` and ``2``/``-2`` use SVD; ``1``/``-1``/``inf``/``-inf``
        use LU factorization, which is cheaper.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    For ``p=None``, ``p=2``, or ``p=-2``, computed via SVD (cost m*n*min(m,n)).
    For ``p=1``, ``p=-1``, ``p=inf``, or ``p=-inf``, computed via LU factorization
    (cost ~min(m,n)^3 + m*n for norm).
    """
    if p is None or p == 2 or p == -2:
        return max(m * n * min(m, n), 1)
    # LU-based: factorization cost + norm computation
    k = min(m, n)
    return max(k**3 + m * n, 1)


@_counted_wrapper
def cond(x: ArrayLike, p: Any = None) -> FlopscopeArray:
    """Condition number with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(x, FlopscopeArray)
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    m, n = x.shape[-2], x.shape[-1]
    batch = _batch_size(x.shape)
    cost = cond_cost(m, n, p=p) * batch if not _has_zero_dim(x.shape) else 0
    has_nan = not _has_zero_dim(x.shape) and bool(
        _np.any(_np.isnan(_to_base_ndarray(x)))
    )
    with budget.deduct(
        "linalg.cond", flop_cost=cost, subscripts=None, shapes=(x.shape,)
    ):
        if has_nan and x.ndim > 2:
            # Batch with NaN: process each matrix individually so NaN
            # propagates per-matrix rather than SVD failing the whole batch.
            batch_shape = x.shape[:-2]
            flat = _to_base_ndarray(x).reshape(-1, x.shape[-2], x.shape[-1])
            out = _call_numpy(_np.empty, flat.shape[0], dtype=_np.float64)
            for i in range(flat.shape[0]):
                try:
                    out[i] = _call_numpy(_np.linalg.cond, flat[i], p=p)
                except _np.linalg.LinAlgError:
                    out[i] = _np.nan
            result = out.reshape(batch_shape)
        elif has_nan:
            # Single matrix with NaN: SVD may fail; return NaN instead.
            try:
                result = _call_numpy(_np.linalg.cond, _to_base_ndarray(x), p=p)
            except _np.linalg.LinAlgError:
                result = _call_numpy(_np.float64, _np.nan)
        else:
            result = _call_numpy(_np.linalg.cond, _to_base_ndarray(x), p=p)
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    cond,
    _np.linalg.cond,
    "linalg",
    r"$m \cdot n \cdot \min(m,n)$ FLOPs (SVD) or $\min(m,n)^3 + mn$ (LU) depending on p",
)


def matrix_rank_cost(m: int, n: int) -> int:
    """FLOP cost of matrix rank.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    int
        Estimated FLOP count: m * n * min(m, n).

    Notes
    -----
    Computed via SVD.
    """
    return max(m * n * min(m, n), 1)


@_counted_wrapper
def matrix_rank(
    A: ArrayLike,
    tol: float | None = None,
    hermitian: bool = False,
    *,
    rtol: float | None = None,
) -> FlopscopeArray | int:
    """Matrix rank with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(A, FlopscopeArray)
    if not isinstance(A, _np.ndarray):
        A = _np.asarray(A)
    if A.ndim < 2:
        # 0D or 1D: cost is trivial, let numpy handle shape/semantics
        cost = max(A.size, 1)
        batch = 1
    else:
        m, n = A.shape[-2], A.shape[-1]
        batch = _batch_size(A.shape)
        cost = matrix_rank_cost(m, n) * batch if not _has_zero_dim(A.shape) else 0
    kwargs = {"hermitian": hermitian}
    if tol is not None:
        kwargs["tol"] = tol  # type: ignore[reportAssignmentType]
    if rtol is not None:
        kwargs["rtol"] = rtol  # type: ignore[reportAssignmentType]
    with budget.deduct(
        "linalg.matrix_rank", flop_cost=cost, subscripts=None, shapes=(A.shape,)
    ):
        result = _call_numpy(_np.linalg.matrix_rank, _to_base_ndarray(A), **kwargs)
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    matrix_rank,
    _np.linalg.matrix_rank,
    "linalg",
    r"$m \cdot n \cdot \min(m,n)$ FLOPs (SVD)",
)
