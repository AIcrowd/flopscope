# src/flopscope/linalg/_properties.py
"""Matrix property wrappers with FLOP counting."""

from __future__ import annotations

from typing import Any

import numpy as _np
from numpy.linalg._linalg import SlogdetResult
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._dtype_billing import (
    linalg_billing_dtypes,
    reduction_billing_dtype,
    sum_accumulator_dtype,
)
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
    cost = trace_cost(n) * _batch_size(x.shape) if not _has_zero_dim(x.shape) else 0
    # trace is a diagonal sum, NOT a LAPACK op, so it is not routed through
    # linalg_billing_dtypes (integer input never forces float64). It DOES widen
    # its accumulator exactly like sum -- trace(int32) runs int64. An explicit
    # dtype= bills as the accumulator numpy runs, in both directions; the
    # default never bills below the input rate.
    _trace_dtypes = (
        reduction_billing_dtype(
            x.dtype,
            explicit_dtype=dtype,
            default_dtype=sum_accumulator_dtype(x.dtype),
        ),
    )
    with budget.deduct(
        "linalg.trace",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape,),
        dtypes=_trace_dtypes,
    ):
        result = _call_numpy(
            _np.linalg.trace, _to_base_ndarray(x), offset=offset, dtype=dtype
        )
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(trace, _np.linalg.trace, "linalg", r"$\min(m,n)$ FLOPs × batch")


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
        Estimated FLOP count: $2n^3/3 + n$.

    Notes
    -----
    2n^3/3 + n FLOPs (LU + product of diagonal).
    """
    return max(2 * n**3 // 3 + n, 1)


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
        "linalg.det",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=linalg_billing_dtypes(a.dtype),
    ):
        result = _call_numpy(_np.linalg.det, _to_base_ndarray(a))
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    det, _np.linalg.det, "linalg", r"$\frac{2}{3}n^3 + n$ FLOPs (LU + diagonal product)"
)


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
        Estimated FLOP count: $2n^3/3 + 18n$.

    Notes
    -----
    2n^3/3 + 18n FLOPs (LU + sum of log|diag|: abs + 16/elem log + reduce).
    """
    return max(2 * n**3 // 3 + 18 * n, 1)


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
        "linalg.slogdet",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=linalg_billing_dtypes(a.dtype),
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


attach_docstring(
    slogdet,
    _np.linalg.slogdet,
    "linalg",
    r"$\frac{2}{3}n^3 + 18n$ FLOPs (LU + sum of log|diag|)",
)


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
    - SVD-based norms (2-norm, nuclear norm): values-only SVD cost
      ``2*a*b^2 + 2*b^3`` where a=max(m,n), b=min(m,n).
    - General-p vector norms (ord not in {None, 0, 1, 2, inf, -inf}):
      ``18 * numel + 16`` (abs + pow per elem + sum-reduce + root pow).
    """
    numel = 1
    for d in shape:
        numel *= d
    numel = max(numel, 1)
    if len(shape) == 1:
        # General-p norm: abs + pow(16) per elem + sum + root pow(16)
        if ord not in (None, 0, 1, 2, _np.inf, -_np.inf):
            return 18 * numel + 16
        # FMA=2: standard vector norms cost 2*numel (one multiply + accumulate per element)
        return 2 * numel
    else:
        m, n = shape[-2], shape[-1]
        if ord is None or ord == "fro":
            return 2 * numel  # FMA=2
        elif ord in (1, -1, _np.inf, -_np.inf):
            return 2 * numel  # FMA=2
        elif ord in (2, -2, "nuc"):
            from flopscope._flops import svd_cost

            return svd_cost(m, n, with_vectors=False)
        return 2 * numel  # FMA=2


@_counted_wrapper
def norm(
    x: ArrayLike,
    ord: Any = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> FlopscopeArray:
    """Matrix or vector norm with FLOP counting.

    Cost = norm_cost(effective_shape, ord) × batch groups.
    When axis=None the whole array is one group (batch groups == 1).
    When axis selects a subset of dimensions, every combination of the
    remaining (non-reduced) dimensions is a separate group.
    """
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
        group_numel = 1
        for dim in effective_shape:
            group_numel *= dim
        n_groups = (x.size // group_numel) if group_numel else 0
        cost = norm_cost(effective_shape, ord=ord) * max(n_groups, 0)
    except (IndexError, ValueError):
        # Let numpy raise the proper error with the right type/message
        return _np.linalg.norm(  # type: ignore[reportReturnType]
            _to_base_ndarray(x), ord=ord, axis=axis, keepdims=keepdims
        )
    with budget.deduct(
        "linalg.norm",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape,),
        dtypes=linalg_billing_dtypes(x.dtype),
    ):
        result = _call_numpy(
            _np.linalg.norm, _to_base_ndarray(x), ord=ord, axis=axis, keepdims=keepdims
        )
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    norm,
    _np.linalg.norm,
    "linalg",
    "depends on ord parameter -- see docstring; × batch groups",
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
    Standard norms (ord in {None, 0, 1, 2, inf, -inf}) cost 2*numel FLOPs
    (FMA=2: one multiply + accumulate per element).
    General-p norms cost 18*numel + 16 FLOPs per group: abs(1) + pow(16)
    per element + sum-reduce(1) + final root pow(16). The wrapper's
    n_groups multiplier scales the +16 correctly per group.
    """
    numel = 1
    for d in shape:
        numel *= d
    numel = max(numel, 1)
    # General-p norm: abs + pow(16) per elem + sum + root pow(16)
    if ord not in (None, 0, 1, 2, _np.inf, -_np.inf):
        return 18 * numel + 16
    # FMA=2: standard norms cost 2*numel (one multiply + accumulate per element).
    return 2 * numel


@_counted_wrapper
def vector_norm(
    x: ArrayLike,
    ord: Any = 2,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> FlopscopeArray:
    """Vector norm with FLOP counting.

    Cost = vector_norm_cost(effective_shape, ord) × batch groups.
    When axis=None the whole array is one group (batch groups == 1).
    When axis selects a subset of dimensions, every combination of the
    remaining (non-reduced) dimensions is a separate group.
    """
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
    group_numel = 1
    for dim in effective_shape:
        group_numel *= dim
    n_groups = (x.size // group_numel) if group_numel else 0
    cost = vector_norm_cost(effective_shape, ord=ord) * max(n_groups, 0)
    with budget.deduct(
        "linalg.vector_norm",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape,),
        dtypes=linalg_billing_dtypes(x.dtype),
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
    vector_norm,
    _np.linalg.vector_norm,
    "linalg",
    "depends on ord parameter; × batch groups",
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
    - SVD-based norms (2-norm, nuclear): values-only SVD cost
      ``2*a*b^2 + 2*b^3`` where a=max(m,n), b=min(m,n).
    """
    m, n = shape[-2], shape[-1]
    numel = m * n
    if ord is None or ord == "fro":
        return 2 * numel  # FMA=2
    elif ord in (1, -1, _np.inf, -_np.inf):
        return 2 * numel  # FMA=2
    elif ord in (2, -2, "nuc"):
        from flopscope._flops import svd_cost

        return svd_cost(m, n, with_vectors=False)
    return 2 * numel  # FMA=2


@_counted_wrapper
def matrix_norm(
    x: ArrayLike, ord: Any = "fro", keepdims: bool = False
) -> FlopscopeArray:
    """Matrix norm with FLOP counting.

    Cost = matrix_norm_cost(x.shape[-2:], ord) × batch groups.
    Batch groups = product of all dimensions except the last two.
    Zero-dim inputs cost 0 FLOPs.
    """
    budget = require_budget()
    inputs_were_whest = isinstance(x, FlopscopeArray)
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    cost = (
        matrix_norm_cost(x.shape, ord=ord) * _batch_size(x.shape)
        if not _has_zero_dim(x.shape)
        else 0
    )
    with budget.deduct(
        "linalg.matrix_norm",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape,),
        dtypes=linalg_billing_dtypes(x.dtype),
    ):
        result = _call_numpy(
            _np.linalg.matrix_norm, _to_base_ndarray(x), ord=ord, keepdims=keepdims
        )  # type: ignore[reportCallIssue]
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    matrix_norm,
    _np.linalg.matrix_norm,
    "linalg",
    "depends on ord parameter; × batch groups",
)


def cond_cost(m: int, n: int, p=None) -> int:
    """FLOP cost of condition number.

    p in {None, 2, -2}: values-only SVD + 1 divide.
    other p (square only): norm(A)*norm(inv(A)) -> inv (2n^3) + two
    elementwise norm passes (2n^2 each) + 1 multiply.
    """
    from flopscope._flops import svd_cost

    if p is None or p == 2 or p == -2:
        return max(svd_cost(m, n, with_vectors=False) + 1, 1)
    k = min(m, n)
    return max(2 * k**3 + 4 * m * n + 1, 1)


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
        "linalg.cond",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape,),
        dtypes=linalg_billing_dtypes(x.dtype),
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
    r"values-only SVD + 1 (p in {None,2,-2}) or 2n^3 + 4n^2 + 1 (inv-based)",
)


def matrix_rank_cost(m: int, n: int) -> int:
    """FLOP cost of matrix rank: values-only SVD + min(m, n) threshold compares."""
    from flopscope._flops import svd_cost

    return max(svd_cost(m, n, with_vectors=False) + min(m, n), 1)


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
        "linalg.matrix_rank",
        flop_cost=cost,
        subscripts=None,
        shapes=(A.shape,),
        dtypes=linalg_billing_dtypes(A.dtype),
    ):
        result = _call_numpy(_np.linalg.matrix_rank, _to_base_ndarray(A), **kwargs)
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    matrix_rank,
    _np.linalg.matrix_rank,
    "linalg",
    r"values-only SVD + min(m,n)",
)
