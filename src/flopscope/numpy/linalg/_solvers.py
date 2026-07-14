# src/flopscope/linalg/_solvers.py
"""Linear solver wrappers with FLOP counting."""

from __future__ import annotations

from typing import Any

import numpy as _np
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._ndarray import FlopscopeArray, _asflopscope, _to_base_ndarray
from flopscope._symmetric import SymmetricTensor, validate_symmetry_groups
from flopscope._validation import require_budget
from flopscope.errors import SymmetryError


def _batch_size(shape):
    """Number of matrices in a batched array."""
    if len(shape) <= 2:
        return 1
    result = 1
    for d in shape[:-2]:
        result *= d
    return result


def _has_zero_dim(shape):
    """Check if any matrix dimension is zero."""
    return len(shape) >= 2 and (shape[-2] == 0 or shape[-1] == 0)


def solve_cost(n: int, nrhs: int = 1, symmetric: bool = False) -> int:
    r"""FLOP cost of solving Ax = b: LU + two triangular solves (FMA=2).

    2n^3/3 (getrf) + 2n^2*nrhs (getrs). G&VL 4e §3.2. ``symmetric`` is
    kept for API compatibility and ignored.

    Parameters
    ----------
    n : int
        Matrix dimension.
    nrhs : int, optional
        Number of right-hand side columns. Default is 1.
    symmetric : bool, optional
        Kept for API compatibility; ignored. Default is False.

    Returns
    -------
    int
        Estimated FLOP count: $\frac{2}{3}n^3 + 2n^2 \cdot nrhs$.
    """
    return max(2 * n**3 // 3 + 2 * n * n * nrhs, 1)


@_counted_wrapper
def solve(a: ArrayLike, b: ArrayLike) -> FlopscopeArray:
    """Solve linear system ``a @ x = b`` with FLOP counting.

    The result adopts the subclass of ``b`` (matching numpy's
    ``np.linalg.solve`` policy): if ``b`` is a plain ndarray the
    solution is plain ndarray even when ``a`` is a ``FlopscopeArray``;
    if ``b`` is a ``FlopscopeArray`` the solution is wrapped accordingly.
    """
    budget = require_budget()
    # Match NumPy's ``linalg.solve`` subclass-return policy: the result
    # adopts the subclass of ``b``. ``np.linalg.solve(FlopscopeArray, plain)``
    # therefore returns plain ndarray to keep parity with raw NumPy.
    b_was_whest = isinstance(b, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    nrhs = b.shape[-1] if b.ndim >= 2 else 1
    cost = solve_cost(n, nrhs=nrhs) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.solve",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(a.dtype, b.dtype),
    ):
        result = _call_numpy(_np.linalg.solve, _to_base_ndarray(a), _to_base_ndarray(b))
    if b_was_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    solve,
    _np.linalg.solve,
    "linalg",
    r"$\frac{2}{3}n^3 + 2n^2 \cdot nrhs$ FLOPs (LU + triangular solves)",
)


def inv_cost(n: int, symmetric: bool = False) -> int:
    """FLOP cost of matrix inverse.

    Parameters
    ----------
    n : int
        Matrix dimension.
    symmetric : bool, optional
        If True, assume symmetric input. Default is False.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    Uses $n^3/3 + n^3$ for symmetric input (Cholesky factorization + n
    triangular solves against identity), or $2n^3$ for general input
    (getrf 2n^3/3 + getri 4n^3/3).
    """
    if symmetric:
        return max(n**3 // 3 + n**3, 1)
    return max(2 * n**3, 1)


@_counted_wrapper
def inv(a: ArrayLike) -> FlopscopeArray:
    """Matrix inverse with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    input_symmetry = a.symmetry if isinstance(a, SymmetricTensor) else None
    is_symmetric = input_symmetry is not None
    cost = (
        inv_cost(n, symmetric=is_symmetric) * batch if not _has_zero_dim(a.shape) else 0
    )
    with budget.deduct(
        "linalg.inv",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(a.dtype,),
    ):
        result = _call_numpy(_np.linalg.inv, _to_base_ndarray(a))
    if is_symmetric:
        # Internal wrapping: use SymmetricTensor directly (D1 pattern — no
        # billing; as_symmetric would double-count the user's inv cost).
        try:
            validate_symmetry_groups(_np.asarray(result), [input_symmetry])
            result = SymmetricTensor(_np.asarray(result), symmetry=input_symmetry)
        except SymmetryError:
            pass
    if inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    inv,
    _np.linalg.inv,
    "linalg",
    r"$2n^3$ FLOPs, or $n^3/3 + n^3$ for SymmetricTensor input. Returns SymmetricTensor if input is symmetric.",
)


def lstsq_cost(m: int, n: int, b_cols: int = 1, b_ndim: int = 1) -> int:
    """FLOP cost of least-squares via SVD.

    Parameters
    ----------
    m : int
        Number of rows in A.
    n : int
        Number of columns in A.
    b_cols : int, default 1
        Number of RHS columns (``b.shape[-1]`` if 2D, else 1).
    b_ndim : int, default 1
        Number of dimensions in b. Use 1 for a 1D RHS vector, 2 for a 2D
        matrix of RHS vectors.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    NumPy uses LAPACK ``gelsd`` (SVD-based). Cost decomposition:
    ``svd(m,n) + ut_b + k*c + reconstruction`` where ``k = min(m, n)``
    and ``c = b_cols``.

    Both 1-D and 2-D RHS branches use ``matmul_cost``:
    ``ut_b = matmul_cost(k, m, b_cols)`` and
    ``reconstruction = matmul_cost(n, k, b_cols)``.

    Issue #69 (was previously just ``svd_cost`` ignoring the
    back-substitution).

    The SVD term uses with_vectors=True (the reconstruction needs U/V); the
    4.0 linalg weight is gone, so this composed value is exactly what is charged.
    """
    from flopscope._flops import matmul_cost, svd_cost

    k = min(m, n)
    c = b_cols
    svd = svd_cost(m, n, with_vectors=True)
    # matmul 2D×1D is now exact (== matmul_cost), so both b_ndim branches use it.
    if b_ndim == 1:
        ut_b = matmul_cost(k, m, 1)
        reconstruction = matmul_cost(n, k, 1)
    else:
        ut_b = matmul_cost(k, m, c)
        reconstruction = matmul_cost(n, k, c)
    divide_by_s = k * c
    return max(svd + ut_b + divide_by_s + reconstruction, 1)


@_counted_wrapper
def lstsq(
    a: ArrayLike, b: ArrayLike, rcond: float | None = None
) -> tuple[FlopscopeArray, FlopscopeArray, int, FlopscopeArray]:
    """Least-squares solution with FLOP counting.

    Returns a 4-tuple ``(solution, residuals, rank, singular_values)``.
    The solution and the array elements adopt the subclass of ``b``
    (matching numpy's ``np.linalg.lstsq`` policy): if ``b`` is a plain
    ndarray the outputs are plain ndarray even when ``a`` is a
    ``FlopscopeArray``; if ``b`` is a ``FlopscopeArray`` they are wrapped
    accordingly.
    """
    budget = require_budget()
    # Match NumPy's ``linalg.lstsq`` subclass-return policy: the solution
    # adopts the subclass of ``b``. The residuals and singular-values
    # arrays follow the same rule (whatever wrapping ``b`` would imply).
    b_was_whest = isinstance(b, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    m, n = a.shape[-2], a.shape[-1]
    batch = _batch_size(a.shape)
    if not isinstance(b, _np.ndarray):
        b_arr = _np.asarray(b)
    else:
        b_arr = b
    b_cols = b_arr.shape[-1] if b_arr.ndim > 1 else 1
    cost = (
        lstsq_cost(m, n, b_cols=b_cols, b_ndim=b_arr.ndim) * batch
        if not _has_zero_dim(a.shape)
        else 0
    )
    with budget.deduct(
        "linalg.lstsq",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(a.dtype, b_arr.dtype),
    ):
        result = _call_numpy(
            _np.linalg.lstsq, _to_base_ndarray(a), _to_base_ndarray(b), rcond=rcond
        )  # type: ignore[reportCallIssue]
    if b_was_whest:
        return tuple(  # type: ignore[reportReturnType]
            _asflopscope(r) if isinstance(r, _np.ndarray) else r for r in result
        )
    return tuple(result)  # type: ignore[reportReturnType]


attach_docstring(
    lstsq,
    _np.linalg.lstsq,
    "linalg",
    r"SVD + back-substitution FLOPs: ``svd(m,n) + matmul(k,m,c) + k*c + matmul(n,k,c)`` (issue #69)",
)


def pinv_cost(m: int, n: int) -> int:
    """FLOP cost of Moore-Penrose pseudoinverse.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    NumPy implements ``pinv`` as: ``svd(A, full_matrices=False)`` →
    threshold tiny singular values → multiply ``vt.T`` by ``s_inv``
    broadcasted → matmul with ``u.T``. We compose the cost from
    ``svd_cost`` and ``matmul_cost`` so this formula tracks those
    helpers automatically (issue #69; was previously missing the
    post-SVD reconstruction).

    Total: ``svd(m,n) + threshold(min(m,n)) + diag_scale(n*min(m,n))
    + matmul(n, min(m,n), m)``.

    The SVD term uses with_vectors=True (the reconstruction needs U/V); the
    4.0 linalg weight is gone, so this composed value is exactly what is charged.
    """
    from flopscope._flops import matmul_cost, svd_cost

    k = min(m, n)
    svd = svd_cost(m, n, with_vectors=True)
    threshold = k
    diag_scale = n * k
    reconstruction = matmul_cost(n, k, m)
    return max(svd + threshold + diag_scale + reconstruction, 1)


@_counted_wrapper
def pinv(
    a: ArrayLike,
    rcond: float | None = None,
    hermitian: bool = False,
    *,
    rtol: float | None = None,
) -> FlopscopeArray:
    """Pseudoinverse with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    m, n = a.shape[-2], a.shape[-1]
    batch = _batch_size(a.shape)
    cost = pinv_cost(m, n) * batch if not _has_zero_dim(a.shape) else 0
    kwargs = {"hermitian": hermitian}
    if rcond is not None:
        kwargs["rcond"] = rcond  # type: ignore[reportAssignmentType]
    if rtol is not None:
        kwargs["rtol"] = rtol  # type: ignore[reportAssignmentType]
    with budget.deduct(
        "linalg.pinv",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(a.dtype,),
    ):
        result = _call_numpy(_np.linalg.pinv, _to_base_ndarray(a), **kwargs)
    if inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    pinv,
    _np.linalg.pinv,
    "linalg",
    r"SVD(with U/V) + min(m,n) + n*min(m,n) + matmul(n, min(m,n), m) FLOPs (see pinv_cost)",
)


def tensorsolve_cost(a_shape: tuple, ind: int | None = None) -> int:
    """FLOP cost of tensor solve.

    Parameters
    ----------
    a_shape : tuple of int
        Shape of the coefficient tensor.
    ind : int or None, optional
        Number of leading indices for the solution. Default is 2.

    Returns
    -------
    int
        Estimated FLOP count: $\frac{2}{3}n^3 + 2n^2$ where $n$ = product of
        trailing dims. Reduces to ``solve_cost(n, 1)``.

    Notes
    -----
    Reduces to a standard linear solve after reshaping.
    """
    if ind is None:
        ind = 2
    n = 1
    for d in a_shape[ind:]:
        n *= d
    return solve_cost(n, nrhs=1)


@_counted_wrapper
def tensorsolve(a: ArrayLike, b: ArrayLike, axes: Any = None) -> FlopscopeArray:
    """Tensor solve with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray) or isinstance(b, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    b_arr = _np.asarray(b)
    cost = tensorsolve_cost(a.shape)
    with budget.deduct(
        "linalg.tensorsolve",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        # tensorsolve delegates to solve: a real `a` with a complex `b` yields a
        # complex result, so both operands must join the billing dtype tuple or
        # the 4x complex factor is bypassed (mirrors solve/lstsq above).
        dtypes=(a.dtype, b_arr.dtype),
    ):
        result = _call_numpy(
            _np.linalg.tensorsolve,
            _to_base_ndarray(a),
            _to_base_ndarray(b),  # type: ignore[arg-type]
            axes=axes,
        )
    if inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    tensorsolve,
    _np.linalg.tensorsolve,
    "linalg",
    r"$\frac{2}{3}n^3 + 2n^2$ FLOPs where n = product of trailing dims (reduces to solve)",
)


def tensorinv_cost(a_shape: tuple, ind: int = 2) -> int:
    """FLOP cost of tensor inverse.

    Parameters
    ----------
    a_shape : tuple of int
        Shape of the input tensor.
    ind : int, optional
        Number of leading indices. Default is 2.

    Returns
    -------
    int
        Estimated FLOP count: $2n^3$ where $n$ = product of leading dims.
        Reduces to ``inv_cost(n)``.

    Notes
    -----
    Reduces to a standard matrix inverse after reshaping.
    """
    n = 1
    for d in a_shape[:ind]:
        n *= d
    return inv_cost(n)


@_counted_wrapper
def tensorinv(a: ArrayLike, ind: int = 2) -> FlopscopeArray:
    """Tensor inverse with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    cost = tensorinv_cost(a.shape, ind=ind)
    with budget.deduct(
        "linalg.tensorinv",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=(a.dtype,),
    ):
        result = _call_numpy(_np.linalg.tensorinv, _to_base_ndarray(a), ind=ind)
    if inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    tensorinv,
    _np.linalg.tensorinv,
    "linalg",
    r"$2n^3$ FLOPs where n = product of leading dims (reduces to inv)",
)
