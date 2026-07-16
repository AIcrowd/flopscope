# src/flopscope/linalg/_decompositions.py
"""Matrix decomposition wrappers with FLOP counting.

Each operation has a co-located cost function documenting the formula,
source, and assumptions. Cost functions are pure (shape params) -> int.
"""

from __future__ import annotations

import numpy as _np
from numpy.linalg._linalg import EighResult, EigResult, QRResult
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._dtype_billing import linalg_billing_dtypes
from flopscope._ndarray import FlopscopeArray, _asflopscope, _to_base_ndarray
from flopscope._validation import require_budget
from flopscope.numpy.linalg._solvers import _batch_size, _has_zero_dim


def cholesky_cost(n: int) -> int:
    """FLOP cost of Cholesky decomposition.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: $n^3/3$.

    Notes
    -----
    n^3/3 FLOPs (G&VL 4e Alg 4.2.1; LAPACK dpotrf).
    """
    return max(n**3 // 3, 1)


@_counted_wrapper
def cholesky(a: ArrayLike, /, *, upper: bool = False) -> FlopscopeArray:
    """Cholesky decomposition with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    cost = cholesky_cost(n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.cholesky",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=linalg_billing_dtypes(a.dtype),
    ):
        result = _call_numpy(_np.linalg.cholesky, _to_base_ndarray(a), upper=upper)
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(cholesky, _np.linalg.cholesky, "linalg", r"$n^3/3$ FLOPs")


def qr_cost(m: int, n: int, mode: str = "reduced") -> int:
    """FLOP cost of QR decomposition (Householder, FMA=2).

    Factorization (dgeqrf): 2*m*n*k - 2*k^3/3, k = min(m, n)
    (G&VL 4e §5.2). Modes "reduced"/"complete" additionally form Q
    explicitly (dorgqr), modeled as the same count again. Modes
    "r"/"raw" bill the factorization only.
    """
    k = min(m, n)
    factor = 2 * m * n * k - 2 * k**3 // 3
    if mode in ("reduced", "complete"):
        return max(2 * factor, 1)
    return max(factor, 1)


@_counted_wrapper
def qr(
    a: ArrayLike, mode: str = "reduced"
) -> tuple[FlopscopeArray, FlopscopeArray] | FlopscopeArray:
    """QR decomposition with FLOP counting.

    Returns vary by ``mode``:

    - ``"reduced"`` (default) / ``"complete"`` → ``QRResult(q, r)``
    - ``"r"`` → ``r`` only
    - ``"raw"`` → 2-tuple ``(h, tau)`` of two ndarrays with mismatched
      shapes (preserved as a tuple rather than collapsed via
      ``_asflopscope``, which would otherwise fail with
      ``ValueError: ... inhomogeneous shape``)
    """
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    m, n = a.shape[-2], a.shape[-1]
    batch = _batch_size(a.shape)
    cost = qr_cost(m, n, mode=mode) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.qr",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=linalg_billing_dtypes(a.dtype),
    ):
        result = _call_numpy(_np.linalg.qr, _to_base_ndarray(a), mode=mode)  # type: ignore[reportCallIssue]
    if mode in ("reduced", "complete"):
        q, r = result  # type: ignore[reportGeneralTypeIssues]
        if inputs_were_whest:
            return QRResult(_asflopscope(q), _asflopscope(r))  # type: ignore[reportReturnType]
        return QRResult(q, r)  # type: ignore[reportReturnType]
    if mode == "raw":
        # ``raw`` returns a (h, tau) tuple of two ndarrays with different
        # shapes — preserve the tuple structure rather than collapsing to
        # a single array via ``_asflopscope``.
        h, tau = result  # type: ignore[reportGeneralTypeIssues]
        if inputs_were_whest:
            return _asflopscope(h), _asflopscope(tau)  # type: ignore[reportReturnType]
        return h, tau  # type: ignore[reportReturnType]
    if inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportArgumentType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    qr,
    _np.linalg.qr,
    "linalg",
    r"$2(2mnk - \frac{2}{3}k^3)$ FLOPs (reduced/complete; form Q) or $2mnk - \frac{2}{3}k^3$ (r/raw), k=min(m,n)",
)


def eig_cost(n: int) -> int:
    """FLOP cost of eigendecomposition.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: ~25n^3.

    Notes
    -----
    Hessenberg reduction + Francis QR with eigenvector backtransform
    (LAPACK Users' Guide Table 3.13 DGEEV / G&VL 4e §7.5).
    Confirmed by the 2026-06 evidence audit (LAPACK Users' Guide Table 3.13
    / G&VL 4e §7.5, §8.3 + runtime scaling); see docs/reference/cost-model.md.
    """
    return max(25 * n**3, 1)


@_counted_wrapper
def eig(a: ArrayLike) -> tuple[FlopscopeArray, FlopscopeArray]:
    """Eigendecomposition with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    cost = eig_cost(n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.eig",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=linalg_billing_dtypes(a.dtype),
    ):
        result = _call_numpy(_np.linalg.eig, _to_base_ndarray(a))
    if inputs_were_whest:
        return EigResult(  # type: ignore[reportReturnType]
            _asflopscope(result.eigenvalues), _asflopscope(result.eigenvectors)
        )
    return result  # type: ignore[reportReturnType]


attach_docstring(
    eig, _np.linalg.eig, "linalg", r"$\sim 25 n^3$ FLOPs (confirmed 2026-06 audit)"
)


def eigh_cost(n: int) -> int:
    """FLOP cost of symmetric eigendecomposition.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: ~9n^3.

    Notes
    -----
    Tridiagonalization + symmetric QR with vectors
    (G&VL 4e §8.3; LAPACK dsyevd).
    Confirmed by the 2026-06 evidence audit (LAPACK Users' Guide Table 3.13
    / G&VL 4e §7.5, §8.3 + runtime scaling); see docs/reference/cost-model.md.
    """
    return max(9 * n**3, 1)


@_counted_wrapper
def eigh(a: ArrayLike, UPLO: str = "L") -> tuple[FlopscopeArray, FlopscopeArray]:
    """Symmetric eigendecomposition with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    cost = eigh_cost(n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.eigh",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=linalg_billing_dtypes(a.dtype),
    ):
        result = _call_numpy(_np.linalg.eigh, _to_base_ndarray(a), UPLO=UPLO)  # type: ignore[reportCallIssue]
    if inputs_were_whest:
        return EighResult(  # type: ignore[reportReturnType]
            _asflopscope(result.eigenvalues), _asflopscope(result.eigenvectors)
        )
    return result  # type: ignore[reportReturnType]


attach_docstring(
    eigh, _np.linalg.eigh, "linalg", r"$\sim 9 n^3$ FLOPs (confirmed 2026-06 audit)"
)


def eigvals_cost(n: int) -> int:
    """FLOP cost of computing eigenvalues.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: ~10n^3.

    Notes
    -----
    ~10n^3, values only (LAPACK Users' Guide Table 3.13 DGEEV values-only
    = 10.00·N^3 exact; G&VL 4e §7.5).
    Confirmed by the 2026-06 evidence audit (LAPACK Users' Guide Table 3.13
    / G&VL 4e §7.5, §8.3 + runtime scaling); see docs/reference/cost-model.md.
    """
    # Note: costs MORE than eigh_cost (9n^3) — nonsymmetric Hessenberg+QR without vectors
    # vs symmetric tridiagonalization with vectors (G&VL §7.5 vs §8.3).
    return max(10 * n**3, 1)


@_counted_wrapper
def eigvals(a: ArrayLike) -> FlopscopeArray:
    """Eigenvalues (nonsymmetric) with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    cost = eigvals_cost(n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.eigvals",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=linalg_billing_dtypes(a.dtype),
    ):
        result = _call_numpy(_np.linalg.eigvals, _to_base_ndarray(a))
    if inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    eigvals,
    _np.linalg.eigvals,
    "linalg",
    r"$\sim 10 n^3$ FLOPs (confirmed 2026-06 audit)",
)


def eigvalsh_cost(n: int) -> int:
    """FLOP cost of computing eigenvalues of a symmetric matrix.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: 4n^3/3.

    Notes
    -----
    4n^3/3: tridiagonalization, values only (G&VL 4e §8.3).
    Confirmed by the 2026-06 evidence audit (LAPACK Users' Guide Table 3.13
    / G&VL 4e §7.5, §8.3 + runtime scaling); see docs/reference/cost-model.md.
    """
    return max(4 * n**3 // 3, 1)


@_counted_wrapper
def eigvalsh(a: ArrayLike, UPLO: str = "L") -> FlopscopeArray:
    """Eigenvalues (symmetric) with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    cost = eigvalsh_cost(n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.eigvalsh",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=linalg_billing_dtypes(a.dtype),
    ):
        result = _call_numpy(_np.linalg.eigvalsh, _to_base_ndarray(a), UPLO=UPLO)  # type: ignore[reportCallIssue]
    if inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    eigvalsh,
    _np.linalg.eigvalsh,
    "linalg",
    r"$\frac{4}{3} n^3$ FLOPs (confirmed 2026-06 audit)",
)


def svdvals_cost(m: int, n: int, k: int | None = None) -> int:
    """FLOP cost of computing singular values (values-only SVD).

    Delegates to :func:`flopscope._flops.svd_cost` with ``with_vectors=False``:
    2*a*b^2 + 2*b^3, a = max(m, n), b = min(m, n). Top-k (1 <= k < min(m, n))
    bills min(4*m*n*k, that cost).
    """
    from flopscope._flops import svd_cost

    return svd_cost(m, n, k, with_vectors=False)


@_counted_wrapper
def svdvals(x: ArrayLike, /, *, k: int | None = None) -> FlopscopeArray:
    """Singular values with FLOP counting."""
    budget = require_budget()
    inputs_were_whest = isinstance(x, FlopscopeArray)
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    m, n = x.shape[-2], x.shape[-1]
    batch = _batch_size(x.shape)
    if k is None:
        k = min(m, n)
    if min(m, n) > 0 and not (1 <= k <= min(m, n)):
        raise ValueError(f"k must satisfy 1 <= k <= min(m, n) = {min(m, n)}, got k={k}")
    cost = svdvals_cost(m, n, k) * batch if not _has_zero_dim(x.shape) else 0
    with budget.deduct(
        "linalg.svdvals",
        flop_cost=cost,
        subscripts=None,
        shapes=(x.shape,),
        dtypes=linalg_billing_dtypes(x.dtype),
    ):
        result = _call_numpy(_np.linalg.svdvals, _to_base_ndarray(x))
    if k < min(m, n):
        result = result[..., :k]
    if inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    svdvals,
    _np.linalg.svdvals,
    "linalg",
    r"$2ab^2 + 2b^3$ FLOPs (values-only SVD; a=max(m,n), b=min(m,n))",
)
