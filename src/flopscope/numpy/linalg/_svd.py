"""SVD with FLOP counting."""

from __future__ import annotations

import numpy as _np
from numpy.linalg._linalg import SVDResult
from numpy.typing import ArrayLike

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._dtype_billing import linalg_billing_dtypes
from flopscope._flops import svd_cost
from flopscope._ndarray import FlopscopeArray, _asflopscope, _to_base_ndarray
from flopscope._validation import maybe_check_nan_inf, require_budget


def _batch_size(shape):
    """Number of matrices in a batched array."""
    from flopscope._batch import _broadcast_batch

    return _broadcast_batch(tuple(shape), core_ranks=(2,))


def _has_zero_dim(shape):
    """Check if any matrix dimension is zero."""
    return len(shape) >= 2 and (shape[-2] == 0 or shape[-1] == 0)


@_counted_wrapper
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
    *,
    k: int | None = None,
) -> tuple[FlopscopeArray, FlopscopeArray, FlopscopeArray] | FlopscopeArray:
    """Singular value decomposition with FLOP counting.

    Matches ``numpy.linalg.svd`` signature with an optional *k* parameter
    for truncated SVD (flopscope extension).

    FLOP Cost
    ---------
    values only (compute_uv=False):         2*a*b^2 + 2*b^3 per matrix
    with thin U/V (full_matrices=False):    6*a*b^2 + 20*b^3 per matrix
    with full U (full_matrices=True, m!=n): 4*a^2*b + 22*b^3 per matrix
    where a = max(m, n), b = min(m, n). Top-k (1 <= k < min(m, n)) bills
    min(4*m*n*k, economy) — the verified randomized-SVD truncated cost;
    results stay exact (numpy economy SVD, sliced). See cost-model.md.
    Constants confirmed by the 2026-06 evidence audit; see
    docs/reference/cost-model.md.

    Parameters
    ----------
    a : array_like
        Input array with shape ``(..., m, n)``.
    full_matrices : bool, optional
        If True (default), U and Vt have shapes ``(..., m, m)`` and
        ``(..., n, n)``. If False, shapes are ``(..., m, K)`` and
        ``(..., K, n)`` where K = min(m, n) or k if specified.
    compute_uv : bool, optional
        If True (default), compute U and Vt. If False, only compute
        singular values.
    hermitian : bool, optional
        If True, a is assumed to be Hermitian. Default is False.
    k : int, optional
        **flopscope extension.** Number of singular values/vectors to return.
        Must satisfy 1 <= k <= min(m, n). If None (default), returns
        all min(m, n) components.

    Returns
    -------
    When compute_uv is True (default):
        U : ndarray, S : ndarray, Vt : ndarray
    When compute_uv is False:
        S : ndarray
    """
    budget = require_budget()
    inputs_were_whest = isinstance(a, FlopscopeArray)
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim < 2:
        raise ValueError(f"svd requires a 2D (or batched) array, got shape {a.shape}")
    m, n = a.shape[-2], a.shape[-1]
    if k is not None and min(m, n) > 0 and not (1 <= k <= min(m, n)):
        raise ValueError(f"k must satisfy 1 <= k <= min(m, n) = {min(m, n)}, got k={k}")
    batch = _batch_size(a.shape)
    cost = (
        svd_cost(m, n, k, with_vectors=compute_uv, full_matrices=full_matrices) * batch
        if not _has_zero_dim(a.shape)
        else 0
    )
    with budget.deduct(
        "linalg.svd",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
        dtypes=linalg_billing_dtypes(a.dtype),
    ):
        if compute_uv:
            # When k is specified, always use economy decomposition then slice.
            fm = full_matrices if k is None else False
            U, S, Vt = _call_numpy(
                _np.linalg.svd,
                _to_base_ndarray(a),
                full_matrices=fm,
                hermitian=hermitian,
            )
            if k is not None:
                S = S[..., :k]
                U = U[..., :k]
                Vt = Vt[..., :k, :]
            maybe_check_nan_inf(S, "linalg.svd")
        else:
            S = _call_numpy(
                _np.linalg.svd,
                _to_base_ndarray(a),
                compute_uv=False,
                hermitian=hermitian,
            )
            if k is not None:
                S = S[..., :k]
            maybe_check_nan_inf(S, "linalg.svd")

    if compute_uv:
        if inputs_were_whest:
            return SVDResult(_asflopscope(U), _asflopscope(S), _asflopscope(Vt))  # type: ignore[reportPossiblyUnbound]
        return SVDResult(U, S, Vt)  # type: ignore[reportPossiblyUnbound]
    else:
        if inputs_were_whest:
            return _asflopscope(S)  # type: ignore[reportReturnType]
        return S  # type: ignore[reportReturnType]
