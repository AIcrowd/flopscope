"""Factorizations numpy picks by option are billed as the one it runs (#280).

``Generator.multivariate_normal(..., method=)`` factors the covariance with an
SVD, ``eigh`` or Cholesky, and ``linalg.svd``/``pinv``/``matrix_rank`` with
``hermitian=True`` run ``eigh``/``eigvalsh`` instead of an SVD. Every one of
these was billed as the full SVD.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._flops import svd_cost
from flopscope._weights import load_weights


@pytest.fixture
def production_weights(monkeypatch):
    """Load the packaged production weights for one test (conftest resets them)."""
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    load_weights()
    yield


def _billed(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=2**62, quiet=True) as ctx:
            before = int(ctx.flops_used)
            out = fn()
            return int(ctx.flops_used) - before, out


def _spd(n, seed=0, batch=()):
    x = np.random.default_rng(seed).standard_normal((*batch, n, n))
    return x @ np.swapaxes(x, -1, -2) / n + np.eye(n)


def _sort_cost(n):
    return n * max(math.ceil(math.log2(n)), 1) if n > 1 else 1


def _hermitian_path(n, with_vectors):
    """Closed form of numpy's hermitian=True branch for one n x n matrix."""
    if with_vectors:
        return 9 * n**3 + 2 * n + 4 * _sort_cost(n) + 4 * (2 * n + n * n) + 2 * n * n
    return 4 * n**3 // 3 + n + 4 * _sort_cost(n)


# --- Generator.multivariate_normal(method=) ---------------------------------


def test_issue_repro_method_bills_its_factorization(production_weights):
    d = 256
    C = _spd(d)
    mean = np.zeros(d)
    svd, _ = _billed(lambda: fnp.linalg.svd(C))
    by_method = {}
    for method in ("svd", "eigh", "cholesky"):
        rng = fnp.random.default_rng(1)
        by_method[method], _ = _billed(
            lambda rng=rng, method=method: rng.multivariate_normal(
                mean, C, method=method
            )
        )
    sampling = by_method["svd"] - svd
    eigh, _ = _billed(lambda: fnp.linalg.eigh(C))
    cholesky, _ = _billed(lambda: fnp.linalg.cholesky(C))
    assert by_method["svd"] == 872_685_568
    assert by_method["eigh"] == eigh + sampling == 302_260_224
    assert by_method["cholesky"] == cholesky + sampling == 11_455_146


@pytest.mark.parametrize(
    ("method", "factorization"),
    [
        ("svd", lambda d: 26 * d**3),
        ("eigh", lambda d: 9 * d**3),
        ("cholesky", lambda d: d**3 // 3),
    ],
)
@pytest.mark.parametrize(("d", "size"), [(5, 7), (12, (3, 4))])
def test_method_closed_form(method, factorization, d, size):
    C = _spd(d, seed=2)
    rng = fnp.random.default_rng(3)
    cost, out = _billed(
        lambda: rng.multivariate_normal(np.zeros(d), C, size=size, method=method)
    )
    n = int(np.prod(size))
    assert np.asarray(out).shape == (*np.atleast_1d(size), d)
    assert cost == factorization(d) + 2 * n * d * d + 16 * n * d


def test_draws_match_numpy_for_every_method():
    d = 6
    C = _spd(d, seed=4)
    for method in ("svd", "eigh", "cholesky"):
        ours = fnp.random.default_rng(5)
        theirs = np.random.default_rng(5)
        _, out = _billed(
            lambda ours=ours, method=method: ours.multivariate_normal(
                np.zeros(d), C, size=3, method=method
            )
        )
        expected = theirs.multivariate_normal(np.zeros(d), C, size=3, method=method)
        assert np.allclose(np.asarray(out), expected)


def test_svd_only_samplers_are_unchanged():
    d, n = 7, 9
    C = _spd(d, seed=6)
    expected = 26 * d**3 + 2 * n * d * d + 16 * n * d
    legacy, _ = _billed(
        lambda: fnp.random.RandomState(0).multivariate_normal(np.zeros(d), C, n)
    )
    module, _ = _billed(lambda: fnp.random.multivariate_normal(np.zeros(d), C, n))
    assert legacy == module == expected


# --- hermitian=True: svd, pinv, matrix_rank ---------------------------------


def _spelled_hermitian_svd(a):
    """numpy's hermitian=True branch of linalg.svd, one metered op per step."""
    s, u = fnp.linalg.eigh(a)
    sgn = fnp.sign(s)
    s = fnp.abs(s)
    order = fnp.argsort(s)[..., ::-1]
    sgn = fnp.take_along_axis(sgn, order, axis=-1)
    s = fnp.take_along_axis(s, order, axis=-1)
    u = fnp.take_along_axis(u, order[..., None, :], axis=-1)
    vt = fnp.conjugate(fnp.matrix_transpose(u * sgn[..., None, :]))
    return u, s, vt


def _spelled_hermitian_svdvals(a):
    s = fnp.abs(fnp.linalg.eigvalsh(a))
    return fnp.sort(s)[..., ::-1]


@pytest.mark.parametrize(
    ("n", "batch"), [(1, ()), (2, ()), (5, ()), (64, ()), (5, (3,))]
)
def test_hermitian_svd_bills_the_eigh_path_numpy_runs(production_weights, n, batch):
    C = _spd(n, seed=n, batch=batch)
    cost, (u, s, vt) = _billed(lambda: fnp.linalg.svd(C, hermitian=True))
    spelled, _ = _billed(lambda: _spelled_hermitian_svd(C))
    values, sv = _billed(lambda: fnp.linalg.svd(C, compute_uv=False, hermitian=True))
    spelled_values, _ = _billed(lambda: _spelled_hermitian_svdvals(C))
    assert cost == spelled
    assert values == spelled_values
    ref_u, ref_s, ref_vt = np.linalg.svd(C, hermitian=True)
    assert np.allclose(np.asarray(s), ref_s) and np.allclose(np.asarray(sv), ref_s)
    assert np.allclose(np.asarray(u) * np.asarray(s)[..., None, :] @ np.asarray(vt), C)


def test_issue_repro_hermitian_rows(production_weights):
    C = _spd(256)
    svd, _ = _billed(lambda: fnp.linalg.svd(C))
    svd_h, _ = _billed(lambda: fnp.linalg.svd(C, hermitian=True))
    eigh, _ = _billed(lambda: fnp.linalg.eigh(C))
    assert svd == 872_415_232
    assert eigh < svd_h < eigh * 1.01
    pinv, _ = _billed(lambda: fnp.linalg.pinv(C))
    pinv_h, _ = _billed(lambda: fnp.linalg.pinv(C, hermitian=True))
    assert pinv_h == pinv - svd + svd_h
    rank, _ = _billed(lambda: fnp.linalg.matrix_rank(C))
    rank_h, _ = _billed(lambda: fnp.linalg.matrix_rank(C, hermitian=True))
    values_h, _ = _billed(lambda: fnp.linalg.svd(C, compute_uv=False, hermitian=True))
    assert rank_h == values_h + 2 * 256 < rank


@pytest.mark.parametrize("n", [3, 16])
def test_hermitian_closed_forms(n):
    C = _spd(n, seed=7)
    vectors = _hermitian_path(n, True)
    values = _hermitian_path(n, False)
    assert _billed(lambda: fnp.linalg.svd(C, hermitian=True))[0] == vectors
    assert (
        _billed(lambda: fnp.linalg.svd(C, compute_uv=False, hermitian=True))[0]
        == values
    )
    pinv, out = _billed(lambda: fnp.linalg.pinv(C, hermitian=True))
    assert pinv == vectors + n + n * n + (2 * n * n * n - n * n)
    assert np.allclose(np.asarray(out), np.linalg.pinv(C, hermitian=True))
    rank, r = _billed(lambda: fnp.linalg.matrix_rank(C, hermitian=True))
    assert rank == values + n
    assert int(r) == np.linalg.matrix_rank(C, hermitian=True)


@pytest.mark.parametrize(
    ("k", "compute_uv"), [(2, True), (2, False), (40, False), (63, True)]
)
def test_hermitian_top_k_bills_the_cheaper_of_the_two(k, compute_uv):
    n = 64
    C = _spd(n, seed=8)
    cost, _ = _billed(
        lambda: fnp.linalg.svd(C, compute_uv=compute_uv, hermitian=True, k=k)
    )
    top_k = svd_cost(n, n, k, with_vectors=compute_uv)
    assert cost == min(top_k, _hermitian_path(n, compute_uv))


def test_hermitian_false_and_non_square_are_unchanged():
    C = _spd(9, seed=9)
    explicit, _ = _billed(lambda: fnp.linalg.svd(C, hermitian=False))
    assert explicit == svd_cost(9, 9, with_vectors=True, full_matrices=True)
    # numpy has no hermitian path for a non-square matrix: it refuses the call.
    A = np.random.default_rng(10).standard_normal((6, 4))
    for f in (fnp.linalg.svd, fnp.linalg.pinv, fnp.linalg.matrix_rank):
        with flops.BudgetContext(flop_budget=2**62, quiet=True):
            with pytest.raises(np.linalg.LinAlgError):
                f(A, hermitian=True)
