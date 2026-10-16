"""Rank pricing covers public APIs, complete-factor recovery, and production rates."""

import numpy as np
import pytest

import flopscope.numpy as fnp
from flopscope._budget import BudgetContext
from flopscope._weights import load_weights
from flopscope.errors import BudgetExhaustedError


@pytest.mark.parametrize("name", ["vectors", "values", "svdvals"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_public_truncated_apis_bill_rank_work_and_preserve_numpy_values(
    name, transpose, batched
):
    matrix = np.random.default_rng(5).normal(size=(16, 8))
    if transpose:
        matrix = matrix.T
    data = np.stack([matrix, matrix * 2]) if batched else matrix
    reference_u, reference_s, reference_vt = np.linalg.svd(data, full_matrices=False)
    u = vt = None
    with BudgetContext(flop_budget=10**9, quiet=True) as budget:
        if name == "svdvals":
            actual = fnp.linalg.svdvals(data, k=2)
        elif name == "values":
            actual = fnp.linalg.svd(data, k=2, compute_uv=False)
        else:
            u, actual, vt = fnp.linalg.svd(data, k=2, full_matrices=True)
    if name == "vectors":
        assert u is not None and vt is not None
        np.testing.assert_allclose(
            (np.asarray(u) * np.asarray(actual)[..., None, :]) @ np.asarray(vt),
            (reference_u[..., :2] * reference_s[..., None, :2])
            @ reference_vt[..., :2, :],
            atol=1e-12,
        )
    np.testing.assert_allclose(actual, reference_s[..., :2])
    assert budget.flops_used == (1760 if name == "vectors" else 1232) * (
        2 if batched else 1
    )


@pytest.mark.parametrize(
    "dtype,multiplier", [(np.float32, 1), (np.float64, 2), (np.complex128, 8)]
)
@pytest.mark.parametrize("vectors", [False, True])
def test_production_dtype_and_complex_multipliers_are_unchanged(
    dtype, multiplier, vectors
):
    load_weights()
    matrix = np.random.default_rng(7).normal(size=(16, 8)).astype(dtype)
    with BudgetContext(flop_budget=10**9, quiet=True) as budget:
        fnp.linalg.svd(matrix, k=2, compute_uv=vectors)
    assert budget.flops_used == (1760 if vectors else 1232) * multiplier


def test_missing_triplet_reconstruction_no_longer_discounts_large_full_svd():
    n = 16
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(n, n))
    with BudgetContext(flop_budget=10**9, quiet=True) as budget:
        u, s, vt = fnp.linalg.svd(matrix, k=n - 1)
    # The report's quadratic complement construction still works numerically.
    left = rng.normal(size=n)
    left -= np.asarray(u) @ (np.asarray(u).T @ left)
    left /= np.linalg.norm(left)
    right = rng.normal(size=n)
    right -= np.asarray(vt).T @ (np.asarray(vt) @ right)
    right /= np.linalg.norm(right)
    coefficient = left @ matrix @ right
    reconstructed = (np.asarray(u) * np.asarray(s)) @ np.asarray(vt)
    reconstructed += coefficient * np.outer(left, right)
    np.testing.assert_allclose(reconstructed, matrix, atol=1e-11)
    assert budget.flops_used == 26 * n**3


def test_padded_topk_returns_original_complete_factors_without_price_discount():
    n = 16
    matrix = np.random.default_rng(9).normal(size=(n, n))
    padded = np.pad(matrix, ((0, 1), (0, 1)))
    with BudgetContext(flop_budget=10**9, quiet=True) as budget:
        u, s, vt = fnp.linalg.svd(padded, k=n)
    np.testing.assert_allclose(
        (np.asarray(u)[:-1] * np.asarray(s)) @ np.asarray(vt)[:, :-1],
        matrix,
        atol=1e-12,
    )
    assert budget.flops_used >= 26 * n**3


def test_new_rank_price_is_checked_before_backend_execution(monkeypatch):
    matrix = np.ones((10, 5))

    def unexpected(*args, **kwargs):
        pytest.fail("SVD backend executed despite an insufficient budget")

    monkeypatch.setattr(np.linalg, "svd", unexpected)
    with BudgetContext(flop_budget=1949, quiet=True) as budget:
        with pytest.raises(BudgetExhaustedError):
            fnp.linalg.svd(matrix, k=3)
    assert budget.flops_used == 0
