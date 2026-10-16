"""Unit tests for rank-dependent SVD pricing and full-decomposition caps."""

import pytest

from flopscope._flops import svd_cost


def economy(m, n, with_vectors):
    a, b = max(m, n), min(m, n)
    return (6 * a * b * b + 20 * b**3) if with_vectors else (2 * a * b * b + 2 * b**3)


def test_topk_discount_small_k():
    # Two matrix passes, basis/factor work, and a small vector-returning SVD.
    assert svd_cost(128, 64, 8, with_vectors=True) == 346_112


def test_values_only_has_its_own_quadratic_and_cubic_terms():
    assert svd_cost(128, 64, 8, with_vectors=False) == 287_744
    assert svd_cost(128, 64, 8, with_vectors=False) < svd_cost(
        128, 64, 8, with_vectors=True
    )


@pytest.mark.parametrize("m,n", [(128, 64), (64, 128), (64, 64), (200, 10), (10, 200)])
@pytest.mark.parametrize("with_vectors", [True, False])
@pytest.mark.parametrize("full_matrices", [True, False])
def test_cap_never_exceeds_full(m, n, with_vectors, full_matrices):
    full = svd_cost(m, n, None, with_vectors=with_vectors, full_matrices=full_matrices)
    for k in range(1, min(m, n) + 1):
        assert (
            svd_cost(m, n, k, with_vectors=with_vectors, full_matrices=full_matrices)
            <= full
        )


@pytest.mark.parametrize("m,n", [(128, 64), (64, 128), (64, 64), (200, 10), (10, 200)])
@pytest.mark.parametrize("with_vectors", [True, False])
def test_all_components_via_k_bills_full_economy(m, n, with_vectors):
    # Asking for all min(m,n) components via k bills economy, the corresponding full decomposition.
    b = min(m, n)
    assert svd_cost(m, n, b, with_vectors=with_vectors) == economy(m, n, with_vectors)
    assert svd_cost(m, n, b, with_vectors=with_vectors) == svd_cost(
        m, n, None, with_vectors=with_vectors, full_matrices=False
    )


def test_full_matrices_premium_only_for_k_none():
    # Non-square with vectors: full-U premium applies ONLY to k=None.
    assert svd_cost(10, 5, None, with_vectors=True, full_matrices=True) == 4750
    # k given (even k == min) -> economy, no premium
    assert svd_cost(10, 5, 5, with_vectors=True, full_matrices=True) == 4000
    assert svd_cost(10, 5, 3, with_vectors=True, full_matrices=True) == 1950


@pytest.mark.parametrize("m,n", [(128, 64), (64, 128), (64, 64), (200, 10)])
@pytest.mark.parametrize("with_vectors", [True, False])
def test_monotonic_non_decreasing_in_k(m, n, with_vectors):
    costs = [
        svd_cost(m, n, k, with_vectors=with_vectors) for k in range(1, min(m, n) + 1)
    ]
    assert all(costs[i] <= costs[i + 1] for i in range(len(costs) - 1))


def test_full_decomposition_unchanged():
    # Regression guard: k=None paths are exactly the pre-change full costs.
    assert svd_cost(100, 50, None) == 750_000  # values-only
    assert (
        svd_cost(10, 5, None, with_vectors=True, full_matrices=True) == 4750
    )  # full U
    assert svd_cost(10, 5, None, with_vectors=True, full_matrices=False) == 4000  # thin


def test_values_only_cheaper_at_cap_region():
    # Near the economy cap, values-only (capped at 2ab^2+2b^3) can be cheaper
    # than with-vectors (capped at 6ab^2+20b^3); the caps differ. Intentional.
    # m=10,n=5,k=4: values-only caps at 750; vectors cost 3520.
    assert svd_cost(10, 5, 4, with_vectors=False) == 750
    assert svd_cost(10, 5, 4, with_vectors=True) == 3520
    assert svd_cost(10, 5, 4, with_vectors=False) < svd_cost(
        10, 5, 4, with_vectors=True
    )


@pytest.mark.parametrize("m,n", [(8, 8), (16, 16), (256, 256), (64, 16), (16, 64)])
@pytest.mark.parametrize("with_vectors", [True, False])
def test_near_full_rank_reaches_the_respective_full_price(m, n, with_vectors):
    assert svd_cost(m, n, min(m, n) - 1, with_vectors=with_vectors) == economy(
        m, n, with_vectors
    )


@pytest.mark.parametrize("m,n", [(8, 8), (256, 256), (64, 16), (16, 64)])
@pytest.mark.parametrize("padding", [1, 4])
@pytest.mark.parametrize("with_vectors", [True, False])
def test_zero_padding_cannot_discount_complete_original_factors(
    m, n, padding, with_vectors
):
    original = economy(m, n, with_vectors)
    assert (
        svd_cost(m + padding, n + padding, min(m, n), with_vectors=with_vectors)
        >= original
    )


def test_small_matrix_truncation_keeps_the_approved_formula():
    # Do not claim that every possible small-matrix reconstruction is prevented.
    assert svd_cost(4, 4, 3, with_vectors=True) == 1164
