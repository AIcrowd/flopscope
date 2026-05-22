"""Tests for the public einsum_clear_caches / einsum_cache_info / fma_cost API."""

from __future__ import annotations

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp


def test_einsum_cache_info_keys():
    info = flops.einsum_cache_info()
    assert set(info.keys()) == {"path", "accumulation"}
    for key, ci in info.items():
        assert hasattr(ci, "hits"), f"{key}: missing .hits attribute"
        assert hasattr(ci, "misses"), f"{key}: missing .misses attribute"
        assert hasattr(ci, "maxsize"), f"{key}: missing .maxsize attribute"
        assert hasattr(ci, "currsize"), f"{key}: missing .currsize attribute"


def test_einsum_clear_caches_clears_both():
    """A call should drop both currsize counters to 0."""
    # Warm both caches.
    A = np.zeros((4, 4))
    flops.einsum_accumulation_cost("ij,jk->ik", A, A)
    with flops.BudgetContext(flop_budget=10**9):
        fnp.einsum_path("ij,jk->ik", A, A)

    before = flops.einsum_cache_info()
    assert before["path"].currsize >= 1, before
    assert before["accumulation"].currsize >= 1, before

    flops.einsum_clear_caches()

    after = flops.einsum_cache_info()
    assert after["path"].currsize == 0, after
    assert after["accumulation"].currsize == 0, after


def test_einsum_clear_caches_independent_of_fnp_clear():
    """The fnp.clear_einsum_cache() narrow version still only clears the path cache."""
    A = np.zeros((4, 4))
    flops.einsum_accumulation_cost("ij,jk->ik", A, A)
    with flops.BudgetContext(flop_budget=10**9):
        fnp.einsum_path("ij,jk->ik", A, A)

    # fnp.clear_einsum_cache only touches the path cache.
    fnp.clear_einsum_cache()
    info = flops.einsum_cache_info()
    assert info["path"].currsize == 0
    assert info["accumulation"].currsize >= 1, (
        "fnp.clear_einsum_cache must NOT touch the accumulation cache"
    )

    # The unified version clears both.
    flops.einsum_clear_caches()
    info = flops.einsum_cache_info()
    assert info["path"].currsize == 0
    assert info["accumulation"].currsize == 0


def test_public_api_in_all():
    assert "einsum_clear_caches" in flops.__all__
    assert "einsum_cache_info" in flops.__all__
    assert "clear_cache" in flops.__all__
    assert "reduction_clear_cache" in flops.__all__
    assert "reduction_cache_info" in flops.__all__
    assert "tier2_reduction_cost" in flops.__all__


def test_reduction_cache_info_shape():
    info = flops.reduction_cache_info()
    assert hasattr(info, "hits")
    assert hasattr(info, "misses")
    assert hasattr(info, "maxsize")
    assert hasattr(info, "currsize")


def test_reduction_clear_cache():
    A = np.zeros((4, 4))
    flops.reduction_accumulation_cost(A)
    assert flops.reduction_cache_info().currsize >= 1
    flops.reduction_clear_cache()
    assert flops.reduction_cache_info().currsize == 0


def test_clear_cache_unified():
    """clear_cache() should drop currsize on both einsum and reduction caches."""
    A = np.zeros((4, 4))
    flops.einsum_accumulation_cost("ij,jk->ik", A, A)
    flops.reduction_accumulation_cost(A)
    with flops.BudgetContext(flop_budget=10**9):
        fnp.einsum_path("ij,jk->ik", A, A)

    flops.clear_cache()

    einsum_info = flops.einsum_cache_info()
    assert einsum_info["path"].currsize == 0
    assert einsum_info["accumulation"].currsize == 0
    assert flops.reduction_cache_info().currsize == 0


def test_tier2_reduction_cost_default_dense_per_output():
    """When dense_per_output_cost is None, defaults to product of reduced axes."""
    # axis=None → full reduction. dense_per_output_cost defaults to prod(shape).
    A = np.zeros(10)
    assert flops.tier2_reduction_cost(A) == 10

    # axis=0 on (6, 8). dense_per_output_cost defaults to shape[0] = 6.
    # num_output_orbits is shape[1] = 8 (unsymmetric).
    B = np.zeros((6, 8))
    assert flops.tier2_reduction_cost(B, axis=0) == 48

    # axis=1 on (6, 8). default dense = 8, num_output_orbits = 6.
    assert flops.tier2_reduction_cost(B, axis=1) == 48


def test_tier2_reduction_cost_explicit_dense():
    """Explicit dense_per_output_cost overrides the axis-product default."""
    A = np.zeros(10)
    # Pretend a Tier-2 op costs 100 per output instead of 10.
    assert flops.tier2_reduction_cost(A, dense_per_output_cost=100) == 100


def test_tier2_reduction_cost_uses_symmetry():
    """num_output_orbits respects declared symmetry on the input.

    S_3 on (4,4,4) reducing axis=0: stabilizer of axis 0 is S_2 on
    {1,2}, so the (4,4) output collapses 16 cells to 4*5/2 = 10 orbits.
    Dense per-output cost = 4 (axis 0 has length 4).
    Expected:  sym = 10 * 4 = 40   vs   dense = 16 * 4 = 64.
    """
    T = flops.as_symmetric(np.zeros((4, 4, 4)), symmetry=(0, 1, 2))
    cost = flops.tier2_reduction_cost(T, axis=0)
    dense_cost = flops.tier2_reduction_cost(np.zeros((4, 4, 4)), axis=0)
    assert cost == 40, f"expected sym=40, got {cost}"
    assert dense_cost == 64, f"expected dense=64, got {dense_cost}"
    assert cost < dense_cost
