"""Tests for _get_accumulation_cost and its cache."""

import numpy as np

import flopscope as fps
from flopscope._einsum import _accumulation_cache, _get_accumulation_cost


def test_get_accumulation_cost_returns_AccumulationCost():
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    cost = _get_accumulation_cost(
        canonical_subscripts="ij,jk->ik",
        input_parts=("ij", "jk"),
        output_subscript="ik",
        shapes=((3, 3), (3, 3)),
        operands=(A, B),
    )
    assert isinstance(cost, fps.AccumulationCost)


def test_accumulation_cache_is_hit_on_repeat():
    _accumulation_cache.cache_clear()
    A = np.zeros((4, 4))
    _get_accumulation_cost(
        canonical_subscripts="ij,jk->ik",
        input_parts=("ij", "jk"),
        output_subscript="ik",
        shapes=((4, 4), (4, 4)),
        operands=(A, A),
    )
    info1 = _accumulation_cache.cache_info()
    _get_accumulation_cost(
        canonical_subscripts="ij,jk->ik",
        input_parts=("ij", "jk"),
        output_subscript="ik",
        shapes=((4, 4), (4, 4)),
        operands=(A, A),
    )
    info2 = _accumulation_cache.cache_info()
    assert info2.hits == info1.hits + 1


def test_einsum_accumulation_cost_public_function_uses_cache():
    """Public einsum_accumulation_cost should hit the cache on repeat calls."""
    A = np.zeros((4, 4))
    A_sym = fps.as_symmetric(A, symmetry=(0, 1))

    _accumulation_cache.cache_clear()
    fps.einsum_accumulation_cost("ij,j->i", A_sym, np.zeros(4))
    info1 = _accumulation_cache.cache_info()

    fps.einsum_accumulation_cost("ij,j->i", A_sym, np.zeros(4))
    info2 = _accumulation_cache.cache_info()

    assert info2.hits == info1.hits + 1, (
        f"expected cache hit on repeat call; "
        f"before: hits={info1.hits} misses={info1.misses}; "
        f"after: hits={info2.hits} misses={info2.misses}"
    )


def test_einsum_accumulation_cost_partition_budget_in_cache_key():
    """Different partition_budget values should NOT share cache entries."""
    A = np.zeros((3, 3))
    _accumulation_cache.cache_clear()
    fps.einsum_accumulation_cost("ij,jk->ik", A, A, partition_budget=10_000)
    fps.einsum_accumulation_cost("ij,jk->ik", A, A, partition_budget=20_000)
    info = _accumulation_cache.cache_info()
    # Two distinct budgets → two misses (no false cache hit).
    assert info.misses == 2, (
        f"expected 2 misses for distinct partition_budget values, got {info}"
    )
