"""LRU cache behavior for reduction-cost lookups."""

from flopscope._accumulation._cache import (
    _reduction_cache,
    get_reduction_cost_cached,
)


def test_cache_hit_on_repeat():
    _reduction_cache.cache_clear()
    fp = ((0, 1), ((1, 0),))  # axes + array_form of single generator
    get_reduction_cost_cached(
        input_shape=(4, 4),
        axes_summed=(1,),
        sym_fingerprint=fp,
        op_factor=1,
        extra_ops=0,
        partition_budget=100_000,
    )
    info_after_miss = _reduction_cache.cache_info()
    get_reduction_cost_cached(
        input_shape=(4, 4),
        axes_summed=(1,),
        sym_fingerprint=fp,
        op_factor=1,
        extra_ops=0,
        partition_budget=100_000,
    )
    info_after_hit = _reduction_cache.cache_info()
    assert info_after_hit.hits == info_after_miss.hits + 1


def test_different_op_factor_is_a_miss():
    _reduction_cache.cache_clear()
    get_reduction_cost_cached(
        input_shape=(4,),
        axes_summed=(0,),
        sym_fingerprint=(None,),
        op_factor=1,
        extra_ops=0,
        partition_budget=100_000,
    )
    get_reduction_cost_cached(
        input_shape=(4,),
        axes_summed=(0,),
        sym_fingerprint=(None,),
        op_factor=2,
        extra_ops=0,
        partition_budget=100_000,
    )
    assert _reduction_cache.cache_info().misses == 2


def test_different_extra_ops_is_a_miss():
    _reduction_cache.cache_clear()
    get_reduction_cost_cached(
        input_shape=(4,),
        axes_summed=(0,),
        sym_fingerprint=(None,),
        op_factor=1,
        extra_ops=0,
        partition_budget=100_000,
    )
    get_reduction_cost_cached(
        input_shape=(4,),
        axes_summed=(0,),
        sym_fingerprint=(None,),
        op_factor=1,
        extra_ops=1,
        partition_budget=100_000,
    )
    assert _reduction_cache.cache_info().misses == 2
