"""Public flopscope.reduction_accumulation_cost wrapper tests."""

import flopscope as fps
import flopscope.numpy as fnp


def test_public_function_exists_at_top_level():
    assert callable(fps.reduction_accumulation_cost)
    assert "reduction_accumulation_cost" in fps.__all__


def test_plain_ndarray_sum_full_axis_returns_n_minus_1():
    a = fnp.zeros(10)
    cost = fps.reduction_accumulation_cost(a, axis=0)
    assert cost.total == 9


def test_symmetric_tensor_input_extracts_symmetry():
    n = 4
    T = fps.as_symmetric(fnp.zeros((n, n, n)), symmetry=(0, 1, 2))
    cost = fps.reduction_accumulation_cost(T, axis=2)
    # n(n+1)/2 output orbits; α determined by ladder; total > 0
    assert cost.total > 0
    assert cost.per_component


def test_axis_none_reduces_all():
    a = fnp.zeros((3, 4))
    cost = fps.reduction_accumulation_cost(a, axis=None)
    # Full reduce: dense baseline = 12, output_orbits = 1.
    # cost = 12 - 1 = 11.
    assert cost.total == 11


def test_extra_ops_kwarg_threaded_through():
    a = fnp.zeros(10)
    cost = fps.reduction_accumulation_cost(a, axis=0, extra_ops=1)
    # 9 (sum) + 1 (extra) = 10 (matches mean's behavior)
    assert cost.total == 10


def test_partition_budget_resolves_before_cache():
    """If partition_budget=None is passed and the global setting changes
    between calls, the second call must reflect the new setting (not return
    a stale cached entry under the None key)."""
    from flopscope._accumulation._cache import _reduction_cache

    _reduction_cache.cache_clear()
    a = fnp.zeros(10)

    # First call: partition_budget=None resolves to default 100_000 setting.
    cost1 = fps.reduction_accumulation_cost(a, axis=0)

    # Change the setting, call again with partition_budget=None.
    fps.configure(partition_budget=50_000)
    try:
        cost2 = fps.reduction_accumulation_cost(a, axis=0)
        info = _reduction_cache.cache_info()
        # Two different effective partition budgets → two cache misses.
        assert info.misses == 2, (
            f"expected 2 misses for distinct effective budgets, got {info}"
        )
    finally:
        fps.configure(partition_budget=100_000)
