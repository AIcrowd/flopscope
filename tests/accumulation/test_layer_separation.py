"""Tests verifying the layered API: shared primitives, einsum aggregator,
and the reduction aggregator stub."""


def test_decompose_into_components_callable_without_einsum_context():
    from flopscope._accumulation._components import decompose_into_components

    assert callable(decompose_into_components)


def test_run_ladder_per_component_is_pure():
    from flopscope._accumulation._components import Component
    from flopscope._accumulation._cost import run_ladder_per_component
    from flopscope._perm_group import _Permutation as Permutation

    c = Component(
        indices=(0,),
        labels=("i",),
        va=("i",),
        wa=(),
        sizes=(3,),
        visible_positions=(0,),
        generators=(),
        elements=(Permutation.identity(1),),
        order=1,
        group_name="trivial",
    )
    out_a = run_ladder_per_component((c,), partition_budget=100_000)
    out_b = run_ladder_per_component((c,), partition_budget=100_000)
    assert out_a == out_b


def test_aggregate_einsum_signature_matches_spec():
    import inspect

    from flopscope._accumulation._cost import aggregate_einsum

    sig = inspect.signature(aggregate_einsum)
    params = list(sig.parameters.keys())
    assert "component_costs" in params
    assert "num_terms" in params
    assert "dense_baseline" in params


def test_aggregate_reduction_signature_locked_for_future_sprint():
    import inspect

    from flopscope._accumulation._cost import aggregate_reduction

    sig = inspect.signature(aggregate_reduction)
    params = list(sig.parameters.keys())
    assert "component_costs" in params
    assert "op_factor" in params
    assert "dense_baseline" in params
    assert "output_dense" in params
    assert "extra_ops" in params


def test_aggregate_reduction_is_implemented():
    # Now that the body is implemented, calling with an empty component list
    # should return an AccumulationCost with total=0 (α_product=1, output_dense=1 → 0).
    from flopscope._accumulation._cost import AccumulationCost, aggregate_reduction

    cost = aggregate_reduction(
        component_costs=(),
        op_factor=1,
        dense_baseline=1,
        output_dense=1,
        extra_ops=0,
    )
    assert isinstance(cost, AccumulationCost)
    assert cost.fallback_used is False
