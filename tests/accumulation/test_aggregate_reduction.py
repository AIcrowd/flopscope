"""Tests for aggregate_reduction body (formerly NotImplementedError stub)."""

import pytest

from flopscope._accumulation._cost import (
    AccumulationCost,
    ComponentCost,
    aggregate_reduction,
)
from flopscope.errors import CostFallbackWarning


def _trivial_component(*, m: int, alpha: int, output_orbits: int) -> ComponentCost:
    """A 1-axis trivial component with explicit m/alpha."""
    return ComponentCost(
        labels=("i",),
        va=("i",),
        wa=(),
        sizes=(output_orbits,),
        m=m,
        alpha=alpha,
        dense_count=m,
        num_output_orbits=output_orbits,
        regime_id="trivial",
        shape="trivial",
        group_name="trivial",
        group_order=1,
        regime_trace=(),
    )


def test_single_component_op_factor_1_no_extra():
    # 1 component, α=10, output_dense=1 (will be num_output_orbits in real use).
    # cost = 1 × (10 - 1) + 0 = 9
    comp = _trivial_component(m=10, alpha=10, output_orbits=1)
    cost = aggregate_reduction(
        [comp],
        op_factor=1,
        dense_baseline=10,
        output_dense=1,
        extra_ops=0,
    )
    assert cost.total == 9


def test_mean_alias_extra_ops_equals_output_orbits():
    # Mean(a) = sum + 1 divide. α=10, output_dense=1, extra_ops=1.
    # cost = 1 × (10 - 1) + 1 = 10
    comp = _trivial_component(m=10, alpha=10, output_orbits=1)
    cost = aggregate_reduction(
        [comp],
        op_factor=1,
        dense_baseline=10,
        output_dense=1,
        extra_ops=1,
    )
    assert cost.total == 10


def test_op_factor_2_doubles_per_event_cost():
    # Sum of squares: op_factor=2.
    # cost = 2 × (10 - 1) + 0 = 18
    comp = _trivial_component(m=10, alpha=10, output_orbits=1)
    cost = aggregate_reduction(
        [comp],
        op_factor=2,
        dense_baseline=10,
        output_dense=1,
        extra_ops=0,
    )
    assert cost.total == 18


def test_two_components_alpha_product_minus_output_dense():
    # Two components: α_total = α_1 × α_2 = 6 × 4 = 24
    # output_dense passed by orchestrator = num_output_orbits = 6
    # cost = 1 × (24 - 6) + 0 = 18
    c1 = _trivial_component(m=6, alpha=6, output_orbits=3)
    c2 = _trivial_component(m=4, alpha=4, output_orbits=2)
    cost = aggregate_reduction(
        [c1, c2],
        op_factor=1,
        dense_baseline=24,
        output_dense=6,
        extra_ops=0,
    )
    assert cost.total == 18


def test_unavailable_component_triggers_fallback():
    # When alpha is None for a component, fallback formula applies:
    # total = output_dense × (input_axis_size - 1) × op_factor + extra_ops
    # input_axis_size = dense_baseline / output_dense.
    comp = ComponentCost(
        labels=("i",),
        va=("i",),
        wa=(),
        sizes=(1,),
        m=1,
        alpha=None,
        dense_count=1,
        num_output_orbits=1,
        regime_id="unavailable",
        shape="trivial",
        group_name="unknown",
        group_order=1,
        regime_trace=(),
    )
    with pytest.warns(CostFallbackWarning, match="partition_budget"):
        cost = aggregate_reduction(
            [comp],
            op_factor=1,
            dense_baseline=10,
            output_dense=1,
            extra_ops=0,
        )
    # input_axis_size = 10/1 = 10. cost = 1 × (10 - 1) × 1 + 0 = 9
    assert cost.total == 9
    assert cost.fallback_used is True


def test_returns_accumulation_cost_instance():
    comp = _trivial_component(m=5, alpha=5, output_orbits=1)
    cost = aggregate_reduction(
        [comp],
        op_factor=1,
        dense_baseline=5,
        output_dense=1,
        extra_ops=0,
    )
    assert isinstance(cost, AccumulationCost)
    assert cost.num_terms == 1
    assert cost.per_component == (comp,)
