"""Tests for AccumulationCost fallback to dense when a component is unavailable."""

import warnings

import pytest

from flopscope._accumulation._cost import (
    ComponentCost,
    aggregate_einsum,
)
from flopscope.errors import CostFallbackWarning


def _unavailable_component(labels=("i", "j"), sizes=(3, 3), dense=9):
    return ComponentCost(
        labels=labels,
        va=labels[:1],
        wa=labels[1:],
        sizes=sizes,
        m=4,
        alpha=None,
        dense_count=dense,
        num_output_orbits=sizes[0],
        regime_id="unavailable",
        shape="mixed",
        group_name="S?{i,j}",
        group_order=2,
        regime_trace=(),
        unavailable_reason="partition budget 0 exceeded",
    )


def _trivial_component(label="k", size=4):
    return ComponentCost(
        labels=(label,),
        va=(label,),
        wa=(),
        sizes=(size,),
        m=size,
        alpha=size,
        dense_count=size,
        num_output_orbits=size,
        regime_id="trivial",
        shape="trivial",
        group_name="trivial",
        group_order=1,
        regime_trace=(),
    )


def test_fallback_total_is_k_times_dense_baseline():
    c_avail = _trivial_component()
    c_unavail = _unavailable_component()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CostFallbackWarning)
        cost = aggregate_einsum(
            component_costs=(c_avail, c_unavail),
            num_terms=3,
            dense_baseline=4 * 9,  # 36
        )
    # Fallback: total = k · dense_baseline = 3 · 36 = 108
    assert cost.total == 108
    assert cost.fallback_used is True
    assert cost.alpha is None
    assert cost.mu is None
    assert cost.m_total == 4 * 4  # M is always computable


def test_fallback_emits_cost_fallback_warning():
    c = _unavailable_component()
    with pytest.warns(CostFallbackWarning, match="partition_budget"):
        aggregate_einsum(component_costs=(c,), num_terms=1, dense_baseline=9)


def test_fallback_records_unavailable_components_indices():
    c1 = _trivial_component()
    c2 = _unavailable_component()
    c3 = _unavailable_component()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CostFallbackWarning)
        cost = aggregate_einsum(
            component_costs=(c1, c2, c3),
            num_terms=2,
            dense_baseline=4 * 9 * 9,
        )
    assert cost.unavailable_components == (1, 2)
    assert cost.unavailable_reason == "partition budget 0 exceeded"


def test_fallback_does_not_fire_when_all_components_available():
    c1 = _trivial_component("i", 3)
    c2 = _trivial_component("j", 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CostFallbackWarning)
        # Must not raise — no warning should fire.
        cost = aggregate_einsum(
            component_costs=(c1, c2),
            num_terms=2,
            dense_baseline=12,
        )
    assert cost.fallback_used is False
