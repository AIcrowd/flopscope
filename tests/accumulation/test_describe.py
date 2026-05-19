"""Tests for ComponentCost.describe() and AccumulationCost.describe()."""

from flopscope._accumulation._cost import AccumulationCost, ComponentCost
from flopscope._accumulation._shape import Shape


def _comp(regime_id, shape: Shape = "mixed"):
    return ComponentCost(
        labels=("i",),
        va=("i",),
        wa=(),
        sizes=(3,),
        m=3,
        alpha=3,
        dense_count=3,
        num_output_orbits=3,
        regime_id=regime_id,
        shape=shape,
        group_name="S2{i,j}",
        group_order=2,
        regime_trace=(),
    )


def test_describe_component_includes_latex_for_each_regime():
    for regime in [
        "trivial",
        "functionalProjection",
        "singleton",
        "young",
        "partitionCount",
    ]:
        d = _comp(regime).describe()
        assert "latex" in d
        assert d["latex"]  # non-empty


def test_describe_component_unavailable_has_distinct_latex():
    d = _comp("unavailable").describe()
    assert "unavailable" in d["latex"].lower()


def test_describe_total_includes_summary_fields():
    c = _comp("trivial", "trivial")
    cost = AccumulationCost(
        total=3,  # mu + alpha − num_output_orbits = 3 + 3 − 3
        mu=3,
        alpha=3,
        m_total=3,
        dense_baseline=3,
        num_terms=2,
        per_component=(c,),
        fallback_used=False,
    )
    d = cost.describe()
    assert "total" in d
    assert "savings_ratio" in d
    assert d["total"] == 3
