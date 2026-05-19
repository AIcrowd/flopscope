"""Tests for _cost.py — ComponentCost, run_ladder_per_component, and aggregate_einsum."""

from flopscope._accumulation._components import Component
from flopscope._accumulation._cost import ComponentCost, run_ladder_per_component
from flopscope._perm_group import _dimino
from flopscope._perm_group import _Permutation as Permutation


def _trivial_component(labels=("i",), sizes=(3,)):
    return Component(
        indices=tuple(range(len(labels))),
        labels=labels,
        va=labels,
        wa=(),
        sizes=sizes,
        visible_positions=tuple(range(len(labels))),
        generators=(),
        elements=(Permutation.identity(len(labels)),),
        order=1,
        group_name="trivial",
    )


def test_component_cost_trivial():
    component = _trivial_component()
    [cost] = run_ladder_per_component((component,), partition_budget=100_000)
    assert isinstance(cost, ComponentCost)
    assert cost.m == 3
    assert cost.alpha == 3
    assert cost.dense_count == 3
    assert cost.regime_id == "trivial"
    assert cost.shape == "trivial"
    assert cost.group_name == "trivial"


def test_component_cost_s2_visible():
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    component = Component(
        indices=(0, 1),
        labels=("i", "j"),
        va=("i", "j"),
        wa=(),
        sizes=(4, 4),
        visible_positions=(0, 1),
        generators=(swap,),
        elements=tuple(elements),
        order=2,
        group_name="S2{i,j}",
    )
    [cost] = run_ladder_per_component((component,), partition_budget=100_000)
    # M = α (functional projection) = 4·5/2 = 10
    assert cost.m == 10
    assert cost.alpha == 10
    assert cost.regime_id == "functionalProjection"
    assert cost.shape == "allVisible"


def test_component_cost_unavailable_when_budget_zero():
    """A mixed S_2 component with partition_budget=0 → singleton fires before partitionCount."""
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    component = Component(
        indices=(0, 1),
        labels=("i", "j"),
        va=("i",),
        wa=("j",),
        sizes=(6, 6),
        visible_positions=(0,),
        generators=(swap,),
        elements=tuple(elements),
        order=2,
        group_name="S2{i,j}",
    )
    [cost] = run_ladder_per_component((component,), partition_budget=0)
    # singleton fires before partitionCount, so this still gets a number.
    assert cost.alpha == 36
    assert cost.regime_id == "singleton"


def test_component_cost_dense_count_is_product_of_sizes():
    # Use a trivial group so we don't need equal sizes (a swap on unequal sizes
    # is physically invalid because any cycle must have uniform dimension).
    component = Component(
        indices=(0, 1),
        labels=("i", "j"),
        va=("i",),
        wa=("j",),
        sizes=(7, 11),
        visible_positions=(0,),
        generators=(),
        elements=(Permutation.identity(2),),
        order=1,
        group_name="trivial",
    )
    [cost] = run_ladder_per_component((component,), partition_budget=100_000)
    assert cost.dense_count == 77


# ── aggregate_einsum + AccumulationCost ──────────────────────────────


from flopscope._accumulation._cost import AccumulationCost, aggregate_einsum


def test_aggregate_einsum_total_for_two_trivial_components():
    """Two trivial components with V=L: no actual reduction → off-by-one
    correction zeroes out the α contribution.
    M_total = ∏ sizes; α = ∏ sizes; ∏num_output_orbits = ∏ sizes;
    total = (k-1)·M + α − ∏num_output_orbits = mu + 0."""
    c1 = run_ladder_per_component(
        (_trivial_component(labels=("i",), sizes=(3,)),), partition_budget=100_000
    )[0]
    c2 = run_ladder_per_component(
        (_trivial_component(labels=("j",), sizes=(4,)),), partition_budget=100_000
    )[0]
    cost = aggregate_einsum(
        component_costs=(c1, c2),
        num_terms=2,  # k = 2 operands
        dense_baseline=3 * 4,  # ∏ n_ℓ
    )
    assert isinstance(cost, AccumulationCost)
    assert cost.m_total == 12
    assert cost.alpha == 12
    assert cost.mu == 12  # (k-1) · m_total = 1 · 12
    assert cost.total == 12  # mu + alpha − ∏num_output_orbits = 12 + 12 − 12
    assert cost.fallback_used is False
    assert cost.unavailable_components == ()


def test_aggregate_einsum_with_symmetry_savings():
    """A single S_2 component with V=L: M = α = num_output_orbits = 10
    (S_2 on (4,4)). No reduction → off-by-one zeroes α contribution."""
    c = ComponentCost(
        labels=("i", "j"),
        va=("i", "j"),
        wa=(),
        sizes=(4, 4),
        m=10,
        alpha=10,
        dense_count=16,
        num_output_orbits=10,  # same as α when V=L (no reduction)
        regime_id="functionalProjection",
        shape="allVisible",
        group_name="S2{i,j}",
        group_order=2,
        regime_trace=(),
    )
    cost = aggregate_einsum(component_costs=(c,), num_terms=2, dense_baseline=16)
    assert cost.total == 1 * 10 + 10 - 10  # (k-1)·M + α − num_output_orbits


def test_aggregate_einsum_records_num_terms_and_dense_baseline():
    c = ComponentCost(
        labels=("i",),
        va=("i",),
        wa=(),
        sizes=(5,),
        m=5,
        alpha=5,
        dense_count=5,
        num_output_orbits=5,
        regime_id="trivial",
        shape="trivial",
        group_name="trivial",
        group_order=1,
        regime_trace=(),
    )
    cost = aggregate_einsum(component_costs=(c,), num_terms=3, dense_baseline=5)
    assert cost.num_terms == 3
    assert cost.dense_baseline == 5


# ── compute_accumulation_cost end-to-end ─────────────────────────────


from flopscope._accumulation._cost import compute_accumulation_cost


def test_compute_cost_matmul_no_symmetry():
    """ij,jk -> ik on (3,3,3): no symmetry → trivial component per label.
    M_total = 27, α = 27, ∏num_output_orbits = 3·1·3 = 9
    (i,k visible → 3 each; j summed → 1),
    total = (2-1)·27 + 27 − 9 = 45 (= textbook 2·n³ − n² for n=3)."""
    cost = compute_accumulation_cost(
        canonical_subscripts="ij,jk->ik",
        input_parts=("ij", "jk"),
        output_subscript="ik",
        shapes=((3, 3), (3, 3)),
        per_op_symmetries=(None, None),
        identity_pattern=None,
    )
    assert cost.dense_baseline == 27  # ∏ over {i, j, k} = 27 (each has size 3)
    assert cost.m_total == 27
    assert cost.alpha == 27
    assert cost.total == 45


def test_compute_cost_with_one_symmetric_input():
    """ij,jk -> ik with A symmetric in (i,j). Detected G_pt acts on {i,j}."""
    from flopscope._perm_group import SymmetryGroup

    s2 = SymmetryGroup.symmetric(axes=(0, 1))
    cost = compute_accumulation_cost(
        canonical_subscripts="ij,jk->ik",
        input_parts=("ij", "jk"),
        output_subscript="ik",
        shapes=((4, 4), (4, 4)),
        per_op_symmetries=(s2, None),
        identity_pattern=None,
    )
    # The detected G_pt's action depends on the bipartite structure.
    # We assert positively that cost is computed (no exception, no fallback).
    assert cost.fallback_used is False
    assert cost.total > 0


def test_compute_cost_identity_pattern_is_used_for_repeated_operands():
    """A·A: same operand object passed twice → identity_pattern triggers wreath swaps."""
    cost_distinct = compute_accumulation_cost(
        canonical_subscripts="ij,jk->ik",
        input_parts=("ij", "jk"),
        output_subscript="ik",
        shapes=((3, 3), (3, 3)),
        per_op_symmetries=(None, None),
        identity_pattern=None,
    )
    cost_aliased = compute_accumulation_cost(
        canonical_subscripts="ij,jk->ik",
        input_parts=("ij", "jk"),
        output_subscript="ik",
        shapes=((3, 3), (3, 3)),
        per_op_symmetries=(None, None),
        identity_pattern=((0, 1),),  # same object at positions 0 and 1
    )
    # The aliased version may detect additional G_pt (if the wreath swap
    # produces a valid pi). For ij,jk it doesn't, so both should agree.
    assert cost_distinct.total == cost_aliased.total


def test_accumulation_cost_has_per_step_and_path_fields():
    """For k=2 einsums these fields default to empty/None.
    Populated for k>=3 (covered in test_path_aware_cost.py)."""
    from flopscope._accumulation._cost import AccumulationCost, ComponentCost

    c = ComponentCost(
        labels=("i",),
        va=("i",),
        wa=(),
        sizes=(3,),
        m=3,
        alpha=3,
        dense_count=3,
        num_output_orbits=3,
        regime_id="trivial",
        shape="trivial",
        group_name="trivial",
        group_order=1,
        regime_trace=(),
    )
    cost = AccumulationCost(
        total=3,
        mu=3,
        alpha=3,
        m_total=3,
        dense_baseline=3,
        num_terms=2,
        per_component=(c,),
        fallback_used=False,
    )
    assert cost.per_step == ()
    assert cost.path is None
