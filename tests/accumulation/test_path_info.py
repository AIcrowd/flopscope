"""Tests for FlopscopePathInfo wrapper."""

from flopscope._accumulation._cost import AccumulationCost, ComponentCost
from flopscope._accumulation._path_info import FlopscopePathInfo


def _trivial_cost():
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
    return AccumulationCost(
        total=3,  # mu + alpha − num_output_orbits = 3 + 3 − 3
        mu=3,
        alpha=3,
        m_total=3,
        dense_baseline=3,
        num_terms=2,
        per_component=(c,),
        fallback_used=False,
    )


def test_path_info_wrapper_carries_accumulation():
    fpi = FlopscopePathInfo.from_inner(inner=None, accumulation=_trivial_cost())
    assert fpi.accumulation is not None
    assert fpi.accumulation.total == 3


def test_path_info_optimized_cost_returns_accumulation_total():
    fpi = FlopscopePathInfo.from_inner(inner=None, accumulation=_trivial_cost())
    assert fpi.optimized_cost == 3


def test_path_info_falls_back_to_inner_when_no_accumulation():
    """If no accumulation attached, optimized_cost should fall back to the
    wrapped inner PathInfo's optimized_cost (legacy behavior)."""

    class _FakeInner:
        optimized_cost = 99
        path = []
        steps = []

    fpi = FlopscopePathInfo.from_inner(inner=_FakeInner(), accumulation=None)
    assert fpi.optimized_cost == 99


def test_str_shows_flopscope_optimized_cost():
    """str(info) must render the α/M optimized_cost, not the upstream one.

    Regression for the bug surfaced during PR #91 docs review where
    info.optimized_cost (α/M) showed an upstream value instead. With the
    off-by-one fix in aggregate_einsum, matmul (n=4) costs 112
    (= 2·n³ − n² = 128 − 16) — not the upstream raw 64 from opt_einsum's
    own counter.
    """
    import numpy as np

    import flopscope as flops
    import flopscope.numpy as fnp

    A = np.zeros((4, 4))
    B = np.zeros((4, 4))

    with flops.BudgetContext(flop_budget=10**12):
        _, info = fnp.einsum_path("ij,jk->ik", A, B)

    assert info.optimized_cost == 112, (
        f"setup precondition violated: optimized_cost={info.optimized_cost}"
    )

    rendered = str(info)
    # The optimized_cost row must show the reconciled flopscope value 112.
    assert "112" in rendered, f"expected 112 in str(info); got:\n{rendered}"
    # naive_cost is not reconciled — the inner's own value (64) is acceptable
    # in the header after the monkey-patch was removed (commit 69d88ec8f).
