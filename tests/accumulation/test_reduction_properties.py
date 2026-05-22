"""Property tests for the reduction model."""

import flopscope as fps
import flopscope.numpy as fnp


def test_56_offbyone_dense_sum_charges_n_minus_1():
    """#56 regression: dense sum charges n-1, not n."""
    for n in [1, 2, 5, 10, 100]:
        a = fnp.zeros(n)
        with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
            fnp.sum(a)
        expected = max(n - 1, 1)
        actual = bc.summary_dict()["flops_used"]
        assert actual == expected, f"n={n}: expected {expected}, got {actual}"


def test_gaming_resistance_cost_never_exceeds_dense():
    """Symmetric reductions can never charge more than the dense baseline."""
    cases = [
        ((4, 4), (0, 1), fps.SymmetryGroup.symmetric(axes=(0, 1))),
        ((4, 4, 4), (2,), fps.SymmetryGroup.symmetric(axes=(0, 1, 2))),
        ((4, 4, 4), (0, 1), fps.SymmetryGroup.cyclic(axes=(0, 1, 2))),
    ]
    from flopscope._accumulation._reduction import compute_reduction_accumulation_cost

    for shape, axes, sym in cases:
        cost = compute_reduction_accumulation_cost(
            input_shape=shape,
            axes_summed=axes,
            symmetry=sym,
        )
        dense = 1
        for d in shape:
            dense *= d
        assert cost.total <= dense, f"{shape} {axes}: {cost.total} > {dense}"


def test_einsum_parity_reduction_via_sum_equals_via_einsum():
    """np.sum(T, axis=k) charge matches einsum_accumulation_cost('...', T) exactly.

    Both surfaces now apply the same off-by-one correction ("first cell of
    each output orbit is a free copy"), so the totals agree without a
    num_output_orbits offset. Previously the einsum surface overcharged by
    num_output_orbits; this regression test guards against that bug
    returning.
    """
    n = 4
    T = fps.as_symmetric(fnp.zeros((n, n, n)), symmetry=(0, 1, 2))
    sum_cost = fps.reduction_accumulation_cost(T, axis=2).total
    einsum_cost = fps.einsum_accumulation_cost("abc->ab", T).total
    assert einsum_cost == sum_cost, f"einsum_cost={einsum_cost}, sum_cost={sum_cost}"
