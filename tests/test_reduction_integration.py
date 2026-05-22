"""End-to-end integration tests via BudgetContext for the new reduction model."""

import flopscope as fps
import flopscope.numpy as fnp


def _flops_used(bc):
    return bc.summary_dict()["flops_used"]


def test_dense_sum_charges_n_minus_1_end_to_end():
    a = fnp.zeros(10)
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.sum(a)
    assert _flops_used(bc) == 9


def test_symmetric_sum_charges_via_orbit_mapping():
    n = 4
    T = fps.as_symmetric(fnp.zeros((n, n, n)), symmetry=(0, 1, 2))
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.sum(T, axis=2)
    expected = fps.reduction_accumulation_cost(T, axis=2).total
    assert _flops_used(bc) == expected


def test_prod_uses_tier1_model():
    a = fnp.ones(5)
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.prod(a)
    # 5 - 1 = 4 multiplications
    assert _flops_used(bc) == 4


def test_max_uses_tier1_model():
    a = fnp.zeros(5)
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.max(a)
    # 5 - 1 = 4 comparisons
    assert _flops_used(bc) == 4


def test_mean_charges_sum_plus_num_output_orbits():
    a = fnp.zeros(10)
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.mean(a)
    # sum cost = 9 (10 - 1); + 1 divide = 10
    assert _flops_used(bc) == 10


def test_mean_on_symmetric_tensor_uses_orbit_count_for_divides():
    n = 4
    T = fps.as_symmetric(fnp.zeros((n, n, n)), symmetry=(0, 1, 2))
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.mean(T, axis=2)
    sum_cost = fps.reduction_accumulation_cost(T, axis=2).total
    # num_output_orbits = n(n+1)/2 = 10 for the (n,n) S_2 output
    expected = sum_cost + 4 * 5 // 2
    assert _flops_used(bc) == expected


def test_median_on_symmetric_tensor_uses_tier2_discount():
    n = 4
    T = fps.as_symmetric(fnp.zeros((n, n, n)), symmetry=(0, 1, 2))
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.median(T, axis=2)
    # Output is (n, n) S_2-symmetric → n(n+1)/2 output orbits.
    # dense_per_output for median ≈ axis_dim (one partition pass).
    expected_min = (n * (n + 1) // 2) * n  # lower bound
    assert _flops_used(bc) >= expected_min
    assert _flops_used(bc) < n * n * n  # cheaper than dense


def test_percentile_uses_tier2_discount():
    n = 4
    T = fps.as_symmetric(fnp.zeros((n, n, n)), symmetry=(0, 1, 2))
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.percentile(T, 50, axis=2)
    expected_min = (n * (n + 1) // 2) * n
    assert _flops_used(bc) >= expected_min


def test_quantile_uses_tier2_discount():
    n = 4
    T = fps.as_symmetric(fnp.zeros((n, n, n)), symmetry=(0, 1, 2))
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.quantile(T, 0.5, axis=2)
    expected_min = (n * (n + 1) // 2) * n
    assert _flops_used(bc) >= expected_min
