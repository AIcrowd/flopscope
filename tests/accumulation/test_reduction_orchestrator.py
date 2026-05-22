"""End-to-end tests for compute_reduction_accumulation_cost."""

import flopscope as fps
from flopscope._accumulation._reduction import compute_reduction_accumulation_cost


def test_dense_sum_charges_n_minus_1():
    """sum of (n,) → scalar charges n - 1 (off-by-one fixed)."""
    cost = compute_reduction_accumulation_cost(
        input_shape=(10,),
        axes_summed=(0,),
        symmetry=None,
    )
    assert cost.total == 9


def test_dense_sum_partial_axis_charges_dense_minus_outputs():
    """A.sum(axis=1) on (4, 5) → (4,). Cost = 4 × (5 - 1) = 16."""
    cost = compute_reduction_accumulation_cost(
        input_shape=(4, 5),
        axes_summed=(1,),
        symmetry=None,
    )
    # Total - num_output_orbits = 20 - 4 = 16.
    assert cost.total == 16


def test_symmetric_full_reduce_to_scalar():
    """T: (n, n, n) S_3 fully reduced → scalar.

    Input unique orbits = n(n+1)(n+2)/6 (multiset count of 3 from n).
    Output = 1 orbit.
    Cost = α - 1 where α = input orbits (all project to the same scalar).
    """
    n = 3
    sym = fps.SymmetryGroup.symmetric(axes=(0, 1, 2))
    cost = compute_reduction_accumulation_cost(
        input_shape=(n, n, n),
        axes_summed=(0, 1, 2),
        symmetry=sym,
    )
    # For n=3, multiset count = C(n+2, 3) = C(5, 3) = 10.
    assert cost.total == 10 - 1


def test_mean_via_extra_ops():
    """mean = sum cost + extra_ops = num_output_orbits."""
    # plain (10,), reduce all → mean cost = (n-1) + 1 = n
    cost = compute_reduction_accumulation_cost(
        input_shape=(10,),
        axes_summed=(0,),
        symmetry=None,
        extra_ops=1,
    )
    assert cost.total == 10  # 9 + 1


def test_op_factor_2_doubles_accumulation_cost():
    """Sum of squares with op_factor=2 doubles the per-event cost."""
    cost = compute_reduction_accumulation_cost(
        input_shape=(10,),
        axes_summed=(0,),
        symmetry=None,
        op_factor=2,
    )
    # 2 × (10 - 1) = 18
    assert cost.total == 18


def test_returns_accumulation_cost_with_per_component():
    cost = compute_reduction_accumulation_cost(
        input_shape=(4, 5),
        axes_summed=(1,),
        symmetry=None,
    )
    assert cost.per_component
    assert all(c.alpha is not None for c in cost.per_component)
