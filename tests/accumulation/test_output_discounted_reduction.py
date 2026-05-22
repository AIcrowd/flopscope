"""Tier-2 reduction cost: dense × num_output_orbits / num_output_elems."""

import flopscope as fps
from flopscope._accumulation._reduction import output_discounted_reduction_cost


def test_no_symmetry_returns_dense_cost():
    # No symmetry: discount = 1, cost = dense_per_output × num_output_elems.
    cost = output_discounted_reduction_cost(
        input_shape=(4, 5, 6),
        axes_summed=(2,),
        symmetry=None,
        dense_per_output_cost=6,  # e.g. partition-based median on axis of size 6
    )
    # output_elems = 4 * 5 = 20; discount = 1; total = 20 * 6 = 120
    assert cost == 20 * 6


def test_s3_full_reduce_gives_one_output_orbit():
    sym = fps.SymmetryGroup.symmetric(axes=(0, 1, 2))
    cost = output_discounted_reduction_cost(
        input_shape=(4, 4, 4),
        axes_summed=(0, 1, 2),
        symmetry=sym,
        dense_per_output_cost=10,
    )
    # full reduce → 1 output orbit; cost = 1 * 10 = 10
    assert cost == 10


def test_s3_reduce_axis_0_output_is_s2_symmetric():
    # T: (n, n, n) S_3. Reduce axis 0 → output (n, n) S_2.
    # num_output_orbits = n(n+1)/2.
    n = 4
    sym = fps.SymmetryGroup.symmetric(axes=(0, 1, 2))
    cost = output_discounted_reduction_cost(
        input_shape=(n, n, n),
        axes_summed=(0,),
        symmetry=sym,
        dense_per_output_cost=n,  # one pass over axis 0 per output cell
    )
    # output_orbits = n(n+1)/2 = 10; total = 10 * 4 = 40
    assert cost == (n * (n + 1) // 2) * n


def test_dense_per_output_zero_yields_zero():
    sym = fps.SymmetryGroup.symmetric(axes=(0, 1))
    cost = output_discounted_reduction_cost(
        input_shape=(3, 3),
        axes_summed=(0,),
        symmetry=sym,
        dense_per_output_cost=0,
    )
    assert cost == 0
