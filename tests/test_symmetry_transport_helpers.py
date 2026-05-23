"""Tests for the new low-level helpers in _symmetry_utils that back transport_*."""

from __future__ import annotations

from flopscope._perm_group import SymmetryGroup
from flopscope._symmetry_utils import (
    _normalize_reps_for_output,
    group_orbits_on_axes,
    setwise_stabilizer,
)


class TestSetwiseStabilizer:
    def test_S3_with_singleton_fixed_set_returns_S2(self):
        G = SymmetryGroup.symmetric(axes=(0, 1, 2))
        result = setwise_stabilizer(G, fixed_set={1})
        assert result is not None
        # Stabilizer of {1} in S_3 = perms mapping 1->1 = S_2 on {0, 2}
        assert result.order() == 2
        assert set(result.axes) == {0, 1, 2}

    def test_full_set_returns_full_group(self):
        G = SymmetryGroup.symmetric(axes=(0, 1, 2))
        result = setwise_stabilizer(G, fixed_set={0, 1, 2})
        assert result is not None
        assert result.order() == G.order()

    def test_empty_set_returns_full_group(self):
        G = SymmetryGroup.symmetric(axes=(0, 1, 2))
        result = setwise_stabilizer(G, fixed_set=set())
        assert result is not None
        assert result.order() == G.order()

    def test_cyclic_C3_singleton_drops(self):
        G = SymmetryGroup.cyclic(axes=(0, 1, 2))
        result = setwise_stabilizer(G, fixed_set={0})
        # C_3 has no element other than identity that fixes {0}
        assert result is None

    def test_cyclic_C4_pair_yields_Z2(self):
        G = SymmetryGroup.cyclic(axes=(0, 1, 2, 3))
        result = setwise_stabilizer(G, fixed_set={0, 2})
        # C_4 element (0 2)(1 3) maps {0, 2} -> {0, 2} setwise. Z_2 survives.
        assert result is not None
        assert result.order() == 2

    def test_returns_none_for_input_None(self):
        assert setwise_stabilizer(None, fixed_set={0}) is None

    def test_filters_out_of_block_axes(self):
        G = SymmetryGroup.symmetric(axes=(0, 1))
        # fixed_set contains axis 5 which is NOT in G.axes — should be filtered.
        result = setwise_stabilizer(G, fixed_set={0, 5})
        # Effective fixed = {0}; stabilizer of {0} in S_2 = trivial -> None.
        assert result is None


class TestGroupOrbitsOnAxes:
    def test_S3_single_orbit_on_all_axes(self):
        G = SymmetryGroup.symmetric(axes=(0, 1, 2))
        orbits = group_orbits_on_axes(G, [0, 1, 2])
        assert len(orbits) == 1
        assert orbits[0] == {0, 1, 2}

    def test_C3_single_orbit(self):
        G = SymmetryGroup.cyclic(axes=(0, 1, 2))
        orbits = group_orbits_on_axes(G, [0, 1, 2])
        assert len(orbits) == 1
        assert orbits[0] == {0, 1, 2}

    def test_two_disjoint_S2_orbits_via_direct_product(self):
        # S_2 on (0,1) direct-product S_2 on (3,4): two separate orbits.
        gA = SymmetryGroup.symmetric(axes=(0, 1))
        gB = SymmetryGroup.symmetric(axes=(3, 4))
        from flopscope._symmetry_utils import direct_product_groups

        G = direct_product_groups(gA, gB)
        orbits = group_orbits_on_axes(G, [0, 1, 3, 4])
        # Order may vary; compare as set of frozensets.
        as_sets = {frozenset(o) for o in orbits}
        assert as_sets == {frozenset({0, 1}), frozenset({3, 4})}

    def test_singleton_orbit_for_axis_outside_group(self):
        # Axis 5 isn't acted on by G, so it's its own orbit.
        G = SymmetryGroup.symmetric(axes=(0, 1))
        orbits = group_orbits_on_axes(G, [0, 1, 5])
        as_sets = {frozenset(o) for o in orbits}
        assert as_sets == {frozenset({0, 1}), frozenset({5})}


class TestNormalizeRepsForOutput:
    def test_scalar_reps(self):
        # tile(arr, 2) with output_ndim=3 -> (1, 1, 2)? No - scalar reps means
        # replicate every axis by that count. NumPy behavior: scalar => (reps,)
        # then right-align. For output_ndim=3 with reps=2: output rank stays
        # max(1, ndim)=3 so reps padded to (1, 1, 2).
        assert _normalize_reps_for_output(2, output_ndim=3) == (1, 1, 2)

    def test_tuple_shorter_than_output(self):
        assert _normalize_reps_for_output((2, 3), output_ndim=4) == (1, 1, 2, 3)

    def test_tuple_equal_length(self):
        assert _normalize_reps_for_output((2, 3, 4), output_ndim=3) == (2, 3, 4)

    def test_tuple_already_padded(self):
        assert _normalize_reps_for_output((1, 2, 3), output_ndim=3) == (1, 2, 3)

    def test_list_input(self):
        assert _normalize_reps_for_output([2, 3], output_ndim=3) == (1, 2, 3)
