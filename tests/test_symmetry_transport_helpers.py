"""Tests for the new low-level helpers in _symmetry_utils that back transport_*."""

from __future__ import annotations

import pytest

import flopscope as flops
from flopscope._perm_group import SymmetryGroup
from flopscope._symmetry_utils import (
    setwise_stabilizer,
    group_orbits_on_axes,
    _normalize_reps_for_output,
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
