"""Regression tests for issue #66: propagate symmetry through None/np.newaxis indexing.

See .aicrowd/superpowers/specs/2026-05-22-issue-66-newaxis-symmetry-design.md.
"""

import numpy
import pytest

import flopscope as flops
import flopscope._free_ops as ops
import flopscope.numpy as fnp
from flopscope import SymmetryGroup
from flopscope._symmetry_utils import inserted_axes_symmetry


# ---------------------------------------------------------------------------
# inserted_axes_symmetry() unit tests
# ---------------------------------------------------------------------------


def test_inserted_axes_symmetry_returns_none_for_empty():
    assert inserted_axes_symmetry([]) is None


def test_inserted_axes_symmetry_returns_none_for_single_position():
    assert inserted_axes_symmetry([5]) is None


def test_inserted_axes_symmetry_returns_symmetric_pair_for_two_positions():
    expected = SymmetryGroup.symmetric(axes=(0, 2))
    assert inserted_axes_symmetry([0, 2]) == expected


def test_inserted_axes_symmetry_returns_symmetric_triple_for_three_positions():
    expected = SymmetryGroup.symmetric(axes=(1, 3, 5))
    assert inserted_axes_symmetry([1, 3, 5]) == expected


def test_inserted_axes_symmetry_accepts_tuple_input():
    expected = SymmetryGroup.symmetric(axes=(0, 4))
    assert inserted_axes_symmetry((0, 4)) == expected
