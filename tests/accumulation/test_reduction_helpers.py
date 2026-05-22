"""Unit tests for _accumulation/_reduction.py axis + orbit helpers."""

import flopscope as fps
from flopscope._accumulation._reduction import _normalize_axis, _num_output_orbits


def test_normalize_axis_none_returns_all_axes():
    assert _normalize_axis(None, ndim=3) == (0, 1, 2)


def test_normalize_axis_int_returns_singleton():
    assert _normalize_axis(1, ndim=3) == (1,)


def test_normalize_axis_negative_int_is_wrapped():
    assert _normalize_axis(-1, ndim=3) == (2,)


def test_normalize_axis_tuple_is_sorted_and_dedup():
    assert _normalize_axis((2, 0, 0), ndim=3) == (0, 2)


def test_normalize_axis_tuple_with_negatives():
    assert _normalize_axis((-1, 0), ndim=3) == (0, 2)


def test_num_output_orbits_no_symmetry_full_reduce_to_scalar():
    assert _num_output_orbits((4, 4), axes_summed=(0, 1), symmetry=None) == 1


def test_num_output_orbits_no_symmetry_partial_reduce():
    assert _num_output_orbits((4, 5), axes_summed=(1,), symmetry=None) == 4


def test_num_output_orbits_s2_reduce_axis_1_strips_symmetry():
    # A: (n, n) symmetric S_2{0, 1}. Reduce axis 1 → output shape (n,).
    # The output stabilizer is trivial (S_2 swap doesn't survive partial reduction).
    # Output orbits = n.
    sym = fps.SymmetryGroup.symmetric(axes=(0, 1))
    assert _num_output_orbits((4, 4), axes_summed=(1,), symmetry=sym) == 4


def test_num_output_orbits_s3_reduce_axis_2_keeps_s2_on_remaining():
    # T: (n, n, n) symmetric S_3{0, 1, 2}. Reduce axis 2 → output (n, n) with S_2{0, 1}.
    # Output orbits = n(n+1)/2.
    sym = fps.SymmetryGroup.symmetric(axes=(0, 1, 2))
    assert _num_output_orbits((4, 4, 4), axes_summed=(2,), symmetry=sym) == 4 * 5 // 2


def test_num_output_orbits_full_reduce_with_symmetry_is_scalar():
    # T: (n, n, n) symmetric S_3. Reduce all axes → scalar.
    sym = fps.SymmetryGroup.symmetric(axes=(0, 1, 2))
    assert _num_output_orbits((4, 4, 4), axes_summed=(0, 1, 2), symmetry=sym) == 1
