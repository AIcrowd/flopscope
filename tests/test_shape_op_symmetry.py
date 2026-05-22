"""Tests for shape-op symmetry transport. See issue #68."""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._perm_group import SymmetryGroup
from flopscope._symmetric import SymmetricTensor, as_symmetric


def _sym(*axes):
    return SymmetryGroup.symmetric(axes=axes)


def _cyc(*axes):
    return SymmetryGroup.cyclic(axes=axes)


def _dih(*axes):
    return SymmetryGroup.dihedral(axes=axes)


def test_every_issue_68_op_has_transport():
    """Coverage assertion: every shape op in issue #68 has a transport function."""
    from flopscope import _symmetry_transport as st

    required = {
        "reshape", "concatenate", "stack", "vstack", "hstack", "column_stack",
        "split", "hsplit", "vsplit", "dsplit",
        "atleast_1d", "atleast_2d", "atleast_3d",
        "broadcast_to", "expand_dims", "squeeze",
        "flip", "roll", "tile", "repeat",
        "transpose", "swapaxes", "moveaxis", "matrix_transpose",
        "ravel",
    }
    for op in required:
        assert hasattr(st, f"transport_{op}"), f"missing transport_{op}"


class TestTransportSqueeze:
    def test_squeeze_outside_block_shifts_block_axes(self):
        from flopscope._symmetry_transport import transport_squeeze
        # input shape (1, 3, 3), S_2 on (1, 2)
        G = _sym(1, 2)
        result = transport_squeeze(G, input_shape=(1, 3, 3), axis=0)
        # After squeeze, block axes (1, 2) shift to (0, 1).
        assert result is not None
        assert set(result.axes) == {0, 1}
        assert result.order() == 2

    def test_squeeze_inside_block_drops(self):
        from flopscope._symmetry_transport import transport_squeeze
        # input shape (1, 1), S_2 on (0, 1). Squeeze axis 0 (in block).
        G = _sym(0, 1)
        result = transport_squeeze(G, input_shape=(1, 1), axis=0)
        assert result is None

    def test_squeeze_axis_none_removes_all_length_1(self):
        from flopscope._symmetry_transport import transport_squeeze
        # input shape (1, 3, 3, 1), S_2 on (1, 2). axis=None removes axes 0 and 3.
        G = _sym(1, 2)
        result = transport_squeeze(G, input_shape=(1, 3, 3, 1), axis=None)
        assert result is not None
        assert set(result.axes) == {0, 1}

    def test_squeeze_none_input_returns_none(self):
        from flopscope._symmetry_transport import transport_squeeze
        assert transport_squeeze(None, input_shape=(1, 3, 3), axis=0) is None


class TestTransportAtleast:
    def test_atleast_1d_noop_on_2d(self):
        from flopscope._symmetry_transport import transport_atleast_1d
        G = _sym(0, 1)
        result = transport_atleast_1d(G, input_shape=(3, 3))
        assert result is not None and set(result.axes) == {0, 1}

    def test_atleast_2d_noop_on_2d(self):
        from flopscope._symmetry_transport import transport_atleast_2d
        G = _sym(0, 1)
        result = transport_atleast_2d(G, input_shape=(3, 3))
        assert result is not None and set(result.axes) == {0, 1}

    def test_atleast_3d_appends_trailing_axis_for_2d(self):
        from flopscope._symmetry_transport import transport_atleast_3d
        G = _sym(0, 1)
        # NumPy: (M, N) -> (M, N, 1). Block axes unchanged.
        result = transport_atleast_3d(G, input_shape=(3, 3))
        assert result is not None
        assert set(result.axes) == {0, 1}
        assert result.order() == 2

    def test_atleast_3d_noop_on_3d(self):
        from flopscope._symmetry_transport import transport_atleast_3d
        G = _sym(1, 2)
        result = transport_atleast_3d(G, input_shape=(4, 3, 3))
        assert result is not None and set(result.axes) == {1, 2}

    def test_atleast_kd_none_input(self):
        from flopscope._symmetry_transport import (
            transport_atleast_1d, transport_atleast_2d, transport_atleast_3d,
        )
        assert transport_atleast_1d(None, input_shape=(3, 3)) is None
        assert transport_atleast_2d(None, input_shape=(3, 3)) is None
        assert transport_atleast_3d(None, input_shape=(3, 3)) is None


class TestTransportSplit:
    def test_split_non_block_axis_preserves(self):
        from flopscope._symmetry_transport import transport_split
        # (4, 3, 3) with S_2 on (1, 2), split along axis 0.
        G = _sym(1, 2)
        result = transport_split(G, input_shape=(4, 3, 3), axis=0)
        assert result is not None
        assert set(result.axes) == {1, 2}
        assert result.order() == 2

    def test_split_in_block_drops_for_degree2(self):
        from flopscope._symmetry_transport import transport_split
        G = _sym(0, 1)
        result = transport_split(G, input_shape=(3, 3), axis=0)
        assert result is None

    def test_split_in_block_preserves_subgroup_for_S3(self):
        from flopscope._symmetry_transport import transport_split
        # S_3 on (0, 1, 2), split along axis 0 -> pieces carry S_2 on (1, 2).
        G = _sym(0, 1, 2)
        result = transport_split(G, input_shape=(3, 3, 3), axis=0)
        assert result is not None
        assert set(result.axes) == {1, 2}
        assert result.order() == 2

    def test_hsplit_for_1d(self):
        from flopscope._symmetry_transport import transport_hsplit
        # hsplit on 1-D uses axis=0, and 1-D inputs can't carry multi-axis groups.
        assert transport_hsplit(None, input_shape=(6,)) is None

    def test_hsplit_for_2d_uses_axis_1(self):
        from flopscope._symmetry_transport import transport_hsplit
        G = _sym(0, 1)
        # hsplit on (3, 6) splits along axis 1 - which IS in block; degree 2 -> drops.
        assert transport_hsplit(G, input_shape=(3, 6)) is None

    def test_vsplit_uses_axis_0(self):
        from flopscope._symmetry_transport import transport_vsplit
        G = _sym(1, 2)
        # vsplit on (4, 3, 3) splits along axis 0 (outside block) -> preserves.
        result = transport_vsplit(G, input_shape=(4, 3, 3))
        assert result is not None and set(result.axes) == {1, 2}

    def test_dsplit_uses_axis_2(self):
        from flopscope._symmetry_transport import transport_dsplit
        G = _sym(0, 1)
        # dsplit on (3, 3, 6) splits along axis 2 (outside block) -> preserves.
        result = transport_dsplit(G, input_shape=(3, 3, 6))
        assert result is not None and set(result.axes) == {0, 1}


class TestTransportRepeatRoll:
    def test_repeat_outside_block_preserves(self):
        from flopscope._symmetry_transport import transport_repeat
        G = _sym(0, 1)
        result = transport_repeat(G, input_shape=(3, 3, 4), axis=2)
        assert result is not None and set(result.axes) == {0, 1}

    def test_repeat_inside_block_drops(self):
        from flopscope._symmetry_transport import transport_repeat
        G = _sym(0, 1)
        assert transport_repeat(G, input_shape=(3, 3), axis=1) is None

    def test_repeat_axis_negative_normalized(self):
        from flopscope._symmetry_transport import transport_repeat
        G = _sym(0, 1)
        # axis=-1 on (3, 3, 4) -> axis 2, outside block.
        result = transport_repeat(G, input_shape=(3, 3, 4), axis=-1)
        assert result is not None

    def test_roll_outside_block_preserves(self):
        from flopscope._symmetry_transport import transport_roll
        G = _sym(0, 1)
        result = transport_roll(G, input_shape=(3, 3, 4), axis=2)
        assert result is not None and set(result.axes) == {0, 1}

    def test_roll_inside_block_drops(self):
        from flopscope._symmetry_transport import transport_roll
        G = _sym(0, 1)
        assert transport_roll(G, input_shape=(3, 3), axis=0) is None

    def test_roll_multi_axis_any_in_block_drops(self):
        from flopscope._symmetry_transport import transport_roll
        G = _sym(0, 1)
        # Even one block axis rolled => drop.
        assert transport_roll(G, input_shape=(3, 3, 4), axis=(0, 2)) is None


class TestTransportAxisPermutation:
    def test_transpose_reverses_block_axes(self):
        from flopscope._symmetry_transport import transport_transpose
        G = _sym(0, 1)
        # transpose with axes=None reverses all axes.
        result = transport_transpose(G, ndim=2, axes=None)
        assert result is not None and set(result.axes) == {0, 1}

    def test_transpose_explicit_perm(self):
        from flopscope._symmetry_transport import transport_transpose
        G = _sym(0, 1)
        # transpose with axes=(1, 0) swaps; S_2 still on (0, 1) but generators relabeled.
        result = transport_transpose(G, ndim=2, axes=(1, 0))
        assert result is not None
        assert set(result.axes) == {0, 1}

    def test_swapaxes(self):
        from flopscope._symmetry_transport import transport_swapaxes
        G = _sym(0, 1)
        # On a (3,3,4) tensor with S_2 on (0,1), swapaxes(0, 2) sends:
        # axis 0 -> 2, axis 1 -> 1, axis 2 -> 0. So new block axes are (2, 1).
        result = transport_swapaxes(G, ndim=3, axis1=0, axis2=2)
        assert result is not None
        assert set(result.axes) == {1, 2}

    def test_moveaxis(self):
        from flopscope._symmetry_transport import transport_moveaxis
        G = _sym(0, 1)
        # moveaxis source=0, destination=2 on rank-3: axis 0 moves to position 2.
        # After move: original axis 0 is now at position 2, axes 1,2 shift down.
        result = transport_moveaxis(G, ndim=3, source=0, destination=2)
        assert result is not None
        # Original axes (0, 1) -> new positions (2, 0).
        assert set(result.axes) == {0, 2}

    def test_matrix_transpose_swaps_last_two(self):
        from flopscope._symmetry_transport import transport_matrix_transpose
        G = _sym(1, 2)
        # matrix_transpose = swapaxes(-2, -1) on rank-3: swap axes 1 and 2.
        result = transport_matrix_transpose(G, ndim=3)
        # Block (1, 2) under swap(1, 2) -> still (1, 2) (set unchanged).
        assert result is not None and set(result.axes) == {1, 2}


class TestTransportConcatenate:
    def test_identical_groups_off_block_axis_preserves(self):
        from flopscope._symmetry_transport import transport_concatenate
        G1 = _sym(0, 1)
        G2 = _sym(0, 1)
        result = transport_concatenate(
            [G1, G2], output_ndim=3, axis=2,
        )
        assert result is not None and set(result.axes) == {0, 1}

    def test_plain_input_forces_drop(self):
        from flopscope._symmetry_transport import transport_concatenate
        G1 = _sym(0, 1)
        result = transport_concatenate(
            [G1, None], output_ndim=3, axis=2,
        )
        assert result is None

    def test_concat_on_block_axis_for_degree2_drops(self):
        from flopscope._symmetry_transport import transport_concatenate
        G1 = _sym(0, 1)
        G2 = _sym(0, 1)
        # Concat along axis 0 (in block); restriction to {1} is degree-1 -> drop.
        result = transport_concatenate([G1, G2], output_ndim=2, axis=0)
        assert result is None

    def test_concat_S3_along_block_axis_yields_S2(self):
        from flopscope._symmetry_transport import transport_concatenate
        G1 = _sym(0, 1, 2)
        G2 = _sym(0, 1, 2)
        # Restrict S_3 to {1, 2} -> S_2.
        result = transport_concatenate([G1, G2], output_ndim=3, axis=0)
        assert result is not None
        assert set(result.axes) == {1, 2}
        assert result.order() == 2

    def test_concat_axis_none_drops(self):
        from flopscope._symmetry_transport import transport_concatenate
        G1 = _sym(0, 1)
        # axis=None ravels first -> always drops.
        assert transport_concatenate([G1, G1], output_ndim=1, axis=None) is None


class TestTransportStack:
    def test_stack_two_identical_S2_along_axis_0(self):
        from flopscope._symmetry_transport import transport_stack
        G = _sym(0, 1)
        # New axis at position 0; existing block axes (0, 1) shift to (1, 2).
        result = transport_stack([G, G], output_ndim=3, axis=0)
        assert result is not None
        assert set(result.axes) == {1, 2}

    def test_stack_S3_and_C3_intersect_to_C3(self):
        from flopscope._symmetry_transport import transport_stack
        gS = _sym(0, 1, 2)
        gC = _cyc(0, 1, 2)
        # New axis at position 0; block axes shift to (1, 2, 3).
        # Intersect S_3 ∩ C_3 = C_3.
        result = transport_stack([gS, gC], output_ndim=4, axis=0)
        assert result is not None
        assert set(result.axes) == {1, 2, 3}
        # C_3 has order 3.
        assert result.order() == 3

    def test_stack_plain_input_drops(self):
        from flopscope._symmetry_transport import transport_stack
        G = _sym(0, 1)
        assert transport_stack([G, None], output_ndim=3, axis=0) is None

    def test_stack_at_end(self):
        from flopscope._symmetry_transport import transport_stack
        G = _sym(0, 1)
        # New axis at position 2 (end). Block axes 0,1 don't shift.
        result = transport_stack([G, G], output_ndim=3, axis=2)
        assert result is not None
        assert set(result.axes) == {0, 1}


class TestTransportVHColumnStack:
    def test_vstack_two_2d_concat_along_0(self):
        from flopscope._symmetry_transport import transport_vstack
        G = _sym(0, 1)
        # vstack of two (3,3) S_2 along axis 0; restrict to {1} -> degree-1 -> drop.
        result = transport_vstack(
            [G, G], output_ndim=2, input_ndims=[2, 2],
        )
        assert result is None

    def test_vstack_two_3d_along_0(self):
        from flopscope._symmetry_transport import transport_vstack
        G = _sym(1, 2)
        # vstack of two (2,3,3) S_2(1,2) along axis 0 (outside block).
        result = transport_vstack(
            [G, G], output_ndim=3, input_ndims=[3, 3],
        )
        assert result is not None and set(result.axes) == {1, 2}

    def test_hstack_2d_along_1(self):
        from flopscope._symmetry_transport import transport_hstack
        G = _sym(0, 1)
        # hstack of two (3,3) S_2; axis=1 in block -> drop.
        result = transport_hstack(
            [G, G], output_ndim=2, input_ndims=[2, 2],
        )
        assert result is None

    def test_hstack_1d_all(self):
        from flopscope._symmetry_transport import transport_hstack
        # All 1-D inputs -> concat axis 0. 1-D inputs have no multi-axis group.
        assert transport_hstack(
            [None, None], output_ndim=1, input_ndims=[1, 1],
        ) is None

    def test_column_stack_2d_inputs(self):
        from flopscope._symmetry_transport import transport_column_stack
        G = _sym(0, 1)
        # Two (3,3) S_2 inputs concat along axis 1; axis 1 in block -> drop.
        result = transport_column_stack(
            [G, G], output_ndim=2, input_ndims=[2, 2],
        )
        assert result is None


class TestTransportReshape:
    def test_identity_reshape_preserves(self):
        from flopscope._symmetry_transport import transport_reshape
        G = _sym(0, 1)
        result = transport_reshape(G, input_shape=(3, 3), output_shape=(3, 3))
        assert result is not None and set(result.axes) == {0, 1}

    def test_identity_with_interior_length_1_preserves(self):
        # The bug found by math-olympiad review: (3,1,3) -> (3,1,3) with S_2 on (0,2).
        from flopscope._symmetry_transport import transport_reshape
        G = _sym(0, 2)
        result = transport_reshape(G, input_shape=(3, 1, 3), output_shape=(3, 1, 3))
        assert result is not None and set(result.axes) == {0, 2}

    def test_suffix_split_off_block_preserves(self):
        from flopscope._symmetry_transport import transport_reshape
        G = _sym(1, 2)
        # (2, 3, 3, 4) -> (2, 3, 3, 2, 2): last axis 4 splits to (2,2), block intact.
        result = transport_reshape(
            G, input_shape=(2, 3, 3, 4), output_shape=(2, 3, 3, 2, 2),
        )
        assert result is not None and set(result.axes) == {1, 2}

    def test_merge_inside_block_drops(self):
        from flopscope._symmetry_transport import transport_reshape
        G = _sym(1, 2)
        # (2, 3, 3, 4) -> (2, 9, 4): merges two block axes -> drop.
        result = transport_reshape(
            G, input_shape=(2, 3, 3, 4), output_shape=(2, 9, 4),
        )
        assert result is None

    def test_merge_block_with_prefix_drops(self):
        from flopscope._symmetry_transport import transport_reshape
        G = _sym(1, 2)
        # (2, 3, 3, 4) -> (6, 3, 4): merges axis 0 with axis 1 (block).
        result = transport_reshape(
            G, input_shape=(2, 3, 3, 4), output_shape=(6, 3, 4),
        )
        assert result is None

    def test_prepend_length_1(self):
        from flopscope._symmetry_transport import transport_reshape
        G = _sym(0, 1)
        # (3, 3) -> (1, 3, 3): block shifts to (1, 2).
        result = transport_reshape(
            G, input_shape=(3, 3), output_shape=(1, 3, 3),
        )
        assert result is not None and set(result.axes) == {1, 2}

    def test_non_contiguous_block_preserved(self):
        from flopscope._symmetry_transport import transport_reshape
        G = _sym(0, 2)
        # (3, 5, 3) -> (1, 3, 5, 3): non-contig block preserved at (1, 3).
        result = transport_reshape(
            G, input_shape=(3, 5, 3), output_shape=(1, 3, 5, 3),
        )
        assert result is not None and set(result.axes) == {1, 3}

    def test_swap_block_with_non_block_drops(self):
        from flopscope._symmetry_transport import transport_reshape
        G = _sym(0, 2)
        # (3, 5, 3) -> (3, 3, 5): cannot be done by reshape; segment match fails.
        result = transport_reshape(
            G, input_shape=(3, 5, 3), output_shape=(3, 3, 5),
        )
        assert result is None


class TestTransportRavel:
    def test_ravel_drops_for_nondegenerate(self):
        from flopscope._symmetry_transport import transport_ravel
        G = _sym(0, 1)
        result = transport_ravel(G, input_shape=(3, 3))
        assert result is None  # Block of size 3 can't fit in 1-axis output of size 9.

    def test_ravel_for_none_input(self):
        from flopscope._symmetry_transport import transport_ravel
        assert transport_ravel(None, input_shape=(3, 3)) is None


class TestTransportFlip:
    def test_no_axes_flipped_preserves(self):
        from flopscope._symmetry_transport import transport_flip
        G = _sym(0, 1)
        result = transport_flip(G, ndim=2, axes_flipped=())
        assert result is not None and set(result.axes) == {0, 1}

    def test_all_block_axes_flipped_preserves(self):
        from flopscope._symmetry_transport import transport_flip
        G = _sym(0, 1)
        result = transport_flip(G, ndim=2, axes_flipped=(0, 1))
        assert result is not None and set(result.axes) == {0, 1}

    def test_partial_S3_flip_yields_S2(self):
        from flopscope._symmetry_transport import transport_flip
        G = _sym(0, 1, 2)
        # Flip axis 1 only: setwise stab of {1} in S_3 = perms fixing 1 = S_2 on {0,2}.
        result = transport_flip(G, ndim=3, axes_flipped=(1,))
        assert result is not None
        assert result.order() == 2

    def test_partial_C3_flip_drops(self):
        from flopscope._symmetry_transport import transport_flip
        G = _cyc(0, 1, 2)
        # C_3 has no element fixing a singleton -> drops.
        result = transport_flip(G, ndim=3, axes_flipped=(0,))
        assert result is None

    def test_partial_C4_flip_pair_yields_Z2(self):
        from flopscope._symmetry_transport import transport_flip
        G = _cyc(0, 1, 2, 3)
        # C_4 with F_A = {0, 2}: element (0 2)(1 3) survives -> Z_2.
        result = transport_flip(G, ndim=4, axes_flipped=(0, 2))
        assert result is not None
        assert result.order() == 2

    def test_partial_D3_flip_yields_Z2(self):
        from flopscope._symmetry_transport import transport_flip
        G = _dih(0, 1, 2)
        # D_3 with F_A = {1}: the reflection fixing axis 1 survives -> Z_2.
        result = transport_flip(G, ndim=3, axes_flipped=(1,))
        assert result is not None
        assert result.order() == 2

    def test_axes_outside_block_flipped_preserves(self):
        from flopscope._symmetry_transport import transport_flip
        G = _sym(0, 1)
        # Flipping axis 2 (outside block) doesn't affect the block.
        result = transport_flip(G, ndim=3, axes_flipped=(2,))
        assert result is not None and set(result.axes) == {0, 1}

    def test_negative_axis_normalized(self):
        from flopscope._symmetry_transport import transport_flip
        G = _sym(0, 1)
        # axis=-1 on ndim=3 means axis 2 (outside block).
        result = transport_flip(G, ndim=3, axes_flipped=(-1,))
        assert result is not None and set(result.axes) == {0, 1}


class TestTransportTile:
    def test_constant_reps_on_orbit_preserves(self):
        from flopscope._symmetry_transport import transport_tile
        G = _sym(0, 1)
        # reps (2, 2) is constant on the S_2 orbit {0, 1}; output (4, 4).
        result = transport_tile(
            G, input_shape=(2, 2), output_shape=(4, 4), reps=(2, 2),
        )
        assert result is not None and set(result.axes) == {0, 1}

    def test_non_constant_reps_drops(self):
        from flopscope._symmetry_transport import transport_tile
        G = _sym(0, 1)
        # reps (2, 1) NOT constant on orbit {0, 1}; output (4, 2).
        result = transport_tile(
            G, input_shape=(2, 2), output_shape=(4, 2), reps=(2, 1),
        )
        assert result is None

    def test_reps_outside_block_preserves(self):
        from flopscope._symmetry_transport import transport_tile
        G = _sym(0, 1)
        # reps (1, 1, 2) on (3, 3, 3): axis 2 outside block; constant on orbit {0,1}.
        result = transport_tile(
            G, input_shape=(3, 3, 3), output_shape=(3, 3, 6), reps=(1, 1, 2),
        )
        assert result is not None and set(result.axes) == {0, 1}

    def test_C3_constant_reps_preserves(self):
        from flopscope._symmetry_transport import transport_tile
        G = _cyc(0, 1, 2)
        result = transport_tile(
            G, input_shape=(3, 3, 3), output_shape=(6, 6, 6), reps=(2, 2, 2),
        )
        assert result is not None and result.order() == 3

    def test_reps_longer_than_input_shifts_block(self):
        from flopscope._symmetry_transport import transport_tile
        G = _sym(0, 1)
        # Input (3, 3), reps=(1, 1, 1) -> output rank 3, input prepended with 1 axis.
        # Block axes shift from (0, 1) to (1, 2).
        result = transport_tile(
            G, input_shape=(3, 3), output_shape=(1, 3, 3), reps=(1, 1, 1),
        )
        assert result is not None and set(result.axes) == {1, 2}

    def test_none_input_returns_none(self):
        from flopscope._symmetry_transport import transport_tile
        assert transport_tile(
            None, input_shape=(3, 3), output_shape=(6, 6), reps=(2, 2),
        ) is None
