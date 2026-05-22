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
