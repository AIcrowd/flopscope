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
