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
