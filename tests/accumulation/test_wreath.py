"""Tests for _wreath.py — port of wreath.js."""

import math

from flopscope._accumulation._wreath import (
    WreathElement,
    enumerate_h,
    enumerate_wreath,
)
from flopscope._perm_group import SymmetryGroup


def test_enumerate_h_returns_identity_for_none_symmetry():
    perms = list(enumerate_h(None, rank=3))
    assert len(perms) == 1
    assert perms[0].is_identity


def test_enumerate_h_full_symmetric_for_symmetric_declaration():
    # 'symmetric' on rank 3 → S_3, 6 elements
    perms = list(enumerate_h("symmetric", rank=3))
    assert len(perms) == math.factorial(3)


def test_enumerate_h_cyclic_for_cyclic_declaration():
    # 'cyclic' on rank 3 → C_3, 3 elements
    perms = list(enumerate_h("cyclic", rank=3))
    assert len(perms) == 3


def test_enumerate_h_dihedral_for_dihedral_declaration():
    # 'dihedral' on rank 4 → D_4, 8 elements
    perms = list(enumerate_h("dihedral", rank=4))
    assert len(perms) == 8


def test_enumerate_h_from_symmetry_group_object():
    """Accept a SymmetryGroup directly (flopscope's primary symmetry type)."""
    s2 = SymmetryGroup.symmetric(axes=(0, 1))
    perms = list(enumerate_h(s2, rank=3))
    # S_2 on first two axes, third axis fixed: 2 elements at rank 3
    assert len(perms) == 2


def test_enumerate_wreath_no_symmetry_no_repeats_yields_identity_only():
    elements = list(
        enumerate_wreath(
            identical_groups=((0,), (1,)),
            per_op_symmetry=(None, None),
            axis_ranks=(2, 2),
            u_offsets=(0, 2),
        )
    )
    assert len(elements) == 1
    assert elements[0].row_perm.is_identity


def test_enumerate_wreath_two_identical_operands_includes_swap():
    """Two copies of the same 2-axis operand → wreath has S_2 on operands × identity^2.
    Total: 2 elements."""
    elements = list(
        enumerate_wreath(
            identical_groups=((0, 1),),
            per_op_symmetry=(None, None),
            axis_ranks=(2, 2),
            u_offsets=(0, 2),
        )
    )
    assert len(elements) == 2


def test_enumerate_wreath_declared_symmetry_grows_count():
    """One operand with rank 2 + 'symmetric' declaration: H_0 = S_2 → 2 elements."""
    elements = list(
        enumerate_wreath(
            identical_groups=((0,),),
            per_op_symmetry=("symmetric",),
            axis_ranks=(2,),
            u_offsets=(0,),
        )
    )
    assert len(elements) == 2


def test_wreath_element_has_factorization():
    """Each WreathElement carries provenance for diagnostic display."""
    elements = list(
        enumerate_wreath(
            identical_groups=((0, 1),),
            per_op_symmetry=(None, None),
            axis_ranks=(2, 2),
            u_offsets=(0, 2),
        )
    )
    for e in elements:
        assert isinstance(e, WreathElement)
        assert hasattr(e, "factorization")
