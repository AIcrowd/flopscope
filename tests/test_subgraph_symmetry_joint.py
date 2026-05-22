"""Sprint 2 Task 2: verify SubsetSymmetry.joint exposes the value-preserving
joint group on V ∪ W.
"""

import numpy as np

from flopscope import SymmetryGroup
from flopscope._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle


def test_subset_symmetry_joint_for_4_identical_S2():
    """4-cycle ij,jk,kl,li->ijkl with 4 identical S₂: joint order = 8 (D₄).

    The joint group includes both rotations (from operand-cycle swap, Source B)
    and reflections (from per-operand S₂ Source A composed with operand swaps).
    Since V = (i,j,k,l) and W = (), joint must equal output (which is D₄).
    """
    S2 = SymmetryGroup.symmetric(axes=(0, 1))
    dummy = np.empty((4, 4))
    ops = [dummy, dummy, dummy, dummy]
    oracle = SubgraphSymmetryOracle(
        operands=ops,
        subscript_parts=["ij", "jk", "kl", "li"],
        per_op_groups=[[S2], [S2], [S2], [S2]],
        output_chars="ijkl",
    )
    ss = oracle.sym(frozenset({0, 1, 2, 3}))
    assert ss.joint is not None
    assert ss.output is not None
    # Joint must equal output when V = full label set (no W).
    output_order = len(list(ss.output.elements()))
    joint_order = len(list(ss.joint.elements()))
    assert joint_order == output_order == 8, (
        f"joint order {joint_order} should equal output order {output_order} == 8"
    )
    # Joint labels should cover the full V ∪ W set
    assert ss.joint._labels is not None
    assert set(ss.joint._labels) == {"i", "j", "k", "l"}


def test_subset_symmetry_joint_for_dense_distinct():
    """All distinct dense operands: joint should be None (no symmetry)."""
    dummy_a = np.empty((4, 4))
    dummy_b = np.empty((4, 4))
    oracle = SubgraphSymmetryOracle(
        operands=[dummy_a, dummy_b],
        subscript_parts=["ij", "jk"],
        per_op_groups=[None, None],
        output_chars="ik",
    )
    ss = oracle.sym(frozenset({0, 1}))
    assert ss.joint is None


def test_joint_contains_output_projection():
    """For any subset query, the joint group's V-projection should equal
    the output group (when V is preserved setwise by the joint).
    """
    S2 = SymmetryGroup.symmetric(axes=(0, 1))
    dummy = np.empty((4, 4))
    oracle = SubgraphSymmetryOracle(
        operands=[dummy, dummy],
        subscript_parts=["ij", "jk"],
        per_op_groups=[[S2], [S2]],
        output_chars="ik",
    )
    ss = oracle.sym(frozenset({0, 1}))
    # The joint group's elements, projected onto V positions, should generate
    # the output group.  Order(joint) >= order(output) always.
    if ss.joint is not None and ss.output is not None:
        joint_order = len(list(ss.joint.elements()))
        output_order = len(list(ss.output.elements()))
        assert joint_order >= output_order, (
            f"joint order {joint_order} < output order {output_order} — joint is too small"
        )
