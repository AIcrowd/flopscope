"""Unit tests for compute_step_cost_from_joint_group (Sprint 2 Cat C helper)."""

from flopscope._accumulation._cost import compute_step_cost_from_joint_group
from flopscope._perm_group import SymmetryGroup
from flopscope._perm_group import _Permutation as Permutation


def test_joint_helper_4cycle_step2_D4_to_55():
    """§5(a) step 2 case: D4 on (i,j,k,l) for n=4.  Expected total = 55.

    Step 2's subscript is `lik,jki->ijkl`.  V = (i,j,k,l), W = ().
    D4 acting on V gives M = O = 55 orbits via Burnside.
    Formula: (2-1)·M + M − O = M = 55.
    """
    d4 = SymmetryGroup.dihedral(axes=(0, 1, 2, 3))
    d4._labels = ("i", "j", "k", "l")

    total = compute_step_cost_from_joint_group(
        joint_group=d4,
        v_labels=("i", "j", "k", "l"),
        w_labels=(),
        sizes={"i": 4, "j": 4, "k": 4, "l": 4},
        num_terms=2,
        dimino_budget=10000,
    )
    assert total == 55, f"expected 55, got {total}"


def test_joint_helper_returns_None_on_cross_V_W():
    """Cyclic C3 on (a,b,c) with V = (a,b) and W = (c): the cycle moves c -> a
    (W -> V), so V is NOT preserved.  Helper must return None.
    """
    c3 = SymmetryGroup.cyclic(axes=(0, 1, 2))
    c3._labels = ("a", "b", "c")

    total = compute_step_cost_from_joint_group(
        joint_group=c3,
        v_labels=("a", "b"),
        w_labels=("c",),
        sizes={"a": 4, "b": 4, "c": 4},
        num_terms=2,
        dimino_budget=10000,
    )
    assert total is None


def test_joint_helper_returns_None_for_trivial_group():
    """Group with no _labels -> returns None (label mismatch guard)."""
    # Construct a group with identity-only generator but no _labels set.
    trivial = SymmetryGroup(Permutation([0]), axes=(0,))
    # _labels is None by default (not set), so helper returns None at label check.
    assert trivial._labels is None

    total = compute_step_cost_from_joint_group(
        joint_group=trivial,
        v_labels=("a",),
        w_labels=(),
        sizes={"a": 4},
        num_terms=2,
        dimino_budget=10000,
    )
    assert total is None


def test_joint_helper_returns_None_when_None_input():
    """joint_group=None -> returns None."""
    total = compute_step_cost_from_joint_group(
        joint_group=None,
        v_labels=("a",),
        w_labels=(),
        sizes={"a": 4},
        num_terms=2,
        dimino_budget=10000,
    )
    assert total is None
