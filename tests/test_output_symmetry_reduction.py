"""Regression tests for single-operand reduction output-symmetry tagging.

Closes the pre-existing `_relabel_group_to_output` KeyError-on-summed-label bug
identified during Sprint 1 Task 2 and re-confirmed by the Sprint 2 adversarial
analysis.  After the fix, `_infer_pathless_output_symmetry` applies the
setwise-stabilizer-then-restrict pipeline (via the existing `reduce_group`
helper) when the operand's declared symmetry covers axes that get summed out.
"""

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp
from flopscope import SymmetryGroup
from flopscope._perm_group import _Permutation as Permutation
from flopscope._symmetric import SymmetricTensor


def test_S3_reduce_one_axis_yields_S2():
    """Canonical case: S₃ on (a,b,c), sum c → output S₂{a,b}.

    Setwise stabilizer of {c} in S₃ is {identity, (ab)} (order 2). Projected
    to surviving axes (a, b), this is S₂{a,b}.
    """
    T = flops.as_symmetric(
        np.zeros((4, 4, 4)),
        symmetry=SymmetryGroup.symmetric(axes=(0, 1, 2)),
    )
    result = fnp.einsum("abc->ab", T)
    assert isinstance(result, SymmetricTensor), (
        f"expected SymmetricTensor, got {type(result).__name__}"
    )
    assert result.symmetry is not None
    assert len(list(result.symmetry.elements())) == 2
    assert result.symmetry.axes == (0, 1)


def test_S3_reduce_two_axes_yields_trivial():
    """Over-reduction: S₃ on (a,b,c), sum b and c → only one axis left.

    Setwise stabilizer of {b, c} in S₃ is {identity, (bc)} but restricted to
    the single surviving axis (a), it collapses to trivial → no output symmetry.
    """
    T = flops.as_symmetric(
        np.zeros((4, 4, 4)),
        symmetry=SymmetryGroup.symmetric(axes=(0, 1, 2)),
    )
    result = fnp.einsum("abc->a", T)
    # Result should be a plain rank-1 tensor with no symmetry
    assert getattr(result, "symmetry", None) is None, (
        f"unexpected symmetry on rank-1 reduction: {getattr(result, 'symmetry', None)}"
    )


def test_cyclic_C3_dies_under_reduction():
    """Cyclic C₃ on (a,b,c), sum c → no surviving symmetry.

    C₃ = {identity, (abc), (acb)}.  Neither 3-cycle fixes {c} setwise (each
    moves c to a or b).  Setwise stabilizer of {c} = {identity} → trivial.
    """
    T = flops.as_symmetric(
        np.zeros((4, 4, 4)),
        symmetry=SymmetryGroup.cyclic(axes=(0, 1, 2)),
    )
    result = fnp.einsum("abc->ab", T)
    assert getattr(result, "symmetry", None) is None, (
        f"unexpected symmetry from cyclic group losing an axis: "
        f"{getattr(result, 'symmetry', None)}"
    )


def test_S2xS2_custom_reduce_partial():
    """Custom S₂{a,b} × S₂{c,d} on rank-4, sum c and d → S₂{a,b} survives.

    The (a↔b) generator fixes {c, d} setwise; the (c↔d) generator does NOT
    fix {a, b} setwise (it acts on the reduced axes).  After stabilizer
    filter + restrict: only (a↔b) survives → S₂ on (0, 1) of output.
    """
    T = flops.as_symmetric(
        np.zeros((3, 3, 3, 3)),
        symmetry=SymmetryGroup(
            Permutation([1, 0, 2, 3]),  # swap axes 0, 1 (a↔b)
            Permutation([0, 1, 3, 2]),  # swap axes 2, 3 (c↔d)
            axes=(0, 1, 2, 3),
        ),
    )
    result = fnp.einsum("abcd->ab", T)
    assert isinstance(result, SymmetricTensor), (
        f"expected SymmetricTensor, got {type(result).__name__}"
    )
    assert result.symmetry is not None
    assert len(list(result.symmetry.elements())) == 2
    assert result.symmetry.axes == (0, 1)


def test_partial_axis_S2_loses_when_reduced():
    """T rank-4 with S₂ on (0, 1) only (axes c, d untouched).
    Sum axis 0 (a): the S₂ swap involves the reduced axis → group dies.

    Setwise stabilizer of {0} in S₂{0,1} = {identity} (the (01) swap
    sends 0 → 1, not into {0}).  Restricted to surviving axes (1, 2, 3)
    = trivial.
    """
    T = flops.as_symmetric(
        np.zeros((3, 3, 3, 3)),
        symmetry=SymmetryGroup.symmetric(axes=(0, 1)),
    )
    result = fnp.einsum("abcd->bcd", T)
    assert getattr(result, "symmetry", None) is None, (
        f"unexpected symmetry after summing axis bound to S₂: "
        f"{getattr(result, 'symmetry', None)}"
    )


def test_output_transpose_with_reduction():
    """S₃ on (a,b,c), sum c, output transposed: `abc → ba`.

    Surviving S₂{a,b} must be remapped from operand axis-order (a at 0, b at 1)
    to output axis-order (b at 0, a at 1).  The S₂ swap is the same group
    element; just lives on axes (0, 1) of the output.
    """
    T = flops.as_symmetric(
        np.zeros((4, 4, 4)),
        symmetry=SymmetryGroup.symmetric(axes=(0, 1, 2)),
    )
    result = fnp.einsum("abc->ba", T)
    assert isinstance(result, SymmetricTensor), (
        f"expected SymmetricTensor, got {type(result).__name__}"
    )
    assert result.symmetry is not None
    assert len(list(result.symmetry.elements())) == 2
    assert result.symmetry.axes == (0, 1)
