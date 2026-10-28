"""One-operand einsum diagonals over a partly tagged tensor (issue #348).

A SymmetricTensor whose group acts on some of its axes (as ``stack`` and
``as_symmetric`` on a subset of axes produce) had its permutation moved onto
the output of a diagonal even when a label on a moved axis also sat on an axis
the group leaves alone, so ``einsum('iij->ij', stack)`` claimed a symmetric
result and raised SymmetryError. These tests compare one-operand einsums with
numpy for several tag placements, check that tags which do hold are still
carried, and check that the error reports the measured deviation.
"""

import itertools

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope import SymmetryGroup
from flopscope._perm_group import _PermutationCompat as Permutation
from flopscope._symmetric import SymmetricTensor
from flopscope.errors import SymmetryError

EQUATIONS_3D = [
    "".join(t) + "->" + "".join(o)
    for t in itertools.product("ij", repeat=3)
    if len(set(t)) == 2
    for r in range(3)
    for o in itertools.permutations("ij", r)
]
EQUATIONS_4D = [
    "".join(t) + "->" + "".join(o)
    for t in itertools.product("ijk", repeat=4)
    if len(set(t)) == 3
    for r in (2, 3)
    for o in itertools.permutations("ijk", r)
]


@pytest.fixture
def ctx():
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as budget:
        yield budget


def _symmetrized(shape, axes, seed=0):
    """Random data made invariant under every permutation of ``axes``."""
    base = np.random.default_rng(seed).standard_normal(shape)
    total = np.zeros(shape)
    for perm in itertools.permutations(axes):
        order = list(range(len(shape)))
        for src, dst in zip(axes, perm, strict=True):
            order[src] = dst
        total += base.transpose(order)
    return total


def _tagged(data, group):
    tensor = flops.as_symmetric(fnp.asarray(data), symmetry=group)
    assert type(tensor) is SymmetricTensor
    return tensor


def _stack(seed=0) -> SymmetricTensor:
    rng = np.random.default_rng(seed)
    vs = [fnp.asarray(rng.standard_normal(3)) for _ in range(3)]
    stack = fnp.stack([fnp.outer(v, v) for v in vs])
    assert isinstance(stack, SymmetricTensor)
    return stack


def _mismatches(tensor, data, equations):
    bad = []
    for eq in equations:
        try:
            got = np.asarray(fnp.einsum(eq, tensor))
        except Exception as exc:
            bad.append(f"{eq}: {type(exc).__name__}")
            continue
        if not np.allclose(got, np.einsum(eq, data)):
            bad.append(f"{eq}: wrong value")
    return bad


def test_equation_sweeps_have_the_expected_size():
    assert len(EQUATIONS_3D) == 30
    assert len(EQUATIONS_4D) == 432


@pytest.mark.parametrize("eq", EQUATIONS_3D)
@pytest.mark.parametrize("axes", [(0, 1), (0, 2), (1, 2)])
def test_diagonal_over_a_degree_2_tag_matches_numpy(ctx, axes, eq):
    data = _symmetrized((3, 3, 3), axes)
    tensor = _tagged(data, SymmetryGroup.symmetric(axes=axes))
    np.testing.assert_allclose(np.asarray(fnp.einsum(eq, tensor)), np.einsum(eq, data))


@pytest.mark.parametrize("eq", EQUATIONS_3D)
def test_diagonal_over_a_stack_of_symmetric_matrices_matches_numpy(ctx, eq):
    stack = _stack()
    assert stack.symmetry.axes == (1, 2)
    data = np.asarray(stack).copy()
    np.testing.assert_allclose(np.asarray(fnp.einsum(eq, stack)), np.einsum(eq, data))


@pytest.mark.parametrize("eq", EQUATIONS_3D)
def test_group_stored_with_a_fixed_point_matches_numpy(ctx, eq):
    data = _symmetrized((3, 3, 3), (0, 1))
    tensor = _tagged(data, SymmetryGroup(Permutation([1, 0, 2]), axes=(0, 1, 2)))
    np.testing.assert_allclose(np.asarray(fnp.einsum(eq, tensor)), np.einsum(eq, data))


@pytest.mark.parametrize("eq", EQUATIONS_3D)
def test_both_storage_forms_of_one_group_give_the_same_result(ctx, eq):
    data = _symmetrized((3, 3, 3), (0, 1))
    short = fnp.einsum(eq, _tagged(data, SymmetryGroup.symmetric(axes=(0, 1))))
    padded = fnp.einsum(
        eq, _tagged(data, SymmetryGroup(Permutation([1, 0, 2]), axes=(0, 1, 2)))
    )
    np.testing.assert_allclose(np.asarray(short), np.asarray(padded))
    assert getattr(short, "symmetry", None) == getattr(padded, "symmetry", None)


def test_untagged_copies_match_numpy(ctx):
    for axes in [(0, 1), (0, 2), (1, 2)]:
        data = _symmetrized((3, 3, 3), axes)
        assert _mismatches(fnp.asarray(data), data, EQUATIONS_3D) == []


@pytest.mark.parametrize(
    "axes", [(0, 1), (1, 2), (2, 3), (1, 2, 3), (0, 1, 2, 3)], ids=str
)
def test_four_axis_diagonals_match_numpy(ctx, axes):
    data = _symmetrized((3, 3, 3, 3), axes)
    tensor = _tagged(data, SymmetryGroup.symmetric(axes=axes))
    assert _mismatches(tensor, data, EQUATIONS_4D) == []


def test_tag_on_axes_away_from_the_diagonal_is_kept(ctx):
    data = _symmetrized((3, 3, 3, 3), (2, 3))
    out = fnp.einsum("iijk->ijk", _tagged(data, SymmetryGroup.symmetric(axes=(2, 3))))
    np.testing.assert_allclose(np.asarray(out), np.einsum("iijk->ijk", data))
    assert isinstance(out, SymmetricTensor)
    assert out.symmetry == SymmetryGroup.symmetric(axes=(1, 2))


def test_fixed_point_label_repeated_elsewhere_keeps_the_swap(ctx):
    data = _symmetrized((3, 3, 3, 3), (0, 1))
    tensor = _tagged(data, SymmetryGroup(Permutation([1, 0, 2]), axes=(0, 1, 2)))
    out = fnp.einsum("ijkk->ijk", tensor)
    np.testing.assert_allclose(np.asarray(out), np.einsum("ijkk->ijk", data))
    assert isinstance(out, SymmetricTensor)
    assert out.symmetry == SymmetryGroup(Permutation([1, 0, 2]), axes=(0, 1, 2))


def test_partial_trace_of_a_fully_symmetric_tensor_keeps_the_swap(ctx):
    data = _symmetrized((3, 3, 3, 3), (0, 1, 2, 3))
    tensor = _tagged(data, SymmetryGroup.symmetric(axes=(0, 1, 2, 3)))
    out = fnp.einsum("ijkk->ij", tensor)
    np.testing.assert_allclose(np.asarray(out), np.einsum("ijkk->ij", data))
    assert isinstance(out, SymmetricTensor)
    assert out.symmetry == SymmetryGroup.symmetric(axes=(0, 1))


def test_batched_transpose_of_a_stack_keeps_the_tag(ctx):
    stack = _stack()
    out = fnp.einsum("bij->bji", stack)
    np.testing.assert_allclose(np.asarray(out), np.asarray(stack).transpose(0, 2, 1))
    assert isinstance(out, SymmetricTensor)
    assert out.symmetry == stack.symmetry


def test_symmetry_error_reports_the_measured_deviation(ctx):
    data = np.arange(9.0).reshape(3, 3)
    with pytest.raises(SymmetryError) as info:
        fnp.einsum(
            "ij->ij", fnp.asarray(data), symmetry=SymmetryGroup.symmetric(axes=(0, 1))
        )
    assert info.value.max_deviation == pytest.approx(np.abs(data - data.T).max())


def test_max_generator_deviation_by_dtype():
    from flopscope._symmetric import _max_generator_deviation

    swap = SymmetryGroup.symmetric(axes=(0, 1))
    assert _max_generator_deviation(np.array([[True, False], [True, True]]), swap) == 1
    unsigned = np.array([[0, 1], [3, 0]], dtype=np.uint8)
    assert _max_generator_deviation(unsigned, swap) == 2
    assert _max_generator_deviation(np.array([["a", "b"], ["c", "d"]]), swap) == float(
        "inf"
    )
