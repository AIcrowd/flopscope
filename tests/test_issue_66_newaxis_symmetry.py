"""Regression tests for issue #66: propagate symmetry through None/np.newaxis indexing.

See .aicrowd/superpowers/specs/2026-05-22-issue-66-newaxis-symmetry-design.md.
"""

import numpy
import pytest

import flopscope as flops
import flopscope._free_ops as ops
import flopscope.numpy as fnp
from flopscope import SymmetryGroup
from flopscope._symmetry_utils import inserted_axes_symmetry


# ---------------------------------------------------------------------------
# inserted_axes_symmetry() unit tests
# ---------------------------------------------------------------------------


def test_inserted_axes_symmetry_returns_none_for_empty():
    assert inserted_axes_symmetry([]) is None


def test_inserted_axes_symmetry_returns_none_for_single_position():
    assert inserted_axes_symmetry([5]) is None


def test_inserted_axes_symmetry_returns_symmetric_pair_for_two_positions():
    expected = SymmetryGroup.symmetric(axes=(0, 2))
    assert inserted_axes_symmetry([0, 2]) == expected


def test_inserted_axes_symmetry_returns_symmetric_triple_for_three_positions():
    expected = SymmetryGroup.symmetric(axes=(1, 3, 5))
    assert inserted_axes_symmetry([1, 3, 5]) == expected


def test_inserted_axes_symmetry_accepts_tuple_input():
    expected = SymmetryGroup.symmetric(axes=(0, 4))
    assert inserted_axes_symmetry((0, 4)) == expected


# ---------------------------------------------------------------------------
# Contract: tensor[K_with_Nones].symmetry matches expand_dims equivalent
# ---------------------------------------------------------------------------


def test_issue_66_example_1_corrected_full_slices():
    """eye(3)[None, :, None, :] should match expand_dims(eye(3), axis=(0, 2))."""
    a = fnp.eye(3)
    by_slice = a[None, :, None, :]
    assert by_slice.shape == (1, 3, 1, 3)
    assert by_slice.symmetry == SymmetryGroup.young(blocks=((0, 2), (1, 3)))


def test_issue_66_example_2_corrected_asymmetric_slice():
    """eye(3)[None, :, None, 1:2] keeps inserted-axis pair; original sym broken."""
    a = fnp.eye(3)
    by_slice = a[None, :, None, 1:2]
    assert by_slice.shape == (1, 3, 1, 1)
    assert by_slice.symmetry == SymmetryGroup.young(blocks=((0, 2),))


def test_original_sym_broken_inserted_still_survives():
    """eye(3)[None, None, 0:1, 1:2] retains the two-inserted-axes pair only."""
    a = fnp.eye(3)
    by_slice = a[None, None, 0:1, 1:2]
    assert by_slice.shape == (1, 1, 1, 1)
    assert by_slice.symmetry == SymmetryGroup.symmetric(axes=(0, 1))


def test_single_none_builds_no_inserted_group():
    """eye(3)[None, :, :] has remapped original sym only; no inserted group."""
    a = fnp.eye(3)
    by_slice = a[None, :, :]
    assert by_slice.shape == (1, 3, 3)
    assert by_slice.symmetry == SymmetryGroup.symmetric(axes=(1, 2))


def test_preexisting_size_one_not_lumped_with_inserted():
    """A (3,1) SymmetricTensor indexed [None, :, None] makes only the two fresh
    Nones symmetric, NOT lumping with the pre-existing size-1 axis.

    Note: fnp.zeros returns a FlopscopeArray (not SymmetricTensor), which
    doesn't go through our __getitem__ path. We use .view(SymmetricTensor)
    to construct a SymmetricTensor whose _symmetry is None via
    __array_finalize__.
    """
    from flopscope._symmetric import SymmetricTensor

    data = numpy.zeros((3, 1))
    tensor = data.view(SymmetricTensor)
    assert tensor._symmetry is None
    by_slice = tensor[None, :, None]
    assert by_slice.shape == (1, 3, 1, 1)
    # Inserted axes [0, 2]. Pre-existing size-1 at axis 3 is NOT lumped.
    assert getattr(by_slice, "symmetry", None) == SymmetryGroup.symmetric(axes=(0, 2))


def test_ellipsis_and_none_combine():
    """eye(3)[None, ..., None] gives remapped sym((1,2)) + inserted sym((0,3))."""
    a = fnp.eye(3)
    by_slice = a[None, ..., None]
    assert by_slice.shape == (1, 3, 3, 1)
    expected = SymmetryGroup.direct_product(
        SymmetryGroup.symmetric(axes=(1, 2)),
        SymmetryGroup.symmetric(axes=(0, 3)),
    )
    assert by_slice.symmetry == expected


def test_integer_index_mid_key_with_nones():
    """eye(3)[None, 0, None, :] removes input axis 0; output positions 0, 1 are inserted."""
    a = fnp.eye(3)
    by_slice = a[None, 0, None, :]
    assert by_slice.shape == (1, 1, 3)
    # Output axes 0 and 1 are the freshly-inserted Nones; sym((0, 1)).
    # (Original sym((0,1)) is dropped: input axis 0 was removed by integer index,
    #  so pointwise stabilizer leaves nothing on the kept axis 1 alone.)
    assert getattr(by_slice, "symmetry", None) == SymmetryGroup.symmetric(axes=(0, 1))


def test_no_input_symmetry_still_gains_inserted_group():
    """A SymmetricTensor with _symmetry=None still constructs inserted-axis sym."""
    from flopscope._symmetric import SymmetricTensor

    # Construct a SymmetricTensor whose _symmetry is None via __array_finalize__:
    # numpy.asarray(...).view(SymmetricTensor) goes through __array_finalize__,
    # which initializes _symmetry to None.
    base = numpy.zeros((3, 3))
    sym_tensor = base.view(SymmetricTensor)
    assert sym_tensor._symmetry is None
    result = sym_tensor[None, None, :, :]
    assert result.shape == (1, 1, 3, 3)
    # Inserted positions 0, 1 form a free sym((0,1)) — symmetry from nothing.
    assert getattr(result, "symmetry", None) == SymmetryGroup.symmetric(axes=(0, 1))
