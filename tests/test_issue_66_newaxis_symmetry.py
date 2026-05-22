"""Regression tests for issue #66: propagate symmetry through None/np.newaxis indexing.

See .aicrowd/superpowers/specs/2026-05-22-issue-66-newaxis-symmetry-design.md.
"""

import numpy
import pytest

import flopscope as flops
import flopscope._free_ops as ops
import flopscope.numpy as fnp
from flopscope import SymmetryGroup
from flopscope._symmetric import SymmetricTensor
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
    assert isinstance(by_slice, SymmetricTensor)
    assert by_slice.shape == (1, 3, 1, 3)
    assert by_slice.symmetry == SymmetryGroup.young(blocks=((0, 2), (1, 3)))


def test_issue_66_example_2_corrected_asymmetric_slice():
    """eye(3)[None, :, None, 1:2] keeps inserted-axis pair; original sym broken."""
    a = fnp.eye(3)
    by_slice = a[None, :, None, 1:2]
    assert isinstance(by_slice, SymmetricTensor)
    assert by_slice.shape == (1, 3, 1, 1)
    assert by_slice.symmetry == SymmetryGroup.young(blocks=((0, 2),))


def test_original_sym_broken_inserted_still_survives():
    """eye(3)[None, None, 0:1, 1:2] retains the two-inserted-axes pair only."""
    a = fnp.eye(3)
    by_slice = a[None, None, 0:1, 1:2]
    assert isinstance(by_slice, SymmetricTensor)
    assert by_slice.shape == (1, 1, 1, 1)
    assert by_slice.symmetry == SymmetryGroup.symmetric(axes=(0, 1))


def test_single_none_builds_no_inserted_group():
    """eye(3)[None, :, :] has remapped original sym only; no inserted group."""
    a = fnp.eye(3)
    by_slice = a[None, :, :]
    assert isinstance(by_slice, SymmetricTensor)
    assert by_slice.shape == (1, 3, 3)
    assert by_slice.symmetry == SymmetryGroup.symmetric(axes=(1, 2))


def test_preexisting_size_one_not_lumped_when_no_input_symmetry():
    """No-input-symmetry path: a (3,1) view-as-SymmetricTensor indexed
    [None, :, None] gives sym((0, 2)) over the two fresh Nones, NOT
    sym((0, 2, 3)) which would also include the pre-existing size-1.

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
    # Inserted axes [0, 2]. Pre-existing size-1 at output axis 3 is NOT lumped.
    assert getattr(by_slice, "symmetry", None) == SymmetryGroup.symmetric(axes=(0, 2))


def test_preexisting_size_one_not_lumped_with_existing_symmetry():
    """A SymmetricTensor with non-trivial symmetry AND a pre-existing size-1
    axis: indexing with Nones produces a direct product of the remapped original
    sym and the inserted-axis sym, with the pre-existing size-1 axis excluded
    from both. This is the case the "not lumped" rule is really about.

    Input: shape (3, 1, 3) with sym((0, 2)); axis 1 is the pre-existing size-1.
    Key:   [None, :, :, None, :]
    Output: shape (1, 3, 1, 1, 3); pre-existing size-1 lands at output axis 2.
    Expected:
      - remapped original sym → sym((1, 4))  (input axes 0, 2 → output axes 1, 4)
      - inserted axes (None positions 0, 3 in output) → sym((0, 3))
      - Output axis 2 (the pre-existing size-1) is in NEITHER group.
    """
    sym_matrix = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [3.0, 5.0, 6.0],
        ]
    )
    data = sym_matrix[:, numpy.newaxis, :]
    tensor = flops.as_symmetric(
        data, symmetry=SymmetryGroup.symmetric(axes=(0, 2))
    )
    assert tensor.shape == (3, 1, 3)
    assert tensor.symmetry == SymmetryGroup.symmetric(axes=(0, 2))

    by_slice = tensor[None, :, :, None, :]
    assert isinstance(by_slice, SymmetricTensor)
    assert by_slice.shape == (1, 3, 1, 1, 3)

    expected = SymmetryGroup.direct_product(
        SymmetryGroup.symmetric(axes=(1, 4)),  # remapped original
        SymmetryGroup.symmetric(axes=(0, 3)),  # inserted axes only
    )
    assert by_slice.symmetry == expected

    # Negative assertion: explicitly NOT the version where the pre-existing
    # size-1 (output axis 2) is lumped with the freshly-inserted Nones.
    wrong_if_lumped = SymmetryGroup.direct_product(
        SymmetryGroup.symmetric(axes=(1, 4)),
        SymmetryGroup.symmetric(axes=(0, 2, 3)),
    )
    assert by_slice.symmetry != wrong_if_lumped


def test_ellipsis_and_none_combine():
    """eye(3)[None, ..., None] gives remapped sym((1,2)) + inserted sym((0,3))."""
    a = fnp.eye(3)
    by_slice = a[None, ..., None]
    assert isinstance(by_slice, SymmetricTensor)
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


# ---------------------------------------------------------------------------
# Warning predicate: SymmetryLossWarning fires only on real structural
# reduction (new.order() < old.order()), not on gains or axis-relabels.
# ---------------------------------------------------------------------------


def _count_symmetry_warnings(callable_):
    """Helper: run `callable_()` and count SymmetryLossWarning instances."""
    import warnings as _warnings

    from flopscope.errors import SymmetryLossWarning

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        callable_()
        return sum(
            1 for w in caught if issubclass(w.category, SymmetryLossWarning)
        )


def test_no_warning_on_gained_symmetry_via_none_insertion():
    """`a[None, :, None, :]` produces RICHER symmetry, not loss — no warning."""
    a = fnp.eye(3)
    n = _count_symmetry_warnings(lambda: a[None, :, None, :])
    assert n == 0


def test_no_warning_on_axis_relabel_via_single_none():
    """`a[None, :, :]` shifts axes but preserves order — no warning."""
    a = fnp.eye(3)
    n = _count_symmetry_warnings(lambda: a[None, :, :])
    assert n == 0


def test_no_warning_on_ellipsis_and_none():
    """`a[None, ..., None]` gains an inserted pair on top of remapped original — no warning."""
    a = fnp.eye(3)
    n = _count_symmetry_warnings(lambda: a[None, ..., None])
    assert n == 0


def test_warning_fires_on_real_order_reduction():
    """`a[:, 0]` removes axis 1 from the symmetric group; original order drops to 1."""
    a = fnp.eye(3)
    n = _count_symmetry_warnings(lambda: a[:, 0])
    assert n == 1


def test_warning_fires_on_asymmetric_slice():
    """`a[0:2, 0:2]` breaks the original sym; resulting tensor is no longer symmetric."""
    a = fnp.eye(3)
    n = _count_symmetry_warnings(lambda: a[0:2, 0:2])
    assert n == 1


# ---------------------------------------------------------------------------
# Bool indexing bailout: boolean masks are not integer scalars.
# `isinstance(True, int)` is True in Python, so without an explicit bool check
# the propagator would treat `True` as an integer index (removes axis 0) when
# numpy actually treats it as a boolean mask (adds a size-1 batch axis).
# Symmetry conservatively drops to None.
# ---------------------------------------------------------------------------


def test_bool_scalar_index_drops_symmetry_safely():
    """tensor[True] is a boolean mask in numpy; bail out to no symmetry."""
    a = fnp.eye(3)
    assert a.shape == (3, 3)
    result = a[True]
    assert result.shape == (1, 3, 3)
    assert getattr(result, "symmetry", None) is None


def test_bool_index_combined_with_none_drops_symmetry_safely():
    """Combining True with None must not produce a misclassified inserted-axis
    group: positions 0 and 2 are size-1 in output, but so is position 1 (the
    bool-mask axis), and the propagator has no way to distinguish them. Drop
    to None rather than report a wrong group.
    """
    a = fnp.eye(3)
    result = a[None, True, None]
    assert result.shape == (1, 1, 1, 3, 3)
    assert getattr(result, "symmetry", None) is None


def test_numpy_bool_scalar_index_drops_symmetry_safely():
    """numpy.bool_ behaves like Python bool for indexing; same bailout."""
    a = fnp.eye(3)
    result = a[numpy.bool_(True)]
    assert result.shape == (1, 3, 3)
    assert getattr(result, "symmetry", None) is None


# ---------------------------------------------------------------------------
# Equivalence property: tensor[K] == expand_dims(tensor[slice_part], newaxes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, slice_part, expand_dims_axis",
    [
        # (None, :, None, :) — strip Nones → (:, :); expand at output positions (0, 2)
        (
            (None, slice(None), None, slice(None)),
            (slice(None), slice(None)),
            (0, 2),
        ),
        # (None, :, None, 1:2) — strip Nones → (:, 1:2); expand at (0, 2)
        (
            (None, slice(None), None, slice(1, 2)),
            (slice(None), slice(1, 2)),
            (0, 2),
        ),
        # (None, None, 0:1, 1:2) — strip Nones → (0:1, 1:2); expand at (0, 1)
        (
            (None, None, slice(0, 1), slice(1, 2)),
            (slice(0, 1), slice(1, 2)),
            (0, 1),
        ),
        # (None, 0, None, :) — strip Nones → (0, :); expand at output positions (0, 1)
        (
            (None, 0, None, slice(None)),
            (0, slice(None)),
            (0, 1),
        ),
    ],
)
def test_newaxis_indexing_equivalent_to_slice_then_expand_dims(
    key, slice_part, expand_dims_axis
):
    a = fnp.eye(3)
    by_slice = a[key]
    intermediate = a[slice_part]
    by_expand = ops.expand_dims(intermediate, axis=expand_dims_axis)

    assert numpy.array_equal(by_slice, by_expand)
    by_slice_sym = getattr(by_slice, "symmetry", None)
    by_expand_sym = getattr(by_expand, "symmetry", None)
    assert by_slice_sym == by_expand_sym
