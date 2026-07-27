"""Index-form accounting for ufunc.at.

Ground truth: ``ufunc.at`` applies the ufunc once per selected cell and does
NOT deduplicate repeated indices, so the application count is exactly the size
of the indexing result. We assert against that oracle rather than against magic
numbers, so these tests keep their value as the cost model evolves.
"""

from __future__ import annotations

import numpy as np
import pytest

from flopscope._pointwise import _canonical_index, _ufunc_at_touched_cells


def oracle(shape, indices) -> int:
    """True ufunc.at application count = size of the indexing result."""
    return int(np.empty(shape, np.uint8)[indices].size)


# (shape, indices, label)
INDEX_FORMS = [
    ((6,), 3, "int scalar"),
    ((6,), -1, "negative int"),
    ((6,), np.array([0, 1, 2]), "int ndarray"),
    ((6,), [0, 0, 0, 0], "repeated list"),
    ((6,), np.zeros(1000, np.intp), "heavy repeat"),
    ((6,), range(3), "range"),
    ((6,), memoryview(np.array([1, 2, 3], np.intp)), "memoryview"),
    ((6,), np.array([], np.intp), "empty ndarray"),
    ((6,), [], "empty list"),
    ((6,), slice(None), "full slice"),
    ((6,), slice(1, 5, 2), "stepped slice"),
    ((6,), slice(None, None, -1), "negative step"),
    ((6,), slice(4, 2), "empty slice"),
    ((6,), True, "bool scalar True"),
    ((6,), False, "bool scalar False"),
    ((6,), np.bool_(True), "np.bool_ True"),
    ((6,), np.array(True), "0-d bool array"),
    ((6,), np.array(2), "0-d int array"),
    ((6,), np.array([True, False, True, True, False, True]), "1-d mask"),
    ((6,), [True, False, True, True, False, True], "list of bools"),
    ((6,), Ellipsis, "bare Ellipsis"),
    ((4, 5), (slice(None), slice(None)), "two full slices"),
    ((4, 5), (slice(0, 3), slice(1, 4)), "two partial slices"),
    ((4, 5), (np.array([0, 1]), np.array([2, 3])), "two int arrays"),
    ((4, 5), (np.array([0, 1]), slice(None)), "advanced + slice"),
    ((4, 5), (np.array([0, 1]), 2), "advanced + int"),
    ((4, 5), (np.array([0, 1]), Ellipsis), "advanced + Ellipsis"),
    ((4, 5), (Ellipsis, np.array([0, 1])), "Ellipsis + advanced"),
    ((4, 5), np.ones((4, 5), bool), "2-d full mask"),
    ((4, 5), np.zeros((4, 5), bool), "2-d empty mask"),
    ((4, 5), (None, 0), "newaxis leading"),
    ((4, 5), (0, None), "newaxis trailing"),
    ((2, 3, 4), (np.array([0, 1]), Ellipsis, np.array([0, 1])), "adv-Ellipsis-adv"),
    ((2, 3, 4), np.ones((2, 3), bool), "k-d partial mask"),
    ((1, 4096), (np.zeros(1000, np.intp), Ellipsis), "repeat + Ellipsis"),
    ((1, 4096), (np.zeros(1000, np.intp), slice(None)), "repeat + slice"),
]


@pytest.mark.parametrize(
    "shape,indices,label",
    INDEX_FORMS,
    ids=[lbl for _, _, lbl in INDEX_FORMS],
)
def test_touched_cells_matches_oracle(shape, indices, label):
    canon = _canonical_index(indices)
    expected = max(oracle(shape, indices), 1)
    assert _ufunc_at_touched_cells(np.empty(shape, np.float32), canon) == expected


@pytest.mark.parametrize(
    "shape,indices,label",
    INDEX_FORMS,
    ids=[lbl for _, _, lbl in INDEX_FORMS],
)
def test_canonical_index_preserves_write(shape, indices, label):
    """The canonical index must write exactly what the raw index writes."""
    raw = np.zeros(shape, np.float64)
    canon_target = np.zeros(shape, np.float64)
    np.add.at(raw, indices, 1.0)
    np.add.at(canon_target, _canonical_index(indices), 1.0)
    np.testing.assert_array_equal(raw, canon_target)


def test_bool_is_not_treated_as_int():
    """Python bool subclasses int; numpy treats it as a 0-d mask adding an axis."""
    assert (
        _ufunc_at_touched_cells(np.empty((6,), np.float32), _canonical_index(True)) == 6
    )
    assert _ufunc_at_touched_cells(np.empty((6,), np.float32), _canonical_index(1)) == 1


def test_non_integer_array_index_raises_indexerror():
    with pytest.raises(IndexError):
        _canonical_index(np.array([1.5, 2.5]))


def test_canonical_index_does_not_freeze_caller_array():
    """We must never make a participant's own array read-only."""
    caller = np.array([0, 1, 2], np.intp)

    class Holder:
        def __array__(self, dtype=None, copy=None):
            return caller

    _canonical_index(Holder())
    assert caller.flags.writeable is True
