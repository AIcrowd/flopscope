"""Tests for numpy 2.3+ behavioral shims preserving pre-2.3 semantics."""

from __future__ import annotations

import numpy as np
import pytest

import flopscope.numpy as fnp

fnp = fnp  # backwards-compat local alias for this test
# -------------------------------------------------------------------------
# count_nonzero: axis=None must return Python int on all numpy versions
# -------------------------------------------------------------------------


def test_count_nonzero_axis_none_returns_python_int():
    arr = np.array([0, 1, 2, 0, 3])
    result = fnp.count_nonzero(arr)
    assert type(result) is int, f"expected exact `int`, got {type(result).__name__}"
    assert result == 3


def test_count_nonzero_explicit_axis_none_returns_python_int():
    arr = np.array([[0, 1], [2, 0]])
    result = fnp.count_nonzero(arr, axis=None)
    assert type(result) is int
    assert result == 2


def test_count_nonzero_with_axis_returns_ndarray():
    """Negative case: shim does not interfere when axis is not None."""
    arr = np.array([[0, 1, 2], [3, 0, 4]])
    result = fnp.count_nonzero(arr, axis=0)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 1, 2])


def test_count_nonzero_int_return_is_unconditional():
    """Coercion to int for axis=None is unconditional regardless of numpy version."""
    import flopscope._pointwise as _pointwise

    arr = np.array([0, 1, 2, 0, 3])
    result = _pointwise.count_nonzero(arr)
    assert type(result) is int
    assert result == 3


# -------------------------------------------------------------------------
# unique: string / complex input must return sorted values on all versions
#
# The shim's ``_UNSORTED_IN_NP_2_3`` set names four kinds (str 'U', bytes
# 'S', object 'O', complex 'c'). U/S/O are now outside the numeric allowlist
# the dtype ban enforces, so a string/bytes/object array can never reach
# `unique` at all anymore -- the shim's re-sort branch is only reachable for
# complex today. The string-exemplar tests below are converted to pin the
# refusal instead; complex coverage is unaffected (complex is numeric).
# -------------------------------------------------------------------------


def test_unique_strings_are_refused():
    from flopscope.errors import UnsupportedDtypeError

    arr = np.array(["banana", "apple", "cherry", "apple"])
    with pytest.raises(UnsupportedDtypeError):
        fnp.unique(arr)


def test_unique_complex_returns_sorted():
    arr = np.array([2 + 1j, 1 + 0j, 2 + 0j, 1 + 0j], dtype=np.complex128)
    result = fnp.unique(arr)
    expected = np.sort(np.array([1 + 0j, 2 + 0j, 2 + 1j]))
    np.testing.assert_array_equal(result, expected)


def test_unique_numeric_sort_unaffected():
    """Negative case: shim is a no-op for non-string / non-complex dtypes."""
    arr = np.array([3.0, 1.0, 2.0, 1.0])
    result = fnp.unique(arr)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_unique_with_return_index_is_refused_for_strings():
    """Shim does not attempt to re-sort when auxiliary arrays are requested
    -- moot for strings now, since the operand itself is refused before the
    shim's return_index branch is ever reached."""
    from flopscope.errors import UnsupportedDtypeError

    arr = np.array(["b", "a", "c"])
    with pytest.raises(UnsupportedDtypeError):
        fnp.unique(arr, return_index=True)


def test_unique_shim_forced_for_strings_is_unreachable(monkeypatch):
    """The re-sort shim's string branch used to be forceable on numpy <2.3
    by faking ``_NUMPY_GE_2_3``; now the string operand is refused before
    ``unique``'s body -- and so before that branch -- ever runs, regardless
    of ``_NUMPY_GE_2_3``. Pins that the refusal wins the race, not the shim."""
    import flopscope._sorting_ops as _sorting_ops
    from flopscope.errors import UnsupportedDtypeError

    monkeypatch.setattr(_sorting_ops, "_NUMPY_GE_2_3", True)
    arr = np.array(["banana", "apple", "cherry"])
    with pytest.raises(UnsupportedDtypeError):
        _sorting_ops.unique(arr)
