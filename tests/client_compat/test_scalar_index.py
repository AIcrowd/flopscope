"""A 0-D RemoteScalar must be usable as a Python index (numpy parity).

Prod regression (subs 310044, 310045, 310059, 310067): "tuple indices must be
integers or slices, not RemoteScalar" — a scalar from indexing/reduction used to
index a Python tuple/list. v22 added __index__ to RemoteArray, not RemoteScalar.
"""

from __future__ import annotations

import pytest

import flopscope as fnp


def _int_scalar(v):
    s = fnp.array([v, v + 1])[0]
    assert type(s).__name__ == "RemoteScalar"
    return s


def test_scalar_indexes_tuple():
    s = _int_scalar(2)
    assert ("a", "b", "c", "d")[s] == "c"


def test_scalar_indexes_list_and_range():
    s = _int_scalar(1)
    assert [10, 20, 30][s] == 20
    assert list(range(s, s + 2)) == [1, 2]


def test_noninteger_scalar_rejects_index():
    s_frac = fnp.array([1.5, 2.5])[0]  # fractional float scalar
    with pytest.raises((TypeError, ValueError)):
        ("a", "b")[s_frac]
    s_whole = fnp.array([2.0, 3.0])[0]  # WHOLE-valued float scalar: numpy still rejects
    with pytest.raises((TypeError, ValueError)):
        ("a", "b", "c")[s_whole]
