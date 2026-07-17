"""Fancy indexing with plain Python list keys (numpy parity).

``x[[0, 1]]`` selects rows; it must not be conflated with ``x[0, 1]``
(element access). The wire protocol used to encode both as a msgpack list
and the server heuristically decoded len>1 lists as tuples, so ``x[[0, 1]]``
on a 2-D array silently returned the scalar ``x[0, 1]`` and raised
IndexError on 1-D arrays.
"""

from __future__ import annotations

import flopscope as fnp


def test_list_key_2d_selects_rows():
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert a[[0, 1]].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_list_key_2d_reorders_rows():
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert a[[1, 0]].tolist() == [[3.0, 4.0], [1.0, 2.0]]


def test_list_key_1d():
    a = fnp.array([10.0, 20.0, 30.0])
    assert a[[0, 1]].tolist() == [10.0, 20.0]


def test_single_element_list_key_keeps_dimension():
    a = fnp.array([10.0, 20.0, 30.0])
    assert a[[1]].tolist() == [20.0]


def test_nested_list_key():
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    # 2-D index array selects rows -> shape (2, 2, 2)
    assert a[[[0, 1], [1, 0]]].tolist() == [
        [[1.0, 2.0], [3.0, 4.0]],
        [[3.0, 4.0], [1.0, 2.0]],
    ]


def test_bool_mask_list_2d():
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    # Boolean mask selects row 0 only; int coercion of the bools would
    # instead reorder rows as a[[1, 0]].
    assert a[[True, False]].tolist() == [[1.0, 2.0]]


def test_bool_mask_list_1d():
    a = fnp.array([10.0, 20.0, 30.0])
    assert a[[False, True, True]].tolist() == [20.0, 30.0]


def test_tuple_key_still_element_access():
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert float(a[0, 1]) == 2.0


def test_tuple_with_list_component():
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert a[:, [1]].tolist() == [[2.0], [4.0]]


def test_slice_key_roundtrip():
    a = fnp.array([10.0, 20.0, 30.0, 40.0])
    assert a[1:3].tolist() == [20.0, 30.0]


def test_ellipsis_key_roundtrip():
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert a[..., 0].tolist() == [1.0, 3.0]
