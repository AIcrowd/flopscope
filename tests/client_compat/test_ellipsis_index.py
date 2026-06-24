"""Ellipsis (...) in indexing must work (numpy parity).

Prod regression (sub 310351): "can not serialize 'ellipsis' object" — the client
index encoder had no branch for Ellipsis.
"""

from __future__ import annotations

import flopscope as fnp


def test_trailing_ellipsis():
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert a[0, ...].tolist() == [1.0, 2.0]


def test_leading_ellipsis():
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert a[..., 1].tolist() == [2.0, 4.0]


def test_bare_ellipsis_returns_full():
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert a[...].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_one_element_ellipsis_tuple():
    # ``arr[..., ]`` passes the key as ``(Ellipsis,)`` -> a one-element list on
    # the wire; the server must decode it back to a tuple ``(Ellipsis,)``, not a
    # bare ``[Ellipsis]`` (which numpy rejects). Equivalent to ``arr[...]``.
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert a[...,].tolist() == [[1.0, 2.0], [3.0, 4.0]]
