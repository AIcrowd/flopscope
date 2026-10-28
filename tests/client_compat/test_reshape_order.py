"""RemoteArray.reshape takes ``order=`` as ndarray.reshape does (issue #347).

The method used to accept only the shape, so ``x.reshape(2, 8, order="F")``
raised TypeError on the client for every array. Expected values are built in
plain Python so that nothing here depends on the patched numpy.
"""

from __future__ import annotations

import pytest

import flopscope as fnp

FLAT = [float(i) for i in range(16)]


def _c_order(flat, rows, cols):
    return [[flat[i * cols + j] for j in range(cols)] for i in range(rows)]


def _f_order(flat, rows, cols):
    return [[flat[i + rows * j] for j in range(cols)] for i in range(rows)]


SPELLINGS = {"ints": (2, 8), "tuple": ((2, 8),), "list": ([2, 8],)}


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_reshape_order_f_on_a_vector(spelling):
    a = fnp.array(FLAT)
    assert a.reshape(*SPELLINGS[spelling], order="F").tolist() == _f_order(FLAT, 2, 8)


def test_reshape_order_f_on_a_matrix_reads_it_column_by_column():
    rows = _c_order(FLAT, 4, 4)
    by_column = [rows[i][j] for j in range(4) for i in range(4)]
    a = fnp.array(rows)
    assert a.reshape(2, 8, order="F").tolist() == _f_order(by_column, 2, 8)


def test_reshape_explicit_order_c_is_the_default():
    a = fnp.array(FLAT)
    assert a.reshape(2, 8, order="C").tolist() == _c_order(FLAT, 2, 8)


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_reshape_without_order_stays_c_order(spelling):
    a = fnp.array(FLAT)
    assert a.reshape(*SPELLINGS[spelling]).tolist() == _c_order(FLAT, 2, 8)
