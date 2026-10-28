"""RemoteArray has the real, imag and mT attributes ndarray has (issue #352).

They were missing, so ``fnp.linalg.eigvals(W).real`` and ``B.mT`` raised
AttributeError on the client while working in-process. Each attribute must
return what its function form returns; expected layouts are built in plain
Python.
"""

from __future__ import annotations

import flopscope.numpy as fnp

W = [[0.9, -0.4, 0.1], [0.5, 0.8, -0.2], [0.0, 0.3, 0.7]]


def test_real_and_imag_of_complex_eigenvalues():
    ev = fnp.linalg.eigvals(fnp.asarray(W))
    assert any(v != 0.0 for v in fnp.imag(ev).tolist())  # the eigenvalues are complex
    assert ev.real.tolist() == fnp.real(ev).tolist()
    assert ev.imag.tolist() == fnp.imag(ev).tolist()


def test_real_of_eig_eigenvalues():
    ev = fnp.linalg.eig(fnp.asarray(W)).eigenvalues
    assert ev.real.tolist() == fnp.real(ev).tolist()


def test_real_and_imag_of_a_real_array():
    a = fnp.asarray(W)
    assert a.real.tolist() == W
    assert a.imag.tolist() == [[0.0, 0.0, 0.0] for _ in range(3)]


def test_mT_of_a_batch_and_of_a_matrix():
    rows = [[float(4 * i + j) for j in range(4)] for i in range(3)]
    stack = [rows, [[v + 12.0 for v in r] for r in rows]]
    batch = fnp.asarray(stack)
    expected = [[[m[i][j] for i in range(3)] for j in range(4)] for m in stack]
    assert batch.mT.tolist() == expected
    assert batch.mT.tolist() == fnp.matrix_transpose(batch).tolist()
    assert fnp.asarray(rows).mT.tolist() == expected[0]
