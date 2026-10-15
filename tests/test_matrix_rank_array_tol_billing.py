"""``linalg.matrix_rank`` bills its threshold compares per tolerance (#274).

With an array ``tol=`` or ``rtol=``, numpy compares the singular values of
every matrix against every entry of the tolerance array (broadcast against the
batch) and returns one rank per pair. The ``min(m, n)`` compare term of the
cost is paid once per (matrix, tolerance) pair; a scalar threshold gives one
pair per matrix, which is the unchanged ordinary case.
"""

from __future__ import annotations

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp
from flopscope.numpy.linalg import matrix_rank_cost


def _bill(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as bc:
        out = fn()
    return int(bc.flops_used), out


def test_array_tol_bills_every_tolerance_issue_shape():
    rng = np.random.default_rng(0)
    T = rng.uniform(0.0, 6.0, (1024, 1024))
    fT = fnp.asarray(T)
    for k in (1, 16):
        A = fnp.asarray(rng.standard_normal((k, k)))
        billed, got = _bill(lambda A=A: fnp.linalg.matrix_rank(A, tol=fT))
        assert got.shape == (1024, 1024)
        assert billed == matrix_rank_cost(k, k) + k * (1024 * 1024 - 1)
        np.testing.assert_array_equal(
            np.asarray(got), np.linalg.matrix_rank(np.asarray(A), tol=T)
        )


def test_array_rtol_bills_every_tolerance():
    A = fnp.asarray(np.ones((1, 1)))
    R = fnp.asarray(np.random.default_rng(1).random((1024, 1024)))
    billed, got = _bill(lambda: fnp.linalg.matrix_rank(A, rtol=R))
    assert got.shape == (1024, 1024)
    assert billed == matrix_rank_cost(1, 1) + (1024 * 1024 - 1)


def test_tolerance_broadcast_against_a_batch():
    rng = np.random.default_rng(2)
    A = fnp.asarray(rng.standard_normal((4, 8, 8)))
    # tol (3, 1) against the batch (4,): 12 (matrix, tolerance) pairs.
    tol = fnp.asarray(rng.random((3, 1)))
    billed, got = _bill(lambda: fnp.linalg.matrix_rank(A, tol=tol))
    assert got.shape == (3, 4)
    assert billed == 4 * matrix_rank_cost(8, 8) + 8 * (12 - 4)


def test_scalar_and_per_matrix_thresholds_are_unchanged():
    rng = np.random.default_rng(3)
    A = fnp.asarray(rng.standard_normal((4, 8, 8)))
    want = 4 * matrix_rank_cost(8, 8)
    assert _bill(lambda: fnp.linalg.matrix_rank(A))[0] == want
    assert _bill(lambda: fnp.linalg.matrix_rank(A, tol=0.5))[0] == want
    assert _bill(lambda: fnp.linalg.matrix_rank(A, rtol=1e-3))[0] == want
    per_matrix = fnp.asarray(rng.random(4))
    assert _bill(lambda: fnp.linalg.matrix_rank(A, tol=per_matrix))[0] == want
