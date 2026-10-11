"""A group declared with ``einsum(..., symmetry=)`` describes the data it tags (#284).

The declared group is validated with the same tolerant check as
``as_symmetric``, but the tag is read as exact. A result that matched only
within the tolerance must be canonicalized like ``as_symmetric`` does, or a
small asymmetric result keeps its tag after scaling and every later op is
billed at the orbit rate for data that is not symmetric. A result that is
already exactly invariant is returned untouched.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope.errors import SymmetryError

SG = flops.SymmetryGroup


def _ctx():
    return flops.BudgetContext(flop_budget=10**16, quiet=True)


def _bill(fn):
    with _ctx() as bc:
        out = fn()
    return int(bc.flops_used), out


def _tag(arr):
    """The symmetry tag ``arr`` carries, or None."""
    return getattr(arr, "symmetry", None)


def _max_deviation(arr) -> float:
    """Largest difference between ``arr`` and any of its axis permutations."""
    raw = np.asarray(arr)
    return max(
        float(np.max(np.abs(raw - np.transpose(raw, perm))))
        for perm in itertools.permutations(range(raw.ndim))
    )


def test_issue_repro_scaled_result_stays_symmetric():
    B = np.random.default_rng(0).standard_normal((48, 48))  # not symmetric
    G = SG.symmetric(axes=(0, 1))
    with _ctx():
        small = fnp.multiply(fnp.asarray(B), 1e-9)
        tagged = fnp.einsum("ij->ij", small, symmetry=G)
        reference = flops.as_symmetric(small, symmetry=(0, 1))
        X = fnp.multiply(tagged, 1e9)
    assert _tag(tagged) == G
    # The same canonical values as_symmetric produces, so scaling them back
    # up cannot recover the asymmetric entries (4.62 before the fix).
    np.testing.assert_array_equal(np.asarray(tagged), np.asarray(reference))
    assert _tag(X) == G
    assert _max_deviation(X) == 0.0
    assert not np.shares_memory(np.asarray(tagged), np.asarray(small))


def test_three_axis_full_group_is_canonicalized():
    T = np.random.default_rng(1).standard_normal((24, 24, 24))
    G = SG.symmetric(axes=(0, 1, 2))
    with _ctx():
        small = fnp.multiply(fnp.asarray(T), 1e-9)
        Y = fnp.multiply(fnp.einsum("ijk->ijk", small, symmetry=G), 1e9)
        reference = fnp.multiply(flops.as_symmetric(small, symmetry=G), 1e9)
    assert _tag(Y) == G
    assert _max_deviation(Y) == 0.0
    np.testing.assert_array_equal(np.asarray(Y), np.asarray(reference))


def test_later_ops_bill_like_the_as_symmetric_twin_and_do_not_raise():
    B = np.random.default_rng(2).standard_normal((32, 32))
    G = SG.symmetric(axes=(0, 1))
    with _ctx():
        small = fnp.multiply(fnp.asarray(B), 1e-9)
        X = fnp.multiply(fnp.einsum("ij->ij", small, symmetry=G), 1e9)
        twin = fnp.multiply(flops.as_symmetric(small, symmetry=(0, 1)), 1e9)
    for op in (
        fnp.exp,
        lambda A: fnp.multiply(A, A),
        lambda A: fnp.einsum("ij,ij->", A, A),
        # These raised SymmetryError on the asymmetric tagged result.
        lambda A: fnp.matmul(A, A),
        lambda A: fnp.einsum("ij,jk->ik", A, A),
    ):
        billed, got = _bill(lambda op=op: op(X))
        want_bill, want = _bill(lambda op=op: op(twin))
        assert billed == want_bill
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_exactly_symmetric_result_is_returned_unchanged():
    a = np.random.default_rng(3).standard_normal((16, 16))
    S = a + a.T
    G = SG.symmetric(axes=(0, 1))
    plain_bill, plain = _bill(lambda: fnp.einsum("ij->ij", fnp.asarray(S)))
    billed, got = _bill(lambda: fnp.einsum("ij->ij", fnp.asarray(S), symmetry=G))
    assert _tag(got) == G
    np.testing.assert_array_equal(np.asarray(got), S)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(plain))
    assert billed == plain_bill


def test_declared_group_the_data_does_not_satisfy_still_raises():
    B = np.random.default_rng(4).standard_normal((16, 16))
    with _ctx(), pytest.raises(SymmetryError):
        fnp.einsum("ij->ij", fnp.asarray(B), symmetry=SG.symmetric(axes=(0, 1)))
