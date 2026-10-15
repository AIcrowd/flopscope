"""``var``/``std`` bill the broadcast ``a - mean`` when ``mean=`` enlarges ``a`` (#273).

numpy uses a caller-supplied ``mean`` as given: ``x = a - mean`` broadcasts
before it is squared and summed, so a ``mean`` with extra (or longer)
dimensions computes a batch of variances. The bill must follow that broadcast
shape, exactly as the same call on the materialised ``a`` is billed. A mean
that does not enlarge ``a`` (the ordinary ``keepdims`` shape) bills as before.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


def _bill(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as bc:
        out = fn()
    return int(bc.flops_used), out


def _materialised(a, shape):
    """``a`` broadcast to ``shape`` and copied, built outside any budget."""
    return fnp.asarray(np.broadcast_to(np.asarray(a), shape).copy())


@pytest.mark.parametrize("op", ["var", "std"])
def test_broadcast_mean_bills_the_batch_issue_shape(op):
    # The issue's first case: one series, B different means.
    rng = np.random.default_rng(0)
    B, N = 512, 2048
    a = fnp.asarray(rng.standard_normal((1, N)))
    m = fnp.asarray(rng.standard_normal((B, 1)))
    f = getattr(fnp, op)
    billed, got = _bill(lambda: f(a, axis=1, keepdims=True, mean=m))
    full = _materialised(a, (B, N))
    billed_full, want = _bill(lambda: f(full, axis=1, keepdims=True, mean=m))
    # 4 passes over the (B, N) centred array, + B square roots for std.
    sqrt = B if op == "std" else 0
    assert billed == 4 * B * N + sqrt
    assert billed == billed_full
    assert got.shape == (B, 1)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("op", ["var", "std"])
def test_pairwise_mean_bills_every_pair(op):
    # The issue's second case, scaled down: X (A, 1, D) against Y (1, Bq, D)
    # yields all A x Bq mean squared distances.
    rng = np.random.default_rng(1)
    A, Bq, D = 24, 16, 32
    X = fnp.asarray(rng.standard_normal((A, 1, D)))
    Y = fnp.asarray(rng.standard_normal((1, Bq, D)))
    f = getattr(fnp, op)
    billed, got = _bill(lambda: f(X, axis=-1, mean=Y))
    full = _materialised(X, (A, Bq, D))
    billed_full, want = _bill(lambda: f(full, axis=-1, mean=Y))
    sqrt = A * Bq if op == "std" else 0
    assert billed == 4 * A * Bq * D + sqrt
    assert billed == billed_full
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("op", ["var", "std"])
def test_non_enlarging_mean_is_unchanged(op):
    # A mean of the reduced (keepdims) shape does not broadcast ``a`` to
    # anything larger: the bill is the plain four-pass formula on ``a``.
    rng = np.random.default_rng(2)
    a = fnp.asarray(rng.standard_normal((64, 128)))
    m = fnp.asarray(rng.standard_normal((64, 1)))
    f = getattr(fnp, op)
    with_mean, _ = _bill(lambda: f(a, axis=1, keepdims=True, mean=m))
    without, _ = _bill(lambda: f(a, axis=1, keepdims=True))
    sqrt = 64 if op == "std" else 0
    assert with_mean == without == 4 * 64 * 128 + sqrt
    # A full-shape mean and a scalar mean do not enlarge ``a`` either.
    full_mean, _ = _bill(lambda: f(a, axis=1, mean=fnp.zeros((64, 128))))
    scalar_mean, _ = _bill(lambda: f(a, axis=None, mean=0.0))
    assert full_mean == with_mean
    assert scalar_mean == 4 * 64 * 128 + (1 if op == "std" else 0)


def test_broadcast_mean_drops_the_operand_symmetry_tag():
    # ``a``'s tag describes ``a``; the broadcast ``a - mean`` is a different
    # array, so it is billed dense and the result carries no tag.
    rng = np.random.default_rng(3)
    n, B = 12, 5
    t = rng.standard_normal((n, n))
    sym = flops.as_symmetric(fnp.asarray(t + t.T), symmetry=(0, 1))
    m = fnp.asarray(rng.standard_normal((B, 1, 1)))
    billed, got = _bill(lambda: fnp.var(sym, axis=-1, mean=m))
    assert billed == 4 * B * n * n
    assert got.shape == (B, n)
    assert getattr(got, "symmetry", None) is None


def test_nanvar_with_enlarging_mean_is_still_refused():
    # nanvar/nanstd centre ``a`` in place, so numpy refuses a mean that would
    # enlarge it; the fix applies to var/std only.
    a = fnp.asarray(np.ones((1, 8)))
    m = fnp.asarray(np.zeros((3, 1)))
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        for f in (fnp.nanvar, fnp.nanstd):
            with pytest.raises(ValueError):
                f(a, axis=1, keepdims=True, mean=m)
