"""``mean``, ``var`` and ``std`` bill the count numpy takes of a ``where=`` mask (#330).

With a mask, numpy's ``_count_reduce_items`` sums the mask broadcast to the
input's shape before dividing (``umr_sum(broadcast_to(where, arr.shape),
axis, ...)``), one more full reduction the unmasked call skips. The bill adds
what that count bills when written out as ``sum`` of the broadcast mask.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


def _billed(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        before = int(ctx.flops_used)
        out = fn()
        return int(ctx.flops_used) - before, out


def _count_bill(mask, shape, axis, keepdims):
    """What numpy's mask count bills written out: ``sum`` of the broadcast mask."""
    full = fnp.asarray(np.broadcast_to(np.asarray(mask), shape).copy())
    billed, _ = _billed(lambda: fnp.sum(full, axis=axis, keepdims=keepdims))
    return billed


OPS = ["mean", "var", "std"]


@pytest.mark.parametrize("op", OPS)
def test_issue_repro_where_adds_the_mask_count(op):
    rng = np.random.default_rng(0)
    a = fnp.asarray(rng.standard_normal((100, 100)))
    M = rng.random((100, 100)) > 0.3
    f = getattr(fnp, op)
    plain, _ = _billed(lambda: f(a, axis=1))
    masked, got = _billed(lambda: f(a, axis=1, where=M))
    count = _count_bill(M, (100, 100), 1, False)
    assert count == 100 * 100 - 100
    assert masked == plain + count
    np.testing.assert_array_equal(
        np.asarray(got), getattr(np, op)(np.asarray(a), axis=1, where=M)
    )


CASES = [
    # input shape, mask shape, axis, keepdims
    ((100, 100), (1, 100), 1, False),
    ((100, 100), (100, 1), 0, False),
    ((6, 7), (6, 7), None, False),
    ((6, 7), (6, 7), 0, True),
    ((4, 5, 6), (5, 6), (0, 2), False),
    ((4, 5, 6), (4, 5, 6), -1, True),
]


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(("shape", "mask_shape", "axis", "keepdims"), CASES)
def test_where_mask_bills_the_broadcast_count(op, shape, mask_shape, axis, keepdims):
    # A mask smaller than the input is broadcast first, so it costs the same
    # full-size count.
    rng = np.random.default_rng(1)
    a = fnp.asarray(rng.standard_normal(shape))
    mask = rng.random(mask_shape) > 0.1
    f = getattr(fnp, op)
    plain, _ = _billed(lambda: f(a, axis=axis, keepdims=keepdims))
    masked, got = _billed(lambda: f(a, axis=axis, keepdims=keepdims, where=mask))
    count = _count_bill(mask, shape, axis, keepdims)
    assert count > 0
    assert masked == plain + count
    want = getattr(np, op)(np.asarray(a), axis=axis, keepdims=keepdims, where=mask)
    np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("op", OPS)
def test_method_spelling_bills_the_count_too(op):
    rng = np.random.default_rng(2)
    a = fnp.asarray(rng.standard_normal((30, 40)))
    M = rng.random((30, 40)) > 0.1
    plain, _ = _billed(lambda: getattr(fnp, op)(a, axis=1))
    via_method, _ = _billed(lambda: getattr(a, op)(axis=1, where=M))
    assert via_method == plain + _count_bill(M, (30, 40), 1, False)


@pytest.mark.parametrize("op", OPS)
def test_where_true_takes_numpys_fast_path(op):
    a = fnp.asarray(np.random.default_rng(3).standard_normal((30, 40)))
    f = getattr(fnp, op)
    plain, _ = _billed(lambda: f(a, axis=0))
    explicit, _ = _billed(lambda: f(a, axis=0, where=True))
    assert explicit == plain


def test_sum_runs_no_count():
    rng = np.random.default_rng(4)
    a = fnp.asarray(rng.standard_normal((30, 40)))
    M = rng.random((30, 40)) > 0.5
    plain, _ = _billed(lambda: fnp.sum(a, axis=1))
    masked, _ = _billed(lambda: fnp.sum(a, axis=1, where=M))
    assert masked == plain
