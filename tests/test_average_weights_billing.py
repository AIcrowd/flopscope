"""``average(weights=)`` bills the weight sum numpy runs (#339).

numpy's ``average`` sums the weights once, ``scl = wgt.sum(axis=axis)``, and
refuses weights that sum to zero, ``np.any(scl == 0.0)``. Weights of shape
``a.shape[axis]`` are reshaped to broadcast along ``axis`` first, so their sum
adds ``numel(w) - 1`` numbers. The bill priced that sum over the whole input,
``numel - M``, whatever shape the weights had, and left out the zero check.
It now equals numpy's own steps written out in flopscope ops.
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


def _average_steps(a, axis, w, keepdims=False):
    """numpy's ``average(a, axis, weights=w)`` in flopscope ops, in numpy's order."""
    if tuple(np.shape(w)) != tuple(a.shape):
        ax = (axis,) if isinstance(axis, int) else tuple(axis)
        ax = tuple(i % a.ndim for i in ax)
        # numpy's own transpose and reshape of the weights: views, no arithmetic
        wn = np.transpose(np.asarray(w), tuple(np.argsort(ax)))
        w = fnp.asarray(
            wn.reshape(tuple(s if i in ax else 1 for i, s in enumerate(a.shape)))
        )
    scl = fnp.sum(w, axis=axis, keepdims=keepdims)
    fnp.any(fnp.equal(scl, 0.0))
    total = fnp.sum(fnp.multiply(a, w), axis=axis, keepdims=keepdims)
    return fnp.divide(total, scl)


CASES = [
    # (input shape, axis, weights shape)
    ((1000, 1000), 0, (1000,)),
    ((1000, 1000), 1, (1000,)),
    ((40, 30, 20), 1, (30,)),
    ((40, 30, 20), -1, (20,)),
    ((40, 30, 20), (0, 2), (40, 20)),
    ((40, 30, 20), (2, 0), (20, 40)),
    ((1000, 1000), 0, (1000, 1000)),
    ((1000, 1000), None, (1000, 1000)),
    ((40, 30, 20), (0, 2), (40, 30, 20)),
]


@pytest.mark.parametrize(("shape", "axis", "wshape"), CASES)
def test_average_bills_numpys_steps(shape, axis, wshape):
    rng = np.random.default_rng(0)
    an, wn = rng.standard_normal(shape), rng.random(wshape) + 0.5
    a, w = fnp.asarray(an), fnp.asarray(wn)
    billed, out = _billed(lambda: fnp.average(a, axis=axis, weights=w))
    steps, _ = _billed(lambda: _average_steps(a, axis, w))
    assert billed == steps
    np.testing.assert_allclose(np.asarray(out), np.average(an, axis=axis, weights=wn))


@pytest.mark.parametrize("keepdims", [False, True])
def test_keepdims_and_plain_weights(keepdims):
    rng = np.random.default_rng(1)
    an, wn = rng.standard_normal((50, 60)), rng.random(60) + 0.5
    a = fnp.asarray(an)
    billed, out = _billed(
        lambda: fnp.average(a, axis=1, weights=list(wn), keepdims=keepdims)
    )
    steps, _ = _billed(lambda: _average_steps(a, 1, wn, keepdims=keepdims))
    assert billed == steps
    ref = np.average(an, axis=1, weights=wn, keepdims=keepdims)
    np.testing.assert_allclose(np.asarray(out), ref)


def test_axis_weights_cost_far_less_than_full_shape_weights():
    rng = np.random.default_rng(2)
    a = fnp.asarray(rng.standard_normal((1000, 1000)))
    one_d, _ = _billed(lambda: fnp.average(a, axis=0, weights=rng.random(1000)))
    full, _ = _billed(lambda: fnp.average(a, axis=0, weights=rng.random((1000, 1000))))
    # the full-shape weight sum adds 999,000 numbers, the 1-D one 999
    # and its zero check looks at 1000 sums instead of one (1000 + 999 against 1 + 1)
    assert full - one_d == (1000 * 1000 - 1000) - (1000 - 1) + (1000 + 999) - (1 + 1)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_unweighted_average_is_still_mean(axis):
    a = fnp.asarray(np.random.default_rng(3).standard_normal((300, 200)))
    unweighted, _ = _billed(lambda: fnp.average(a, axis=axis))
    mean, _ = _billed(lambda: fnp.mean(a, axis=axis))
    assert unweighted == mean
