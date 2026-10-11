"""``trapezoid`` bills once per integral numpy forms, not once per ``y`` (#271).

numpy sums ``d * (y[1:] + y[:-1]) / 2`` along ``axis`` with ``d = diff(x)``.
An N-D ``x`` broadcasts ``d`` against ``y``, so a stack of K sample grids
integrates ``y`` K times. The bill is ``4 * numel(y) * K``; for a 1-D ``x``,
an ``x`` of ``y``'s own shape, or ``dx`` (K = 1) it is the old
``4 * numel(y)``.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp

_OPS = ["trapezoid"] + (["trapz"] if hasattr(np, "trapz") else [])

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _bill(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as bc:
        out = fn()
    return int(bc.flops_used), out


@pytest.mark.parametrize("op", _OPS)
def test_stack_of_sample_grids_bills_every_integral(op):
    # The issue's shape: y (1024,) against 512 sample grids x (512, 1024).
    rng = np.random.default_rng(0)
    y = rng.standard_normal(1024).astype(np.float32)
    x = np.cumsum(rng.random((512, 1024)).astype(np.float32), axis=1)
    fy, fx = fnp.asarray(y), fnp.asarray(x)
    f = getattr(fnp, op)
    billed, got = _bill(lambda: f(fy, fx))
    assert got.shape == (512,)
    assert billed == 4 * 1024 * 512
    # The same integrals one grid at a time cost the same per grid.
    one_grid, _ = _bill(lambda: f(fy, fx[0]))
    assert billed == 512 * one_grid
    np.testing.assert_array_equal(np.asarray(got), np.trapezoid(y, x))


@pytest.mark.parametrize("op", _OPS)
def test_ordinary_calls_are_unchanged(op):
    rng = np.random.default_rng(2)
    f = getattr(fnp, op)
    y2 = fnp.asarray(rng.standard_normal((16, 64)))
    x1 = fnp.asarray(np.cumsum(rng.random(64)))
    x2 = fnp.asarray(np.cumsum(rng.random((16, 64)), axis=1))
    x0 = fnp.asarray(np.cumsum(rng.random(16)))
    want = 4 * 16 * 64
    # scalar dx, 1-D x along the last axis, x of y's own shape and a 1-D x
    # along axis 0: all K = 1.
    assert _bill(lambda: f(y2))[0] == want
    assert _bill(lambda: f(y2, dx=0.5))[0] == want
    assert _bill(lambda: f(y2, x1))[0] == want
    assert _bill(lambda: f(y2, x2))[0] == want
    assert _bill(lambda: f(y2, x0, axis=0))[0] == want


@pytest.mark.parametrize("op", _OPS)
def test_higher_rank_y_against_a_grid_stack_on_axis_0(op):
    # y (64, 3) integrated along axis 0 over x (5, 64, 3): 5 grids, each
    # integrating all 3 columns.
    rng = np.random.default_rng(3)
    y = rng.standard_normal((64, 3))
    x = np.cumsum(rng.random((5, 64, 3)), axis=1)
    fy, fx = fnp.asarray(y), fnp.asarray(x)
    billed, got = _bill(lambda: getattr(fnp, op)(fy, fx, axis=-2))
    assert got.shape == (5, 3)
    assert billed == 4 * 64 * 3 * 5
    np.testing.assert_allclose(np.asarray(got), np.trapezoid(y, x, axis=-2))
