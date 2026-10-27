"""``polyfit`` bills its column scaling and its ``w=`` weighting (#333).

numpy's ``polyfit`` scales every column of the Vandermonde matrix before the
``lstsq`` solve and unscales the coefficients after it, on every call
(``scale = sqrt((lhs*lhs).sum(axis=0))``, ``lhs /= scale``,
``c = (c.T/scale).T``), and with ``w=`` it first weights both sides. The bill
covered only the Vandermonde build and the solve.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._polynomial import polyfit_covariance_cost
from flopscope.numpy.linalg import lstsq_cost


def _billed(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        before = int(ctx.flops_used)
        out = fn()
        return int(ctx.flops_used) - before, out


def _solve_price(m, deg, ncols):
    """The Vandermonde build plus the solve, the part already billed."""
    b_ndim = 1 if ncols == 1 else 2
    return m * deg + lstsq_cost(m, deg + 1, b_cols=ncols, b_ndim=b_ndim)


def _scale_and_weight_bill(x, y, deg, w=None):
    """What numpy's steps around the solve bill, written out in flopscope ops.

    The Vandermonde matrix and the coefficients are built outside the budget;
    only the weighting, the scaling and the unscaling are billed.
    """
    n = deg + 1
    lhs = fnp.asarray(np.vander(np.asarray(x), n))
    coef = fnp.asarray(np.ones((n,) + tuple(np.shape(y)[1:])))

    def steps():
        v = lhs
        if w is not None:
            ww = w + 0.0
            v = v * ww[:, None]
            _ = y * (ww[:, None] if y.ndim == 2 else ww)
        scale = fnp.sqrt(fnp.sum(v * v, axis=0))
        v = v / scale
        return (coef.T / scale).T

    billed, _ = _billed(steps)
    return billed


def _numpy_fit(x, y, deg, w, **kw):
    return np.polyfit(
        np.asarray(x), np.asarray(y), deg, w=None if w is None else np.asarray(w), **kw
    )


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("deg", [1, 3])
def test_issue_repro_polyfit_bills_scaling_and_weighting(deg, weighted):
    rng = np.random.default_rng(0)
    m = 1000
    x = fnp.asarray(rng.standard_normal(m))
    y = fnp.asarray(rng.standard_normal(m))
    w = fnp.asarray(rng.random(m) + 0.5) if weighted else None
    billed, got = _billed(lambda: fnp.polyfit(x, y, deg, w=w))
    assert np.array_equal(np.asarray(got), _numpy_fit(x, y, deg, w))
    assert billed == _solve_price(m, deg, 1) + _scale_and_weight_bill(x, y, deg, w)


CASES = [
    # m, deg, ncols (0 for a 1-D y), weighted
    (50, 0, 0, False),
    (50, 0, 0, True),
    (200, 2, 0, False),
    (200, 5, 0, True),
    (120, 2, 3, False),
    (120, 2, 3, True),
]


@pytest.mark.parametrize(("m", "deg", "ncols", "weighted"), CASES)
def test_polyfit_bills_numpys_steps_around_the_solve(m, deg, ncols, weighted):
    rng = np.random.default_rng(1)
    x = fnp.asarray(rng.standard_normal(m))
    y = fnp.asarray(rng.standard_normal((m, ncols) if ncols else m))
    w = fnp.asarray(rng.random(m) + 0.5) if weighted else None
    billed, got = _billed(lambda: fnp.polyfit(x, y, deg, w=w))
    assert np.array_equal(np.asarray(got), _numpy_fit(x, y, deg, w))
    extra = _scale_and_weight_bill(x, y, deg, w)
    assert billed == _solve_price(m, deg, max(ncols, 1)) + extra


def test_full_output_bills_the_scaling_too():
    rng = np.random.default_rng(2)
    x = fnp.asarray(rng.standard_normal(80))
    y = fnp.asarray(rng.standard_normal(80))
    billed, got = _billed(lambda: fnp.polyfit(x, y, 2, full=True))
    assert np.array_equal(np.asarray(got[0]), _numpy_fit(x, y, 2, None, full=True)[0])
    assert billed == _solve_price(80, 2, 1) + _scale_and_weight_bill(x, y, 2)


@pytest.mark.parametrize("cov", [True, "unscaled"])
def test_covariance_branch_keeps_its_price(cov):
    rng = np.random.default_rng(3)
    m, deg = 200, 2
    x = fnp.asarray(rng.standard_normal(m))
    y = fnp.asarray(rng.standard_normal(m))
    plain, _ = _billed(lambda: fnp.polyfit(x, y, deg))
    with_cov, _ = _billed(lambda: fnp.polyfit(x, y, deg, cov=cov))
    scaled = cov != "unscaled"
    expected = polyfit_covariance_cost(m, deg, (deg + 1) ** 2, scaled=scaled)
    assert with_cov - plain == expected
