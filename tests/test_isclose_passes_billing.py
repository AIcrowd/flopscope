"""``isclose`` and ``allclose`` bill the passes numpy's ``isclose`` runs (#343).

numpy's ``isclose`` is ``less_equal(abs(x-y), atol + rtol * abs(y)) &
isfinite(y) | (x == y)``, with ``result |= isnan(x) & isnan(y)`` under
``equal_nan=True``. The bill priced the tolerance core alone, six passes per
element; it now prices the four passes after it, and the four more that
``equal_nan=True`` adds. ``allclose`` is ``all(isclose(...))``.
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


def _isclose_steps(x, y, rtol=1e-05, atol=1e-08, equal_nan=False):
    """numpy's ``isclose`` for float arrays in flopscope ops, in numpy's order."""
    core = fnp.less_equal(
        fnp.abs(fnp.subtract(x, y)), fnp.add(atol, fnp.multiply(rtol, fnp.abs(y)))
    )
    result = fnp.bitwise_or(fnp.bitwise_and(core, fnp.isfinite(y)), fnp.equal(x, y))
    if equal_nan:
        both_nan = fnp.bitwise_and(fnp.isnan(x), fnp.isnan(y))
        result = fnp.bitwise_or(result, both_nan)
    return result


def _pair(shape, seed=0):
    rng = np.random.default_rng(seed)
    xn = rng.standard_normal(shape)
    yn = xn + rng.standard_normal(shape) * 1e-6
    xn.flat[::7] = np.nan
    yn.flat[::5] = np.nan
    return xn, yn


@pytest.mark.parametrize("equal_nan", [False, True])
@pytest.mark.parametrize("shape", [(1000,), (100, 100), (7, 11, 13)])
def test_isclose_bills_numpys_steps(shape, equal_nan):
    xn, yn = _pair(shape)
    x, y = fnp.asarray(xn), fnp.asarray(yn)
    billed, out = _billed(lambda: fnp.isclose(x, y, equal_nan=equal_nan))
    steps, _ = _billed(lambda: _isclose_steps(x, y, equal_nan=equal_nan))
    assert billed == steps
    np.testing.assert_array_equal(
        np.asarray(out), np.isclose(xn, yn, equal_nan=equal_nan)
    )


def test_isclose_with_tolerances_bills_numpys_steps():
    xn, yn = _pair((50, 40), seed=1)
    x, y = fnp.asarray(xn), fnp.asarray(yn)
    billed, out = _billed(lambda: fnp.isclose(x, y, rtol=1e-3, atol=1e-6))
    steps, _ = _billed(lambda: _isclose_steps(x, y, rtol=1e-3, atol=1e-6))
    assert billed == steps
    np.testing.assert_array_equal(
        np.asarray(out), np.isclose(xn, yn, rtol=1e-3, atol=1e-6)
    )


@pytest.mark.parametrize("equal_nan", [False, True])
def test_allclose_bills_numpys_steps(equal_nan):
    xn, yn = _pair((300, 30), seed=2)
    x, y = fnp.asarray(xn), fnp.asarray(yn)
    billed, out = _billed(lambda: fnp.allclose(x, y, equal_nan=equal_nan))
    steps, _ = _billed(lambda: fnp.all(_isclose_steps(x, y, equal_nan=equal_nan)))
    assert billed == steps
    assert bool(out) == np.allclose(xn, yn, equal_nan=equal_nan)
