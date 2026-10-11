"""``interp(period=)`` bills the work numpy's period branch runs (#276).

With ``period=`` numpy reduces ``x`` and ``xp`` modulo the period, argsorts
``xp``, gathers ``xp`` and ``fp`` in that order and pads both with one wrapped
point at each end before it interpolates on the padded table. None of that was
billed: the call cost what ``interp`` costs on a presorted table, although the
argsort of ``xp`` alone is far more expensive. The charge now equals the same
sequence spelled out with the ops numpy runs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._weights import load_weights


@pytest.fixture
def production_weights(monkeypatch):
    """Load the packaged production weights for one test (conftest resets them)."""
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    load_weights()
    yield


def _billed(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        before = int(ctx.flops_used)
        out = fn()
        return int(ctx.flops_used) - before, out


def _tables(m, n, period, seed=0):
    rng = np.random.default_rng(seed)
    xp = rng.random(m) * period  # unsorted, as period= allows
    fp = rng.standard_normal(m)
    x = rng.random(n) * period * 3 - period  # queries outside one period too
    return x, xp, fp


def _spelled_out(x, xp, fp, period):
    """numpy's period branch, one metered op per step."""
    xm = fnp.remainder(x, period)
    xpm = fnp.remainder(xp, period)
    order = fnp.argsort(xpm)
    xs, fs = fnp.take(xpm, order), fnp.take(fp, order)
    xs = fnp.concatenate((xs[-1:] - period, xs, xs[0:1] + period))
    fs = fnp.concatenate((fs[-1:], fs, fs[0:1]))
    return fnp.interp(xm, xs, fs)


def _log2_ceil(k):
    return max(math.ceil(math.log2(k)), 1)


def _expected_flop_cost(n, m):
    branch = 16 * (n + m) + 4 * m * _log2_ceil(m) + 8 * m + 2 * (m + 2) + 2
    return branch + 3 * n + n * _log2_ceil(m + 2)


@pytest.mark.parametrize(("m", "n"), [(1 << 20, 4), (1000, 1), (37, 250)])
def test_period_bills_the_spelled_out_branch(production_weights, m, n):
    x, xp, fp = _tables(m, n, 360.0)
    cost, out = _billed(lambda: fnp.interp(x, xp, fp, period=360.0))
    spelled, ref = _billed(lambda: _spelled_out(x, xp, fp, 360.0))
    assert cost == spelled
    assert np.array_equal(np.asarray(out), np.asarray(ref))
    assert np.array_equal(np.asarray(out), np.interp(x, xp, fp, period=360.0))


def test_issue_repro_production_bill(production_weights):
    x, xp, fp = _tables(1 << 20, 4, 360.0)
    cost, _ = _billed(lambda: fnp.interp(x, xp, fp, period=360.0))
    assert cost == 222_298_444


@pytest.mark.parametrize(("m", "n"), [(16, 4), (5, 9), (1, 3)])
def test_period_flop_cost_formula(m, n):
    x, xp, fp = _tables(m, n, 2.0, seed=1)
    cost, _ = _billed(lambda: fnp.interp(x, xp, fp, period=2.0))
    assert cost == _expected_flop_cost(n, m)


def test_no_period_bill_is_unchanged():
    x, xp, fp = _tables(32, 10, 1.0, seed=2)
    xp = np.sort(xp)
    cost, _ = _billed(lambda: fnp.interp(x, xp, fp))
    assert cost == 3 * 10 + 10 * 5
    explicit_none, _ = _billed(lambda: fnp.interp(x, xp, fp, period=None))
    assert explicit_none == cost


def test_negative_period_bills_like_its_absolute_value(production_weights):
    x, xp, fp = _tables(64, 8, 5.0, seed=3)
    pos, out_pos = _billed(lambda: fnp.interp(x, xp, fp, period=5.0))
    neg, out_neg = _billed(lambda: fnp.interp(x, xp, fp, period=-5.0))
    assert pos == neg
    assert np.array_equal(np.asarray(out_pos), np.asarray(out_neg))


def test_complex_fp_period_runs_and_is_billed():
    x, xp, _ = _tables(64, 8, 5.0, seed=4)
    fp = np.exp(1j * xp)
    cost, out = _billed(lambda: fnp.interp(x, xp, fp, period=5.0))
    plain, _ = _billed(lambda: fnp.interp(x, np.sort(xp), fp))
    assert cost > plain
    assert np.allclose(np.asarray(out), np.interp(x, xp, fp, period=5.0))


def test_tracked_period_is_accepted():
    x, xp, fp = _tables(64, 8, 5.0, seed=5)
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        period = fnp.asarray(5.0)
        before = int(ctx.flops_used)
        out = fnp.interp(x, xp, fp, period=period)
        tracked = int(ctx.flops_used) - before
    plain, _ = _billed(lambda: fnp.interp(x, xp, fp, period=5.0))
    assert tracked == plain
    assert np.array_equal(np.asarray(out), np.interp(x, xp, fp, period=5.0))


@pytest.mark.parametrize(
    ("xp", "fp", "period"),
    [
        (np.arange(4.0), np.arange(4.0), 0.0),
        (np.ones((2, 2)), np.ones((2, 2)), 1.0),
        (np.arange(4.0), np.arange(5.0), 1.0),
    ],
)
def test_period_branch_refusals_charge_nothing(xp, fp, period):
    x = np.linspace(0.0, 1.0, 7)
    with pytest.raises(ValueError) as numpy_error:
        np.interp(x, xp, fp, period=period)
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        with pytest.raises(ValueError) as ours:
            fnp.interp(x, xp, fp, period=period)
        assert ctx.flops_used == 0
    assert str(ours.value) == str(numpy_error.value)
