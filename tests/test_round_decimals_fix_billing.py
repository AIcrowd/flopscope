"""``round``/``around`` with ``decimals != 0`` and ``fix`` bill their three passes (#332).

For a float input and ``decimals != 0``, numpy's ``round`` runs a multiply by
``10**decimals``, ``rint`` and a divide (the divide first and the multiply
last for negative ``decimals``); ``fix`` runs ``ceil``, ``greater_equal`` and
``floor(where=)``. Each was billed as one pass.
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


def _round_steps(a, decimals):
    """numpy's ``round`` for ``decimals != 0``, pass by pass."""
    scale = 10.0 ** abs(decimals)
    if decimals > 0:
        return fnp.true_divide(fnp.rint(fnp.multiply(a, scale)), scale)
    return fnp.multiply(fnp.rint(fnp.true_divide(a, scale)), scale)


def _fix_steps(a):
    """numpy's ``fix``, line for line."""
    res = fnp.ceil(a)
    return fnp.floor(a, out=res, where=fnp.greater_equal(a, 0))


SPELLINGS = {
    "round": lambda x, d: fnp.round(x, d),
    "around": lambda x, d: fnp.around(x, d),
    "method": lambda x, d: x.round(d),
}


@pytest.fixture
def a():
    return fnp.asarray(np.random.default_rng(0).standard_normal(10000))


@pytest.mark.parametrize("decimals", [2, -1])
@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_issue_repro_round_bills_its_three_passes(a, spelling, decimals):
    billed, got = _billed(lambda: SPELLINGS[spelling](a, decimals))
    twin, ref = _billed(lambda: _round_steps(a, decimals))
    want = np.round(np.asarray(a), decimals)
    assert np.array_equal(np.asarray(got), want)
    assert np.array_equal(np.asarray(ref), want)
    assert billed == twin == 3 * a.size


def test_issue_repro_fix_bills_its_three_passes(a):
    billed, got = _billed(lambda: fnp.fix(a))
    twin, ref = _billed(lambda: _fix_steps(a))
    want = np.fix(np.asarray(a))
    assert np.array_equal(np.asarray(got), want)
    assert np.array_equal(np.asarray(ref), want)
    assert billed == twin == 3 * a.size


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
@pytest.mark.parametrize("decimals", [1, 3, 5, -1, -2])
@pytest.mark.parametrize("op", ["round", "around"])
def test_float_decimals_bill_three_passes(op, decimals, dtype):
    # float16 overflows at 10**5, so numpy's own result is NaN there.
    raw = np.random.default_rng(1).standard_normal((40, 25)) * 100
    x = fnp.asarray(raw.astype(dtype))
    with np.errstate(over="ignore", invalid="ignore"):
        billed, got = _billed(lambda: getattr(fnp, op)(x, decimals))
        want = np.round(np.asarray(x), decimals)
    assert np.array_equal(np.asarray(got), want, equal_nan=True)
    assert billed == 3 * x.size


def test_complex_decimals_bill_three_passes():
    rng = np.random.default_rng(2)
    z = fnp.asarray(rng.standard_normal(500) + 1j * rng.standard_normal(500))
    one, _ = _billed(lambda: fnp.round(z, 0))
    billed, got = _billed(lambda: fnp.round(z, 2))
    assert np.array_equal(np.asarray(got), np.round(np.asarray(z), 2))
    assert billed == 3 * one


def test_integer_negative_decimals_bill_three_passes():
    # true_divide, rint and multiply in float64, then a cast back
    i = fnp.asarray(np.arange(-500, 500, dtype=np.int64) * 7)
    one, _ = _billed(lambda: fnp.round(i, 0))
    billed, got = _billed(lambda: fnp.round(i, -1))
    assert np.array_equal(np.asarray(got), np.round(np.asarray(i), -1))
    assert billed == 3 * one


def test_integer_positive_decimals_keep_one_pass():
    i = fnp.asarray(np.arange(-500, 500, dtype=np.int64) * 7)
    one, _ = _billed(lambda: fnp.round(i, 0))
    billed, got = _billed(lambda: fnp.round(i, 2))
    assert np.array_equal(np.asarray(got), np.round(np.asarray(i), 2))
    assert billed == one


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_fix_bills_three_passes(dtype):
    x = fnp.asarray(
        (np.random.default_rng(3).standard_normal((30, 30)) * 10).astype(dtype)
    )
    billed, got = _billed(lambda: fnp.fix(x))
    twin, _ = _billed(lambda: _fix_steps(x))
    assert np.array_equal(np.asarray(got), np.fix(np.asarray(x)))
    assert billed == twin == 3 * x.size


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_decimals_zero_is_one_rint(a, spelling):
    billed, got = _billed(lambda: SPELLINGS[spelling](a, 0))
    twin, _ = _billed(lambda: fnp.rint(a))
    assert np.array_equal(np.asarray(got), np.round(np.asarray(a), 0))
    assert billed == twin == a.size
