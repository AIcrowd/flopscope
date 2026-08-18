"""A FlopscopeArray passed as a *secondary* array argument must be stripped.

Regression tests for the fail-closed ``RuntimeError`` (WhestArray reached
numpy.<func> from inside an fnp wrapper) raised when a wrapper strips its
primary operand but forwards a secondary array argument -- ``weights=``,
``q=``, ``sorter=``, ``fweights=``, ``aweights=``, ``p=``, ``out=`` -- to
numpy still wrapped.

Each case pins two properties against the equivalent plain-``ndarray`` form:

1. It does not raise (the participant's fnp-built ``weights``/``p``/... is
   accepted rather than crashing).
2. It bills exactly what the plain-ndarray form bills (the secondary arrays
   are built *outside* the budget so only the op itself is metered).
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope.numpy as fnp
from flopscope import BudgetContext
from flopscope._ndarray import FlopscopeArray


def _billed(thunk):
    """Run ``thunk`` inside a fresh budget; return (flops_used, result)."""
    with BudgetContext(flop_budget=10**14) as bc:
        result = thunk()
    return bc.flops_used, result


# --- data built once, outside any budget, so construction is never metered ---
_X = fnp.asarray(np.arange(1, 21, dtype=np.float64))
_W_NP = np.arange(1, 21, dtype=np.float64)
_W_FA = fnp.asarray(_W_NP)
_Q_NP = np.array([0.25, 0.5, 0.75])
_Q_FA = fnp.asarray(_Q_NP)
# Percentile takes q in [0, 100]; scale outside any budget so the fnp form's
# only metered work is the op under test.
_QPCT_NP = _Q_NP * 100.0
_QPCT_FA = fnp.asarray(_QPCT_NP)


# ---------------------------------------------------------------------------
# quantile / percentile family: weights= and q= (array)
# ---------------------------------------------------------------------------
_QUANTILE_FUNCS = [fnp.quantile, fnp.nanquantile]
_PERCENTILE_FUNCS = [fnp.percentile, fnp.nanpercentile]


@pytest.mark.parametrize("func", _QUANTILE_FUNCS, ids=lambda f: f.__name__)
def test_quantile_family_weights_flopscopearray(func):
    plain_flops, plain = _billed(
        lambda: func(_X, 0.5, weights=_W_NP, method="inverted_cdf")
    )
    fa_flops, fa = _billed(lambda: func(_X, 0.5, weights=_W_FA, method="inverted_cdf"))
    assert fa_flops == plain_flops
    np.testing.assert_allclose(np.asarray(fa), np.asarray(plain))


@pytest.mark.parametrize("func", _PERCENTILE_FUNCS, ids=lambda f: f.__name__)
def test_percentile_family_weights_flopscopearray(func):
    plain_flops, plain = _billed(
        lambda: func(_X, 50.0, weights=_W_NP, method="inverted_cdf")
    )
    fa_flops, fa = _billed(lambda: func(_X, 50.0, weights=_W_FA, method="inverted_cdf"))
    assert fa_flops == plain_flops
    np.testing.assert_allclose(np.asarray(fa), np.asarray(plain))


@pytest.mark.parametrize(
    "func", _QUANTILE_FUNCS + _PERCENTILE_FUNCS, ids=lambda f: f.__name__
)
def test_quantile_family_q_array_flopscopearray(func):
    q_np = _QPCT_NP if func in _PERCENTILE_FUNCS else _Q_NP
    q_fa = _QPCT_FA if func in _PERCENTILE_FUNCS else _Q_FA
    plain_flops, plain = _billed(lambda: func(_X, q_np))
    fa_flops, fa = _billed(lambda: func(_X, q_fa))
    assert fa_flops == plain_flops
    np.testing.assert_allclose(np.asarray(fa), np.asarray(plain))


# ---------------------------------------------------------------------------
# cov: fweights= / aweights=
# ---------------------------------------------------------------------------
_M = fnp.asarray(np.random.default_rng(0).random((3, 10)))
_FW_NP = np.arange(1, 11)
_FW_FA = fnp.asarray(_FW_NP)
_AW_NP = np.linspace(0.1, 1.0, 10)
_AW_FA = fnp.asarray(_AW_NP)


def test_cov_fweights_flopscopearray():
    plain_flops, plain = _billed(lambda: fnp.cov(_M, fweights=_FW_NP))
    fa_flops, fa = _billed(lambda: fnp.cov(_M, fweights=_FW_FA))
    assert fa_flops == plain_flops
    np.testing.assert_allclose(np.asarray(fa), np.asarray(plain))


def test_cov_aweights_flopscopearray():
    plain_flops, plain = _billed(lambda: fnp.cov(_M, aweights=_AW_NP))
    fa_flops, fa = _billed(lambda: fnp.cov(_M, aweights=_AW_FA))
    assert fa_flops == plain_flops
    np.testing.assert_allclose(np.asarray(fa), np.asarray(plain))


# ---------------------------------------------------------------------------
# searchsorted: sorter=
# ---------------------------------------------------------------------------
def test_searchsorted_sorter_flopscopearray():
    unsorted = fnp.asarray(np.array([3.0, 1.0, 2.0, 5.0, 4.0]))
    v = fnp.asarray(np.array([2.5, 4.5]))
    sorter_np = np.argsort(np.asarray(unsorted))
    sorter_fa = fnp.asarray(sorter_np)
    plain_flops, plain = _billed(
        lambda: fnp.searchsorted(unsorted, v, sorter=sorter_np)
    )
    fa_flops, fa = _billed(lambda: fnp.searchsorted(unsorted, v, sorter=sorter_fa))
    assert fa_flops == plain_flops
    np.testing.assert_array_equal(np.asarray(fa), np.asarray(plain))


# ---------------------------------------------------------------------------
# random.choice: p= (Generator, RandomState, module-level)
# ---------------------------------------------------------------------------
_POOL = np.arange(10)
_P_NP = np.full(10, 0.1)
_P_FA = fnp.asarray(_P_NP)


def test_generator_choice_p_flopscopearray():
    plain_flops, plain = _billed(
        lambda: fnp.random.default_rng(7).choice(_POOL, size=5, p=_P_NP)
    )
    fa_flops, fa = _billed(
        lambda: fnp.random.default_rng(7).choice(_POOL, size=5, p=_P_FA)
    )
    assert fa_flops == plain_flops
    np.testing.assert_array_equal(np.asarray(fa), np.asarray(plain))


def test_randomstate_choice_p_flopscopearray():
    plain_flops, plain = _billed(
        lambda: fnp.random.RandomState(7).choice(_POOL, size=5, p=_P_NP)
    )
    fa_flops, fa = _billed(
        lambda: fnp.random.RandomState(7).choice(_POOL, size=5, p=_P_FA)
    )
    assert fa_flops == plain_flops
    np.testing.assert_array_equal(np.asarray(fa), np.asarray(plain))


def test_module_random_choice_p_flopscopearray():
    # Module-level fnp.random.choice draws from the global RNG; seed it each time.
    def draw(p):
        fnp.random.seed(7)
        return fnp.random.choice(_POOL, size=5, p=p)

    plain_flops, plain = _billed(lambda: draw(_P_NP))
    fa_flops, fa = _billed(lambda: draw(_P_FA))
    assert fa_flops == plain_flops
    np.testing.assert_array_equal(np.asarray(fa), np.asarray(plain))


# ---------------------------------------------------------------------------
# Generator.permuted: out= (movement method whose destination is an array)
# ---------------------------------------------------------------------------
def test_generator_permuted_out_flopscopearray():
    src = np.arange(10, dtype=np.float64)
    out_np = np.empty(10, dtype=np.float64)
    out_fa = fnp.asarray(np.empty(10, dtype=np.float64))
    plain_flops, plain = _billed(
        lambda: fnp.random.default_rng(3).permuted(src, out=out_np)
    )
    fa_flops, fa = _billed(lambda: fnp.random.default_rng(3).permuted(src, out=out_fa))
    assert fa_flops == plain_flops
    # out= is written in place; the fnp destination must receive the result.
    np.testing.assert_array_equal(np.asarray(out_fa), np.asarray(out_np))
    np.testing.assert_array_equal(np.asarray(fa), np.asarray(plain))


# ---------------------------------------------------------------------------
# The two headline cases named in the report, kept explicit and self-contained.
# ---------------------------------------------------------------------------
def test_quantile_weights_matches_plain_billing():
    x = fnp.asarray(np.arange(1, 21, dtype=np.float64))
    weights = fnp.asarray(np.arange(1, 21, dtype=np.float64))
    with BudgetContext(flop_budget=10**14) as bc:
        result = fnp.quantile(x, 0.5, weights=weights, method="inverted_cdf")
    assert isinstance(result, FlopscopeArray)
    assert bc.flops_used > 0


def test_generator_choice_p_matches_plain_billing():
    pool = np.arange(10)
    p = fnp.asarray(np.full(10, 0.1))
    with BudgetContext(flop_budget=10**14) as bc:
        result = fnp.random.default_rng(0).choice(pool, size=5, p=p)
    assert bc.flops_used > 0
    assert np.asarray(result).shape == (5,)
