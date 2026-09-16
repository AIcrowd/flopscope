"""NaN propagation must not depend on recycled allocation contents."""

import numpy as np
import pytest
from scipy import special, stats

from flopscope import BudgetContext
from flopscope.stats import lognorm, norm, truncnorm
from flopscope.stats._erf import _erf
from flopscope.stats._ndtri import _ndtri


@pytest.fixture
def finite_empty_buffers(monkeypatch):
    """Make uninitialized outputs deterministic rather than allocator-dependent.

    Filling empty buffers with a finite sentinel is valid: empty_like makes
    no promise about its contents. This avoids false passes when recycled
    buffers happen to contain NaN already.
    """
    original = np.empty_like

    def empty_with_sentinel(*args, **kwargs):
        result = original(*args, **kwargs)
        result.fill(0.125)
        return result

    monkeypatch.setattr(np, "empty_like", empty_with_sentinel)


@pytest.mark.parametrize(
    "actual,expected,values",
    [
        (_erf, special.erf, [-np.inf, -2, 0, np.nan, 1, np.inf]),
        (_ndtri, special.ndtri, [-0.1, 0, 0.25, np.nan, 0.75, 1, 1.1]),
        (norm.cdf, stats.norm.cdf, [-np.inf, -2, 0, np.nan, 1, np.inf]),
        (norm.ppf, stats.norm.ppf, [-0.1, 0, 0.25, np.nan, 0.75, 1, 1.1]),
        (
            lambda x: lognorm.ppf(x, 0.5),
            lambda x: stats.lognorm.ppf(x, 0.5),
            [0, 0.25, np.nan, 0.75, 1],
        ),
        (
            lambda x: truncnorm.cdf(x, -2, 2),
            lambda x: stats.truncnorm.cdf(x, -2, 2),
            [-3, -1, np.nan, 1, 3],
        ),
        (
            lambda x: truncnorm.ppf(x, -2, 2),
            lambda x: stats.truncnorm.ppf(x, -2, 2),
            [0, 0.25, np.nan, 0.75, 1],
        ),
    ],
)
def test_nan_propagates_with_mixed_valid_values(
    finite_empty_buffers, actual, expected, values
):
    values = np.asarray(values, dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(actual(values)),
        expected(values),
        rtol=1e-11,
        atol=1e-11,
        equal_nan=True,
    )


@pytest.mark.parametrize("fn", [_erf, _ndtri, norm.cdf, norm.ppf])
def test_scalar_nan_propagates(finite_empty_buffers, fn):
    assert np.isnan(np.asarray(fn(np.nan)))


@pytest.mark.parametrize("fn", [norm.cdf, norm.ppf])
def test_nan_input_keeps_same_shape_flop_cost(fn):
    with BudgetContext(flop_budget=10**6) as finite_budget:
        fn(np.array([0.25, 0.5, 0.75]))
    with BudgetContext(flop_budget=10**6) as nan_budget:
        fn(np.array([0.25, np.nan, 0.75]))
    assert nan_budget.flops_used == finite_budget.flops_used > 0
