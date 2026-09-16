"""Tests for distributions derived from the normal distribution."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from flopscope._budget import BudgetContext

# ============================================================
# lognorm
# ============================================================


class TestLognormPdf:
    @pytest.mark.parametrize("s,loc,scale", [(1, 0, 1), (0.5, 0, 1), (1, 1, 2)])
    def test_accuracy(self, s, loc, scale):
        from flopscope.stats import lognorm

        x = np.linspace(loc + 0.01, loc + 10 * scale, 500)
        result = np.asarray(lognorm.pdf(x, s, loc=loc, scale=scale))
        expected = sp_stats.lognorm.pdf(x, s, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-13, rtol=1e-13)


class TestLognormCdf:
    @pytest.mark.parametrize("s,loc,scale", [(1, 0, 1), (0.5, 0, 1), (1, 1, 2)])
    def test_accuracy(self, s, loc, scale):
        from flopscope.stats import lognorm

        x = np.linspace(loc + 0.01, loc + 10 * scale, 500)
        result = np.asarray(lognorm.cdf(x, s, loc=loc, scale=scale))
        expected = sp_stats.lognorm.cdf(x, s, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-13, rtol=1e-13)


class TestLognormPpf:
    @pytest.mark.parametrize("s,loc,scale", [(1, 0, 1), (0.5, 0, 1), (1, 1, 2)])
    def test_accuracy(self, s, loc, scale):
        from flopscope.stats import lognorm

        q = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
        result = np.asarray(lognorm.ppf(q, s, loc=loc, scale=scale))
        expected = sp_stats.lognorm.ppf(q, s, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-11, rtol=1e-11)

    def test_roundtrip(self):
        from flopscope.stats import lognorm

        s = 0.5
        q = np.linspace(0.01, 0.99, 50)
        roundtrip = np.asarray(lognorm.cdf(np.asarray(lognorm.ppf(q, s)), s))
        np.testing.assert_allclose(roundtrip, q, atol=1e-11, rtol=1e-11)

    def test_flop_cost(self):
        from flopscope.stats import lognorm

        q = np.random.rand(60)
        with BudgetContext(flop_budget=10**6) as b:
            lognorm.ppf(q, 1.0)
            assert b.flops_used == 106 * 60


# ============================================================
# truncnorm
# ============================================================


class TestTruncnormPdf:
    @pytest.mark.parametrize(
        "a,b,loc,scale", [(-2, 2, 0, 1), (-1, 3, 1, 2), (0, 5, 0, 1)]
    )
    def test_accuracy(self, a, b, loc, scale):
        from flopscope.stats import truncnorm

        x = np.linspace(a * scale + loc - 0.5, b * scale + loc + 0.5, 500)
        result = np.asarray(truncnorm.pdf(x, a, b, loc=loc, scale=scale))
        expected = sp_stats.truncnorm.pdf(x, a, b, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-13, rtol=1e-13)


class TestTruncnormCdf:
    @pytest.mark.parametrize(
        "a,b,loc,scale", [(-2, 2, 0, 1), (-1, 3, 1, 2), (0, 5, 0, 1)]
    )
    def test_accuracy(self, a, b, loc, scale):
        from flopscope.stats import truncnorm

        x = np.linspace(a * scale + loc - 0.5, b * scale + loc + 0.5, 500)
        result = np.asarray(truncnorm.cdf(x, a, b, loc=loc, scale=scale))
        expected = sp_stats.truncnorm.cdf(x, a, b, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-13, rtol=1e-13)


class TestTruncnormPpf:
    @pytest.mark.parametrize(
        "a,b,loc,scale", [(-2, 2, 0, 1), (-1, 3, 1, 2), (0, 5, 0, 1)]
    )
    def test_accuracy(self, a, b, loc, scale):
        from flopscope.stats import truncnorm

        q = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
        result = np.asarray(truncnorm.ppf(q, a, b, loc=loc, scale=scale))
        expected = sp_stats.truncnorm.ppf(q, a, b, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-11, rtol=1e-11)

    def test_roundtrip(self):
        from flopscope.stats import truncnorm

        a, b = -2, 2
        q = np.linspace(0.01, 0.99, 50)
        roundtrip = np.asarray(truncnorm.cdf(np.asarray(truncnorm.ppf(q, a, b)), a, b))
        np.testing.assert_allclose(roundtrip, q, atol=1e-11, rtol=1e-11)

    def test_flop_cost(self):
        from flopscope.stats import truncnorm

        x = np.random.randn(40)
        with BudgetContext(flop_budget=10**6) as b:
            truncnorm.cdf(x, -2, 2)
            # Stable tail kernel's analytical bound, stats weight 1.0.
            assert b.flops_used == 844 * 40
