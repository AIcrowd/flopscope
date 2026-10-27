"""Tail and narrow-interval regression tests through the counted public API."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, special, stats

from flopscope._budget import BudgetContext
from flopscope._ndarray import FlopscopeArray
from flopscope._weights import load_weights
from flopscope.errors import BudgetExhaustedError, UnsupportedDtypeError
from flopscope.stats import _truncnorm_kernels as kernels
from flopscope.stats import truncnorm


@pytest.mark.parametrize(
    "a,b",
    [
        (9, 10),
        (40, 41),
        (-10, -9),
        (-41, -40),
        (9, np.inf),
        (-np.inf, -40),
        (-np.inf, np.inf),
        (-2, 3),
        (0, np.inf),
    ],
)
def test_tail_accuracy_and_inverse(a, b):
    q = np.unique(
        np.r_[
            0,
            1e-300,
            np.geomspace(1e-12, 0.1, 20),
            np.linspace(0.1, 0.9, 41),
            1 - 1e-12,
            1,
        ]
    )
    quantiles = np.asarray(truncnorm.ppf(q, a, b))
    expected = stats.truncnorm.ppf(q, a, b)
    np.testing.assert_allclose(quantiles, expected, rtol=3e-14, atol=4e-14)
    assert np.all(np.diff(quantiles) >= 0)
    assert np.all((quantiles >= a) & (quantiles <= b))
    assert quantiles[0] == a and quantiles[-1] == b
    actual_cdf = np.asarray(truncnorm.cdf(quantiles, a, b))
    np.testing.assert_allclose(actual_cdf, q, atol=5e-13, rtol=5e-13)
    np.testing.assert_allclose(
        actual_cdf, stats.truncnorm.cdf(quantiles, a, b), atol=5e-13, rtol=5e-13
    )
    np.testing.assert_allclose(
        np.asarray(truncnorm.pdf(quantiles, a, b)),
        stats.truncnorm.pdf(quantiles, a, b),
        atol=1e-13,
        rtol=5e-13,
    )


@pytest.mark.parametrize(
    "a,width", [(0.0, 1e-12), (1.0, 1e-12), (40.0, 1e-10), (-40.0, 1e-10)]
)
def test_narrow_interval_independent_integral(a, width):
    # Subtracting SciPy CDFs also loses precision here. Integrate the factored
    # density over a unit interval with adaptive Gauss-Kronrod as an oracle.
    b = a + width
    width = b - a

    def integral(u):
        return integrate.quad(
            lambda v: np.exp(-a * width * v - 0.5 * (width * v) ** 2),
            0,
            u,
            epsabs=2e-14,
            epsrel=2e-14,
        )[0]

    denominator = integral(1.0)
    x = np.linspace(a, b, 17)
    offset = x - a
    expected_pdf = np.exp(-a * offset - 0.5 * offset**2) / (width * denominator)
    expected_cdf = np.array([integral(t / width) / denominator for t in offset])
    np.testing.assert_allclose(
        np.asarray(truncnorm.pdf(x, a, b)), expected_pdf, rtol=2e-14
    )
    np.testing.assert_allclose(
        np.asarray(truncnorm.cdf(x, a, b)), expected_cdf, atol=2e-14
    )
    q = np.linspace(0, 1, 41)
    quantiles = np.asarray(truncnorm.ppf(q, a, b))
    assert np.all(np.diff(quantiles) >= 0)
    assert np.all((quantiles >= a) & (quantiles <= b))
    # Quantile rounding cannot be more accurate than one representable x step.
    error = np.abs(np.asarray(truncnorm.cdf(quantiles, a, b)) - q)
    resolution = np.abs(np.spacing(quantiles)) * np.asarray(
        truncnorm.pdf(quantiles, a, b)
    )
    assert np.all(error <= resolution + 2e-14)


def test_adjacent_bounds_and_empty_input():
    a, b = 1.0, np.nextafter(1.0, np.inf)
    values = np.asarray(truncnorm.ppf([0.0, 0.25, 0.5, 0.75, 1.0], a, b))
    assert np.all((values == a) | (values == b))
    np.testing.assert_array_equal(np.asarray(truncnorm.cdf([a, b], a, b)), [0.0, 1.0])
    for method in (truncnorm.pdf, truncnorm.cdf, truncnorm.ppf):
        result = method(np.empty((0, 2)), 40.0, 41.0)
        assert result.shape == (0, 2)
        assert result.dtype == np.float64


def test_log_tail_helpers_against_scipy():
    x = np.r_[np.linspace(-100, 100, 1001), -np.inf, np.inf, np.nan]
    np.testing.assert_allclose(
        kernels.log_ndtr(x), special.log_ndtr(x), rtol=3e-14, atol=3e-14
    )
    assert np.isfinite(kernels.log_mass(40, 41))
    np.testing.assert_allclose(
        kernels.log_mass(40, 41), -804.6084420137538, rtol=0, atol=2e-13
    )


def test_seeded_mixed_intervals():
    rng = np.random.default_rng(2901)
    a = rng.uniform(-50, 50, 256)
    b = a + np.exp(rng.uniform(np.log(0.003), np.log(10), a.size))
    q = rng.uniform(1e-6, 1 - 1e-6, a.size)
    x = np.asarray(truncnorm.ppf(q, a, b))
    np.testing.assert_allclose(x, stats.truncnorm.ppf(q, a, b), rtol=3e-14, atol=5e-14)
    assert np.all((x >= a) & (x <= b))
    np.testing.assert_allclose(np.asarray(truncnorm.cdf(x, a, b)), q, atol=5e-12)


@pytest.mark.parametrize("method", ["pdf", "cdf", "ppf"])
def test_invalid_domains_are_nan(method):
    values = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, np.nan, 0.5, 0.5, 0.5])
    a = np.array([1, 2, np.nan, -2, -2, -2, -2, -2, np.inf, -np.inf])
    b = np.array([1, 1, 2, 2, 2, 2, 2, 2, np.inf, -np.inf])
    scale = np.array([1, 1, 1, 0, -1, np.nan, 1, 1, 1, 1])
    loc = np.array([0, 0, 0, 0, 0, 0, 0, np.nan, 0, 0])
    with np.errstate(all="ignore"):
        actual = np.asarray(
            getattr(truncnorm, method)(values, a, b, loc=loc, scale=scale)
        )
        expected = getattr(stats.truncnorm, method)(values, a, b, loc=loc, scale=scale)
    assert np.all(np.isnan(actual))
    np.testing.assert_equal(actual, expected)


def test_invalid_quantiles_and_exact_endpoints():
    q = [-np.inf, -0.1, 0.0, 1.0, 1.1, np.inf, np.nan]
    result = np.asarray(truncnorm.ppf(q, 9, 10, loc=3, scale=2))
    np.testing.assert_equal(
        result, [np.nan, np.nan, 21.0, 23.0, np.nan, np.nan, np.nan]
    )


@pytest.mark.parametrize("method", ["pdf", "cdf", "ppf"])
def test_broadcast_and_scalar_contract(method):
    fn = getattr(truncnorm, method)
    oracle = getattr(stats.truncnorm, method)
    x = 0.25 if method == "ppf" else 42.0
    a = np.array([[9.0], [40.0]])
    b = a + 1
    loc = np.array([0.0, 1.0, 2.0])
    scale = np.array([1.0, 1.25, 2.0])
    with BudgetContext(10**9, quiet=True) as scalar_budget:
        scalar = fn(x, 40.0, 41.0)
    assert isinstance(scalar, FlopscopeArray)
    assert scalar.shape == () and scalar.dtype == np.float64
    with BudgetContext(10**9, quiet=True) as array_budget:
        result = fn(x, a, b, loc=loc, scale=scale)
    assert result.shape == (2, 3) and result.dtype == np.float64
    assert array_budget.flops_used == 6 * scalar_budget.flops_used
    np.testing.assert_allclose(
        np.asarray(result),
        oracle(x, a, b, loc=loc, scale=scale),
        rtol=5e-13,
        atol=5e-13,
    )


@pytest.mark.parametrize("method", ["pdf", "cdf", "ppf"])
def test_budget_refusal_precedes_numerical_compute(method, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("kernel ran before budget refusal")

    monkeypatch.setattr(kernels, method, unexpected)
    with BudgetContext(1, quiet=True):
        with pytest.raises(BudgetExhaustedError):
            getattr(truncnorm, method)(0.5, 40, 41)


@pytest.mark.parametrize("method", ["pdf", "cdf", "ppf"])
def test_incompatible_shapes_are_billed_and_raise(method):
    fn = getattr(truncnorm, method)
    with BudgetContext(10**9, quiet=True) as scalar_budget:
        fn(0.5, 40, 41)
    with BudgetContext(10**9, quiet=True) as failed_budget:
        with pytest.raises(ValueError, match="broadcast"):
            fn(np.ones(3), np.ones(4), 41)
    assert failed_budget.flops_used == 3 * scalar_budget.flops_used


@pytest.mark.parametrize("method", ["pdf", "cdf", "ppf"])
@pytest.mark.parametrize("position", ["x", "a", "b", "loc", "scale"])
def test_object_source_refused_before_conversion(method, position):
    class Uncastable:
        def __float__(self):
            raise AssertionError("object source converted before dtype refusal")

    args: dict[str, object] = {"x": 0.5, "a": 40.0, "b": 41.0, "loc": 0.0, "scale": 1.0}
    args[position] = np.array([Uncastable()], dtype=object)
    x = args.pop("x")
    with BudgetContext(10**9, quiet=True) as budget:
        with pytest.raises(UnsupportedDtypeError):
            getattr(truncnorm, method)(x, **args)
    assert budget.flops_used == 0


@pytest.mark.parametrize("method,cost", [("pdf", 315), ("cdf", 844), ("ppf", 1392)])
@pytest.mark.parametrize("packaged_weights", [False, True])
def test_fixed_cost_for_broadcast_and_empty(
    method, cost, packaged_weights, monkeypatch
):
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    if packaged_weights:
        monkeypatch.delenv("FLOPSCOPE_DISABLE_WEIGHTS", raising=False)
        load_weights()
    else:
        monkeypatch.setenv("FLOPSCOPE_DISABLE_WEIGHTS", "1")
    multiplier = 2 if packaged_weights else 1
    fn = getattr(truncnorm, method)
    with BudgetContext(10**9, quiet=True) as budget:
        fn(0.5, np.full((2, 3), 40.0), 41.0)
    assert budget.flops_used == cost * 6 * multiplier
    with BudgetContext(10**9, quiet=True) as budget:
        fn(np.empty(0), 40.0, 41.0)
    assert budget.flops_used == cost * multiplier
