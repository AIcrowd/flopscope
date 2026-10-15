"""Polyfit covariance must pay for its shared matrix work and output scaling."""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._polynomial import polyfit_cost
from flopscope._weights import load_weights


def _bill(y, **kwargs):
    x = np.arange(8, dtype=np.float64)
    with flops.BudgetContext(flop_budget=10**8) as budget:
        result = fnp.polyfit(x, y, 3, **kwargs)
    expected = np.polyfit(x, y, 3, **kwargs)
    if isinstance(result, tuple):
        for got, want in zip(result, expected, strict=True):
            np.testing.assert_allclose(np.asarray(got), want)
    else:
        np.testing.assert_allclose(np.asarray(result), expected)
    return budget.flops_used, result


@pytest.mark.parametrize("cov,increment", [(False, 0), (True, 411), ("unscaled", 410)])
def test_reported_example(cov, increment):
    y = np.array([1.0, 3.0, 2.0, 5.0, 6.0, 4.0, 9.0, 8.0])
    charge, _ = _bill(y, cov=cov)
    assert charge == polyfit_cost(8, 3) + increment


@pytest.mark.parametrize("cov,increment", [(True, 445), ("unscaled", 410)])
def test_multiple_rhs_share_inverse_and_gram(cov, increment):
    y = np.arange(24, dtype=np.float64).reshape(8, 3) ** 2 + np.sin(
        np.arange(24).reshape(8, 3)
    )
    charge, result = _bill(y, cov=cov)
    assert charge == polyfit_cost(8, 3, 3) + increment
    assert result[1].shape == (4, 4, 3 if cov is True else 1)


def test_full_takes_precedence_over_covariance():
    y = np.arange(8, dtype=np.float64) ** 2
    charge, result = _bill(y, full=True, cov=True)
    assert len(result) == 5
    assert charge == polyfit_cost(8, 3)


def test_production_precision_price():
    load_weights()
    y = np.array([1.0, 3.0, 2.0, 5.0, 6.0, 4.0, 9.0, 8.0])
    baseline, _ = _bill(y, cov=False)
    scaled, _ = _bill(y, cov=True)
    unscaled, _ = _bill(y, cov="unscaled")
    assert baseline == 4328
    assert scaled - baseline == 822
    assert unscaled - baseline == 820


def test_invalid_scaled_covariance_does_not_charge():
    x = np.arange(4, dtype=np.float64)
    with flops.BudgetContext(flop_budget=10**8) as budget:
        with pytest.raises(ValueError, match="number of data points"):
            fnp.polyfit(x, x * x, 3, cov=True)
    assert budget.flops_used == 0


@pytest.mark.parametrize("cov", [True, "unscaled"])
def test_increment_matches_counted_covariance_primitives(cov):
    x = np.arange(8, dtype=np.float64)
    y = np.array([1.0, 3.0, 2.0, 5.0, 6.0, 4.0, 9.0, 8.0])
    lhs = np.vander(x, 4)
    scale = np.sqrt((lhs * lhs).sum(axis=0))
    normalized = fnp.asarray(lhs / scale)
    fscale = fnp.asarray(scale)
    residual = fnp.asarray(np.linalg.lstsq(lhs / scale, y, rcond=None)[1])
    with flops.BudgetContext(flop_budget=10**8) as primitives:
        gram = fnp.dot(normalized.T, normalized)
        inverse = fnp.linalg.inv(gram)
        base = inverse / fnp.outer(fscale, fscale)
        factor = residual / 4 if cov is True else 1
        result = base * factor
    total, _ = _bill(y, cov=cov)
    assert total - polyfit_cost(8, 3) == primitives.flops_used
    np.testing.assert_allclose(np.asarray(result), np.polyfit(x, y, 3, cov=cov)[1])


@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.complex128])
def test_covariance_dtype_and_weighted_values_match_numpy(dtype):
    x = np.arange(8, dtype=np.float32)
    y = np.array([1.0, 3.0, 2.0, 5.0, 6.0, 4.0, 9.0, 8.0], dtype=dtype)
    weights = np.linspace(1.0, 2.0, 8)
    with flops.BudgetContext(flop_budget=10**8):
        result = fnp.polyfit(x, y, 3, cov=True, w=weights)
    expected = np.polyfit(x, y, 3, cov=True, w=weights)
    for got, want in zip(result, expected, strict=True):
        assert got.dtype == want.dtype
        np.testing.assert_allclose(np.asarray(got), want)


def test_insufficient_budget_prevents_numpy_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy must not execute before budget approval")

    monkeypatch.setattr(np, "polyfit", forbidden)
    x = np.arange(8, dtype=np.float64)
    with flops.BudgetContext(flop_budget=1):
        with pytest.raises(flops.errors.BudgetExhaustedError):
            fnp.polyfit(x, x * x, 3, cov=True)


def test_negative_degree_covariance_preserves_numpy_error():
    x = np.arange(8, dtype=np.float64)
    with flops.BudgetContext(flop_budget=10**8) as budget:
        with pytest.raises(ValueError, match="expected deg >= 0"):
            fnp.polyfit(x, x * x, -1, cov=True)
    assert budget.flops_used == 0
