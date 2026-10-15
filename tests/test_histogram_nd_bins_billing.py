"""Equivalent histogram bin spellings must not select a cheap fallback."""

import numpy as np
import pytest

import flopscope.numpy as fnp
from flopscope import BudgetContext


@pytest.mark.parametrize(
    "bins", [8, np.int64(8), np.array(8), [8, 8], np.array([8, 8])]
)
@pytest.mark.parametrize("name", ["histogram2d", "histogramdd"])
def test_counts_have_equivalent_price(name, bins):
    sample = np.arange(40, dtype=np.float64).reshape(20, 2)
    args = (sample[:, 0], sample[:, 1]) if name == "histogram2d" else (sample,)
    with BudgetContext(flop_budget=10**7) as budget:
        result = getattr(fnp, name)(*args, bins=bins)
    expected = getattr(np, name)(*args, bins=bins)
    np.testing.assert_array_equal(np.asarray(result[0]), expected[0])
    assert budget.flops_used == 20 * (3 + 3)


@pytest.mark.parametrize(
    "bins", [[8, np.arange(17)], [np.arange(17), np.int64(8)], np.arange(17)]
)
def test_histogram2d_mixed_and_shared_edges(bins):
    x = np.arange(20, dtype=np.float64)
    with BudgetContext(flop_budget=10**7) as budget:
        result = fnp.histogram2d(x, x, bins=bins)
    np.testing.assert_array_equal(
        np.asarray(result[0]), np.histogram2d(x, x, bins=bins)[0]
    )
    assert budget.flops_used == 20 * (8 if isinstance(bins, list) else 10)


@pytest.mark.parametrize(
    "bins", [[np.int64(8), np.arange(17)], np.array([[0, 1, 2], [0, 1, 2]])]
)
def test_histogramdd_mixed_and_array_edges(bins):
    sample = np.arange(40, dtype=np.float64).reshape(20, 2)
    with BudgetContext(flop_budget=10**7) as budget:
        result = fnp.histogramdd(sample, bins=bins)
    np.testing.assert_array_equal(
        np.asarray(result[0]), np.histogramdd(sample, bins=bins)[0]
    )
    assert budget.flops_used == 20 * (8 if isinstance(bins, list) else 4)


def test_production_bin_spellings_have_equal_price():
    from flopscope._weights import load_weights

    load_weights()
    sample = np.arange(40, dtype=np.float64).reshape(20, 2)
    charges = []
    for bins in (8, np.int64(8), np.array(8), np.array([8, 8])):
        with BudgetContext(flop_budget=10**7) as budget:
            fnp.histogramdd(sample, bins=bins)
        charges.append(budget.flops_used)
    assert charges == [960] * 4


@pytest.mark.parametrize("name", ["histogram2d", "histogramdd"])
def test_insufficient_budget_prevents_numpy_execution(name, monkeypatch):
    from flopscope.errors import BudgetExhaustedError

    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy must not execute before budget approval")

    monkeypatch.setattr(np, name, forbidden)
    sample = np.arange(40, dtype=np.float64).reshape(20, 2)
    args = (sample[:, 0], sample[:, 1]) if name == "histogram2d" else (sample,)
    with BudgetContext(flop_budget=1):
        with pytest.raises(BudgetExhaustedError):
            getattr(fnp, name)(*args, bins=np.array([8, 8]))
