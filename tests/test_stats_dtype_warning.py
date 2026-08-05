"""Warnings for stats inputs promoted to SciPy-compatible float64."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from flopscope._budget import BudgetContext
from flopscope._weights import load_weights
from flopscope.errors import FlopscopeWarning
from flopscope.stats import (
    cauchy,
    expon,
    laplace,
    logistic,
    lognorm,
    norm,
    truncnorm,
    uniform,
)

_DISTRIBUTIONS = (
    ("norm", norm, {}),
    ("uniform", uniform, {}),
    ("expon", expon, {}),
    ("cauchy", cauchy, {}),
    ("logistic", logistic, {}),
    ("laplace", laplace, {}),
    ("lognorm", lognorm, {"s": 1.0}),
    ("truncnorm", truncnorm, {"a": -2, "b": 2}),
)


class _DtypeSensitiveArrayLike:
    """Array-like whose dtype-directed conversion returns distinct values."""

    dtype = np.dtype(np.float32)

    def __init__(self):
        self.array_calls = []

    def __array__(self, dtype=None, copy=None):
        del copy
        resolved_dtype = None if dtype is None else np.dtype(dtype)
        self.array_calls.append(resolved_dtype)
        value = 8.0 if resolved_dtype is None else 0.5
        return np.array([value], dtype=resolved_dtype or self.dtype)


class _ForeignDtype:
    """Sentinel metadata that NumPy cannot normalize as a dtype."""


class _ForeignDtypeArrayLike:
    """Array-like with foreign metadata but valid NumPy array conversion."""

    dtype = _ForeignDtype()

    def __init__(self):
        self.array_calls = []

    def __array__(self, dtype=None, copy=None):
        del copy
        resolved_dtype = None if dtype is None else np.dtype(dtype)
        self.array_calls.append(resolved_dtype)
        value = 8.0 if resolved_dtype is None else 0.5
        return np.array([value], dtype=resolved_dtype or np.float32)


def _promotion_message(op_name: str, dtype_name: str) -> str:
    return (
        f"stats.{op_name} promoted its {dtype_name} input to float64 to match "
        "scipy.stats. If float64 output was not intended, cast the result back "
        f"with result.astype(np.{dtype_name}) before downstream operations; "
        f"float64 operations are billed at twice the {dtype_name} dtype rate."
    )


@pytest.mark.parametrize("method", ("pdf", "cdf", "ppf"))
@pytest.mark.parametrize("distribution_name,distribution,kwargs", _DISTRIBUTIONS)
def test_float32_input_warns_and_returns_float64(
    distribution_name, distribution, kwargs, method
):
    x = np.array([0.5], dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FlopscopeWarning)
        result = getattr(distribution, method)(x, **kwargs)

    assert len(caught) == 1
    assert issubclass(caught[0].category, FlopscopeWarning)
    assert str(caught[0].message) == _promotion_message(
        f"{distribution_name}.{method}", "float32"
    )
    assert result.dtype == np.dtype(np.float64)


def test_float16_warning_names_dtype_and_points_to_call_site():
    x = np.array([0.5], dtype=np.float16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FlopscopeWarning)
        result = norm.ppf(x)

    assert len(caught) == 1
    assert str(caught[0].message) == _promotion_message("norm.ppf", "float16")
    assert caught[0].filename == __file__
    assert result.dtype == np.dtype(np.float64)


def test_dtype_sensitive_array_like_preserves_direct_float64_conversion():
    x = _DtypeSensitiveArrayLike()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FlopscopeWarning)
        result = norm.pdf(x)

    expected = np.exp(-0.5 * 0.5**2) / np.sqrt(2.0 * np.pi)
    np.testing.assert_allclose(np.asarray(result), [expected], rtol=1e-15)
    assert x.array_calls == [None, np.dtype(np.float64)]
    assert len(caught) == 1
    assert str(caught[0].message) == _promotion_message("norm.pdf", "float32")


def test_foreign_dtype_metadata_falls_back_to_array_inference():
    x = _ForeignDtypeArrayLike()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FlopscopeWarning)
        result = norm.pdf(x)

    expected = np.exp(-0.5 * 0.5**2) / np.sqrt(2.0 * np.pi)
    np.testing.assert_allclose(np.asarray(result), [expected], rtol=1e-15)
    assert result.dtype == np.dtype(np.float64)
    assert x.array_calls == [None, None, np.dtype(np.float64)]
    assert len(caught) == 1
    assert str(caught[0].message) == _promotion_message("norm.pdf", "float32")


@pytest.mark.parametrize("dtype", (np.float64, np.longdouble, np.int32, np.bool_))
def test_other_primary_input_dtypes_do_not_warn(dtype):
    x = np.array([0.5], dtype=dtype)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FlopscopeWarning)
        norm.pdf(x)

    assert caught == []


def test_float32_promotion_keeps_float64_production_billing(monkeypatch):
    monkeypatch.delenv("FLOPSCOPE_DISABLE_WEIGHTS", raising=False)
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    load_weights()
    x = np.ones(4, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FlopscopeWarning)
        with BudgetContext(10**9, quiet=True) as budget:
            norm.pdf(x)

    assert budget.flops_used == 216
