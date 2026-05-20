"""Tests for the new fma_cost() function in _cost_model.py."""

import pytest
from flopscope._cost_model import fma_cost  # pyright: ignore[reportMissingImports]

from flopscope._config import get_setting, set_setting


def test_fma_cost_function_exists():
    assert callable(fma_cost)


def test_fma_cost_function_returns_default():
    assert fma_cost() == 1


def test_fma_cost_function_returns_set_value():
    original = get_setting("fma_cost")
    try:
        set_setting("fma_cost", 2)
        assert fma_cost() == 2
    finally:
        set_setting("fma_cost", original)


def test_fma_cost_function_returns_int():
    assert isinstance(fma_cost(), int)


def test_existing_FMA_COST_constant_is_gone():
    """After Task 3, the constant FMA_COST is removed in favor of the function.
    Direct imports of FMA_COST should fail.
    """
    # Use getattr through importlib so pyright doesn't flag the missing symbol.
    import importlib

    cost_model = importlib.import_module("flopscope._cost_model")
    with pytest.raises(AttributeError):
        cost_model.FMA_COST  # noqa: B018
