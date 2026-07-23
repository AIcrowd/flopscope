"""Tests for global default BudgetContext auto-activation."""

import os
from unittest.mock import patch

import pytest

from flopscope._budget import (
    BudgetContext,
    _reset_global_default,
    get_active_budget,
)
from flopscope._validation import require_budget


@pytest.fixture(autouse=True)
def clean_budget_state():
    """Ensure no budget context leaks between tests."""
    _reset_global_default()
    yield
    _reset_global_default()


def test_require_budget_auto_activates_global():
    assert get_active_budget() is None
    budget = require_budget()
    assert budget is not None
    assert budget.flop_budget == int(1e15)


def test_global_default_is_singleton():
    b1 = require_budget()
    b2 = require_budget()
    assert b1 is b2


def test_explicit_context_overrides_global():
    global_ctx = require_budget()
    with BudgetContext(flop_budget=500) as explicit:
        assert get_active_budget() is explicit
        assert get_active_budget() is not global_ctx
    # Global default resumes after explicit context exits
    assert get_active_budget() is global_ctx


def test_global_default_env_var():
    _reset_global_default()
    with patch.dict(os.environ, {"FLOPSCOPE_DEFAULT_BUDGET": "1e9"}):
        budget = require_budget()
        assert budget.flop_budget == int(1e9)


def test_global_default_tracks_flops():
    budget = require_budget()
    budget.deduct("add", flop_cost=100, subscripts=None, shapes=(), dtypes=())
    assert budget.flops_used == 100
    same = require_budget()
    assert same.flops_used == 100
