"""Tests for BudgetContext decorator form."""

import pytest

from flopscope._budget import BudgetContext, budget, get_active_budget


def test_budget_context_as_decorator():
    @BudgetContext(flop_budget=1000, namespace="train")
    def my_func():
        budget_ctx = get_active_budget()
        assert budget_ctx is not None
        assert budget_ctx.namespace == "train"
        budget_ctx.deduct("add", flop_cost=10, subscripts=None, shapes=(), dtypes=())
        return 42

    result = my_func()
    assert result == 42
    assert get_active_budget() is None


def test_decorator_cleans_up_on_exception():
    @BudgetContext(flop_budget=1000)
    def bad_func():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        bad_func()
    assert get_active_budget() is None


def test_decorator_passes_args_and_kwargs():
    @BudgetContext(flop_budget=1000, namespace="test")
    def add_values(a, b, offset=0):
        return a + b + offset

    assert add_values(1, 2, offset=10) == 13


def test_decorator_preserves_function_name():
    @BudgetContext(flop_budget=1000)
    def my_special_func():
        pass

    assert my_special_func.__name__ == "my_special_func"
    assert my_special_func.__qualname__.endswith("my_special_func")


def test_budget_function_returns_budget_context():
    ctx = budget(flop_budget=5000, namespace="test")
    assert isinstance(ctx, BudgetContext)
    assert ctx.flop_budget == 5000
    assert ctx.namespace == "test"


def test_budget_function_as_context_manager():
    with budget(flop_budget=5000, namespace="ctx") as ctx:
        assert get_active_budget() is ctx
        assert ctx.namespace == "ctx"
    assert get_active_budget() is None


def test_budget_function_as_decorator():
    @budget(flop_budget=5000, namespace="dec")
    def compute():
        b = get_active_budget()
        assert b is not None
        assert b.namespace == "dec"
        return 99

    assert compute() == 99
