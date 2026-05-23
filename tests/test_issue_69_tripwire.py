"""Boundary tripwire tests for issue #69."""

from __future__ import annotations

import numpy as np
import pytest

import flopscope.numpy as fnp
from flopscope._budget import BudgetContext, _called_from_wrapper, _counted_wrapper


def test_called_from_wrapper_false_at_module_top_level():
    """At module-top-level call site, marker is not on the stack."""
    assert _called_from_wrapper() is False


def test_called_from_wrapper_true_inside_counted_wrapper():
    """Inside a _counted_wrapper-decorated function, marker IS on the stack."""

    @_counted_wrapper
    def probe():
        return _called_from_wrapper()

    with BudgetContext(flop_budget=10**6):
        assert probe() is True


def test_called_from_wrapper_true_under_nested_calls():
    """Marker is detected even when wrapper is several frames deep."""

    @_counted_wrapper
    def outer():
        def helper_1():
            def helper_2():
                return _called_from_wrapper()
            return helper_2()
        return helper_1()

    with BudgetContext(flop_budget=10**6):
        assert outer() is True
