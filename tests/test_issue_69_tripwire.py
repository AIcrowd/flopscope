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


def test_top_level_np_func_on_whestarray_warns_and_routes():
    """np.diff(whest) at top level should emit UserWarning AND still auto-route to fnp.diff."""
    a = fnp.asarray(np.random.default_rng(0).random((10,)))
    with BudgetContext(flop_budget=10**14):
        with pytest.warns(UserWarning, match=r"np\.diff.*auto-routed to fnp\.diff"):
            result = np.diff(a)
    # result should have correct shape (auto-routing called fnp.diff successfully)
    assert result.shape == (9,)


def test_inside_wrapper_array_function_leak_raises():
    """An fnp wrapper that forgets to strip and calls _np.<func> on a WhestArray must raise."""
    import numpy as _np

    @_counted_wrapper
    def buggy_wrapper(a):
        # Intentionally forget _to_base_ndarray: pass WhestArray straight to numpy.
        # numpy.diff goes through __array_function__ which detects we're inside
        # an fnp wrapper (depth>0) and raises.
        return _np.diff(a)

    a = fnp.asarray(np.random.default_rng(0).random((10,)))
    with BudgetContext(flop_budget=10**14):
        with pytest.raises(
            RuntimeError,
            match=r"WhestArray reached numpy\.diff from inside an fnp wrapper",
        ):
            buggy_wrapper(a)
