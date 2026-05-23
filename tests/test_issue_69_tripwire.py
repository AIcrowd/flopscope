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


def test_top_level_np_ufunc_on_whestarray_warns_and_routes():
    """np.add(whest, whest) at top level should emit UserWarning AND route."""
    a = fnp.asarray(np.random.default_rng(0).random((10,)))
    with BudgetContext(flop_budget=10**14):
        with pytest.warns(UserWarning, match=r"np\.add.*auto-routed to fnp\.add"):
            result = np.add(a, a)
    from flopscope._ndarray import FlopscopeArray

    assert isinstance(result, FlopscopeArray)


def test_inside_wrapper_array_ufunc_leak_raises():
    """An fnp wrapper that calls _np.<ufunc> on a WhestArray must raise."""
    import numpy as _np

    @_counted_wrapper
    def buggy_ufunc_wrapper(a):
        # Forget the strip; pass WhestArray to numpy ufunc.
        return _np.add(a, a)

    a = fnp.asarray(np.random.default_rng(0).random((10,)))
    with BudgetContext(flop_budget=10**14):
        with pytest.raises(
            RuntimeError,
            match=r"WhestArray reached numpy\.add from inside an fnp wrapper",
        ):
            buggy_ufunc_wrapper(a)


def test_passthrough_from_inside_wrapper_does_not_raise():
    """PASSTHROUGH ops (np.shape, np.ndim, etc.) on un-stripped FlopscopeArray
    inside a wrapper should NOT raise the tripwire — they are zero-FLOP queries,
    not real bugs."""

    @_counted_wrapper
    def wrapper_using_passthrough(a):
        # np.shape goes through __array_function__ but is in PASSTHROUGH set
        return np.shape(a)

    a = fnp.asarray(np.zeros((4, 5)))
    with BudgetContext(flop_budget=10**6):
        result = wrapper_using_passthrough(a)
    assert result == (4, 5)


def test_polyval_works_on_flopscopearray_inputs():
    """Regression for issue #69: polyval used to TypeError on FlopscopeArray.

    After fix: fnp.polyval(p_whest, x_whest) produces the correct numeric
    result, charges nonzero FLOPs, does NOT emit auto-route warnings (since
    we're calling fnp.polyval directly, not np.polyval), and does NOT raise
    the in-wrapper tripwire (since we strip before calling _np.polyval).
    """
    p = fnp.asarray([1.0, 2.0, 3.0])
    x = fnp.asarray([1.0, 2.0, 3.0])
    with BudgetContext(flop_budget=10**14) as bc:
        result = fnp.polyval(p, x)
    np.testing.assert_array_equal(np.asarray(result), np.array([6.0, 11.0, 18.0]))
    assert bc.flops_used > 0
