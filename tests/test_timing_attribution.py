"""Timing-bucket attribution tests (issue: callback/data-movement misattribution)."""

import cProfile
import gc
import sys
import threading
import time
import weakref

import numpy as np
import pytest

import flopscope as flops
import flopscope._budget as budget_module
import flopscope.numpy as fnp
from flopscope._budget import (
    _call_numpy,
    _call_numpy_with_python_callbacks,
    _call_user_code,
    _counted_wrapper,
    get_active_budget,
)
from flopscope._config import get_setting


def test_ordinary_backend_call_reset_keeps_only_post_reset_overlap(monkeypatch):
    logical_clock = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: logical_clock[0])

    def backend():
        logical_clock[0] = 2.0
        flops.budget_reset()
        logical_clock[0] = 5.0
        return "ok"

    with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
        with budget.deduct(
            "ordinary", flop_cost=1, subscripts=None, shapes=(), dtypes=()
        ):
            assert _call_numpy(backend) == "ok"

    summary = flops.budget_summary_dict()
    assert summary["wall_time_s"] == 3.0
    assert summary["flopscope_backend_time_s"] == 3.0
    assert summary["flopscope_overhead_time_s"] == 0.0
    assert summary["residual_wall_time_s"] == 0.0
    assert summary["operations"]["ordinary"]["flopscope_backend_time_s"] == 3.0
    assert budget._live_backend_calls == set()


def test_callback_tracked_call_reset_rebases_active_callback_root(monkeypatch):
    logical_clock = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: logical_clock[0])

    def callback_backend():
        logical_clock[0] = 2.0
        flops.budget_reset()
        logical_clock[0] = 5.0
        return "ok"

    with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
        with budget.deduct(
            "callback", flop_cost=1, subscripts=None, shapes=(), dtypes=()
        ):
            assert _call_numpy_with_python_callbacks(callback_backend) == "ok"

    summary = flops.budget_summary_dict()
    assert budget._total_user_code_time == 3.0
    assert summary["wall_time_s"] == 3.0
    assert summary["flopscope_backend_time_s"] == 0.0
    assert summary["flopscope_overhead_time_s"] == 0.0
    assert summary["residual_wall_time_s"] == 3.0
    assert budget._live_backend_calls == set()


def test_non_callable_profiler_fallback_reset_keeps_post_reset_overlap(monkeypatch):
    logical_clock = [0.0]
    profiler = object()
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: logical_clock[0])
    monkeypatch.setattr(budget_module._sys, "getprofile", lambda: profiler)
    monkeypatch.setattr(budget_module._sys, "setprofile", lambda _profile: None)

    def backend():
        logical_clock[0] = 2.0
        flops.budget_reset()
        logical_clock[0] = 5.0
        return "ok"

    with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
        with budget.deduct(
            "fallback", flop_cost=1, subscripts=None, shapes=(), dtypes=()
        ):
            assert _call_numpy_with_python_callbacks(backend) == "ok"

    summary = flops.budget_summary_dict()
    assert summary["wall_time_s"] == 3.0
    assert summary["flopscope_backend_time_s"] == 3.0
    assert summary["flopscope_overhead_time_s"] == 0.0
    assert summary["residual_wall_time_s"] == 0.0
    assert budget._live_backend_calls == set()


def test_backend_call_multiple_resets_uses_latest_epoch(monkeypatch):
    logical_clock = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: logical_clock[0])

    def backend():
        logical_clock[0] = 1.0
        flops.budget_reset()
        logical_clock[0] = 3.0
        flops.budget_reset()
        logical_clock[0] = 7.0

    with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
        with budget.deduct(
            "ordinary", flop_cost=1, subscripts=None, shapes=(), dtypes=()
        ):
            _call_numpy(backend)

    summary = flops.budget_summary_dict()
    assert summary["wall_time_s"] == 4.0
    assert summary["flopscope_backend_time_s"] == 4.0
    assert summary["flopscope_overhead_time_s"] == 0.0
    assert summary["residual_wall_time_s"] == 0.0
    assert budget._live_backend_calls == set()


def test_backend_registration_setup_is_overhead_not_backend(monkeypatch):
    logical_clock = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: logical_clock[0])

    class ClockAdvancingSet(set):
        def add(self, value):
            logical_clock[0] = 2.0
            super().add(value)

    def backend():
        logical_clock[0] = 5.0
        return "ok"

    with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
        budget._live_backend_calls = ClockAdvancingSet()
        with budget.deduct(
            "ordinary", flop_cost=1, subscripts=None, shapes=(), dtypes=()
        ):
            assert _call_numpy(backend) == "ok"

    summary = flops.budget_summary_dict()
    assert summary["wall_time_s"] == 5.0
    assert summary["flopscope_backend_time_s"] == 3.0
    assert summary["flopscope_overhead_time_s"] == 2.0
    assert summary["residual_wall_time_s"] == 0.0
    assert summary["operations"]["ordinary"]["flopscope_backend_time_s"] == 3.0
    assert summary["operations"]["ordinary"]["flopscope_overhead_time_s"] == 2.0
    assert budget._live_backend_calls == set()


def test_nested_ordinary_backend_calls_count_overlap_once(monkeypatch):
    logical_clock = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: logical_clock[0])

    def inner_backend():
        logical_clock[0] = 3.0
        return "inner"

    def outer_backend():
        logical_clock[0] = 1.0
        assert _call_numpy(inner_backend) == "inner"
        logical_clock[0] = 5.0
        return "outer"

    with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
        with budget.deduct(
            "ordinary", flop_cost=1, subscripts=None, shapes=(), dtypes=()
        ):
            assert _call_numpy(outer_backend) == "outer"

    summary = flops.budget_summary_dict()
    assert summary["wall_time_s"] == 5.0
    assert summary["flopscope_backend_time_s"] == 5.0
    assert summary["flopscope_overhead_time_s"] == 0.0
    assert summary["residual_wall_time_s"] == 0.0
    assert summary["operations"]["ordinary"]["flopscope_backend_time_s"] == 5.0
    assert budget._live_backend_calls == set()


def test_nested_ordinary_backend_calls_reset_count_post_reset_overlap_once(
    monkeypatch,
):
    logical_clock = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: logical_clock[0])

    def inner_backend():
        logical_clock[0] = 2.0
        flops.budget_reset()
        logical_clock[0] = 4.0
        return "inner"

    def outer_backend():
        logical_clock[0] = 1.0
        assert _call_numpy(inner_backend) == "inner"
        logical_clock[0] = 5.0
        return "outer"

    with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
        with budget.deduct(
            "ordinary", flop_cost=1, subscripts=None, shapes=(), dtypes=()
        ):
            assert _call_numpy(outer_backend) == "outer"

    summary = flops.budget_summary_dict()
    assert summary["wall_time_s"] == 3.0
    assert summary["flopscope_backend_time_s"] == 3.0
    assert summary["flopscope_overhead_time_s"] == 0.0
    assert summary["residual_wall_time_s"] == 0.0
    assert summary["operations"]["ordinary"]["flopscope_backend_time_s"] == 3.0
    assert budget._live_backend_calls == set()


def test_reset_after_backend_return_before_commit_drops_the_finished_call(
    monkeypatch,
):
    logical_clock = [0.0]
    backend_returned = threading.Event()
    end_sample_entered = threading.Event()
    reset_finished = threading.Event()
    end_sample_taken = [False]
    main_thread_id = threading.get_ident()

    def perf_counter():
        if (
            threading.get_ident() == main_thread_id
            and backend_returned.is_set()
            and not end_sample_taken[0]
        ):
            end_sample_taken[0] = True
            end_sample_entered.set()
            assert reset_finished.wait(timeout=5.0)
            return 5.0
        return logical_clock[0]

    monkeypatch.setattr(budget_module.time, "perf_counter", perf_counter)

    def backend():
        logical_clock[0] = 5.0
        backend_returned.set()

    def reset_after_end_sample_starts():
        assert end_sample_entered.wait(timeout=5.0)
        logical_clock[0] = 6.0
        flops.budget_reset()
        reset_finished.set()

    reset_thread = threading.Thread(target=reset_after_end_sample_starts)
    reset_thread.start()
    try:
        with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
            with budget.deduct(
                "ordinary", flop_cost=1, subscripts=None, shapes=(), dtypes=()
            ):
                _call_numpy(backend)
                logical_clock[0] = 7.0
    finally:
        reset_finished.set()
        reset_thread.join(timeout=5.0)

    assert not reset_thread.is_alive()
    summary = flops.budget_summary_dict()
    assert summary["wall_time_s"] == 1.0
    assert summary["flopscope_backend_time_s"] == 0.0
    assert summary["flopscope_overhead_time_s"] == 1.0
    assert summary["residual_wall_time_s"] == 0.0
    assert budget._live_backend_calls == set()


@pytest.mark.parametrize("raises", [False, True])
def test_cross_thread_reset_during_backend_keeps_only_post_reset_overlap(
    monkeypatch, raises
):
    logical_clock = [0.0]
    backend_started = threading.Event()
    reset_finished = threading.Event()
    release_backend = threading.Event()
    reset_errors: list[BaseException] = []
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: logical_clock[0])

    def backend():
        logical_clock[0] = 2.0
        backend_started.set()
        assert release_backend.wait(timeout=5.0)
        logical_clock[0] = 5.0
        if raises:
            raise RuntimeError("backend failed")
        return "ok"

    def reset_while_backend_is_blocked():
        try:
            assert backend_started.wait(timeout=5.0)
            flops.budget_reset()
        except BaseException as exc:  # pragma: no cover - asserted below
            reset_errors.append(exc)
        finally:
            reset_finished.set()
            release_backend.set()

    reset_thread = threading.Thread(target=reset_while_backend_is_blocked)
    reset_thread.start()
    try:
        with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
            if raises:
                with pytest.raises(RuntimeError, match="backend failed"):
                    with budget.deduct(
                        "ordinary",
                        flop_cost=1,
                        subscripts=None,
                        shapes=(),
                        dtypes=(),
                    ):
                        _call_numpy(backend)
            else:
                with budget.deduct(
                    "ordinary", flop_cost=1, subscripts=None, shapes=(), dtypes=()
                ):
                    assert _call_numpy(backend) == "ok"
    finally:
        release_backend.set()
        reset_thread.join(timeout=5.0)

    assert reset_finished.is_set()
    assert not reset_thread.is_alive()
    assert not reset_errors
    summary = flops.budget_summary_dict()
    assert summary["wall_time_s"] == 3.0
    assert summary["flopscope_backend_time_s"] == 3.0
    assert summary["flopscope_overhead_time_s"] == 0.0
    assert summary["residual_wall_time_s"] == 0.0
    assert summary["operations"]["ordinary"]["flopscope_backend_time_s"] == 3.0
    assert budget._live_backend_calls == set()


class _NumericSleepyUfuncDuck:
    """Numeric duck whose ufunc protocol spends observable time in Python."""

    def __init__(self, payload, *, sleep_s=0.04, before_return=None, raises=None):
        self.payload = np.asarray(payload)
        self.sleep_s = sleep_s
        self.before_return = before_return
        self.raises = raises
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.payload, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.calls += 1
        time.sleep(self.sleep_s)
        if self.raises is not None:
            raise self.raises
        if self.before_return is not None:
            self.before_return()
        raw_inputs = tuple(self.payload if value is self else value for value in inputs)
        return getattr(ufunc, method)(*raw_inputs, **kwargs)


class _NumericSleepyUfuncArray(np.ndarray):
    """Foreign ndarray output whose ufunc protocol sleeps in Python."""

    def __new__(cls, payload, *, sleep_s=0.04):
        result = np.asarray(payload).view(cls)
        result.sleep_s = sleep_s
        result.calls = 0
        return result

    def __array_finalize__(self, original):
        if original is not None:
            self.sleep_s = getattr(original, "sleep_s", 0.04)
            self.calls = getattr(original, "calls", 0)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.calls += 1
        time.sleep(self.sleep_s)
        raw_inputs = tuple(
            np.asarray(value)
            if isinstance(value, (flops.FlopscopeArray, _NumericSleepyUfuncArray))
            else value
            for value in inputs
        )
        if "out" in kwargs and kwargs["out"] is not None:
            kwargs["out"] = tuple(
                None if value is None else np.asarray(value) for value in kwargs["out"]
            )
        return getattr(ufunc, method)(*raw_inputs, **kwargs)


def _run_callback_aware_add(duck):
    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**6, quiet=True) as budget:
        with budget.deduct(
            "add",
            flop_cost=2,
            subscripts=None,
            shapes=((2,), (2,)),
            dtypes=(np.float64,),
        ):
            result = _call_numpy_with_python_callbacks(
                np.add, np.array([1.0, 2.0]), duck
            )
    return result, budget


def test_numpy_protocol_callback_time_lands_in_residual_not_backend():
    duck = _NumericSleepyUfuncDuck([3.0, 4.0])

    result, budget = _run_callback_aware_add(duck)

    assert np.array_equal(result, np.array([4.0, 6.0]))
    assert duck.calls == 1
    summary = budget.summary_dict()
    assert summary["residual_wall_time_s"] >= 0.03, summary
    assert summary["flopscope_backend_time_s"] < 0.02, summary


@pytest.mark.parametrize(
    "op_name, make_duck, invoke",
    [
        (
            "add",
            lambda: _NumericSleepyUfuncDuck([3.0, 4.0]),
            lambda duck: fnp.add(fnp.array([1.0, 2.0]), duck),
        ),
        (
            "add.outer",
            lambda: _NumericSleepyUfuncDuck([3.0, 4.0]),
            lambda duck: np.add.outer(fnp.array([1.0, 2.0]), duck),
        ),
        (
            "subtract.reduce",
            lambda: _NumericSleepyUfuncArray(0.0),
            lambda duck: np.subtract.reduce(fnp.array([3.0, 4.0]), out=duck, axis=0),
        ),
        (
            "subtract.accumulate",
            lambda: _NumericSleepyUfuncArray([0.0, 0.0]),
            lambda duck: np.subtract.accumulate(
                fnp.array([3.0, 4.0]), out=duck, axis=0
            ),
        ),
        (
            "add.reduceat",
            lambda: _NumericSleepyUfuncArray([0.0, 0.0]),
            lambda duck: np.add.reduceat(
                fnp.array([3.0, 4.0]), [0, 1], out=duck, axis=0
            ),
        ),
        (
            "add.at",
            lambda: _NumericSleepyUfuncDuck([3.0, 4.0]),
            lambda duck: np.add.at(fnp.zeros(2), [0, 1], duck),
        ),
    ],
)
def test_foreign_ufunc_protocol_time_lands_in_residual(op_name, make_duck, invoke):
    duck = make_duck()
    flops.budget_reset()
    previous_callback_warnings = get_setting("callback_warnings")
    flops.configure(callback_warnings=True)
    try:
        with flops.BudgetContext(flop_budget=10**6, quiet=True) as budget:
            with pytest.warns(flops.errors.RemoteCallbackWarning, match=op_name):
                invoke(duck)
    finally:
        flops.configure(callback_warnings=previous_callback_warnings)

    assert duck.calls == 1
    summary = budget.summary_dict()
    assert summary["residual_wall_time_s"] >= 0.03, summary
    assert summary["flopscope_backend_time_s"] < 0.02, summary


def test_foreign_ufunc_symmetric_out_callback_time_lands_in_residual():
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    symmetric_input = flops.symmetrize(
        fnp.array([[1.0, 2.0], [2.0, 3.0]]), symmetry=symmetry
    )
    out = flops.symmetrize(fnp.zeros((2, 2)), symmetry=symmetry)
    duck = _NumericSleepyUfuncDuck(10.0)

    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**6, quiet=True) as budget:
        with pytest.warns(flops.errors.RemoteCallbackWarning, match="add") as caught:
            result = fnp.add(symmetric_input, duck, out=out)

    assert result is out
    assert duck.calls == 1
    assert len(caught) == 1
    np.testing.assert_array_equal(np.asarray(out), np.asarray(symmetric_input) + 10.0)
    summary = budget.summary_dict()
    assert summary["residual_wall_time_s"] >= 0.03, summary
    assert summary["flopscope_backend_time_s"] < 0.02, summary


@pytest.mark.parametrize(
    "op_name, invoke",
    [
        ("negative", lambda duck: fnp.negative(duck)),
        ("modf", lambda duck: fnp.modf(duck)),
    ],
)
def test_unary_ufunc_protocol_time_lands_in_residual(op_name, invoke):
    duck = _NumericSleepyUfuncDuck([3.0, 4.0])
    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**6, quiet=True) as budget:
        with pytest.warns(flops.errors.RemoteCallbackWarning, match=op_name):
            invoke(duck)

    assert duck.calls == 1
    summary = budget.summary_dict()
    assert summary["residual_wall_time_s"] >= 0.03, summary
    assert summary["flopscope_backend_time_s"] < 0.02, summary


def test_numpy_protocol_callback_chains_and_restores_existing_profile_hook():
    duck = _NumericSleepyUfuncDuck([3.0, 4.0], sleep_s=0.0)
    events = []

    def prior_profile(frame, event, arg):
        if frame.f_code.co_name == "__array_ufunc__":
            events.append(event)

    original_profile = sys.getprofile()
    sys.setprofile(prior_profile)
    try:
        _run_callback_aware_add(duck)
        assert sys.getprofile() is prior_profile
    finally:
        sys.setprofile(original_profile)

    assert "call" in events
    assert "return" in events


def test_callback_helper_skips_non_callable_existing_profiler(monkeypatch):
    profiler = object()
    setprofile_calls = []
    monkeypatch.setattr(budget_module._sys, "getprofile", lambda: profiler)
    monkeypatch.setattr(
        budget_module._sys,
        "setprofile",
        lambda profile: setprofile_calls.append(profile),
    )

    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**6, quiet=True) as budget:
        with budget.deduct(
            "add",
            flop_cost=2,
            subscripts=None,
            shapes=((2,), (2,)),
            dtypes=(np.float64,),
        ):
            result = _call_numpy_with_python_callbacks(
                np.add, np.array([1.0, 2.0]), np.array([3.0, 4.0])
            )
        with budget.deduct(
            "add",
            flop_cost=2,
            subscripts=None,
            shapes=((1,), ()),
            dtypes=(np.float64,),
        ):
            with pytest.raises(TypeError):
                _call_numpy_with_python_callbacks(np.add, np.array([1.0]), object())

    assert np.array_equal(result, np.array([4.0, 6.0]))
    assert budget_module._sys.getprofile() is profiler
    assert setprofile_calls == []


def test_callback_helper_preserves_active_cprofile_when_exposed():
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        if sys.getprofile() is None:
            pytest.skip("this runtime does not expose the active cProfile object")
        result, budget = _run_callback_aware_add(
            _NumericSleepyUfuncDuck([3.0, 4.0], sleep_s=0.0)
        )
        assert sys.getprofile() is profiler
    finally:
        profiler.disable()

    assert np.array_equal(result, np.array([4.0, 6.0]))
    assert budget.flopscope_backend_time_s >= 0.0


def _run_cprofiler_fallback_case(kind):
    callback_sleep = 0.02
    classified_sleep = 0.02

    if kind == "nested_counted":

        @_counted_wrapper
        def classified_work():
            budget = get_active_budget()
            assert budget is not None
            with budget.deduct(
                "add",
                flop_cost=2,
                subscripts=None,
                shapes=((2,), (2,)),
                dtypes=(np.float64,),
            ):
                _call_numpy(time.sleep, classified_sleep)

    elif kind == "same_timer":

        def classified_work():
            _call_numpy(time.sleep, classified_sleep)

    else:
        raise AssertionError(f"unknown fallback test kind: {kind}")

    duck = _NumericSleepyUfuncDuck(
        [3.0, 4.0], sleep_s=callback_sleep, before_return=classified_work
    )
    result, budget = _run_callback_aware_add(duck)
    return result, budget


def _assert_fallback_decomposition(budget):
    summary = budget.summary_dict()
    bucket_sum = (
        summary["flopscope_backend_time_s"]
        + summary["flopscope_overhead_time_s"]
        + summary["residual_wall_time_s"]
    )
    assert bucket_sum <= summary["wall_time_s"] + 0.015, summary
    assert bucket_sum == pytest.approx(summary["wall_time_s"], abs=0.015)
    return summary


@pytest.mark.parametrize("kind", ["nested_counted", "same_timer"])
def test_non_callable_profiler_fallback_excludes_already_classified_work(
    monkeypatch, kind
):
    profiler = object()
    setprofile_calls = []
    monkeypatch.setattr(budget_module._sys, "getprofile", lambda: profiler)
    monkeypatch.setattr(
        budget_module._sys,
        "setprofile",
        lambda profile: setprofile_calls.append(profile),
    )

    result, budget = _run_cprofiler_fallback_case(kind)

    assert np.array_equal(result, np.array([4.0, 6.0]))
    assert budget_module._sys.getprofile() is profiler
    assert setprofile_calls == []
    summary = _assert_fallback_decomposition(budget)
    assert summary["flopscope_backend_time_s"] >= 0.015, summary
    if kind == "nested_counted":
        assert len(budget.op_log) == 2
        nested_backend = budget.op_log[-1].flopscope_backend_duration_s
        assert nested_backend is not None
        assert nested_backend >= 0.015
    else:
        assert len(budget.op_log) == 1


@pytest.mark.parametrize("kind", ["nested_counted", "same_timer"])
def test_cprofile_fallback_excludes_already_classified_work_when_exposed(kind):
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        if sys.getprofile() is None:
            pytest.skip("this runtime does not expose the active cProfile object")
        result, budget = _run_cprofiler_fallback_case(kind)
        assert sys.getprofile() is profiler
    finally:
        profiler.disable()

    assert np.array_equal(result, np.array([4.0, 6.0]))
    _assert_fallback_decomposition(budget)


def test_numpy_protocol_callback_profiles_only_actual_callback_roots():
    """Cleanup must not create frames mistaken for NumPy protocol callbacks."""

    duck = _NumericSleepyUfuncDuck([3.0, 4.0], sleep_s=0.0)
    callback_roots = []

    def prior_profile(frame, event, arg):
        if (
            event == "call"
            and frame.f_back is not None
            and frame.f_back.f_code.co_name == "_call_numpy_impl"
            and sys.getprofile() is not prior_profile
        ):
            callback_roots.append(frame.f_code.co_name)

    original_profile = sys.getprofile()
    sys.setprofile(prior_profile)
    try:
        _run_callback_aware_add(duck)
        assert sys.getprofile() is prior_profile
    finally:
        sys.setprofile(original_profile)

    assert callback_roots == ["__array_ufunc__"]


def test_numpy_protocol_callback_restores_profile_after_profile_hook_error():
    duck = _NumericSleepyUfuncDuck([3.0, 4.0], sleep_s=0.0)

    def prior_profile(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "__array_ufunc__":
            raise RuntimeError("profile callback boom")

    original_profile = sys.getprofile()
    sys.setprofile(prior_profile)
    try:
        with pytest.raises(RuntimeError, match="profile callback boom"):
            _run_callback_aware_add(duck)
        assert sys.getprofile() is prior_profile
    finally:
        sys.setprofile(original_profile)


def test_numpy_protocol_callback_does_not_chain_hook_while_restoring_profile():
    duck = _NumericSleepyUfuncDuck([3.0, 4.0], sleep_s=0.0)
    restore_events = []

    def prior_profile(frame, event, arg):
        if (
            event == "c_call"
            and arg is sys.setprofile
            and frame.f_code.co_name == "_call_numpy_impl"
        ):
            restore_events.append(event)
            if len(restore_events) == 2:
                raise RuntimeError("profile restore boom")

    original_profile = sys.getprofile()
    sys.setprofile(prior_profile)
    try:
        _run_callback_aware_add(duck)
        assert sys.getprofile() is prior_profile
    finally:
        sys.setprofile(original_profile)

    assert restore_events == ["c_call"]


@pytest.mark.parametrize("replacement_kind", ["none", "replacement"])
def test_numpy_protocol_callback_reclaims_profile_after_prior_hook_mutation(
    replacement_kind,
):
    duck = _NumericSleepyUfuncDuck([3.0, 4.0])
    mutations = []

    def replacement_profile(frame, event, arg):
        pass

    def prior_profile(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "__array_ufunc__":
            mutations.append(event)
            sys.setprofile(None if replacement_kind == "none" else replacement_profile)

    original_profile = sys.getprofile()
    sys.setprofile(prior_profile)
    try:
        result, budget = _run_callback_aware_add(duck)
        assert sys.getprofile() is prior_profile
    finally:
        sys.setprofile(original_profile)

    assert np.array_equal(result, np.array([4.0, 6.0]))
    assert mutations == ["call"]
    summary = budget.summary_dict()
    assert summary["residual_wall_time_s"] >= 0.03, summary
    assert summary["flopscope_backend_time_s"] < 0.02, summary


def test_callback_tracker_does_not_retain_raw_call_arguments_without_gc():
    def call_with_large_argument():
        large = np.ones(1_000_000)
        reference = weakref.ref(large)
        duck = _NumericSleepyUfuncDuck([3.0], sleep_s=0.0)
        flops.budget_reset()
        with flops.BudgetContext(flop_budget=10**6, quiet=True) as budget:
            with budget.deduct(
                "add",
                flop_cost=2,
                subscripts=None,
                shapes=(large.shape, (1,)),
                dtypes=(np.float64,),
            ):
                _call_numpy_with_python_callbacks(np.add, large, duck)
        return reference

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        reference = call_with_large_argument()
        assert reference() is None
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


def test_numpy_protocol_callback_exception_stays_residual_and_propagates():
    error = RuntimeError("protocol boom")
    duck = _NumericSleepyUfuncDuck([3.0, 4.0], raises=error)

    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**6, quiet=True) as budget:
        with budget.deduct(
            "add",
            flop_cost=2,
            subscripts=None,
            shapes=((2,), (2,)),
            dtypes=(np.float64,),
        ):
            with pytest.raises(RuntimeError, match="protocol boom") as raised:
                _call_numpy_with_python_callbacks(np.add, np.array([1.0, 2.0]), duck)

    assert raised.value is error
    assert duck.calls == 1
    summary = budget.summary_dict()
    assert summary["residual_wall_time_s"] >= 0.03, summary
    assert summary["flopscope_backend_time_s"] < 0.02, summary


def test_numpy_protocol_callback_excludes_nested_flopscope_operation_from_residual():
    def nested_op():
        fnp.add(fnp.array([1.0, 2.0]), fnp.array([3.0, 4.0]))

    duck = _NumericSleepyUfuncDuck([3.0, 4.0], before_return=nested_op)

    result, budget = _run_callback_aware_add(duck)

    assert np.array_equal(result, np.array([4.0, 6.0]))
    assert duck.calls == 1
    assert sum(record.op_name == "add" for record in budget.op_log) >= 2
    summary = budget.summary_dict()
    assert summary["residual_wall_time_s"] >= 0.03, summary
    assert summary["flopscope_backend_time_s"] >= 0.0, summary
    assert summary["flopscope_overhead_time_s"] >= 0.0, summary
    assert summary["wall_time_s"] == pytest.approx(
        summary["flopscope_backend_time_s"]
        + summary["flopscope_overhead_time_s"]
        + summary["residual_wall_time_s"],
        abs=0.01,
    )


def test_nested_callback_aware_calls_exclude_same_timer_backend_from_user_time(
    monkeypatch,
):
    callback_sleep = 0.02
    backend_sleep = 0.02
    logical_clock = 0.0

    def perf_counter():
        return logical_clock

    def sleep(duration):
        nonlocal logical_clock
        logical_clock += duration

    monkeypatch.setattr(budget_module.time, "perf_counter", perf_counter)
    monkeypatch.setattr(time, "sleep", sleep)

    inner = _NumericSleepyUfuncDuck([3.0, 4.0], sleep_s=callback_sleep)

    def same_timer_work():
        _call_numpy(time.sleep, backend_sleep)
        _call_numpy_with_python_callbacks(np.add, np.array([1.0, 2.0]), inner)

    outer = _NumericSleepyUfuncDuck(
        [3.0, 4.0], sleep_s=callback_sleep, before_return=same_timer_work
    )

    result, budget = _run_callback_aware_add(outer)

    assert np.array_equal(result, np.array([4.0, 6.0]))
    assert (outer.calls, inner.calls) == (1, 1)
    assert budget._total_user_code_time == pytest.approx(  # type: ignore[attr-defined]
        2 * callback_sleep, abs=1e-12
    )
    summary = budget.summary_dict()
    assert summary["flopscope_backend_time_s"] == pytest.approx(
        backend_sleep, abs=1e-12
    ), summary
    assert summary["residual_wall_time_s"] == pytest.approx(
        2 * callback_sleep, abs=1e-12
    ), summary
    assert summary["wall_time_s"] == pytest.approx(
        2 * callback_sleep + backend_sleep, abs=1e-12
    ), summary
    assert summary["wall_time_s"] == pytest.approx(
        summary["flopscope_backend_time_s"]
        + summary["flopscope_overhead_time_s"]
        + summary["residual_wall_time_s"],
        abs=1e-12,
    ), summary


def test_nested_counted_callback_aware_call_is_not_double_counted_as_user_time(
    monkeypatch,
):
    callback_sleep = 0.02
    logical_clock = 0.0

    def perf_counter():
        return logical_clock

    def sleep(duration):
        nonlocal logical_clock
        logical_clock += duration

    monkeypatch.setattr(budget_module.time, "perf_counter", perf_counter)
    monkeypatch.setattr(time, "sleep", sleep)

    inner = _NumericSleepyUfuncDuck([3.0, 4.0], sleep_s=callback_sleep)

    @_counted_wrapper
    def nested_callback_op():
        budget = get_active_budget()
        assert budget is not None
        with budget.deduct(
            "add",
            flop_cost=2,
            subscripts=None,
            shapes=((2,), (2,)),
            dtypes=(np.float64,),
        ):
            _call_numpy_with_python_callbacks(np.add, np.array([1.0, 2.0]), inner)

    outer = _NumericSleepyUfuncDuck(
        [3.0, 4.0], sleep_s=callback_sleep, before_return=nested_callback_op
    )

    result, budget = _run_callback_aware_add(outer)

    assert np.array_equal(result, np.array([4.0, 6.0]))
    assert (outer.calls, inner.calls) == (1, 1)
    assert sum(record.op_name == "add" for record in budget.op_log) == 2
    assert budget._total_user_code_time == pytest.approx(  # type: ignore[attr-defined]
        2 * callback_sleep, abs=1e-12
    )
    summary = budget.summary_dict()
    assert summary["flopscope_backend_time_s"] == pytest.approx(0.0, abs=1e-12), summary
    assert summary["residual_wall_time_s"] == pytest.approx(
        2 * callback_sleep, abs=1e-12
    ), summary
    assert summary["wall_time_s"] == pytest.approx(2 * callback_sleep, abs=1e-12), (
        summary
    )
    assert summary["wall_time_s"] == pytest.approx(
        summary["flopscope_backend_time_s"]
        + summary["flopscope_overhead_time_s"]
        + summary["residual_wall_time_s"],
        abs=1e-12,
    )


def test_user_code_time_lands_in_residual_not_overhead():
    """Wall time spent in _call_user_code must bill to residual, not overhead."""

    @_counted_wrapper
    def fake_callback_op():
        budget = get_active_budget()
        assert budget is not None
        _call_user_code(budget, time.sleep, 0.05)

    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**6, quiet=True) as b:
        fake_callback_op()
    s = b.summary_dict()
    assert s["residual_wall_time_s"] >= 0.03, s
    assert s["flopscope_overhead_time_s"] < 0.02, s
    assert s["wall_time_s"] == pytest.approx(
        s["flopscope_backend_time_s"]
        + s["flopscope_overhead_time_s"]
        + s["residual_wall_time_s"],
        abs=1e-6,
    )


def test_user_code_nested_flopscope_op_not_double_counted():
    """Callback that runs a real flopscope op: the op's time stays in
    backend/overhead, and the pure-Python remainder (sleep) goes to residual."""

    @_counted_wrapper
    def fake_callback_with_inner_op():
        budget = get_active_budget()
        assert budget is not None

        def cb():
            time.sleep(0.03)
            fnp.add(fnp.array([1.0, 2.0]), fnp.array([3.0, 4.0]))
            return 0.0

        _call_user_code(budget, cb)

    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**6, quiet=True) as b:
        fake_callback_with_inner_op()
    s = b.summary_dict()
    # the inner flopscope op ran and was counted (nested > 0 path exercised)
    assert any(rec.op_name == "add" for rec in b.op_log), [r.op_name for r in b.op_log]
    # the 0.03s sleep (pure user time) lands in residual, not overhead
    assert s["residual_wall_time_s"] >= 0.02, s
    assert s["flopscope_overhead_time_s"] < 0.02, s
    # decomposition identity holds
    assert s["wall_time_s"] == pytest.approx(
        s["flopscope_backend_time_s"]
        + s["flopscope_overhead_time_s"]
        + s["residual_wall_time_s"],
        abs=1e-6,
    )


CALLBACK_SLEEP = 0.05


def _sleepy(*_a, **_k):
    time.sleep(CALLBACK_SLEEP)
    return 0.0


def _lazy_sleepy_gen():
    yield _sleepy()


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: fnp.apply_along_axis(
            lambda row: _sleepy(), 1, fnp.array(np.zeros((1, 3)))
        ),
        lambda: fnp.apply_over_axes(
            lambda a, ax: (_sleepy(), np.sum(a, axis=ax, keepdims=True))[1],
            fnp.array(np.zeros((1, 3))),
            [1],
        ),
        lambda: fnp.piecewise(
            fnp.array(np.zeros(3)),
            [np.array([True, False, False])],
            [lambda v: (_sleepy(), 0.0)[1], 0.0],
        ),
        lambda: fnp.fromfunction(
            lambda i, j: (_sleepy(), i + j)[1], (2, 2), dtype=float
        ),
        lambda: fnp.fromiter(_lazy_sleepy_gen(), dtype=float),
    ],
    ids=[
        "apply_along_axis",
        "apply_over_axes",
        "piecewise",
        "fromfunction",
        "fromiter",
    ],
)
def test_callback_ops_bill_callback_to_residual(invoke):
    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as b:
        invoke()
    s = b.summary_dict()
    assert s["residual_wall_time_s"] >= 0.03, s
    assert s["flopscope_overhead_time_s"] < 0.02, s


def test_deduct_after_attributes_call_to_backend_and_charges():
    from flopscope._budget import _call_numpy

    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as b:

        @_counted_wrapper
        def fake_movement():
            budget = get_active_budget()
            assert budget is not None
            with budget.deduct_after(
                "tile", subscripts=None, shapes=(), dtypes=()
            ) as op:
                _call_numpy(time.sleep, 0.05)  # stand-in for numpy data movement
                op.set_cost(1000)

        fake_movement()
    s = b.summary_dict()
    assert b.flops_used == 1000  # weight("tile") == 1.0
    assert s["flopscope_backend_time_s"] >= 0.03, s
    assert s["flopscope_overhead_time_s"] < 0.02, s
    assert s["wall_time_s"] == pytest.approx(
        s["flopscope_backend_time_s"]
        + s["flopscope_overhead_time_s"]
        + s["residual_wall_time_s"],
        abs=1e-6,
    )


def test_deduct_after_overshoot_raises_without_recording():
    flops.budget_reset()
    with flops.BudgetContext(flop_budget=100, quiet=True) as b:

        @_counted_wrapper
        def fake():
            budget = get_active_budget()
            assert budget is not None
            with pytest.raises(flops.errors.BudgetExhaustedError):
                with budget.deduct_after(
                    "tile", subscripts=None, shapes=(), dtypes=()
                ) as op:
                    op.set_cost(1000)  # exceeds budget of 100

        fake()
    assert b.flops_used == 0
    assert all(rec.op_name != "tile" for rec in b.op_log)


def test_deduct_after_without_set_cost_raises_runtime_error():
    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**9, quiet=True):

        @_counted_wrapper
        def fake():
            budget = get_active_budget()
            assert budget is not None
            with pytest.raises(RuntimeError, match="set_cost"):
                with budget.deduct_after("tile", subscripts=None, shapes=(), dtypes=()):
                    pass  # forgot to call set_cost

        fake()


def test_deduct_after_attributes_backend_even_when_block_raises():
    from flopscope._budget import _call_numpy

    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as b:

        @_counted_wrapper
        def fake():
            budget = get_active_budget()
            assert budget is not None
            with pytest.raises(ValueError):
                with budget.deduct_after(
                    "tile", subscripts=None, shapes=(), dtypes=()
                ) as op:
                    _call_numpy(time.sleep, 0.04)
                    raise ValueError("boom")

        fake()
    s = b.summary_dict()
    assert s["flopscope_backend_time_s"] >= 0.02, s  # backend attributed despite raise
    assert b.flops_used == 0  # nothing charged on the raising path
    assert all(rec.op_name != "tile" for rec in b.op_log)  # nothing recorded


def test_counted_wrapper_preflight_preserves_incremental_op_timing():
    """A custom preflight and incremental timing attribution compose safely."""
    events = []

    def preflight(args, kwargs):
        events.append(("preflight", args, kwargs))

    @_counted_wrapper(preflight=preflight)
    def fake(value, *, scale=1):
        events.append(("body", value, scale))
        budget = get_active_budget()
        assert budget is not None
        with budget.deduct(
            "preflight_test", flop_cost=3, subscripts=None, shapes=(), dtypes=()
        ):
            pass
        return value * scale

    with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
        assert fake(4, scale=2) == 8

    assert events == [("preflight", (4,), {"scale": 2}), ("body", 4, 2)]
    summary = budget.summary_dict()
    operation = summary["operations"]["preflight_test"]
    assert summary["flops_used"] == 3
    assert operation["calls"] == 1
    assert operation["flopscope_overhead_time_s"] == pytest.approx(
        budget.op_log[0].flopscope_overhead_duration_s
    )


@pytest.mark.parametrize(
    "invoke",
    [
        lambda big: flops.numpy.tile(big, (2, 2)),
        lambda big: flops.numpy.repeat(big, 4, axis=0),
        lambda big: flops.numpy.take(
            flops.numpy.reshape(big, (-1,)), np.arange(big.size // 2)
        ),
        lambda big: flops.numpy.resize(big, (big.shape[0] * 2, big.shape[1] * 2)),
    ],
    ids=["tile", "repeat", "take", "resize"],
)
def test_data_movement_ops_bill_to_backend(invoke):
    big = flops.numpy.array(np.random.randn(2000, 2000))
    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**12, quiet=True) as b:
        invoke(big)
    s = b.summary_dict()
    assert s["flopscope_backend_time_s"] > s["flopscope_overhead_time_s"], s


@pytest.mark.parametrize(
    "invoke",
    [
        lambda big: np.subtract.outer(big[0], big[0]),
        lambda big: np.subtract.reduce(big, axis=0),
        lambda big: np.subtract.accumulate(big, axis=0),
        lambda big: np.subtract.reduceat(big, [0, big.shape[0] // 2], axis=0),
        lambda big: np.add.at(big, (np.arange(big.shape[0]),), 1.0),
    ],
    ids=["outer", "reduce", "accumulate", "reduceat", "at"],
)
def test_ufunc_methods_bill_to_backend(invoke):
    """The ufunc-method wrappers invoked numpy directly instead of through
    ``_call_numpy``, so their whole backend cost was misfiled as flopscope
    overhead -- ``flopscope_backend_time_s`` came back at exactly 0.0 for calls
    taking milliseconds. The same omission is why their writes went unrecorded
    (pinned in ``test_symmetry_tag_forgery``); this is the timing half.
    """
    big = flops.numpy.array(np.random.randn(2000, 2000))
    invoke(big)
    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**12, quiet=True) as b:
        invoke(big)
    s = b.summary_dict()
    assert s["flopscope_backend_time_s"] > 0.0, s
    assert s["flopscope_backend_time_s"] > s["flopscope_overhead_time_s"], s
