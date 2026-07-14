"""Timing-bucket attribution tests (issue: callback/data-movement misattribution)."""

import time

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._budget import _call_user_code, _counted_wrapper, get_active_budget


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
