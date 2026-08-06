"""Tests for BudgetAccumulator and budget_summary_dict()."""

from flopscope._budget import (
    BudgetContext,
    NamespaceRecord,
    budget_reset,
    budget_summary_dict,
)


class _PoisonedHistory(list):
    """Fail if a summary regresses to scanning diagnostic operation history."""

    def __iter__(self):
        raise AssertionError("summary traversed historical operation records")

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError("summary sliced historical operation records")
        return super().__getitem__(index)


def test_namespace_record_fields():
    rec = NamespaceRecord(
        namespace="train",
        flop_budget=1000,
        flops_used=500,
        op_log=[],
    )
    assert rec.namespace == "train"
    assert rec.flop_budget == 1000
    assert rec.flops_used == 500


def test_budget_summary_dict_unlabeled():
    with BudgetContext(flop_budget=1000, namespace="a", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())
        ctx.deduct("mul", flop_cost=200, subscripts=None, shapes=())

    data = budget_summary_dict()
    assert data["flops_used"] == 300
    assert data["flop_budget"] == 1000
    assert data["flops_remaining"] == 700
    assert data["operations"]["add"]["flop_cost"] == 100
    assert data["operations"]["add"]["calls"] == 1
    assert data["operations"]["mul"]["flop_cost"] == 200


def test_budget_summary_dict_by_namespace():
    import flopscope as flops

    with BudgetContext(flop_budget=1000, namespace="predict", quiet=True) as ctx:
        with ctx.deduct("mul", flop_cost=10, subscripts=None, shapes=()):
            pass
        with flops.namespace("precompute"):
            with ctx.deduct("add", flop_cost=25, subscripts=None, shapes=()):
                pass
    with BudgetContext(flop_budget=500, namespace="predict", quiet=True) as ctx:
        with flops.namespace("precompute"):
            with ctx.deduct("add", flop_cost=15, subscripts=None, shapes=()):
                pass
    with BudgetContext(flop_budget=250, quiet=True) as ctx:
        with ctx.deduct("sum", flop_cost=5, subscripts=None, shapes=()):
            pass

    data = budget_summary_dict(by_namespace=True)
    assert data["flops_used"] == 55
    assert set(data["by_namespace"]) == {"predict", "predict.precompute", None}

    root_bucket = data["by_namespace"]["predict"]
    assert root_bucket["flops_used"] == 10
    assert root_bucket["calls"] == 1
    assert root_bucket["flopscope_backend_time_s"] >= 0
    assert root_bucket["operations"]["mul"]["flop_cost"] == 10
    assert "flop_budget" not in root_bucket
    assert "wall_time_s" not in root_bucket
    assert "residual_wall_time_s" not in root_bucket

    nested_bucket = data["by_namespace"]["predict.precompute"]
    assert nested_bucket["flops_used"] == 40
    assert nested_bucket["calls"] == 2
    assert nested_bucket["flopscope_backend_time_s"] >= 0
    assert nested_bucket["operations"]["add"]["flop_cost"] == 40
    assert nested_bucket["operations"]["add"]["calls"] == 2

    unlabeled_bucket = data["by_namespace"][None]
    assert unlabeled_bucket["flops_used"] == 5
    assert unlabeled_bucket["calls"] == 1


def test_budget_summary_dict_by_namespace_uses_nested_op_namespace():
    import flopscope as flops

    with BudgetContext(flop_budget=1000, namespace="predict..raw", quiet=True) as ctx:
        with flops.namespace("precompute"):
            ctx.deduct("add", flop_cost=25, subscripts=None, shapes=())

    data = budget_summary_dict(by_namespace=True)
    assert "predict..raw.precompute" in data["by_namespace"]
    assert (
        data["by_namespace"]["predict..raw.precompute"]["operations"]["add"][
            "flop_cost"
        ]
        == 25
    )


def test_budget_summary_dict_accumulates_across_contexts():
    with BudgetContext(flop_budget=1000, namespace="a", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())
    with BudgetContext(flop_budget=2000, namespace="b", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=300, subscripts=None, shapes=())

    data = budget_summary_dict()
    assert data["flops_used"] == 400
    assert data["operations"]["add"]["flop_cost"] == 400
    assert data["operations"]["add"]["calls"] == 2
    assert data["flopscope_backend_time_s"] == 0.0
    assert data["residual_wall_time_s"] is not None
    assert data["residual_wall_time_s"] >= 0


def test_budget_reset():
    with BudgetContext(flop_budget=1000, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())
    budget_reset()
    data = budget_summary_dict()
    assert data["flops_used"] == 0
    assert data["operations"] == {}


def test_budget_reset_clears_summary_baselines_without_resetting_enforcement():
    import flopscope

    with BudgetContext(flop_budget=100, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=20, subscripts=None, shapes=())
        flopscope.budget_reset()

        assert flopscope.budget_summary_dict()["flops_used"] == 0
        assert flopscope.current_budget() == flopscope.BudgetSnapshot(100, 20, 80)

        ctx.deduct("multiply", flop_cost=30, subscripts=None, shapes=())
        data = flopscope.budget_summary_dict()
        assert data["flops_used"] == 30
        assert set(data["operations"]) == {"multiply"}
        assert data["operations"]["multiply"]["flop_cost"] == 30
        assert data["operations"]["multiply"]["calls"] == 1


def test_summaries_use_incremental_aggregates_not_operation_history():
    import flopscope

    with BudgetContext(flop_budget=100, quiet=True) as ctx:
        timer = ctx.deduct("add", flop_cost=5, subscripts=None, shapes=())
        with timer:
            pass

    ctx._op_log = _PoisonedHistory(ctx._op_log)
    assert ctx.summary_dict(by_namespace=True)["operations"]["add"]["calls"] == 1

    import flopscope._budget as budget_module

    original_records = budget_module._accumulator._records
    budget_module._accumulator._records = _PoisonedHistory(original_records)
    try:
        assert (
            flopscope.budget_summary_dict(by_namespace=True)["operations"]["add"][
                "flop_cost"
            ]
            == 5
        )
    finally:
        budget_module._accumulator._records = original_records


def test_budget_reset_mid_timer_preserves_post_reset_operation_timing():
    import time

    import flopscope
    from flopscope._budget import _call_numpy

    with BudgetContext(flop_budget=100, quiet=True) as ctx:
        timer = ctx.deduct("add", flop_cost=5, subscripts=None, shapes=())
        flopscope.budget_reset()
        with timer:
            _call_numpy(time.sleep, 0.001)

        operation = flopscope.budget_summary_dict()["operations"]["add"]
        assert operation["calls"] == 0
        assert operation["flop_cost"] == 0
        assert operation["flopscope_backend_time_s"] >= 0.001
        assert operation["flopscope_overhead_time_s"] >= 0.0


def test_closed_summary_includes_context_close_bookkeeping_time():
    import pytest

    import flopscope

    with BudgetContext(flop_budget=100, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=5, subscripts=None, shapes=())

    summary = flopscope.budget_summary_dict()
    assert summary["wall_time_s"] == pytest.approx(ctx.wall_time_s, abs=1e-9)
    assert summary["flopscope_overhead_time_s"] == pytest.approx(
        ctx.flopscope_overhead_time_s, abs=1e-9
    )


def test_budget_summary_dict_does_not_double_count_reused_decorator_context():
    import flopscope as flops
    from flopscope._budget import get_active_budget

    seen_totals = []

    @flops.budget(flop_budget=5000, namespace="dec", quiet=True)
    def compute():
        ctx = get_active_budget()
        assert ctx is not None
        ctx.deduct("add", flop_cost=10, subscripts=None, shapes=())
        seen_totals.append(flops.budget_summary_dict()["flops_used"])

    compute()
    compute()

    data = flops.budget_summary_dict()
    assert seen_totals == [10, 20]
    assert data["flop_budget"] == 5000
    assert data["flops_used"] == 20
    assert data["operations"]["add"]["flop_cost"] == 20
    assert data["operations"]["add"]["calls"] == 2


def test_reused_decorator_context_resets_live_timing_state_between_calls():
    import time

    import flopscope as flops
    from flopscope._budget import get_active_budget

    budget_ctx = flops.budget(flop_budget=5000, namespace="dec", quiet=True)
    seen_ctx_wall_times = []
    seen_context_live_wall_times = []
    seen_global_live_wall_times = []

    @budget_ctx
    def compute(post_sleep_s: float) -> None:
        ctx = get_active_budget()
        assert ctx is not None
        assert ctx is budget_ctx
        seen_ctx_wall_times.append(ctx.wall_time_s)

        with ctx.deduct("add", flop_cost=10, subscripts=None, shapes=()):
            pass

        seen_context_live_wall_times.append(ctx.summary_dict()["wall_time_s"])
        seen_global_live_wall_times.append(flops.budget_summary_dict()["wall_time_s"])

        if post_sleep_s:
            time.sleep(post_sleep_s)

    compute(0.03)
    first_closed_wall_time = budget_ctx.wall_time_s
    compute(0.0)

    assert first_closed_wall_time is not None
    assert first_closed_wall_time >= 0.03
    assert seen_ctx_wall_times == [None, None]
    assert seen_context_live_wall_times[1] is not None
    assert seen_context_live_wall_times[1] < first_closed_wall_time / 2
    assert seen_global_live_wall_times[1] is not None
    assert seen_global_live_wall_times[1] < first_closed_wall_time * 1.5
