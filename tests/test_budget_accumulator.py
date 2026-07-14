"""Tests for BudgetAccumulator and budget_summary_dict()."""

from flopscope._budget import (
    BudgetContext,
    NamespaceRecord,
    budget_reset,
    budget_summary_dict,
)


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
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=(), dtypes=())
        ctx.deduct("mul", flop_cost=200, subscripts=None, shapes=(), dtypes=())

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
        with ctx.deduct("mul", flop_cost=10, subscripts=None, shapes=(), dtypes=()):
            pass
        with flops.namespace("precompute"):
            with ctx.deduct("add", flop_cost=25, subscripts=None, shapes=(), dtypes=()):
                pass
    with BudgetContext(flop_budget=500, namespace="predict", quiet=True) as ctx:
        with flops.namespace("precompute"):
            with ctx.deduct("add", flop_cost=15, subscripts=None, shapes=(), dtypes=()):
                pass
    with BudgetContext(flop_budget=250, quiet=True) as ctx:
        with ctx.deduct("sum", flop_cost=5, subscripts=None, shapes=(), dtypes=()):
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
            ctx.deduct("add", flop_cost=25, subscripts=None, shapes=(), dtypes=())

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
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=(), dtypes=())
    with BudgetContext(flop_budget=2000, namespace="b", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=300, subscripts=None, shapes=(), dtypes=())

    data = budget_summary_dict()
    assert data["flops_used"] == 400
    assert data["operations"]["add"]["flop_cost"] == 400
    assert data["operations"]["add"]["calls"] == 2
    assert data["flopscope_backend_time_s"] == 0.0
    assert data["residual_wall_time_s"] is not None
    assert data["residual_wall_time_s"] >= 0


def test_budget_reset():
    with BudgetContext(flop_budget=1000, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=(), dtypes=())
    budget_reset()
    data = budget_summary_dict()
    assert data["flops_used"] == 0
    assert data["operations"] == {}


def test_budget_summary_dict_does_not_double_count_reused_decorator_context():
    import flopscope as flops
    from flopscope._budget import get_active_budget

    seen_totals = []

    @flops.budget(flop_budget=5000, namespace="dec", quiet=True)
    def compute():
        ctx = get_active_budget()
        assert ctx is not None
        ctx.deduct("add", flop_cost=10, subscripts=None, shapes=(), dtypes=())
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

        with ctx.deduct("add", flop_cost=10, subscripts=None, shapes=(), dtypes=()):
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
