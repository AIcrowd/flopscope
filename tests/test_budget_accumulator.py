"""Tests for BudgetAccumulator and budget_summary_dict()."""

from flopscope._budget import (
    BudgetAccumulator,
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


def test_get_data_falls_back_to_legacy_namespace_record_op_log() -> None:
    from flopscope._budget import OpRecord

    accumulator = BudgetAccumulator()
    operation = OpRecord(
        op_name="add",
        subscripts=None,
        shapes=(),
        flop_cost=5,
        cumulative=5,
        namespace="train",
        flopscope_backend_duration_s=0.25,
        flopscope_overhead_duration_s=0.05,
    )
    accumulator._records.append(
        NamespaceRecord(
            namespace="train",
            flop_budget=100,
            flops_used=5,
            op_log=[operation],
        )
    )

    result = accumulator.get_data(by_namespace=True)
    assert result["operations"]["add"]["calls"] == 1
    assert result["operations"]["add"]["flop_cost"] == 5
    assert result["operations"]["add"]["flopscope_backend_time_s"] == 0.25
    assert (
        result["by_namespace"]["train"]["operations"]["add"]
        == result["operations"]["add"]
    )


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


def test_accumulator_reset_clears_incremental_snapshot(monkeypatch) -> None:
    import flopscope._budget as budget_module

    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)
    with BudgetContext(flop_budget=100, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
    assert accumulator.snapshot()["flops_used"] == 5

    accumulator.reset()

    result = accumulator.snapshot()
    assert result["flop_budget"] == 0
    assert result["flops_used"] == 0
    assert result["operations"] == {}
    assert result["wall_time_s"] is None


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


def test_closed_session_timing_matches_final_context_timing() -> None:
    import pytest

    import flopscope as flops

    with flops.BudgetContext(100, quiet=True) as ctx:
        with ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=()):
            pass
    session = flops.budget_summary_dict()
    assert session["wall_time_s"] == pytest.approx(ctx.wall_time_s, abs=1e-9)
    assert session["flopscope_backend_time_s"] == pytest.approx(
        ctx.flopscope_backend_time_s, abs=1e-9
    )
    assert session["flopscope_overhead_time_s"] == pytest.approx(
        ctx.flopscope_overhead_time_s, abs=1e-9
    )


def test_reused_context_does_not_bill_inter_call_gap() -> None:
    import time

    import flopscope as flops

    ctx = flops.BudgetContext(100, quiet=True)
    with ctx:
        ctx.deduct("add", flop_cost=1, subscripts=None, shapes=(), dtypes=())
    first_wall = ctx.wall_time_s
    assert first_wall is not None
    time.sleep(0.03)
    with ctx:
        ctx.deduct("add", flop_cost=1, subscripts=None, shapes=(), dtypes=())
    session = flops.budget_summary_dict()
    assert ctx.wall_time_s is not None
    assert ctx.wall_time_s < 0.02
    assert session["wall_time_s"] < first_wall + 0.02


def test_active_record_then_close_does_not_merge_prior_wall_twice(
    monkeypatch,
) -> None:
    import flopscope._budget as budget_module

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)

    ctx = BudgetContext(100, quiet=True)
    with ctx:
        ctx._add_flopscope_overhead(0.1)
        now[0] = 2.0
        accumulator.record(ctx)
        assert accumulator.snapshot()["wall_time_s"] == 2.0
        now[0] = 5.0
        ctx._add_flopscope_overhead(0.1)
        assert ctx._snapshot_record().wall_time_s == 3.0
        assert budget_summary_dict()["wall_time_s"] == 5.0

    assert ctx.wall_time_s == 5.0
    assert accumulator.snapshot()["wall_time_s"] == 5.0


def test_budget_reset_mid_context_excludes_prior_wall_on_close(monkeypatch) -> None:
    import pytest

    import flopscope._budget as budget_module

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)

    ctx = BudgetContext(100, quiet=True)
    now[0] = 1.0
    with ctx:
        ctx._add_flopscope_overhead(0.1)
        now[0] = 4.0
        budget_reset()
        assert accumulator.snapshot()["wall_time_s"] is None
        now[0] = 9.0
        ctx._add_flopscope_overhead(0.1)
        assert ctx._snapshot_record().wall_time_s == 5.0
        assert budget_summary_dict()["wall_time_s"] == 5.0

    assert ctx.wall_time_s == 9.0
    assert accumulator.snapshot()["wall_time_s"] == 5.0
    assert accumulator.snapshot()["flopscope_overhead_time_s"] == pytest.approx(0.1)


def test_context_created_before_reset_excludes_pre_reset_construction(
    monkeypatch,
) -> None:
    import flopscope._budget as budget_module

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)

    ctx = BudgetContext(100, quiet=True)
    now[0] = 4.0
    budget_reset()
    now[0] = 6.0
    with ctx:
        now[0] = 9.0

    result = accumulator.snapshot()
    assert ctx.wall_time_s == 9.0
    assert result["wall_time_s"] == 5.0
    assert result["flopscope_overhead_time_s"] == 2.0
    assert result["residual_wall_time_s"] == 3.0


def test_interim_record_after_reset_preserves_pending_pre_enter_overhead(
    monkeypatch,
) -> None:
    import pytest

    import flopscope._budget as budget_module

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)

    ctx = BudgetContext(100, quiet=True)
    now[0] = 1.0
    with ctx:
        ctx._add_flopscope_overhead(0.1)
        now[0] = 4.0
        budget_reset()
        now[0] = 6.0
        ctx._add_flopscope_overhead(0.2)
        accumulator.record(ctx)
        now[0] = 9.0

    result = accumulator.snapshot()
    assert result["wall_time_s"] == 5.0
    assert result["flopscope_overhead_time_s"] == pytest.approx(0.2)
    assert result["residual_wall_time_s"] == pytest.approx(4.8)


def test_global_default_reset_uses_no_process_age_timing_origin(monkeypatch) -> None:
    import flopscope._budget as budget_module

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    global_ctx = budget_module._get_global_default()
    global_ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
    global_ctx._add_flopscope_backend(0.5)
    global_ctx._add_flopscope_overhead(0.25)

    now[0] = 100.0
    budget_reset()
    now[0] = 101.0
    global_ctx.deduct("add", flop_cost=7, subscripts=None, shapes=(), dtypes=())
    global_ctx._add_flopscope_backend(2.0)
    global_ctx._add_flopscope_overhead(3.0)

    first = budget_summary_dict(by_namespace=True)
    second = budget_summary_dict(by_namespace=True)
    assert first == second
    assert first["wall_time_s"] is None
    assert first["residual_wall_time_s"] is None
    assert first["flops_used"] == 7
    assert first["operations"]["add"]["flop_cost"] == 7
    assert first["operations"]["add"]["calls"] == 1
    assert first["flopscope_backend_time_s"] == 2.0
    assert first["flopscope_overhead_time_s"] == 3.0
    assert (
        first["by_namespace"][None]["operations"]["add"] == first["operations"]["add"]
    )


def test_reentry_resets_recorded_wall_baseline(monkeypatch) -> None:
    import flopscope._budget as budget_module

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)

    ctx = BudgetContext(100, quiet=True)
    with ctx:
        now[0] = 2.0
    assert ctx._recorded_wall_time_s == 2.0

    now[0] = 10.0
    with ctx:
        assert ctx._recorded_wall_time_s == 0.0
        now[0] = 13.0

    assert ctx.wall_time_s == 3.0
    assert accumulator.snapshot()["wall_time_s"] == 5.0


def test_context_close_timing_includes_accumulator_bookkeeping(monkeypatch) -> None:
    import flopscope._budget as budget_module

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)

    ctx = BudgetContext(100, quiet=True)
    original_mark_recorded = ctx._mark_recorded
    mark_calls = 0

    def mark_recorded(**kwargs):
        nonlocal mark_calls
        original_mark_recorded(**kwargs)
        mark_calls += 1
        if mark_calls == 1:
            now[0] += 2.0

    original_invalidate_caches = accumulator._invalidate_caches

    def invalidate_caches(*, rollup_changed):
        original_invalidate_caches(rollup_changed=rollup_changed)
        now[0] += 3.0

    monkeypatch.setattr(ctx, "_mark_recorded", mark_recorded)
    monkeypatch.setattr(accumulator, "_invalidate_caches", invalidate_caches)

    with ctx:
        now[0] = 5.0

    result = accumulator.snapshot()
    assert mark_calls == 2
    assert ctx.wall_time_s == 10.0
    assert result["wall_time_s"] == 10.0
    assert result["flopscope_overhead_time_s"] == 5.0
    assert result["residual_wall_time_s"] == 5.0
    assert accumulator._records[-1].wall_time_s == 10.0
    assert accumulator._records[-1].total_flopscope_overhead_time == 5.0


def test_record_commits_wall_boundary_from_delta_snapshot(monkeypatch) -> None:
    import flopscope._budget as budget_module

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)

    ctx = BudgetContext(100, quiet=True)
    with ctx:
        ctx._add_flopscope_overhead(0.1)
        now[0] = 2.0
        original_snapshot = ctx._snapshot_summary_delta

        def snapshot_then_advance(**kwargs):
            delta = original_snapshot(**kwargs)
            now[0] = 4.0
            return delta

        monkeypatch.setattr(ctx, "_snapshot_summary_delta", snapshot_then_advance)
        accumulator.record(ctx)
        assert accumulator.snapshot()["wall_time_s"] == 2.0

        monkeypatch.setattr(ctx, "_snapshot_summary_delta", original_snapshot)
        now[0] = 6.0
        ctx._add_flopscope_overhead(0.1)
        accumulator.record(ctx)

        assert accumulator.snapshot()["wall_time_s"] == 6.0
