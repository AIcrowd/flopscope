from __future__ import annotations

from copy import deepcopy

from flopscope._budget import (
    OpRecord,
    _summarize_by_namespace,
    _summarize_operations,
    _SummaryRollup,
)


class _PoisonedHistory(list):
    def __iter__(self):
        raise AssertionError("summary traversed historical operation records")


def _op(
    name: str = "add",
    *,
    cost: int = 5,
    namespace: str | None = "predict",
    backend: float | None = None,
    overhead: float | None = None,
) -> OpRecord:
    return OpRecord(
        op_name=name,
        subscripts=None,
        shapes=((1,),),
        flop_cost=cost,
        cumulative=cost,
        namespace=namespace,
        flopscope_backend_duration_s=backend,
        flopscope_overhead_duration_s=overhead,
    )


def _assert_matches_scan(rollup: _SummaryRollup, records: list[OpRecord]) -> None:
    assert rollup.operations_dict() == _summarize_operations(records)
    assert rollup.namespaces_dict() == _summarize_by_namespace(records)


def test_rollup_add_replace_remove_matches_scan() -> None:
    rollup = _SummaryRollup()
    first = _op()
    final = first._replace(
        namespace="predict.precompute",
        flopscope_backend_duration_s=0.25,
        flopscope_overhead_duration_s=0.05,
    )

    rollup.apply_record(None, first)
    _assert_matches_scan(rollup, [first])

    rollup.apply_record(first, final)
    _assert_matches_scan(rollup, [final])

    rollup.apply_record(final, None)
    _assert_matches_scan(rollup, [])


def test_rollup_retains_zero_call_timing_delta_until_merge() -> None:
    staged = _op(backend=None, overhead=0.01)
    final = staged._replace(
        flopscope_backend_duration_s=0.25,
        flopscope_overhead_duration_s=0.05,
    )
    delta = _SummaryRollup()

    delta.apply_record(staged, final)

    expected_operation = {
        "flop_cost": 0,
        "calls": 0,
        "flopscope_backend_time_s": 0.25,
        "flopscope_overhead_time_s": 0.04,
    }
    assert delta.operations_dict() == {"add": expected_operation}
    namespace = delta.namespaces_dict()["predict"]
    assert namespace == {
        "flops_used": 0,
        "calls": 0,
        "flopscope_backend_time_s": 0.25,
        "flopscope_overhead_time_s": 0.04,
        "operations": {"add": expected_operation},
    }

    aggregate = _SummaryRollup()
    aggregate.apply_record(None, staged)
    aggregate.merge(delta)
    _assert_matches_scan(aggregate, [final])


def test_rollup_merge_does_not_alias_nested_buckets() -> None:
    left = _SummaryRollup()
    right = _SummaryRollup()
    right.apply_record(None, _op())
    left.merge(right)

    result = left.namespaces_dict()
    result["predict"]["operations"]["add"]["calls"] = 999

    assert left.namespaces_dict()["predict"]["operations"]["add"]["calls"] == 1


def test_rollup_negative_self_merge_clears_multiple_buckets() -> None:
    rollup = _SummaryRollup()
    rollup.apply_record(None, _op("add", namespace="predict"))
    rollup.apply_record(
        None,
        _op(
            "multiply",
            cost=7,
            namespace="train",
            backend=0.2,
            overhead=0.03,
        ),
    )

    rollup.merge(rollup, sign=-1)

    assert rollup.operations_dict() == {}
    assert rollup.namespaces_dict() == {}


def test_rollup_positive_self_merge_doubles_totals() -> None:
    records = [
        _op("add", namespace="predict", backend=0.1, overhead=0.01),
        _op(
            "multiply",
            cost=7,
            namespace="train",
            backend=0.2,
            overhead=0.02,
        ),
    ]
    rollup = _SummaryRollup()
    for record in records:
        rollup.apply_record(None, record)

    rollup.merge(rollup)

    _assert_matches_scan(rollup, records * 2)


def test_rollup_copy_is_independent_and_clear_empties_copy() -> None:
    record = _op(backend=0.1, overhead=0.01)
    original = _SummaryRollup()
    original.apply_record(None, record)
    copied = original.copy()

    copied.clear()

    assert copied.operations_dict() == {}
    assert copied.namespaces_dict() == {}
    _assert_matches_scan(original, [record])


def test_rollup_merge_keeps_source_and_destination_independent() -> None:
    first = _op(backend=0.1, overhead=0.01)
    second = _op("multiply", cost=7, namespace="train")
    source = _SummaryRollup()
    source.apply_record(None, first)
    destination = _SummaryRollup()
    destination.merge(source)

    source.apply_record(None, second)
    _assert_matches_scan(destination, [first])

    destination.apply_record(first, None)
    _assert_matches_scan(source, [first, second])


def test_context_rollup_tracks_staged_record_replacement() -> None:
    from flopscope._budget import BudgetContext

    ctx = BudgetContext(100, quiet=True)
    with ctx:
        timer = ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
        before = ctx._summary_rollup.operations_dict()
        with timer:
            pass
        after = ctx._summary_rollup.operations_dict()

    assert before["add"]["calls"] == 1
    assert after == _summarize_operations(ctx.op_log)
    assert after["add"]["flopscope_overhead_time_s"] >= 0.0


def test_public_op_log_keeps_existing_backing_list_contract() -> None:
    from flopscope._budget import BudgetContext

    ctx = BudgetContext(100, quiet=True)
    assert ctx.op_log is ctx._op_log


def test_context_summary_does_not_iterate_op_log() -> None:
    from flopscope._budget import BudgetContext

    with BudgetContext(100, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
    ctx._op_log = _PoisonedHistory(ctx._op_log)
    assert ctx.summary_dict(by_namespace=True)["flops_used"] == 5


def test_context_summary_returns_deep_defensive_copies() -> None:
    from flopscope._budget import BudgetContext

    with BudgetContext(100, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
    first = ctx.summary_dict(by_namespace=True)
    expected = deepcopy(first)
    first["operations"]["add"]["calls"] = 99
    first["by_namespace"][None]["operations"].clear()
    assert ctx.summary_dict(by_namespace=True) == expected


def test_context_live_reads_reuse_rollup_mapping_but_advance_wall(monkeypatch) -> None:
    from flopscope._budget import BudgetContext, _SummaryRollup

    operations_calls = 0
    namespaces_calls = 0
    original_operations = _SummaryRollup.operations_dict
    original_namespaces = _SummaryRollup.namespaces_dict

    def counted_operations(self):
        nonlocal operations_calls
        operations_calls += 1
        return original_operations(self)

    def counted_namespaces(self):
        nonlocal namespaces_calls
        namespaces_calls += 1
        return original_namespaces(self)

    monkeypatch.setattr(_SummaryRollup, "operations_dict", counted_operations)
    monkeypatch.setattr(_SummaryRollup, "namespaces_dict", counted_namespaces)
    with BudgetContext(100, quiet=True) as ctx:
        first_flat = ctx.summary_dict()
        second_flat = ctx.summary_dict()
        first_namespaced = ctx.summary_dict(by_namespace=True)
        second_namespaced = ctx.summary_dict(by_namespace=True)
        assert (operations_calls, namespaces_calls) == (2, 1)

        timer = ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
        staged_flat = ctx.summary_dict()
        staged_namespaced = ctx.summary_dict(by_namespace=True)
        assert (operations_calls, namespaces_calls) == (4, 2)
        assert staged_flat["operations"]["add"]["calls"] == 1
        assert staged_namespaced["by_namespace"][None]["calls"] == 1

        with timer:
            pass
        final_flat = ctx.summary_dict()
        final_namespaced = ctx.summary_dict(by_namespace=True)
        assert (operations_calls, namespaces_calls) == (6, 3)
        assert final_flat["operations"] == _summarize_operations(ctx.op_log)
        assert final_namespaced["operations"] == _summarize_operations(ctx.op_log)
        assert final_namespaced["by_namespace"] == _summarize_by_namespace(ctx.op_log)

    assert second_flat["wall_time_s"] > first_flat["wall_time_s"]
    assert (
        second_flat["flopscope_overhead_time_s"]
        > first_flat["flopscope_overhead_time_s"]
    )
    assert second_namespaced["wall_time_s"] > first_namespaced["wall_time_s"]
    assert (
        second_namespaced["flopscope_overhead_time_s"]
        > first_namespaced["flopscope_overhead_time_s"]
    )


def test_cross_thread_summary_waits_for_context_snapshot_lock() -> None:
    import threading

    from flopscope._budget import BudgetContext

    ctx = BudgetContext(100, quiet=True)
    started = threading.Event()
    finished = threading.Event()

    def inspect() -> None:
        started.set()
        ctx.summary_dict(by_namespace=True)
        finished.set()

    with ctx._summary_lock:
        thread = threading.Thread(target=inspect)
        thread.start()
        assert started.wait(timeout=1.0)
        assert not finished.wait(timeout=0.05)
    thread.join(timeout=1.0)
    assert finished.is_set()


def test_live_wall_is_sampled_inside_context_snapshot_lock(monkeypatch) -> None:
    import threading

    import flopscope._budget as budget_module
    from flopscope._budget import BudgetContext

    ctx = BudgetContext(100, quiet=True)
    ctx._start_time = 0.0
    meter_started = threading.Event()
    wall_sampled = threading.Event()
    release_clock = threading.Event()
    reader_finished = threading.Event()
    clock_lock = threading.Lock()
    clock_phase = [1.0]
    clock_calls = 0
    summaries: list[dict] = []
    errors: list[BaseException] = []

    def fake_perf_counter() -> float:
        nonlocal clock_calls
        with clock_lock:
            clock_calls += 1
            call = clock_calls
        if call == 1:
            meter_started.set()
            return 0.0
        if call == 2:
            wall_sampled.set()
            assert release_clock.wait(timeout=5.0)
            return clock_phase[0]
        raise AssertionError(f"unexpected clock call {call}")

    monkeypatch.setattr(budget_module.time, "perf_counter", fake_perf_counter)

    def inspect() -> None:
        try:
            summaries.append(ctx.summary_dict())
        except BaseException as exc:
            errors.append(exc)
        finally:
            reader_finished.set()

    reader = threading.Thread(target=inspect)
    try:
        with ctx._summary_lock:
            reader.start()
            assert meter_started.wait(timeout=5.0)
            assert not wall_sampled.wait(timeout=0.05), (
                "live wall sampled before acquiring the context snapshot lock"
            )
            ctx._add_flopscope_backend(5.0)
            clock_phase[0] = 10.0
            release_clock.set()
    finally:
        release_clock.set()
        if reader.ident is not None:
            reader.join(timeout=5.0)

    assert not reader.is_alive()
    assert reader_finished.is_set()
    assert not errors
    summary = summaries[0]
    assert wall_sampled.is_set()
    wall = summary["wall_time_s"]
    assert wall is not None
    measured = (
        summary["flopscope_backend_time_s"]
        + summary["flopscope_overhead_time_s"]
        + summary["residual_wall_time_s"]
    )
    assert abs(wall - measured) <= 1e-12


def test_timing_only_activity_advances_summary_generation() -> None:
    from flopscope._budget import BudgetContext

    ctx = BudgetContext(100, quiet=True)
    generation = ctx._summary_generation
    rollup_generation = ctx._rollup_generation
    ctx._add_flopscope_overhead(0.25)
    assert ctx._summary_generation == generation + 1
    assert ctx._rollup_generation == rollup_generation


def test_summary_overhead_is_not_retroactive() -> None:
    from flopscope._budget import BudgetContext

    with BudgetContext(100, quiet=True) as ctx:
        before = ctx.flopscope_overhead_time_s
        first = ctx.summary_dict()
        after_first = ctx.flopscope_overhead_time_s
        second = ctx.summary_dict()
    assert first["flopscope_overhead_time_s"] == before
    assert after_first > before
    assert second["flopscope_overhead_time_s"] >= after_first


def test_plain_text_formatting_is_measured_as_summary_overhead() -> None:
    from flopscope._budget import BudgetContext
    from flopscope._display import _plain_text_summary

    with BudgetContext(100, quiet=True) as ctx:
        before = ctx.flopscope_overhead_time_s
        _plain_text_summary()
        assert ctx.flopscope_overhead_time_s > before


def test_summary_imports_are_measured_as_summary_overhead(monkeypatch) -> None:
    import builtins

    import flopscope._budget as budget_module
    from flopscope._budget import BudgetContext
    from flopscope._display import _rich_summary

    import_depths: dict[str, list[int]] = {
        "flopscope._display": [],
        "rich.console": [],
        "rich.panel": [],
    }
    original_import = builtins.__import__

    def recording_import(name: str, *args, **kwargs):
        if name in import_depths:
            import_depths[name].append(
                getattr(budget_module._thread_local, "summary_overhead_depth", 0)
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", recording_import)

    with BudgetContext(100, quiet=True) as ctx:
        ctx.summary()
        _rich_summary()

    zero_depth_imports = {name for name, depths in import_depths.items() if 0 in depths}
    assert all(import_depths.values())
    assert not zero_depth_imports


def test_cross_context_summary_avoids_lock_inversion(monkeypatch) -> None:
    import threading

    import flopscope._budget as budget_module
    from flopscope._budget import BudgetContext

    first = BudgetContext(100, quiet=True)
    second = BudgetContext(100, quiet=True)
    first_lock = threading.Lock()
    second_lock = threading.Lock()
    first._summary_lock = first_lock  # type: ignore[assignment]
    second._summary_lock = second_lock  # type: ignore[assignment]
    contexts_entered = threading.Barrier(2)
    both_snapshot_locks_held = threading.Event()
    snapshot_barrier = threading.Barrier(2, action=both_snapshot_locks_held.set)
    summaries: list[dict] = []
    errors: list[BaseException] = []
    original_materialize = budget_module._SummaryRollup.operations_dict

    def synchronized_materialize(self: _SummaryRollup) -> dict[str, dict]:
        snapshot_barrier.wait(timeout=5.0)
        return original_materialize(self)

    monkeypatch.setattr(
        budget_module._SummaryRollup,
        "operations_dict",
        synchronized_materialize,
    )

    def cross_inspect(active: BudgetContext, inspected: BudgetContext) -> None:
        try:
            with active:
                contexts_entered.wait(timeout=5.0)
                summaries.append(inspected.summary_dict())
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(
            target=cross_inspect,
            args=(first, second),
            name="inspect-second",
            daemon=True,
        ),
        threading.Thread(
            target=cross_inspect,
            args=(second, first),
            name="inspect-first",
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    alive: list[str] = []
    try:
        assert both_snapshot_locks_held.wait(timeout=5.0)
        for thread in threads:
            thread.join(timeout=0.5)
        alive = [thread.name for thread in threads if thread.is_alive()]
    finally:
        contexts_entered.abort()
        snapshot_barrier.abort()
        if any(thread.is_alive() for thread in threads) and first_lock.locked():
            first_lock.release()
        for thread in threads:
            thread.join(timeout=5.0)

    assert not alive
    assert not errors
    assert len(summaries) == 2


def test_summary_waits_for_complete_op_timing_transition(monkeypatch) -> None:
    import threading

    from flopscope._budget import BudgetContext

    ctx = BudgetContext(100, quiet=True)
    timing_totals_updated = threading.Event()
    release_writer = threading.Event()
    reader_started = threading.Event()
    reader_finished = threading.Event()
    writer_errors: list[BaseException] = []
    original_replace = ctx._replace_op_record

    def paused_replace(index: int, record: OpRecord) -> None:
        if record.flopscope_backend_duration_s is not None:
            timing_totals_updated.set()
            assert release_writer.wait(timeout=5.0)
        original_replace(index, record)

    monkeypatch.setattr(ctx, "_replace_op_record", paused_replace)

    def write_timing() -> None:
        try:
            timer = ctx.deduct(
                "add", flop_cost=5, subscripts=None, shapes=(), dtypes=()
            )
            with timer:
                pass
        except BaseException as exc:
            writer_errors.append(exc)

    def inspect() -> None:
        reader_started.set()
        ctx.summary_dict()
        reader_finished.set()

    writer = threading.Thread(target=write_timing)
    reader = threading.Thread(target=inspect)
    writer.start()
    try:
        assert timing_totals_updated.wait(timeout=5.0)
        reader.start()
        assert reader_started.wait(timeout=5.0)
        assert not reader_finished.wait(timeout=0.05)
    finally:
        release_writer.set()
        writer.join(timeout=5.0)
        if reader.ident is not None:
            reader.join(timeout=5.0)

    assert not writer_errors
    assert not writer.is_alive()
    assert not reader.is_alive()
    assert reader_finished.is_set()


def test_session_merge_consumes_only_unrecorded_rollup(monkeypatch) -> None:
    import flopscope._budget as budget_module
    from flopscope._budget import BudgetAccumulator, BudgetContext

    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)
    ctx = BudgetContext(100, quiet=True)
    with ctx:
        ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
        accumulator.record(ctx)
        ctx.deduct("add", flop_cost=7, subscripts=None, shapes=(), dtypes=())
    result = accumulator.snapshot(by_namespace=True)
    assert result["flops_used"] == 12
    assert result["operations"]["add"]["flop_cost"] == 12
    assert result["operations"]["add"]["calls"] == 2
    assert [record.flops_used for record in accumulator._records] == [5, 7]


def test_inflight_op_completion_merges_zero_call_timing_delta(monkeypatch) -> None:
    import flopscope._budget as budget_module
    from flopscope._budget import BudgetAccumulator, BudgetContext

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)

    ctx = BudgetContext(100, quiet=True, namespace="train")
    with ctx:
        timer = ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
        accumulator.record(ctx)
        first_generation = accumulator._generation
        first_rollup_generation = accumulator._rollup_generation
        accumulator._closed_snapshot_cache[False] = (first_generation, {})
        accumulator._stable_rollup_cache[()] = ({}, None)

        with timer:
            timer._backend_duration_s = 2.0
            now[0] = 3.0

        accumulator.record(ctx)
        result = accumulator.snapshot(by_namespace=True)

    operation = result["operations"]["add"]
    namespace = result["by_namespace"]["train"]
    assert result["flops_used"] == 5
    assert operation["calls"] == 1
    assert operation["flop_cost"] == 5
    assert operation["flopscope_backend_time_s"] == 2.0
    assert operation["flopscope_overhead_time_s"] == 1.0
    assert namespace["calls"] == 1
    assert namespace["flops_used"] == 5
    assert namespace["flopscope_backend_time_s"] == 2.0
    assert namespace["flopscope_overhead_time_s"] == 1.0
    assert namespace["operations"]["add"] == operation
    assert result["flopscope_backend_time_s"] == 2.0
    assert result["flopscope_overhead_time_s"] == 1.0
    assert (
        sum(
            record.total_flopscope_backend_time or 0.0
            for record in accumulator._records
        )
        == result["flopscope_backend_time_s"]
    )
    assert (
        sum(
            record.total_flopscope_overhead_time or 0.0
            for record in accumulator._records
        )
        == result["flopscope_overhead_time_s"]
    )
    diagnostic_ops = [
        operation for record in accumulator._records for operation in record.op_log
    ]
    assert _summarize_operations(diagnostic_ops) == result["operations"]
    assert _summarize_by_namespace(diagnostic_ops) == result["by_namespace"]
    assert accumulator._generation == first_generation + 2
    assert accumulator._rollup_generation == first_rollup_generation + 1
    assert accumulator._closed_snapshot_cache == {}
    assert accumulator._stable_rollup_cache == {}


def test_reset_drops_inflight_diagnostic_provenance_but_keeps_later_timing(
    monkeypatch,
) -> None:
    import flopscope._budget as budget_module
    from flopscope._budget import BudgetAccumulator, BudgetContext

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)

    ctx = BudgetContext(100, quiet=True, namespace="train")
    with ctx:
        timer = ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
        accumulator.record(ctx)
        detached_record = accumulator._records[0]
        now[0] = 1.0
        budget_module.budget_reset()

        with timer:
            timer._backend_duration_s = 2.0
            now[0] = 4.0

    result = accumulator.snapshot(by_namespace=True)
    operation = result["operations"]["add"]
    namespace = result["by_namespace"]["train"]
    assert result["flops_used"] == 0
    assert result["wall_time_s"] == 3.0
    assert result["flopscope_backend_time_s"] == 2.0
    assert result["flopscope_overhead_time_s"] == 1.0
    assert operation["calls"] == 0
    assert operation["flop_cost"] == 0
    assert operation["flopscope_backend_time_s"] == 2.0
    assert operation["flopscope_overhead_time_s"] == 1.0
    assert namespace["calls"] == 0
    assert namespace["flops_used"] == 0
    assert namespace["operations"]["add"] == operation
    assert [
        operation.flopscope_backend_duration_s for operation in detached_record.op_log
    ] == [None]
    assert all(not record.op_log for record in accumulator._records)
    assert accumulator.get_data(by_namespace=True) == result
    assert budget_module.budget_summary_dict(by_namespace=True) == result


def test_diagnostic_provenance_tracks_only_unfinished_operations(monkeypatch) -> None:
    import flopscope._budget as budget_module
    from flopscope._budget import BudgetAccumulator, BudgetContext

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)

    ctx = BudgetContext(100, quiet=True)
    with ctx:
        for _ in range(10):
            with ctx.deduct("add", flop_cost=1, subscripts=None, shapes=(), dtypes=()):
                pass
        accumulator.record(ctx)
        assert ctx not in accumulator._diagnostic_op_locations

        timer = ctx.deduct("add", flop_cost=1, subscripts=None, shapes=(), dtypes=())
        unfinished_index = len(ctx.op_log) - 1
        accumulator.record(ctx)
        assert set(accumulator._diagnostic_op_locations[ctx]) == {unfinished_index}

        with timer:
            pass
        accumulator.record(ctx)
        assert ctx not in accumulator._diagnostic_op_locations


def test_timing_only_delta_is_unrecorded_activity() -> None:
    from flopscope._budget import BudgetContext

    ctx = BudgetContext(100, quiet=True)
    ctx._add_flopscope_overhead(0.1)
    assert ctx._has_unrecorded_activity()


def test_mark_recorded_rebases_unrecorded_rollup_to_new_mutations() -> None:
    from flopscope._budget import BudgetContext

    ctx = BudgetContext(100, quiet=True)
    ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())

    ctx._mark_recorded()

    assert ctx._unrecorded_rollup.operations_dict() == {}
    assert ctx._unrecorded_rollup.namespaces_dict() == {}
    assert ctx._recorded_summary_generation == ctx._summary_generation

    ctx.deduct("multiply", flop_cost=7, subscripts=None, shapes=(), dtypes=())

    new_records = ctx.op_log[ctx._recorded_op_count :]
    assert [record.op_name for record in new_records] == ["multiply"]
    _assert_matches_scan(ctx._unrecorded_rollup, new_records)


def test_mark_reset_baseline_rebases_unrecorded_rollup_to_new_mutations() -> None:
    from flopscope._budget import BudgetContext

    ctx = BudgetContext(100, quiet=True)
    ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())

    ctx._mark_reset_baseline()

    assert ctx._unrecorded_rollup.operations_dict() == {}
    assert ctx._unrecorded_rollup.namespaces_dict() == {}
    assert ctx._recorded_summary_generation == ctx._summary_generation

    ctx.deduct("multiply", flop_cost=7, subscripts=None, shapes=(), dtypes=())

    new_records = ctx.op_log[ctx._recorded_op_count :]
    assert [record.op_name for record in new_records] == ["multiply"]
    _assert_matches_scan(ctx._unrecorded_rollup, new_records)
