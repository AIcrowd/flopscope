from __future__ import annotations

from flopscope._budget import (
    OpRecord,
    _summarize_by_namespace,
    _summarize_operations,
    _SummaryRollup,
)


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
        timer = ctx.deduct(
            "add", flop_cost=5, subscripts=None, shapes=(), dtypes=()
        )
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

    zero_depth_imports = {
        name for name, depths in import_depths.items() if 0 in depths
    }
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
    original_summarize = budget_module._summarize_operations

    def synchronized_summarize(op_log: list[OpRecord]) -> dict[str, dict]:
        snapshot_barrier.wait(timeout=5.0)
        return original_summarize(op_log)

    monkeypatch.setattr(
        budget_module, "_summarize_operations", synchronized_summarize
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
