from __future__ import annotations

from copy import deepcopy

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from flopscope._budget import (
    OpRecord,
    _summarize_by_namespace,
    _summarize_operations,
    _SummaryRollup,
    _timing_summary,
)


class _PoisonedHistory(list):
    def __iter__(self):
        raise AssertionError("summary traversed historical operation records")

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError("summary sliced historical operation records")
        return super().__getitem__(index)


class _ScriptedClock:
    def __init__(self, now: float) -> None:
        self.now = now
        self._scripted = False
        self._values: list[float] = []

    def __call__(self) -> float:
        if not self._scripted:
            return self.now
        if not self._values:
            raise AssertionError("accessor made an unexpected perf_counter call")
        self.now = self._values.pop(0)
        return self.now

    def advance(self, duration: float) -> None:
        assert not self._scripted
        self.now += duration

    def begin_accessor(self, *, sample_calls: int, overhead: float | None) -> None:
        assert not self._scripted
        started = self.now
        self._values = [started] * (1 + sample_calls)
        if overhead is not None:
            self._values.append(started + overhead)
        self._scripted = True

    def end_accessor(self) -> None:
        was_scripted = self._scripted
        remaining = tuple(self._values)
        self._values.clear()
        self._scripted = False
        assert was_scripted
        assert not remaining, (
            f"accessor skipped {len(remaining)} expected perf_counter call(s)"
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


def test_scripted_clock_recovers_after_incomplete_accessor() -> None:
    clock = _ScriptedClock(10.0)
    clock.begin_accessor(sample_calls=1, overhead=0.5)
    clock()

    with pytest.raises(AssertionError, match="skipped 2 expected"):
        clock.end_accessor()

    assert not clock._scripted
    assert clock._values == []
    clock.advance(1.0)
    assert clock() == 11.0


def _assert_matches_scan(rollup: _SummaryRollup, records: list[OpRecord]) -> None:
    _assert_bucket_mapping_matches(
        rollup.operations_dict(), _summarize_operations(records)
    )
    _assert_bucket_mapping_matches(
        rollup.namespaces_dict(), _summarize_by_namespace(records)
    )


def _merge_bucket_mapping(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            _merge_bucket_mapping(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0) + value


def _scan_namespace_records(
    records,
    by_namespace: bool,
    *,
    orphan_operations: dict | None = None,
    orphan_namespaces: dict | None = None,
) -> dict:
    total_budget = 0
    total_used = 0
    total_wall: float | None = None
    total_backend = 0.0
    total_overhead = 0.0
    operations: dict = {}
    namespaces: dict = {}
    for record in records:
        total_budget += record.flop_budget
        total_used += record.flops_used
        record_operations = _summarize_operations(record.op_log)
        record_namespaces = _summarize_by_namespace(record.op_log)
        _merge_bucket_mapping(operations, record_operations)
        _merge_bucket_mapping(namespaces, record_namespaces)
        if record.wall_time_s is not None:
            total_wall = (total_wall or 0.0) + record.wall_time_s
        total_backend += record.total_flopscope_backend_time or 0.0
        total_overhead += record.total_flopscope_overhead_time or 0.0
    if orphan_operations is not None:
        _merge_bucket_mapping(operations, orphan_operations)
    if orphan_namespaces is not None:
        _merge_bucket_mapping(namespaces, orphan_namespaces)
    wall, backend, overhead, residual = _timing_summary(
        total_wall, total_backend, total_overhead
    )
    result = {
        "flop_budget": total_budget,
        "flops_used": total_used,
        "flops_remaining": total_budget - total_used,
        "operations": operations,
        "wall_time_s": wall,
        "flopscope_backend_time_s": backend,
        "flopscope_overhead_time_s": overhead,
        "residual_wall_time_s": residual,
    }
    if by_namespace:
        result["by_namespace"] = namespaces
    return result


def _assert_bucket_mapping_matches(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], dict):
            _assert_bucket_mapping_matches(actual[key], expected[key])
        elif isinstance(actual[key], float):
            assert actual[key] == pytest.approx(expected[key], abs=1e-9)
        else:
            assert actual[key] == expected[key]


def _scan_context(ctx, by_namespace: bool, *, now: float | None = None) -> dict:
    wall = ctx.wall_time_s
    if wall is None and ctx._start_time is not None:
        wall = ctx.elapsed_s if now is None else now - ctx._start_time
    wall, backend, overhead, residual = _timing_summary(
        wall,
        ctx._total_flopscope_backend_time,
        ctx._total_flopscope_overhead_time,
    )
    result = {
        "flop_budget": ctx.flop_budget,
        "flops_used": ctx.flops_used,
        "flops_remaining": ctx.flops_remaining,
        "operations": _summarize_operations(ctx.op_log),
        "wall_time_s": wall,
        "flopscope_backend_time_s": backend,
        "flopscope_overhead_time_s": overhead,
        "residual_wall_time_s": residual,
    }
    if by_namespace:
        result["by_namespace"] = _summarize_by_namespace(ctx.op_log)
    return result


def _scan_global(
    by_namespace: bool,
    *,
    orphan_operations: dict | None = None,
    orphan_namespaces: dict | None = None,
) -> dict:
    import flopscope._budget as budget_module

    records = list(budget_module._accumulator._records)
    active = budget_module.get_active_budget()
    global_default = budget_module._global_default
    if global_default is not None and global_default._has_unrecorded_activity():
        records.append(global_default._snapshot_record())
    if active is not None and active is not global_default:
        # Active explicit contexts contribute their budget even before their
        # first operation, so presence cannot be gated on recorded activity.
        records.append(active._snapshot_record())
    return _scan_namespace_records(
        records,
        by_namespace,
        orphan_operations=orphan_operations,
        orphan_namespaces=orphan_namespaces,
    )


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["add", "mul", "einsum"]),
            st.integers(min_value=0, max_value=10_000),
            st.one_of(st.none(), st.sampled_from(["a", "a.b", "x"])),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
        max_size=100,
    )
)
def test_rollup_matches_scan_for_generated_records(specs) -> None:
    records = [
        _op(name, cost=cost, namespace=namespace, backend=backend, overhead=overhead)
        for name, cost, namespace, backend, overhead in specs
    ]
    rollup = _SummaryRollup()
    for record in records:
        rollup.apply_record(None, record)
    _assert_matches_scan(rollup, records)


@settings(max_examples=200, deadline=None)
@example(events=["reenter"], by_namespace=False)
@example(events=["close", "reenter"], by_namespace=True)
@example(events=["default", "reenter"], by_namespace=True)
@example(events=["inflight", "reset", "inflight"], by_namespace=True)
@example(events=["inflight", "reset", "reset", "inflight"], by_namespace=True)
@example(events=["inflight_backend_reset", "inflight"], by_namespace=True)
@example(
    events=["inflight_backend", "reset", "inflight_backend", "inflight"],
    by_namespace=True,
)
@given(
    st.lists(
        st.sampled_from(
            [
                "immediate",
                "deferred",
                "exception",
                "namespace",
                "inflight",
                "inflight_backend",
                "inflight_backend_reset",
                "close",
                "reenter",
                "reset",
                "default",
            ]
        ),
        min_size=1,
        max_size=30,
    ),
    st.booleans(),
)
def test_generated_context_and_session_transitions_match_scan(
    events, by_namespace
) -> None:
    import flopscope as flops
    import flopscope._budget as budget_module

    budget_module._reset_global_default()
    flops.budget_reset()
    context_accessor_overhead = 0.01
    global_accessor_overhead = 0.02
    clock = _ScriptedClock(100.0)
    real_perf_counter = budget_module.time.perf_counter
    budget_module.time.perf_counter = clock
    ctx = budget_module.BudgetContext(1_000_000, namespace="explicit", quiet=True)
    ctx.__enter__()
    entered = True
    inflight = None
    inflight_segment_started_at: float | None = None
    inflight_backend_baseline: float | None = None
    inflight_overhead_baseline: float | None = None
    inflight_usercode_baseline: float | None = None
    inflight_expected_backend = 0.0
    inflight_namespace: str | None = None
    inflight_reset_epoch: int | None = None
    reset_epoch = 0
    orphan_operations: dict = {}
    orphan_namespaces: dict = {}

    def ensure_entered() -> None:
        nonlocal entered
        if not entered:
            ctx.__enter__()
            entered = True

    def finish_inflight() -> None:
        nonlocal inflight, inflight_namespace, inflight_reset_epoch
        nonlocal inflight_segment_started_at, inflight_backend_baseline
        nonlocal inflight_overhead_baseline, inflight_usercode_baseline
        nonlocal inflight_expected_backend
        if inflight is not None:
            assert inflight_reset_epoch is not None
            assert inflight_segment_started_at is not None
            assert inflight_backend_baseline is not None
            assert inflight_overhead_baseline is not None
            assert inflight_usercode_baseline is not None
            backend_delta = inflight_expected_backend
            nested_delta = (
                ctx._total_flopscope_backend_time - inflight_backend_baseline
            ) + (ctx._total_flopscope_overhead_time - inflight_overhead_baseline)
            usercode_delta = ctx._total_user_code_time - inflight_usercode_baseline
            overhead_delta = max(
                clock.now
                - inflight_segment_started_at
                - backend_delta
                - nested_delta
                - usercode_delta,
                0.0,
            )
            inflight.__exit__(None, None, None)
            if inflight_reset_epoch != reset_epoch:
                operation_delta = {
                    "inflight_add": {
                        "flop_cost": 0,
                        "calls": 0,
                        "flopscope_backend_time_s": backend_delta,
                        "flopscope_overhead_time_s": overhead_delta,
                    }
                }
                namespace_delta = {
                    inflight_namespace: {
                        "flops_used": 0,
                        "calls": 0,
                        "flopscope_backend_time_s": backend_delta,
                        "flopscope_overhead_time_s": overhead_delta,
                        "operations": operation_delta,
                    }
                }
                _merge_bucket_mapping(orphan_operations, operation_delta)
                _merge_bucket_mapping(orphan_namespaces, namespace_delta)
            inflight = None
            inflight_segment_started_at = None
            inflight_backend_baseline = None
            inflight_overhead_baseline = None
            inflight_usercode_baseline = None
            inflight_expected_backend = 0.0
            inflight_namespace = None
            inflight_reset_epoch = None

    def begin_inflight() -> None:
        nonlocal inflight, inflight_namespace, inflight_reset_epoch
        nonlocal inflight_segment_started_at, inflight_backend_baseline
        nonlocal inflight_overhead_baseline, inflight_usercode_baseline
        nonlocal inflight_expected_backend
        assert inflight is None
        inflight = ctx.deduct(
            "inflight_add",
            flop_cost=6,
            subscripts=None,
            shapes=(),
            dtypes=(),
        )
        inflight.__enter__()
        inflight_segment_started_at = clock.now
        inflight_backend_baseline = ctx._total_flopscope_backend_time
        inflight_overhead_baseline = ctx._total_flopscope_overhead_time
        inflight_usercode_baseline = ctx._total_user_code_time
        inflight_expected_backend = 0.0
        inflight_namespace = ctx.namespace
        inflight_reset_epoch = reset_epoch

    def reset_epoch_boundary() -> None:
        nonlocal reset_epoch, inflight_segment_started_at
        nonlocal inflight_backend_baseline, inflight_overhead_baseline
        nonlocal inflight_usercode_baseline, inflight_expected_backend
        flops.budget_reset()
        reset_epoch += 1
        orphan_operations.clear()
        orphan_namespaces.clear()
        if inflight is not None:
            inflight_segment_started_at = clock.now
            inflight_backend_baseline = ctx._total_flopscope_backend_time
            inflight_overhead_baseline = ctx._total_flopscope_overhead_time
            inflight_usercode_baseline = ctx._total_user_code_time
            inflight_expected_backend = 0.0

    try:
        for event in events:
            clock.advance(0.001)
            if event == "close" and entered:
                finish_inflight()
                ctx.__exit__(None, None, None)
                entered = False
            elif event == "reenter":
                ensure_entered()
            elif event == "reset":
                reset_epoch_boundary()
            elif event == "default":
                finish_inflight()
                if entered:
                    ctx.__exit__(None, None, None)
                    entered = False
                default = budget_module._get_global_default()
                with default.deduct(
                    "default_add",
                    flop_cost=1,
                    subscripts=None,
                    shapes=(),
                    dtypes=(),
                ):
                    pass
            else:
                ensure_entered()
                if event == "immediate":
                    with ctx.deduct(
                        "add", flop_cost=2, subscripts=None, shapes=(), dtypes=()
                    ):
                        pass
                elif event == "deferred":
                    with ctx.deduct_after(
                        "take", subscripts=None, shapes=(), dtypes=()
                    ) as timer:
                        timer.set_cost(3)
                elif event == "exception":
                    with pytest.raises(RuntimeError):
                        with ctx.deduct(
                            "mul",
                            flop_cost=4,
                            subscripts=None,
                            shapes=(),
                            dtypes=(),
                        ):
                            raise RuntimeError("expected")
                elif event == "namespace":
                    with flops.namespace("nested"):
                        with ctx.deduct(
                            "sum",
                            flop_cost=5,
                            subscripts=None,
                            shapes=(),
                            dtypes=(),
                        ):
                            pass
                elif event == "inflight":
                    if inflight is None:
                        begin_inflight()
                    else:
                        finish_inflight()
                elif event == "inflight_backend":
                    if inflight is None:
                        begin_inflight()
                    backend_duration = 0.003
                    budget_module._call_numpy(clock.advance, backend_duration)
                    inflight_expected_backend += backend_duration
                elif event == "inflight_backend_reset":
                    if inflight is None:
                        begin_inflight()

                    def backend_reset() -> None:
                        clock.advance(0.001)
                        reset_epoch_boundary()
                        clock.advance(0.002)

                    budget_module._call_numpy(backend_reset)
                    inflight_expected_backend += 0.002

            expected_context = _scan_context(ctx, by_namespace, now=clock.now)
            active_during_context_summary = budget_module.get_active_budget()
            active_overhead_before_context_summary = (
                active_during_context_summary._total_flopscope_overhead_time
                if active_during_context_summary is not None
                else None
            )
            context_sample_calls = int(
                ctx.wall_time_s is None and ctx._start_time is not None
            )
            clock.begin_accessor(
                sample_calls=context_sample_calls,
                overhead=(
                    context_accessor_overhead
                    if active_during_context_summary is not None
                    else None
                ),
            )
            try:
                actual_context = ctx.summary_dict(by_namespace=by_namespace)
            finally:
                clock.end_accessor()
            _assert_bucket_mapping_matches(actual_context, expected_context)
            if active_during_context_summary is not None:
                assert active_overhead_before_context_summary is not None
                assert active_during_context_summary._total_flopscope_overhead_time == (
                    pytest.approx(
                        active_overhead_before_context_summary
                        + context_accessor_overhead,
                        abs=1e-12,
                    )
                )
                if active_during_context_summary is ctx:
                    next_context = _scan_context(ctx, by_namespace, now=clock.now)
                    assert next_context["flopscope_overhead_time_s"] == pytest.approx(
                        expected_context["flopscope_overhead_time_s"]
                        + context_accessor_overhead,
                        abs=1e-12,
                    )

            expected_global = _scan_global(
                by_namespace,
                orphan_operations=orphan_operations,
                orphan_namespaces=orphan_namespaces,
            )
            active_during_global_summary = budget_module.get_active_budget()
            active_overhead_before_global_summary = (
                active_during_global_summary._total_flopscope_overhead_time
                if active_during_global_summary is not None
                else None
            )
            global_default = budget_module._global_default
            global_sample_calls = 0
            if (
                global_default is not None
                and global_default._has_unrecorded_activity()
                and global_default._wall_time_s is None
                and global_default._start_time is not None
            ):
                global_sample_calls += 1
            if (
                active_during_global_summary is not None
                and active_during_global_summary is not global_default
                and active_during_global_summary._wall_time_s is None
                and active_during_global_summary._start_time is not None
            ):
                global_sample_calls += 1
            clock.begin_accessor(
                sample_calls=global_sample_calls,
                overhead=(
                    global_accessor_overhead
                    if active_during_global_summary is not None
                    else None
                ),
            )
            try:
                actual_global = flops.budget_summary_dict(by_namespace=by_namespace)
            finally:
                clock.end_accessor()
            _assert_bucket_mapping_matches(actual_global, expected_global)
            if active_during_global_summary is not None:
                assert active_overhead_before_global_summary is not None
                assert active_during_global_summary._total_flopscope_overhead_time == (
                    pytest.approx(
                        active_overhead_before_global_summary
                        + global_accessor_overhead,
                        abs=1e-12,
                    )
                )
                next_global = _scan_global(
                    by_namespace,
                    orphan_operations=orphan_operations,
                    orphan_namespaces=orphan_namespaces,
                )
                assert next_global["flopscope_overhead_time_s"] == pytest.approx(
                    expected_global["flopscope_overhead_time_s"]
                    + global_accessor_overhead,
                    abs=1e-12,
                )
    finally:
        try:
            try:
                finish_inflight()
            finally:
                if entered:
                    ctx.__exit__(None, None, None)
        finally:
            budget_module.time.perf_counter = real_perf_counter
            try:
                try:
                    budget_module._reset_global_default()
                finally:
                    flops.budget_reset()
            finally:
                budget_module._thread_local.active_budget = None


def test_global_accessor_overhead_is_deferred_until_next_public_snapshot(
    monkeypatch,
) -> None:
    import flopscope as flops
    import flopscope._budget as budget_module

    accessor_overhead = 0.02
    clock = _ScriptedClock(100.0)
    monkeypatch.setattr(budget_module.time, "perf_counter", clock)

    def public_snapshot() -> dict:
        clock.begin_accessor(sample_calls=1, overhead=accessor_overhead)
        try:
            return flops.budget_summary_dict()
        finally:
            clock.end_accessor()

    with budget_module.BudgetContext(100, quiet=True) as ctx:
        before = ctx.flopscope_overhead_time_s
        first = public_snapshot()
        after_first = ctx.flopscope_overhead_time_s
        second = public_snapshot()
        after_second = ctx.flopscope_overhead_time_s

    assert first["flopscope_overhead_time_s"] == before
    assert after_first == pytest.approx(before + accessor_overhead, abs=1e-12)
    assert second["flopscope_overhead_time_s"] == pytest.approx(after_first, abs=1e-12)
    assert after_second == pytest.approx(
        second["flopscope_overhead_time_s"] + accessor_overhead,
        abs=1e-12,
    )


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


def test_global_summary_does_not_iterate_records_or_live_op_log() -> None:
    import flopscope as flops
    import flopscope._budget as budget_module
    from flopscope._budget import BudgetContext

    with BudgetContext(100, quiet=True) as closed:
        closed.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
    records = budget_module._accumulator._records
    budget_module._accumulator._records = _PoisonedHistory(records)
    try:
        with BudgetContext(100, quiet=True) as live:
            live.deduct("multiply", flop_cost=7, subscripts=None, shapes=(), dtypes=())
            diagnostic_log = live._op_log
            live._op_log = _PoisonedHistory(diagnostic_log)
            try:
                summary = flops.budget_summary_dict(by_namespace=True)
            finally:
                live._op_log = diagnostic_log
    finally:
        budget_module._accumulator._records = records

    assert summary["flops_used"] == 12
    assert summary["operations"]["add"]["flop_cost"] == 5
    assert summary["operations"]["multiply"]["flop_cost"] == 7


def test_fresh_active_zero_op_summary_includes_budget_wall_and_next_overhead(
    monkeypatch,
) -> None:
    import flopscope as flops
    import flopscope._budget as budget_module
    from flopscope._budget import BudgetContext

    ticks = [0.0, 0.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]

    def fake_perf_counter() -> float:
        return ticks.pop(0) if ticks else 6.0

    monkeypatch.setattr(budget_module.time, "perf_counter", fake_perf_counter)
    with BudgetContext(100, quiet=True):
        first = flops.budget_summary_dict()
        second = flops.budget_summary_dict()

    assert first["flop_budget"] == 100
    assert first["flops_used"] == 0
    assert first["wall_time_s"] == 5.0
    assert first["flopscope_overhead_time_s"] == 0.0
    assert second["flop_budget"] == 100
    assert second["wall_time_s"] == 6.0
    assert second["flopscope_overhead_time_s"] == 1.0


def test_reentered_zero_op_summary_includes_current_invocation_wall(
    monkeypatch,
) -> None:
    import flopscope as flops
    import flopscope._budget as budget_module
    from flopscope._budget import BudgetContext

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    ctx = BudgetContext(100, quiet=True)
    with ctx:
        now[0] = 2.0

    now[0] = 10.0
    with ctx:
        now[0] = 13.0
        summary = flops.budget_summary_dict()

    assert summary["flop_budget"] == 100
    assert summary["wall_time_s"] == 5.0


def test_first_post_reset_active_summary_includes_budget_and_elapsed_delta(
    monkeypatch,
) -> None:
    import flopscope as flops
    import flopscope._budget as budget_module
    from flopscope._budget import BudgetContext

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    ctx = BudgetContext(100, quiet=True)
    with ctx:
        now[0] = 4.0
        flops.budget_reset()
        now[0] = 9.0
        first = flops.budget_summary_dict()
        now[0] = 10.0
        second = flops.budget_summary_dict()

    assert first["flop_budget"] == 100
    assert first["flops_used"] == 0
    assert first["wall_time_s"] == 5.0
    assert second["flop_budget"] == 100
    assert second["wall_time_s"] == 6.0


def test_closed_session_reuses_complete_canonical_snapshot(monkeypatch) -> None:
    import flopscope as flops
    from flopscope._budget import BudgetContext, _SummaryRollup

    with BudgetContext(100, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
    calls = 0
    original = _SummaryRollup.operations_dict

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(_SummaryRollup, "operations_dict", counted)
    first = flops.budget_summary_dict()
    second = flops.budget_summary_dict()
    assert first == second
    assert calls == 1


def test_closed_snapshot_cache_invalidates_after_record_and_reset(monkeypatch) -> None:
    import flopscope as flops
    from flopscope._budget import BudgetContext, _SummaryRollup

    with BudgetContext(100, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
    calls = 0
    original = _SummaryRollup.operations_dict

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(_SummaryRollup, "operations_dict", counted)
    assert flops.budget_summary_dict()["flops_used"] == 5
    assert flops.budget_summary_dict()["flops_used"] == 5
    assert calls == 1

    with BudgetContext(200, quiet=True) as ctx:
        ctx.deduct("multiply", flop_cost=7, subscripts=None, shapes=(), dtypes=())
    assert flops.budget_summary_dict()["flops_used"] == 12
    assert calls == 2

    flops.budget_reset()
    reset = flops.budget_summary_dict()
    assert reset["flop_budget"] == 0
    assert reset["flops_used"] == 0
    assert reset["operations"] == {}
    assert calls == 3


def test_live_global_summary_reuses_rollup_cache_until_operation_changes(
    monkeypatch,
) -> None:
    import flopscope as flops
    from flopscope._budget import BudgetContext, _SummaryRollup

    calls = 0
    original = _SummaryRollup.operations_dict

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(_SummaryRollup, "operations_dict", counted)
    with BudgetContext(100, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
        first = flops.budget_summary_dict()
        second = flops.budget_summary_dict()
        assert first["operations"] == second["operations"]
        assert calls == 1

        ctx.deduct("multiply", flop_cost=7, subscripts=None, shapes=(), dtypes=())
        third = flops.budget_summary_dict()
        assert third["flops_used"] == 12
        assert calls == 2


def test_global_summary_returns_deep_defensive_copies() -> None:
    import flopscope as flops
    from flopscope._budget import BudgetContext

    with BudgetContext(100, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=5, subscripts=None, shapes=(), dtypes=())
    first = flops.budget_summary_dict(by_namespace=True)
    expected = deepcopy(first)
    first["operations"]["add"]["calls"] = 99
    first["by_namespace"][None]["operations"].clear()
    assert flops.budget_summary_dict(by_namespace=True) == expected


def test_global_summary_overhead_is_visible_on_the_next_snapshot() -> None:
    import flopscope as flops
    from flopscope._budget import BudgetContext

    with BudgetContext(100, quiet=True) as ctx:
        before = ctx.flopscope_overhead_time_s
        first = flops.budget_summary_dict()
        after_first = ctx.flopscope_overhead_time_s
        second = flops.budget_summary_dict()

    assert first["flopscope_overhead_time_s"] == before
    assert after_first > before
    assert second["flopscope_overhead_time_s"] >= after_first


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


def test_external_summary_overhead_is_generation_tracked() -> None:
    from flopscope._budget import BudgetContext

    ctx = BudgetContext(100, quiet=True)
    generation = ctx._summary_generation
    ctx._record_external_flopscope_overhead(0.25)
    assert ctx.flopscope_overhead_time_s == pytest.approx(0.25)
    assert ctx._summary_generation > generation
    assert ctx._has_unrecorded_activity()


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
    stable = accumulator._stable_rollup_cache[
        (True, accumulator._rollup_generation, ())
    ]
    assert stable == (result["operations"], result["by_namespace"])


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


def test_reset_rebases_live_timer_and_nested_timing_to_the_new_epoch(
    monkeypatch,
) -> None:
    import flopscope._budget as budget_module
    from flopscope._budget import BudgetAccumulator, BudgetContext

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)

    ctx = BudgetContext(100, quiet=True, namespace="train")
    ctx.__enter__()
    outer = ctx.deduct("outer", flop_cost=5, subscripts=None, shapes=(), dtypes=())
    outer.__enter__()
    assert ctx._live_op_timers == {outer}

    now[0] = 1.0
    before_reset = ctx.deduct(
        "before_reset", flop_cost=2, subscripts=None, shapes=(), dtypes=()
    )
    before_reset.__enter__()
    assert ctx._live_op_timers == {outer, before_reset}
    before_reset._backend_duration_s = 1.0
    now[0] = 3.0
    before_reset.__exit__(None, None, None)
    assert ctx._live_op_timers == {outer}
    outer._backend_duration_s += 0.5

    now[0] = 4.0
    budget_module.budget_reset()
    assert ctx._live_op_timers == {outer}

    now[0] = 5.0
    after_reset = ctx.deduct(
        "after_reset", flop_cost=3, subscripts=None, shapes=(), dtypes=()
    )
    after_reset.__enter__()
    assert ctx._live_op_timers == {outer, after_reset}
    after_reset._backend_duration_s = 0.5
    now[0] = 7.0
    after_reset.__exit__(None, None, None)
    assert ctx._live_op_timers == {outer}
    outer._backend_duration_s += 0.25

    now[0] = 8.0
    outer.__exit__(None, None, None)
    assert ctx._live_op_timers == set()
    ctx.__exit__(None, None, None)

    result = accumulator.snapshot(by_namespace=True)
    assert result["wall_time_s"] == 4.0
    assert result["flopscope_backend_time_s"] == 0.75
    assert result["flopscope_overhead_time_s"] == 3.25
    assert result["residual_wall_time_s"] == 0.0
    assert result["wall_time_s"] == (
        result["flopscope_backend_time_s"]
        + result["flopscope_overhead_time_s"]
        + result["residual_wall_time_s"]
    )

    operations = result["operations"]
    assert "before_reset" not in operations
    assert operations["after_reset"] == {
        "flop_cost": 3,
        "calls": 1,
        "flopscope_backend_time_s": 0.5,
        "flopscope_overhead_time_s": 1.5,
    }
    assert operations["outer"] == {
        "flop_cost": 0,
        "calls": 0,
        "flopscope_backend_time_s": 0.25,
        "flopscope_overhead_time_s": 1.75,
    }
    namespace = result["by_namespace"]["train"]
    assert namespace["flops_used"] == 3
    assert namespace["calls"] == 1
    assert namespace["operations"] == operations


def test_reset_rebases_live_deferred_timer_to_the_new_epoch(monkeypatch) -> None:
    import flopscope._budget as budget_module
    from flopscope._budget import BudgetAccumulator, BudgetContext

    now = [0.0]
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: now[0])
    accumulator = BudgetAccumulator()
    monkeypatch.setattr(budget_module, "_accumulator", accumulator)

    ctx = BudgetContext(100, quiet=True, namespace="train")
    ctx.__enter__()
    timer = ctx.deduct_after("deferred", subscripts=None, shapes=(), dtypes=())
    timer.__enter__()
    timer.set_cost(4)
    timer._backend_duration_s = 0.75

    now[0] = 2.0
    budget_module.budget_reset()
    timer._backend_duration_s += 0.25
    now[0] = 3.0
    timer.__exit__(None, None, None)
    ctx.__exit__(None, None, None)

    result = accumulator.snapshot(by_namespace=True)
    assert result["wall_time_s"] == 1.0
    assert result["flopscope_backend_time_s"] == 0.25
    assert result["flopscope_overhead_time_s"] == 0.75
    assert result["residual_wall_time_s"] == 0.0
    assert result["operations"]["deferred"] == {
        "flop_cost": 4,
        "calls": 1,
        "flopscope_backend_time_s": 0.25,
        "flopscope_overhead_time_s": 0.75,
    }
    assert ctx._live_op_timers == set()


def test_live_timer_registry_is_cleaned_on_exceptional_exits() -> None:
    from flopscope._budget import BudgetContext
    from flopscope.errors import TimeExhaustedError

    ctx = BudgetContext(100, quiet=True)
    ctx.__enter__()

    missing_cost = ctx.deduct_after(
        "missing_cost", subscripts=None, shapes=(), dtypes=()
    )
    missing_cost.__enter__()
    with pytest.raises(RuntimeError, match=r"set_cost\(\) was never called"):
        missing_cost.__exit__(None, None, None)
    assert ctx._live_op_timers == set()
    assert ctx._current_op_timer is None

    block_error = ctx.deduct_after("block_error", subscripts=None, shapes=(), dtypes=())
    block_error.__enter__()
    assert block_error.__exit__(RuntimeError, RuntimeError("boom"), None) is False
    assert ctx._live_op_timers == set()
    assert ctx._current_op_timer is None

    deadline = ctx.deduct(
        "deadline", flop_cost=1, subscripts=None, shapes=(), dtypes=()
    )
    deadline.__enter__()
    ctx._deadline = float("-inf")
    ctx._wall_time_limit_s = 0.0
    with pytest.raises(TimeExhaustedError):
        deadline.__exit__(None, None, None)
    assert ctx._live_op_timers == set()
    assert ctx._current_op_timer is None

    ctx._deadline = None
    ctx.__exit__(None, None, None)


def test_reset_waits_for_live_timer_exit_accounting(monkeypatch) -> None:
    import threading

    import flopscope._budget as budget_module
    from flopscope._budget import BudgetContext

    ctx = BudgetContext(100, quiet=True)
    ctx.__enter__()
    timer = ctx.deduct("add", flop_cost=1, subscripts=None, shapes=(), dtypes=())
    timer.__enter__()

    exit_clock_entered = threading.Event()
    release_exit_clock = threading.Event()
    reset_started = threading.Event()
    reset_finished = threading.Event()
    errors: list[BaseException] = []

    def blocked_clock() -> float:
        exit_clock_entered.set()
        if not release_exit_clock.wait(timeout=5.0):
            raise AssertionError("timer exit clock was not released")
        return 1.0

    monkeypatch.setattr(budget_module.time, "perf_counter", blocked_clock)

    def exit_timer() -> None:
        try:
            timer.__exit__(None, None, None)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def reset_context() -> None:
        reset_started.set()
        try:
            ctx._mark_reset_baseline()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            reset_finished.set()

    exit_thread = threading.Thread(target=exit_timer)
    reset_thread = threading.Thread(target=reset_context)
    exit_thread.start()
    assert exit_clock_entered.wait(timeout=5.0)
    reset_thread.start()
    assert reset_started.wait(timeout=5.0)
    assert not reset_finished.wait(timeout=0.05)

    release_exit_clock.set()
    exit_thread.join(timeout=5.0)
    reset_thread.join(timeout=5.0)

    assert not errors
    assert not exit_thread.is_alive()
    assert not reset_thread.is_alive()
    assert ctx._live_op_timers == set()
    assert ctx._unrecorded_rollup.operations_dict() == {}
    assert ctx._recorded_summary_generation == ctx._summary_generation
    ctx.__exit__(None, None, None)


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
