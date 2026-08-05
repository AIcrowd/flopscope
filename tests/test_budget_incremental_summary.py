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
