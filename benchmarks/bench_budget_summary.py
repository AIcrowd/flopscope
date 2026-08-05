from __future__ import annotations

import statistics
import time

import flopscope as flops
from flopscope._budget import (
    BudgetContext,
    OpRecord,
    _accumulator,
    _reset_global_default,
    _SummaryRollup,
)


class _PoisonedHistory(list):
    def __iter__(self):
        raise AssertionError("budget summary iterated historical records")

    def __getitem__(self, index):
        raise AssertionError(f"budget summary indexed historical records at {index!r}")


def _build_history(calls: int) -> None:
    flops.budget_reset()
    with _accumulator._lock:
        _accumulator._records = []
    with BudgetContext(calls + 1, namespace="bench", quiet=True) as ctx:
        for index in range(calls):
            ctx.deduct(
                "add" if index % 2 == 0 else "mul",
                flop_cost=1,
                subscripts=None,
                shapes=(),
                dtypes=(),
            )


def _poison_history() -> None:
    with _accumulator._lock:
        records = [
            record._replace(op_log=_PoisonedHistory(record.op_log))
            for record in list.__iter__(_accumulator._records)
        ]
        _accumulator._records = _PoisonedHistory(records)


def _median_cold_snapshot_latency(samples: int = 25) -> float:
    timings = []
    for _ in range(samples):
        with _accumulator._lock:
            _accumulator._closed_snapshot_cache.clear()
        started = time.perf_counter()
        flops.budget_summary_dict(by_namespace=True)
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def _cardinality_latency(cardinality: int) -> float:
    flops.budget_reset()
    rollup = _SummaryRollup()
    for index in range(cardinality):
        rollup.apply_record(
            None,
            OpRecord(
                op_name=f"op_{index}",
                subscripts=None,
                shapes=(),
                flop_cost=1,
                cumulative=index + 1,
                namespace=f"namespace_{index}",
            ),
        )
    with _accumulator._lock:
        _accumulator._records = []
        _accumulator._rollup = rollup
        _accumulator._flop_budget = cardinality + 1
        _accumulator._flops_used = cardinality
        _accumulator._invalidate_caches(rollup_changed=True)
    flops.budget_summary_dict(by_namespace=True)
    return _median_cold_snapshot_latency()


def main() -> None:
    try:
        results = {}
        for calls in (1_000, 100_000):
            _build_history(calls)
            _poison_history()
            flops.budget_summary_dict(by_namespace=True)
            results[calls] = _median_cold_snapshot_latency()
        ratio = results[100_000] / max(results[1_000], 1e-12)
        print(
            f"cold_snapshot_1k={results[1_000]:.6f}s "
            f"cold_snapshot_100k={results[100_000]:.6f}s ratio={ratio:.2f}x"
        )
        assert ratio < 5.0, (
            f"fixed-cardinality summary latency scaled with history: ratio={ratio:.2f}x"
        )
        for cardinality in (10, 100, 1_000):
            elapsed = _cardinality_latency(cardinality)
            print(
                f"cardinality={cardinality} "
                f"diagnostic_cold_snapshot_copy={elapsed:.6f}s"
            )
    finally:
        with _accumulator._lock:
            _accumulator._records = []
        try:
            flops.budget_reset()
        finally:
            _reset_global_default()


if __name__ == "__main__":
    main()
