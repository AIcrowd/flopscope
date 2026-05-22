"""Cold and warm-call latency benchmarks for einsum_accumulation_cost.

Run: `uv run python benchmarks/accumulation/bench_cost_compute.py`

Failure threshold: cold-call latency for the whole suite must stay below
COLD_CALL_BUDGET_SECONDS. This is the R7 risk gate from the spec — protects
against a regime implementation that's accidentally O(n²) on partition counts.
"""

from __future__ import annotations

import time

import numpy as np

import flopscope as fps

COLD_CALL_BUDGET_SECONDS = 0.5  # whole suite, no cache hits


CASES = [
    {
        "name": "matrix_chain_n3",
        "subscripts": "ij,jk",
        "output": "ik",
        "per_op_sym": (None, None),
        "sizes": {"i": 3, "j": 3, "k": 3},
    },
    {
        "name": "symmetric_matvec",
        "subscripts": "ij,j",
        "output": "i",
        "per_op_sym": ("symmetric", None),
        "sizes": {"i": 4, "j": 4},
    },
    {
        "name": "fully_symmetric_self_contract",
        "subscripts": "ijk,ijk",
        "output": "",
        "per_op_sym": ("symmetric", "symmetric"),
        "sizes": {"i": 4, "j": 4, "k": 4},
    },
    {
        "name": "triple_outer",
        "subscripts": "ia,ib,ic",
        "output": "abc",
        "per_op_sym": (None, None, None),
        "sizes": {"i": 4, "a": 4, "b": 4, "c": 4},
    },
    {
        "name": "cyclic_t_to_ab",
        "subscripts": "abc",
        "output": "ab",
        "per_op_sym": ({"type": "cyclic", "axes": [0, 1, 2]},),
        "sizes": {"a": 3, "b": 3, "c": 3},
    },
]


def _build_operands(case_def):
    parts = case_def["subscripts"].split(",")
    operands = []
    for op_idx, part in enumerate(parts):
        shape = tuple(case_def["sizes"][lbl] for lbl in part)
        op = np.zeros(shape) if shape else np.zeros(1)
        sym_decl = case_def["per_op_sym"][op_idx]
        if sym_decl == "symmetric":
            axes = tuple(range(len(part)))
            op = fps.as_symmetric(op, symmetry=axes)
        elif isinstance(sym_decl, dict) and sym_decl.get("type") == "cyclic":
            from flopscope._perm_group import SymmetryGroup

            axes = tuple(sym_decl.get("axes", range(len(part))))
            group = SymmetryGroup.cyclic(axes=axes)
            op = fps.as_symmetric(op, symmetry=group)
        operands.append(op)
    return operands


def bench_cold_call() -> dict[str, float]:
    """Time each case with a fresh cache."""
    timings = {}
    for case in CASES:
        from flopscope._einsum import _accumulation_cache, _path_cache

        _accumulation_cache.cache_clear()
        _path_cache.cache_clear()

        operands = _build_operands(case)
        subscripts = case["subscripts"] + "->" + case["output"]

        start = time.perf_counter()
        fps.einsum_accumulation_cost(subscripts, *operands)
        elapsed = time.perf_counter() - start
        timings[case["name"]] = elapsed
    return timings


def bench_warm_call() -> dict[str, float]:
    """Time each case after a warm-up call (cache hit)."""
    timings = {}
    for case in CASES:
        operands = _build_operands(case)
        subscripts = case["subscripts"] + "->" + case["output"]
        fps.einsum_accumulation_cost(subscripts, *operands)
        start = time.perf_counter()
        for _ in range(100):
            fps.einsum_accumulation_cost(subscripts, *operands)
        elapsed = (time.perf_counter() - start) / 100
        timings[case["name"]] = elapsed
    return timings


def main():
    cold = bench_cold_call()
    warm = bench_warm_call()
    print("Cold-call latency (seconds):")
    for name, t in cold.items():
        print(f"  {name:40s} {t * 1000:8.2f} ms")
    cold_total = sum(cold.values())
    print(f"  {'TOTAL':40s} {cold_total * 1000:8.2f} ms")

    print("\nWarm-call latency (averaged over 100 iterations):")
    for name, t in warm.items():
        print(f"  {name:40s} {t * 1e6:8.2f} µs")

    if cold_total > COLD_CALL_BUDGET_SECONDS:
        print(
            f"\nFAIL: cold-call total {cold_total:.3f}s exceeds budget {COLD_CALL_BUDGET_SECONDS}s"
        )
        raise SystemExit(1)
    print(
        f"\nOK: cold-call total {cold_total:.3f}s within budget {COLD_CALL_BUDGET_SECONDS}s"
    )


if __name__ == "__main__":
    main()
