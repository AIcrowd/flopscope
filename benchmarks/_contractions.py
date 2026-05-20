"""Benchmark BLAS contraction operations."""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_flops

CONTRACTION_OPS: list[str] = [
    "dot",
    "matmul",
    "inner",
    "vdot",
    "vecdot",
    "outer",
    "tensordot",
    "kron",
    "einsum",
]

_FORMULA_STRINGS: dict[str, str] = {
    "dot": "MNK",
    "matmul": "MNK",
    "inner": "N (a.size)",
    "vdot": "N (a.size)",
    "vecdot": "batch * K (output_size * contracted_axis)",
    "outer": "M*N",
    "tensordot": "product of free * contracted dims",
    "kron": "numel(output)",
    "einsum": "α/M model (FMA=2 textbook)",
}

_BENCHMARK_SIZE_STRINGS: dict[str, str] = {
    "dot": "A: (512,512), B: (512,512)",
    "matmul": "A: (512,512), B: (512,512)",
    "inner": "a: (1000000,), b: (1000000,)",
    "vdot": "a: (1000000,), b: (1000000,)",
    "vecdot": "A: (1000,512), B: (1000,512)",
    "outer": "a: (5000,), b: (5000,)",
    "tensordot": "A: (64,64,64), B: (64,64,64), axes=1",
    "kron": "A: (64,64), B: (64,64)",
    "einsum": "A: (512,512), B: (512,512), subscripts='ij,jk->ik'",
}


def _analytical_cost(op: str, **kwargs: int) -> int:
    """Return analytical FLOP count for the benchmark configuration.

    Parameters
    ----------
    op : str
        Operation name (e.g. ``"dot"``).
    **kwargs : int
        Shape parameters used by each formula.

    Returns
    -------
    int
        Analytical FLOP count.
    """
    costs: dict[str, int] = {
        # dot: 2D matrix multiply A(512,512) @ B(512,512), FMA=2 textbook
        "dot": 512 * 512 * 512,
        # matmul: identical to dot for 2D
        "matmul": 512 * 512 * 512,
        # inner: dot product of two 1M-element vectors.
        # Runtime charges a.size — matches flopscope's convention (FMA=2, but
        # a.size is pointwise-shaped so the FMA off-by-one doesn't apply here).
        "inner": 1_000_000,
        # vdot: same as inner for 1D real inputs.
        # Runtime charges a.size (FMA=2, pointwise-shaped — no off-by-one).
        "vdot": 1_000_000,
        # vecdot: batched dot product A(1000,512) . B(1000,512)
        # Output (1000,) with contracted axis 512.
        # Runtime charges result.size * contracted = 1000 * 512 (FMA=2 textbook).
        "vecdot": 1000 * 512,
        # outer: outer product of two 5000-element vectors
        "outer": 5000 * 5000,
        # tensordot: A(64,64,64) . B(64,64,64) axes=1 -> contract last of A with first of B
        "tensordot": 64**5,
        # kron: Kronecker product A(64,64) x B(64,64)
        "kron": 64**4,
        # einsum: 'ij,jk->ik' is matrix multiply (512,512)x(512,512), FMA=2 textbook
        "einsum": 512 * 512 * 512,
    }
    return costs[op]


def benchmark_contractions(
    dtype: str = "float64",
    repeats: int = 10,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Benchmark contraction ops, returning raw measurement per analytical FLOP.

    In perf mode this is actual FP ops / analytical FLOPs (correction factor).
    In timing mode this is nanoseconds / analytical FLOPs (same units as
    pointwise -- the runner normalizes against baseline to get relative weights).

    Parameters
    ----------
    dtype : str
        NumPy dtype string.
    repeats : int
        Number of repetitions per measurement.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict]]
        A pair of (alphas, details). ``alphas`` maps op name to median
        raw measurement per analytical FLOP. ``details`` maps op name to
        a dict of raw benchmark metadata.
    """
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    for op in CONTRACTION_OPS:
        dist_values: list[float] = []
        dist_raw_totals: list[int] = []

        # --- Build setups and bench code per op ---

        if op in ("dot", "matmul", "einsum"):
            # Two 512x512 matrices
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.standard_normal((512, 512)).astype(np.{dtype}); "
                    f"B = rng.standard_normal((512, 512)).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.uniform(0.01, 100, size=(512, 512)).astype(np.{dtype}); "
                    f"B = rng.uniform(0.01, 100, size=(512, 512)).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.uniform(-1000, 1000, size=(512, 512)).astype(np.{dtype}); "
                    f"B = rng.uniform(-1000, 1000, size=(512, 512)).astype(np.{dtype})"
                ),
            ]
            if op == "dot":
                bench = "np.dot(A, B)"
            elif op == "matmul":
                bench = "np.matmul(A, B)"
            else:  # einsum
                bench = "np.einsum('ij,jk->ik', A, B)"

        elif op in ("inner", "vdot"):
            # Two 1M-element vectors — large enough for BLAS ddot FMA to dominate
            # over per-call overhead (10K was too small, overhead inflated alpha)
            vec_n = 1_000_000
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"a = rng.standard_normal({vec_n}).astype(np.{dtype}); "
                    f"b = rng.standard_normal({vec_n}).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"a = rng.uniform(0.01, 100, size={vec_n}).astype(np.{dtype}); "
                    f"b = rng.uniform(0.01, 100, size={vec_n}).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"a = rng.uniform(-1000, 1000, size={vec_n}).astype(np.{dtype}); "
                    f"b = rng.uniform(-1000, 1000, size={vec_n}).astype(np.{dtype})"
                ),
            ]
            bench = f"np.{op}(a, b)"

        elif op == "vecdot":
            # Batched dot: A(1000,512), B(1000,512) -- NumPy 2.x only
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.standard_normal((1000, 512)).astype(np.{dtype}); "
                    f"B = rng.standard_normal((1000, 512)).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.uniform(0.01, 100, size=(1000, 512)).astype(np.{dtype}); "
                    f"B = rng.uniform(0.01, 100, size=(1000, 512)).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.uniform(-1000, 1000, size=(1000, 512)).astype(np.{dtype}); "
                    f"B = rng.uniform(-1000, 1000, size=(1000, 512)).astype(np.{dtype})"
                ),
            ]
            bench = "np.vecdot(A, B)"

        elif op == "outer":
            # Two 5000-element vectors
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"a = rng.standard_normal(5000).astype(np.{dtype}); "
                    f"b = rng.standard_normal(5000).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"a = rng.uniform(0.01, 100, size=5000).astype(np.{dtype}); "
                    f"b = rng.uniform(0.01, 100, size=5000).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"a = rng.uniform(-1000, 1000, size=5000).astype(np.{dtype}); "
                    f"b = rng.uniform(-1000, 1000, size=5000).astype(np.{dtype})"
                ),
            ]
            bench = "np.outer(a, b)"

        elif op == "tensordot":
            # Two (64,64,64) tensors, axes=1
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.standard_normal((64, 64, 64)).astype(np.{dtype}); "
                    f"B = rng.standard_normal((64, 64, 64)).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.uniform(0.01, 100, size=(64, 64, 64)).astype(np.{dtype}); "
                    f"B = rng.uniform(0.01, 100, size=(64, 64, 64)).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.uniform(-1000, 1000, size=(64, 64, 64)).astype(np.{dtype}); "
                    f"B = rng.uniform(-1000, 1000, size=(64, 64, 64)).astype(np.{dtype})"
                ),
            ]
            bench = "np.tensordot(A, B, axes=1)"

        elif op == "kron":
            # Two (64,64) matrices
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.standard_normal((64, 64)).astype(np.{dtype}); "
                    f"B = rng.standard_normal((64, 64)).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.uniform(0.01, 100, size=(64, 64)).astype(np.{dtype}); "
                    f"B = rng.uniform(0.01, 100, size=(64, 64)).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.uniform(-1000, 1000, size=(64, 64)).astype(np.{dtype}); "
                    f"B = rng.uniform(-1000, 1000, size=(64, 64)).astype(np.{dtype})"
                ),
            ]
            bench = "np.kron(A, B)"

        else:
            continue  # pragma: no cover

        analytical = _analytical_cost(op)

        for setup in setups:
            # For vecdot, wrap in try/except since it's NumPy 2.x only
            if op == "vecdot":
                try:
                    result = measure_flops(setup, bench, repeats=repeats)
                except (RuntimeError, AttributeError):
                    continue
            else:
                try:
                    result = measure_flops(setup, bench, repeats=repeats)
                except RuntimeError:
                    continue

            measured = result.total_flops / repeats
            dist_values.append(measured / analytical if analytical else 0.0)
            dist_raw_totals.append(result.total_flops)

        if dist_values:
            results[op] = statistics.median(dist_values)
            details[op] = {
                "category": "counted_custom",
                "measurement_mode": "blas",
                "analytical_formula": _FORMULA_STRINGS.get(op, ""),
                "analytical_flops": analytical,
                "benchmark_size": _BENCHMARK_SIZE_STRINGS.get(op, ""),
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    return results, details
