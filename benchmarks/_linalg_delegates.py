"""Benchmark linalg namespace delegate operations.

These 15 ops live under ``numpy.linalg.*`` and typically delegate to a
primary operation (matmul, SVD, solve, ...).  We benchmark them directly
with perf counters to capture any wrapper overhead.
"""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_flops

LINALG_DELEGATE_OPS: list[str] = [
    "linalg.cond",
    "linalg.cross",
    "linalg.matmul",
    "linalg.matrix_norm",
    "linalg.matrix_power",
    "linalg.matrix_rank",
    "linalg.multi_dot",
    "linalg.norm",
    "linalg.outer",
    "linalg.tensordot",
    "linalg.tensorinv",
    "linalg.tensorsolve",
    "linalg.trace",
    "linalg.vecdot",
    "linalg.vector_norm",
]

_FORMULA_STRINGS: dict[str, str] = {
    "linalg.cond": "m*n*min(m,n)",
    "linalg.cross": "6*n",
    "linalg.matmul": "MNK",
    "linalg.matrix_norm": "numel",
    "linalg.matrix_power": "(ceil(log2(k))+popcount(k)-1)*n^3",
    "linalg.matrix_rank": "m*n*min(m,n)",
    "linalg.multi_dot": "sum of chain MNK costs",
    "linalg.norm": "numel",
    "linalg.outer": "M*N",
    "linalg.tensordot": "product of free * contracted dims",
    "linalg.tensorinv": "n^3",
    "linalg.tensorsolve": "n^3",
    "linalg.trace": "min(m,n)",
    "linalg.vecdot": "batch*K",
    "linalg.vector_norm": "numel",
}

# NumPy 2.x-only ops — skip gracefully on older versions.
_NUMPY2_OPS = {
    "linalg.cross",
    "linalg.matrix_norm",
    "linalg.vector_norm",
    "linalg.outer",
    "linalg.vecdot",
    "linalg.matmul",
    "linalg.tensordot",
}


def _analytical_cost(op_name: str) -> int:
    """Return the analytical FLOP count for *op_name* at the canonical size.

    Each op has a fixed benchmark size (see the table in the module docstring).
    This function returns the textbook cost for that size.

    Parameters
    ----------
    op_name : str
        Fully-qualified operation name, e.g. ``"linalg.cond"``.

    Returns
    -------
    int
        Analytical FLOP count.
    """
    short = op_name.split(".")[-1]
    costs: dict[str, int] = {
        "cond": 512 * 512 * 512,  # m*n*min(m,n) via SVD
        "cross": 6 * 1_000_000,  # 6*n
        "matmul": 2 * 512 * 512 * 512
        - 512 * 512,  # 2*M*N*K - M*N (FMA=2); = 268,173,312
        "matrix_norm": 2 * 512 * 512,  # 2*numel (Frobenius)
        "matrix_power": 3 * 64**3,  # 3 matmuls for n=5
        "matrix_rank": 512 * 512 * 512,  # m*n*min(m,n) via SVD
        "multi_dot": 128 * 64 * 128
        + 128
        * 128
        * 64,  # optimal chain (FMA=2); coincidentally same as FMA=1 value = 2,097,152
        "norm": 2 * 10_000_000,  # 2*numel (FMA=2, vector L2)
        "outer": 5000 * 5000,  # M*N
        "tensordot": 64
        ** 5,  # d^5 (FMA=2 textbook; matches flopscope charge = 1,073,741,824)
        "tensorinv": 64**3,  # n^3 after reshape
        "tensorsolve": 64**3,  # n^3 after reshape
        "trace": 10_000,  # min(m,n)
        "vecdot": 1000 * 512,  # batch*K
        "vector_norm": 2 * 10_000_000,  # 2*numel (FMA=2)
    }
    return costs[short]


# ---------------------------------------------------------------------------
# Per-op setup / bench code builders
# ---------------------------------------------------------------------------


def _op_config(op: str, dtype: str) -> tuple[list[str], str, str]:
    """Return (setups, bench_code, benchmark_size) for a delegate op.

    Each op gets 3 setup variants (distributions) to take the median over.

    Returns
    -------
    tuple[list[str], str, str]
        (list of setup strings, benchmark expression, human-readable size)
    """
    short = op.split(".")[-1]
    d = dtype

    if short == "cond":
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((512, 512)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.uniform(0.1, 1.0, (512, 512)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((512, 512)).astype(np.{d}) + "
            f"512 * np.eye(512, dtype=np.{d})",
        ]
        return setups, "np.linalg.cond(A)", "A: (512,512)"

    if short == "cross":
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"a = rng.standard_normal((1000000, 3)).astype(np.{d}); "
            f"b = rng.standard_normal((1000000, 3)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"a = rng.uniform(-1, 1, (1000000, 3)).astype(np.{d}); "
            f"b = rng.uniform(-1, 1, (1000000, 3)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"a = rng.standard_normal((1000000, 3)).astype(np.{d}) * 100; "
            f"b = rng.standard_normal((1000000, 3)).astype(np.{d}) * 0.01",
        ]
        return setups, "np.linalg.cross(a, b)", "a: (1000000,3), b: (1000000,3)"

    if short == "matmul":
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((512, 512)).astype(np.{d}); "
            f"B = rng.standard_normal((512, 512)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.uniform(0.1, 1.0, (512, 512)).astype(np.{d}); "
            f"B = rng.uniform(0.1, 1.0, (512, 512)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((512, 512)).astype(np.{d}) * 100; "
            f"B = rng.standard_normal((512, 512)).astype(np.{d}) * 0.01",
        ]
        return setups, "np.linalg.matmul(A, B)", "A: (512,512), B: (512,512)"

    if short == "matrix_norm":
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((512, 512)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.uniform(0.1, 1.0, (512, 512)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((512, 512)).astype(np.{d}) * 100",
        ]
        return setups, "np.linalg.matrix_norm(A)", "A: (512,512)"

    if short == "matrix_power":
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((64, 64)).astype(np.{d}) + "
            f"64 * np.eye(64, dtype=np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.uniform(0.1, 1.0, (64, 64)).astype(np.{d}) + "
            f"64 * np.eye(64, dtype=np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((64, 64)).astype(np.{d}) + "
            f"640 * np.eye(64, dtype=np.{d})",
        ]
        return setups, "np.linalg.matrix_power(A, 5)", "A: (64,64), n=5"

    if short == "matrix_rank":
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((512, 512)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.uniform(0.1, 1.0, (512, 512)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((512, 512)).astype(np.{d}) + "
            f"512 * np.eye(512, dtype=np.{d})",
        ]
        return setups, "np.linalg.matrix_rank(A)", "A: (512,512)"

    if short == "multi_dot":
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((128, 64)).astype(np.{d}); "
            f"B = rng.standard_normal((64, 128)).astype(np.{d}); "
            f"C = rng.standard_normal((128, 64)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.uniform(0.1, 1.0, (128, 64)).astype(np.{d}); "
            f"B = rng.uniform(0.1, 1.0, (64, 128)).astype(np.{d}); "
            f"C = rng.uniform(0.1, 1.0, (128, 64)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((128, 64)).astype(np.{d}) * 100; "
            f"B = rng.standard_normal((64, 128)).astype(np.{d}) * 0.01; "
            f"C = rng.standard_normal((128, 64)).astype(np.{d})",
        ]
        return (
            setups,
            "np.linalg.multi_dot([A, B, C])",
            "A: (128,64), B: (64,128), C: (128,64)",
        )

    if short == "norm":
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"x = rng.standard_normal(10000000).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"x = rng.uniform(0.1, 1.0, 10000000).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"x = rng.standard_normal(10000000).astype(np.{d}) * 100",
        ]
        return setups, "np.linalg.norm(x)", "x: (10000000,)"

    if short == "outer":
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"a = rng.standard_normal(5000).astype(np.{d}); "
            f"b = rng.standard_normal(5000).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"a = rng.uniform(0.1, 1.0, 5000).astype(np.{d}); "
            f"b = rng.uniform(0.1, 1.0, 5000).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"a = rng.standard_normal(5000).astype(np.{d}) * 100; "
            f"b = rng.standard_normal(5000).astype(np.{d}) * 0.01",
        ]
        return setups, "np.linalg.outer(a, b)", "a: (5000,), b: (5000,)"

    if short == "tensordot":
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((64, 64, 64)).astype(np.{d}); "
            f"B = rng.standard_normal((64, 64, 64)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.uniform(0.1, 1.0, (64, 64, 64)).astype(np.{d}); "
            f"B = rng.uniform(0.1, 1.0, (64, 64, 64)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((64, 64, 64)).astype(np.{d}) * 100; "
            f"B = rng.standard_normal((64, 64, 64)).astype(np.{d}) * 0.01",
        ]
        return (
            setups,
            "np.linalg.tensordot(A, B, axes=1)",
            "A: (64,64,64), B: (64,64,64)",
        )

    if short == "tensorinv":
        # Build an invertible (64,64) matrix via A@A.T + n*I, then reshape
        # to (8,8,8,8).
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"_A = rng.standard_normal((64, 64)).astype(np.{d}); "
            f"_M = _A @ _A.T + 64 * np.eye(64, dtype=np.{d}); "
            f"A = _M.reshape(8, 8, 8, 8)",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"_A = rng.uniform(0.1, 1.0, (64, 64)).astype(np.{d}); "
            f"_M = _A @ _A.T + 64 * np.eye(64, dtype=np.{d}); "
            f"A = _M.reshape(8, 8, 8, 8)",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"_A = rng.standard_normal((64, 64)).astype(np.{d}); "
            f"_M = _A @ _A.T + 640 * np.eye(64, dtype=np.{d}); "
            f"A = _M.reshape(8, 8, 8, 8)",
        ]
        return setups, "np.linalg.tensorinv(A, ind=2)", "A: (8,8,8,8)"

    if short == "tensorsolve":
        # Build a solvable system: invertible (64,64) reshaped to (8,8,8,8),
        # with b of shape (8,8).
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"_A = rng.standard_normal((64, 64)).astype(np.{d}); "
            f"_M = _A @ _A.T + 64 * np.eye(64, dtype=np.{d}); "
            f"A = _M.reshape(8, 8, 8, 8); "
            f"b = rng.standard_normal((8, 8)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"_A = rng.uniform(0.1, 1.0, (64, 64)).astype(np.{d}); "
            f"_M = _A @ _A.T + 64 * np.eye(64, dtype=np.{d}); "
            f"A = _M.reshape(8, 8, 8, 8); "
            f"b = rng.uniform(0.1, 1.0, (8, 8)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"_A = rng.standard_normal((64, 64)).astype(np.{d}); "
            f"_M = _A @ _A.T + 640 * np.eye(64, dtype=np.{d}); "
            f"A = _M.reshape(8, 8, 8, 8); "
            f"b = rng.standard_normal((8, 8)).astype(np.{d})",
        ]
        return (
            setups,
            "np.linalg.tensorsolve(A, b)",
            "A: (8,8,8,8), b: (8,8)",
        )

    if short == "trace":
        # Use np.ones instead of random arrays to avoid the setup's random
        # number generation dominating the measurement. Trace just sums the
        # diagonal — the values don't affect the FP instruction count.
        setups = [
            f"import numpy as np; A = np.ones((10000, 10000), dtype=np.{d})",
            f"import numpy as np; A = np.ones((10000, 10000), dtype=np.{d}) * 2.5",
            f"import numpy as np; A = np.ones((10000, 10000), dtype=np.{d}) * 100",
        ]
        return setups, "np.linalg.trace(A)", "A: (10000,10000)"

    if short == "vecdot":
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((1000, 512)).astype(np.{d}); "
            f"B = rng.standard_normal((1000, 512)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.uniform(0.1, 1.0, (1000, 512)).astype(np.{d}); "
            f"B = rng.uniform(0.1, 1.0, (1000, 512)).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"A = rng.standard_normal((1000, 512)).astype(np.{d}) * 100; "
            f"B = rng.standard_normal((1000, 512)).astype(np.{d}) * 0.01",
        ]
        return setups, "np.linalg.vecdot(A, B)", "A: (1000,512), B: (1000,512)"

    if short == "vector_norm":
        setups = [
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"x = rng.standard_normal(10000000).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"x = rng.uniform(0.1, 1.0, 10000000).astype(np.{d})",
            f"import numpy as np; rng = np.random.default_rng(42); "
            f"x = rng.standard_normal(10000000).astype(np.{d}) * 100",
        ]
        return setups, "np.linalg.vector_norm(x)", "x: (10000000,)"

    raise ValueError(f"Unknown delegate op: {op}")


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------


def benchmark_linalg_delegates(
    dtype: str = "float64",
    repeats: int = 10,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Benchmark linalg delegate ops via perf counters.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict]]
        (alphas, details) — same schema as ``benchmark_linalg``.
    """
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    for op in LINALG_DELEGATE_OPS:
        # Skip ops that don't exist in this NumPy version.
        if op in _NUMPY2_OPS:
            try:
                import numpy as np  # noqa: F811

                fn = np.linalg
                for part in op.split(".")[1:]:
                    fn = getattr(fn, part)
            except AttributeError:
                continue

        setups, bench, bm_size = _op_config(op, dtype)
        analytical = _analytical_cost(op)

        dist_values: list[float] = []
        dist_raw_totals: list[int] = []

        for setup in setups:
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
                "benchmark_size": bm_size,
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    return results, details
