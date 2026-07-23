"""Benchmark polynomial operations."""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_flops
from flopscope._polynomial import polyfit_cost

POLYNOMIAL_OPS: list[str] = [
    "polyval",
    "polyfit",
    "polyadd",
    "polysub",
    "polymul",
    "polydiv",
    "polyder",
    "polyint",
    "poly",
    "roots",
]

_FORMULA_STRINGS: dict[str, str] = {
    "polyval": "2 * n * degree (FMA=2)",
    "polyfit": "n * degree (Vandermonde build) + lstsq_cost(n, degree+1, ncols=1)",
    "roots": "10*degree^3 (provisional, eigvals_cost)",
    "polymul": "2*(degree+1)^2 - 2*(degree+1) (FMA=2)",
    "polydiv": "1 + Q*(2*n2+1), Q=max(n1-n2+1,0) (quotient length)",
    "polyadd": "degree + 1",
    "polysub": "degree + 1",
    "polyder": "degree (t*n - t*(t+1)//2, t=min(m,n-1); m=1 default gives n-1=degree)",
    "polyint": "degree + 1 (m*n + m*(m-1)//2; m=1 default gives n=degree+1)",
    "poly": "2*degree^2",
}


def _analytical_cost(op: str, n: int, degree: int) -> int:
    """Return the analytical FLOP cost for a polynomial operation.

    These formulas match flopscope's runtime cost model so that the
    benchmark denominator and the budget deduction use the same formula.
    """
    if op == "polyval":
        return (
            2 * n * degree
        )  # Updated for FMA=2 unification (spec 2026-05-20): polyval formula doubled m*deg → 2*m*deg.
    elif op == "polyfit":
        # y is a 1-D array of length n in this benchmark (see benchmark_polynomial
        # below) -> ncols=1. Delegates to the real billing formula (polyfit_cost
        # now prices the Vandermonde build + lstsq solve, matching linalg.lstsq's
        # own cost for the identical shape) so this benchmark's denominator can
        # never drift from what flopscope actually charges.
        return polyfit_cost(n, degree, ncols=1)
    elif op == "roots":
        return 10 * degree**3
    elif op == "polymul":
        n = degree + 1
        return max(2 * n * n - n - n, 1)
    elif op == "polydiv":
        n = degree + 1
        q = max(n - n + 1, 0)  # equal-length benchmark: n1=n2=degree+1, Q=1
        return max(1 + q * (2 * n + 1), 1)
    elif op in ("polyadd", "polysub"):
        return degree + 1
    elif op == "polyder":
        # m=1 default: t=min(1, n-1)=1, cost=n-1=degree (n=degree+1)
        return max(degree, 1)
    elif op == "polyint":
        # m=1 default: m*n + m*(m-1)//2 = n = degree+1
        return degree + 1
    elif op == "poly":
        return 2 * degree**2
    else:
        raise ValueError(f"Unknown polynomial op: {op!r}")


def benchmark_polynomial(
    n: int = 1_000_000,
    dtype: str = "float64",
    repeats: int = 10,
    degree: int = 100,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Benchmark polynomial ops, returning raw measurement per element.

    Each op is normalized by its analytical FLOP cost from
    ``_analytical_cost(op, n, degree)`` so the returned value
    represents raw perf-counter FLOPs per analytical FLOP.

    Parameters
    ----------
    n : int
        Array size for polyval/polyfit.
    dtype : str
        NumPy dtype string.
    repeats : int
        Number of repetitions per measurement.
    degree : int
        Polynomial degree (higher = less overhead-dominated for coeff ops).

    Returns
    -------
    tuple[dict[str, float], dict[str, dict]]
        ``(alphas, details)`` where *alphas* maps op name to median alpha
        and *details* maps op name to a dict of per-op measurement metadata.
    """
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    # 3 distributions with varying coefficient magnitudes
    coeff_setups = [
        f"c = rng.standard_normal({degree + 1}).astype(np.{dtype})",
        f"c = (rng.standard_normal({degree + 1}) * 100).astype(np.{dtype})",
        f"c = (rng.standard_normal({degree + 1}) * 0.01).astype(np.{dtype})",
    ]

    for op in POLYNOMIAL_OPS:
        dist_values: list[float] = []
        perf_instructions: list[int] = []

        for ci, c_setup in enumerate(coeff_setups):
            seed = 42 + ci
            base_setup = (
                f"import numpy as np; rng = np.random.default_rng({seed}); {c_setup}"
            )

            if op == "polyval":
                setup = (
                    base_setup + f"; x = rng.standard_normal({n}).astype(np.{dtype})"
                )
                bench = "np.polyval(c, x)"
            elif op == "polyfit":
                setup = (
                    base_setup
                    + f"; x = np.linspace(-1, 1, {n}).astype(np.{dtype})"
                    + f"; y = np.polyval(c, x) + rng.standard_normal({n}).astype(np.{dtype}) * 0.01"
                )
                bench = f"np.polyfit(x, y, {degree})"
            elif op == "poly":
                setup = (
                    base_setup
                    + f"; r = rng.standard_normal({degree}).astype(np.{dtype})"
                )
                bench = "np.poly(r)"
            elif op == "roots":
                setup = base_setup
                bench = "np.roots(c)"
            elif op in ("polyadd", "polysub"):
                setup = (
                    base_setup
                    + f"; d = rng.standard_normal({degree + 1}).astype(np.{dtype})"
                )
                bench = f"np.{op}(c, d)"
            elif op in ("polymul", "polydiv"):
                setup = (
                    base_setup
                    + f"; d = rng.standard_normal({degree + 1}).astype(np.{dtype})"
                )
                bench = f"np.{op}(c, d)"
            elif op == "polyder":
                setup = base_setup
                bench = "np.polyder(c)"
            elif op == "polyint":
                setup = base_setup
                bench = "np.polyint(c)"
            else:
                setup = base_setup
                bench = f"np.{op}(c)"

            try:
                result = measure_flops(setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            analytical = _analytical_cost(op, n, degree)
            perf_instructions.append(result.total_flops)
            dist_values.append(result.total_flops / (analytical * repeats))

        if dist_values:
            results[op] = statistics.median(dist_values)
            # Build explicit benchmark_size per op
            if op == "polyval":
                bm_size = f"c: ({degree + 1},), x: ({n},)"
            elif op == "polyfit":
                bm_size = f"x: ({n},), y: ({n},), degree={degree}"
            elif op in ("polymul", "polydiv"):
                bm_size = f"c: ({degree + 1},), d: ({degree + 1},)"
            elif op in ("polyadd", "polysub"):
                bm_size = f"c: ({degree + 1},), d: ({degree + 1},)"
            elif op in ("polyder", "polyint"):
                bm_size = f"c: ({degree + 1},)"
            elif op == "poly":
                bm_size = f"r: ({degree},)"
            elif op == "roots":
                bm_size = f"c: ({degree + 1},)"
            else:
                bm_size = f"n={n}, degree={degree}"
            details[op] = {
                "category": "counted_custom",
                "measurement_mode": "custom",
                "analytical_formula": _FORMULA_STRINGS.get(op, "n"),
                "analytical_flops": analytical,
                "benchmark_size": bm_size,
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": perf_instructions,
                "distribution_alphas": dist_values,
            }

    return results, details
