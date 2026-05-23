"""Benchmark miscellaneous custom-formula operations.

Each operation has its own analytical cost formula and benchmark setup.
Categories covered: element-wise comparison/conversion, differencing,
convolution/correlation, statistical, binning/histogram, interpolation,
and linear/generation ops.
"""

from __future__ import annotations

import math
import statistics

from benchmarks._perf import measure_flops

MISC_OPS: list[str] = [
    # Element-wise comparison/conversion
    "allclose",
    "array_equal",
    "array_equiv",
    "clip",
    # Differencing
    "diff",
    "ediff1d",
    "gradient",
    "unwrap",
    # Convolution/correlation
    "convolve",
    "correlate",
    # Statistical
    "corrcoef",
    "cov",
    "cross",
    # Binning/histogram
    "histogram",
    "histogram2d",
    "histogramdd",
    "histogram_bin_edges",
    "digitize",
    "bincount",
    # Interpolation
    "interp",
    # Linear/generation
    "trace",
    "trapezoid",
    "logspace",
    "geomspace",
    "vander",
]


_FORMULA_STRINGS: dict[str, str] = {
    "allclose": "n",
    "array_equal": "n",
    "array_equiv": "n",
    "clip": "n",
    "diff": "n",
    "ediff1d": "n",
    "gradient": "n",
    "unwrap": "7 * n",
    "convolve": "n * k",
    "correlate": "n * k",
    "corrcoef": "f^2 * s",
    "cov": "f^2 * s",
    "cross": "6 * n",
    "histogram": "n * ceil(log2(bins))",
    "histogram2d": "n * 2 * ceil(log2(bins))",
    "histogramdd": "n * ndim * ceil(log2(bins))",
    "histogram_bin_edges": "n",
    "digitize": "n * ceil(log2(bins))",
    "bincount": "n",
    "interp": "n * ceil(log2(xp))",
    "trace": "min(m, n)",
    "trapezoid": "n",
    "logspace": "n",
    "geomspace": "n",
    "vander": "n * (degree - 1)",
}


def _analytical_cost(op: str, **kwargs: int) -> int:
    """Return the analytical FLOP count for the benchmark configuration.

    Parameters
    ----------
    op : str
        Operation name.
    **kwargs
        Operation-specific size parameters: ``n``, ``k`` (kernel size),
        ``f`` (features), ``s`` (samples), ``bins``, ``xp`` (interp knots),
        ``degree``.

    Returns
    -------
    int
        Analytical FLOP count for the benchmark configuration.
    """
    n = kwargs.get("n", 10_000_000)

    # --- Element-wise comparison/conversion (cost = n) ---
    if op in ("allclose", "array_equal", "array_equiv", "clip"):
        return n

    # --- Differencing (cost = n) ---
    if op in ("diff", "ediff1d", "gradient"):
        return n

    # --- Phase unwrapping (~7 ufunc passes, issue #69) ---
    if op == "unwrap":
        return 7 * n

    # --- Convolution/correlation (cost = n * k) ---
    if op in ("convolve", "correlate"):
        k = kwargs.get("k", 1000)
        return n * k

    # --- Statistical ---
    if op in ("corrcoef", "cov"):
        f = kwargs.get("f", 1000)
        s = kwargs.get("s", 10000)
        return f * f * s

    if op == "cross":
        # 6 multiply-adds per cross product
        return 6 * n

    # --- Binning/histogram ---
    if op == "histogram":
        bins = kwargs.get("bins", 100)
        return n * math.ceil(math.log2(bins))

    if op == "histogram2d":
        bins = kwargs.get("bins", 100)
        return n * (math.ceil(math.log2(bins)) + math.ceil(math.log2(bins)))

    if op == "histogramdd":
        bins = kwargs.get("bins", 50)
        ndim = kwargs.get("ndim", 3)
        return n * ndim * math.ceil(math.log2(bins))

    if op == "histogram_bin_edges":
        return n

    if op == "digitize":
        bins = kwargs.get("bins", 100)
        return n * math.ceil(math.log2(bins))

    if op == "bincount":
        return n

    # --- Interpolation ---
    if op == "interp":
        xp = kwargs.get("xp", 10000)
        return n * math.ceil(math.log2(xp))

    # --- Linear/generation ---
    if op == "trace":
        return n  # cost = min(m, n); for square 1000x1000, n=1000

    if op == "trapezoid":
        return n

    if op in ("logspace", "geomspace"):
        return n

    if op == "vander":
        degree = kwargs.get("degree", 100)
        return n * (degree - 1)

    # Fallback
    return n


def _get_op_config(op: str, dtype: str) -> dict:
    """Return setups, bench code, analytical cost, and notes for an op.

    Returns
    -------
    dict with keys: ``setups`` (list[str]), ``bench`` (str),
    ``analytical`` (int).
    """
    n_large = 10_000_000
    rng_seed_a = 42
    rng_seed_b = 43

    def _three_setups_1arr(n: int, extra: str = "") -> list[str]:
        """Three distributions for a single array x."""
        base = f"import numpy as np; rng = np.random.default_rng({rng_seed_a})"
        return [
            f"{base}; x = rng.standard_normal({n}).astype(np.{dtype}){extra}",
            f"{base}; x = rng.uniform(0.01, 100, size={n}).astype(np.{dtype}){extra}",
            f"{base}; x = rng.uniform(-1000, 1000, size={n}).astype(np.{dtype}){extra}",
        ]

    def _three_setups_2arr(n: int, extra: str = "") -> list[str]:
        """Three distributions for two arrays a, b."""
        base = (
            f"import numpy as np; "
            f"rng_a = np.random.default_rng({rng_seed_a}); "
            f"rng_b = np.random.default_rng({rng_seed_b})"
        )
        return [
            (
                f"{base}; "
                f"a = rng_a.standard_normal({n}).astype(np.{dtype}); "
                f"b = rng_b.standard_normal({n}).astype(np.{dtype}){extra}"
            ),
            (
                f"{base}; "
                f"a = rng_a.uniform(0.01, 100, size={n}).astype(np.{dtype}); "
                f"b = rng_b.uniform(0.01, 100, size={n}).astype(np.{dtype}){extra}"
            ),
            (
                f"{base}; "
                f"a = rng_a.uniform(-1000, 1000, size={n}).astype(np.{dtype}); "
                f"b = rng_b.uniform(-1000, 1000, size={n}).astype(np.{dtype}){extra}"
            ),
        ]

    # --- Element-wise comparison/conversion ---
    if op == "allclose":
        # Two similar arrays (b = a + small noise) so allclose scans all.
        base = f"import numpy as np; rng = np.random.default_rng({rng_seed_a})"
        setups = [
            (
                f"{base}; a = rng.standard_normal({n_large}).astype(np.{dtype}); "
                f"b = a + rng.standard_normal({n_large}).astype(np.{dtype}) * 1e-10"
            ),
            (
                f"{base}; a = rng.uniform(0.01, 100, size={n_large}).astype(np.{dtype}); "
                f"b = a + rng.uniform(-1e-10, 1e-10, size={n_large}).astype(np.{dtype})"
            ),
            (
                f"{base}; a = rng.uniform(-1000, 1000, size={n_large}).astype(np.{dtype}); "
                f"b = a + rng.uniform(-1e-10, 1e-10, size={n_large}).astype(np.{dtype})"
            ),
        ]
        return {
            "setups": setups,
            "bench": "np.allclose(a, b)",
            "analytical": _analytical_cost(op, n=n_large),
        }

    if op in ("array_equal", "array_equiv"):
        setups = _three_setups_2arr(n_large)
        return {
            "setups": setups,
            "bench": f"np.{op}(a, b)",
            "analytical": _analytical_cost(op, n=n_large),
        }

    if op == "clip":
        setups = _three_setups_1arr(n_large)
        return {
            "setups": setups,
            "bench": "np.clip(x, -1.0, 1.0)",
            "analytical": _analytical_cost(op, n=n_large),
        }

    # --- Differencing ---
    if op in ("diff", "ediff1d", "gradient", "unwrap"):
        setups = _three_setups_1arr(n_large)
        return {
            "setups": setups,
            "bench": f"np.{op}(x)",
            "analytical": _analytical_cost(op, n=n_large),
        }

    # --- Convolution/correlation ---
    if op in ("convolve", "correlate"):
        n_conv = 100_000
        k_conv = 1000
        base = f"import numpy as np; rng = np.random.default_rng({rng_seed_a})"
        setups = [
            (
                f"{base}; "
                f"x = rng.standard_normal({n_conv}).astype(np.{dtype}); "
                f"k = rng.standard_normal({k_conv}).astype(np.{dtype})"
            ),
            (
                f"{base}; "
                f"x = rng.uniform(0.01, 100, size={n_conv}).astype(np.{dtype}); "
                f"k = rng.uniform(0.01, 100, size={k_conv}).astype(np.{dtype})"
            ),
            (
                f"{base}; "
                f"x = rng.uniform(-1000, 1000, size={n_conv}).astype(np.{dtype}); "
                f"k = rng.uniform(-1000, 1000, size={k_conv}).astype(np.{dtype})"
            ),
        ]
        return {
            "setups": setups,
            "bench": f"np.{op}(x, k, mode='full')",
            "analytical": _analytical_cost(op, n=n_conv, k=k_conv),
        }

    # --- Statistical ---
    if op in ("corrcoef", "cov"):
        f_feat = 1000
        s_samp = 10000
        base = f"import numpy as np; rng = np.random.default_rng({rng_seed_a})"
        setups = [
            f"{base}; x = rng.standard_normal(({f_feat}, {s_samp})).astype(np.{dtype})",
            f"{base}; x = rng.uniform(0.01, 100, size=({f_feat}, {s_samp})).astype(np.{dtype})",
            f"{base}; x = rng.uniform(-1000, 1000, size=({f_feat}, {s_samp})).astype(np.{dtype})",
        ]
        return {
            "setups": setups,
            "bench": f"np.{op}(x)",
            "analytical": _analytical_cost(op, f=f_feat, s=s_samp),
        }

    if op == "cross":
        n_cross = 1_000_000
        base_a = f"import numpy as np; rng_a = np.random.default_rng({rng_seed_a}); rng_b = np.random.default_rng({rng_seed_b})"
        setups = [
            (
                f"{base_a}; "
                f"a = rng_a.standard_normal(({n_cross}, 3)).astype(np.{dtype}); "
                f"b = rng_b.standard_normal(({n_cross}, 3)).astype(np.{dtype})"
            ),
            (
                f"{base_a}; "
                f"a = rng_a.uniform(0.01, 100, size=({n_cross}, 3)).astype(np.{dtype}); "
                f"b = rng_b.uniform(0.01, 100, size=({n_cross}, 3)).astype(np.{dtype})"
            ),
            (
                f"{base_a}; "
                f"a = rng_a.uniform(-1000, 1000, size=({n_cross}, 3)).astype(np.{dtype}); "
                f"b = rng_b.uniform(-1000, 1000, size=({n_cross}, 3)).astype(np.{dtype})"
            ),
        ]
        return {
            "setups": setups,
            "bench": "np.cross(a, b)",
            "analytical": _analytical_cost(op, n=n_cross),
        }

    # --- Binning/histogram ---
    if op == "histogram":
        setups = _three_setups_1arr(n_large)
        return {
            "setups": setups,
            "bench": "np.histogram(x, bins=100)",
            "analytical": _analytical_cost(op, n=n_large, bins=100),
        }

    if op == "histogram2d":
        setups = _three_setups_2arr(n_large, extra="")
        # Rename for histogram2d: need x and y
        setups = [
            s.replace("; a = ", "; x = ").replace("; b = ", "; y = ") for s in setups
        ]
        # Fix variable names in rng references
        setups = [
            s.replace("rng_a", "rng")
            .replace("rng_b", "rng2")
            .replace(
                f"rng = np.random.default_rng({rng_seed_a}); "
                f"rng2 = np.random.default_rng({rng_seed_b})",
                f"rng = np.random.default_rng({rng_seed_a}); "
                f"rng2 = np.random.default_rng({rng_seed_b})",
            )
            for s in setups
        ]
        return {
            "setups": setups,
            "bench": "np.histogram2d(x, y, bins=100)",
            "analytical": _analytical_cost(op, n=n_large, bins=100),
        }

    if op == "histogramdd":
        n_hdd = 1_000_000
        base = f"import numpy as np; rng = np.random.default_rng({rng_seed_a})"
        setups = [
            f"{base}; x = rng.standard_normal(({n_hdd}, 3)).astype(np.{dtype})",
            f"{base}; x = rng.uniform(0.01, 100, size=({n_hdd}, 3)).astype(np.{dtype})",
            f"{base}; x = rng.uniform(-1000, 1000, size=({n_hdd}, 3)).astype(np.{dtype})",
        ]
        return {
            "setups": setups,
            "bench": "np.histogramdd(x, bins=50)",
            "analytical": _analytical_cost(op, n=n_hdd, bins=50, ndim=3),
        }

    if op == "histogram_bin_edges":
        setups = _three_setups_1arr(n_large)
        return {
            "setups": setups,
            "bench": "np.histogram_bin_edges(x, bins=100)",
            "analytical": _analytical_cost(op, n=n_large),
        }

    if op == "digitize":
        base = f"import numpy as np; rng = np.random.default_rng({rng_seed_a})"
        setups = [
            f"{base}; x = rng.standard_normal({n_large}).astype(np.{dtype}); bins = np.linspace(-3, 3, 100)",
            f"{base}; x = rng.uniform(0.01, 100, size={n_large}).astype(np.{dtype}); bins = np.linspace(0, 100, 100)",
            f"{base}; x = rng.uniform(-1000, 1000, size={n_large}).astype(np.{dtype}); bins = np.linspace(-1000, 1000, 100)",
        ]
        return {
            "setups": setups,
            "bench": "np.digitize(x, bins)",
            "analytical": _analytical_cost(op, n=n_large, bins=100),
        }

    if op == "bincount":
        base = f"import numpy as np; rng = np.random.default_rng({rng_seed_a})"
        setups = [
            f"{base}; x = rng.integers(0, 1000, size={n_large})",
            f"{base}; x = rng.integers(0, 100, size={n_large})",
            f"{base}; x = rng.integers(0, 10000, size={n_large})",
        ]
        return {
            "setups": setups,
            "bench": "np.bincount(x)",
            "analytical": _analytical_cost(op, n=n_large),
        }

    # --- Interpolation ---
    if op == "interp":
        n_interp = 10_000_000
        xp_size = 10000
        base = f"import numpy as np; rng = np.random.default_rng({rng_seed_a})"
        setups = [
            (
                f"{base}; "
                f"xp = np.sort(rng.standard_normal({xp_size})); "
                f"fp = rng.standard_normal({xp_size}).astype(np.{dtype}); "
                f"x = rng.standard_normal({n_interp}).astype(np.{dtype})"
            ),
            (
                f"{base}; "
                f"xp = np.sort(rng.uniform(0.01, 100, size={xp_size})); "
                f"fp = rng.uniform(0.01, 100, size={xp_size}).astype(np.{dtype}); "
                f"x = rng.uniform(0.01, 100, size={n_interp}).astype(np.{dtype})"
            ),
            (
                f"{base}; "
                f"xp = np.sort(rng.uniform(-1000, 1000, size={xp_size})); "
                f"fp = rng.uniform(-1000, 1000, size={xp_size}).astype(np.{dtype}); "
                f"x = rng.uniform(-1000, 1000, size={n_interp}).astype(np.{dtype})"
            ),
        ]
        return {
            "setups": setups,
            "bench": "np.interp(x, xp, fp)",
            "analytical": _analytical_cost(op, n=n_interp, xp=xp_size),
        }

    # --- Linear/generation ---
    if op == "trace":
        # Use 10000x10000 to ensure analytical cost (10000) dominates
        # over subprocess overhead (~1000 FP instructions).
        n_trace = 10000
        base = f"import numpy as np; rng = np.random.default_rng({rng_seed_a})"
        setups = [
            f"{base}; A = rng.standard_normal(({n_trace}, {n_trace})).astype(np.{dtype})",
            f"{base}; A = rng.uniform(0.01, 100, size=({n_trace}, {n_trace})).astype(np.{dtype})",
            f"{base}; A = rng.uniform(-1000, 1000, size=({n_trace}, {n_trace})).astype(np.{dtype})",
        ]
        return {
            "setups": setups,
            "bench": "np.trace(A)",
            "analytical": _analytical_cost(op, n=n_trace),
        }

    if op == "trapezoid":
        setups = _three_setups_1arr(n_large)
        # Handle NumPy version differences: trapezoid (>=1.25) vs trapz
        # Use a simpler approach: detect at setup time
        setups_with_compat = []
        for s in setups:
            setups_with_compat.append(
                s + "; _trapfn = getattr(np, 'trapezoid', None) or np.trapz"
            )
        return {
            "setups": setups_with_compat,
            "bench": "_trapfn(x)",
            "analytical": _analytical_cost(op, n=n_large),
        }

    if op == "logspace":
        # logspace generates arrays, not element-wise on existing data.
        # Use three different ranges as "distributions".
        base = "import numpy as np"
        setups = [
            f"{base}",
            f"{base}",
            f"{base}",
        ]
        benches = [
            f"np.logspace(0, 10, {n_large})",
            f"np.logspace(-5, 5, {n_large})",
            f"np.logspace(0, 100, {n_large})",
        ]
        # For generation ops with varying bench code, return special marker.
        return {
            "setups": setups,
            "bench": benches,  # list of bench codes (one per distribution)
            "analytical": _analytical_cost(op, n=n_large),
        }

    if op == "geomspace":
        base = "import numpy as np"
        setups = [
            f"{base}",
            f"{base}",
            f"{base}",
        ]
        benches = [
            f"np.geomspace(1, 1000, {n_large})",
            f"np.geomspace(0.001, 1000, {n_large})",
            f"np.geomspace(1, 1e6, {n_large})",
        ]
        return {
            "setups": setups,
            "bench": benches,
            "analytical": _analytical_cost(op, n=n_large),
        }

    if op == "vander":
        n_vander = 10000
        degree = 100
        base = f"import numpy as np; rng = np.random.default_rng({rng_seed_a})"
        setups = [
            f"{base}; x = rng.standard_normal({n_vander}).astype(np.{dtype})",
            f"{base}; x = rng.uniform(0.01, 100, size={n_vander}).astype(np.{dtype})",
            f"{base}; x = rng.uniform(-1000, 1000, size={n_vander}).astype(np.{dtype})",
        ]
        return {
            "setups": setups,
            "bench": f"np.vander(x, {degree})",
            "analytical": _analytical_cost(op, n=n_vander, degree=degree),
        }

    raise ValueError(f"Unknown misc op: {op!r}")


def _benchmark_size_str(op: str) -> str:
    """Return a human-readable benchmark size string for an op."""
    n_large = 10_000_000
    if op in ("convolve", "correlate"):
        return "x: (100000,), k: (1000,)"
    if op in ("corrcoef", "cov"):
        return "x: (1000,10000)"
    if op == "cross":
        return "a: (1000000,3), b: (1000000,3)"
    if op == "histogramdd":
        return "x: (1000000,3), bins=50"
    if op == "histogram":
        return f"x: ({n_large},), bins=100"
    if op == "histogram2d":
        return f"x: ({n_large},), y: ({n_large},), bins=100"
    if op == "histogram_bin_edges":
        return f"x: ({n_large},), bins=100"
    if op == "digitize":
        return f"x: ({n_large},), bins: (100,)"
    if op == "bincount":
        return f"x: ({n_large},)"
    if op == "interp":
        return f"x: ({n_large},), xp: (10000,), fp: (10000,)"
    if op == "trace":
        return "A: (10000,10000)"
    if op == "vander":
        return "x: (10000,), degree=100"
    if op == "clip":
        return f"x: ({n_large},), a_min=-1.0, a_max=1.0"
    if op in ("allclose", "array_equal", "array_equiv"):
        return f"a: ({n_large},), b: ({n_large},)"
    if op in ("diff", "ediff1d", "gradient", "unwrap"):
        return f"x: ({n_large},)"
    if op == "trapezoid":
        return f"x: ({n_large},)"
    if op in ("logspace", "geomspace"):
        return f"output: ({n_large},)"
    # Default: most ops use n_large
    return f"x: ({n_large},)"


def benchmark_misc(
    dtype: str = "float64",
    repeats: int = 10,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Benchmark misc ops, returning alpha(op) = measured / analytical.

    For each operation, run with 3 input distributions and take the
    median ratio of measured FP work to analytical cost.

    Parameters
    ----------
    dtype : str
        NumPy dtype string (default ``"float64"``).
    repeats : int
        Number of repetitions per measurement.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict]]
        A pair of (alphas, details). ``alphas`` maps op name to median
        alpha(op). ``details`` maps op name to a dict of raw benchmark
        metadata.
    """
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    for op in MISC_OPS:
        try:
            config = _get_op_config(op, dtype)
        except ValueError:
            continue

        setups = config["setups"]
        bench = config["bench"]
        analytical = config["analytical"]

        if analytical <= 0:
            continue

        dist_values: list[float] = []
        dist_raw_totals: list[int] = []

        for i, setup in enumerate(setups):
            # bench can be a single string or a list (one per distribution)
            if isinstance(bench, list):
                bench_code = bench[i]
            else:
                bench_code = bench

            try:
                result = measure_flops(setup, bench_code, repeats=repeats)
            except RuntimeError:
                continue
            dist_values.append(result.total_flops / (analytical * repeats))
            dist_raw_totals.append(result.total_flops)

        if dist_values:
            results[op] = statistics.median(dist_values)
            # For ops with varying bench code (list), store the first one
            display_bench = bench[0] if isinstance(bench, list) else bench
            details[op] = {
                "category": "counted_custom",
                "measurement_mode": "custom",
                "analytical_formula": _FORMULA_STRINGS.get(op, "n"),
                "analytical_flops": analytical,
                "benchmark_size": _benchmark_size_str(op),
                "bench_code": display_bench,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    return results, details
