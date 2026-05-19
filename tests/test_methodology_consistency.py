"""Verify that analytical formulas in benchmarks match flopscope runtime costs.

For a representative subset of operations (one per benchmark category), this
test runs the flopscope-wrapped operation inside a BudgetContext and verifies
that the FLOP cost it charges matches the analytical formula used in the
corresponding benchmark module's denominator.

This guards against drift between the benchmark normalization denominator and
the runtime cost model -- if they diverge, the empirical weight would be
wrong.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

import flopscope.numpy as fnp
from flopscope._budget import BudgetContext

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks._fft import _analytical_cost as fft_analytical_cost  # noqa: E402
from benchmarks._linalg import _analytical_cost as linalg_analytical_cost  # noqa: E402
from benchmarks._polynomial import (
    _analytical_cost as poly_analytical_cost,  # noqa: E402
)
from benchmarks._sorting import (
    _analytical_cost as sorting_analytical_cost,  # noqa: E402
)


def _run_and_get_cost(func, *args, **kwargs) -> int:
    """Run a flopscope function inside a budget context and return the FLOP cost."""
    with BudgetContext(flop_budget=10**18) as ctx:
        func(*args, **kwargs)
        records = ctx.op_log
        assert records, f"No FLOP records for {func}"
        return records[-1].flop_cost


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestSortingConsistency:
    """sort, argsort: cost = n * ceil(log2(n))."""

    def test_sort(self):
        n = 1000
        a = np.random.rand(n)
        runtime_cost = _run_and_get_cost(fnp.sort, a)
        expected = sorting_analytical_cost("sort", n)
        assert runtime_cost == expected, (
            f"sort({n}): runtime={runtime_cost}, benchmark={expected}"
        )

    def test_argsort(self):
        n = 1000
        a = np.random.rand(n)
        runtime_cost = _run_and_get_cost(fnp.argsort, a)
        expected = sorting_analytical_cost("argsort", n)
        assert runtime_cost == expected


# ---------------------------------------------------------------------------
# Contractions
# ---------------------------------------------------------------------------


class TestContractionConsistency:
    """matmul: cost uses the whole-expression direct-event accumulation model.
    For 2D matrix multiply with no symmetry:
        total = (k-1)*M + alpha - num_output_orbits
              = M*N*K + M*N*K - M*N = 2*M*N*K - M*N.
    The final -M*N applies the off-by-one correction: first cell of each
    output orbit is a free copy."""

    def test_matmul(self):
        m, n, k = 32, 32, 32
        a = np.random.rand(m, k)
        b = np.random.rand(k, n)
        runtime_cost = _run_and_get_cost(fnp.matmul, a, b)
        # direct-event model with off-by-one correction:
        # (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
        # = M*N*K + M*N*K - M*N = 2*M*N*K - M*N (textbook 2n^3 - n^2 form).
        expected = 2 * m * n * k - m * n
        assert runtime_cost == expected, (
            f"matmul({m},{k})x({k},{n}): runtime={runtime_cost}, expected={expected}"
        )


# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------


class TestPointwiseConsistency:
    """Unary ops: cost = numel(input). Binary ops: cost = numel(broadcast output)."""

    def test_sin(self):
        n = 1000
        a = np.random.rand(n)
        runtime_cost = _run_and_get_cost(fnp.sin, a)
        assert runtime_cost == n

    def test_add(self):
        n = 1000
        a = np.random.rand(n)
        b = np.random.rand(n)
        runtime_cost = _run_and_get_cost(fnp.add, a, b)
        assert runtime_cost == n

    def test_exp(self):
        n = 500
        a = np.random.rand(n)
        runtime_cost = _run_and_get_cost(fnp.exp, a)
        assert runtime_cost == n


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


class TestReductionConsistency:
    """Reductions: cost = numel(input) - 1 (orbit-mapping model)."""

    def test_sum(self):
        # Updated to orbit-mapping count (was n(n+1)/2 unique-element count).
        # Full reduction of (n,): n - 1 additions.
        n = 1000
        a = np.random.rand(n)
        runtime_cost = _run_and_get_cost(fnp.sum, a)
        assert runtime_cost == n - 1

    def test_mean(self):
        # Updated to orbit-mapping count (was n(n+1)/2 unique-element count).
        # Full reduction of (n,): n - 1 additions (mean includes that reduction).
        n = 1000
        a = np.random.rand(n)
        runtime_cost = _run_and_get_cost(fnp.mean, a)
        assert runtime_cost >= n - 1


# ---------------------------------------------------------------------------
# Polynomial
# ---------------------------------------------------------------------------


class TestPolynomialConsistency:
    """polyval: cost = m * deg (Horner's method, FMA=1)."""

    def test_polyval(self):
        degree = 10
        n = 100
        p = np.random.rand(degree + 1)
        x = np.random.rand(n)
        runtime_cost = _run_and_get_cost(fnp.polyval, p, x)
        expected = poly_analytical_cost("polyval", n, degree)
        assert runtime_cost == expected, (
            f"polyval(deg={degree}, n={n}): runtime={runtime_cost}, expected={expected}"
        )


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------


class TestFFTConsistency:
    """fft.fft: cost = 5 * n * ceil(log2(n))."""

    def test_fft(self):
        n = 1024
        a = np.random.rand(n)
        runtime_cost = _run_and_get_cost(fnp.fft.fft, a)
        expected = fft_analytical_cost("fft.fft", n)
        assert runtime_cost == expected, (
            f"fft.fft({n}): runtime={runtime_cost}, expected={expected}"
        )


# ---------------------------------------------------------------------------
# Histogram (misc)
# ---------------------------------------------------------------------------


class TestMiscConsistency:
    """histogram: cost = n * ceil(log2(bins))."""

    def test_histogram(self):
        n = 1000
        bins = 10
        a = np.random.rand(n)
        runtime_cost = _run_and_get_cost(fnp.histogram, a, bins=bins)
        expected = n * math.ceil(math.log2(bins))
        assert runtime_cost == expected, (
            f"histogram({n}, bins={bins}): runtime={runtime_cost}, expected={expected}"
        )


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


class TestWindowConsistency:
    """bartlett: cost = n (one trig eval per sample)."""

    def test_bartlett(self):
        n = 1000
        runtime_cost = _run_and_get_cost(fnp.bartlett, n)
        assert runtime_cost == n, f"bartlett({n}): runtime={runtime_cost}, expected={n}"


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------


class TestRandomConsistency:
    """Random samplers: cost = numel(output)."""

    def test_standard_normal(self):
        n = 1000
        runtime_cost = _run_and_get_cost(fnp.random.standard_normal, size=n)
        assert runtime_cost == n

    def test_uniform(self):
        n = 500
        runtime_cost = _run_and_get_cost(fnp.random.uniform, 0.0, 1.0, size=n)
        assert runtime_cost == n


# ---------------------------------------------------------------------------
# Linalg
# ---------------------------------------------------------------------------


class TestLinalgConsistency:
    """linalg.cholesky: cost = n^3."""

    def test_cholesky(self):
        n = 64
        a = np.random.rand(n, n)
        a = a @ a.T + np.eye(n) * n  # make SPD
        runtime_cost = _run_and_get_cost(fnp.linalg.cholesky, a)
        expected = linalg_analytical_cost("linalg.cholesky", n)
        assert runtime_cost == expected, (
            f"linalg.cholesky({n}x{n}): runtime={runtime_cost}, expected={expected}"
        )
