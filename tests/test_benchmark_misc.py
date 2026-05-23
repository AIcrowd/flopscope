"""Tests for miscellaneous benchmark module."""

import math
from unittest.mock import patch

import pytest

from benchmarks._misc import (
    _FORMULA_STRINGS,
    MISC_OPS,
    _analytical_cost,
    _get_op_config,
    benchmark_misc,
)
from benchmarks._perf import PerfResult


class TestOpsLists:
    def test_misc_ops_non_empty(self):
        assert len(MISC_OPS) > 0

    def test_expected_ops_present(self):
        expected = [
            "allclose",
            "array_equal",
            "array_equiv",
            "clip",
            "diff",
            "ediff1d",
            "gradient",
            "unwrap",
            "convolve",
            "correlate",
            "corrcoef",
            "cov",
            "cross",
            "histogram",
            "histogram2d",
            "histogramdd",
            "histogram_bin_edges",
            "digitize",
            "bincount",
            "interp",
            "trace",
            "trapezoid",
            "logspace",
            "geomspace",
            "vander",
        ]
        for op in expected:
            assert op in MISC_OPS, f"{op} missing from MISC_OPS"

    def test_no_duplicates(self):
        assert len(MISC_OPS) == len(set(MISC_OPS))


class TestAnalyticalCost:
    """Test _analytical_cost returns correct values per formula."""

    # --- Element-wise (cost = n) ---

    def test_allclose_linear(self):
        assert _analytical_cost("allclose", n=1000) == 1000

    def test_array_equal_linear(self):
        assert _analytical_cost("array_equal", n=500) == 500

    def test_array_equiv_linear(self):
        assert _analytical_cost("array_equiv", n=500) == 500

    def test_clip_linear(self):
        assert _analytical_cost("clip", n=2000) == 2000

    # --- Differencing (cost = n) ---

    def test_diff_linear(self):
        assert _analytical_cost("diff", n=1000) == 1000

    def test_ediff1d_linear(self):
        assert _analytical_cost("ediff1d", n=1000) == 1000

    def test_gradient_linear(self):
        assert _analytical_cost("gradient", n=1000) == 1000

    def test_unwrap_linear(self):
        assert _analytical_cost("unwrap", n=1000) == 7000  # 7 * numel (issue #69)

    # --- Convolution (cost = 2*n*k - n - k, FMA=1 convention, issue #69) ---

    def test_convolve_nk(self):
        assert _analytical_cost("convolve", n=1000, k=100) == 2 * 1000 * 100 - 1000 - 100  # 198900

    def test_correlate_nk(self):
        assert _analytical_cost("correlate", n=1000, k=100) == 2 * 1000 * 100 - 1000 - 100  # 198900

    def test_convolve_default_k(self):
        # Default k=1000
        assert _analytical_cost("convolve", n=500) == 2 * 500 * 1000 - 500 - 1000  # 998500

    # --- Statistical ---

    def test_corrcoef_matmul_dominated(self):
        assert _analytical_cost("corrcoef", f=100, s=200) == 100 * 100 * 200

    def test_cov_matmul_dominated(self):
        assert _analytical_cost("cov", f=100, s=200) == 100 * 100 * 200

    def test_cross_6n(self):
        assert _analytical_cost("cross", n=1000) == 6000

    # --- Binning/histogram ---

    def test_histogram_nlog2bins(self):
        n, bins = 1000, 100
        expected = n * math.ceil(math.log2(bins))
        assert _analytical_cost("histogram", n=n, bins=bins) == expected

    def test_histogram2d_two_log2(self):
        n, bins = 1000, 100
        log2b = math.ceil(math.log2(bins))
        assert _analytical_cost("histogram2d", n=n, bins=bins) == n * 2 * log2b

    def test_histogramdd_ndim_log2(self):
        n, bins, ndim = 1000, 50, 3
        expected = n * ndim * math.ceil(math.log2(bins))
        assert _analytical_cost("histogramdd", n=n, bins=bins, ndim=ndim) == expected

    def test_histogram_bin_edges_linear(self):
        assert _analytical_cost("histogram_bin_edges", n=1000) == 1000

    def test_digitize_nlog2bins(self):
        n, bins = 1000, 100
        expected = n * math.ceil(math.log2(bins))
        assert _analytical_cost("digitize", n=n, bins=bins) == expected

    def test_bincount_linear(self):
        assert _analytical_cost("bincount", n=5000) == 5000

    # --- Interpolation ---

    def test_interp_nlog2xp(self):
        n, xp = 1000, 100
        expected = n * math.ceil(math.log2(xp))
        assert _analytical_cost("interp", n=n, xp=xp) == expected

    # --- Linear/generation ---

    def test_trace_min_mn(self):
        # For the benchmark, trace uses n = min(m, n) = 1000
        assert _analytical_cost("trace", n=1000) == 1000

    def test_trapezoid_linear(self):
        assert _analytical_cost("trapezoid", n=5000) == 5000

    def test_logspace_linear(self):
        assert _analytical_cost("logspace", n=5000) == 5000

    def test_geomspace_linear(self):
        assert _analytical_cost("geomspace", n=5000) == 5000

    def test_vander_n_deg_minus_1(self):
        assert _analytical_cost("vander", n=100, degree=10) == 100 * 9

    def test_vander_default_degree(self):
        # Default degree=100
        assert _analytical_cost("vander", n=100) == 100 * 99

    # --- Default n ---

    def test_default_n_is_10M(self):
        # With no explicit n, ops that use n should default to 10M
        assert _analytical_cost("allclose") == 10_000_000
        assert _analytical_cost("diff") == 10_000_000


class TestGetOpConfig:
    """Test that _get_op_config returns valid configs for all ops."""

    @pytest.mark.parametrize("op", MISC_OPS)
    def test_config_has_required_keys(self, op):
        config = _get_op_config(op, "float64")
        assert "setups" in config
        assert "bench" in config
        assert "analytical" in config

    @pytest.mark.parametrize("op", MISC_OPS)
    def test_config_has_three_setups(self, op):
        config = _get_op_config(op, "float64")
        assert len(config["setups"]) == 3

    @pytest.mark.parametrize("op", MISC_OPS)
    def test_analytical_positive(self, op):
        config = _get_op_config(op, "float64")
        assert config["analytical"] > 0

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError, match="Unknown misc op"):
            _get_op_config("nonexistent_op", "float64")


class TestFormulaStrings:
    def test_all_ops_have_formula(self):
        for op in MISC_OPS:
            assert op in _FORMULA_STRINGS, f"{op} missing from _FORMULA_STRINGS"

    def test_formula_strings_are_strings(self):
        for op, formula in _FORMULA_STRINGS.items():
            assert isinstance(formula, str), f"{op} formula is not a string"


class TestBenchmarkMisc:
    def test_returns_tuple(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._misc.measure_flops", return_value=mock_result):
            ret = benchmark_misc(dtype="float64", repeats=1)

        assert isinstance(ret, tuple)
        assert len(ret) == 2

    def test_returns_dict(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._misc.measure_flops", return_value=mock_result):
            result, details = benchmark_misc(dtype="float64", repeats=1)

        assert isinstance(result, dict)

    def test_all_ops_present(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._misc.measure_flops", return_value=mock_result):
            result, _ = benchmark_misc(dtype="float64", repeats=1)

        for op in MISC_OPS:
            assert op in result, f"{op} missing from benchmark results"

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=500_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._misc.measure_flops", return_value=mock_result):
            result, _ = benchmark_misc(dtype="float64", repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_clip_uses_analytical_denominator(self):
        """Verify clip normalizes by n (analytical cost)."""
        repeats = 2
        total_flops = 10_000
        mock_result = PerfResult(
            scalar_double=total_flops,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._misc.measure_flops", return_value=mock_result):
            result, _ = benchmark_misc(dtype="float64", repeats=repeats)

        # clip analytical cost = n = 10M
        analytical = _analytical_cost("clip", n=10_000_000)
        expected = total_flops / (analytical * repeats)
        assert result["clip"] == pytest.approx(expected)

    def test_convolve_uses_nk_denominator(self):
        """Verify convolve normalizes by n*k."""
        repeats = 1
        total_flops = 50_000
        mock_result = PerfResult(
            scalar_double=total_flops,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._misc.measure_flops", return_value=mock_result):
            result, _ = benchmark_misc(dtype="float64", repeats=repeats)

        analytical = _analytical_cost("convolve", n=100_000, k=1000)
        expected = total_flops / (analytical * repeats)
        assert result["convolve"] == pytest.approx(expected)

    def test_histogram_uses_nlog2bins(self):
        """Verify histogram normalizes by n * ceil(log2(bins))."""
        repeats = 1
        total_flops = 100_000
        mock_result = PerfResult(
            scalar_double=total_flops,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._misc.measure_flops", return_value=mock_result):
            result, _ = benchmark_misc(dtype="float64", repeats=repeats)

        analytical = _analytical_cost("histogram", n=10_000_000, bins=100)
        expected = total_flops / (analytical * repeats)
        assert result["histogram"] == pytest.approx(expected)

    def test_handles_runtime_error_gracefully(self):
        """Benchmark should skip ops that raise RuntimeError."""

        def mock_measure(setup, bench, repeats=10):
            raise RuntimeError("perf not available")

        with patch("benchmarks._misc.measure_flops", side_effect=mock_measure):
            result, details = benchmark_misc(dtype="float64", repeats=1)

        # Should return empty dict (all ops failed)
        assert result == {}
        assert details == {}

    def test_details_populated_for_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._misc.measure_flops", return_value=mock_result):
            result, details = benchmark_misc(dtype="float64", repeats=1)

        assert set(details.keys()) == set(result.keys())
        for op, d in details.items():
            assert d["category"] == "counted_custom"
            assert d["analytical_formula"] == _FORMULA_STRINGS[op]
            assert d["analytical_flops"] > 0
            assert isinstance(d["benchmark_size"], str)
            assert isinstance(d["bench_code"], str)
            assert d["repeats"] == 1
            assert isinstance(d["perf_instructions_total"], list)
            assert isinstance(d["distribution_alphas"], list)
