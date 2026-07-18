"""Tests for window function benchmark module."""

from unittest.mock import patch

import pytest

from benchmarks._perf import PerfResult
from benchmarks._window import (
    _ANALYTICAL_COST,
    _FORMULA_STRINGS,
    WINDOW_OPS,
    benchmark_window,
)


class TestOpsLists:
    def test_window_ops_non_empty(self):
        assert len(WINDOW_OPS) > 0

    def test_contains_all_expected_ops(self):
        expected = ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
        for op in expected:
            assert op in WINDOW_OPS, f"{op} missing from WINDOW_OPS"

    def test_window_ops_length(self):
        assert len(WINDOW_OPS) == 5


class TestAnalyticalCost:
    def test_bartlett_cost(self):
        assert (
            _ANALYTICAL_COST["bartlett"] == 4
        )  # compare+div+add+select per sample (FMA=2)

    def test_blackman_cost(self):
        assert (
            _ANALYTICAL_COST["blackman"] == 40
        )  # 2 cos evals @16 + 8 arith per sample

    def test_hamming_cost(self):
        assert _ANALYTICAL_COST["hamming"] == 18  # cos@16 + mul + sub per sample

    def test_hanning_cost(self):
        assert _ANALYTICAL_COST["hanning"] == 18  # cos@16 + mul + sub per sample

    def test_kaiser_cost(self):
        assert (
            _ANALYTICAL_COST["kaiser"] == 23
        )  # Bessel I0 @16 + 7 arith per sample (FMA=2)

    def test_all_ops_have_cost(self):
        for op in WINDOW_OPS:
            assert op in _ANALYTICAL_COST, f"{op} missing from _ANALYTICAL_COST"


class TestFormulaStrings:
    def test_all_ops_have_formula(self):
        for op in WINDOW_OPS:
            assert op in _FORMULA_STRINGS, f"{op} missing from _FORMULA_STRINGS"


class TestBenchmarkWindow:
    def test_returns_tuple(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._window.measure_flops", return_value=mock_result):
            ret = benchmark_window(n=1_000, dtype="float64", repeats=1)

        assert isinstance(ret, tuple)
        assert len(ret) == 2

    def test_returns_dict_with_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._window.measure_flops", return_value=mock_result):
            result, details = benchmark_window(n=1_000, dtype="float64", repeats=1)

        assert isinstance(result, dict)
        assert set(WINDOW_OPS).issubset(set(result.keys()))

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=500_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._window.measure_flops", return_value=mock_result):
            result, _ = benchmark_window(n=1_000, dtype="float64", repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_bartlett_uses_linear_denominator(self):
        n = 1_000
        repeats = 2
        total_flops = 10_000
        mock_result = PerfResult(
            scalar_double=total_flops,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._window.measure_flops", return_value=mock_result):
            result, _ = benchmark_window(n=n, dtype="float64", repeats=repeats)

        analytical = _ANALYTICAL_COST["bartlett"] * n
        expected = total_flops / (analytical * repeats)
        assert result["bartlett"] == pytest.approx(expected)

    def test_blackman_uses_40n_denominator(self):
        n = 1_000
        repeats = 1
        total_flops = 40_000
        mock_result = PerfResult(
            scalar_double=total_flops,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._window.measure_flops", return_value=mock_result):
            result, _ = benchmark_window(n=n, dtype="float64", repeats=repeats)

        analytical = _ANALYTICAL_COST["blackman"] * n  # 40 * n
        expected = total_flops / (analytical * repeats)
        assert result["blackman"] == pytest.approx(expected)

    def test_kaiser_uses_23n_denominator(self):
        n = 1_000
        repeats = 1
        total_flops = 23_000
        mock_result = PerfResult(
            scalar_double=total_flops,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._window.measure_flops", return_value=mock_result):
            result, _ = benchmark_window(n=n, dtype="float64", repeats=repeats)

        analytical = _ANALYTICAL_COST["kaiser"] * n  # 23 * n
        expected = total_flops / (analytical * repeats)
        assert result["kaiser"] == pytest.approx(expected)

    def test_runtime_error_skipped(self):
        """If measure_flops raises RuntimeError, the op is skipped."""
        with patch(
            "benchmarks._window.measure_flops", side_effect=RuntimeError("no perf")
        ):
            result, details = benchmark_window(n=1_000, dtype="float64", repeats=1)

        assert result == {}
        assert details == {}

    def test_details_populated_for_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._window.measure_flops", return_value=mock_result):
            result, details = benchmark_window(n=1_000, dtype="float64", repeats=1)

        assert set(details.keys()) == set(result.keys())
        for op, d in details.items():
            assert d["category"] == "counted_custom"
            assert d["analytical_formula"] == _FORMULA_STRINGS[op]
            assert d["analytical_flops"] == _ANALYTICAL_COST[op] * 1_000
            assert d["benchmark_size"] == "output: (1000,)"
            assert isinstance(d["bench_code"], str)
            assert d["repeats"] == 1
            assert isinstance(d["perf_instructions_total"], list)
            assert isinstance(d["distribution_alphas"], list)
