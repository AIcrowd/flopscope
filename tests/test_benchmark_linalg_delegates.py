"""Tests for linalg delegate benchmark module."""

from unittest.mock import patch

import numpy as np
import pytest

from benchmarks._linalg_delegates import (
    _FORMULA_STRINGS,
    _NUMPY2_OPS,
    LINALG_DELEGATE_OPS,
    _analytical_cost,
    benchmark_linalg_delegates,
)
from benchmarks._perf import PerfResult


class TestOpsLists:
    def test_delegate_ops_non_empty(self):
        assert len(LINALG_DELEGATE_OPS) == 15

    def test_contains_expected_ops(self):
        for op in (
            "linalg.cond",
            "linalg.matmul",
            "linalg.norm",
            "linalg.trace",
            "linalg.tensorinv",
        ):
            assert op in LINALG_DELEGATE_OPS, f"{op} missing"


class TestAnalyticalCost:
    def test_cond_cost(self):
        assert _analytical_cost("linalg.cond") == 512 * 512 * 512

    def test_cross_cost(self):
        assert _analytical_cost("linalg.cross") == 6 * 1_000_000

    def test_matmul_cost(self):
        assert (
            _analytical_cost("linalg.matmul") == 2 * 512 * 512 * 512 - 512 * 512
        )  # FMA=2

    def test_matrix_norm_cost(self):
        assert _analytical_cost("linalg.matrix_norm") == 2 * 512 * 512

    def test_matrix_power_cost(self):
        assert _analytical_cost("linalg.matrix_power") == 3 * 64**3

    def test_matrix_rank_cost(self):
        assert _analytical_cost("linalg.matrix_rank") == 512**3

    def test_multi_dot_cost(self):
        expected = (
            128 * 64 * 128 + 128 * 128 * 64
        )  # FMA=2 optimal chain (coincidentally equals FMA=1 value)
        assert _analytical_cost("linalg.multi_dot") == expected

    def test_norm_cost(self):
        assert _analytical_cost("linalg.norm") == 2 * 10_000_000  # FMA=2: 2*numel

    def test_outer_cost(self):
        assert _analytical_cost("linalg.outer") == 5000 * 5000

    def test_tensordot_cost(self):
        assert _analytical_cost("linalg.tensordot") == 64**5

    def test_tensorinv_cost(self):
        assert _analytical_cost("linalg.tensorinv") == 64**3

    def test_tensorsolve_cost(self):
        assert _analytical_cost("linalg.tensorsolve") == 64**3

    def test_trace_cost(self):
        assert _analytical_cost("linalg.trace") == 10_000

    def test_vecdot_cost(self):
        assert _analytical_cost("linalg.vecdot") == 1000 * 512

    def test_vector_norm_cost(self):
        assert (
            _analytical_cost("linalg.vector_norm") == 2 * 10_000_000
        )  # FMA=2: 2*numel

    def test_unknown_op_raises(self):
        with pytest.raises((ValueError, KeyError)):
            _analytical_cost("linalg.bogus")


class TestFormulaStrings:
    def test_all_ops_have_formula(self):
        for op in LINALG_DELEGATE_OPS:
            assert op in _FORMULA_STRINGS, f"{op} missing from _FORMULA_STRINGS"
            assert isinstance(_FORMULA_STRINGS[op], str)
            assert len(_FORMULA_STRINGS[op]) > 0


class TestBenchmarkLinalgDelegates:
    def _available_ops(self):
        """Return the subset of ops available in the installed NumPy."""
        available = []
        for op in LINALG_DELEGATE_OPS:
            if op in _NUMPY2_OPS:
                try:
                    fn = np.linalg
                    for part in op.split(".")[1:]:
                        fn = getattr(fn, part)
                except AttributeError:
                    continue
            available.append(op)
        return available

    def test_returns_tuple_with_all_available_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._linalg_delegates.measure_flops",
            return_value=mock_result,
        ):
            result, details = benchmark_linalg_delegates(dtype="float64", repeats=1)

        available = set(self._available_ops())
        assert isinstance(result, dict)
        assert isinstance(details, dict)
        assert set(result.keys()) == available
        assert set(details.keys()) == available

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._linalg_delegates.measure_flops",
            return_value=mock_result,
        ):
            result, _ = benchmark_linalg_delegates(dtype="float64", repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_details_have_required_keys(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._linalg_delegates.measure_flops",
            return_value=mock_result,
        ):
            _, details = benchmark_linalg_delegates(dtype="float64", repeats=1)

        required_keys = {
            "category",
            "measurement_mode",
            "analytical_formula",
            "analytical_flops",
            "benchmark_size",
            "bench_code",
            "repeats",
            "perf_instructions_total",
            "distribution_alphas",
        }
        for op, d in details.items():
            assert required_keys.issubset(d.keys()), (
                f"{op} missing keys: {required_keys - set(d.keys())}"
            )
            assert d["category"] == "counted_custom"
            assert isinstance(d["analytical_flops"], int)
            assert isinstance(d["bench_code"], str)
            assert isinstance(d["distribution_alphas"], list)
            assert len(d["distribution_alphas"]) > 0
