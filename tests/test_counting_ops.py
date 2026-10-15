"""Tests for _counting_ops.py — counted trace, histogram, and generation ops."""

from __future__ import annotations

import numpy

from flopscope._budget import BudgetContext
from flopscope._counting_ops import (
    allclose,
    array_equal,
    array_equiv,
    bincount,
    geomspace,
    histogram,
    histogram2d,
    histogram_bin_edges,
    histogramdd,
    logspace,
    trace,
    vander,
)

# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------


class TestTrace:
    def test_result_matches_numpy(self):
        a = numpy.array([[1, 2], [3, 4]])
        with BudgetContext(flop_budget=10**6):
            result = trace(a)
        assert result == numpy.trace(a)

    def test_cost_rectangular_min_rows(self):
        # 5x8 matrix — cost = min(5, 8) = 5
        a = numpy.random.randn(5, 8)
        with BudgetContext(flop_budget=10**6) as budget:
            trace(a)
        assert budget.flops_used == 5

    def test_cost_square(self):
        # 10x10 — cost = 10
        a = numpy.random.randn(10, 10)
        with BudgetContext(flop_budget=10**6) as budget:
            trace(a)
        assert budget.flops_used == 10

    def test_cost_rectangular_min_cols(self):
        # 8x3 matrix — cost = min(8, 3) = 3
        a = numpy.random.randn(8, 3)
        with BudgetContext(flop_budget=10**6) as budget:
            trace(a)
        assert budget.flops_used == 3

    def test_cost_minimum_one(self):
        # 1x1 — cost must be at least 1
        a = numpy.array([[7.0]])
        with BudgetContext(flop_budget=10**6) as budget:
            trace(a)
        assert budget.flops_used == 1

    def test_result_with_offset(self):
        a = numpy.arange(9).reshape(3, 3)
        with BudgetContext(flop_budget=10**6):
            result = trace(a, offset=1)
        assert result == numpy.trace(a, offset=1)


# ---------------------------------------------------------------------------
# allclose
# ---------------------------------------------------------------------------


class TestAllclose:
    def test_result_matches_numpy_true(self):
        a = numpy.ones(100)
        b = numpy.ones(100) + 1e-12
        with BudgetContext(flop_budget=10**6):
            result = allclose(a, b)
        assert result == numpy.allclose(a, b)

    def test_result_matches_numpy_false(self):
        a = numpy.ones(50)
        b = numpy.zeros(50)
        with BudgetContext(flop_budget=10**6):
            result = allclose(a, b)
        assert result == numpy.allclose(a, b)

    def test_cost(self):
        a = numpy.random.randn(100)
        b = numpy.random.randn(100)
        with BudgetContext(flop_budget=10**6) as budget:
            allclose(a, b)
        # 7*numel - 1: 6/elem tolerance core + all-reduce
        assert budget.flops_used == 7 * 100 - 1

    def test_cost_minimum_one(self):
        a = numpy.array([1.0])
        b = numpy.array([1.0])
        with BudgetContext(flop_budget=10**6) as budget:
            allclose(a, b)
        # 7*1 - 1 = 6 (numel=1)
        assert budget.flops_used == 6


# ---------------------------------------------------------------------------
# array_equal
# ---------------------------------------------------------------------------


class TestArrayEqual:
    def test_result_matches_numpy_equal(self):
        a = numpy.array([1, 2, 3])
        with BudgetContext(flop_budget=10**6):
            result = array_equal(a, a.copy())
        assert result == numpy.array_equal(a, a.copy())

    def test_result_matches_numpy_not_equal(self):
        a = numpy.array([1, 2, 3])
        b = numpy.array([1, 2, 4])
        with BudgetContext(flop_budget=10**6):
            result = array_equal(a, b)
        assert result == numpy.array_equal(a, b)

    def test_cost(self):
        a = numpy.random.randn(50)
        b = numpy.random.randn(50)
        with BudgetContext(flop_budget=10**6) as budget:
            array_equal(a, b)
        assert budget.flops_used == 50


# ---------------------------------------------------------------------------
# array_equiv
# ---------------------------------------------------------------------------


class TestArrayEquiv:
    def test_result_matches_numpy(self):
        a = numpy.array([1, 2, 3])
        b = numpy.array([[1, 2, 3], [1, 2, 3]])
        with BudgetContext(flop_budget=10**6):
            result = array_equiv(a, b)
        assert result == numpy.array_equiv(a, b)

    def test_cost(self):
        a = numpy.random.randn(40)
        b = numpy.random.randn(40)
        with BudgetContext(flop_budget=10**6) as budget:
            array_equiv(a, b)
        assert budget.flops_used == 40


# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------


class TestHistogram:
    def test_result_matches_numpy(self):
        a = numpy.random.randn(100)
        with BudgetContext(flop_budget=10**6):
            counts, edges = histogram(a, bins=10)
        np_counts, np_edges = numpy.histogram(a, bins=10)
        numpy.testing.assert_array_equal(counts, np_counts)
        numpy.testing.assert_array_equal(edges, np_edges)

    def test_cost_int_bins(self):
        # n=100, bins=8 → ceil(log2(8))=3 → cost=300, plus the 9 edges
        # charged as linspace (2 per edge)
        a = numpy.random.randn(100)
        with BudgetContext(flop_budget=10**6) as budget:
            histogram(a, bins=8)
        assert budget.flops_used == 300 + 2 * 9

    def test_cost_int_bins_default(self):
        # default bins=10, n=100 → ceil(log2(10))=4 → cost=400, plus the 11
        # edges charged as linspace
        a = numpy.random.randn(100)
        with BudgetContext(flop_budget=10**6) as budget:
            histogram(a)
        assert budget.flops_used == 400 + 2 * 11

    def test_cost_string_bins(self):
        # bins="auto" → cost = n*(2 + estimator + ceil_log2(resolved_bins))
        # "auto" estimator cost = 1 (percentile), must be >= int-path equivalent
        import math

        a = numpy.random.randn(100)
        n = 100
        nb = len(numpy.histogram_bin_edges(a, "auto")) - 1
        # plus the nb + 1 edges, charged as linspace
        expected = n * (2 + 1 + math.ceil(math.log2(max(nb, 1)))) + 2 * (nb + 1)
        with BudgetContext(flop_budget=10**6) as budget:
            histogram(a, bins="auto")
        assert budget.flops_used == expected

    def test_cost_array_bins(self):
        # array bins of length 11 → binary search: n * ceil(log2(11)) = 100 * 4 = 400
        a = numpy.random.randn(100)
        edges = numpy.linspace(-3, 3, 11)
        with BudgetContext(flop_budget=10**6) as budget:
            histogram(a, bins=edges)  # pyright: ignore[reportArgumentType]
        assert budget.flops_used == 400


# ---------------------------------------------------------------------------
# histogram2d
# ---------------------------------------------------------------------------


class TestHistogram2d:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(50)
        y = numpy.random.randn(50)
        with BudgetContext(flop_budget=10**6):
            counts, xe, ye = histogram2d(x, y, bins=10)
        np_counts, np_xe, np_ye = numpy.histogram2d(x, y, bins=10)
        numpy.testing.assert_array_equal(counts, np_counts)

    def test_cost_int_bins(self):
        # n=50, bins=10 → bx=by=10 → ceil(log2(10))=4 each → cost=50*(4+4)=400
        x = numpy.random.randn(50)
        y = numpy.random.randn(50)
        with BudgetContext(flop_budget=10**6) as budget:
            histogram2d(x, y, bins=10)
        assert budget.flops_used == 400

    def test_cost_int_pair_bins(self):
        # bins=[8, 16] → ceil(log2(8))=3, ceil(log2(16))=4 → cost=50*(3+4)=350
        x = numpy.random.randn(50)
        y = numpy.random.randn(50)
        with BudgetContext(flop_budget=10**6) as budget:
            histogram2d(x, y, bins=[8, 16])
        assert budget.flops_used == 350

    def test_cost_non_int_bins(self):
        # array bins of length 5 each → n * (ceil(log2(5)) + ceil(log2(5))) = 50*(3+3) = 300
        x = numpy.random.randn(50)
        y = numpy.random.randn(50)
        with BudgetContext(flop_budget=10**6) as budget:
            histogram2d(x, y, bins=[numpy.linspace(-3, 3, 5), numpy.linspace(-3, 3, 5)])
        assert budget.flops_used == 300


# ---------------------------------------------------------------------------
# histogramdd
# ---------------------------------------------------------------------------


class TestHistogramdd:
    def test_result_matches_numpy(self):
        sample = numpy.random.randn(50, 3)
        with BudgetContext(flop_budget=10**6):
            counts, edges = histogramdd(sample, bins=5)
        np_counts, np_edges = numpy.histogramdd(sample, bins=5)
        numpy.testing.assert_array_equal(counts, np_counts)

    def test_cost_int_bins(self):
        # n=50, d=3, bins=5 → ceil(log2(5))=3 → cost=50*3*3=450
        sample = numpy.random.randn(50, 3)
        with BudgetContext(flop_budget=10**6) as budget:
            histogramdd(sample, bins=5)
        assert budget.flops_used == 450

    def test_cost_non_int_bins(self):
        # list of ints [5,5,5] → n * (ceil(log2(5))*3) = 50*(3+3+3) = 450
        sample = numpy.random.randn(50, 3)
        with BudgetContext(flop_budget=10**6) as budget:
            histogramdd(sample, bins=[5, 5, 5])
        assert budget.flops_used == 450


# ---------------------------------------------------------------------------
# histogram_bin_edges
# ---------------------------------------------------------------------------


class TestHistogramBinEdges:
    def test_result_matches_numpy(self):
        a = numpy.random.randn(100)
        with BudgetContext(flop_budget=10**6):
            edges = histogram_bin_edges(a, bins=10)
        np_edges = numpy.histogram_bin_edges(a, bins=10)
        numpy.testing.assert_array_equal(edges, np_edges)

    def test_cost(self):
        a = numpy.random.randn(80)
        with BudgetContext(flop_budget=10**6) as budget:
            histogram_bin_edges(a)
        # numel(a), plus the default 11 edges charged as linspace
        assert budget.flops_used == 80 + 2 * 11


# ---------------------------------------------------------------------------
# bincount
# ---------------------------------------------------------------------------


class TestBincount:
    def test_result_matches_numpy(self):
        x = numpy.array([0, 1, 1, 2, 3, 3, 3])
        with BudgetContext(flop_budget=10**6):
            result = bincount(x)
        numpy.testing.assert_array_equal(result, numpy.bincount(x))

    def test_cost(self):
        x = numpy.array([0, 1, 2, 3, 4, 5, 6, 7])
        with BudgetContext(flop_budget=10**6) as budget:
            bincount(x)
        assert budget.flops_used == 8


# ---------------------------------------------------------------------------
# logspace
# ---------------------------------------------------------------------------


class TestLogspace:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            result = logspace(0, 2, num=50)
        numpy.testing.assert_allclose(result, numpy.logspace(0, 2, num=50))

    def test_cost_default_num(self):
        with BudgetContext(flop_budget=10**6) as budget:
            logspace(0, 2)
        assert budget.flops_used == 50

    def test_cost_custom_num(self):
        with BudgetContext(flop_budget=10**6) as budget:
            logspace(1, 3, num=100)
        assert budget.flops_used == 100


# ---------------------------------------------------------------------------
# geomspace
# ---------------------------------------------------------------------------


class TestGeomspace:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            result = geomspace(1, 1000, num=50)
        numpy.testing.assert_allclose(result, numpy.geomspace(1, 1000, num=50))

    def test_cost_default_num(self):
        with BudgetContext(flop_budget=10**6) as budget:
            geomspace(1, 1000)
        assert budget.flops_used == 50

    def test_cost_custom_num(self):
        with BudgetContext(flop_budget=10**6) as budget:
            geomspace(1, 100, num=75)
        assert budget.flops_used == 75


# ---------------------------------------------------------------------------
# vander
# ---------------------------------------------------------------------------


class TestVander:
    def test_result_matches_numpy(self):
        x = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with BudgetContext(flop_budget=10**6):
            result = vander(x, N=4)
        numpy.testing.assert_array_equal(result, numpy.vander(x, N=4))

    def test_cost_explicit_N(self):
        # x=[1,2,3,4,5], N=4 → cost=5*(4-2)=10 (x^0,x^1 free; x^2,x^3 multiply)
        x = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with BudgetContext(flop_budget=10**6) as budget:
            vander(x, N=4)
        assert budget.flops_used == 10

    def test_cost_default_N(self):
        # x=[1,2,3,4], N=None→4 → cost=4*(4-2)=8
        x = numpy.array([1.0, 2.0, 3.0, 4.0])
        with BudgetContext(flop_budget=10**6) as budget:
            vander(x)
        assert budget.flops_used == 8

    def test_cost_N_is_one(self):
        # N=1 → cost = len(x) * (1-2) = -3 → floor to 1
        x = numpy.array([1.0, 2.0, 3.0])
        with BudgetContext(flop_budget=10**6) as budget:
            vander(x, N=1)
        assert budget.flops_used == 1
