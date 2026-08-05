"""Tests for counted sorting, search, and set operation wrappers."""

from __future__ import annotations

import numpy
import pytest

from flopscope._budget import BudgetContext
from flopscope._flops import search_cost, sort_cost
from flopscope._sorting_ops import (
    argpartition,
    argsort,
    digitize,
    in1d,
    intersect1d,
    isin,
    lexsort,
    partition,
    searchsorted,
    setdiff1d,
    setxor1d,
    sort,
    union1d,
    unique,
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
)

# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestSort:
    def test_result_matches_numpy(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(sort(a), numpy.sort(a))

    def test_cost_1d(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6])
        n = len(a)
        expected = sort_cost(n)
        with BudgetContext(flop_budget=10**6) as budget:
            sort(a)
            assert budget.flops_used == expected

    def test_cost_2d_axis0(self):
        a = numpy.array([[3, 1, 4], [1, 5, 9]])
        # axis=0: n=2, num_slices=3
        n = a.shape[0]
        num_slices = a.shape[1]
        expected = num_slices * sort_cost(n)
        with BudgetContext(flop_budget=10**6) as budget:
            sort(a, axis=0)
            assert budget.flops_used == expected

    def test_cost_2d_axis1(self):
        a = numpy.array([[3, 1, 4], [1, 5, 9]])
        # axis=1: n=3, num_slices=2
        n = a.shape[1]
        num_slices = a.shape[0]
        expected = num_slices * sort_cost(n)
        with BudgetContext(flop_budget=10**6) as budget:
            sort(a, axis=1)
            assert budget.flops_used == expected

    def test_result_2d(self):
        a = numpy.array([[3, 1, 4], [1, 5, 9]])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(sort(a), numpy.sort(a))

    def test_order_structured_dtype_is_refused(self):
        """sort(order=...) only means anything for a structured dtype, which
        is now refused unconditionally (kind 'V' is outside the numeric
        allowlist, object-free fields included) -- there is no numeric
        substitute for a keyword that exists solely to name a struct field."""
        from flopscope.errors import UnsupportedDtypeError

        dt = numpy.dtype([("a", "i4"), ("b", "i4")])
        s = numpy.array([(2, 1), (1, 5), (2, 0)], dtype=dt)
        with BudgetContext(flop_budget=10**9, quiet=True):
            with pytest.raises(UnsupportedDtypeError):
                sort(s, order="b")

    def test_kind_stable_float(self):
        """sort(kind='stable') must forward to numpy — duplicate-preserving sort."""
        a = numpy.array([3.0, 1.0, 2.0, 1.0, 3.0, 2.0])
        with BudgetContext(flop_budget=10**9, quiet=True):
            result = numpy.asarray(sort(a, kind="stable"))
        expected = numpy.sort(a, kind="stable")
        numpy.testing.assert_array_equal(result, expected)


class TestArgsort:
    def test_result_matches_numpy(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(argsort(a), numpy.argsort(a))

    def test_cost_1d(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6])
        n = len(a)
        expected = sort_cost(n)
        with BudgetContext(flop_budget=10**6) as budget:
            argsort(a)
            assert budget.flops_used == expected

    def test_cost_2d(self):
        a = numpy.array([[3, 1, 4], [1, 5, 9]])
        # default axis=-1 => axis=1: n=3, num_slices=2
        n = a.shape[1]
        num_slices = a.shape[0]
        expected = num_slices * sort_cost(n)
        with BudgetContext(flop_budget=10**6) as budget:
            argsort(a)
            assert budget.flops_used == expected


class TestLexsort:
    def test_result_matches_numpy(self):
        a = numpy.array([3, 1, 4, 1, 5])
        b = numpy.array([2, 2, 1, 1, 0])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(lexsort((a, b)), numpy.lexsort((a, b)))

    def test_cost(self):
        a = numpy.array([3, 1, 4, 1, 5])
        b = numpy.array([2, 2, 1, 1, 0])
        k = 2
        n = len(a)
        expected = k * sort_cost(n)
        with BudgetContext(flop_budget=10**6) as budget:
            lexsort((a, b))
            assert budget.flops_used == expected

    def test_single_key(self):
        a = numpy.array([3, 1, 4, 1, 5])
        k = 1
        n = len(a)
        expected = k * sort_cost(n)
        with BudgetContext(flop_budget=10**6) as budget:
            lexsort((a,))
            assert budget.flops_used == expected


class TestPartition:
    def test_result_matches_numpy(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6])
        with BudgetContext(flop_budget=10**6):
            result = partition(a, 3)
            expected = numpy.partition(a, 3)
            # partition only guarantees element at kth is in place; check that
            assert result[3] == expected[3]

    def test_cost_1d(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6])
        n = len(a)
        expected = n  # 1 slice * n
        with BudgetContext(flop_budget=10**6) as budget:
            partition(a, 3)
            assert budget.flops_used == expected

    def test_cost_2d(self):
        a = numpy.array([[3, 1, 4], [1, 5, 9]])
        # axis=-1 => axis=1: n=3, num_slices=2
        n = a.shape[1]
        num_slices = a.shape[0]
        expected = num_slices * n
        with BudgetContext(flop_budget=10**6) as budget:
            partition(a, 1)
            assert budget.flops_used == expected


class TestArgpartition:
    def test_result_matches_numpy(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6])
        with BudgetContext(flop_budget=10**6):
            result = argpartition(a, 3)
            expected = numpy.argpartition(a, 3)
            assert a[result[3]] == a[expected[3]]

    def test_cost_1d(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6])
        n = len(a)
        expected = n
        with BudgetContext(flop_budget=10**6) as budget:
            argpartition(a, 3)
            assert budget.flops_used == expected

    def test_cost_2d(self):
        a = numpy.array([[3, 1, 4], [1, 5, 9]])
        n = a.shape[1]
        num_slices = a.shape[0]
        expected = num_slices * n
        with BudgetContext(flop_budget=10**6) as budget:
            argpartition(a, 1)
            assert budget.flops_used == expected


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearchsorted:
    def test_result_matches_numpy(self):
        a = numpy.array([1, 2, 3, 4, 5])
        v = numpy.array([1.5, 2.5])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(searchsorted(a, v), numpy.searchsorted(a, v))

    def test_cost(self):
        a = numpy.array([1, 2, 3, 4, 5])
        v = numpy.array([1.5, 2.5, 3.5])
        n = len(a)
        m = len(v)
        expected = search_cost(m, n)
        with BudgetContext(flop_budget=10**6) as budget:
            searchsorted(a, v)
            assert budget.flops_used == expected

    def test_scalar_query(self):
        a = numpy.array([1, 2, 3, 4, 5])
        v = 2.5
        n = len(a)
        m = 1
        expected = search_cost(m, n)
        with BudgetContext(flop_budget=10**6) as budget:
            searchsorted(a, v)
            assert budget.flops_used == expected


class TestDigitize:
    def test_result_matches_numpy(self):
        x = numpy.array([0.5, 1.5, 2.5, 3.5])
        bins = numpy.array([1.0, 2.0, 3.0])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(digitize(x, bins), numpy.digitize(x, bins))

    def test_cost(self):
        x = numpy.array([0.5, 1.5, 2.5, 3.5])
        bins = numpy.array([1.0, 2.0, 3.0])
        n = len(x)
        b = len(bins)
        expected = search_cost(n, b)
        with BudgetContext(flop_budget=10**6) as budget:
            digitize(x, bins)
            assert budget.flops_used == expected


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------


class TestUnique:
    def test_result_matches_numpy(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(unique(a), numpy.unique(a))

    def test_cost(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
        n = a.size
        expected = sort_cost(n)
        with BudgetContext(flop_budget=10**6) as budget:
            unique(a)
            assert budget.flops_used == expected

    def test_with_kwargs(self):
        a = numpy.array([3, 1, 4, 1, 5])
        with BudgetContext(flop_budget=10**6):
            vals, counts = unique(a, return_counts=True)
            assert numpy.array_equal(vals, numpy.unique(a))


class TestUniqueAll:
    def test_result_matches_numpy(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
        with BudgetContext(flop_budget=10**6):
            result = unique_all(a)
            expected = numpy.unique_all(a)
            assert numpy.array_equal(result.values, expected.values)

    def test_cost(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
        n = a.size
        expected = sort_cost(n)
        with BudgetContext(flop_budget=10**6) as budget:
            unique_all(a)
            assert budget.flops_used == expected


class TestUniqueCounts:
    def test_result_matches_numpy(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
        with BudgetContext(flop_budget=10**6):
            result = unique_counts(a)
            expected = numpy.unique_counts(a)
            assert numpy.array_equal(result.values, expected.values)
            assert numpy.array_equal(result.counts, expected.counts)

    def test_cost(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
        n = a.size
        expected = sort_cost(n)
        with BudgetContext(flop_budget=10**6) as budget:
            unique_counts(a)
            assert budget.flops_used == expected


class TestUniqueInverse:
    def test_result_matches_numpy(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
        with BudgetContext(flop_budget=10**6):
            result = unique_inverse(a)
            expected = numpy.unique_inverse(a)
            assert numpy.array_equal(result.values, expected.values)
            assert numpy.array_equal(result.inverse_indices, expected.inverse_indices)

    def test_cost(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
        n = a.size
        expected = sort_cost(n)
        with BudgetContext(flop_budget=10**6) as budget:
            unique_inverse(a)
            assert budget.flops_used == expected


class TestUniqueValues:
    def test_result_matches_numpy(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
        with BudgetContext(flop_budget=10**6):
            result = unique_values(a)
            expected = numpy.unique_values(a)
            assert numpy.array_equal(result, expected)

    def test_cost(self):
        a = numpy.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
        n = a.size
        expected = sort_cost(n)
        with BudgetContext(flop_budget=10**6) as budget:
            unique_values(a)
            assert budget.flops_used == expected


# ---------------------------------------------------------------------------
# Set operations
# ---------------------------------------------------------------------------


def _set_op_expected_cost(a1, a2):
    n = numpy.asarray(a1).size
    m = numpy.asarray(a2).size
    return sort_cost(n + m)


@pytest.mark.skipif(not hasattr(numpy, "in1d"), reason="numpy 2.4+ removed in1d")
class TestIn1d:
    def test_result_matches_numpy(self):
        ar1 = numpy.array([1, 2, 3, 4, 5])
        ar2 = numpy.array([2, 4])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(in1d(ar1, ar2), numpy.in1d(ar1, ar2))

    def test_cost(self):
        ar1 = numpy.array([1, 2, 3, 4, 5])
        ar2 = numpy.array([2, 4])
        expected = _set_op_expected_cost(ar1, ar2)
        with BudgetContext(flop_budget=10**6) as budget:
            in1d(ar1, ar2)
            assert budget.flops_used == expected


class TestIsin:
    def test_result_matches_numpy(self):
        element = numpy.array([1, 2, 3, 4, 5])
        test_elements = numpy.array([2, 4])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(
                isin(element, test_elements), numpy.isin(element, test_elements)
            )

    def test_cost(self):
        element = numpy.array([1, 2, 3, 4, 5])
        test_elements = numpy.array([2, 4])
        expected = _set_op_expected_cost(element, test_elements)
        with BudgetContext(flop_budget=10**6) as budget:
            isin(element, test_elements)
            assert budget.flops_used == expected


class TestIntersect1d:
    def test_result_matches_numpy(self):
        ar1 = numpy.array([1, 2, 3, 4])
        ar2 = numpy.array([3, 4, 5, 6])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(intersect1d(ar1, ar2), numpy.intersect1d(ar1, ar2))

    def test_cost(self):
        # Default assume_unique=False: numpy unique()-sorts both inputs first,
        # so honest cost = sort_cost(n) + sort_cost(m) + sort_cost(n+m).
        ar1 = numpy.array([1, 2, 3, 4])
        ar2 = numpy.array([3, 4, 5, 6])
        n, m = ar1.size, ar2.size
        expected = sort_cost(n) + sort_cost(m) + sort_cost(n + m)
        with BudgetContext(flop_budget=10**6) as budget:
            intersect1d(ar1, ar2)
            assert budget.flops_used == expected

    def test_cost_assume_unique(self):
        # assume_unique=True: no pre-sorting, only sort_cost(n+m).
        ar1 = numpy.array([1, 2, 3, 4])
        ar2 = numpy.array([3, 4, 5, 6])
        n, m = ar1.size, ar2.size
        expected = sort_cost(n + m)
        with BudgetContext(flop_budget=10**6) as budget:
            intersect1d(ar1, ar2, assume_unique=True)
            assert budget.flops_used == expected


class TestUnion1d:
    def test_result_matches_numpy(self):
        ar1 = numpy.array([1, 2, 3, 4])
        ar2 = numpy.array([3, 4, 5, 6])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(union1d(ar1, ar2), numpy.union1d(ar1, ar2))

    def test_cost(self):
        ar1 = numpy.array([1, 2, 3, 4])
        ar2 = numpy.array([3, 4, 5, 6])
        expected = _set_op_expected_cost(ar1, ar2)
        with BudgetContext(flop_budget=10**6) as budget:
            union1d(ar1, ar2)
            assert budget.flops_used == expected


class TestSetdiff1d:
    def test_result_matches_numpy(self):
        ar1 = numpy.array([1, 2, 3, 4])
        ar2 = numpy.array([3, 4, 5, 6])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(setdiff1d(ar1, ar2), numpy.setdiff1d(ar1, ar2))

    def test_cost(self):
        ar1 = numpy.array([1, 2, 3, 4])
        ar2 = numpy.array([3, 4, 5, 6])
        expected = _set_op_expected_cost(ar1, ar2)
        with BudgetContext(flop_budget=10**6) as budget:
            setdiff1d(ar1, ar2)
            assert budget.flops_used == expected


class TestSetxor1d:
    def test_result_matches_numpy(self):
        ar1 = numpy.array([1, 2, 3, 4])
        ar2 = numpy.array([3, 4, 5, 6])
        with BudgetContext(flop_budget=10**6):
            assert numpy.array_equal(setxor1d(ar1, ar2), numpy.setxor1d(ar1, ar2))

    def test_cost(self):
        ar1 = numpy.array([1, 2, 3, 4])
        ar2 = numpy.array([3, 4, 5, 6])
        expected = _set_op_expected_cost(ar1, ar2)
        with BudgetContext(flop_budget=10**6) as budget:
            setxor1d(ar1, ar2)
            assert budget.flops_used == expected
