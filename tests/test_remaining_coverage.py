"""Tests targeting low-coverage modules to push total coverage toward 95%.

Covers uncovered branches in:
  - _counting_ops (array_equiv, histogram2d, histogramdd, apply_*, piecewise)
  - _validation (validate_ndarray, check_nan_inf, coerce_arrays)
  - _ndarray (reverse ops, in-place ops, __array_wrap__)
  - _config (unknown setting)
  - _docstrings (attach_docstring with empty docstring)
  - random (dims sampler no-args, size-only sampler, choice, bytes, shuffle, __getattr__)
  - linalg/_solvers (inv with symmetric, tensorsolve, tensorinv, lstsq, pinv)
  - linalg/_decompositions (svdvals with k, eigvalsh, qr)
  - linalg/_properties (norm variants, cond with p, matrix_rank, vector_norm, matrix_norm)
  - linalg/_aliases (cross, outer, tensordot, vecdot, diagonal, matrix_transpose)
  - fft/_transforms (hfft, ihfft, irfft2, irfftn, fft2 with s, batch helpers)
  - _sorting_ops (lexsort, partition, argpartition, digitize, set ops)
  - _opt_einsum/_contract (memory_limit branches, PathInfo formatting)

Note: _opt_einsum/_testing and _opt_einsum/_blas sections were removed when those
vendored modules were deleted in Task 7+8.
"""

from __future__ import annotations

import warnings

import numpy
import numpy as np
import pytest

from flopscope._budget import BudgetContext

# ============================================================================
# _counting_ops
# ============================================================================


class TestArrayEquivBroadcastFail:
    """Cover the ValueError branch when broadcast_shapes fails (lines 79-80)."""

    def test_incompatible_shapes_returns_false(self):
        # Shapes (2,3) and (4,5) cannot broadcast
        a = numpy.ones((2, 3))
        b = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import array_equiv

            result = array_equiv(a, b)
        assert result is np.False_ or result == False  # noqa: E712
        # Cost should be max(a.size, b.size, 1) = max(6, 20, 1) = 20
        assert budget.flops_used == 20


class TestHistogram2dTupleBinsNonInt:
    """Cover lines 135-145: tuple bins with non-int arrays and scalar fallback."""

    def test_tuple_bins_with_arrays(self):
        x = numpy.random.rand(20)
        y = numpy.random.rand(20)
        bins_x = numpy.linspace(0, 1, 5)
        bins_y = numpy.linspace(0, 1, 8)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import histogram2d

            histogram2d(x, y, bins=[bins_x, bins_y])
        assert budget.flops_used > 0

    def test_tuple_bins_with_scalar_fallback(self):
        """Cover lines 142-143: scalar bin entries in tuple."""
        x = numpy.random.rand(10)
        y = numpy.random.rand(10)
        # Pass scalars (0-d arrays) as bins -- hits the else branch
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import histogram2d

            histogram2d(x, y, bins=[numpy.array(5), numpy.array(5)])
        assert budget.flops_used == 10  # fallback: max(n, 1) = 10

    def test_non_tuple_bins_fallback(self):
        """Cover line 144-145: bins is not int, not list/tuple of length 2."""
        x = numpy.random.rand(10)
        y = numpy.random.rand(10)
        bins_arr = numpy.linspace(0, 1, 6)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import histogram2d

            histogram2d(x, y, bins=bins_arr)
        assert budget.flops_used == 10


class TestHistogramddNonIntBins:
    """Cover lines 178-183, 186: non-int bins in list and else fallback."""

    def test_list_bins_with_arrays(self):
        sample = numpy.random.rand(15, 2)
        bins = [numpy.linspace(0, 1, 5), numpy.linspace(0, 1, 8)]
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import histogramdd

            histogramdd(sample, bins=bins)
        assert budget.flops_used > 0

    def test_list_bins_with_empty_array(self):
        """Cover line 182-183: empty bin array -> total_log += 1."""
        sample = numpy.random.rand(10, 1)
        bins = [numpy.array(5)]  # 0-d array, ndim=0 -> total_log += 1
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import histogramdd

            histogramdd(sample, bins=bins)
        assert budget.flops_used > 0

    def test_1d_sample(self):
        """Cover lines 166-167: 1-d sample."""
        sample = numpy.random.rand(20)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import histogramdd

            histogramdd(sample, bins=10)
        assert budget.flops_used > 0

    def test_non_list_bins_fallback(self):
        """Cover line 186: bins is not int and not list/tuple."""
        sample = numpy.random.rand(10, 2)
        bins_arr = numpy.array(5)  # scalar, not list/tuple, not int
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import histogramdd

            histogramdd(sample, bins=bins_arr)
        assert budget.flops_used == 10


class TestApplyAlongAxis:
    """Cover lines 273-281: apply_along_axis cost = result.size."""

    def test_basic(self):
        a = numpy.arange(12).reshape(3, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import apply_along_axis

            result = apply_along_axis(numpy.sum, 1, a)
        assert result.shape == (3,)
        assert budget.flops_used == 3

    def test_with_list_input(self):
        """Cover line 274-275: non-ndarray input conversion."""
        a = [[1, 2, 3], [4, 5, 6]]
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import apply_along_axis

            result = apply_along_axis(numpy.sum, 0, a)
        assert budget.flops_used == result.size


class TestApplyOverAxes:
    """Cover lines 294-300: apply_over_axes cost = result.size."""

    def test_basic(self):
        a = numpy.arange(24).reshape(2, 3, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import apply_over_axes

            result = apply_over_axes(numpy.sum, a, [0, 2])
        assert budget.flops_used == result.size

    def test_with_list_input(self):
        """Cover line 295-296: non-ndarray input conversion."""
        a = [[1, 2], [3, 4]]
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import apply_over_axes

            result = apply_over_axes(numpy.sum, a, [0])
        assert budget.flops_used == result.size


class TestPiecewise:
    """Cover lines 313-319: piecewise cost = x.size."""

    def test_basic(self):
        x = numpy.linspace(-2, 2, 10)
        condlist = [x < 0, x >= 0]
        funclist = [lambda x: -x, lambda x: x]
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import piecewise

            result = piecewise(x, condlist, funclist)
        assert budget.flops_used == 10
        numpy.testing.assert_array_equal(result, numpy.abs(x))

    def test_with_list_input(self):
        """Cover line 314-315: non-ndarray input conversion."""
        x = [1.0, -2.0, 3.0]
        condlist = [numpy.array([True, False, True]), numpy.array([False, True, False])]
        funclist = [lambda x: x * 2, lambda x: x * 3]
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope._counting_ops import piecewise

            piecewise(x, condlist, funclist)
        assert budget.flops_used == 3


# ============================================================================
# _validation
# ============================================================================


class TestValidateNdarray:
    """Cover lines 25-27: type validation raises TypeError."""

    def test_raises_for_list(self):
        from flopscope._validation import validate_ndarray

        with pytest.raises(TypeError, match="Expected numpy.ndarray"):
            validate_ndarray([1, 2, 3])

    def test_raises_for_int(self):
        from flopscope._validation import validate_ndarray

        with pytest.raises(TypeError, match="Expected numpy.ndarray"):
            validate_ndarray(42)

    def test_accepts_ndarray(self):
        from flopscope._validation import validate_ndarray

        validate_ndarray(numpy.array([1, 2, 3]))  # Should not raise


class TestCheckNanInf:
    """Cover lines 32-34, 39: NaN/Inf warnings and dtype skipping."""

    def test_float_with_nan(self):
        from flopscope._validation import check_nan_inf
        from flopscope.errors import FlopscopeWarning

        arr = numpy.array([1.0, float("nan"), 3.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_inf(arr, "test_op")
        assert len(w) == 1
        assert issubclass(w[0].category, FlopscopeWarning)
        assert "1 NaN" in str(w[0].message)

    def test_float_with_inf(self):
        from flopscope._validation import check_nan_inf

        arr = numpy.array([1.0, float("inf"), -float("inf")])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_inf(arr, "test_op")
        assert len(w) == 1
        assert "2 Inf" in str(w[0].message)

    def test_complex_with_nan(self):
        from flopscope._validation import check_nan_inf

        arr = numpy.array([1 + 0j, float("nan") + 0j])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_inf(arr, "test_op")
        assert len(w) == 1

    def test_integer_dtype_skipped(self):
        from flopscope._validation import check_nan_inf

        arr = numpy.array([1, 2, 3], dtype=numpy.int64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_inf(arr, "test_op")
        assert len(w) == 0

    def test_object_dtype_skipped(self):
        from flopscope._validation import check_nan_inf

        arr = numpy.array(["a", "b", "c"], dtype=object)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_inf(arr, "test_op")
        assert len(w) == 0

    def test_non_array_skipped(self):
        from flopscope._validation import check_nan_inf

        # Non-ndarray input should be silently skipped
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_inf(42, "test_op")  # pyright: ignore[reportArgumentType]
        assert len(w) == 0


class TestMaybeCheckNanInf:
    """Cover the gated wrapper around check_nan_inf (issue #77)."""

    def test_default_off_skips_check(self):
        from flopscope._config import get_setting
        from flopscope._validation import maybe_check_nan_inf

        assert get_setting("check_nan_inf") is False
        arr = numpy.array([1.0, float("nan"), 3.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            maybe_check_nan_inf(arr, "test_op")
        assert len(w) == 0

    def test_when_on_runs_check(self):
        import flopscope as flops
        from flopscope._config import get_setting
        from flopscope._validation import maybe_check_nan_inf
        from flopscope.errors import FlopscopeWarning

        original = get_setting("check_nan_inf")
        try:
            flops.configure(check_nan_inf=True)
            arr = numpy.array([1.0, float("nan"), 3.0])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                maybe_check_nan_inf(arr, "test_op")
            assert len(w) == 1
            assert issubclass(w[0].category, FlopscopeWarning)
            assert "1 NaN" in str(w[0].message)
        finally:
            flops.configure(check_nan_inf=original)

    def test_non_array_skipped_when_on(self):
        import flopscope as flops
        from flopscope._config import get_setting
        from flopscope._validation import maybe_check_nan_inf

        original = get_setting("check_nan_inf")
        try:
            flops.configure(check_nan_inf=True)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                maybe_check_nan_inf(42, "test_op")
            assert len(w) == 0
        finally:
            flops.configure(check_nan_inf=original)

    def test_wrapper_op_silent_by_default(self):
        """Production-scoring path: a wrapper op that produces Inf does not warn."""
        import flopscope.numpy as fnp
        from flopscope.errors import FlopscopeWarning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with BudgetContext(flop_budget=10_000), np.errstate(divide="ignore"):
                fnp.divide(numpy.array([1.0]), numpy.array([0.0]))
        assert not any(
            issubclass(rec.category, FlopscopeWarning) and "Inf" in str(rec.message)
            for rec in w
        )

    def test_wrapper_op_warns_when_enabled(self):
        """Debug path: opting in surfaces the Inf-producing op."""
        import flopscope as flops
        import flopscope.numpy as fnp
        from flopscope._config import get_setting
        from flopscope.errors import FlopscopeWarning

        original = get_setting("check_nan_inf")
        try:
            flops.configure(check_nan_inf=True)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with BudgetContext(flop_budget=10_000), np.errstate(divide="ignore"):
                    fnp.divide(numpy.array([1.0]), numpy.array([0.0]))
            assert any(
                issubclass(rec.category, FlopscopeWarning) and "Inf" in str(rec.message)
                for rec in w
            )
        finally:
            flops.configure(check_nan_inf=original)


class TestCoerceArrays:
    """Cover line 39: coerce_arrays tuple conversion."""

    def test_coerce_multiple(self):
        from flopscope._validation import coerce_arrays

        a, b = coerce_arrays([1, 2], numpy.array([3, 4]))
        assert isinstance(a, numpy.ndarray)
        assert isinstance(b, numpy.ndarray)

    def test_coerce_preserves_ndarray(self):
        from flopscope._validation import coerce_array

        arr = numpy.array([1, 2, 3])
        result = coerce_array(arr)
        assert result is arr  # same object, not a copy


# ============================================================================
# _ndarray (reverse and in-place operators, __array_wrap__)
# ============================================================================


class TestFlopscopeArrayReverseOps:
    """Cover lines for __radd__, __rsub__, __rmul__, etc."""

    def test_radd(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([1.0, 2.0, 3.0])
            result = 5.0 + arr  # triggers __radd__
            numpy.testing.assert_array_equal(result, [6.0, 7.0, 8.0])

    def test_rsub(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([1.0, 2.0, 3.0])
            result = 10.0 - arr  # triggers __rsub__
            numpy.testing.assert_array_equal(result, [9.0, 8.0, 7.0])

    def test_rmul(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([1.0, 2.0, 3.0])
            result = 2.0 * arr  # triggers __rmul__
            numpy.testing.assert_array_equal(result, [2.0, 4.0, 6.0])

    def test_rtruediv(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([1.0, 2.0, 3.0])
            result = 6.0 / arr  # triggers __rtruediv__
            numpy.testing.assert_array_equal(result, [6.0, 3.0, 2.0])

    def test_rfloordiv(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([1.0, 2.0, 3.0])
            result = 7.0 // arr  # triggers __rfloordiv__
            numpy.testing.assert_array_equal(result, [7.0, 3.0, 2.0])

    def test_rmod(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([1.0, 2.0, 3.0])
            result = 7.0 % arr  # triggers __rmod__
            numpy.testing.assert_array_equal(result, [0.0, 1.0, 1.0])

    def test_rpow(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([1.0, 2.0, 3.0])
            result = 2.0**arr  # triggers __rpow__
            numpy.testing.assert_array_equal(result, [2.0, 4.0, 8.0])

    def test_rmatmul(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
            b = numpy.array([[5.0, 6.0], [7.0, 8.0]])
            result = b @ a  # triggers __rmatmul__ on a
            expected = numpy.array([[5.0, 6.0], [7.0, 8.0]]) @ numpy.array(
                [[1.0, 2.0], [3.0, 4.0]]
            )
            numpy.testing.assert_array_almost_equal(result, expected)


class TestFlopscopeArrayInPlaceOps:
    """Cover lines for __iadd__, __isub__, __imul__, etc."""

    def test_iadd(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([1.0, 2.0, 3.0])
            arr += 1.0
            numpy.testing.assert_array_equal(arr, [2.0, 3.0, 4.0])

    def test_isub(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([10.0, 20.0, 30.0])
            arr -= 5.0
            numpy.testing.assert_array_equal(arr, [5.0, 15.0, 25.0])

    def test_imul(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([1.0, 2.0, 3.0])
            arr *= 3.0
            numpy.testing.assert_array_equal(arr, [3.0, 6.0, 9.0])

    def test_itruediv(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([10.0, 20.0, 30.0])
            arr /= 10.0
            numpy.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_ifloordiv(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([10.0, 21.0, 35.0])
            arr //= 10.0
            numpy.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_imod(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([10.0, 21.0, 35.0])
            arr %= 10.0
            numpy.testing.assert_array_equal(arr, [0.0, 1.0, 5.0])

    def test_ipow(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([2.0, 3.0, 4.0])
            arr **= 2.0
            numpy.testing.assert_array_equal(arr, [4.0, 9.0, 16.0])

    def test_imatmul(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([[1.0, 0.0], [0.0, 1.0]])
            arr @= numpy.array([[2.0, 3.0], [4.0, 5.0]])
            numpy.testing.assert_array_equal(arr, [[2.0, 3.0], [4.0, 5.0]])


class TestFlopscopeArrayBitwiseOps:
    """Cover reverse/in-place bitwise and shift operators."""

    def test_rand(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([0b1010, 0b1100], dtype=numpy.int32)
            result = numpy.int32(0b1111) & arr
            numpy.testing.assert_array_equal(result, [0b1010, 0b1100])

    def test_iand(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([0b1010, 0b1100], dtype=numpy.int32)
            arr &= numpy.int32(0b1010)
            numpy.testing.assert_array_equal(arr, [0b1010, 0b1000])

    def test_ror(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([0b1010, 0b0001], dtype=numpy.int32)
            result = numpy.int32(0b0100) | arr
            numpy.testing.assert_array_equal(result, [0b1110, 0b0101])

    def test_ior(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([0b1010, 0b0001], dtype=numpy.int32)
            arr |= numpy.int32(0b0100)
            numpy.testing.assert_array_equal(arr, [0b1110, 0b0101])

    def test_rxor(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([0b1010, 0b1100], dtype=numpy.int32)
            result = numpy.int32(0b1111) ^ arr
            numpy.testing.assert_array_equal(result, [0b0101, 0b0011])

    def test_ixor(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([0b1010, 0b1100], dtype=numpy.int32)
            arr ^= numpy.int32(0b1111)
            numpy.testing.assert_array_equal(arr, [0b0101, 0b0011])

    def test_rlshift(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([1, 2], dtype=numpy.int32)
            result = numpy.int32(1).__lshift__(arr)  # pyright: ignore[reportCallIssue, reportArgumentType]
            # numpy.int32(1) << arr doesn't trigger __rlshift__,
            # but fnp can still test __lshift__ and __ilshift__
            arr2 = fnp.array([1, 2], dtype=numpy.int32)
            result2 = arr2 << numpy.int32(2)
            numpy.testing.assert_array_equal(result2, [4, 8])

    def test_ilshift(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([1, 2], dtype=numpy.int32)
            arr <<= numpy.int32(2)
            numpy.testing.assert_array_equal(arr, [4, 8])

    def test_rrshift(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([4, 8], dtype=numpy.int32)
            result = arr >> numpy.int32(1)
            numpy.testing.assert_array_equal(result, [2, 4])

    def test_irshift(self):
        import flopscope.numpy as fnp

        with BudgetContext(flop_budget=10**9):
            arr = fnp.array([4, 8], dtype=numpy.int32)
            arr >>= numpy.int32(1)
            numpy.testing.assert_array_equal(arr, [2, 4])


class TestArrayWrapReturnScalar:
    """Cover line 62-63: __array_wrap__ with return_scalar=True."""

    def test_return_scalar(self):
        from flopscope._ndarray import FlopscopeArray

        arr = numpy.array([42.0]).view(FlopscopeArray)
        # Calling __array_wrap__ with return_scalar=True should return a scalar
        result = arr.__array_wrap__(numpy.array(42.0), return_scalar=True)
        assert isinstance(result, (int, float, numpy.floating))

    def test_return_non_scalar(self):
        from flopscope._ndarray import FlopscopeArray

        arr = numpy.array([1.0, 2.0]).view(FlopscopeArray)
        out = numpy.array([3.0, 4.0])
        result = arr.__array_wrap__(out, return_scalar=False)
        assert isinstance(result, numpy.ndarray)


# ============================================================================
# _config
# ============================================================================


class TestConfigureUnknownSetting:
    """Cover line 26: unknown setting raises ValueError."""

    def test_unknown_setting(self):
        from flopscope._config import configure

        with pytest.raises(ValueError, match="Unknown setting"):
            configure(nonexistent_option=True)


class TestCheckNanInfSetting:
    """Cover the new check_nan_inf config flag (issue #77)."""

    def test_default_false(self):
        from flopscope._config import get_setting

        assert get_setting("check_nan_inf") is False

    def test_can_be_toggled(self):
        import flopscope as flops
        from flopscope._config import get_setting

        original = get_setting("check_nan_inf")
        try:
            flops.configure(check_nan_inf=True)
            assert get_setting("check_nan_inf") is True
            flops.configure(check_nan_inf=False)
            assert get_setting("check_nan_inf") is False
        finally:
            flops.configure(check_nan_inf=original)


# ============================================================================
# _docstrings
# ============================================================================


class TestAttachDocstringEmpty:
    """Cover lines 18-21: attach_docstring when np_func has empty docstring."""

    def test_empty_docstring(self):
        from flopscope._docstrings import attach_docstring

        def dummy():
            pass

        class FakeNpFunc:
            __name__ = "fake_func"
            __doc__ = ""

        attach_docstring(dummy, FakeNpFunc(), "test", "10 FLOPs")
        assert "FLOP Cost" in dummy.__doc__  # pyright: ignore[reportOperatorIssue]
        assert "fake_func" in dummy.__doc__  # pyright: ignore[reportOperatorIssue]

    def test_none_docstring(self):
        from flopscope._docstrings import attach_docstring

        def dummy():
            pass

        class FakeNpFunc:
            __name__ = "fake_func2"
            __doc__ = None

        attach_docstring(dummy, FakeNpFunc(), "test", "5 FLOPs")
        assert "FLOP Cost" in dummy.__doc__  # pyright: ignore[reportOperatorIssue]


# ============================================================================
# random
# ============================================================================


class TestRandomDimsSamplerNoArgs:
    """Cover line 79: dims sampler with no args (returns scalar)."""

    def test_rand_no_args(self):
        from flopscope.numpy import random as merandom

        with BudgetContext(flop_budget=10**6) as budget:
            result = merandom.rand()
        assert isinstance(result, float)
        assert budget.flops_used == 1

    def test_randn_no_args(self):
        from flopscope.numpy import random as merandom

        with BudgetContext(flop_budget=10**6) as budget:
            result = merandom.randn()
        assert isinstance(result, float)
        assert budget.flops_used == 1


class TestRandomSizeOnlySampler:
    """Cover lines 161-165: size-only samplers."""

    def test_random_with_size(self):
        from flopscope.numpy import random as merandom

        with BudgetContext(flop_budget=10**6) as budget:
            result = merandom.random(size=(3, 4))
        assert result.shape == (3, 4)
        assert budget.flops_used == 12

    def test_random_no_size(self):
        from flopscope.numpy import random as merandom

        with BudgetContext(flop_budget=10**6) as budget:
            result = merandom.random()
        assert budget.flops_used == 1


class TestRandomChoice:
    """Cover lines 223-234: choice with replace=False."""

    def test_choice_without_replacement(self):
        from flopscope._flops import sort_cost
        from flopscope.numpy import random as merandom

        n = 20
        with BudgetContext(flop_budget=10**6) as budget:
            merandom.choice(n, size=5, replace=False)
        assert budget.flops_used == sort_cost(n)

    def test_choice_from_array(self):
        from flopscope.numpy import random as merandom

        arr = numpy.arange(10)
        with BudgetContext(flop_budget=10**6) as budget:
            merandom.choice(arr, size=3, replace=True)
        assert budget.flops_used == 3


class TestRandomBytes:
    """Cover lines 242-245: bytes sampler."""

    def test_bytes(self):
        from flopscope.numpy import random as merandom

        with BudgetContext(flop_budget=10**6) as budget:
            result = merandom.bytes(16)
        assert len(result) == 16
        assert budget.flops_used == 16


class TestRandomShuffle:
    """Cover lines 207: shuffle with list input."""

    def test_shuffle_list(self):
        from flopscope.numpy import random as merandom

        lst = list(range(10))
        with BudgetContext(flop_budget=10**6) as budget:
            merandom.shuffle(lst)
        assert budget.flops_used > 0


class TestRandomGetattr:
    """Cover line 255: __getattr__ fallback for unknown attrs."""

    def test_getattr_existing(self):
        from flopscope.numpy import random as merandom

        # Access something that exists in numpy.random but not explicitly wrapped
        assert hasattr(merandom, "get_state")

    def test_getattr_missing(self):
        from flopscope.numpy import random as merandom

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = merandom.totally_nonexistent_attr_xyz


class TestRandomOutputSize:
    """Cover lines 35-37: _output_size with dims."""

    def test_output_size_with_dims(self):
        from flopscope.numpy.random import _output_size

        assert _output_size(3, 4, 5) == 60
        assert _output_size() == 1
        assert _output_size(size=(2, 3)) == 6
        assert _output_size(size=10) == 10


# ============================================================================
# linalg/_solvers
# ============================================================================


class TestLinalgSolversExtended:
    """Cover remaining linalg solver branches."""

    def test_inv_with_symmetric(self):
        from flopscope._symmetric import as_symmetric
        from flopscope.numpy.linalg._solvers import inv, inv_cost

        # Symmetric cost: n^3/3 + n^3
        n = 4
        a = numpy.eye(n) * 2.0
        sym_a = as_symmetric(a, symmetry=(0, 1))
        with BudgetContext(flop_budget=10**9) as budget:
            result = inv(sym_a)
        expected_cost = inv_cost(n, symmetric=True)
        assert budget.flops_used == expected_cost
        assert expected_cost == max(n**3 // 3 + n**3, 1)

    def test_lstsq(self):
        from flopscope.numpy.linalg._solvers import lstsq

        a = numpy.random.rand(5, 3)
        b = numpy.random.rand(5)
        with BudgetContext(flop_budget=10**9) as budget:
            lstsq(a, b, rcond=None)
        # lstsq_cost(5,3,b_cols=1,b_ndim=1): svd=45, ut_b=k*m*m=3*5*5=75, divide=3, recon=n*k*k=3*3*3=27 -> 150
        assert budget.flops_used == 150

    def test_pinv(self):
        from flopscope.numpy.linalg._solvers import pinv

        a = numpy.random.rand(4, 6)
        with BudgetContext(flop_budget=10**9) as budget:
            pinv(a)
        assert budget.flops_used == 292  # svd(4,6) + threshold + diag_scale + matmul(6,4,4)

    def test_tensorsolve(self):
        from flopscope.numpy.linalg._solvers import tensorsolve

        # a must be such that prod(a.shape[ind:]) == prod(b.shape)
        a = numpy.eye(24).reshape(6, 4, 6, 4)
        b = numpy.ones((6, 4))
        with BudgetContext(flop_budget=10**12) as budget:
            tensorsolve(a, b)
        assert budget.flops_used > 0

    def test_tensorinv(self):
        from flopscope.numpy.linalg._solvers import tensorinv

        a = numpy.eye(24).reshape(4, 6, 24)
        with BudgetContext(flop_budget=10**12) as budget:
            tensorinv(a, ind=2)
        assert budget.flops_used > 0


# ============================================================================
# linalg/_decompositions
# ============================================================================


class TestLinalgDecompositionsExtended:
    """Cover remaining decomposition branches."""

    def test_svdvals_with_k(self):
        from flopscope.numpy.linalg._decompositions import svdvals

        a = numpy.random.rand(6, 4)
        with BudgetContext(flop_budget=10**9) as budget:
            result = svdvals(a, k=2)
        assert len(result) == 2
        assert budget.flops_used == 6 * 4 * 2

    def test_svdvals_invalid_k(self):
        from flopscope.numpy.linalg._decompositions import svdvals

        a = numpy.random.rand(4, 3)
        with BudgetContext(flop_budget=10**9):
            with pytest.raises(ValueError, match="k must satisfy"):
                svdvals(a, k=5)

    def test_eigvalsh(self):
        from flopscope.numpy.linalg._decompositions import eigvalsh

        a = numpy.array([[2.0, 1.0], [1.0, 3.0]])
        with BudgetContext(flop_budget=10**9) as budget:
            eigvalsh(a)
        assert budget.flops_used == 8  # n^3 = 2^3

    def test_qr(self):
        from flopscope.numpy.linalg._decompositions import qr

        a = numpy.random.rand(5, 3)
        with BudgetContext(flop_budget=10**9) as budget:
            qr(a)
        assert budget.flops_used == 5 * 3 * 3  # m * n * min(m, n)


# ============================================================================
# linalg/_properties
# ============================================================================


class TestLinalgPropertiesExtended:
    """Cover remaining norm, cond, and matrix_rank branches."""

    def test_norm_vector_general_ord(self):
        from flopscope.numpy.linalg._properties import norm

        x = numpy.array([1.0, 2.0, 3.0])
        with BudgetContext(flop_budget=10**9) as budget:
            norm(x, ord=3)
        # FMA=2: all vector norms cost 2*numel
        assert budget.flops_used == 6  # FMA=2: 2*numel

    def test_norm_matrix_2(self):
        from flopscope.numpy.linalg._properties import norm

        a = numpy.random.rand(4, 3)
        with BudgetContext(flop_budget=10**9) as budget:
            norm(a, ord=2)
        # SVD-based: 4 * m * n * min(m, n)
        assert budget.flops_used == 4 * 4 * 3 * 3

    def test_norm_matrix_nuc(self):
        from flopscope.numpy.linalg._properties import norm

        a = numpy.random.rand(3, 5)
        with BudgetContext(flop_budget=10**9) as budget:
            norm(a, ord="nuc")
        assert budget.flops_used == 4 * 3 * 5 * 3

    def test_norm_matrix_1(self):
        from flopscope.numpy.linalg._properties import norm

        a = numpy.random.rand(3, 4)
        with BudgetContext(flop_budget=10**9) as budget:
            norm(a, ord=1)
        assert budget.flops_used == 24  # FMA=2: 2*numel

    def test_norm_with_axis_tuple(self):
        from flopscope.numpy.linalg._properties import norm

        a = numpy.random.rand(3, 4, 5)
        with BudgetContext(flop_budget=10**9) as budget:
            norm(a, axis=(1, 2))
        assert budget.flops_used > 0

    def test_cond_with_p_1(self):
        from flopscope.numpy.linalg._properties import cond

        a = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**9) as budget:
            cond(a, p=1)
        # LU-based: k^3 + m*n where k=min(m,n)=2
        assert budget.flops_used == 2**3 + 2 * 2

    def test_cond_with_p_inf(self):
        from flopscope.numpy.linalg._properties import cond

        a = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**9) as budget:
            cond(a, p=numpy.inf)
        assert budget.flops_used == 2**3 + 2 * 2

    def test_matrix_rank(self):
        from flopscope.numpy.linalg._properties import matrix_rank

        a = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**9) as budget:
            matrix_rank(a)
        assert budget.flops_used == 2 * 2 * 2  # m * n * min(m, n)

    def test_vector_norm(self):
        from flopscope.numpy.linalg._properties import vector_norm

        x = numpy.array([1.0, 2.0, 3.0])
        with BudgetContext(flop_budget=10**9) as budget:
            vector_norm(x, ord=3)
        # FMA=2: all norms cost 2*numel
        assert budget.flops_used == 6  # FMA=2: 2*numel

    def test_vector_norm_default(self):
        from flopscope.numpy.linalg._properties import vector_norm

        x = numpy.array([3.0, 4.0])
        with BudgetContext(flop_budget=10**9) as budget:
            vector_norm(x)
        assert budget.flops_used == 4  # FMA=2: 2*numel

    def test_vector_norm_with_axis(self):
        from flopscope.numpy.linalg._properties import vector_norm

        x = numpy.random.rand(3, 4)
        with BudgetContext(flop_budget=10**9) as budget:
            vector_norm(x, axis=1)
        assert budget.flops_used > 0

    def test_matrix_norm_fro(self):
        from flopscope.numpy.linalg._properties import matrix_norm

        a = numpy.random.rand(3, 4)
        with BudgetContext(flop_budget=10**9) as budget:
            matrix_norm(a)
        # FMA=2: Frobenius norm costs 2*numel
        assert budget.flops_used == 2 * 3 * 4  # FMA=2: 2*numel

    def test_matrix_norm_2(self):
        from flopscope.numpy.linalg._properties import matrix_norm

        a = numpy.random.rand(3, 4)
        with BudgetContext(flop_budget=10**9) as budget:
            matrix_norm(a, ord=2)
        assert budget.flops_used == 4 * 3 * 4 * 3

    def test_matrix_norm_nuc(self):
        from flopscope.numpy.linalg._properties import matrix_norm

        a = numpy.random.rand(3, 4)
        with BudgetContext(flop_budget=10**9) as budget:
            matrix_norm(a, ord="nuc")
        assert budget.flops_used == 4 * 3 * 4 * 3


# ============================================================================
# linalg/_aliases
# ============================================================================


class TestLinalgAliases:
    """Cover linalg alias functions that delegate to top-level flopscope."""

    def test_cross(self):
        from flopscope.numpy.linalg._aliases import cross

        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**9):
            result = cross(a, b)
        numpy.testing.assert_array_almost_equal(result, numpy.cross(a, b))

    def test_outer(self):
        from flopscope.numpy.linalg._aliases import outer

        a = numpy.array([1.0, 2.0])
        b = numpy.array([3.0, 4.0, 5.0])
        with BudgetContext(flop_budget=10**9):
            result = outer(a, b)
        numpy.testing.assert_array_almost_equal(result, numpy.outer(a, b))

    def test_tensordot(self):
        from flopscope.numpy.linalg._aliases import tensordot

        a = numpy.random.rand(3, 4)
        b = numpy.random.rand(4, 5)
        with BudgetContext(flop_budget=10**9):
            result = tensordot(a, b, axes=1)
        numpy.testing.assert_array_almost_equal(result, numpy.tensordot(a, b, axes=1))

    def test_vecdot(self):
        from flopscope.numpy.linalg._aliases import vecdot

        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**9):
            result = vecdot(a, b)
        assert float(result) == pytest.approx(numpy.vecdot(a, b))

    def test_diagonal(self):
        from flopscope.numpy.linalg._aliases import diagonal

        a = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**9):
            result = diagonal(a)
        numpy.testing.assert_array_equal(result, [1.0, 4.0])

    def test_matrix_transpose(self):
        from flopscope.numpy.linalg._aliases import matrix_transpose

        a = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**9):
            result = matrix_transpose(a)
        numpy.testing.assert_array_equal(result, a.T)


# ============================================================================
# fft/_transforms
# ============================================================================


class TestFFTExtended:
    """Cover remaining FFT transform branches."""

    def test_hfft(self):
        from flopscope.numpy.fft._transforms import hfft

        a = numpy.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
        with BudgetContext(flop_budget=10**9) as budget:
            hfft(a)
        assert budget.flops_used > 0

    def test_ihfft(self):
        from flopscope.numpy.fft._transforms import ihfft

        a = numpy.array([1.0, 2.0, 3.0, 4.0])
        with BudgetContext(flop_budget=10**9) as budget:
            ihfft(a)
        assert budget.flops_used > 0

    def test_irfft2(self):
        from flopscope.numpy.fft._transforms import irfft2

        a = numpy.random.rand(4, 3) + 1j * numpy.random.rand(4, 3)
        with BudgetContext(flop_budget=10**9) as budget:
            irfft2(a)
        assert budget.flops_used > 0

    def test_irfftn(self):
        from flopscope.numpy.fft._transforms import irfftn

        a = numpy.random.rand(4, 3, 5) + 1j * numpy.random.rand(4, 3, 5)
        with BudgetContext(flop_budget=10**9) as budget:
            irfftn(a)
        assert budget.flops_used > 0

    def test_irfftn_with_axes(self):
        from flopscope.numpy.fft._transforms import irfftn

        a = numpy.random.rand(4, 3, 5) + 1j * numpy.random.rand(4, 3, 5)
        with BudgetContext(flop_budget=10**9) as budget:
            irfftn(a, axes=(0, 1, 2))
        assert budget.flops_used > 0

    def test_fft2_with_s(self):
        from flopscope.numpy.fft._transforms import fft2

        a = numpy.random.rand(4, 6)
        with BudgetContext(flop_budget=10**9) as budget:
            fft2(a, s=(8, 8))
        assert budget.flops_used > 0

    def test_ifft2_with_s(self):
        from flopscope.numpy.fft._transforms import ifft2

        a = numpy.random.rand(4, 6) + 1j * numpy.random.rand(4, 6)
        with BudgetContext(flop_budget=10**9) as budget:
            ifft2(a, s=(4, 6))
        assert budget.flops_used > 0

    def test_rfft2_with_s(self):
        from flopscope.numpy.fft._transforms import rfft2

        a = numpy.random.rand(4, 6)
        with BudgetContext(flop_budget=10**9) as budget:
            rfft2(a, s=(8, 8))
        assert budget.flops_used > 0

    def test_irfft2_with_s(self):
        from flopscope.numpy.fft._transforms import irfft2

        a = numpy.random.rand(4, 5) + 1j * numpy.random.rand(4, 5)
        with BudgetContext(flop_budget=10**9) as budget:
            irfft2(a, s=(4, 8))
        assert budget.flops_used > 0

    def test_fftn_with_s_and_axes(self):
        from flopscope.numpy.fft._transforms import fftn

        a = numpy.random.rand(4, 5, 6)
        with BudgetContext(flop_budget=10**9) as budget:
            fftn(a, s=(4, 5), axes=(0, 1))
        assert budget.flops_used > 0

    def test_ifftn_with_s(self):
        from flopscope.numpy.fft._transforms import ifftn

        a = numpy.random.rand(4, 5) + 1j * numpy.random.rand(4, 5)
        with BudgetContext(flop_budget=10**9) as budget:
            ifftn(a, s=(4, 5))
        assert budget.flops_used > 0

    def test_rfftn_with_s(self):
        from flopscope.numpy.fft._transforms import rfftn

        a = numpy.random.rand(4, 5, 6)
        with BudgetContext(flop_budget=10**9) as budget:
            rfftn(a, s=(4, 5, 6))
        assert budget.flops_used > 0

    def test_irfftn_with_s(self):
        from flopscope.numpy.fft._transforms import irfftn

        a = numpy.random.rand(4, 5, 3) + 1j * numpy.random.rand(4, 5, 3)
        with BudgetContext(flop_budget=10**9) as budget:
            irfftn(a, s=(4, 5, 4))
        assert budget.flops_used > 0


# ============================================================================
# _sorting_ops
# ============================================================================


class TestSortingOpsExtended:
    """Cover remaining sorting/search branches."""

    def test_lexsort(self):
        from flopscope._sorting_ops import lexsort

        a = numpy.array([1, 2, 3, 1, 2])
        b = numpy.array([5, 4, 3, 2, 1])
        with BudgetContext(flop_budget=10**9) as budget:
            lexsort((a, b))
        assert budget.flops_used > 0

    def test_lexsort_single_key(self):
        from flopscope._sorting_ops import lexsort

        a = numpy.array([3, 1, 2])
        with BudgetContext(flop_budget=10**9) as budget:
            lexsort((a,))
        assert budget.flops_used > 0

    def test_partition(self):
        from flopscope._sorting_ops import partition

        a = numpy.array([3, 1, 2, 5, 4])
        with BudgetContext(flop_budget=10**9) as budget:
            partition(a, 2)
        assert budget.flops_used > 0

    def test_partition_from_list(self):
        from flopscope._sorting_ops import partition

        a = [3, 1, 2, 5, 4]  # non-ndarray input triggers coercion
        with BudgetContext(flop_budget=10**9) as budget:
            partition(a, 2)
        assert budget.flops_used > 0

    def test_argpartition(self):
        from flopscope._sorting_ops import argpartition

        a = numpy.array([3, 1, 2, 5, 4])
        with BudgetContext(flop_budget=10**9) as budget:
            argpartition(a, 2)
        assert budget.flops_used > 0

    def test_argpartition_from_list(self):
        from flopscope._sorting_ops import argpartition

        a = [3, 1, 2, 5, 4]  # non-ndarray input triggers coercion
        with BudgetContext(flop_budget=10**9) as budget:
            argpartition(a, 2)
        assert budget.flops_used > 0

    def test_digitize(self):
        from flopscope._sorting_ops import digitize

        x = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins = numpy.array([0.0, 1.0, 2.5, 4.0, 10.0])
        with BudgetContext(flop_budget=10**9) as budget:
            digitize(x, bins)
        assert budget.flops_used > 0

    def test_setdiff1d(self):
        from flopscope._sorting_ops import setdiff1d

        a = numpy.array([1, 2, 3, 4, 5])
        b = numpy.array([3, 4])
        with BudgetContext(flop_budget=10**9) as budget:
            result = setdiff1d(a, b)
        numpy.testing.assert_array_equal(result, [1, 2, 5])
        assert budget.flops_used > 0

    def test_setxor1d(self):
        from flopscope._sorting_ops import setxor1d

        a = numpy.array([1, 2, 3])
        b = numpy.array([2, 3, 4])
        with BudgetContext(flop_budget=10**9) as budget:
            result = setxor1d(a, b)
        numpy.testing.assert_array_equal(result, [1, 4])
        assert budget.flops_used > 0

    def test_sort_axis_none(self):
        from flopscope._sorting_ops import sort

        a = numpy.array([[3, 1], [2, 4]])
        with BudgetContext(flop_budget=10**9) as budget:
            sort(a, axis=None)
        assert budget.flops_used > 0

    def test_argsort_axis_none(self):
        from flopscope._sorting_ops import argsort

        a = numpy.array([[3, 1], [2, 4]])
        with BudgetContext(flop_budget=10**9) as budget:
            argsort(a, axis=None)
        assert budget.flops_used > 0

    def test_sort_from_list(self):
        from flopscope._sorting_ops import sort

        a = [3, 1, 2]  # non-ndarray input triggers coercion
        with BudgetContext(flop_budget=10**9) as budget:
            sort(a)
        assert budget.flops_used > 0

    def test_argsort_from_list(self):
        from flopscope._sorting_ops import argsort

        a = [3, 1, 2]  # non-ndarray input triggers coercion
        with BudgetContext(flop_budget=10**9) as budget:
            argsort(a)
        assert budget.flops_used > 0


# ============================================================================
# _opt_einsum/_contract
# ============================================================================


class TestContractPathMemoryLimit:
    """Cover memory_limit handling (now delegated to upstream opt_einsum)."""

    def test_memory_limit_max_input(self):
        from flopscope._opt_einsum import contract_path

        path, info = contract_path(
            "ij,jk,kl->il",
            (2, 3),
            (3, 4),
            (4, 5),
            shapes=True,
            memory_limit="max_input",
        )
        assert len(path) > 0

    def test_memory_limit_explicit(self):
        from flopscope._opt_einsum import contract_path

        path, info = contract_path(
            "ij,jk,kl->il",
            (2, 3),
            (3, 4),
            (4, 5),
            shapes=True,
            memory_limit=100,
        )
        assert len(path) > 0

    def test_memory_limit_invalid_string(self):
        # _choose_memory_arg was removed along with local contract_path.
        # Upstream opt_einsum raises ValueError for invalid memory_limit strings.
        from flopscope._opt_einsum import contract_path

        with pytest.raises((ValueError, TypeError)):
            contract_path(
                "ij,jk,kl->il",
                (2, 3),
                (3, 4),
                (4, 5),
                shapes=True,
                memory_limit="invalid",  # type: ignore[arg-type]
            )

    def test_memory_limit_negative(self):
        # -1 is accepted by upstream as "no limit".
        from flopscope._opt_einsum import contract_path

        path, info = contract_path(
            "ij,jk->ik", (3, 4), (4, 5), shapes=True, memory_limit=-1
        )
        assert len(path) > 0

    def test_memory_limit_negative_invalid(self):
        # Values other than -1 and None: upstream raises ValueError.
        from flopscope._opt_einsum import contract_path

        with pytest.raises((ValueError, TypeError)):
            contract_path("ij,jk->ik", (3, 4), (4, 5), shapes=True, memory_limit=-2)


class TestPathInfoFormatTable:
    """Cover PathInfo.format_table and __repr__."""

    def test_format_table(self):
        from flopscope._opt_einsum import contract_path

        path, info = contract_path("ij,jk,kl->il", (2, 3), (3, 4), (4, 5), shapes=True)
        table = info.format_table()
        assert "Naive cost" in table
        assert "Optimized cost" in table

    def test_format_table_verbose(self):
        from flopscope._opt_einsum import contract_path

        path, info = contract_path("ij,jk,kl->il", (2, 3), (3, 4), (4, 5), shapes=True)
        table = info.format_table(verbose=True)
        assert "subset=" in table
        assert "cumulative=" in table

    def test_repr(self):
        from flopscope._opt_einsum import contract_path

        path, info = contract_path("ij,jk->ik", (3, 4), (4, 5), shapes=True)
        repr_str = repr(info)
        assert "contraction" in repr_str.lower() or "cost" in repr_str.lower()

    def test_optimize_false(self):
        from flopscope._opt_einsum import contract_path

        path, info = contract_path(
            "ij,jk->ik", (3, 4), (4, 5), shapes=True, optimize=False
        )
        assert len(path) == 1

    def test_opt_cost_property(self):
        from flopscope._opt_einsum import contract_path

        _, info = contract_path("ij,jk->ik", (3, 4), (4, 5), shapes=True)
        assert info.opt_cost >= 0

    def test_eq_property(self):
        from flopscope._opt_einsum import contract_path

        _, info = contract_path("ij,jk->ik", (3, 4), (4, 5), shapes=True)
        assert info.eq == "ij,jk->ik"
