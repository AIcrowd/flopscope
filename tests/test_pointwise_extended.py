"""Extended tests for uncovered counted pointwise operations.

Covers: modf, frexp, divmod, binary ops (copysign, heaviside, etc.),
convolve, correlate, corrcoef, cov, diff, gradient, interp, ediff1d,
trapezoid, and uncovered symmetric / scalar paths in factory functions.
"""

import numpy
import pytest

import flopscope._pointwise as ops
from flopscope._budget import BudgetContext
from flopscope._symmetric import SymmetricTensor, as_symmetric
from flopscope.errors import SymmetryLossWarning

# ---------------------------------------------------------------------------
# Multi-output unary ops
# ---------------------------------------------------------------------------


def test_modf():
    x = numpy.array([1.5, 2.7, -0.3])
    with BudgetContext(flop_budget=10**6) as budget:
        frac, intpart = ops.modf(x)
        # modf writes two output buffers (fractional + integral part) per
        # call, so it bills nout=2 * numel(x).
        assert budget.flops_used == 2 * x.size
    assert numpy.allclose(frac + intpart, x)


def test_frexp():
    x = numpy.array([1.0, 4.0, 0.5])
    with BudgetContext(flop_budget=10**6) as budget:
        mant, exp = ops.frexp(x)
        # frexp writes two output buffers (mantissa + exponent) per call, so
        # it bills nout=2 * numel(x).
        assert budget.flops_used == 2 * x.size
    assert numpy.allclose(mant * (2**exp), x)


# ---------------------------------------------------------------------------
# Multi-output binary ops
# ---------------------------------------------------------------------------


def test_divmod():
    x = numpy.array([7.0, 8.0, 9.0])
    y = numpy.array([3.0, 3.0, 3.0])
    with BudgetContext(flop_budget=10**6) as budget:
        quot, rem = ops.divmod(x, y)
        # divmod writes a quotient AND a remainder buffer per call, so it
        # bills nout=2 * numel(output).
        assert budget.flops_used == 2 * x.size
    assert numpy.allclose(quot * y + rem, x)


# ---------------------------------------------------------------------------
# Binary ops (new, previously uncovered)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_name,x,y",
    [
        ("copysign", numpy.array([1.0, -2.0, 3.0]), numpy.array([-1.0, 1.0, -1.0])),
        ("heaviside", numpy.array([-1.0, 0.0, 1.0]), numpy.array([0.5, 0.5, 0.5])),
        ("arctan2", numpy.array([1.0, 1.0]), numpy.array([1.0, -1.0])),
        ("atan2", numpy.array([1.0]), numpy.array([1.0])),
        ("float_power", numpy.array([2.0, 3.0]), numpy.array([3.0, 2.0])),
        ("floor_divide", numpy.array([7.0, 8.0]), numpy.array([2.0, 3.0])),
        ("fmax", numpy.array([1.0, float("nan")]), numpy.array([2.0, 1.0])),
        ("fmin", numpy.array([1.0, float("nan")]), numpy.array([2.0, 1.0])),
        ("fmod", numpy.array([7.0, 8.0]), numpy.array([3.0, 3.0])),
        ("hypot", numpy.array([3.0]), numpy.array([4.0])),
        ("logaddexp", numpy.array([0.0, 1.0]), numpy.array([0.0, 0.0])),
        ("logaddexp2", numpy.array([0.0, 1.0]), numpy.array([0.0, 0.0])),
        ("logical_and", numpy.array([True, False]), numpy.array([True, True])),
        ("logical_or", numpy.array([True, False]), numpy.array([False, False])),
        ("logical_xor", numpy.array([True, False]), numpy.array([True, True])),
        ("nextafter", numpy.array([1.0]), numpy.array([2.0])),
        ("not_equal", numpy.array([1.0, 2.0]), numpy.array([1.0, 3.0])),
        ("remainder", numpy.array([7.0]), numpy.array([3.0])),
        ("true_divide", numpy.array([6.0]), numpy.array([2.0])),
        ("equal", numpy.array([1.0, 2.0]), numpy.array([1.0, 3.0])),
        ("greater", numpy.array([2.0, 1.0]), numpy.array([1.0, 2.0])),
        ("greater_equal", numpy.array([2.0, 1.0]), numpy.array([2.0, 2.0])),
        ("less", numpy.array([1.0, 2.0]), numpy.array([2.0, 1.0])),
        ("less_equal", numpy.array([1.0, 2.0]), numpy.array([1.0, 1.0])),
    ],
)
def test_binary_op_runs(op_name, x, y):
    with BudgetContext(flop_budget=10**6) as budget:
        result = getattr(ops, op_name)(x, y)
        assert budget.flops_used == x.size


def test_pow():
    x = numpy.array([2.0, 3.0])
    y = numpy.array([3.0, 2.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.pow(x, y)
    assert numpy.allclose(result, numpy.array([8.0, 9.0]))


def test_gcd():
    x = numpy.array([12, 15, 8], dtype=int)
    y = numpy.array([8, 10, 6], dtype=int)
    with BudgetContext(flop_budget=10**6):
        result = ops.gcd(x, y)
    assert list(result) == [4, 5, 2]


def test_lcm():
    x = numpy.array([4, 6], dtype=int)
    y = numpy.array([6, 4], dtype=int)
    with BudgetContext(flop_budget=10**6):
        result = ops.lcm(x, y)
    assert list(result) == [12, 12]


def test_bitwise_ops():
    x = numpy.array([5, 6], dtype=numpy.uint8)
    y = numpy.array([3, 3], dtype=numpy.uint8)
    with BudgetContext(flop_budget=10**6):
        ops.bitwise_and(x, y)
        ops.bitwise_or(x, y)
        ops.bitwise_xor(x, y)
        ops.bitwise_left_shift(x, numpy.array([1, 1], dtype=numpy.uint8))
        ops.bitwise_right_shift(x, numpy.array([1, 1], dtype=numpy.uint8))
        ops.left_shift(x, numpy.array([1, 1], dtype=numpy.uint8))
        ops.right_shift(x, numpy.array([1, 1], dtype=numpy.uint8))


def test_ldexp():
    x = numpy.array([0.5, 1.0])
    y = numpy.array([2, 3], dtype=int)
    with BudgetContext(flop_budget=10**6):
        result = ops.ldexp(x, y)
    assert numpy.allclose(result, numpy.array([2.0, 8.0]))


def test_vecdot():
    a = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    b = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    with BudgetContext(flop_budget=10**6):
        result = ops.vecdot(a, b)
    assert numpy.allclose(result, numpy.array([1.0, 4.0]))


# ---------------------------------------------------------------------------
# Custom ops (diff, gradient, ediff1d, convolve, correlate, corrcoef, cov,
#             trapezoid, interp)
# ---------------------------------------------------------------------------


def test_diff():
    x = numpy.array([1.0, 3.0, 6.0, 10.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = ops.diff(x)
        assert budget.flops_used == 3
    assert list(result) == [2.0, 3.0, 4.0]


def test_gradient():
    f = numpy.array([1.0, 2.0, 4.0, 7.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = ops.gradient(f)
        assert budget.flops_used == f.size
    assert isinstance(result, numpy.ndarray)


def test_ediff1d():
    a = numpy.array([1.0, 2.0, 4.0, 7.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = ops.ediff1d(a)
        assert budget.flops_used == 3
    assert list(result) == [1.0, 2.0, 3.0]


def test_convolve():
    a = numpy.array([1.0, 2.0, 3.0])
    v = numpy.array([1.0, 1.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = ops.convolve(a, v)
        assert (
            budget.flops_used == 2 * a.size * v.size - a.size - v.size
        )  # FMA=1 convention (issue #69)
    assert result.shape == (4,)


def test_correlate():
    a = numpy.array([1.0, 2.0, 3.0])
    v = numpy.array([1.0, 2.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = ops.correlate(a, v)
        # default mode='valid': honest = (2*min-1)*(max-min+1) = (2*2-1)*(3-2+1) = 3*2 = 6
        # old formula was mode-blind full: 2*3*2 - 3 - 2 = 7 (overcount on valid default)
        mn, mx = min(a.size, v.size), max(a.size, v.size)
        assert budget.flops_used == (2 * mn - 1) * (mx - mn + 1)


def test_corrcoef():
    x = numpy.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    with BudgetContext(flop_budget=10**6) as budget:
        result = ops.corrcoef(x)
    assert result.shape == (2, 2)


def test_corrcoef_with_y():
    x = numpy.array([1.0, 2.0, 3.0])
    y = numpy.array([1.0, 2.0, 3.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.corrcoef(x, y)
    assert result.shape == (2, 2)


def test_cov():
    m = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with BudgetContext(flop_budget=10**6) as budget:
        result = ops.cov(m)
    assert result.shape == (2, 2)


def test_cov_with_y():
    m = numpy.array([1.0, 2.0, 3.0])
    y = numpy.array([1.0, 2.0, 3.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.cov(m, y)
    assert result.shape == (2, 2)


@pytest.mark.parametrize("fn_name", ["cov", "corrcoef"])
@pytest.mark.parametrize("rowvar", [True, False])
@pytest.mark.parametrize("zero_d", [numpy.float64(2.0), numpy.array(2.0)])
@pytest.mark.parametrize("x", [numpy.ones((3, 4)), numpy.ones((2,)), numpy.ones((1,))])
def test_zero_d_y_raises_value_error_not_index_error(fn_name, rowvar, zero_d, x):
    """A 0-d ``y`` must be refused with ``ValueError``, not flopscope's own
    ``IndexError``.

    The cost helpers read ``y.shape[0]``/``shape[1]`` behind an ``ndim == 1``
    guard, so a 0-d y crashed with an IndexError out of flopscope's shape
    arithmetic. Nothing is billed either way: the cost helper runs before
    ``budget.deduct``, which is what the flops_used assertion pins.

    numpy's own outcome for the identical call is recorded here rather than
    asserted equal, because the two genuinely differ in one place and the
    difference should be visible instead of hidden -- see
    :func:`test_numpy_accepts_zero_d_y_for_a_single_observation`.
    """
    with BudgetContext(flop_budget=10**12, quiet=True) as budget:
        with pytest.raises(ValueError):
            getattr(ops, fn_name)(x, zero_d, rowvar=rowvar)
        assert budget.flops_used == 0, "a refused call must not be billed"


@pytest.mark.parametrize("fn_name", ["cov", "corrcoef"])
def test_numpy_accepts_zero_d_y_for_a_single_observation(fn_name):
    """A KNOWN, deliberate divergence from numpy, pinned so it cannot drift.

    With a single observation numpy broadcasts a 0-d ``y`` to ``(1, 1)`` and
    returns a 2x2 result; flopscope refuses it. Un-refusing it would change
    accept/refuse on a *billable* call, which is a repricing decision and does
    not belong in this batch -- so the divergence is recorded, not closed.
    Before this change flopscope refused the same call with ``IndexError``, so
    nothing here regressed; only the exception class moved.
    """
    x = numpy.ones((1,))
    y = numpy.float64(2.0)

    accepted = getattr(numpy, fn_name)(x, y)
    assert accepted.shape == (2, 2), (
        "numpy no longer accepts a 0-d y for a single observation; the "
        "divergence this test records may be closable -- re-triage it"
    )

    with BudgetContext(flop_budget=10**12, quiet=True) as budget:
        with pytest.raises(ValueError, match="0-d"):
            getattr(ops, fn_name)(x, y)
        assert budget.flops_used == 0


def test_trapezoid():
    y = numpy.array([0.0, 1.0, 2.0, 3.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = ops.trapezoid(y)
        assert budget.flops_used == 4 * y.size
    assert numpy.isclose(result, 4.5)


def test_interp():
    x = numpy.array([0.5, 1.5, 2.5])
    xp = numpy.array([0.0, 1.0, 2.0, 3.0])
    fp = numpy.array([0.0, 1.0, 4.0, 9.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = ops.interp(x, xp, fp)
        # cost = 3*n + n*ceil(log2(len(xp))) = 3*3 + 3*ceil(log2(4)) = 9 + 3*2 = 15
        assert budget.flops_used == 15
    assert result.shape == (3,)


# ---------------------------------------------------------------------------
# Symmetric tensor paths in binary factory (_counted_binary)
# ---------------------------------------------------------------------------


def test_binary_op_symmetric_x_scalar():
    """x is SymmetricTensor, y is scalar — result should preserve symmetry."""
    data = numpy.array([[1.0, 2.0], [2.0, 3.0]])
    x = as_symmetric(data, symmetry=(0, 1))
    y = numpy.array(2.0)
    with BudgetContext(flop_budget=10**6):
        result = ops.add(x, y)
    assert isinstance(result, SymmetricTensor)


def test_binary_op_scalar_symmetric_y():
    """x is scalar, y is SymmetricTensor — result should preserve symmetry."""
    data = numpy.array([[1.0, 2.0], [2.0, 3.0]])
    y = as_symmetric(data, symmetry=(0, 1))
    x = numpy.array(1.0)
    with BudgetContext(flop_budget=10**6):
        result = ops.add(x, y)
    assert isinstance(result, SymmetricTensor)


def test_binary_op_mismatched_symmetry_returns_plain():
    """x and y have different symmetric dims — result should be plain array."""
    d1 = numpy.array([[1.0, 2.0], [2.0, 3.0]])
    d2 = numpy.array([[4.0, 5.0], [5.0, 6.0]])
    x = as_symmetric(d1, symmetry=(0, 1))
    # y is a plain ndarray (no symmetry_info)
    with BudgetContext(flop_budget=10**6):
        with pytest.warns(SymmetryLossWarning):
            result = ops.add(x, d2)
    # Plain array — no SymmetricTensor wrapping
    assert not isinstance(result, SymmetricTensor)


# ---------------------------------------------------------------------------
# Unary factory with SymmetricTensor — preserves symmetry
# ---------------------------------------------------------------------------


def test_unary_op_symmetric_result():
    data = numpy.array([[1.0, 2.0], [2.0, 3.0]])
    x = as_symmetric(data, symmetry=(0, 1))
    with BudgetContext(flop_budget=10**6):
        result = ops.exp(x)
    assert isinstance(result, SymmetricTensor)


# ---------------------------------------------------------------------------
# isclose (special binary-ish)
# ---------------------------------------------------------------------------


def test_isclose():
    a = numpy.array([1.0, 2.0, 3.0])
    b = numpy.array([1.0, 2.0 + 1e-10, 4.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = ops.isclose(a, b)
        # 6/elem tolerance core (sub+2*abs+mul+add+cmp); numel=3
        assert budget.flops_used == 6 * 3
    assert list(result) == [True, True, False]  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Additional reductions not tested in test_pointwise.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_name",
    [
        "all",
        "any",
        "amax",
        "amin",
        "count_nonzero",
        "nanmax",
        "nanmin",
        "nansum",
        "nanprod",
        "nanargmax",
        "nanargmin",
    ],
)
def test_reduction_runs(op_name):
    x = numpy.array([1.0, 2.0, 3.0, 4.0])
    with BudgetContext(flop_budget=10**6):
        result = getattr(ops, op_name)(x)
    assert numpy.ndim(result) == 0 or result is not None


def test_median():
    x = numpy.array([3.0, 1.0, 2.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.median(x)
    assert numpy.isclose(result, 2.0)


def test_average():
    x = numpy.array([1.0, 2.0, 3.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.average(x)
    assert numpy.isclose(result, 2.0)


def test_ptp():
    x = numpy.array([1.0, 5.0, 3.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.ptp(x)
    assert numpy.isclose(result, 4.0)


def test_percentile():
    x = numpy.array([1.0, 2.0, 3.0, 4.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.percentile(x, q=50)
    assert numpy.isclose(result, 2.5)


def test_quantile():
    x = numpy.array([1.0, 2.0, 3.0, 4.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.quantile(x, q=0.5)
    assert numpy.isclose(result, 2.5)


def test_nanpercentile():
    x = numpy.array([1.0, float("nan"), 3.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.nanpercentile(x, q=50)
    assert numpy.isclose(result, 2.0)


def test_nanquantile():
    x = numpy.array([1.0, float("nan"), 3.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.nanquantile(x, q=0.5)
    assert numpy.isclose(result, 2.0)


def test_nancumprod():
    x = numpy.array([1.0, float("nan"), 3.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.nancumprod(x)
    assert result.shape == (3,)


def test_nancumsum():
    x = numpy.array([1.0, float("nan"), 3.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.nancumsum(x)
    assert result.shape == (3,)


def test_nanmean():
    x = numpy.array([1.0, float("nan"), 3.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.nanmean(x)
    assert numpy.isclose(result, 2.0)


def test_nanmedian():
    x = numpy.array([1.0, float("nan"), 3.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.nanmedian(x)
    assert numpy.isclose(result, 2.0)


def test_nanstd():
    x = numpy.array([1.0, 2.0, float("nan"), 4.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.nanstd(x)
    assert not numpy.isnan(result)


def test_nanvar():
    x = numpy.array([1.0, 2.0, float("nan"), 4.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.nanvar(x)
    assert not numpy.isnan(result)


@pytest.mark.skipif(
    not hasattr(numpy, "cumulative_prod"), reason="requires numpy >= 2.1"
)
def test_cumulative_prod():
    x = numpy.array([1.0, 2.0, 3.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.cumulative_prod(x)
    assert result.shape == (3,)


@pytest.mark.skipif(
    not hasattr(numpy, "cumulative_sum"), reason="requires numpy >= 2.1"
)
def test_cumulative_sum():
    x = numpy.array([1.0, 2.0, 3.0])
    with BudgetContext(flop_budget=10**6):
        result = ops.cumulative_sum(x)
    assert result.shape == (3,)
