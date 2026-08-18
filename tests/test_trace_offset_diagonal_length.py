"""``numpy.trace`` must bill the diagonal length that ``offset`` actually selects.

The bare ``numpy.trace`` cost path historically summed ``min(m, n)`` diagonal
elements regardless of ``offset=``, while ``numpy.linalg.trace`` already shrank
the diagonal.  That over-billed every off-diagonal trace (up to ``min(m, n)``x)
and broke spelling invariance between the two ``trace`` spellings.  These tests
pin that the two spellings agree wherever they compute the same operation, and
that the billed count tracks numpy's real diagonal length.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


def _billed(fn) -> int:
    with flops.budget(10**15, quiet=True) as b:
        fn()
        return b.flops_used


_DTYPES = [np.float64, np.float32, np.int32, np.complex64, np.complex128]
_SHAPES_2D = [(256, 256), (100, 300), (300, 100), (64, 64), (1, 50), (50, 1)]
_OFFSETS = [0, 1, 50, 99, 100, 250, 255, 256, 300, -1, -50, -99, -255, -300]


@pytest.mark.parametrize("shape", _SHAPES_2D)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("offset", _OFFSETS)
def test_bare_trace_matches_linalg_trace_2d(shape, dtype, offset):
    """On 2-D input the two spellings are the same op and must bill identically."""
    m = fnp.array(np.ones(shape, dtype=dtype))
    assert _billed(lambda: fnp.trace(m, offset=offset)) == _billed(
        lambda: fnp.linalg.trace(m, offset=offset)
    )


@pytest.mark.parametrize("shape", [(5, 256, 256), (3, 4, 100, 300), (2, 64, 64)])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("offset", [0, 50, 100, -50, 255])
def test_bare_trace_matches_linalg_trace_stacked_matched_axes(shape, dtype, offset):
    """>2-D: with axes matched to linalg's (-2, -1) the two spellings agree.

    (Bare ``trace`` defaults to axis1=0/axis2=1, a genuinely different op on
    >2-D input, so invariance only holds when the axes are matched.)
    """
    m = fnp.array(np.ones(shape, dtype=dtype))
    assert _billed(lambda: fnp.trace(m, offset=offset, axis1=-2, axis2=-1)) == _billed(
        lambda: fnp.linalg.trace(m, offset=offset)
    )


@pytest.mark.parametrize("shape", [(256, 256), (100, 300), (300, 100)])
@pytest.mark.parametrize("offset", [1, 50, 100, 250, 255, -1, -50, -99])
def test_bare_trace_count_tracks_diagonal_length(shape, offset):
    """Billed FLOPs scale with numpy's real diagonal length, not min(m, n).

    Weight-independent: the cost is a linear function of diagonal length with
    zero intercept (floored at 1), so ``billed(k) * diag_len(0)`` must equal
    ``billed(0) * diag_len(k)``.  Cross-multiplying avoids any weight/rate
    constant entirely.
    """
    m = fnp.array(np.ones(shape, dtype=np.float64))
    diag0 = int(np.diagonal(np.ones(shape), offset=0).shape[-1])
    diagk = max(int(np.diagonal(np.ones(shape), offset=offset).shape[-1]), 1)
    billed0 = _billed(lambda: fnp.trace(m))
    billedk = _billed(lambda: fnp.trace(m, offset=offset))
    assert billedk * diag0 == billed0 * diagk


def test_offset_shrinks_the_bill_below_the_full_diagonal():
    """A guard that fails if ``offset=`` is ever ignored again.

    Weight-independent: an offset that selects a length-1 diagonal must bill
    exactly ``1 / min(m, n)`` of the offset=0 main-diagonal bill.
    """
    shape = (256, 256)
    m = fnp.array(np.ones(shape, dtype=np.float64))
    billed0 = _billed(lambda: fnp.trace(m))
    billed_edge = _billed(lambda: fnp.trace(m, offset=255))  # length-1 diagonal
    assert billed_edge < billed0
    assert billed_edge * min(shape) == billed0  # 1 element vs 256


def test_offset_zero_is_the_full_main_diagonal():
    """offset=0 still bills the full min(m, n) diagonal (common case unchanged).

    Weight-independent: equals the bill of an explicit length-min(m, n)
    computation via the linalg spelling, already pinned invariant above; here
    we assert the offset=0 bare bill is strictly the largest across offsets.
    """
    m = fnp.array(np.ones((256, 256), dtype=np.float64))
    bills = [_billed(lambda o=o: fnp.trace(m, offset=o)) for o in (0, 1, 50, 200, 255)]
    assert bills[0] == max(bills)
    assert bills == sorted(bills, reverse=True)  # monotone non-increasing in |offset|


def test_trace_value_is_still_correct_with_offset():
    """The billing change must not alter the returned value."""
    base = np.arange(256 * 256).reshape(256, 256).astype(float)
    m = fnp.array(base)
    with flops.budget(10**15, quiet=True):
        got = np.asarray(fnp.trace(m, offset=100))
    assert np.allclose(got, np.trace(base, offset=100))
