"""Extended tests for FFT transform wrappers — covers all untested variants.

Covers: ifft (cost/result), rfft (result), irfft (result), fft2 (cost variants),
ifft2, rfft2, irfft2, fftn, ifftn, rfftn, irfftn, hfft, ihfft.
Also exercises the cost helper functions directly.
"""

import math

import numpy

from flopscope._budget import BudgetContext
from flopscope.numpy.fft import (
    fft,
    fft2,
    fftn,
    hfft,
    ifft,
    ifft2,
    ifftn,
    ihfft,
    irfft,
    irfft2,
    irfftn,
    rfft,
    rfft2,
    rfftn,
)
from flopscope.numpy.fft._transforms import (
    fft_cost,
    fftn_cost,
    hfft_cost,
    rfft_cost,
    rfftn_cost,
)

# ---------------------------------------------------------------------------
# Cost helper functions
# ---------------------------------------------------------------------------


def test_fft_cost_small():
    assert fft_cost(0) == 0
    assert fft_cost(1) == 0
    assert fft_cost(8) == 5 * 8 * 3


def test_rfft_cost_small():
    assert rfft_cost(0) == 0
    assert rfft_cost(1) == 0
    assert rfft_cost(16) == 5 * 8 * math.ceil(math.log2(16))


def test_fftn_cost():
    assert fftn_cost((1,)) == 0
    assert fftn_cost((4, 4)) == 5 * 16 * math.ceil(math.log2(16))


def test_rfftn_cost():
    assert rfftn_cost((1,)) == 0
    assert rfftn_cost((4, 4)) == 5 * 8 * math.ceil(math.log2(16))


def test_hfft_cost():
    assert hfft_cost(0) == 0
    assert hfft_cost(1) == 0
    # hfft_cost(8) = 5 * (8//2) * ceil(log2(8)) = 5 * 4 * 3 = 60
    assert hfft_cost(8) == 5 * (8 // 2) * 3


# ---------------------------------------------------------------------------
# ifft
# ---------------------------------------------------------------------------


class TestIfftExtended:
    def test_result_roundtrip(self):
        x = numpy.random.randn(16)
        with BudgetContext(flop_budget=10**6):
            spectrum = fft(x)
            recovered = ifft(spectrum)
        assert numpy.allclose(x, recovered.real, atol=1e-10)

    def test_op_log_name(self):
        x = numpy.random.randn(8) + 1j * numpy.random.randn(8)
        with BudgetContext(flop_budget=10**6) as budget:
            ifft(x)
        assert budget.op_log[-1].op_name == "fft.ifft"

    def test_with_n_param(self):
        n = 32
        x = numpy.random.randn(16) + 1j * numpy.random.randn(16)
        with BudgetContext(flop_budget=10**6) as budget:
            ifft(x, n=n)
        assert budget.flops_used == fft_cost(n)


# ---------------------------------------------------------------------------
# rfft
# ---------------------------------------------------------------------------


class TestRfftExtended:
    def test_output_shape(self):
        n = 16
        x = numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6):
            result = rfft(x)
        assert result.shape == (n // 2 + 1,)

    def test_op_log_name(self):
        x = numpy.random.randn(8)
        with BudgetContext(flop_budget=10**6) as budget:
            rfft(x)
        assert budget.op_log[-1].op_name == "fft.rfft"


# ---------------------------------------------------------------------------
# irfft
# ---------------------------------------------------------------------------


class TestIrfft:
    def test_result_matches_numpy(self):
        n = 16
        x = numpy.random.randn(n)
        freq = numpy.fft.rfft(x)
        with BudgetContext(flop_budget=10**6):
            result = irfft(freq, n=n)
        assert numpy.allclose(result, x, atol=1e-10)

    def test_output_shape_default_n(self):
        n = 16
        freq = numpy.fft.rfft(numpy.random.randn(n))  # shape = (9,)
        with BudgetContext(flop_budget=10**6):
            result = irfft(freq)
        # default n = 2*(9-1) = 16
        assert result.shape == (n,)

    def test_op_log_name(self):
        freq = numpy.fft.rfft(numpy.random.randn(8))
        with BudgetContext(flop_budget=10**6) as budget:
            irfft(freq)
        assert budget.op_log[-1].op_name == "fft.irfft"


# ---------------------------------------------------------------------------
# fft2 extra coverage
# ---------------------------------------------------------------------------


class TestFft2Extended:
    def test_op_log_name(self):
        x = numpy.random.randn(4, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            fft2(x)
        assert budget.op_log[-1].op_name == "fft.fft2"

    def test_with_s_param(self):
        # Padding both axes of a (4, 6) input up to s=(8, 8) is priced from
        # numpy's actual staged cascade (batch taken from the shape at each
        # stage), not the flat final-shape fftn_cost(s)=1920: axis 1 pads
        # first at batch=4 (4*fft_cost(8)=4*120=480), then axis 0 pads at
        # the now-8-wide batch (8*fft_cost(8)=8*120=960) -> 480+960=1440.
        x = numpy.random.randn(4, 6)
        s = (8, 8)
        with BudgetContext(flop_budget=10**6) as budget:
            fft2(x, s=s)
        assert budget.flops_used == 1440


# ---------------------------------------------------------------------------
# ifft2
# ---------------------------------------------------------------------------


class TestIfft2:
    def test_result_roundtrip(self):
        x = numpy.random.randn(8, 8)
        with BudgetContext(flop_budget=10**6):
            spectrum = fft2(x)
            recovered = ifft2(spectrum)
        assert numpy.allclose(x, recovered.real, atol=1e-10)

    def test_cost(self):
        x = numpy.random.randn(8, 8)
        with BudgetContext(flop_budget=10**6) as budget:
            ifft2(x)
        assert budget.flops_used == fftn_cost((8, 8))

    def test_op_log_name(self):
        x = numpy.random.randn(4, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            ifft2(x)
        assert budget.op_log[-1].op_name == "fft.ifft2"

    def test_with_s_param(self):
        # Staged cost, not the flat final-shape fftn_cost(s)=1920 -- see
        # TestFft2Extended.test_with_s_param for the per-stage derivation
        # (same 1440 total: axis 1 pads at batch=4, then axis 0 pads at the
        # now-8-wide batch=8).
        x = numpy.random.randn(4, 4)
        s = (8, 8)
        with BudgetContext(flop_budget=10**6) as budget:
            ifft2(x, s=s)
        assert budget.flops_used == 1440


# ---------------------------------------------------------------------------
# rfft2
# ---------------------------------------------------------------------------


class TestRfft2:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(8, 8)
        with BudgetContext(flop_budget=10**6):
            result = rfft2(x)
        assert numpy.allclose(result, numpy.fft.rfft2(x))

    def test_cost(self):
        # rfft2 stages a real FFT on the last axis (batch=8, rfft_cost(8)=60
        # -> 480) then a complex FFT on axis 0 over the now Hermitian-reduced
        # last axis (batch=5, fft_cost(8)=120 -> 600); 480+600=1080. The flat
        # rfftn_cost((8,8))=960 assumes a uniform half-cost across both axes,
        # which is not what the cascade actually does once the last axis has
        # already been reduced to n//2+1=5.
        x = numpy.random.randn(8, 8)
        with BudgetContext(flop_budget=10**6) as budget:
            rfft2(x)
        assert budget.flops_used == 1080

    def test_op_log_name(self):
        x = numpy.random.randn(4, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            rfft2(x)
        assert budget.op_log[-1].op_name == "fft.rfft2"

    def test_with_s_param(self):
        # Staged: real FFT on axis 1 first (batch=4, rfft_cost(8)=60 -> 240),
        # then complex FFT on axis 0 over the Hermitian-reduced axis 1
        # (batch=5, fft_cost(8)=120 -> 600); 240+600=840.
        x = numpy.random.randn(4, 6)
        s = (8, 8)
        with BudgetContext(flop_budget=10**6) as budget:
            rfft2(x, s=s)
        assert budget.flops_used == 840


# ---------------------------------------------------------------------------
# irfft2
# ---------------------------------------------------------------------------


class TestIrfft2:
    def test_result_roundtrip(self):
        n = 8
        x = numpy.random.randn(n, n)
        freq = numpy.fft.rfft2(x)
        with BudgetContext(flop_budget=10**6):
            result = irfft2(freq, s=(n, n))
        assert numpy.allclose(result, x, atol=1e-10)

    def test_default_s(self):
        n = 8
        x = numpy.random.randn(n, n)
        freq = numpy.fft.rfft2(x)  # shape (8, 5)
        with BudgetContext(flop_budget=10**6):
            result = irfft2(freq)
        assert result.shape[0] == n

    def test_op_log_name(self):
        freq = numpy.fft.rfft2(numpy.random.randn(4, 4))
        with BudgetContext(flop_budget=10**6) as budget:
            irfft2(freq, s=(4, 4))
        assert budget.op_log[-1].op_name == "fft.irfft2"


# ---------------------------------------------------------------------------
# fftn
# ---------------------------------------------------------------------------


class TestFftnExtended:
    def test_cost(self):
        x = numpy.random.randn(4, 4, 4)
        N = 64
        with BudgetContext(flop_budget=10**8) as budget:
            fftn(x)
        assert budget.flops_used == fftn_cost((4, 4, 4))

    def test_op_log_name(self):
        x = numpy.random.randn(4, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            fftn(x)
        assert budget.op_log[-1].op_name == "fft.fftn"

    def test_with_s_and_axes(self):
        # Truncating both axes of an (8, 8) input down to s=(4, 4) is priced
        # from the staged cascade: axis 1 truncates first at the original
        # batch=8 (8*fft_cost(4)=8*40=320), then axis 0 truncates at the
        # now-4-wide batch (4*fft_cost(4)=160); 320+160=480. The flat
        # fftn_cost((4,4))=320 assumes both axes always saw the same,
        # already-truncated batch.
        x = numpy.random.randn(8, 8)
        with BudgetContext(flop_budget=10**6) as budget:
            fftn(x, s=(4, 4), axes=(0, 1))
        assert budget.flops_used == 480


# ---------------------------------------------------------------------------
# ifftn
# ---------------------------------------------------------------------------


class TestIfftn:
    def test_result_roundtrip(self):
        x = numpy.random.randn(4, 4)
        with BudgetContext(flop_budget=10**6):
            spectrum = fftn(x)
            recovered = ifftn(spectrum)
        assert numpy.allclose(x, recovered.real, atol=1e-10)

    def test_cost(self):
        x = numpy.random.randn(4, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            ifftn(x)
        assert budget.flops_used == fftn_cost((4, 4))

    def test_op_log_name(self):
        x = numpy.random.randn(4, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            ifftn(x)
        assert budget.op_log[-1].op_name == "fft.ifftn"

    def test_with_axes(self):
        x = numpy.random.randn(8, 8)
        with BudgetContext(flop_budget=10**6) as budget:
            ifftn(x, axes=(0,))
        # 8 independent 1-D transforms of length 8 (batch=8)
        assert budget.flops_used == 8 * fftn_cost((8,))


# ---------------------------------------------------------------------------
# rfftn
# ---------------------------------------------------------------------------


class TestRfftn:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(4, 4)
        with BudgetContext(flop_budget=10**6):
            result = rfftn(x)
        assert numpy.allclose(result, numpy.fft.rfftn(x))

    def test_cost(self):
        # rfftn stages a real FFT on the last axis (batch=4, rfft_cost(4)=20
        # -> 80) then a complex FFT on axis 0 over the now Hermitian-reduced
        # last axis (batch=3, fft_cost(4)=40 -> 120); 80+120=200. The flat
        # rfftn_cost((4,4))=160 assumes a uniform half-cost across both axes.
        x = numpy.random.randn(4, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            rfftn(x)
        assert budget.flops_used == 200

    def test_op_log_name(self):
        x = numpy.random.randn(4, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            rfftn(x)
        assert budget.op_log[-1].op_name == "fft.rfftn"

    def test_with_axes(self):
        x = numpy.random.randn(8, 8)
        with BudgetContext(flop_budget=10**6) as budget:
            rfftn(x, axes=(0,))
        # 8 independent 1-D transforms of length 8 (batch=8)
        assert budget.flops_used == 8 * rfftn_cost((8,))


# ---------------------------------------------------------------------------
# irfftn
# ---------------------------------------------------------------------------


class TestIrfftn:
    def test_result_roundtrip(self):
        n = 8
        x = numpy.random.randn(n, n)
        freq = numpy.fft.rfftn(x)
        with BudgetContext(flop_budget=10**6):
            result = irfftn(freq, s=(n, n))
        assert numpy.allclose(result, x, atol=1e-10)

    def test_default_s(self):
        x = numpy.random.randn(8, 8)
        freq = numpy.fft.rfftn(x)
        with BudgetContext(flop_budget=10**6):
            result = irfftn(freq)
        assert result.shape == (8, 8)

    def test_with_axes(self):
        x = numpy.random.randn(8, 8)
        freq = numpy.fft.rfftn(x, axes=(0,))
        with BudgetContext(flop_budget=10**6) as budget:
            result = irfftn(freq, axes=(0,))
        assert budget.flops_used > 0

    def test_op_log_name(self):
        freq = numpy.fft.rfftn(numpy.random.randn(4, 4))
        with BudgetContext(flop_budget=10**6) as budget:
            irfftn(freq, s=(4, 4))
        assert budget.op_log[-1].op_name == "fft.irfftn"


# ---------------------------------------------------------------------------
# hfft
# ---------------------------------------------------------------------------


class TestHfftExtended:
    def test_result_matches_numpy(self):
        n = 8
        x = numpy.random.randn(n // 2 + 1) + 1j * numpy.random.randn(n // 2 + 1)
        with BudgetContext(flop_budget=10**6):
            result = hfft(x, n=n)
        assert numpy.allclose(result, numpy.fft.hfft(x, n=n))

    def test_op_log_name(self):
        x = numpy.random.randn(8) + 1j * numpy.random.randn(8)
        with BudgetContext(flop_budget=10**6) as budget:
            hfft(x)
        assert budget.op_log[-1].op_name == "fft.hfft"


# ---------------------------------------------------------------------------
# ihfft
# ---------------------------------------------------------------------------


class TestIhfft:
    def test_result_matches_numpy(self):
        n = 16
        x = numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6):
            result = ihfft(x)
        assert numpy.allclose(result, numpy.fft.ihfft(x))

    def test_cost(self):
        n = 16
        x = numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6) as budget:
            ihfft(x)
        assert budget.flops_used == hfft_cost(n)

    def test_op_log_name(self):
        x = numpy.random.randn(8)
        with BudgetContext(flop_budget=10**6) as budget:
            ihfft(x)
        assert budget.op_log[-1].op_name == "fft.ihfft"

    def test_with_n_param(self):
        x = numpy.random.randn(16)
        n = 8
        with BudgetContext(flop_budget=10**6) as budget:
            ihfft(x, n=n)
        assert budget.flops_used == hfft_cost(n)
