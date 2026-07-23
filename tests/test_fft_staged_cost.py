import numpy as np

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights
from flopscope.numpy.fft._transforms import _NUMPY_GE_2_1


def _billed(fn):
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn()
        return b.flops_used


def test_reviewer_repro():
    load_weights()
    x = np.random.default_rng(0).standard_normal((1024, 1))
    assert _billed(lambda: fnp.fft.fftn(x, s=(1, 1024), axes=(0, 1))) == 104857600


def test_fused_equals_explicit_1d_sequence():
    load_weights()
    x = np.random.default_rng(1).standard_normal((32, 8))
    fused = _billed(lambda: fnp.fft.fftn(x, s=(4, 16), axes=(0, 1)))

    def staged():  # numpy c2c order: reverse axes
        y = fnp.fft.fft(x, n=16, axis=1)
        return fnp.fft.fft(y, n=4, axis=0)

    assert fused == _billed(staged)


def test_no_op_s_matches_legacy_formula():
    load_weights()
    # s == input shape (and s=None) must be byte-identical to the old bill.
    x = np.random.default_rng(2).standard_normal((16, 8))
    a = _billed(lambda: fnp.fft.fftn(x))
    b = _billed(lambda: fnp.fft.fftn(x, s=(16, 8), axes=(0, 1)))
    assert a == b > 0


def test_rfftn_and_irfftn_shape_change():
    load_weights()
    x = np.random.default_rng(3).standard_normal((64, 9))
    # rfftn fused == rfft(last) then fft(other) in numpy's order
    fused_r = _billed(lambda: fnp.fft.rfftn(x, s=(4, 16), axes=(0, 1)))

    def staged_r():
        y = fnp.fft.rfft(x, n=16, axis=1)
        return fnp.fft.fft(y, n=4, axis=0)

    assert fused_r == _billed(staged_r)


def test_rfftn_3axis_matches_numpy_cascade():
    load_weights()
    x = np.random.default_rng(0).standard_normal((4, 2, 4))

    def fused():
        return fnp.fft.rfftn(x, s=(4, 16, 4), axes=(0, 1, 2))

    # numpy: rfft(last) first, then the remaining c2c axes -- REVERSED from
    # numpy 2.1 onward, forward on 2.0.x. Same `_NUMPY_GE_2_1` gate that
    # staged_fftn_cost's r2c branch uses, so this test tracks numpy's actual
    # per-version order instead of hardcoding one of the two.
    def cascade():
        y = fnp.fft.rfft(x, n=4, axis=2)
        if _NUMPY_GE_2_1:
            y = fnp.fft.fft(y, n=16, axis=1)
            return fnp.fft.fft(y, n=4, axis=0)
        y = fnp.fft.fft(y, n=4, axis=0)
        return fnp.fft.fft(y, n=16, axis=1)

    assert _billed(fused) == _billed(cascade)

    # Anchor to numpy's actual behavior, not just internal self-consistency:
    # the same-order manual cascade run through plain numpy must reproduce
    # np.fft.rfftn's own output.
    y_ref = np.fft.rfft(x, n=4, axis=2)
    if _NUMPY_GE_2_1:
        y_ref = np.fft.fft(y_ref, n=16, axis=1)
        y_ref = np.fft.fft(y_ref, n=4, axis=0)
    else:
        y_ref = np.fft.fft(y_ref, n=4, axis=0)
        y_ref = np.fft.fft(y_ref, n=16, axis=1)
    assert np.allclose(y_ref, np.fft.rfftn(x, s=(4, 16, 4), axes=(0, 1, 2)))


def test_irfftn_3axis_matches_numpy_cascade():
    load_weights()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 3, 5)) + 1j * rng.standard_normal((4, 3, 5))

    def fused():
        return fnp.fft.irfftn(x, s=(4, 6, 8), axes=(0, 1, 2))

    def cascade():  # numpy: remaining ifft FORWARD, then irfft(last)
        y = fnp.fft.ifft(x, n=4, axis=0)
        y = fnp.fft.ifft(y, n=6, axis=1)
        return fnp.fft.irfft(y, n=8, axis=2)

    assert _billed(fused) == _billed(cascade)
