import warnings

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp


def _bill(thunk):
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as bc:
        thunk()
    return bc.flops_used


def test_solve_bills_broadcast_batch_from_b():
    rng = np.random.default_rng(0)
    C = fnp.asarray(rng.standard_normal((64, 64)))
    B1 = fnp.asarray(rng.standard_normal((64, 32)))
    B8 = fnp.asarray(rng.standard_normal((8, 64, 32)))
    b1 = _bill(lambda: fnp.linalg.solve(C, B1))
    b8 = _bill(lambda: fnp.linalg.solve(C, B8))
    assert b8 == 8 * b1  # batch of 8 independent solves


def test_tensorsolve_degenerate_scales_with_n():
    rng = np.random.default_rng(1)
    A50 = fnp.asarray(rng.standard_normal((50, 50)))
    b50 = fnp.asarray(rng.standard_normal(50))
    A100 = fnp.asarray(rng.standard_normal((100, 100)))
    b100 = fnp.asarray(rng.standard_normal(100))
    c50 = _bill(lambda: fnp.linalg.tensorsolve(A50, b50))
    c100 = _bill(lambda: fnp.linalg.tensorsolve(A100, b100))
    assert c50 > 100  # not the flat 4-FLOP floor
    assert c100 >= 7 * c50  # ~ (100/50)^3


def test_tensorsolve_axes_bills_true_solved_dimension():
    # Regression pin for the axes-reordering under-bill: numpy's
    # tensorsolve(a, b, axes=...) transposes `axes` to the tail of `a`
    # *before* reshaping to the real (n, n) system it solves, so a formula
    # that reads the split point off the UN-transposed `a.shape` (e.g. the
    # earlier `n = prod(a.shape[b.ndim:])`) can diverge arbitrarily from the
    # true solved dimension once `axes` is passed.
    #
    # a=(6,1,3,2), b=(1,3,2), axes=(0,): axis 0 moves to the tail, giving a
    # transposed shape (1,3,2,6); the real system solved is n=6 (matches
    # `honest_n` cross-checked against a manual transpose+reshape+solve in
    # .superpowers/sdd/reprobe_tensorsolve.py-style probes). The old
    # ind=b.ndim formula instead read n=prod(a.shape[3:])=2 from the
    # untransposed shape -> solve_cost(2,1)=13 (analytical), 26 under
    # production rates (float64 dtype_rate=2.0) -- an under-bill even
    # relative to the pre-`ind`-threading baseline for this shape.
    from tests.test_dtype_cost import _billed_with_production_rates

    rng = np.random.default_rng(7)
    a = fnp.asarray(rng.standard_normal((6, 1, 3, 2)))
    b = fnp.asarray(rng.standard_normal((1, 3, 2)))
    billed, dt = _billed_with_production_rates(
        lambda: fnp.linalg.tensorsolve(a, b, axes=(0,))
    )
    assert dt == "float64"
    assert billed == 432  # solve_cost(6, 1)=216 x dtype_rate(float64)=2.0
    assert billed != 26  # the old under-billed (ind-based) production value


def test_percentile_family_scales_with_q_count():
    a = fnp.asarray(np.random.default_rng(2).standard_normal(1000))
    for op in (fnp.percentile, fnp.quantile, fnp.nanpercentile, fnp.nanquantile):
        hi = 100.0 if op in (fnp.percentile, fnp.nanpercentile) else 1.0
        scalar = _bill(lambda op=op, hi=hi: op(a, hi / 2))
        many = _bill(lambda op=op, hi=hi: op(a, np.linspace(0, hi, 1000)))
        assert many > scalar, f"{op.__name__} flat-bills q-array"
        assert many >= scalar + 999  # at least +1 FLOP per extra quantile


def test_fftn_family_bills_leading_batch_when_s_given():
    # numpy: when `s` is given and `axes` is omitted, the transform runs
    # over the TRAILING len(s) axes and batches over every leading axis.
    # `_batch_count_nd` short-circuits `axes is None` to batch=1, which is
    # only correct when `s` is also None -- so this path was silently
    # dropping the leading batch dimension from the bill.
    rng = np.random.default_rng(3)
    a1 = fnp.asarray(rng.standard_normal((16, 16)))
    a8 = fnp.asarray(rng.standard_normal((8, 16, 16)))
    with warnings.catch_warnings():
        # numpy 2.x DeprecationWarning: "`axes` should not be `None` if `s`
        # is not `None`" -- expected on this path, not under test here.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        for op in (fnp.fft.fftn, fnp.fft.ifftn):
            b1 = _bill(lambda op=op: op(a1, s=(16, 16)))
            b8 = _bill(lambda op=op: op(a8, s=(16, 16)))
            assert b8 == 8 * b1, (
                f"{op.__name__} drops leading batch on axes=None,s given"
            )


def test_rfftn_family_bills_leading_batch_when_s_given():
    # Same bug, real-FFT siblings: rfftn transforms real input, irfftn
    # consumes the (conjugate-symmetric) complex spectrum rfftn produces.
    rng = np.random.default_rng(4)
    raw1 = rng.standard_normal((16, 16))
    raw8 = rng.standard_normal((8, 16, 16))
    a1 = fnp.asarray(raw1)
    a8 = fnp.asarray(raw8)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        b1 = _bill(lambda: fnp.fft.rfftn(a1, s=(16, 16)))
        b8 = _bill(lambda: fnp.fft.rfftn(a8, s=(16, 16)))
        assert b8 == 8 * b1, "rfftn drops leading batch on axes=None,s given"

        # Reference complex spectra built with plain numpy (outside any
        # BudgetContext, so building them doesn't pollute the bill below).
        c1 = fnp.asarray(np.fft.rfftn(raw1, s=(16, 16)))
        c8 = fnp.asarray(np.fft.rfftn(raw8, s=(16, 16)))
        i1 = _bill(lambda: fnp.fft.irfftn(c1, s=(16, 16)))
        i8 = _bill(lambda: fnp.fft.irfftn(c8, s=(16, 16)))
        assert i8 == 8 * i1, "irfftn drops leading batch on axes=None,s given"


def test_fft2_family_bills_leading_batch_when_axes_explicitly_none():
    # fft2/ifft2/rfft2/irfft2 route through the same `_batch_count_nd`
    # helper as the fftn family, and `axes` defaults to (-2, -1) rather
    # than None -- but numpy still accepts an explicit `axes=None`
    # (fft2(a, s, axes) delegates straight to fftn(a, s, axes)), so the
    # same leading-batch-axis under-bill is reachable here too.
    rng = np.random.default_rng(5)
    raw1 = rng.standard_normal((16, 16))
    raw8 = rng.standard_normal((8, 16, 16))
    a1 = fnp.asarray(raw1)
    a8 = fnp.asarray(raw8)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        for op in (fnp.fft.fft2, fnp.fft.ifft2):
            b1 = _bill(lambda op=op: op(a1, s=(16, 16), axes=None))
            b8 = _bill(lambda op=op: op(a8, s=(16, 16), axes=None))
            assert b8 == 8 * b1, (
                f"{op.__name__} drops leading batch on axes=None,s given"
            )

        r1 = _bill(lambda: fnp.fft.rfft2(a1, s=(16, 16), axes=None))
        r8 = _bill(lambda: fnp.fft.rfft2(a8, s=(16, 16), axes=None))
        assert r8 == 8 * r1, "rfft2 drops leading batch on axes=None,s given"

        c1 = fnp.asarray(np.fft.rfft2(raw1, s=(16, 16)))
        c8 = fnp.asarray(np.fft.rfft2(raw8, s=(16, 16)))
        j1 = _bill(lambda: fnp.fft.irfft2(c1, s=(16, 16), axes=None))
        j8 = _bill(lambda: fnp.fft.irfft2(c8, s=(16, 16), axes=None))
        assert j8 == 8 * j1, "irfft2 drops leading batch on axes=None,s given"


def test_fftn_s_negative_one_sentinel_bills_real_cost():
    # numpy >= 2.0 treats `-1` in `s` as "use the whole input" along that
    # transform axis. The old s_for_cost builder passed `-1` straight
    # through into fftn_cost's `prod(shape)`, collapsing N to <= 1 and
    # billing 0 for a transform that numpy actually runs at full size -- a
    # live budget bypass (the real result is still computed and returned).
    from tests.test_dtype_cost import _billed_with_production_rates

    rng = np.random.default_rng(6)
    a = fnp.asarray(rng.standard_normal((6, 5, 8, 8)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        billed_neg1, _ = _billed_with_production_rates(
            lambda: fnp.fft.fftn(a, s=(-1, 8))
        )
    assert billed_neg1 > 0
    # -1 on axis 2 (size 8) resolves to the same real cost as writing the
    # size out explicitly: fftn(a, s=(8, 8)) bills 115200 (production
    # rates: float64 input -> complex128 compute dtype, rate 2.0).
    assert billed_neg1 == 115200


def test_fftn_s_none_sentinel_resolves_to_transform_axis_not_leading_axis():
    # When `axes` is omitted and `s` is given, numpy transforms the
    # TRAILING len(s) axes (here axes (2, 3), both size 8) and a `None`
    # entry in `s` resolves to the input size along ITS transform axis. The
    # s_for_cost None-fill used `range(a.ndim)` (leading axes 0, 1 = sizes
    # 6, 5) instead of the same trailing axes the batch count uses,
    # under-billing 86400 vs the real 115200 for this shape.
    rng = np.random.default_rng(6)
    a = fnp.asarray(rng.standard_normal((6, 5, 8, 8)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        billed_none = _bill(lambda: fnp.fft.fftn(a, s=(None, 8)))
        billed_concrete = _bill(lambda: fnp.fft.fftn(a, s=(8, 8)))
    assert billed_none == billed_concrete
