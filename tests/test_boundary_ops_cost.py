"""Cost-model tests for boundary/padding ops under the writes-consistent model.

``pad`` allocates a fresh output and writes every cell (interior copy +
border fill), so every mode bills a numel(output) base: movement modes
(constant/edge/empty/wrap and reflect/symmetric with reflect_type='even') add
nothing on top; value-computing modes (maximum/minimum/mean/median,
linear_ramp, and reflect/symmetric with reflect_type='odd') add their own
extra on top of that base; ``mode=<callable>`` is rejected outright.
"""

import numpy as np
import pytest

import flopscope.numpy as fnp


def billed(fn):
    from flopscope import BudgetContext

    with BudgetContext(flop_budget=10**15, quiet=True) as b:
        fn()
    return int(b.flops_used)


def test_pad_constant_free():
    a = fnp.asarray(np.zeros(100))
    # writes-consistent base: numel(out) = 100 + 1 + 1 = 102, no mode extra
    assert billed(lambda: fnp.pad(a, (1, 1), mode="constant")) == 102


def test_pad_even_reflect_free():
    a = fnp.asarray(np.arange(10.0))
    # writes-consistent base: numel(out) = 10 + 1 + 1 = 12, no mode extra
    assert billed(lambda: fnp.pad(a, (1, 1), mode="reflect")) == 12


def test_pad_mean_1d_charged():
    a = fnp.asarray(np.arange(10.0))
    # numel(out) = 15; full-axis stat, both sides padded -> dedup: 10 reduce
    # + 1 divide = 11 stat extra; total 15 + 11 = 26
    assert billed(lambda: fnp.pad(a, (2, 3), mode="mean")) == 26


def test_pad_maximum_2d_charged():
    a = fnp.asarray(np.arange(20.0).reshape(4, 5))
    # numel(out) = 6*7 = 42; stat extra: axis0 is processed first, so its
    # cross-section still uses axis1's ORIGINAL size: cross=5,sl=4 -> 20;
    # axis1 is processed second, so its cross-section uses axis0's ALREADY
    # GROWN size (4+1+1=6), not axis0's original size 4: cross=6,sl=5 -> 30;
    # stat extra 50; total 42 + 50 = 92
    assert billed(lambda: fnp.pad(a, ((1, 1), (0, 2)), mode="maximum")) == 92


def test_pad_median_charged():
    a = fnp.asarray(np.arange(1000.0))
    # numel(out) = 1002; stat extra 1000 (full-axis dedup); total 2002
    assert billed(lambda: fnp.pad(a, (1, 1), mode="median")) == 2002


def test_pad_linear_ramp_charged():
    a = fnp.asarray(np.zeros(100))
    # numel(out) = 150; extra = out - in = 50; total 150 + 50 = 200
    assert (
        billed(lambda: fnp.pad(a, (0, 50), mode="linear_ramp", end_values=5.0)) == 200
    )


def test_pad_odd_reflect_charged():
    a = fnp.asarray(np.arange(10.0))
    # numel(out) = 12; extra = out - in = 2; total 12 + 2 = 14
    assert billed(lambda: fnp.pad(a, (1, 1), mode="reflect", reflect_type="odd")) == 14


def test_pad_callable_rejected():
    a = fnp.asarray(np.arange(10.0))
    with pytest.raises(ValueError, match="callable"):
        fnp.pad(a, (1, 1), mode=lambda *args, **kw: None)


def test_pad_mean_asymmetric_stat_length():
    a = fnp.asarray(np.arange(10.0))
    # numel(out) = 12; both sides padded, stat_length (3,4) not full-axis ->
    # no dedup: reduce 3+4=7, +2 divides = 9 stat extra; total 12 + 9 = 21
    assert billed(lambda: fnp.pad(a, (1, 1), mode="mean", stat_length=(3, 4))) == 21


def test_pad_one_sided_bills_the_discarded_side_too():
    a = fnp.asarray(np.arange(10.0))
    # numel(out) = 13; pad after only (before=0), stat_length=2 -> numpy's
    # _get_stats ALWAYS computes a left-side reduction (even though before=0
    # means its result is discarded into a width-0 output region), plus the
    # right-side reduction the output actually uses: cross(1)*2 [left] +
    # cross(1)*2 [right] = 4 stat extra; total 13 + 4 = 17. (Previously this
    # test asserted the discarded left-side reduction was NOT billed -- that
    # was an under-bill: numpy really performs that reduction.)
    assert billed(lambda: fnp.pad(a, (0, 3), mode="maximum", stat_length=2)) == 17


def test_pad_2d_mean_charged():
    a = fnp.asarray(np.arange(20.0).reshape(4, 5))
    # numel(out) = 6*7 = 42; stat extra: axis0 (1,1) full-axis dedup, cross=5
    # (axis1's ORIGINAL size -- axis0 is processed first): 5*4 reduce + 5
    # divides = 25; axis1 (1,1) full-axis dedup, cross=6 (axis0's ALREADY
    # GROWN size 4+1+1=6, not its original 4 -- axis0 was padded first):
    # 6*5 reduce + 6 divides = 36; stat extra 61; total 42 + 61 = 103
    assert billed(lambda: fnp.pad(a, ((1, 1), (1, 1)), mode="mean")) == 103


def test_pad_zero_width_still_reduces_the_full_axis():
    a = fnp.asarray(np.arange(10.0))
    # pad_width=(0,0) -> numel(out) == numel(in) == 10, nothing is actually
    # placed in the output from this axis's stat -- but numpy's per-axis
    # stat loop runs unconditionally on pad width (it iterates every axis
    # and always computes at least the left-side reduction), so it still
    # performs a full-axis maximum reduction here; the result is simply
    # discarded into a width-0 output region. stat extra = cross(1)*10 = 10;
    # total 10 + 10 = 20. (Previously this test asserted a (0, 0) axis was
    # "free" -- that was an under-bill: numpy really performs the reduction.
    # A wholly EMPTY input -- some axis length 0 -- is the one case that
    # truly skips the reduction loop; see test_pad_empty_input_floors_at_one
    # in test_triage_price_pins.py, which uses mode='constant'.)
    assert billed(lambda: fnp.pad(a, (0, 0), mode="maximum")) == 20


def test_pad_constant_malformed_pad_width_raises_numpy_error():
    a = fnp.asarray(np.arange(10.0))
    # free mode must surface numpy's ValueError (not an IndexError from cost calc)
    with pytest.raises(ValueError):
        fnp.pad(a, ((1, 2), (3, 4)), mode="constant")


def test_ravel_multi_index_charged():
    rows = fnp.asarray(np.arange(100) % 10)
    cols = fnp.asarray(np.arange(100) % 10)
    # numel(output) = N = 100 (Task 8: replaces the old 2*(ndim-1)*N formula)
    assert billed(lambda: fnp.ravel_multi_index((rows, cols), (10, 10))) == 100


def test_ravel_multi_index_mode_does_not_change_cost():
    """Task 8: cost is numel(output) regardless of mode -- clip/wrap no
    longer add +N (the old 2*(ndim-1)*N(+N for clip/wrap) formula did)."""
    rows = fnp.asarray(np.arange(100) % 10)
    cols = fnp.asarray(np.arange(100) % 10)
    assert (
        billed(lambda: fnp.ravel_multi_index((rows, cols), (10, 10), mode="clip"))
        == 100
    )


def test_trim_zeros_charged():
    a = fnp.asarray(np.array([0, 0, 1, 2, 3, 0, 0], dtype=float))
    # value scan = numel(input) = 7
    assert billed(lambda: fnp.trim_zeros(a)) == 7


def test_copyto_same_dtype_free():
    dst = fnp.zeros(100, dtype=np.float64)
    src = fnp.asarray(np.ones(100, dtype=np.float64))
    # Even same-dtype copy bills per element written: 100 elements at unit rate 1.0
    assert billed(lambda: fnp.copyto(dst, src)) == 100


def test_copyto_value_changing_cast_charged():
    dst = fnp.zeros(100, dtype=np.int64)
    src = fnp.asarray(np.random.default_rng(0).standard_normal(100))
    # float64 -> int64 cast computes per element -> numel(dst) = 100
    assert billed(lambda: fnp.copyto(dst, src, casting="unsafe")) == 100


def test_copyto_lossless_widening_free():
    # copyto bills per element written: 100 elements at unit rate 1.0
    dst = fnp.zeros(100, dtype=np.float64)
    src = fnp.asarray(np.ones(100, dtype=np.float32))
    assert billed(lambda: fnp.copyto(dst, src)) == 100


def test_charged_modes_billed_under_production_weights():
    from flopscope._weights import load_weights, reset_weights

    load_weights()
    try:
        a = fnp.asarray(np.arange(1000.0))
        assert billed(lambda: fnp.pad(a, (1, 1), mode="mean")) > 0
        assert billed(lambda: fnp.pad(a, (1, 1), mode="median")) > 0
        rows = fnp.asarray(np.arange(100) % 10)
        assert billed(lambda: fnp.ravel_multi_index((rows, rows), (10, 10))) > 0
        assert billed(lambda: fnp.trim_zeros(a)) > 0
    finally:
        reset_weights()
