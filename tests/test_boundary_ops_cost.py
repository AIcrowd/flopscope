"""Cost-model tests for boundary/padding ops under the data-movement free-tier.

``pad`` is free for pure data-movement modes (constant/edge/empty/wrap and
reflect/symmetric with reflect_type='even') but must bill a real analytic cost
for value-computing modes (maximum/minimum/mean/median/linear_ramp and
reflect/symmetric with reflect_type='odd'), and reject ``mode=<callable>``.
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
    assert billed(lambda: fnp.pad(a, (1, 1), mode="constant")) == 0


def test_pad_even_reflect_free():
    a = fnp.asarray(np.arange(10.0))
    assert billed(lambda: fnp.pad(a, (1, 1), mode="reflect")) == 0


def test_pad_mean_1d_charged():
    a = fnp.asarray(np.arange(10.0))
    # full-axis stat, both sides padded -> dedup: 10 reduce + 1 divide = 11
    assert billed(lambda: fnp.pad(a, (2, 3), mode="mean")) == 11


def test_pad_maximum_2d_charged():
    a = fnp.asarray(np.arange(20.0).reshape(4, 5))
    # axis0 (1,1): cross=5, sl=4 -> 20 ; axis1 (0,2): cross=4, sl=5 -> 20 ; total 40
    assert billed(lambda: fnp.pad(a, ((1, 1), (0, 2)), mode="maximum")) == 40


def test_pad_median_charged():
    a = fnp.asarray(np.arange(1000.0))
    assert billed(lambda: fnp.pad(a, (1, 1), mode="median")) == 1000


def test_pad_linear_ramp_charged():
    a = fnp.asarray(np.zeros(100))
    assert (
        billed(lambda: fnp.pad(a, (0, 50), mode="linear_ramp", end_values=5.0)) == 100
    )


def test_pad_odd_reflect_charged():
    a = fnp.asarray(np.arange(10.0))
    assert billed(lambda: fnp.pad(a, (1, 1), mode="reflect", reflect_type="odd")) == 4


def test_pad_callable_rejected():
    a = fnp.asarray(np.arange(10.0))
    with pytest.raises(ValueError, match="callable"):
        fnp.pad(a, (1, 1), mode=lambda *args, **kw: None)


def test_pad_mean_asymmetric_stat_length():
    a = fnp.asarray(np.arange(10.0))
    # both sides padded, stat_length (3,4) not full-axis -> no dedup:
    # reduce 3+4=7, +2 divides = 9
    assert billed(lambda: fnp.pad(a, (1, 1), mode="mean", stat_length=(3, 4))) == 9


def test_pad_one_sided_only_charges_padded_side():
    a = fnp.asarray(np.arange(10.0))
    # pad after only, stat_length=2 -> charge only the after side: cross(1)*2 = 2
    # (numpy also computes a discarded before-stat; we intentionally do not bill it)
    assert billed(lambda: fnp.pad(a, (0, 3), mode="maximum", stat_length=2)) == 2


def test_pad_2d_mean_charged():
    a = fnp.asarray(np.arange(20.0).reshape(4, 5))
    # axis0 (1,1) full-axis dedup: 5*4 reduce + 5 divides = 25
    # axis1 (1,1) full-axis dedup: 4*5 reduce + 4 divides = 24 ; total 49
    assert billed(lambda: fnp.pad(a, ((1, 1), (1, 1)), mode="mean")) == 49


def test_pad_zero_width_free():
    a = fnp.asarray(np.arange(10.0))
    assert billed(lambda: fnp.pad(a, (0, 0), mode="maximum")) == 0


def test_pad_constant_malformed_pad_width_raises_numpy_error():
    a = fnp.asarray(np.arange(10.0))
    # free mode must surface numpy's ValueError (not an IndexError from cost calc)
    with pytest.raises(ValueError):
        fnp.pad(a, ((1, 2), (3, 4)), mode="constant")


def test_ravel_multi_index_charged():
    rows = fnp.asarray(np.arange(100) % 10)
    cols = fnp.asarray(np.arange(100) % 10)
    # ndim=2, N=100 -> 2*(2-1)*100 = 200
    assert billed(lambda: fnp.ravel_multi_index((rows, cols), (10, 10))) == 200


def test_ravel_multi_index_clip_adds_n():
    rows = fnp.asarray(np.arange(100) % 10)
    cols = fnp.asarray(np.arange(100) % 10)
    # 200 + N(=100) for clip = 300
    assert (
        billed(lambda: fnp.ravel_multi_index((rows, cols), (10, 10), mode="clip"))
        == 300
    )


def test_trim_zeros_charged():
    a = fnp.asarray(np.array([0, 0, 1, 2, 3, 0, 0], dtype=float))
    # value scan = numel(input) = 7
    assert billed(lambda: fnp.trim_zeros(a)) == 7


def test_copyto_same_dtype_free():
    dst = fnp.zeros(100, dtype=np.float64)
    src = fnp.asarray(np.ones(100, dtype=np.float64))
    # same-dtype copy bills per element written: 100 elements
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
