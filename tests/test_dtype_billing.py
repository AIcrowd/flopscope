"""Resolved-dtype computation and complex factor lookup."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from flopscope._dtype_billing import (
    billing_operand,
    complex_factor_for,
    rate_for,
    resolve_billing_dtype,
)
from flopscope._weights import load_weights
from flopscope.errors import UnsupportedDtypeError


def test_resolve_empty_is_dtype_neutral():
    assert resolve_billing_dtype(()) is None


def test_resolve_promotes_like_numpy():
    assert resolve_billing_dtype((np.dtype("int32"), np.dtype("float32"))) == np.float64
    assert (
        resolve_billing_dtype((np.dtype("float64"), np.dtype("complex64")))
        == np.complex128
    )


def test_resolve_includes_explicit_output_dtype():
    # matmul(int32, float32, dtype=int8) still bills float64
    resolved = resolve_billing_dtype(
        (np.dtype("int32"), np.dtype("float32"), np.dtype("int8"))
    )
    assert resolved == np.float64


def test_billing_operand_keeps_python_scalars_weak():
    # NEP 50: f32_array * 2.0 stays float32
    arr = np.ones(3, dtype=np.float32)
    ops = (billing_operand(arr, arr), billing_operand(2.0, np.asarray(2.0)))
    assert resolve_billing_dtype(ops) == np.float32


def test_billing_operand_coerces_lists():
    coerced = np.asarray([1.0, 2.0])
    assert billing_operand([1.0, 2.0], coerced) == np.float64


def test_rate_for_uses_active_table():
    load_weights()
    assert rate_for(np.dtype("float64")) == 2.0
    assert rate_for(np.dtype("complex64")) == 1.0
    # Extended precision is PRICED by width class, not banned: float128 costs
    # 2x float64 (4.0 in fp32 units); packing two f32-payload products into
    # its mantissa still loses (4.0 vs honest 2.0).
    from flopscope._weights import get_dtype_rate

    assert get_dtype_rate("float128") == 4.0
    assert get_dtype_rate("complex256") == 4.0  # component width float128
    # Numeric dtypes genuinely OUTSIDE the table (future types) still fail
    # closed. Name-level check: platform-independent.
    with pytest.raises(UnsupportedDtypeError):
        get_dtype_rate("float256")


def test_rate_for_non_numeric_kinds_bill_neutral():
    # object/str/bytes/datetime64/timedelta64 are not floating-point
    # arithmetic; no precision-packing exploit is possible through them, and
    # the numpy-compat guarantee requires them to keep working. They bill at
    # the neutral rate 1.0 rather than failing closed.
    load_weights()
    assert rate_for(np.dtype("object")) == 1.0
    assert rate_for(np.dtype("str_")) == 1.0
    assert rate_for(np.dtype("datetime64[s]")) == 1.0
    assert rate_for(np.dtype("timedelta64[s]")) == 1.0


def test_complex_factor_real_dtype_is_one():
    assert complex_factor_for("multiply", np.dtype("float64")) == 1.0


def test_complex_factor_reads_registry():
    assert complex_factor_for("multiply", np.dtype("complex128")) == 6.0
    assert complex_factor_for("add", np.dtype("complex128")) == 2.0


def test_complex_factor_fails_closed_when_illegal():
    # Ops explicitly marked "illegal" (numpy raises on complex) fail closed.
    # This is distinct from an UNCLASSIFIED op (factor None), which now returns
    # 2.0 -- see test_complex_factor_free_op_is_two_not_raise below.
    with pytest.raises(UnsupportedDtypeError):
        complex_factor_for("left_shift", np.dtype("complex128"))  # complex-illegal op


def test_complex_factor_free_op_is_two_not_raise():
    # Free / data-movement / blacklisted ops carry no complex_factor; they
    # relocate or allocate whole complex values, and a complex value is two
    # real components, so the default factor is 2.0 (one unit per component),
    # never raise.
    assert complex_factor_for("reshape", np.dtype("complex128")) == 2.0
    assert complex_factor_for("asarray", np.dtype("complex128")) == 2.0
    assert complex_factor_for("some_unknown_op_xyz", np.dtype("complex128")) == 2.0


def test_complex_factor_exact_requires_override():
    with pytest.raises(RuntimeError):
        complex_factor_for("einsum", np.dtype("complex128"))


def test_complex_factor_ufunc_method_falls_back_to_base():
    c = np.dtype("complex128")
    assert complex_factor_for("multiply.reduce", c) == 6.0
    assert complex_factor_for("add.reduce", c) == 2.0
    assert complex_factor_for("subtract.accumulate", c) == 2.0
    assert complex_factor_for("multiply.outer", c) == 6.0


def test_complex_factor_ufunc_method_illegal_base_still_raises():
    with pytest.raises(UnsupportedDtypeError):
        complex_factor_for("logaddexp.reduce", np.dtype("complex128"))


def test_complex_factor_dotted_registry_key_is_not_stripped():
    # "linalg.outer" is itself a registry key (not a generic ufunc-method
    # name) and must resolve on the direct lookup, never via the ".outer"
    # suffix-stripping fallback (there is no "linalg" registry entry).
    from flopscope._registry import REGISTRY

    direct = REGISTRY["linalg.outer"]["complex_factor"]
    assert complex_factor_for("linalg.outer", np.dtype("complex128")) == direct


# Complex real-FLOP total for contractions
from flopscope._accumulation._cost import AccumulationCost, complex_real_total


class _FakeAcc:
    # mu is the authoritative multiply count (aggregate_einsum sets it to
    # (num_terms-1)*m_total for k<=2, and to the summed per-step mu for a path).
    def __init__(self, total, num_terms, m_total, mu, fallback_used=False):
        self.total = total
        self.num_terms = num_terms
        self.m_total = m_total
        self.mu = mu
        self.fallback_used = fallback_used


def test_complex_real_total_matmul_shape():
    # ij,jk->ik with m=n=8, K=8: total = 2*512 - 64 = 960 (mults 512, adds 448)
    acc = cast(AccumulationCost, _FakeAcc(total=960, num_terms=2, m_total=512, mu=512))
    assert complex_real_total(acc) == 6 * 512 + 2 * 448  # 3968


def test_complex_real_total_pure_product():
    # i,i->i elementwise product: no accumulation, all units are multiplies
    acc = cast(AccumulationCost, _FakeAcc(total=100, num_terms=2, m_total=100, mu=100))
    assert complex_real_total(acc) == 600


def test_complex_real_total_multistep_uses_mu_not_m_total():
    # k>=3 path: aggregate m_total is the output-orbit product, NOT the multiply
    # basis; only mu carries the true multiply count. Here mu=130 while
    # (num_terms-1)*m_total = 2*1000 = 2000 would wrongly trip the adds<0
    # fallback. Correct: mults=130, adds=250-130=120 -> 6*130+2*120=1020.
    acc = cast(
        AccumulationCost,
        _FakeAcc(total=250, num_terms=3, m_total=1000, mu=130),
    )
    assert complex_real_total(acc) == 6 * 130 + 2 * 120  # 1020


def test_complex_real_total_fallback_is_conservative():
    acc = cast(
        AccumulationCost,
        _FakeAcc(total=1000, num_terms=3, m_total=100, mu=200, fallback_used=True),
    )
    assert complex_real_total(acc) == 6000


def test_complex_real_total_mu_none_is_conservative():
    # mu unavailable (no component data): bill every unit as a multiply.
    acc = cast(
        AccumulationCost,
        _FakeAcc(total=500, num_terms=2, m_total=100, mu=None),
    )
    assert complex_real_total(acc) == 3000


def test_resolve_promotion_failure_falls_back_to_heaviest():
    # Some numpy loops accept operand mixes with NO common promoted dtype
    # (logical ufuncs on anything; timedelta64/float64 division). result_type
    # raises DTypePromotionError there; billing falls back to the heaviest
    # individual operand rate instead of failing a call numpy allows.
    load_weights()
    assert (
        resolve_billing_dtype((np.dtype("timedelta64[s]"), np.dtype("float64")))
        == np.float64
    )
    assert (
        resolve_billing_dtype((np.dtype("V8"), np.dtype("float64"), np.dtype("int32")))
        == np.float64
    )


def test_reduction_billing_dtype_resolution_order():
    from flopscope._dtype_billing import reduction_billing_dtype

    load_weights()

    i32, i64 = np.dtype(np.int32), np.dtype(np.int64)
    f32, f64 = np.dtype(np.float32), np.dtype(np.float64)

    # No explicit/out: the implicit default accumulator wins.
    assert reduction_billing_dtype(i32, default_dtype=i64) == i64
    # Explicit dtype REPLACES the implicit default (mattmotoki's case).
    assert (
        reduction_billing_dtype(i32, explicit_dtype=np.int32, default_dtype=i64) == i32
    )
    # Explicit wide dtype is honored (positional-dtype case feeds this too).
    assert reduction_billing_dtype(f32, explicit_dtype=np.float64) == f64
    # out= without dtype= sets the accumulator (numpy ufunc.reduce semantics).
    assert reduction_billing_dtype(f32, out_dtype=f64) == f64
    # dtype-like inputs (type objects, strings) are normalized.
    assert (
        reduction_billing_dtype(i32, explicit_dtype="int32", default_dtype=i64) == i32
    )


def test_reduction_billing_dtype_floors_at_input_rate():
    from flopscope._dtype_billing import reduction_billing_dtype

    load_weights()

    f32, f64 = np.dtype(np.float32), np.dtype(np.float64)
    # Narrowing below the input does NOT discount: sum(f64, dtype=f32)
    # bills the f64 rate (per-element lossy downcast is astype-priced).
    assert reduction_billing_dtype(f64, explicit_dtype=np.float32) == f64
    # mean-family: integer input computing in f32 per explicit dtype is honest.
    assert (
        reduction_billing_dtype(
            np.dtype(np.int32),
            explicit_dtype=np.float32,
            default_dtype=np.dtype(np.float64),
        )
        == f32
    )
    # Max-by-rate, not result_type: int64 + float32 out must NOT promote to
    # a third dtype (result_type would say float64).
    got = reduction_billing_dtype(
        np.dtype(np.int64), explicit_dtype=np.float32, out_dtype=f32
    )
    assert got == np.dtype(np.int64)


def test_unary_float_loop_dtype_maps_ints_by_size():
    from flopscope._dtype_billing import unary_float_loop_dtype as u

    # numpy unary float-only ufuncs pick the same-size float loop:
    # exp(int8)->float16, exp(int16)->float32, exp(int32/int64)->float64.
    assert u(np.dtype(np.bool_)) == np.dtype(np.float16)
    assert u(np.dtype(np.int8)) == np.dtype(np.float16)
    assert u(np.dtype(np.uint8)) == np.dtype(np.float16)
    assert u(np.dtype(np.int16)) == np.dtype(np.float32)
    assert u(np.dtype(np.int32)) == np.dtype(np.float64)
    assert u(np.dtype(np.int64)) == np.dtype(np.float64)
    # floats and complex pass through untouched
    assert u(np.dtype(np.float32)) == np.dtype(np.float32)
    assert u(np.dtype(np.complex64)) == np.dtype(np.complex64)
    # non-numeric kinds pass through (neutral billing handles them)
    assert u(np.dtype("O")) == np.dtype("O")


def test_binary_float_loop_dtype_maps_all_ints_to_f64():
    from flopscope._dtype_billing import binary_float_loop_dtype as b

    # binary float-only ufuncs have no int loops: divide(int8,int8)->float64.
    for dt in (np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint32):
        assert b(np.dtype(dt)) == np.dtype(np.float64)
    assert b(np.dtype(np.float32)) == np.dtype(np.float32)
    assert b(np.dtype(np.complex128)) == np.dtype(np.complex128)


def test_fft_billing_dtype():
    from flopscope._dtype_billing import fft_billing_dtype as fbd

    assert fbd(np.dtype(np.float16)) == np.dtype(np.complex64)
    assert fbd(np.dtype(np.float32)) == np.dtype(np.complex64)
    assert fbd(np.dtype(np.complex64)) == np.dtype(np.complex64)
    for dt in (np.int8, np.int32, np.int64, np.float64, np.complex128):
        assert fbd(np.dtype(dt)) == np.dtype(np.complex128)


def test_linalg_compute_dtype_and_tuple_helper():
    from flopscope._dtype_billing import linalg_billing_dtypes, linalg_compute_dtype

    assert linalg_compute_dtype(np.dtype(np.int32)) == np.dtype(np.float64)
    assert linalg_compute_dtype(np.dtype(np.float32)) == np.dtype(np.float32)
    assert linalg_compute_dtype(np.dtype(np.complex64)) == np.dtype(np.complex64)
    assert linalg_billing_dtypes(np.dtype(np.int64), np.dtype(np.int64)) == (
        np.dtype(np.float64),
    )
    assert linalg_billing_dtypes(np.dtype(np.float32)) == (np.dtype(np.float32),)
    # numpy.linalg._commonType: ANY int/bool operand forces the double driver.
    assert linalg_billing_dtypes(np.dtype(np.bool_), np.dtype(np.float32)) == (
        np.dtype(np.float64),
    )
    assert linalg_billing_dtypes(np.dtype(np.uint16), np.dtype(np.float32)) == (
        np.dtype(np.float64),
    )
    assert linalg_billing_dtypes(np.dtype(np.int8), np.dtype(np.complex64)) == (
        np.dtype(np.complex128),
    )
    # all-inexact operands keep the promoted single-precision driver
    assert linalg_billing_dtypes(np.dtype(np.complex64), np.dtype(np.float32)) == (
        np.dtype(np.complex64),
    )
