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
    with pytest.raises(UnsupportedDtypeError):
        rate_for(np.dtype("object"))


def test_complex_factor_real_dtype_is_one():
    assert complex_factor_for("multiply", np.dtype("float64")) == 1.0


def test_complex_factor_reads_registry():
    assert complex_factor_for("multiply", np.dtype("complex128")) == 6.0
    assert complex_factor_for("add", np.dtype("complex128")) == 2.0


def test_complex_factor_fails_closed_when_unclassified():
    with pytest.raises(UnsupportedDtypeError):
        complex_factor_for("left_shift", np.dtype("complex128"))  # complex-illegal op


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
