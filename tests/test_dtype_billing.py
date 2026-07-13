"""Resolved-dtype computation and complex factor lookup."""

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
    assert resolve_billing_dtype((np.dtype("float64"), np.dtype("complex64"))) == np.complex128


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
