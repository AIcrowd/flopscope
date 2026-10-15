"""Unary constraints select computation independently of destination storage."""

import warnings
from typing import Any

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights


@pytest.mark.parametrize("constraint", ["dtype", "signature", "sig", "tuple"])
@pytest.mark.parametrize("dtype,price", [(np.float64, 128), (np.complex128, 396)])
def test_exp_forced_loop_price_and_numpy_parity(constraint, dtype, price):
    load_weights()
    x = np.arange(4, dtype=np.float32)
    code = np.dtype(dtype).char
    kw: dict[str, Any] = (
        {constraint: dtype}
        if constraint == "dtype"
        else {
            "signature" if constraint == "tuple" else constraint: (dtype, dtype)
            if constraint == "tuple"
            else f"{code}->{code}"
        }
    )
    expected = np.exp(x, **kw)
    with f.budget(10**9, quiet=True) as budget:
        actual = fnp.exp(x, **kw)
    np.testing.assert_array_equal(np.asarray(actual), expected)
    assert actual.dtype == expected.dtype
    assert budget.flops_used == price


@pytest.mark.parametrize("name", ["exp", "log", "sqrt", "sin", "cos", "negative"])
@pytest.mark.parametrize("constraint", [{"dtype": np.float32}, {"sig": "f->f"}])
def test_wider_destination_is_independent_of_forced_loop(name, constraint):
    load_weights()
    x = np.arange(1, 5, dtype=np.float64)
    out = fnp.empty(4, dtype=np.float64)
    expected = np.empty(4, dtype=np.float64)
    getattr(np, name)(x, out=expected, **constraint)
    # Same compute loop, but the wider destination must retain float64 pricing.
    with f.budget(10**9, quiet=True) as baseline:
        getattr(fnp, name)(x, dtype=np.float64)
    with f.budget(10**9, quiet=True) as budget:
        actual = getattr(fnp, name)(x, out=out, **constraint)
    assert actual is out
    np.testing.assert_array_equal(np.asarray(actual), expected)
    assert budget.flops_used == baseline.flops_used


@pytest.mark.parametrize("constraint", [{"dtype": np.float64}, {"sig": "d->d"}])
def test_narrow_destination_cannot_hide_wider_loop(constraint):
    load_weights()
    x = np.arange(4, dtype=np.float32)
    out = fnp.empty(4, dtype=np.float32)
    with f.budget(10**9, quiet=True) as budget:
        fnp.exp(x, out=out, **constraint)
    assert budget.flops_used == 128


def test_explicit_narrow_loop_replaces_wide_source_floor():
    load_weights()
    x = np.arange(4, dtype=np.float64)
    with f.budget(10**9, quiet=True) as budget:
        actual = fnp.exp(x, sig="f->f")
    assert actual.dtype == np.float32
    assert budget.flops_used == 64


def test_predicate_signature_prices_input_loop_not_bool_output():
    load_weights()
    x = np.arange(4, dtype=np.float32)
    with f.budget(10**9, quiet=True) as budget:
        actual = fnp.signbit(x, sig="d->?")
    assert actual.dtype == np.bool_
    assert budget.flops_used == 8


@pytest.mark.parametrize("name,signature", [("modf", "d->dd"), ("frexp", "d->di")])
@pytest.mark.parametrize("destination", ["none", "natural", "partial", "wider"])
def test_multioutput_constraints_and_destinations(name, signature, destination):
    load_weights()
    x = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    # A float32 forced loop costs 8 units; a wider destination costs 16.
    signature = signature.replace("d", "f")
    second_dtype = np.int32 if name == "frexp" else np.float32
    out = None
    if destination != "none":
        out = (
            None
            if destination == "partial"
            else fnp.empty(
                4, dtype=np.float64 if destination == "wider" else np.float32
            ),
            fnp.empty(4, dtype=second_dtype),
        )
    with f.budget(10**9, quiet=True) as budget:
        actual = getattr(fnp, name)(x, signature=signature, out=out)
    expected = getattr(np, name)(x, signature=signature)
    for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_array_equal(np.asarray(a), e)
        if out is not None and out[index] is not None:
            assert a is out[index]
    assert budget.flops_used == (16 if destination == "wider" else 8)


@pytest.mark.parametrize("name,signature", [("modf", "d->dd"), ("frexp", "d->di")])
def test_multioutput_forced_wider_loop(name, signature):
    load_weights()
    x = np.arange(4, dtype=np.float32)
    with f.budget(10**9, quiet=True) as budget:
        actual = getattr(fnp, name)(x, sig=signature)
    assert actual[0].dtype == np.float64
    assert budget.flops_used == 16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signature": "f->f", "sig": "f->f"},
        {"signature": "f->f", "dtype": None},
        {"sig": "d->d", "casting": "no"},
        {"signature": "i->i"},
        {"signature": (np.float32,) * 3},
    ],
)
def test_invalid_constraints_cost_nothing(kwargs):
    x = np.arange(4, dtype=np.float32)
    with pytest.raises((TypeError, ValueError)) as reference:
        np.exp(x, **kwargs)
    with f.budget(10**9, quiet=True) as budget:
        with pytest.raises(type(reference.value)):
            fnp.exp(x, **kwargs)
    assert budget.flops_used == 0
    assert not budget.op_log


def test_signature_dtype_like_is_frozen_once():
    load_weights()
    x = np.arange(4, dtype=np.float32)

    class StatefulDtype:
        reads = 0

        @property
        def dtype(self):
            self.reads += 1
            return np.dtype("d" if self.reads == 1 else "f")

    slot = StatefulDtype()
    with f.budget(10**9, quiet=True) as budget:
        actual = fnp.exp(x, signature=(slot, None))
    assert slot.reads == 1
    assert actual.dtype == np.float64
    assert budget.flops_used == 128


def test_forced_complex_loop_narrowed_to_real_retains_complex_price():
    load_weights()
    x = np.arange(4, dtype=np.float32)
    out = fnp.empty(4, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        with f.budget(10**9, quiet=True) as budget:
            fnp.exp(x, signature="D->D", casting="unsafe", out=out)
    assert budget.flops_used == 396


@pytest.mark.parametrize("name,signature", [("exp", "d->d"), ("modf", "d->dd")])
def test_invalid_destination_cast_costs_nothing(name, signature):
    x = np.arange(4, dtype=np.float32)
    out = fnp.empty(4, dtype=np.int32)
    destinations = out if name == "exp" else (out, None)
    with f.budget(10**9, quiet=True) as budget:
        with pytest.raises(TypeError):
            getattr(fnp, name)(x, signature=signature, out=destinations)
    assert budget.flops_used == 0
    assert not budget.op_log


@pytest.mark.parametrize("constraint", [{"dtype": np.float32}, {"sig": "f->ff"}])
def test_modf_explicit_narrow_loop_still_bills_wide_destination(constraint):
    load_weights()
    x = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    out = (fnp.empty(4, dtype=np.float64), fnp.empty(4, dtype=np.float32))
    with f.budget(10**9, quiet=True) as budget:
        actual = fnp.modf(x, out=out, **constraint)
    assert actual[0] is out[0]
    assert budget.flops_used == 16


def test_explicit_dtype_like_is_frozen_once():
    load_weights()

    class StatefulDtype:
        reads = 0

        @property
        def dtype(self):
            self.reads += 1
            return np.dtype("d" if self.reads == 1 else "f")

    x = np.arange(4, dtype=np.float32)
    dtype = StatefulDtype()
    with f.budget(10**9, quiet=True) as budget:
        actual = fnp.exp(x, dtype=dtype)
    assert dtype.reads == 1
    assert actual.dtype == np.float64
    assert budget.flops_used == 128
