"""A non-numeric ``out=`` must not launder the arithmetic's billing rate.

The billed rate is meant to track the arithmetic a call actually performs.
Non-numeric kinds (object, str, bytes, void, datetime, timedelta) bill at the
neutral rate 1.0, which is right when they are the *operands* -- object
arithmetic is interpreted Python, so there is no precision-packing to exploit
and the wall time shows up as residual anyway.

It is wrong when the non-numeric dtype arrives as ``out=``. Contractions
compute in the operands' dtypes and only then store into ``out``, so the
arithmetic stays native-speed while the stored kind drags the resolved billing
dtype down -- dropping both the dtype rate and the complex factor. These tests
pin that ``out=`` can only widen the bill, never launder it.

For ``einsum`` specifically, the str/bytes destinations are no longer billed
honestly -- they are refused, for free, because plain numpy refuses them too.
See ``test_einsum_out_casting_parity.py``; the pin here is that the refusal
costs nothing, since an unbillable refusal is what keeps it from becoming its
own exploit.
"""

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights

N = 24


def _billed(fn):
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn()
        return b.flops_used


def _complex_pair(n=N):
    rng = np.random.default_rng(0)
    a = (rng.random((n, n)) + 1j * rng.random((n, n))).astype(np.complex128)
    b = (rng.random((n, n)) + 1j * rng.random((n, n))).astype(np.complex128)
    return a, b


def _real_pair(n=N):
    rng = np.random.default_rng(1)
    return rng.random((n, n)), rng.random((n, n))


def test_einsum_nonnumeric_out_does_not_discount_complex():
    load_weights()
    a, b = _complex_pair()
    honest = _billed(lambda: fnp.einsum("ij,jk->ik", a, b))
    out = np.empty((N, N), dtype=object)
    laundered = _billed(lambda: fnp.einsum("ij,jk->ik", a, b, out=out))
    assert laundered == honest


def test_einsum_nonnumeric_out_does_not_discount_float64():
    load_weights()
    a, b = _real_pair()
    honest = _billed(lambda: fnp.einsum("ij,jk->ik", a, b))
    out = np.empty((N, N), dtype=object)
    assert _billed(lambda: fnp.einsum("ij,jk->ik", a, b, out=out)) == honest


@pytest.mark.parametrize("out_dtype", ["U64", "S64", "U32", "S16"])
@pytest.mark.parametrize("pair", ["complex", "real"])
def test_einsum_string_out_is_refused_for_free(out_dtype, pair):
    """The str/bytes destinations used to be the sharpest form of this launder:
    they reached the contraction, billed at the neutral rate, and then stored
    a *rendering* of the answer. numpy has never allowed it -- einsum has no
    string inner loop -- so the destination-casting parity fix now refuses
    them outright, which is a stronger pin than "bills the honest rate".

    Refusal must cost nothing. There are no refunds, so an op that raises
    after billing would be a free way to burn a rival's budget, and an op that
    raises after billing *its own* cost would still charge for arithmetic the
    caller never received.
    """
    load_weights()
    a, b = _complex_pair() if pair == "complex" else _real_pair()
    out = np.empty((N, N), dtype=out_dtype)
    with f.BudgetContext(flop_budget=10**18, quiet=True) as budget:
        with pytest.raises(TypeError):
            fnp.einsum("ij,jk->ik", a, b, out=out)
        assert budget.flops_used == 0


def test_contraction_helper_nonnumeric_out_does_not_discount():
    """vecdot/matvec/vecmat route through _einsum_routed_binary, a second
    place that folds out's dtype into the billing resolution."""
    load_weights()
    a, b = _complex_pair()
    honest = _billed(lambda: fnp.vecdot(a, b))
    out = np.empty(N, dtype=object)
    assert _billed(lambda: fnp.vecdot(a, b, out=out)) == honest


def test_ufunc_nonnumeric_out_does_not_discount():
    load_weights()
    a, b = _complex_pair()
    honest = _billed(lambda: fnp.multiply(a, b))
    out = np.empty((N, N), dtype=object)
    assert (
        _billed(
            lambda: fnp.multiply(a, b, out=out)  # pyright: ignore[reportArgumentType]
        )
        >= honest
    )


# --- the widest-buffer doctrine for NUMERIC out= must survive -------------


def test_numeric_widening_out_still_bills_wider():
    """PR #151 doctrine: a wider out= bills the wider rate, at astype parity."""
    load_weights()
    a32 = np.ones(1000, dtype=np.float32)
    via_out = _billed(
        lambda: fnp.add(
            a32,
            0.0,
            out=np.empty(1000, np.float64),  # pyright: ignore[reportArgumentType]
        )
    )
    via_astype = _billed(lambda: fnp.astype(a32, np.float64))
    assert via_out == via_astype == 2000


@pytest.mark.parametrize("op", ["divmod", "modf", "frexp"])
def test_multi_output_nonnumeric_out_does_not_discount(op):
    """The multi-output ufuncs fold each out= element's dtype separately."""
    load_weights()
    a = np.random.default_rng(0).random((32, 32)) + 1.0
    args = (a, a + 1.0) if op == "divmod" else (a,)
    fn = getattr(fnp, op)
    honest = _billed(lambda: fn(*args))
    outs = (np.empty((32, 32), dtype=object), np.empty((32, 32), dtype=object))
    assert _billed(lambda: fn(*args, out=outs)) == honest


@pytest.mark.parametrize("other", ["float64", "complex128"])
def test_mixed_nonnumeric_operand_still_bills_neutral(other):
    """A non-numeric *operand* describes the arithmetic even when a numeric
    operand is alongside it: numpy runs its object loop and returns object.
    Filtering it out would price the object loop at the numeric rate -- and at
    the complex factor too, for a complex partner."""
    load_weights()
    n = 8
    o = np.array(np.random.default_rng(0).random((n, n)), dtype=object)
    partner = np.ones((n, n), dtype=other)
    both_object = _billed(lambda: fnp.multiply(o, o))
    mixed = _billed(lambda: fnp.multiply(o, partner))
    assert np.asarray(fnp.multiply(o, partner)).dtype == object
    assert mixed == both_object


def test_nonnumeric_operands_still_bill_neutral():
    """The 1.0 neutral rate is still correct when the operands themselves are
    non-numeric -- that arithmetic really is interpreted Python."""
    load_weights()
    o = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=object)
    neutral = _billed(lambda: fnp.multiply(o, o))
    same_shape_f64 = _billed(lambda: fnp.multiply(np.ones((2, 2)), np.ones((2, 2))))
    assert neutral < same_shape_f64  # f64 rate is 2.0, object stays 1.0


# --- the gate must not be reachable around, via a container ---------------


@pytest.mark.parametrize("wrap", [lambda d: d, lambda d: (d,)], ids=["bare", "tuple"])
def test_a_nonnumeric_out_does_not_discount_through_a_container(wrap):
    """The whole gate is ``isinstance(out, np.ndarray)``, so a container used
    to slip past it -- billing as though no destination were supplied at all.
    Normalizing ``out`` before the gate is what keeps the tuple honest."""
    load_weights()
    a, b = _complex_pair()
    honest = _billed(lambda: fnp.vecdot(a, b))
    out = np.empty(N, dtype=object)
    assert _billed(lambda: fnp.vecdot(a, b, out=wrap(out))) == honest


@pytest.mark.parametrize("wrap", [lambda d: d, lambda d: (d,)], ids=["bare", "tuple"])
def test_a_numeric_widening_out_bills_wider_through_a_container(wrap):
    """The converse direction: the widest-participating-buffer charge must
    apply to a wrapped destination exactly as it does to a bare one, or the
    container becomes a way to buy a wide accumulator at the narrow rate."""
    load_weights()
    a, b = _real_pair()
    narrow = _billed(lambda: fnp.vecdot(a, b))
    wide = np.zeros(N, dtype=np.complex128)
    assert _billed(lambda: fnp.vecdot(a, b, out=wrap(wide))) > narrow
