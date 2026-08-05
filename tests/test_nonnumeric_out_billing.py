"""A non-numeric ``out=`` (or operand) is refused, not priced.

This file predates the numeric-allowlist dtype ban: it used to pin that a
non-numeric ``out=`` bills at the operand rate rather than laundering the
arithmetic down to the neutral rate 1.0, using str/bytes/datetime/timedelta
as exemplars precisely because they were still priced back then (only object
was refused outright). The ban now covers every non-numeric kind -- str,
bytes, structured/void, datetime64, timedelta64, and object alike -- so there
is no remaining non-numeric exemplar left to price at all: every case this
file used to demonstrate "bills correctly, does not launder" is refused
before pricing runs, a strictly stronger guarantee. This file is therefore
largely obsolete as a *pricing* test file; what survives below either
(a) still demonstrates real pricing behaviour with purely NUMERIC dtypes
(the widest-participating-buffer doctrine), or (b) is retained, not deleted,
and now pins the refusal itself. See ``test_object_dtype_ban.py`` for the
ban's own dedicated coverage.

For ``einsum`` specifically, a non-numeric destination is refused for free --
flopscope's own ban fires before numpy's einsum call ever runs, which is a
stronger, earlier refusal than numpy's own casting/loop failure (see
``test_einsum_out_casting_parity.py``). The pin here is that refusal costs
nothing, since an unbillable refusal is what keeps a refusal from becoming
its own exploit.
"""

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights
from flopscope.errors import UnsupportedDtypeError

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


def test_einsum_object_out_is_refused_for_free_complex_operands():
    """object used to be the sharpest form of this launder for a complex
    contraction: it reached the contraction, billed at the neutral rate, and
    stored a live Python-object rendering of the answer. As of the dtype ban
    it is refused before any billing happens -- the same free-refusal
    guarantee ``test_einsum_string_out_is_refused_for_free`` pins for
    str/bytes destinations, checked here for object specifically since it is
    refused via ``refuse_non_numeric_dtype`` before numpy's einsum call ever
    runs, same as every other non-numeric destination now."""
    load_weights()
    a, b = _complex_pair()
    out = np.empty((N, N), dtype=object)
    with f.BudgetContext(flop_budget=10**18, quiet=True) as budget:
        with pytest.raises(UnsupportedDtypeError):
            fnp.einsum("ij,jk->ik", a, b, out=out)
        assert budget.flops_used == 0


def test_einsum_object_out_is_refused_for_free_real_operands():
    """Same refusal, real (float64) operands -- object no longer discounts a
    real contraction's rate to 1.0 either."""
    load_weights()
    a, b = _real_pair()
    out = np.empty((N, N), dtype=object)
    with f.BudgetContext(flop_budget=10**18, quiet=True) as budget:
        with pytest.raises(UnsupportedDtypeError):
            fnp.einsum("ij,jk->ik", a, b, out=out)
        assert budget.flops_used == 0


@pytest.mark.parametrize("out_dtype", ["U64", "S64", "U32", "S16"])
@pytest.mark.parametrize("pair", ["complex", "real"])
def test_einsum_string_out_is_refused_for_free(out_dtype, pair):
    """The str/bytes destinations used to be the sharpest form of this launder:
    they reached the contraction, billed at the neutral rate, and then stored
    a *rendering* of the answer. numpy has never allowed it -- einsum has no
    string inner loop -- so a destination-casting parity fix once refused
    them outright via numpy's own casting/loop failure. Now flopscope's own
    non-numeric-dtype ban refuses the same destinations even earlier, before
    numpy's einsum call runs at all -- ``UnsupportedDtypeError`` is a
    ``TypeError`` subclass, so the accept/refuse contract (and this
    ``pytest.raises(TypeError)``) is unaffected by which of the two now
    fires first.

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


def test_contraction_helper_nonnumeric_out_is_refused():
    """vecdot/matvec/vecmat route through _einsum_routed_binary, a second
    place that folds out's dtype into the billing resolution.

    numpy accepts a complex128 vecdot result written into a str destination
    (it renders the value as text) -- str used to keep the neutral rate 1.0
    that store_billing_dtypes was designed to ignore, discounting the bill.
    Now every non-numeric out=, str included, is refused outright before
    that rate check would even run (see test_object_dtype_ban.py), which
    closes the discount categorically rather than repricing it."""
    load_weights()
    a, b = _complex_pair()
    out = np.empty(N, dtype="U64")
    with f.BudgetContext(flop_budget=10**18, quiet=True) as budget:
        with pytest.raises(UnsupportedDtypeError):
            fnp.vecdot(a, b, out=out)
        assert budget.flops_used == 0


def test_ufunc_nonnumeric_out_is_refused():
    """Same fix as test_contraction_helper_nonnumeric_out_is_refused above,
    for a plain elementwise ufunc rather than the contraction-helper path."""
    load_weights()
    a, b = _complex_pair()
    out = np.empty((N, N), dtype="U64")
    with f.BudgetContext(flop_budget=10**18, quiet=True) as budget:
        with pytest.raises(UnsupportedDtypeError):
            fnp.multiply(a, b, out=out)  # pyright: ignore[reportArgumentType]
        assert budget.flops_used == 0


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
def test_multi_output_nonnumeric_out_is_refused(op):
    """The multi-output ufuncs fold each out= element's dtype separately.

    str used to stand in for the discount object would otherwise have shown
    (object is refused outright by the ban before this rate check would even
    run) -- now every non-numeric out= slot, str included, is refused
    outright too, so both destinations here are refused rather than repriced."""
    load_weights()
    a = np.random.default_rng(0).random((32, 32)) + 1.0
    args = (a, a + 1.0) if op == "divmod" else (a,)
    fn = getattr(fnp, op)
    outs = (np.empty((32, 32), dtype="U64"), np.empty((32, 32), dtype="U64"))
    with f.BudgetContext(flop_budget=10**18, quiet=True) as budget:
        with pytest.raises(UnsupportedDtypeError):
            fn(*args, out=outs)
        assert budget.flops_used == 0


@pytest.mark.parametrize("other", ["float64", "complex128"])
def test_mixed_object_operand_is_refused(other):
    """A non-numeric *operand* alongside a numeric partner used to describe
    the arithmetic and bill at the neutral rate: numpy ran its object loop
    and returned object. As of the dtype ban this is refused instead. What
    this pins is that mixing in a numeric partner does not let an object
    operand slip past the refusal a pure-object call already hits -- see
    test_nonnumeric_operand_is_refused below and test_object_dtype_ban.py."""
    load_weights()
    n = 8
    o = np.array(np.random.default_rng(0).random((n, n)), dtype=object)
    partner = np.ones((n, n), dtype=other)
    with f.BudgetContext(flop_budget=10**18, quiet=True) as budget:
        with pytest.raises(UnsupportedDtypeError):
            fnp.multiply(o, partner)
        assert budget.flops_used == 0


def test_nonnumeric_operand_is_refused():
    """The 1.0 neutral rate used to be "correct" when the operands themselves
    were non-numeric -- object was the sharpest exemplar ("that arithmetic
    really is interpreted Python"), and timedelta64 stood in for it here
    (numpy's native timedelta add loop is a non-numeric result kind that
    billed at the neutral rate 1.0, cheaper than float64's real-weights rate
    of 2.0 -- exactly the itemsize-blind under-bill the ban now closes for
    every non-numeric operand, not merely for object). Both are refused
    outright now; there is no remaining non-numeric operand dtype left that
    reaches a rate at all."""
    load_weights()
    o = np.ones((2, 2), dtype="m8[ns]")
    with f.BudgetContext(flop_budget=10**18, quiet=True) as budget:
        with pytest.raises(UnsupportedDtypeError):
            fnp.add(o, o)
        assert budget.flops_used == 0


# --- the gate must not be reachable around, via a container ---------------


@pytest.mark.parametrize("wrap", [lambda d: d, lambda d: (d,)], ids=["bare", "tuple"])
def test_a_nonnumeric_out_is_refused_through_a_container(wrap):
    """The whole gate is ``isinstance(out, np.ndarray)``, so a container used
    to slip past it -- billing as though no destination were supplied at all.
    Normalizing ``out`` before the gate is what keeps the tuple honest, and
    that normalization has to run before the refusal check too, or a
    1-tuple-wrapped non-numeric destination would dodge the ban the same way
    it used to dodge the rate resolution."""
    load_weights()
    a, b = _complex_pair()
    out = np.empty(N, dtype="U64")
    with f.BudgetContext(flop_budget=10**18, quiet=True) as budget:
        with pytest.raises(UnsupportedDtypeError):
            fnp.vecdot(a, b, out=wrap(out))
        assert budget.flops_used == 0


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
