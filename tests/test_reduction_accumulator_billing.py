"""A narrower ``out=`` must never discount a reduction.

``out=`` does not select numpy's accumulator -- numpy's own ``ufunc.reduce``
docs are stale on this point (they say ``dtype`` "defaults to the data-type of
the output array if this is provided"). Two bit-level probes in
``test_numpy_out_does_not_select_the_accumulator`` show numpy 2.2.6 keeps the
family default and only casts the final store, so a narrower ``out=`` must not
move the bill. A genuinely wider ``out=`` does widen the loop and still bills
wider; an explicit ``dtype=`` IS the accumulator and still bills as requested
in both directions.

Figures are production-rate billed FLOPs (rate 2.0 for 64-bit dtypes, 1.0
below; complex factors 2.0 for the add family, 6.0 for the multiply family).
Inputs and ``out=`` destinations are built OUTSIDE every measured region --
building an array costs FLOPs -- and use asymmetric data so no symmetry tag is
inferred.
"""

import warnings

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._dtype_billing import reduction_billing_dtype
from flopscope._weights import load_weights

N = 32
# reduction over axis 0 of an (N, N) array touches numel - M elements
REDUCE_COST = N * N - N  # 992
MEAN_COST = REDUCE_COST + N  # 1024: one divide per output orbit


def _billed(fn) -> int:
    """Delta-billed FLOPs for one call, at production rates."""
    load_weights()
    with flops.BudgetContext(flop_budget=10**15, quiet=True) as b:
        before = b.flops_used
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn()
        return b.flops_used - before


def _arr(dtype):
    """Asymmetric input array -- a square constant fill would pick up a
    symmetry tag and change the cost."""
    rng = np.random.default_rng(0)
    return fnp.asarray(rng.integers(1, 9, size=(N, N)).astype(dtype))


# --------------------------------------------------------------------------
# The numpy behaviour this whole module rests on.
# --------------------------------------------------------------------------


def test_numpy_out_does_not_select_the_accumulator():
    """Bit-level proof that ``out=`` only casts the store."""
    # float64 input: dtype=float32 truly accumulates in float32 and loses the
    # tail; out=float32 is bit-identical to casting the float64 accumulation.
    a = np.array([1.0, 5e-8, 5e-8, 5e-8], dtype=np.float64)
    full = np.add.reduce(a)
    assert np.float32(np.add.reduce(a, dtype=np.float32)).view(np.uint32) == 1065353216
    dst = np.zeros((), dtype=np.float32)
    assert np.float32(np.add.reduce(a, out=dst)).view(np.uint32) == 1065353217
    assert np.float32(np.add.reduce(a, out=dst)).view(np.uint32) == np.float32(
        full
    ).view(np.uint32)

    # int32 input: the int64 accumulation survives a float32 out=.
    b = np.array([2**24 + 1, 1, 1, 1], dtype=np.int32)
    assert np.add.reduce(b) == 16777220
    assert np.add.reduce(b, dtype=np.float32) == 16777216.0
    assert np.add.reduce(b, out=np.zeros((), dtype=np.float32)) == 16777220.0

    # A WIDER out= does genuinely widen the loop.
    c = np.array([1.0, 5e-8, 5e-8, 5e-8], dtype=np.float32)
    assert np.add.reduce(c) == np.float32(1.0)
    widened = np.add.reduce(c, out=np.zeros((), dtype=np.float64))
    assert widened == np.add.reduce(c, dtype=np.float64)
    assert widened != np.float64(np.add.reduce(c))

    # A real out= cannot make a complex reduction real: the result is the real
    # PART of the complex product, not the product of the real parts.
    d = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
    assert np.prod(d, out=np.zeros((), dtype=np.float64)) == -85.0
    assert np.prod(d.real) == 15.0


# --------------------------------------------------------------------------
# A narrower out= does not discount the bill.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["sum", "prod", "nansum", "nanprod"])
def test_narrower_out_does_not_discount_integer_reduction(name):
    """int32 sum/prod accumulate in int64; an int32 out= only casts the store."""
    a = _arr(np.int32)
    dst = fnp.zeros(N, dtype=np.int32)
    fn = getattr(fnp, name)
    bare = _billed(lambda: fn(a, axis=0))
    assert bare == REDUCE_COST * 2  # int64 accumulator -> rate 2.0
    assert _billed(lambda: fn(a, axis=0, out=dst)) == bare


@pytest.mark.parametrize("name", ["cumsum", "cumprod", "nancumsum", "nancumprod"])
def test_narrower_out_does_not_discount_integer_scan(name):
    a = _arr(np.int32)
    dst = fnp.zeros((N, N), dtype=np.int32)
    fn = getattr(fnp, name)
    bare = _billed(lambda: fn(a, axis=0))
    assert bare == REDUCE_COST * 2
    assert _billed(lambda: fn(a, axis=0, out=dst)) == bare


@pytest.mark.parametrize("name", ["mean", "nanmean"])
def test_narrower_out_does_not_discount_integer_mean(name):
    """An integer mean computes in float64; a float32 out= only casts."""
    a = _arr(np.int32)
    dst = fnp.zeros(N, dtype=np.float32)
    fn = getattr(fnp, name)
    bare = _billed(lambda: fn(a, axis=0))
    assert bare == MEAN_COST * 2  # float64 compute -> rate 2.0
    assert _billed(lambda: fn(a, axis=0, out=dst)) == bare


@pytest.mark.parametrize("name", ["var", "std"])
def test_narrower_out_does_not_discount_integer_variance(name):
    a = _arr(np.int32)
    dst = fnp.zeros(N, dtype=np.float32)
    fn = getattr(fnp, name)
    bare = _billed(lambda: fn(a, axis=0))
    assert _billed(lambda: fn(a, axis=0, out=dst)) == bare


def test_narrower_out_does_not_discount_float_power_reduce():
    """float_power always runs a float64 loop, even on float32 input, so a
    float32 out= must not halve it."""
    a = _arr(np.float32)
    dst = fnp.zeros(N, dtype=np.float32)
    bare = _billed(lambda: np.float_power.reduce(a, axis=0))
    assert bare == REDUCE_COST * 2  # float64 loop -> rate 2.0
    assert _billed(lambda: np.float_power.reduce(a, axis=0, out=dst)) == bare


def test_narrower_out_does_not_discount_trace():
    """trace widens its accumulator exactly like sum."""
    a = _arr(np.int32)
    dst = fnp.zeros((), dtype=np.int32)
    bare = _billed(lambda: fnp.trace(a))
    assert _billed(lambda: fnp.trace(a, out=dst)) == bare


# --------------------------------------------------------------------------
# A real out= cannot strip the complex factor off a complex accumulator.
# --------------------------------------------------------------------------


def test_real_out_does_not_strip_complex_factor_from_prod():
    """complex64 prod bills rate 1.0 x factor 6.0; a float64 out= would rank
    higher on rate alone (2.0) while dropping the factor to 1.0 -- a 3x
    discount if out= were allowed to replace the accumulator."""
    a = _arr(np.complex64)
    dst = fnp.zeros(N, dtype=np.float64)
    bare = _billed(lambda: fnp.prod(a, axis=0))
    assert bare == REDUCE_COST * 6  # rate 1.0 * complex factor 6.0
    # A real store cannot strip the complex factor -- that is the point of
    # this test -- but it does contribute its own rate, so the widened form
    # bills ABOVE the bare one rather than equal to it. What must never happen
    # is falling below `bare`, which is what stripping the factor would do.
    widened = _billed(lambda: fnp.prod(a, axis=0, out=dst))
    assert widened >= bare
    assert widened == bare * 2  # float64 store rate 2.0 over complex64's 1.0


def test_real_out_does_not_strip_complex_factor_from_bool_loop_reduce():
    """logical_xor resolves a BOOL loop dtype, so the complex input carries the
    factor via the operand-width floor rather than the family default."""
    a = _arr(np.complex128)
    dst = fnp.zeros(N, dtype=np.int64)
    bare = _billed(lambda: np.logical_xor.reduce(a, axis=0))
    assert bare == REDUCE_COST * 2 * 2  # rate 2.0 * complex factor 2.0
    assert _billed(lambda: np.logical_xor.reduce(a, axis=0, out=dst)) == bare


# --------------------------------------------------------------------------
# A wider out= must still widen.
# --------------------------------------------------------------------------


def test_wider_out_still_widens_the_bill():
    a = _arr(np.float32)
    dst = fnp.zeros(N, dtype=np.float64)
    bare = _billed(lambda: fnp.sum(a, axis=0))
    assert bare == REDUCE_COST  # float32 accumulator -> rate 1.0
    assert _billed(lambda: fnp.sum(a, axis=0, out=dst)) == REDUCE_COST * 2


def test_wider_out_still_widens_generic_ufunc_reduce():
    a = _arr(np.float32)
    dst = fnp.zeros(N, dtype=np.float64)
    bare = _billed(lambda: np.add.reduce(a, axis=0))
    assert bare == REDUCE_COST
    assert _billed(lambda: np.add.reduce(a, axis=0, out=dst)) == REDUCE_COST * 2


# --------------------------------------------------------------------------
# An explicit dtype= IS the accumulator: still billed as requested, both ways.
# --------------------------------------------------------------------------


def test_explicit_dtype_still_bills_narrower_as_requested():
    a = _arr(np.float64)
    bare = _billed(lambda: fnp.sum(a, axis=0))
    assert bare == REDUCE_COST * 2
    assert _billed(lambda: fnp.sum(a, axis=0, dtype=np.float32)) == REDUCE_COST


def test_explicit_dtype_still_bills_wider_as_requested():
    a = _arr(np.float32)
    assert _billed(lambda: fnp.sum(a, axis=0)) == REDUCE_COST
    assert _billed(lambda: fnp.sum(a, axis=0, dtype=np.float64)) == REDUCE_COST * 2


def test_explicit_dtype_beats_out_in_both_directions():
    """dtype= wins over out= -- it really is the accumulator numpy runs."""
    a = _arr(np.int32)
    wide = fnp.zeros(N, dtype=np.float64)
    assert _billed(lambda: fnp.sum(a, axis=0, dtype=np.int32, out=wide)) == REDUCE_COST


# --------------------------------------------------------------------------
# The correct siblings must not move: numpy's accumulator IS the input dtype.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["amin", "amax", "min", "max"])
def test_extremum_siblings_are_unmoved_by_out(name):
    a = _arr(np.int32)
    dst = fnp.zeros(N, dtype=np.int32)
    fn = getattr(fnp, name)
    bare = _billed(lambda: fn(a, axis=0))
    assert bare == REDUCE_COST  # int32 accumulator -> rate 1.0, no widening
    assert _billed(lambda: fn(a, axis=0, out=dst)) == bare


def test_power_reduce_sibling_is_unmoved_by_out():
    """power (unlike float_power) keeps the float32 loop, so an f32 out= is
    neither a widening nor a narrowing."""
    a = _arr(np.float32)
    dst = fnp.zeros(N, dtype=np.float32)
    bare = _billed(lambda: np.power.reduce(a, axis=0))
    assert bare == REDUCE_COST
    assert _billed(lambda: np.power.reduce(a, axis=0, out=dst)) == bare


# --------------------------------------------------------------------------
# Unit-level resolution.
# --------------------------------------------------------------------------


def test_out_combines_with_the_family_default_instead_of_replacing_it():
    load_weights()
    i32, i64 = np.dtype(np.int32), np.dtype(np.int64)
    f32, f64 = np.dtype(np.float32), np.dtype(np.float64)
    c64 = np.dtype(np.complex64)
    c128 = np.dtype(np.complex128)

    # A narrower out= cannot pull the bill below the family default...
    assert reduction_billing_dtype(i32, out_dtype=i32, default_dtype=i64) == i64
    assert reduction_billing_dtype(i32, out_dtype=f32, default_dtype=f64) == f64
    # ...nor below the operand width when there is no family default.
    assert reduction_billing_dtype(f64, out_dtype=f32) == f64
    # ...and a real out= cannot make a complex accumulator real.
    # complex128, not complex64: restoring the complex kind must not cost the
    # rate the float64 store already earned. This function ranks by rate and
    # cannot see the op's complex factor, so it cannot tell which participant
    # is genuinely pricier; promoting keeps BOTH properties and errs toward
    # over-billing, which is the safe direction for a meter.
    assert reduction_billing_dtype(c64, out_dtype=f64, default_dtype=c64) == c128
    # Also complex128 rather than complex64, and for the same reason: the
    # int64 accumulator's rate 2.0 has been earned and restoring the complex
    # kind must not spend it.
    assert (
        reduction_billing_dtype(c64, out_dtype=i64, default_dtype=np.dtype(np.bool_))
        == c128
    )
    # A wider out= still widens.
    assert reduction_billing_dtype(f32, out_dtype=f64, default_dtype=f32) == f64
    assert reduction_billing_dtype(f32, out_dtype=f64) == f64
    # An explicit dtype= is the accumulator, in both directions, and beats out=.
    assert reduction_billing_dtype(f32, explicit_dtype=np.float64) == f64
    assert reduction_billing_dtype(f64, explicit_dtype=np.float32) == f32
    assert (
        reduction_billing_dtype(i32, explicit_dtype=np.int32, default_dtype=i64) == i32
    )
    assert reduction_billing_dtype(i32, explicit_dtype=np.int32, out_dtype=f64) == i32


def test_reduction_billing_dtype_never_bills_below_the_bare_form():
    """Property: adding an out= may only ever raise the resolved rate."""
    load_weights()
    from flopscope._dtype_billing import rate_for

    dtypes = [
        np.dtype(d)
        for d in (
            np.bool_,
            np.int8,
            np.int32,
            np.int64,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        )
    ]
    for a_dtype in dtypes:
        for default in dtypes:
            bare = reduction_billing_dtype(a_dtype, default_dtype=default)
            for out in dtypes:
                got = reduction_billing_dtype(
                    a_dtype, out_dtype=out, default_dtype=default
                )
                assert rate_for(got) >= rate_for(bare), (a_dtype, default, out)
                # complex structure is never traded away for a wider rate
                if bare.kind == "c":
                    assert got.kind == "c", (a_dtype, default, out)


# ---------------------------------------------------------------------------
# The kind axis: a complex destination must not lose its factor to a tie-break
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,expect_ratio",
    [
        (lambda a, o: fnp.prod(a, axis=0, out=o), 6.0),
        (lambda a, o: fnp.sum(a, axis=0, out=o), 2.0),
        (lambda a, o: fnp.mean(a, axis=0, out=o), 2.0),
        (lambda a, o: fnp.amax(a, axis=0, out=o), 2.0),
    ],
)
def test_a_complex_destination_never_lowers_the_bill(op, expect_ratio):
    """Complex-ness belongs to any participating buffer, including ``out=``.

    The rate axis cannot see it -- complex64 and float32 both rate 1.0 -- so
    whichever dtype wins the tie decides whether the complex factor survives.
    An earlier form of this guard consulted only the accumulator and the
    input, so a complex64 destination on a real accumulator silently dropped
    the factor: prod fell from 391,680 to 65,280, a 6x under-bill of exactly
    the kind this module exists to prevent, pointing the other way.
    """
    rng = np.random.default_rng(11)
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        a = fnp.asarray(rng.standard_normal((256, 256)).astype("float32"))
        real_dest = np.zeros(256, dtype="float32")
        complex_dest = np.zeros(256, dtype="complex64")

        before = ctx.flops_used
        op(a, real_dest)
        real_cost = ctx.flops_used - before

        before = ctx.flops_used
        op(a, complex_dest)
        complex_cost = ctx.flops_used - before

    assert complex_cost >= real_cost, (
        f"a complex destination billed {complex_cost} against {real_cost} for a "
        f"real one -- the complex factor was dropped"
    )
    assert complex_cost == real_cost * expect_ratio
