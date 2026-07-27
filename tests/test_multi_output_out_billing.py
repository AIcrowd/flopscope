"""A multi-output op's NATURAL destinations must be price-neutral.

The ``out=`` doctrine prices the widest participating buffer -- a destination
wider than the compute loop is a real materialization and bills at the wider
rate (see ``tests/test_out_billing_doctrine.py``). A multi-output signature
adds one correction to that: an output whose dtype belongs to the op
SIGNATURE rather than to its arithmetic is not a wider buffer at all, and
naming it must cost exactly what letting numpy allocate it costs.

``frexp`` is the case: its second output is always ``int32``, whatever
precision the mantissa runs at, so folding a supplied exponent buffer into
``np.result_type`` promoted ``(float32, float32, int32)`` to float64 and
``frexp(a32, out=(m32, e32))`` billed 20,000 against the bare call's 10,000 --
the caller paying the float64 rate for arithmetic numpy runs at float32,
purely for naming the destinations numpy would have created itself.

Every pin below is measured in BOTH directions: the natural destination is
free, and a genuinely wider one still widens.
"""

from typing import Any

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights

N = 10_000


def _out(*arrays: object) -> Any:
    """Destination tuple, deliberately loosely typed.

    The wrappers annotate ``out=`` as a tuple of ``FlopscopeArray`` but accept
    plain ndarrays exactly as numpy does, and plain ndarrays are what the
    billing has to be measured on.
    """
    return tuple(arrays)


def _billed(fn):
    with f.BudgetContext(flop_budget=10**15, quiet=True) as b:
        fn()
        return b.flops_used


@pytest.fixture(autouse=True)
def _weights():
    load_weights()


def _f32():
    return np.random.default_rng(0).standard_normal(N).astype(np.float32)


# ---------------------------------------------------------------------------
# frexp -- the op the correction exists for
# ---------------------------------------------------------------------------


def test_frexp_natural_destinations_are_price_neutral():
    a = _f32()
    bare = _billed(lambda: fnp.frexp(a))
    natural = _billed(
        lambda: fnp.frexp(a, out=_out(np.empty(N, np.float32), np.empty(N, np.int32)))
    )
    assert bare == natural == N  # was 10,000 vs 20,000


def test_frexp_natural_destinations_price_neutral_at_float64():
    a = np.random.default_rng(1).standard_normal(N)
    bare = _billed(lambda: fnp.frexp(a))
    natural = _billed(
        lambda: fnp.frexp(a, out=_out(np.empty(N, np.float64), np.empty(N, np.int32)))
    )
    assert bare == natural == 2 * N


def test_frexp_wider_mantissa_still_widens_the_rate():
    a = _f32()
    wide = _billed(
        lambda: fnp.frexp(a, out=_out(np.empty(N, np.float64), np.empty(N, np.int32)))
    )
    assert wide == 2 * N  # float64 rate, not the float32 loop's


def test_frexp_wider_exponent_still_widens_the_rate():
    """int64 is wider than the int32 numpy would have allocated -- it pays."""
    a = _f32()
    wide = _billed(
        lambda: fnp.frexp(a, out=_out(np.empty(N, np.float32), np.empty(N, np.int64)))
    )
    assert wide == 2 * N


def test_frexp_narrower_exponent_never_discounts():
    """A narrow destination is a store, not a loop: it can only cast down."""
    a = _f32()
    narrow = _billed(
        lambda: fnp.frexp(a, out=_out(np.empty(N, np.float32), np.empty(N, np.int16)))
    )
    assert narrow == _billed(lambda: fnp.frexp(a)) == N


def test_frexp_partial_out_slots_price_like_the_slots_given():
    a = _f32()
    assert _billed(lambda: fnp.frexp(a, out=_out(None, np.empty(N, np.int32)))) == N
    assert (
        _billed(lambda: fnp.frexp(a, out=_out(np.empty(N, np.float64), None))) == 2 * N
    )


def test_frexp_int32_exponent_floor_survives_a_narrow_mantissa():
    """The fixed-width exponent is priced by the floor, not by the buffer.

    An int8 input maps to a float16 mantissa loop; the exponent is still
    int32, so the bill must not fall to the float16 rate whether or not the
    caller supplies the buffers.
    """
    a = np.random.default_rng(2).integers(-100, 100, N).astype(np.int8)
    bare = _billed(lambda: fnp.frexp(a))
    natural = _billed(
        lambda: fnp.frexp(a, out=_out(np.empty(N, np.float16), np.empty(N, np.int32)))
    )
    assert bare == natural == N


# ---------------------------------------------------------------------------
# siblings -- modf (both outputs are the loop dtype) and divmod
# ---------------------------------------------------------------------------


def test_modf_natural_destinations_are_price_neutral():
    a = _f32()
    bare = _billed(lambda: fnp.modf(a))
    natural = _billed(
        lambda: fnp.modf(a, out=_out(np.empty(N, np.float32), np.empty(N, np.float32)))
    )
    assert bare == natural == N


def test_modf_wider_destination_still_widens_the_rate():
    a = _f32()
    wide = _billed(
        lambda: fnp.modf(a, out=_out(np.empty(N, np.float32), np.empty(N, np.float64)))
    )
    assert wide == 2 * N


def test_divmod_natural_destinations_are_price_neutral():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(N).astype(np.float32)
    y = (rng.standard_normal(N) + 5.0).astype(np.float32)
    bare = _billed(lambda: fnp.divmod(x, y))
    natural = _billed(
        lambda: fnp.divmod(
            x, y, out=_out(np.empty(N, np.float32), np.empty(N, np.float32))
        )
    )
    assert bare == natural


def test_divmod_wider_destination_still_widens_the_rate():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(N).astype(np.float32)
    y = (rng.standard_normal(N) + 5.0).astype(np.float32)
    bare = _billed(lambda: fnp.divmod(x, y))
    wide = _billed(
        lambda: fnp.divmod(
            x, y, out=_out(np.empty(N, np.float64), np.empty(N, np.float64))
        )
    )
    assert wide == 2 * bare


def test_weak_scalar_operand_does_not_inflate_the_natural_destination():
    """NEP 50: ``divmod(f32, 2.0)`` computes float32, so float32 is natural.

    Resolving the natural output from the RAW operand dtypes instead of the
    promoted ones would have reported float64 as natural here and handed the
    float64 destination away for free.
    """
    x = (np.random.default_rng(5).standard_normal(N).astype(np.float32) + 5.0).astype(
        np.float32
    )
    bare = _billed(lambda: fnp.divmod(x, 2.0))
    natural = _billed(
        lambda: fnp.divmod(
            x, 2.0, out=_out(np.empty(N, np.float32), np.empty(N, np.float32))
        )
    )
    wide = _billed(
        lambda: fnp.divmod(
            x, 2.0, out=_out(np.empty(N, np.float64), np.empty(N, np.float64))
        )
    )
    assert bare == natural
    assert wide == 2 * bare


# ---------------------------------------------------------------------------
# The sweep. Derived from the module surface, not hand-listed -- a hand-written
# list is how a defect at one of three sites goes unnoticed.
# ---------------------------------------------------------------------------


def _multi_output_ops():
    """Every wrapped op whose numpy counterpart produces more than one array."""
    found = []
    for name in dir(fnp):
        if name.startswith("_"):
            continue
        np_func = getattr(np, name, None)
        if isinstance(np_func, np.ufunc) and np_func.nout > 1:
            found.append(name)
    return sorted(found)


def test_multi_output_surface_is_the_expected_three():
    assert _multi_output_ops() == ["divmod", "frexp", "modf"]


@pytest.mark.parametrize("op_name", _multi_output_ops())
def test_every_multi_output_op_prices_its_natural_destinations_at_zero_markup(op_name):
    """Supplying the buffers numpy would have allocated must change nothing."""
    rng = np.random.default_rng(6)
    fs_func = getattr(fnp, op_name)
    np_func = getattr(np, op_name)
    x = rng.standard_normal(N).astype(np.float32)
    args = (
        (x,)
        if np_func.nin == 1
        else (x, (rng.standard_normal(N) + 5.0).astype(np.float32))
    )

    bare_result = np_func(*args)
    natural = tuple(np.empty(r.shape, r.dtype) for r in bare_result)

    bare = _billed(lambda: fs_func(*args))
    with_out = _billed(lambda: fs_func(*args, out=_out(*natural)))
    assert with_out == bare, f"{op_name}: natural out= billed {with_out} vs bare {bare}"


@pytest.mark.parametrize("op_name", _multi_output_ops())
def test_every_multi_output_op_still_charges_a_genuinely_wider_destination(op_name):
    """The neutrality above must not have opened a discount on real widening."""
    rng = np.random.default_rng(7)
    fs_func = getattr(fnp, op_name)
    np_func = getattr(np, op_name)
    x = rng.standard_normal(N).astype(np.float32)
    args = (
        (x,)
        if np_func.nin == 1
        else (x, (rng.standard_normal(N) + 5.0).astype(np.float32))
    )

    bare_result = np_func(*args)
    # Widen every slot by rate while keeping numpy's kind, so the call is one
    # numpy still accepts: float32 -> float64, int32 -> int64.
    wider = tuple(np.empty(r.shape, np.dtype(f"{r.dtype.kind}8")) for r in bare_result)

    bare = _billed(lambda: fs_func(*args))
    with_out = _billed(lambda: fs_func(*args, out=_out(*wider)))
    assert with_out == 2 * bare, (
        f"{op_name}: wide out= billed {with_out} vs bare {bare}"
    )
