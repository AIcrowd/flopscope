"""``out=`` on the fft family: the destination must be priced, and written.

Two defects lived in every one of the fourteen ``flopscope.numpy.fft``
transform wrappers, and each is pinned here.

**The destination never reached the rate.** Every wrapper deducted with
``dtypes=(fft_billing_dtype(a.dtype),)`` -- a function of the INPUT dtype
alone -- so ``store_billing_dtypes(out)`` never joined the resolution. A
complex64 transform written into a complex128 destination therefore bought
genuine double-precision work at the single-precision rate: exactly half
price, for a correctly computed and correctly written result. The pointwise
and contraction families already fold the destination in (see
``src/flopscope/_dtype_billing.py`` -- the widest participating buffer is
``max(compute width, store width)``); the fft family now does the same.

**A destination numpy never wrote was returned anyway.** ``np.fft.hfft``,
``np.fft.ifft2`` and ``np.fft.irfft2`` hardcode ``out=None`` into their inner
call on numpy 2.0 through 2.4 -- fixed upstream only in 2.5.1, outside this
package's ``<2.5.0`` pin. numpy silently ignored the destination, allocated
a fresh array and returned that; the wrapper handed back ``out``, so the
caller received an untouched buffer AS the transform result, at full price.
A plausible-looking array of the wrong values, billed in full, is the worst
failure class a metering system has.

Why the coverage here is DERIVED rather than hand-listed
--------------------------------------------------------
A hand-written list of fourteen names pins whichever wrappers the author
happened to think of, and says nothing about a fifteenth added later --
which is precisely how a defect present in all fourteen sites survived. So
the ops under test are discovered from the module surface (public, mirrored
by ``numpy.fft``, and taking an ``out`` parameter), and their arguments are
derived from the transform's own name: the real/complex/hermitian input kind
from the base name, the dimensionality from the ``2``/``n`` suffix, and the
destination's natural shape and dtype from plain numpy itself. A transform
that lands tomorrow is enumerated the day it lands, and one whose name this
module cannot classify fails ``test_every_discovered_op_is_driven`` rather
than skipping in silence.

Inputs are deliberately single-precision (float32 / complex64), so that a
complex128 destination is genuinely WIDER than the transform's own working
precision and the rate has something to widen to.
"""

import inspect

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights

# Base transforms by what they consume and produce. The suffix (``2``/``n``)
# only picks the dimensionality, so classification is by base name.
_C2C = frozenset({"fft", "ifft"})  # complex in, complex out
_R2C = frozenset({"rfft", "ihfft"})  # real in, complex out
_C2R = frozenset({"irfft", "hfft"})  # hermitian-complex in, real out

SENTINEL = 1234.5


def _discover_ops() -> dict:
    """Every public fft wrapper that accepts a destination."""
    ops = {}
    for name in dir(fnp.fft):
        if name.startswith("_"):
            continue
        fn = getattr(fnp.fft, name)
        if not callable(fn) or not hasattr(np.fft, name):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if "out" in sig.parameters:
            ops[name] = fn
    return ops


FFT_OPS = _discover_ops()
OP_NAMES = sorted(FFT_OPS)


def _base(name: str) -> str:
    """``fftn`` -> ``fft``, ``irfft2`` -> ``irfft``, ``ihfft`` -> ``ihfft``."""
    return name.rstrip("2n")


def _is_nd(name: str) -> bool:
    return name.endswith(("2", "n"))


def _make_input(name: str) -> np.ndarray:
    """A single-precision input of the kind this transform consumes.

    Asymmetric random data on a non-square shape: zeros/ones on a square
    shape would let an inferred symmetry tag change what the call costs.
    """
    base = _base(name)
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    # A c2r transform reconstructs a length-8 real axis from 5 hermitian
    # bins; everything else takes the full length directly.
    last = 5 if base in _C2R else 8
    shape = (4, last) if _is_nd(name) else (last,)
    if base in _R2C:
        return rng.standard_normal(shape).astype(np.float32)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (real + 1j * imag).astype(np.complex64)


def _expected(name: str, a: np.ndarray) -> np.ndarray:
    """What plain numpy computes -- the reference for every content check."""
    return getattr(np.fft, name)(np.asarray(a))


def _billed(fn) -> int:
    with f.BudgetContext(flop_budget=10**15, quiet=True) as b:
        fn()
        return b.flops_used


def _dest(shape, dtype, *, flopscope_array: bool = False):
    """A destination pre-filled with a sentinel, built OUTSIDE any measured
    region -- constructing an array costs FLOPs of its own."""
    if flopscope_array:
        with f.BudgetContext(flop_budget=10**15, quiet=True):
            return fnp.full(shape, SENTINEL, dtype=dtype)
    return np.full(shape, SENTINEL, dtype=dtype)


@pytest.fixture(autouse=True)
def _weights():
    load_weights()


def test_every_discovered_op_is_driven():
    """No destination-taking transform may go untested by accident."""
    assert OP_NAMES, "discovery found no fft ops taking out="
    unclassified = [n for n in OP_NAMES if _base(n) not in _C2C | _R2C | _C2R]
    assert not unclassified, (
        f"fft ops {unclassified} take out= but this module cannot derive their "
        f"arguments; classify their base name in _C2C / _R2C / _C2R"
    )
    # The count is asserted so that a transform DISAPPEARING from the surface
    # is as loud as one appearing unclassified.
    assert len(OP_NAMES) == 14, OP_NAMES


@pytest.mark.parametrize("name", OP_NAMES)
def test_wider_destination_widens_the_rate(name):
    """A complex128 destination buys double-precision work at double price.

    Before the fix the rate came from the input dtype alone, so all three
    columns billed identically and the wide destination was free.
    """
    fn = FFT_OPS[name]
    a = _make_input(name)
    natural = _expected(name, a)

    bare = _billed(lambda: fn(a))
    same = _billed(lambda: fn(a, out=_dest(natural.shape, natural.dtype)))
    wide = _billed(lambda: fn(a, out=_dest(natural.shape, np.complex128)))

    assert same == bare, f"{name}: a same-dtype destination must not move the rate"
    assert wide == 2 * bare, (
        f"{name}: complex128 destination billed {wide}, expected {2 * bare} "
        f"(complex64 rate 1.0 -> complex128 rate 2.0)"
    )


def test_wider_destination_exact_anchor():
    """One concrete number, so a silent global re-rating cannot pass unseen."""
    a = (np.arange(8) + 1j * np.arange(8, 0, -1)).astype(np.complex64)
    assert _billed(lambda: fnp.fft.fft(a)) == 120  # 5 * 8 * log2(8), rate 1.0
    assert _billed(lambda: fnp.fft.fft(a, out=_dest((8,), np.complex128))) == 240


@pytest.mark.parametrize("name", OP_NAMES)
def test_out_tuple_bills_exactly_what_bare_out_bills(name):
    """``out=(d,)`` is numpy's own spelling of ``out=d`` and costs the same."""
    fn = FFT_OPS[name]
    a = _make_input(name)
    shape = _expected(name, a).shape

    bare = _billed(lambda: fn(a, out=_dest(shape, np.complex128)))
    tupled = _billed(lambda: fn(a, out=(_dest(shape, np.complex128),)))

    assert tupled == bare


@pytest.mark.parametrize("name", OP_NAMES)
def test_refused_out_form_costs_zero(name):
    """A list is not a destination numpy accepts, and refusing it is free."""
    fn = FFT_OPS[name]
    a = _make_input(name)
    shape = _expected(name, a).shape

    with f.BudgetContext(flop_budget=10**15, quiet=True) as b:
        with pytest.raises(TypeError):
            fn(a, out=[_dest(shape, np.complex128)])
        assert b.flops_used == 0


@pytest.mark.parametrize("name", OP_NAMES)
def test_flops_used_never_decreases_across_out_forms(name):
    """No ``out=`` form, accepted or refused, ever hands budget back."""
    fn = FFT_OPS[name]
    a = _make_input(name)
    natural = _expected(name, a)

    with f.BudgetContext(flop_budget=10**15, quiet=True) as b:
        seen = b.flops_used
        for make_out in (
            lambda: None,
            lambda: _dest(natural.shape, natural.dtype),
            lambda: _dest(natural.shape, np.complex128),
            lambda: (_dest(natural.shape, np.complex128),),
        ):
            out = make_out()
            fn(a, out=out) if out is not None else fn(a)
            assert b.flops_used >= seen
            seen = b.flops_used
        with pytest.raises(TypeError):
            fn(a, out=[_dest(natural.shape, np.complex128)])
        assert b.flops_used == seen


@pytest.mark.parametrize("flavor", ["ndarray", "flopscope_array"])
@pytest.mark.parametrize("name", OP_NAMES)
def test_accepted_destination_is_actually_written(name, flavor):
    """THE contract test: a destination that is accepted must be WRITTEN.

    Asserted by CONTENT against plain numpy, never by identity -- identity
    passed happily while ``hfft``, ``ifft2`` and ``irfft2`` returned an
    all-sentinel buffer at full price. Both the destination and the returned
    object are checked, because the wrapper hands back the destination and a
    caller reads the result off either one.
    """
    fn = FFT_OPS[name]
    a = _make_input(name)
    expected = _expected(name, a)
    dest = _dest(
        expected.shape, expected.dtype, flopscope_array=(flavor == "flopscope_array")
    )

    with f.BudgetContext(flop_budget=10**15, quiet=True):
        returned = fn(a, out=dest)

    assert not np.any(np.asarray(dest) == SENTINEL), (
        f"{name}: destination left untouched -- numpy ignored out= and the "
        f"caller was handed the sentinel buffer as the transform result"
    )
    np.testing.assert_allclose(np.asarray(dest), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(returned), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", OP_NAMES)
def test_mis_shaped_destination_is_refused_everywhere(name):
    """Every transform validates a bad destination, not just the ones numpy
    bothered to forward ``out=`` to."""
    fn = FFT_OPS[name]
    a = _make_input(name)
    bad_shape = tuple(d + 1 for d in _expected(name, a).shape)

    with f.BudgetContext(flop_budget=10**15, quiet=True):
        with pytest.raises(ValueError, match="wrong shape"):
            fn(a, out=_dest(bad_shape, np.complex128))
