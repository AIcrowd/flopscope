"""flopscope accepts ``out=(dest,)`` more widely than NumPy, on purpose.

NumPy's own boundary here is not a design. The tuple is its canonical *vector
of destinations, one per output* -- the 1.10 release note calls the tuple the
general form and the bare array the sugar, and internally a bare array is
*wrapped* into a tuple before dispatch. Which callables accept it is then an
artifact of which C parser consumes the argument first: the ufunc parser takes
it, ``PyArray_OutputConverter`` ("output must be an array") does not, and
``einsum`` and ``dot`` each have their own inline check with their own wording.
Two more refusals are not validation at all but crashes inside NumPy's Python
wrappers (``nanmean`` raises ``AttributeError: 'tuple' object has no attribute
'dtype'``).

Worse for anyone hoping to mirror it, for three ops acceptance is not a
property of the op at all -- it flips on a keyword *value*:

    np.percentile(v, 50, out=(d,))                      accepted
    np.percentile(v, 50, out=(d,), method='nearest')    TypeError
    np.cumulative_sum(v, out=(d,), include_initial=True) TypeError

So "match NumPy exactly" is unreachable by any mechanism, and every candidate
detection predicate was measured wrong: ``isinstance(f, np.ufunc)`` misses 32
of 175 ops, unwrapping ``_implementation``/``__wrapped__`` buys literally
nothing (no NumPy dispatcher in the set has a ufunc underneath), and an
import-time probe is unsound because acceptance depends on arguments it cannot
know.

flopscope therefore applies ONE rule everywhere: a length-``nout`` tuple of
ndarray-or-``None`` is unwrapped, anything else is refused. That is a deliberate
superset. It is also the safe direction -- accepting more than NumPy costs a
caller nothing at run time, while accepting *less* is a hard failure for code
that works in plain NumPy.

This module is the ledger for that decision, and its real job is the second
test: flopscope must never become a SUBSET. If NumPy widens, or a flopscope
change narrows, an op that NumPy accepts and flopscope refuses is a regression
that this catches on the version matrix CI already runs.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as fl
import flopscope.numpy as fnp

#: Ops where NumPy refuses the 1-tuple and flopscope accepts it. Recorded so
#: the divergence is a decision on the record rather than an accident nobody
#: measured. Not exhaustive of the surface -- exhaustive of what is driven
#: below, which is what a test can honestly claim.
_KNOWN_SUPERSET = (
    "sum",
    "prod",
    "mean",
    "cumsum",
    "trace",
    "einsum",
)


def _driver(name):
    """A (flopscope_call, numpy_call) pair taking the destination form."""
    if name == "einsum":
        return (
            lambda a, out: fnp.einsum("ij,jk->ik", a, a, out=out),
            lambda a, out: np.einsum("ij,jk->ik", a, a, out=out),
        )
    if name == "trace":
        return (
            lambda a, out: fnp.trace(a, out=out),
            lambda a, out: np.trace(a, out=out),
        )
    if name == "cumsum":
        return (
            lambda a, out: fnp.cumsum(a, out=out),
            lambda a, out: np.cumsum(a, out=out),
        )
    return (
        lambda a, out: getattr(fnp, name)(a, axis=0, out=out),
        lambda a, out: getattr(np, name)(a, axis=0, out=out),
    )


def _destination(name, a):
    """A correctly shaped, correctly typed destination for *name*."""
    _, np_call = _driver(name)
    reference = np_call(np.asarray(a), None)
    return np.zeros(np.shape(reference), dtype=np.asarray(reference).dtype)


def _accepts(call, a, out):
    try:
        call(a, out)
    except Exception:
        return False
    return True


@pytest.mark.parametrize("name", _KNOWN_SUPERSET)
def test_the_recorded_superset_is_still_exactly_that(name):
    """NumPy refuses the 1-tuple here and flopscope accepts it, deliberately.

    If this starts failing because NumPy began accepting the tuple, the entry
    is simply stale and should be dropped -- the divergence closed on its own.
    If it fails because flopscope began refusing, that is the regression the
    next test exists to catch, and it should be fixed rather than recorded.
    """
    # Square: the einsum driver contracts ij,jk->ik, which needs it.
    raw = np.arange(16.0).reshape(4, 4)
    fnp_call, np_call = _driver(name)
    dest_np = _destination(name, raw)

    assert not _accepts(np_call, raw, (dest_np,)), (
        f"numpy now accepts a 1-tuple out= for {name}; this ledger entry is "
        f"stale and should be removed rather than kept passing"
    )

    with fl.BudgetContext(flop_budget=10**12, quiet=True):
        a = fnp.asarray(raw)
        dest = _destination(name, raw)
        assert _accepts(fnp_call, a, (dest,)), (
            f"flopscope refuses a 1-tuple out= for {name} -- it has become "
            f"STRICTER than this ledger records"
        )


@pytest.mark.parametrize("name", _KNOWN_SUPERSET)
def test_flopscope_never_becomes_a_subset_of_numpy(name):
    """The guard that actually matters.

    Accepting more than NumPy costs a caller nothing at run time. Accepting
    LESS breaks code that works in plain NumPy, which is why the superset was
    chosen in the first place -- so the invariant worth enforcing on every
    numpy in the CI matrix is one-directional: whatever NumPy accepts,
    flopscope accepts too.
    """
    # Square: the einsum driver contracts ij,jk->ik, which needs it.
    raw = np.arange(16.0).reshape(4, 4)
    fnp_call, np_call = _driver(name)

    for label, make_out in (
        ("bare array", lambda d: d),
        ("one-tuple", lambda d: (d,)),
        ("None", lambda d: None),
    ):
        numpy_took_it = _accepts(
            np_call, np.asarray(raw), make_out(_destination(name, raw))
        )
        if not numpy_took_it:
            continue
        with fl.BudgetContext(flop_budget=10**12, quiet=True):
            a = fnp.asarray(raw)
            dest = _destination(name, raw)
            assert _accepts(fnp_call, a, make_out(dest)), (
                f"numpy accepts {label} out= for {name} and flopscope refuses "
                f"it -- flopscope has become a SUBSET of numpy, which breaks "
                f"code that works in plain numpy"
            )


#: Ops where NumPy ITSELF accepts the 1-tuple, because they reach the ufunc
#: argument parser. These are where the subset guard has teeth: the tests above
#: skip the tuple form for the recorded-superset ops precisely because NumPy
#: refuses it there, so without these the guard would never exercise the case
#: it exists for.
_NUMPY_ACCEPTS_THE_TUPLE = ("multiply", "add", "sqrt", "negative", "matmul")


def _ufunc_driver(name):
    if name in ("sqrt", "negative"):
        return (
            lambda a, out: getattr(fnp, name)(a, out=out),
            lambda a, out: getattr(np, name)(a, out=out),
        )
    return (
        lambda a, out: getattr(fnp, name)(a, a, out=out),
        lambda a, out: getattr(np, name)(a, a, out=out),
    )


@pytest.mark.parametrize("name", _NUMPY_ACCEPTS_THE_TUPLE)
def test_the_forms_numpy_accepts_are_accepted_here_too(name):
    """One-directional parity, on the ops where NumPy takes the tuple.

    This is the regression that would actually hurt: code written against
    plain NumPy, submitted against flopscope, refused. Narrowing
    ``_normalize_out`` -- or losing the guard from a wrapper -- shows up here.
    """
    raw = np.arange(16.0).reshape(4, 4) + 1.0
    fnp_call, np_call = _ufunc_driver(name)
    reference = np_call(raw, None)

    def fresh():
        return np.zeros(reference.shape, dtype=reference.dtype)

    for label, make_out in (
        ("bare array", lambda d: d),
        ("one-tuple", lambda d: (d,)),
        ("None", lambda d: None),
    ):
        assert _accepts(np_call, raw, make_out(fresh())), (
            f"test bug: numpy refuses {label} out= for {name}"
        )
        with fl.BudgetContext(flop_budget=10**12, quiet=True):
            a = fnp.asarray(raw)
            assert _accepts(fnp_call, a, make_out(fresh())), (
                f"numpy accepts {label} out= for {name} and flopscope refuses "
                f"it -- flopscope has become a SUBSET of numpy"
            )
