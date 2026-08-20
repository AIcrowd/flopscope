"""A unary ufunc's compute dtype comes from numpy's loop, not from a name list.

``_counted_binary`` has always asked numpy which loop it resolved and billed
that; ``_counted_unary`` consulted a hand-maintained frozenset instead. That
asymmetry is why every binary float-only op (copysign, heaviside, nextafter,
hypot, logaddexp) has always billed correctly with no entry, while the unary
side undercharged whenever a name was missing -- angle's bool anomaly, then
signbit/isneginf/isposinf. The sets survive only for the composites, which
publish no loop to resolve.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import flopscope
import flopscope as flops
import flopscope.numpy as fnp
from flopscope._dtype_billing import heavier_billing_dtype, rate_for
from flopscope._pointwise import _UNARY_FLOAT64_MIN_OPS, _UNARY_FLOAT_LOOP_OPS
from flopscope._weights import load_weights
from flopscope.errors import UnsupportedDtypeError

N = 256
DTYPES = [
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]


def _billed_dtype(op: str, arr: np.ndarray) -> np.dtype | None:
    load_weights()
    with flops.budget(10**15, quiet=True) as b:
        getattr(fnp, op)(arr)
        recs = [r for r in b.op_log if r.op_name == op]
    return np.dtype(recs[-1].resolved_dtype) if recs else None


def _wired_unary_ops() -> dict[str, str]:
    """op_name -> the numpy name flopscope wraps, read from the wiring lines.

    Anchored to the installed package rather than the working directory, so
    the map does not depend on where pytest was invoked from.
    """
    src = (Path(flopscope.__file__).parent / "_pointwise.py").read_text()
    return {
        m.group(2): m.group(1)
        for m in re.finditer(
            r'^\w+ = _counted_unary(?:_multi)?\(\s*_np\.(\w+),\s*"([\w.]+)"\)',
            src,
            re.M,
        )
    }


WIRING = _wired_unary_ops()
UFUNC_OPS = sorted(
    op
    for op, np_name in WIRING.items()
    if isinstance(getattr(np, np_name, None), np.ufunc)
)


# ---------------------------------------------------------------------------
# The behaviour change: numpy has no bool loop for these four
# ---------------------------------------------------------------------------

NO_BOOL_LOOP = ["square", "conj", "conjugate", "reciprocal"]


@pytest.mark.parametrize("op", NO_BOOL_LOOP)
def test_bool_operand_bills_the_int8_loop_numpy_promotes_to(op):
    """numpy publishes no bool loop here: it computes AND returns int8.

    Billing `bool` named a loop numpy does not have. The rates tie on the
    shipped table so no amount moves, but `bool` and `int8` are separate,
    independently editable entries in `dtype_rates` -- the recorded dtype has
    to name the loop that actually ran.
    """
    np_fn = getattr(np, op)
    assert np_fn.resolve_dtypes((np.dtype(bool), None))[0] == np.dtype(np.int8), (
        "probe stale: numpy now resolves a different loop for a bool operand"
    )
    assert _billed_dtype(op, np.ones(N, dtype=bool)) == np.dtype(np.int8)


# ---------------------------------------------------------------------------
# The structural property: no name list is consulted for a ufunc-backed op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", UFUNC_OPS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_ufunc_unary_bills_numpys_resolved_loop(op, dtype):
    """Billed dtype == heavier(rate-bearing loop slots) floored by the operand.

    Asserted against numpy's own resolution rather than a table, so an op
    flopscope has never heard of is priced correctly the moment it is wired.
    """
    load_weights()
    np_fn = getattr(np, WIRING[op])
    dt = np.dtype(dtype)
    try:
        arr = np.ones(N, dtype=dt)
        billed = _billed_dtype(op, arr)
    except Exception:  # noqa: BLE001 - numpy refuses this operand entirely
        pytest.skip(f"{op} does not accept {dtype}")
    if billed is None:
        pytest.skip(f"{op} logged no record for {dtype}")
    # Ask NUMPY, not flopscope's own resolution helper. An oracle built out of
    # the code under test can only confirm that code agrees with itself -- and
    # calling the helper with its default sentinels is fragile besides, since
    # test_numpy_version_support reloads _pointwise and rebinds them while a
    # function default still holds the old object. No unary ufunc has a
    # control-input slot, so every resolved slot is rate-bearing here.
    loop = np_fn.resolve_dtypes((dt,) + (None,) * np_fn.nout)
    expected = heavier_billing_dtype(*loop, dt)
    assert rate_for(billed) >= rate_for(expected), (
        f"{op}({dtype}) billed {billed} (rate {rate_for(billed)}) but numpy "
        f"resolved {loop} (needs rate >= {rate_for(expected)})"
    )


def test_the_name_sets_hold_only_composites():
    """Every remaining set member must be an op with NO loop to resolve.

    This is the test that makes the refactor stick: re-adding a ufunc-backed
    name to either set means someone is hand-maintaining what numpy already
    answers, and it fails here rather than rotting until the next undercharge.
    """
    wiring = WIRING
    listed = _UNARY_FLOAT_LOOP_OPS | _UNARY_FLOAT64_MIN_OPS
    ufunc_backed = sorted(
        op
        for op in listed
        if isinstance(getattr(np, wiring.get(op, ""), None), np.ufunc)
    )
    assert not ufunc_backed, (
        "these set members ARE numpy ufuncs, so their compute dtype is "
        f"resolvable and the entry is dead weight: {ufunc_backed}"
    )


# ---------------------------------------------------------------------------
# Two regressions the A/B matrix caught while this refactor was being written.
# Both were invisible to a plain-call probe; they are pinned here by name.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["int32", "uint32"])
def test_bool_dtype_kwarg_still_bills_the_operand_loop(dtype):
    """``dtype=bool`` names the OUTPUT of a value test, not the loop's input.

    The wrapper deliberately ignores a bool ``dtype=`` and bills the operand,
    so the operand's loop still has to be resolved. Gating loop resolution on
    ``explicit_dtype is None`` alone reopened signbit's undercharge through
    this door: ``signbit(int32, dtype=bool)`` billed int32 while numpy read
    float64.
    """
    load_weights()
    arr = np.ones(N, dtype=dtype)
    with flops.budget(10**15, quiet=True) as b:
        fnp.signbit(arr, dtype=bool)
        forced = b.flops_used
    with flops.budget(10**15, quiet=True) as b:
        fnp.signbit(arr)
        plain = b.flops_used
    assert forced == plain, "a bool dtype= must not change what the call costs"


@pytest.mark.parametrize("op", ["modf", "frexp"])
def test_multi_output_destination_still_widens_the_rate(op):
    """A wider ``out=`` may only widen. ``heavier`` does not widen on a rate
    tie, so folding the destination into the operand FLOOR resolved a float32
    destination down to the float16 loop -- the narrowing #243 closed for the
    single-output wrapper. The destination stays its own billing entry."""
    load_weights()
    arr = np.ones(N, dtype=np.int8)  # float16 loop
    dest = tuple(np.empty(N, dtype=np.float32) for _ in range(2))
    with flops.budget(10**15, quiet=True) as b:
        getattr(fnp, op)(arr, out=dest)
        recs = [r for r in b.op_log if r.op_name == op]
    assert np.dtype(recs[-1].resolved_dtype) == np.dtype(np.float32)


@pytest.mark.parametrize("op", ["modf", "frexp"])
def test_multi_output_destination_is_still_seen_by_the_refusal(op):
    """Dropping the destination from the billing entries also hid it from the
    refusal ``deduct`` keys on them -- a complex destination silently became
    billable. It must still be refused."""
    load_weights()
    arr = np.ones(N, dtype=np.int32)
    dest = tuple(np.empty(N, dtype=np.complex128) for _ in range(2))
    with flops.budget(10**15, quiet=True):
        with pytest.raises(UnsupportedDtypeError):
            getattr(fnp, op)(arr, out=dest)
