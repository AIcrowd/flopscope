"""Spelling invariance for ops reachable as both ``fnp.X`` and ``fnp.linalg.X``.

``trace`` and ``cross`` each have a bare ``numpy`` spelling and a ``numpy.linalg``
spelling that compute a bit-identical result. They must therefore bill
identically. A regression (GitHub #178) gave the ``numpy.linalg`` registry entry
a *blanket* ``complex_factor = 4.0`` that discarded the op-specific complex factor
carried by its bare twin, so on complex dtypes ``linalg.trace`` over-billed 2x and
``linalg.cross`` under-billed (a 15% discount a participant could claim just by
choosing the spelling).

These guards assert per-op spelling invariance across dtypes -- crucially the
complex ones, where the divergence lived -- and also pin the registry factors to
their bare twins so any future blanket override is caught at the source.
"""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._registry import REGISTRY

# Real dtypes read complex_factor 1.0, so they were already invariant; they are
# included to prove the fix leaves real-dtype billing untouched. complex64 and
# complex128 are where the blanket factor diverged.
DTYPES = [np.float32, np.float64, np.int32, np.complex64, np.complex128]


def _billed(call):
    with flops.budget(10**15, quiet=True) as budget:
        call()
    return budget.flops_used


# --- trace ----------------------------------------------------------------


# Only 2-D and matched-axis stacked cases: bare np.trace defaults to
# axis1=0/axis2=1 while np.linalg.trace uses the trailing pair, so the *default*
# stacked results differ by numpy semantics, not by billing.
def _trace_cases(dtype):
    square = fnp.array(np.arange(256 * 256, dtype=dtype).reshape(256, 256))
    tall = fnp.array(np.arange(128 * 300, dtype=dtype).reshape(128, 300))
    yield "square", lambda: fnp.trace(square), lambda: fnp.linalg.trace(square)
    yield "non_square", lambda: fnp.trace(tall), lambda: fnp.linalg.trace(tall)
    stacked = fnp.array(np.arange(10 * 64 * 64, dtype=dtype).reshape(10, 64, 64))
    yield (
        "stacked",
        lambda: fnp.trace(stacked, axis1=1, axis2=2),
        lambda: fnp.linalg.trace(stacked),
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
def test_trace_spelling_invariance(dtype):
    for label, bare, linalg in _trace_cases(dtype):
        # Anchor: the two spellings really do compute the same numbers.
        assert np.allclose(np.asarray(bare()), np.asarray(linalg())), label
        bare_bill = _billed(bare)
        linalg_bill = _billed(linalg)
        assert bare_bill == linalg_bill, (
            f"trace {label} {np.dtype(dtype).name}: "
            f"bare={bare_bill} linalg={linalg_bill}"
        )


# --- cross -----------------------------------------------------------------


def _cross_cases(dtype):
    a1 = fnp.array(np.array([1, 2, 3], dtype=dtype))
    b1 = fnp.array(np.array([4, 5, 6], dtype=dtype))
    yield "single", lambda: fnp.cross(a1, b1), lambda: fnp.linalg.cross(a1, b1)
    a2 = fnp.array(np.arange(3000 * 3, dtype=dtype).reshape(3000, 3))
    b2 = fnp.array(np.ones((3000, 3), dtype=dtype))
    yield "batched", lambda: fnp.cross(a2, b2), lambda: fnp.linalg.cross(a2, b2)
    a3 = fnp.array(np.arange(50 * 40 * 3, dtype=dtype).reshape(50, 40, 3))
    b3 = fnp.array(np.ones((50, 40, 3), dtype=dtype))
    yield "stacked", lambda: fnp.cross(a3, b3), lambda: fnp.linalg.cross(a3, b3)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
def test_cross_spelling_invariance(dtype):
    for label, bare, linalg in _cross_cases(dtype):
        assert np.allclose(np.asarray(bare()), np.asarray(linalg())), label
        bare_bill = _billed(bare)
        linalg_bill = _billed(linalg)
        assert bare_bill == linalg_bill, (
            f"cross {label} {np.dtype(dtype).name}: "
            f"bare={bare_bill} linalg={linalg_bill}"
        )


# --- source-level guard against re-introducing a blanket factor ------------


@pytest.mark.parametrize("op", ["trace", "cross"])
def test_linalg_twin_shares_bare_complex_factor(op):
    """The linalg spelling must carry its bare twin's op-specific factor.

    Directly pins the registry so a future blanket ``complex_factor = 4.0`` on
    these entries (the #178 defect) fails here, not just in a billing assertion.
    """
    bare = REGISTRY[op]["complex_factor"]
    linalg = REGISTRY[f"linalg.{op}"]["complex_factor"]
    assert linalg == bare, (
        f"linalg.{op} complex_factor {linalg!r} must equal bare {op} {bare!r}; "
        "a blanket override would silently mis-bill the linalg spelling"
    )
