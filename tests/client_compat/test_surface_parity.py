"""EXHAUSTIVE operator/method surface parity: RemoteArray vs numpy.ndarray.

Why this exists: the numpy-suite harness patches numpy *functions* and runs them
on native ndarrays, so it NEVER exercises ``RemoteArray``'s own attribute surface
— a participant doing ``a.argsort()`` or ``a & b`` on a client array is invisible
to it. Mining prod failures is also not exhaustive: it only finds gaps that some
submission happened to hit.

The participant's local reference is ``FlopscopeArray``, a ``numpy.ndarray``
SUBCLASS, so locally they inherit the ENTIRE ndarray method/operator surface. The
client ``RemoteArray`` is a hand-built proxy that implements only a subset. The
COMPLETE set of operator/method parity gaps is therefore enumerable and bounded
by the native API:

    dir(numpy.ndarray)  −  dir(RemoteArray)

filtered to the participant-relevant surface (excluding what a remote proxy
genuinely cannot/should not mirror). This test pins that set to a measured
baseline so that:
  * a NEW gap (client regression, or a numpy version adding a method the client
    doesn't mirror) fails the test, and
  * a CLOSED gap (Phase-2 fix) also fails the test, forcing the baseline — and
    thus the inventory — to be pruned and kept honest.

No server needed: this is a static class-surface diff.
"""

from __future__ import annotations

import numpy as np
from flopscope._remote_array import RemoteArray

# ndarray surface a remote proxy genuinely cannot/should not mirror: local memory
# layout, numpy-subclass internals, device/IO buffers, object plumbing.
_PROXY_IMPOSSIBLE = {
    "data",
    "strides",
    "flags",
    "base",
    "ctypes",
    "itemset",
    "newbyteorder",
    "dtype",
    "getfield",
    "setfield",
    "byteswap",
    "view",
    "to_device",
    "tobytes",
    "tofile",
    "tostring",
    "dump",
    "dumps",
    "setflags",
    "__buffer__",
    "__array_interface__",
    "__array_struct__",
    "__array_priority__",
    "__array_finalize__",
    "__array_wrap__",
    "__array_namespace__",
    "__dlpack__",
    "__dlpack_device__",
    "__class_getitem__",
    "__init_subclass__",
    "__subclasshook__",
    "__reduce__",
    "__reduce_ex__",
    "__dir__",
    "__sizeof__",
    "__init__",
    "__new__",
    "__getstate__",
    "__setstate__",
    "__delattr__",
    "__setattr__",
    "__getattribute__",
}
# In-place mutation: RemoteArray is immutable by design (server-held handle).
_BY_DESIGN_IMMUTABLE = {
    "fill",
    "partition",
    "put",
    "resize",
    "sort",
    "__delitem__",
    "__setitem__",
    "__iadd__",
    "__iand__",
    "__ifloordiv__",
    "__ilshift__",
    "__imatmul__",
    "__imod__",
    "__imul__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__itruediv__",
    "__ixor__",
}

# Measured 2026-06-23 against flopscope-client 0.8.0rc2 / numpy 2.2. Every entry
# is a real participant-usable ndarray method/operator the client is MISSING
# (10/10 sampled raise AttributeError/TypeError live). This is the exhaustive
# operator/method gap inventory — Phase 2 shrinks it; do not edit by hand except
# to remove an entry a fix has genuinely closed.
KNOWN_MISSING = {
    # read-only methods (function forms like np.argsort may work; the METHOD does not)
    "all",
    "any",
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "choose",
    "clip",
    "compress",
    "conj",
    "conjugate",
    "cumprod",
    "cumsum",
    "diagonal",
    "item",
    "nonzero",
    "prod",
    "repeat",
    "round",
    "searchsorted",
    "squeeze",
    "std",
    "swapaxes",
    "take",
    "trace",
    "var",
    # operator / conversion dunders
    "__and__",
    "__or__",
    "__xor__",
    "__invert__",
    "__lshift__",
    "__rshift__",
    "__rand__",
    "__ror__",
    "__rxor__",
    "__rlshift__",
    "__rrshift__",
    "__divmod__",
    "__rdivmod__",
    "__pos__",
    "__contains__",
    "__index__",
    "__complex__",
    "__copy__",
    "__deepcopy__",
    "__array__",
    "__array_ufunc__",
    "__array_function__",
}


def _participant_relevant_surface() -> set[str]:
    surface = {n for n in dir(np.ndarray) if callable(getattr(np.ndarray, n, None))}
    return surface - _PROXY_IMPOSSIBLE - _BY_DESIGN_IMMUTABLE


def test_remote_array_surface_matches_measured_baseline():
    missing = _participant_relevant_surface() - set(dir(RemoteArray))
    new = sorted(missing - KNOWN_MISSING)
    closed = sorted(KNOWN_MISSING - missing)
    assert not new, (
        "NEW RemoteArray surface gap(s) not in the baseline — a client regression "
        f"or a numpy method the client does not mirror: {new}"
    )
    assert not closed, (
        "Surface gap(s) now CLOSED — remove them from KNOWN_MISSING (and update "
        f"INVENTORY.md): {closed}"
    )


def test_baseline_is_an_honest_count():
    # Guards the documented inventory number against silent drift.
    assert len(KNOWN_MISSING) == 48
