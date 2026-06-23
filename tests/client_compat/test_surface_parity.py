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

# Excluded from the fix-now 48 for two DIFFERENT reasons (do not conflate):
#  - GENUINELY INCOMPATIBLE: raw-buffer / zero-copy interop whose contract is a
#    pointer to THIS array's local memory (data, __array_interface__,
#    __array_struct__, ctypes, __buffer__) and memory-reinterpret/view aliasing
#    (view, getfield, setfield, byteswap). The buffer lives on the server; there
#    is no local pointer, and a copy silently breaks the zero-copy/aliasing
#    contract. These are the only truly un-bridgeable entries.
#  - BRIDGEABLE METADATA we just haven't wired: strides, flags, itemsize are
#    plain server-reportable values (the client already bridges shape/ndim/size/
#    dtype/nbytes — the line is currently inconsistent). Worth bridging; parked
#    here so they don't inflate the fix-now count.
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
# In-place mutation. Immutable BY DESIGN on BOTH sides (competition rule
# #immutable-arrays): native FlopscopeArray and the client RemoteArray both raise
# a "flopscope arrays are immutable" error for item assignment, augmented
# assignment, and the C-level mutators (fill/put/resize/sort/partition). Excluded
# from the required surface because the client deliberately does not mirror the
# mutating API; the behavioural parity (both raise identically) is locked by
# dedicated tests (test_audit_gaps.py + tests/test_array_protocols.py), not by
# this static surface diff. (Bridging to true in-place mutation was rejected: it
# would need the server to model handle aliasing + views, and a half-bridge would
# be silently wrong.)
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

# Measured 2026-06-23 against flopscope-client 0.8.0rc3 / numpy 2.2.
# rc3 closed 46 of the original 48 gaps (read-only methods, bitwise/shift
# operators, conversion dunders, dtype type-objects, layout metadata, accounting,
# is_symmetric, native immutability, flopscope.numpy package).
# Only the two numpy dispatch-protocol hooks remain — DEFERRED, not participant-
# relevant for the evaluation model (participant code never registers custom ufuncs
# or __array_function__ overrides against the flopscope server).
KNOWN_MISSING = {
    "__array_ufunc__",  # numpy dispatch protocol — deferred, not eval-model
    "__array_function__",  # numpy dispatch protocol — deferred, not eval-model
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
    # rc3: down from 48 to 2 (only the numpy dispatch-protocol hooks remain, deferred).
    assert len(KNOWN_MISSING) == 2
