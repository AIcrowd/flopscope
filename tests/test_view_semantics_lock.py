"""Locks the view/copy semantics the cost model prices against.

If a numpy upgrade changes any of these, this file fails and the op's
price must be consciously re-decided (triage ledger 2026-07-17).
"""

import numpy as np

NP_GE_21 = tuple(int(p) for p in np.__version__.split(".")[:2]) >= (2, 1)

C2 = np.arange(24.0).reshape(4, 6)
SQ = np.arange(16.0).reshape(4, 4)
V1 = np.arange(24.0)


def _views(out, base):
    arrs = out if isinstance(out, (list, tuple)) else [out]
    return all(
        np.shares_memory(o, base) for o in arrs if isinstance(o, np.ndarray) and o.size
    )


def test_free_ops_are_views():
    assert _views(np.split(C2, 2, axis=0), C2)
    assert _views(np.hsplit(C2, 2), C2)
    assert _views(np.vsplit(C2, 2), C2)
    assert _views(np.array_split(V1, 7), V1)
    if NP_GE_21:  # np.unstack needs numpy >= 2.1; CI matrix includes 2.0
        assert _views(np.unstack(C2), C2)
    assert _views(np.broadcast_to(V1[:6], (4, 6)), V1)
    assert _views(np.diagonal(SQ), SQ)
    assert _views(np.diag(SQ), SQ)  # 2-D extract: view -> billed 0
    assert _views(np.transpose(C2), C2)
    assert _views(np.rot90(C2), C2)


def test_charged_copies_are_copies():
    assert not _views(np.diag(V1[:4]), V1)  # 1-D construct
    assert not _views(np.roll(V1, 3), V1)
    assert not _views(np.tile(V1, 2), V1)
    assert not _views(np.fft.fftshift(V1), V1)
    assert not _views(np.copy(C2), C2)
    # astype is deliberately FREE (conversions policy) — locked here only as
    # the semantics fact behind that accepted copy-tier bypass.
    assert not _views(np.astype(C2, np.float64), C2)
