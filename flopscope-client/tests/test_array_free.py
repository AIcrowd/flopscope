"""Client-side server-handle release (free-on-GC) — queue, finalizer, flush."""

from __future__ import annotations

import gc

import msgpack


def test_enqueue_drain_count_roundtrip():
    from flopscope import _handles

    _handles.drain_pending()  # start clean
    assert _handles.pending_count() == 0
    _handles.enqueue_free("a0")
    _handles.enqueue_free("a1")
    _handles.enqueue_free("a0")  # dedup
    assert _handles.pending_count() == 2
    snap = _handles.drain_pending()
    assert set(snap) == {"a0", "a1"}
    assert _handles.pending_count() == 0  # cleared by drain
    assert _handles.drain_pending() == []  # empty is safe
