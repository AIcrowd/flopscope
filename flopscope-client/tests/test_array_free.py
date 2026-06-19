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


def test_remote_array_enqueues_handle_on_gc():
    from flopscope import _handles
    from flopscope._remote_array import RemoteArray

    _handles.drain_pending()
    arr = RemoteArray(handle_id="a7", shape=(2, 2), dtype="float32")
    assert _handles.pending_count() == 0  # alive -> not enqueued
    del arr
    gc.collect()
    assert _handles.drain_pending() == ["a7"]


def test_remote_scalar_does_not_enqueue():
    from flopscope import _handles
    from flopscope._remote_array import RemoteScalar

    _handles.drain_pending()
    s = RemoteScalar(value=1.5, dtype="float64")
    del s
    gc.collect()
    assert _handles.pending_count() == 0  # no server handle -> nothing to free
