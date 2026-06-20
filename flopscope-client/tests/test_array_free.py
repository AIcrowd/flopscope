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
    from flopscope._remote_array import RemoteArray

    from flopscope import _handles

    _handles.drain_pending()
    arr = RemoteArray(handle_id="a7", shape=(2, 2), dtype="float32")
    assert _handles.pending_count() == 0  # alive -> not enqueued
    del arr
    gc.collect()
    assert _handles.drain_pending() == ["a7"]


def test_remote_scalar_does_not_enqueue():
    from flopscope._remote_array import RemoteScalar

    from flopscope import _handles

    _handles.drain_pending()
    s = RemoteScalar(value=1.5, dtype="float64")
    del s
    gc.collect()
    assert _handles.pending_count() == 0  # no server handle -> nothing to free


class _FakeSocket:
    """Records sends, replays scripted recvs (mirrors test_version_handshake)."""

    def __init__(self, recv_payloads):
        self.sent = []
        self._recv_payloads = list(recv_payloads)

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        return self._recv_payloads.pop(0)


def test_send_recv_flushes_pending_frees_first():
    from flopscope._connection import Connection
    from flopscope._protocol import encode_request

    from flopscope import _handles

    _handles.drain_pending()
    _handles.enqueue_free("a0")
    _handles.enqueue_free("a1")

    sock = _FakeSocket(
        [
            msgpack.packb({"status": "ok"}, use_bin_type=True),  # reply to free
            msgpack.packb(
                {"status": "ok", "result": 7}, use_bin_type=True
            ),  # reply to op
        ]
    )
    conn = Connection()
    conn._socket = sock
    conn._handshake_done = True  # bypass the lazy hello

    resp = conn.send_recv(encode_request("some_op"))
    assert resp["status"] == "ok"

    first = msgpack.unpackb(sock.sent[0], raw=False)
    assert first["op"] == "free"
    assert set(first["kwargs"]["handles"]) == {"a0", "a1"}
    second = msgpack.unpackb(sock.sent[1], raw=False)
    assert second["op"] == "some_op"
    assert _handles.pending_count() == 0


def test_send_recv_no_frees_sends_only_the_op():
    from flopscope._connection import Connection
    from flopscope._protocol import encode_request

    from flopscope import _handles

    _handles.drain_pending()
    sock = _FakeSocket(
        [msgpack.packb({"status": "ok", "result": 1}, use_bin_type=True)]
    )
    conn = Connection()
    conn._socket = sock
    conn._handshake_done = True

    conn.send_recv(encode_request("op2"))
    assert len(sock.sent) == 1
    assert msgpack.unpackb(sock.sent[0], raw=False)["op"] == "op2"
