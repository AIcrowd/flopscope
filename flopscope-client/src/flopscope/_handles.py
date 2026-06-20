"""Deferred release of server-side array handles.

A RemoteArray finalizer ENQUEUES its handle here (pure append, no I/O — safe at
GC time and during interpreter shutdown). Connection.send_recv drains the queue
onto the next op's round-trip (see _connection.py), so frees ride existing
traffic and the strict REQ/REP socket is never used mid-op.
"""

from __future__ import annotations

_pending: set[str] = set()


def enqueue_free(handle_id: str) -> None:
    """Mark a server handle for release. Pure; no I/O. Finalizer-safe."""
    _pending.add(handle_id)


def drain_pending() -> list[str]:
    """Return a snapshot of pending handles and clear the queue."""
    if not _pending:
        return []
    snapshot = list(_pending)
    _pending.clear()
    return snapshot


def pending_count() -> int:
    return len(_pending)
