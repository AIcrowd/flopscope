"""Tests for Session — written first (TDD)."""

from __future__ import annotations

import numpy as np
import pytest
from flopscope_server._connection_store import ConnectionStore
from flopscope_server._session import Session

import flopscope as flops

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def session():
    s = Session(flop_budget=100_000)
    yield s
    # Ensure session is closed after each test even if the test didn't close it
    if s.is_open:
        s.close()


# ---------------------------------------------------------------------------
# Creating a session
# ---------------------------------------------------------------------------


def test_session_is_open_after_creation(session):
    assert session.is_open is True


def test_session_has_budget_remaining(session):
    assert session.budget_remaining == 100_000


def test_session_budget_context_is_accessible(session):
    ctx = session.budget_context
    assert ctx is not None
    assert ctx.flop_budget == 100_000


def test_session_comms_tracker_is_accessible(session):
    tracker = session.comms_tracker
    assert tracker is not None
    # Should have zero requests initially
    assert tracker.summary()["request_count"] == 0


# ---------------------------------------------------------------------------
# Store and retrieve array
# ---------------------------------------------------------------------------


def test_store_array_returns_handle(session):
    arr = np.array([1.0, 2.0, 3.0])
    handle = session.store_array(arr)
    assert isinstance(handle, str)


def test_get_array_returns_original(session):
    arr = np.array([1.0, 2.0, 3.0])
    handle = session.store_array(arr)
    retrieved = session.get_array(handle)
    np.testing.assert_array_equal(retrieved, arr)


def test_get_array_returns_same_object(session):
    arr = np.zeros((3, 4))
    handle = session.store_array(arr)
    assert session.get_array(handle) is arr


def test_store_multiple_arrays(session):
    h0 = session.store_array(np.array([0]))
    h1 = session.store_array(np.array([1]))
    assert h0 != h1
    np.testing.assert_array_equal(session.get_array(h0), np.array([0]))
    np.testing.assert_array_equal(session.get_array(h1), np.array([1]))


# ---------------------------------------------------------------------------
# Array metadata
# ---------------------------------------------------------------------------


def test_array_metadata_returns_id(session):
    arr = np.zeros((2, 3))
    handle = session.store_array(arr)
    meta = session.array_metadata(handle)
    assert meta["id"] == handle


def test_array_metadata_returns_shape(session):
    arr = np.zeros((4, 5))
    handle = session.store_array(arr)
    meta = session.array_metadata(handle)
    assert meta["shape"] == [4, 5]


def test_array_metadata_returns_dtype(session):
    arr = np.array([1.0, 2.0], dtype=np.float32)
    handle = session.store_array(arr)
    meta = session.array_metadata(handle)
    assert meta["dtype"] == "float32"


def test_array_metadata_missing_raises_keyerror(session):
    with pytest.raises(KeyError):
        session.array_metadata("nonexistent")


# ---------------------------------------------------------------------------
# free_arrays
# ---------------------------------------------------------------------------


def test_free_arrays_removes_handles(session):
    h0 = session.store_array(np.array([1]))
    h1 = session.store_array(np.array([2]))
    session.free_arrays([h0])
    with pytest.raises(KeyError):
        session.get_array(h0)
    # h1 still accessible
    np.testing.assert_array_equal(session.get_array(h1), np.array([2]))


def test_free_arrays_silently_ignores_unknown(session):
    session.free_arrays(["nonexistent"])  # should not raise


# ---------------------------------------------------------------------------
# Budget status
# ---------------------------------------------------------------------------


def test_budget_status_has_required_keys(session):
    status = session.budget_status()
    assert "flop_budget" in status
    assert "flops_used" in status
    assert "flops_remaining" in status


def test_budget_status_initial_values(session):
    status = session.budget_status()
    assert status["flop_budget"] == 100_000
    assert status["flops_used"] == 0
    assert status["flops_remaining"] == 100_000


def test_budget_status_remaining_equals_budget_minus_used(session):
    status = session.budget_status()
    assert status["flops_remaining"] == status["flop_budget"] - status["flops_used"]


# ---------------------------------------------------------------------------
# Close session
# ---------------------------------------------------------------------------


def test_close_returns_dict(session):
    result = session.close()
    assert isinstance(result, dict)


def test_close_returns_budget_summary(session):
    result = session.close()
    assert "budget_summary" in result
    assert isinstance(result["budget_summary"], str)


def test_close_returns_budget_breakdown(session):
    result = session.close()
    assert "budget_breakdown" in result
    assert isinstance(result["budget_breakdown"], dict)
    assert "by_namespace" in result["budget_breakdown"]


def test_close_keeps_flat_summary_for_unlabeled_sessions(session):
    result = session.close()
    assert "By namespace:" not in result["budget_summary"]
    assert result["budget_breakdown"]["by_namespace"] == {}


def test_close_includes_namespace_breakdown_when_session_is_labeled(session):
    with flops.namespace("phase"):
        session.budget_context.deduct("add", flop_cost=1, subscripts=None, shapes=())

    result = session.close()
    assert "By namespace:" in result["budget_summary"]
    assert "phase" in result["budget_summary"]
    assert result["budget_breakdown"]["by_namespace"]["phase"]["flops_used"] == 1
    assert result["budget_breakdown"]["by_namespace"]["phase"]["calls"] == 1


def test_close_budget_breakdown_keeps_unlabeled_ops_structured(session):
    session.budget_context.deduct("add", flop_cost=1, subscripts=None, shapes=())

    result = session.close()
    assert result["budget_breakdown"]["by_namespace"][None]["flops_used"] == 1
    assert result["budget_breakdown"]["by_namespace"][None]["calls"] == 1


def test_close_returns_comms_summary(session):
    result = session.close()
    assert "comms_summary" in result
    assert isinstance(result["comms_summary"], dict)


def test_close_marks_session_closed():
    s = Session(flop_budget=1000)
    s.close()
    assert s.is_open is False


def test_close_does_not_clear_bare_session_store():
    # A bare Session owns a private ConnectionStore. close() exits the budget
    # but does NOT clear the store — handles are connection-lifetime, not
    # session-lifetime (issue #107). The array remains resolvable afterward.
    s = Session(flop_budget=1000)
    handle = s.store_array(np.array([1, 2, 3]))
    s.close()
    assert s.is_open is False
    np.testing.assert_array_equal(s._conn.arrays.get(handle), np.array([1, 2, 3]))


def test_close_twice_raises_runtime_error():
    s = Session(flop_budget=1000)
    s.close()
    with pytest.raises(RuntimeError):
        s.close()


# ---------------------------------------------------------------------------
# budget_context raises when closed
# ---------------------------------------------------------------------------


def test_budget_context_raises_when_closed():
    s = Session(flop_budget=1000)
    s.close()
    with pytest.raises(RuntimeError):
        _ = s.budget_context


# ---------------------------------------------------------------------------
# budget_remaining property
# ---------------------------------------------------------------------------


def test_budget_remaining_reflects_context():
    s = Session(flop_budget=100_000)
    assert s.budget_remaining == 100_000
    s.close()


# ---------------------------------------------------------------------------
# Server contract: close() surfaces recorded compute time
# ---------------------------------------------------------------------------


def test_close_surfaces_recorded_compute_time(session):
    """close()'s comms_summary carries the compute time the client reads as backend."""
    session.comms_tracker.record_request(
        bytes_received=10,
        bytes_sent=20,
        comms_overhead_ns=100,
        compute_time_ns=5000,
        is_fetch=False,
    )
    summary = session.close()
    assert summary["comms_summary"]["total_compute_time_ns"] == 5000


# ---------------------------------------------------------------------------
# Connection-lifetime store (fix #1, issue #107)
# ---------------------------------------------------------------------------


def test_close_does_not_clear_injected_store():
    """A handle stored in session 1 survives close() when a ConnectionStore is shared."""
    conn = ConnectionStore()
    s1 = Session(flop_budget=100_000, conn_store=conn)
    handle = s1.store_array(np.array([1.0, 2.0, 3.0]))
    s1.close()
    # The store is owned by the connection, not the session — handle still resolves.
    np.testing.assert_array_equal(conn.arrays.get(handle), np.array([1.0, 2.0, 3.0]))


def test_handle_resolves_in_a_later_session_sharing_the_store():
    """The warm-child shape: session 2 (same ConnectionStore) can read session 1's handle."""
    conn = ConnectionStore()
    s1 = Session(flop_budget=100_000, conn_store=conn)
    floor = s1.store_array(np.float32(1e-12))
    s1.close()

    s2 = Session(flop_budget=100_000, conn_store=conn)
    later = s2.store_array(np.arange(5))
    assert later != floor  # process-global monotonic id, never reused
    np.testing.assert_array_equal(s2.get_array(floor), np.float32(1e-12))
    s2.close()


def test_generator_handle_survives_across_sessions_sharing_the_store():
    conn = ConnectionStore()
    s1 = Session(flop_budget=100_000, conn_store=conn)
    g = s1.store_generator(np.random.default_rng(0))
    s1.close()

    s2 = Session(flop_budget=100_000, conn_store=conn)
    assert s2.get_generator(g) is not None
    s2.close()


def test_bare_session_still_works_with_private_store():
    """No injected store -> Session makes its own ConnectionStore (standalone use)."""
    s = Session(flop_budget=100_000)
    h = s.store_array(np.array([1]))
    np.testing.assert_array_equal(s.get_array(h), np.array([1]))
    s.close()
