"""Tests for ArrayStore — written first (TDD)."""

import numpy as np
import pytest
from flopscope_server._array_store import ArrayStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store():
    return ArrayStore()


# ---------------------------------------------------------------------------
# put / get round-trip
# ---------------------------------------------------------------------------


def test_put_returns_handle(store):
    arr = np.array([1, 2, 3])
    handle = store.put(arr)
    assert isinstance(handle, str)


def test_get_returns_same_array(store):
    arr = np.array([1.0, 2.0, 3.0])
    handle = store.put(arr)
    retrieved = store.get(handle)
    np.testing.assert_array_equal(retrieved, arr)


def test_get_returns_exact_object(store):
    """The store should return the exact array object, not a copy."""
    arr = np.zeros((3, 4))
    handle = store.put(arr)
    assert store.get(handle) is arr


# ---------------------------------------------------------------------------
# Sequential IDs
# ---------------------------------------------------------------------------


def test_first_id_is_a0(store):
    handle = store.put(np.array([]))
    assert handle == "a0"


def test_sequential_ids(store):
    handles = [store.put(np.array([i])) for i in range(5)]
    assert handles == ["a0", "a1", "a2", "a3", "a4"]


def test_ids_keep_incrementing_after_free(store):
    h0 = store.put(np.array([0]))
    store.free([h0])
    h1 = store.put(np.array([1]))
    assert h1 == "a1"


# ---------------------------------------------------------------------------
# get — missing handle
# ---------------------------------------------------------------------------


def test_get_missing_raises_keyerror(store):
    with pytest.raises(KeyError):
        store.get("a99")


def test_get_missing_keyerror_contains_handle(store):
    try:
        store.get("a99")
    except KeyError as exc:
        assert "a99" in str(exc)
    else:
        pytest.fail("Expected KeyError")


# ---------------------------------------------------------------------------
# free
# ---------------------------------------------------------------------------


def test_free_single(store):
    handle = store.put(np.array([1]))
    store.free([handle])
    with pytest.raises(KeyError):
        store.get(handle)


def test_free_multiple(store):
    h0 = store.put(np.array([0]))
    h1 = store.put(np.array([1]))
    h2 = store.put(np.array([2]))
    store.free([h0, h2])
    with pytest.raises(KeyError):
        store.get(h0)
    np.testing.assert_array_equal(store.get(h1), np.array([1]))
    with pytest.raises(KeyError):
        store.get(h2)


def test_free_unknown_handle_is_silent(store):
    """Freeing a handle that was never stored should not raise."""
    store.free(["nonexistent"])


def test_free_empty_list_is_silent(store):
    store.free([])


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_removes_all(store):
    store.put(np.array([1]))
    store.put(np.array([2]))
    store.clear()
    assert store.count == 0


def test_clear_makes_handles_invalid(store):
    h = store.put(np.array([1]))
    store.clear()
    with pytest.raises(KeyError):
        store.get(h)


# ---------------------------------------------------------------------------
# count property
# ---------------------------------------------------------------------------


def test_count_starts_at_zero(store):
    assert store.count == 0


def test_count_increments_on_put(store):
    store.put(np.array([1]))
    assert store.count == 1
    store.put(np.array([2]))
    assert store.count == 2


def test_count_decrements_on_free(store):
    h = store.put(np.array([1]))
    store.put(np.array([2]))
    store.free([h])
    assert store.count == 1


def test_count_zero_after_clear(store):
    store.put(np.array([1]))
    store.put(np.array([2]))
    store.clear()
    assert store.count == 0


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def test_metadata_id(store):
    arr = np.zeros((2, 3))
    handle = store.put(arr)
    meta = store.metadata(handle)
    assert meta["id"] == handle


def test_metadata_shape(store):
    arr = np.zeros((4, 5))
    handle = store.put(arr)
    meta = store.metadata(handle)
    assert meta["shape"] == [4, 5]


def test_metadata_shape_is_list(store):
    arr = np.zeros((2, 3))
    handle = store.put(arr)
    meta = store.metadata(handle)
    assert isinstance(meta["shape"], list)


def test_metadata_dtype(store):
    arr = np.array([1, 2, 3], dtype=np.float32)
    handle = store.put(arr)
    meta = store.metadata(handle)
    assert meta["dtype"] == "float32"


def test_metadata_dtype_int64(store):
    arr = np.array([1, 2, 3], dtype=np.int64)
    handle = store.put(arr)
    meta = store.metadata(handle)
    assert meta["dtype"] == "int64"


def test_metadata_1d_shape(store):
    arr = np.array([1, 2, 3, 4, 5])
    handle = store.put(arr)
    meta = store.metadata(handle)
    assert meta["shape"] == [5]


def test_metadata_0d_shape(store):
    arr = np.array(42.0)
    handle = store.put(arr)
    meta = store.metadata(handle)
    assert meta["shape"] == []


def test_metadata_missing_raises_keyerror(store):
    with pytest.raises(KeyError):
        store.metadata("a999")


# ---------------------------------------------------------------------------
# Cross-session handle uniqueness (Fix B regression test)
# ---------------------------------------------------------------------------


def test_handles_unique_across_stores():
    """Handle ids must NOT reset per ArrayStore/session — two stores (simulating
    two sessions) must mint distinct ids, so a stale free can't alias a live
    handle in a new session."""
    import flopscope_server._array_store as mod

    mod._reset_handle_counter()
    s1 = ArrayStore()
    h1 = s1.put(np.array([1.0]))
    s2 = ArrayStore()
    h2 = s2.put(np.array([2.0]))
    assert h1 != h2, f"handle reuse across stores: {h1!r} == {h2!r}"
