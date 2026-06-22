"""Tests for GeneratorStore — written first (TDD)."""

import numpy as np
import pytest
from flopscope_server._generator_store import GeneratorStore


@pytest.fixture()
def store():
    return GeneratorStore()


def test_put_returns_handle(store):
    handle = store.put(np.random.default_rng(0))
    assert handle == "g0"


def test_get_returns_same_object(store):
    gen = np.random.default_rng(0)
    handle = store.put(gen)
    assert store.get(handle) is gen


def test_sequential_ids(store):
    gens = [store.put(np.random.default_rng(i)) for i in range(3)]
    assert gens == ["g0", "g1", "g2"]


def test_ids_are_process_global_across_instances():
    # No autouse reset between these two stores within one test: ids must NOT
    # restart at g0 for the second store (process-global monotonic counter).
    s1 = GeneratorStore()
    h0 = s1.put(np.random.default_rng(0))
    s2 = GeneratorStore()
    h1 = s2.put(np.random.default_rng(1))
    assert h0 == "g0"
    assert h1 == "g1"  # continues across instances, never reused


def test_get_missing_raises_keyerror_with_handle_and_message(store):
    try:
        store.get("g99")
    except KeyError as exc:
        assert "g99" in str(exc)
        assert "not found in store" in str(exc)
    else:
        pytest.fail("Expected KeyError for missing handle")


def test_ids_keep_incrementing_after_free(store):
    # Freeing a handle must NOT rewind the process-global counter — the next id
    # continues monotonically (the counter lives in the module, not the dict).
    h0 = store.put(np.random.default_rng(0))
    store.free([h0])
    h1 = store.put(np.random.default_rng(1))
    assert h0 == "g0"
    assert h1 == "g1"


def test_free_removes_handle(store):
    handle = store.put(np.random.default_rng(0))
    store.free([handle])
    with pytest.raises(KeyError):
        store.get(handle)


def test_free_unknown_handle_is_silent(store):
    store.free(["nonexistent"])  # must not raise


def test_free_empty_list_is_silent(store):
    store.free([])  # must not raise


def test_count_reflects_size(store):
    assert store.count == 0
    h0 = store.put(np.random.default_rng(0))
    assert store.count == 1
    store.free([h0])
    assert store.count == 0
    store.put(np.random.default_rng(1))
    store.clear()
    assert store.count == 0
