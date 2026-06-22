"""Tests for ConnectionStore — written first (TDD)."""

import numpy as np
from flopscope_server._array_store import ArrayStore
from flopscope_server._connection_store import ConnectionStore
from flopscope_server._generator_store import GeneratorStore


def test_bundles_array_and_generator_stores():
    conn = ConnectionStore()
    assert isinstance(conn.arrays, ArrayStore)
    assert isinstance(conn.generators, GeneratorStore)


def test_array_and_generator_stores_are_independent():
    conn = ConnectionStore()
    a = conn.arrays.put(np.array([1, 2, 3]))
    g = conn.generators.put(np.random.default_rng(0))
    assert a == "a0"
    assert g == "g0"
    np.testing.assert_array_equal(conn.arrays.get(a), np.array([1, 2, 3]))
    assert conn.generators.get(g) is not None


def test_each_connection_store_has_its_own_dicts():
    c1 = ConnectionStore()
    c2 = ConnectionStore()
    c1.arrays.put(np.array([1]))
    assert c1.arrays.count == 1
    assert c2.arrays.count == 0  # independent dicts
