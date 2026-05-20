"""Tests for einsum path caching.

Path and accumulation caches are now separate (see _einsum.py Decision Q1):
- _path_cache: caches (subscripts, shapes, optimize) -> path_info
- _accumulation_cache: caches (subscripts, shapes, sym_fingerprint) -> AccumulationResult

The deleted _symmetry_fingerprint / SubgraphSymmetryOracle tests are removed entirely.
"""

import numpy

from flopscope._config import configure, get_setting
from flopscope._einsum import _identity_pattern
from flopscope._perm_group import SymmetryGroup
from flopscope._symmetric import SymmetricTensor


def test_einsum_path_cache_size_default():
    assert get_setting("einsum_path_cache_size") == 4096


def test_einsum_path_cache_size_configurable():
    original = get_setting("einsum_path_cache_size")
    try:
        configure(einsum_path_cache_size=256)
        assert get_setting("einsum_path_cache_size") == 256
    finally:
        configure(einsum_path_cache_size=original)


def test_identity_pattern_distinct():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    pat = _identity_pattern([A, B])
    # Distinct objects -> no grouping, all singletons -> normalized to None
    assert pat is None


def test_identity_pattern_same_object():
    A = numpy.ones((3, 3))
    pat = _identity_pattern([A, A])
    # Same object at positions 0 and 1
    assert pat is not None
    assert isinstance(pat, tuple)
    hash(pat)  # Must be hashable


def test_identity_pattern_mixed():
    A = numpy.ones((3, 3))
    B = numpy.ones((3, 3))
    pat = _identity_pattern([A, B, A])
    # A is at positions 0 and 2 (same object), B at 1 (distinct)
    assert pat is not None
    hash(pat)


from flopscope._budget import BudgetContext
from flopscope._einsum import clear_einsum_cache, einsum, einsum_cache_info


def test_cache_hit_on_repeated_call():
    clear_einsum_cache()
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6):
        einsum("ij,jk->ik", A, B)
        einsum("ij,jk->ik", A, B)
    info = einsum_cache_info()
    assert info.hits >= 1
    assert info.misses >= 1


def test_cache_miss_on_different_shapes():
    clear_einsum_cache()
    with BudgetContext(flop_budget=10**6):
        einsum("ij,jk->ik", numpy.ones((3, 4)), numpy.ones((4, 5)))
        einsum("ij,jk->ik", numpy.ones((5, 6)), numpy.ones((6, 7)))
    info = einsum_cache_info()
    assert info.misses >= 2


def test_cache_miss_on_different_subscripts():
    clear_einsum_cache()
    A = numpy.ones((3, 3))
    with BudgetContext(flop_budget=10**6):
        einsum("ij,jk->ik", A, A)
        einsum("ij,ji->", A, A)
    info = einsum_cache_info()
    assert info.misses >= 2


def test_clear_einsum_cache():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6):
        einsum("ij,jk->ik", A, B)
    clear_einsum_cache()
    info = einsum_cache_info()
    assert info.currsize == 0


def test_cache_correctness():
    """Cached result must produce the same numerical output."""
    clear_einsum_cache()
    A = numpy.random.randn(4, 5)
    B = numpy.random.randn(5, 3)
    with BudgetContext(flop_budget=10**6):
        result1 = einsum("ij,jk->ik", A, B)
        result2 = einsum("ij,jk->ik", A, B)
    numpy.testing.assert_array_equal(result1, result2)
    expected = numpy.einsum("ij,jk->ik", A, B)
    numpy.testing.assert_allclose(result1, expected)


from flopscope._einsum import einsum_path


def test_einsum_path_warms_cache():
    """einsum_path() should warm the cache for subsequent einsum() calls."""
    clear_einsum_cache()
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6):
        einsum_path("ij,jk->ik", A, B)
    info_after_path = einsum_cache_info()
    assert info_after_path.misses >= 1

    with BudgetContext(flop_budget=10**6):
        einsum("ij,jk->ik", A, B)
    info_after_einsum = einsum_cache_info()
    # The einsum call should be a cache hit
    assert info_after_einsum.hits >= 1


def test_einsum_with_symmetric_input_cached():
    """Symmetric tensor inputs should be cached correctly."""
    clear_einsum_cache()
    data = numpy.random.randn(4, 4)
    sym_data = (data + data.T) / 2
    S = SymmetricTensor(sym_data, symmetry=SymmetryGroup.symmetric(axes=(0, 1)))
    B = numpy.random.randn(4, 3)
    with BudgetContext(flop_budget=10**6):
        r1 = einsum("ij,jk->ik", S, B)
        r2 = einsum("ij,jk->ik", S, B)
    numpy.testing.assert_array_equal(r1, r2)
    info = einsum_cache_info()
    assert info.hits >= 1


def test_einsum_same_object_cached():
    """einsum with the same object passed twice should cache correctly."""
    clear_einsum_cache()
    A = numpy.random.randn(3, 3)
    A_sym = (A + A.T) / 2
    with BudgetContext(flop_budget=10**6):
        r1 = einsum("ij,ji->", A_sym, A_sym)
        r2 = einsum("ij,ji->", A_sym, A_sym)
    numpy.testing.assert_array_equal(r1, r2)
    info = einsum_cache_info()
    assert info.hits >= 1


def test_configure_rebuilds_cache():
    """Changing einsum_path_cache_size should rebuild the cache."""
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6):
        einsum("ij,jk->ik", A, B)
    assert einsum_cache_info().currsize >= 1

    original = get_setting("einsum_path_cache_size")
    try:
        configure(einsum_path_cache_size=128)
        # Cache should be rebuilt (empty)
        assert einsum_cache_info().currsize == 0
        assert einsum_cache_info().maxsize == 128
    finally:
        configure(einsum_path_cache_size=original)


import flopscope.numpy as fnp


def test_public_api_clear():
    fnp.clear_einsum_cache()
    info = fnp.einsum_cache_info()
    assert info.currsize == 0


def test_public_api_cache_info():
    info = fnp.einsum_cache_info()
    assert hasattr(info, "hits")
    assert hasattr(info, "misses")
    assert hasattr(info, "maxsize")
    assert hasattr(info, "currsize")


