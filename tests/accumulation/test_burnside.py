"""Tests for _burnside.py — port of sizeAware/burnside.js."""

import pytest

from flopscope._accumulation._burnside import size_aware_burnside
from flopscope._perm_group import _Permutation as Permutation


def test_burnside_trivial_group_returns_product():
    # Trivial group → orbit count = |X| = ∏ sizes
    identity = Permutation.identity(3)
    assert size_aware_burnside((identity,), (2, 3, 4)) == 24


def test_burnside_s2_uniform_size():
    # S_2 on 2 labels of size n: orbit count = n*(n+1)/2
    swap = Permutation([1, 0])
    identity = Permutation.identity(2)
    elements = (identity, swap)
    # n=4: 4·5/2 = 10
    assert size_aware_burnside(elements, (4, 4)) == 10
    # n=5: 5·6/2 = 15
    assert size_aware_burnside(elements, (5, 5)) == 15


def test_burnside_s3_uniform_size():
    # S_3 on 3 labels of size n: orbit count = C(n+2, 3) = n(n+1)(n+2)/6
    s01 = Permutation([1, 0, 2])
    s12 = Permutation([0, 2, 1])
    from flopscope._perm_group import _dimino

    elements = _dimino((s01, s12))
    # n=4: 4·5·6/6 = 20
    assert size_aware_burnside(elements, (4, 4, 4)) == 20


def test_burnside_heterogeneous_disjoint():
    # S_2 on first two labels (size 3), S_2 on last two (size 5). Disjoint actions.
    swap_first = Permutation([1, 0, 2, 3])
    swap_last = Permutation([0, 1, 3, 2])
    from flopscope._perm_group import _dimino

    elements = _dimino((swap_first, swap_last))
    # Expected: (3·4/2) * (5·6/2) = 6 · 15 = 90
    assert size_aware_burnside(elements, (3, 3, 5, 5)) == 90


def test_burnside_rejects_cycle_with_mixed_sizes():
    # A swap on positions of unequal size violates a precondition.
    swap = Permutation([1, 0])
    with pytest.raises(ValueError, match="cycle size mismatch"):
        size_aware_burnside((swap, Permutation.identity(2)), (3, 5))


def test_burnside_rejects_empty_group():
    with pytest.raises(ValueError, match="at least one group element"):
        size_aware_burnside((), (3,))


def test_burnside_returns_integer_for_all_inputs():
    # Property check: any well-formed group + size pair should give integer count.
    swap = Permutation([1, 0])
    identity = Permutation.identity(2)
    result = size_aware_burnside((identity, swap), (5, 5))
    assert isinstance(result, int)
