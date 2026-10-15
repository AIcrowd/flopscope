"""Module-level shuffle records in-place writes, including through aliases (B3)."""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


@pytest.mark.parametrize("alias", [False, True])
def test_shuffle_invalidates_gram_tag_and_matches_numpy(alias):
    a = np.random.default_rng(12).normal(size=(8, 8))
    gram = a @ a.T
    expected = gram.copy()
    np.random.seed(42)
    np.random.shuffle(expected)
    with flops.BudgetContext(flop_budget=10**9, quiet=True):
        tagged = flops.as_symmetric(fnp.asarray(gram), symmetry=(0, 1))
        target = np.asarray(tagged) if alias else tagged
        fnp.random.seed(42)
        assert fnp.random.shuffle(target) is None
        assert tagged.symmetry is None
        np.testing.assert_array_equal(np.asarray(tagged), expected)
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as bill:
        fnp.einsum("ij,ij->", tagged, tagged)
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as reference:
        fnp.einsum("ij,ij->", expected, expected)
    assert bill.flops_used == reference.flops_used


def test_shuffle_drops_even_unaffected_tags_conservatively():
    # Reordering axis 0 does preserve symmetry between axes 1 and 2, but
    # invalidating the whole buffer matches other in-place write routes and
    # also protects tags held by aliases with different axis interpretations.
    values = np.arange(12.0).reshape(3, 2, 2)
    values = values + values.swapaxes(1, 2)
    with flops.BudgetContext(flop_budget=10**9, quiet=True):
        tagged = flops.as_symmetric(fnp.asarray(values), symmetry=(1, 2))
        fnp.random.shuffle(tagged)
        assert tagged.symmetry is None
        np.testing.assert_array_equal(
            np.asarray(tagged), np.asarray(tagged).swapaxes(1, 2)
        )
