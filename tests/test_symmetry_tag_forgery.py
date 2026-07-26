"""Regression pins: a symmetry tag must not survive a write that invalidates it.

A symmetry tag is a billing-critical claim about buffer contents. The cost
model grants a ``k!``-scale discount on the strength of the tag alone and
never re-reads the data, and tags are validated once, at creation
(``flops.as_symmetric``) or inferred for free from shape on a constant fill
(``fnp.zeros`` and friends). So every route that writes into a tagged buffer --
or into any buffer a tagged array aliases -- must either refuse the write or
drop the claim. Otherwise a caller obtains a symmetric-rate discount on
asymmetric data, with numerically correct results and nothing to signal the
discrepancy.

Each test performs a write that makes the data asymmetric and then asserts the
invariant directly: either the write raised, or the array no longer buys the
discount. Both outcomes are acceptable; keeping the discount is not.
"""

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights

N = 32


def _billed(fn):
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn()
        return b.flops_used


def _asymmetric(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, n))


def _symmetric(n=N):
    a = _asymmetric(n)
    return a + a.T


def _probe(arr):
    """Contraction whose price depends only on the operands' symmetry tags."""
    return _billed(lambda: fnp.einsum("ij,ij->", arr, arr))


def _honest():
    return _probe(_asymmetric())


def _assert_claim_does_not_survive(tagged, write):
    """The invariant: no route yields asymmetric data that still bills symmetric."""
    load_weights()
    honest = _honest()
    try:
        write(tagged)
    except Exception:
        return  # write refused -- the claim was protected
    assert not np.allclose(np.asarray(tagged), np.asarray(tagged).T), (
        "test bug: the write did not actually break symmetry"
    )
    assert _probe(tagged) == honest


# --- free inferred tags from constant-fill constructors ------------------


def test_put_does_not_forge_tag():
    _assert_claim_does_not_survive(fnp.zeros((N, N)), lambda z: fnp.put(z, [1], [99.0]))


def test_fill_diagonal_keeps_symmetry_but_never_forges():
    z = fnp.zeros((N, N))
    load_weights()
    fnp.fill_diagonal(z, 3.0)
    # A diagonal write preserves symmetry, so the discount stays legitimate.
    assert np.allclose(np.asarray(z), np.asarray(z).T)


def test_copyto_does_not_forge_tag():
    _assert_claim_does_not_survive(
        fnp.zeros((N, N)), lambda z: fnp.copyto(z, _asymmetric())
    )


def test_ufunc_out_does_not_forge_tag():
    _assert_claim_does_not_survive(
        fnp.zeros((N, N)), lambda z: fnp.multiply(_asymmetric(), 1.0, out=z)
    )


def test_putmask_does_not_forge_tag():
    _assert_claim_does_not_survive(
        fnp.zeros((N, N)),
        lambda z: fnp.putmask(z, np.ones((N, N), dtype=bool), _asymmetric()),
    )


def test_place_does_not_forge_tag():
    _assert_claim_does_not_survive(
        fnp.zeros((N, N)),
        lambda z: fnp.place(z, np.ones((N, N), dtype=bool), _asymmetric().ravel()),
    )


def test_contraction_out_does_not_forge_tag():
    """vecdot/matvec/vecmat route through _einsum_routed_binary, which has no guard."""
    # Distinct operands, so the contraction result is genuinely asymmetric --
    # vecdot of an array against itself yields a symmetric Gram matrix and
    # would not exercise the forgery at all.
    _assert_claim_does_not_survive(
        fnp.zeros((N, N)),
        lambda z: fnp.vecdot(
            _asymmetric(seed=1)[:, None, :], _asymmetric(seed=2)[None, :, :], out=z
        ),
    )


# --- paid hard tags from as_symmetric, reached through aliases -----------


def test_write_through_untagged_parent_does_not_forge_tag():
    """as_symmetric returns a VIEW, so the parent aliases the tagged buffer."""
    load_weights()
    parent = _symmetric()
    tagged = f.as_symmetric(parent, symmetry=(0, 1))
    _assert_claim_does_not_survive(tagged, lambda _: fnp.copyto(parent, _asymmetric()))


def test_write_through_untagged_alias_does_not_forge_tag():
    """fnp.asarray/ravel hand back untagged aliases of a tagged buffer."""
    load_weights()
    tagged = fnp.zeros((N, N))
    alias = fnp.asarray(tagged)
    _assert_claim_does_not_survive(tagged, lambda _: fnp.copyto(alias, _asymmetric()))


def test_rng_out_does_not_forge_tag():
    """RNG methods call numpy directly, bypassing the shared out= write hook."""
    load_weights()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        z = fnp.zeros((N, N))
        fnp.random.default_rng(0).random(out=z)
        assert z.symmetry is None


def test_unverified_symmetry_claim_is_not_minted():
    """Validation is skipped for non-finite results, so no tag may be stamped."""
    load_weights()
    poisoned = _asymmetric()
    poisoned[0, 1] = np.inf
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        r = fnp.einsum("ij,jk->ik", poisoned, _asymmetric(seed=3), symmetry=(0, 1))
        assert getattr(r, "symmetry", None) is None


def test_tag_inherited_by_a_constant_fill_still_voids_on_write():
    """``zeros_like`` keeps a propagated trusted tag (pinned elsewhere); the
    epoch is what makes that safe once the fresh buffer is written."""
    load_weights()
    with f.BudgetContext(flop_budget=10**18, quiet=True):
        tagged = f.as_symmetric(_symmetric(), symmetry=(0, 1))
        arena = fnp.zeros_like(tagged)
        fnp.copyto(arena, _asymmetric())
        assert arena.symmetry is None


# --- the discount must still work when it is honestly earned -------------


def test_genuinely_symmetric_data_still_earns_the_discount():
    load_weights()
    tagged = f.as_symmetric(_symmetric(), symmetry=(0, 1))
    assert _probe(tagged) < _honest()


def test_scratch_buffer_idiom_still_works():
    """arena = fnp.zeros(...) then writing into it must keep working."""
    load_weights()
    arena = fnp.zeros((N, N))
    fnp.copyto(arena, _asymmetric())
    assert np.allclose(np.asarray(arena), _asymmetric())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
