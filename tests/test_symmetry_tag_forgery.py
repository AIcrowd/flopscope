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


def test_write_through_a_shim_interposed_view_does_not_forge_tag():
    """as_strided interposes a non-array object in the view chain, so the root
    walk has to see through it to reach the tags on the real buffer."""
    load_weights()
    parent = _symmetric()
    tagged = f.as_symmetric(parent, symmetry=(0, 1))
    strided = np.lib.stride_tricks.as_strided(
        np.asarray(tagged), shape=(N, N), strides=np.asarray(tagged).strides
    )
    _assert_claim_does_not_survive(tagged, lambda _: fnp.copyto(strided, _asymmetric()))


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
        assert getattr(z, "symmetry", None) is None


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
        assert getattr(arena, "symmetry", None) is None


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


# --- the same routes, with the destination inside a container ------------


def _assert_the_write_landed_and_voided_the_tag(tagged, write, expected):
    """Stricter than :func:`_assert_claim_does_not_survive`, on purpose.

    That helper accepts a raised exception as a pass, which is the right
    contract for it: for most of these routes, refusing the write and dropping
    the claim are equally safe outcomes. It is the wrong contract for the
    container tests. A container that is refused, or one whose write silently
    lands in a temporary, produces no forgery to detect — so a test built on
    that helper passes whether or not the guard exists, and one of the two
    added with this fix never reached its assertion at all: einsum raises a
    SymmetryError for the container and for the bare array alike, so it was
    not testing the container.

    Here the write must HAPPEN — asserted against the expected values, in the
    tagged buffer — and the tag must then be gone. Both halves can fail:
    unwrap the container into a temporary and the values assertion fails;
    skip the symmetry check for containers and the tag assertion fails.
    """
    load_weights()
    honest = _honest()

    write(tagged)

    written = np.asarray(tagged)
    assert np.allclose(written, expected), (
        "the write did not reach the tagged buffer — it landed in a temporary "
        "built from the container, which is the silent-wrong-answer failure"
    )
    assert not np.allclose(written, written.T), (
        "test bug: the write did not actually break symmetry"
    )
    assert getattr(tagged, "symmetry", None) is None, (
        "the symmetry tag survived a write that invalidated it"
    )
    assert _probe(tagged) == honest, (
        "asymmetric data still bills at the symmetric rate through a container"
    )


def test_ufunc_out_does_not_forge_tag_through_a_container():
    """A tuple-wrapped destination must void the tag exactly as a bare one does.

    It reaches ``_prepare_symmetric_out`` only once ``out`` is normalized;
    before that the symmetry check was simply skipped for the container, while
    numpy wrote through the tuple perfectly happily — data in, claim intact.
    """
    tagged = fnp.zeros((N, N))
    source = _asymmetric()
    _assert_the_write_landed_and_voided_the_tag(
        tagged,
        lambda z: fnp.multiply(source, 1.0, out=(z,)),  # pyright: ignore[reportArgumentType]
        expected=source,
    )


def test_contraction_out_does_not_forge_tag_through_a_container():
    """einsum is where a container is worst: it copies into ``out`` itself.

    The destination here is an untagged ALIAS of the tagged buffer — the
    spelling a caller reaches for when handing a scratch array to a routine —
    so the write is not refused and the tag has to be voided by the write
    itself. Unwrap the alias out of its tuple and everything works; leave the
    tuple standing and ``_np.asarray(container)`` builds a new array, the
    tagged buffer keeps its zeros AND its claim, and the caller pays full
    price for a contraction they never receive.
    """
    tagged = fnp.zeros((N, N))
    alias = fnp.asarray(tagged)
    assert getattr(alias, "symmetry", None) is None, "the alias must be untagged"
    source = _asymmetric()
    _assert_the_write_landed_and_voided_the_tag(
        tagged,
        lambda _: fnp.einsum("ij,jk->ik", source, fnp.eye(N), out=(alias,)),
        expected=source,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
