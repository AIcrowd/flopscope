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

One route is NOT closed, and is pinned as an expected failure at the bottom of
this file rather than left to look covered: Python's buffer protocol.
``memoryview(tagged)[i, j] = ...`` writes the buffer with nothing in the Python
layer to observe it. Closing it needs either ``__buffer__`` (PEP 688, and so
Python 3.12+, above this package's floor) or making tagged buffers
non-writeable, which would break their legitimate use as ``out=``
destinations. ``.data`` is not a second instance -- NumPy refuses that itself.

Each test performs a write that makes the data asymmetric and then asserts the
invariant directly: either the write raised, or the array no longer buys the
discount. Both outcomes are acceptable; keeping the discount is not.
"""

import functools
import inspect

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights
from flopscope._write_epoch import epoch_of

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


# --- writes that never reached the shared out= hook ----------------------
#
# The hook that voids a tag lives in ``_call_numpy`` and fires on the ``out=``
# KEYWORD. Two families of write never presented one: a reduction whose
# destination travels in a positional slot instead, and the ufunc methods,
# which called numpy directly rather than through ``_call_numpy`` at all. In
# both the destination below is an untagged ALIAS of the tagged buffer, which
# is what a guard on tagged arrays cannot reach and what makes the write land
# instead of being refused.
#
# Each of these asserts the BILLED COST of a contraction on the tagged buffer,
# not just the epoch counter: an epoch that moves while the discount survives
# would be a passing test over a live under-bill.


def _alias_of_a_tagged_buffer():
    tagged = fnp.zeros((N, N))
    alias = fnp.asarray(tagged)
    assert getattr(alias, "symmetry", None) is None, "the alias must be untagged"
    return tagged, alias


def test_reduction_positional_out_does_not_forge_tag():
    """``fnp.sum(a, axis, dtype, dest)`` -- ``out`` in its positional slot.

    The positional route hands the destination back to numpy inside the
    positional argument list, so no ``out=`` keyword exists for the write hook
    to see, and the tag outlived the data it described.
    """
    tagged, alias = _alias_of_a_tagged_buffer()
    source = np.stack([_asymmetric(seed=1), np.zeros((N, N))])
    _assert_the_write_landed_and_voided_the_tag(
        tagged,
        lambda _: fnp.sum(source, 0, None, alias),  # pyright: ignore[reportCallIssue]
        expected=source.sum(axis=0),
    )


_POSITIONAL_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _reductions_with_a_positional_out_slot():
    """Re-derive, from numpy's signatures, which reductions can take ``out``
    positionally -- the exact condition the production wrapper keys on.

    Derived rather than listed so a reduction that joins the family is swept
    instead of remembered. The derivation is independent of the production
    one (this reads ``numpy``'s signature, the wrapper reads its own captured
    function), so a divergence shows up as a call that fails loudly.
    """
    found = {}
    for name in dir(fnp):
        if name.startswith("_"):
            continue
        fn = getattr(fnp, name, None)
        np_func = getattr(np, name, None)
        if not callable(fn) or not callable(np_func):
            continue
        try:
            params = inspect.signature(np_func).parameters
        except (TypeError, ValueError):
            continue
        names = list(params)
        if len(names) < 2 or names[1] != "axis" or "out" not in names:
            continue
        if params["axis"].kind not in _POSITIONAL_KINDS:
            continue
        slot = names.index("out") - 2
        if slot >= 0:
            found[name] = (fn, slot)
    return found


def test_every_positional_out_reduction_matches_its_keyword_spelling():
    """Sweep, not a spot check: for every reduction that accepts ``out`` in a
    positional slot, the positional spelling must void the tag and bill exactly
    what the keyword spelling bills.

    The equal-billing half is the standing invariant that ``out=(d,)`` bills
    what ``out=d`` bills, applied to the other spelling of the same argument;
    the voided-tag half is the fix.
    """
    load_weights()
    honest = _honest()
    source = np.stack([_asymmetric(seed=s) for s in range(4)])
    swept = []
    for name, (fn, slot) in sorted(_reductions_with_a_positional_out_slot().items()):
        bills = {}
        for spelling in ("keyword", "positional"):
            tagged, alias = _alias_of_a_tagged_buffer()
            if spelling == "keyword":
                call = functools.partial(fn, source, 0, out=alias)
            else:
                call = functools.partial(fn, source, 0, *([None] * slot), alias)
            try:
                bills[spelling] = _billed(call)
            except Exception:
                # Not every discovered name accepts this input or this
                # destination (``argmax`` refuses a float buffer, ``cumsum``
                # keeps the input's shape). Those are pinned elsewhere; the
                # sweep is about the ops that do write.
                break
            assert epoch_of(tagged) == 1, f"{name} ({spelling}) recorded no write"
            assert _probe(tagged) == honest, (
                f"{name} ({spelling}) left asymmetric data billing at the "
                "symmetric rate"
            )
        else:
            assert bills["positional"] == bills["keyword"], (
                f"{name}: positional out= bills {bills['positional']}, keyword "
                f"out= bills {bills['keyword']}"
            )
            swept.append(name)
    # Guards the sweep against silently degenerating to nothing if the
    # discovery or the call form stops matching.
    assert len(swept) >= 15, swept


@pytest.mark.parametrize(
    ("method", "write"),
    [
        (
            "outer",
            lambda dest, src: np.subtract.outer(src[0, 0], src[0, 0], out=dest),
        ),
        ("reduce", lambda dest, src: np.subtract.reduce(src, axis=0, out=dest)),
        (
            "accumulate",
            lambda dest, src: np.subtract.accumulate(src[0], axis=0, out=dest),
        ),
        (
            "at",
            lambda dest, src: np.add.at(
                dest, (np.array([0, 2]), np.array([1, 3])), np.array([5.0, 7.0])
            ),
        ),
    ],
)
def test_ufunc_method_does_not_forge_tag(method, write):
    """``ufunc.outer`` / ``.reduce`` / ``.accumulate`` / ``.at`` called numpy
    directly rather than through ``_call_numpy``, so no write was ever recorded.

    ``.at`` writes into its FIRST argument rather than an ``out=``, so even
    routing it through the shared helper does not cover it -- its destination
    is recorded explicitly. Its refusal of a tagged ``SymmetricTensor`` is not
    protection either: the alias used here is untagged and sails past it.
    """
    load_weights()
    honest = _honest()
    tagged, alias = _alias_of_a_tagged_buffer()
    source = np.stack([_asymmetric(seed=s) for s in range(4)])

    write(alias, source)

    written = np.asarray(tagged)
    assert not np.allclose(written, written.T), (
        f"test bug: {method} did not actually break symmetry"
    )
    assert getattr(tagged, "symmetry", None) is None, (
        f"the symmetry tag survived a write through ufunc.{method}"
    )
    assert _probe(tagged) == honest, (
        f"asymmetric data still bills at the symmetric rate after ufunc.{method}"
    )


def test_ufunc_reduceat_does_not_forge_tag():
    """Separated from the sweep above because ``reduceat``'s output shape is
    set by the segment count, so its destination is a slice of the tagged
    buffer rather than the whole of it -- a different alias route to the same
    root, and the one that has to reach the tag."""
    load_weights()
    honest = _honest()
    tagged = fnp.zeros((N, N))
    rows = fnp.asarray(tagged)[:2]
    source = _asymmetric(seed=1)

    np.subtract.reduceat(source, [0, N // 2], axis=0, out=rows)

    written = np.asarray(tagged)
    assert not np.allclose(written, written.T), (
        "test bug: reduceat did not actually break symmetry"
    )
    assert getattr(tagged, "symmetry", None) is None, (
        "the symmetry tag survived a write through ufunc.reduceat"
    )
    assert _probe(tagged) == honest


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Routes that write the buffer beneath the Python layer
# ---------------------------------------------------------------------------


def test_flat_assignment_does_not_forge_tag():
    """``arr.flat[:] = ...`` is the same category as fill/put/resize.

    A C-level route that writes the buffer without passing through
    ``__setitem__``, and so without the write-epoch hook ever seeing it.
    """
    _assert_claim_does_not_survive(
        fnp.zeros((N, N)),
        lambda z: z.flat.__setitem__(slice(None), list(_asymmetric().ravel())),
    )


def test_setfield_does_not_forge_tag():
    _assert_claim_does_not_survive(
        fnp.zeros((N, N)), lambda z: z.setfield(_asymmetric(), np.float64)
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known gap: the buffer protocol writes beneath the Python layer, so "
        "nothing observes memoryview(tagged)[i, j] = value. Closing it needs "
        "__buffer__ (PEP 688, Python 3.12+, above this package's floor) or "
        "non-writeable tagged buffers, which would break out= destinations. "
        "Pinned strict so that whoever closes it is forced to delete this "
        "marker rather than leave a passing test mislabelled."
    ),
)
def test_memoryview_assignment_does_not_forge_tag():
    _assert_claim_does_not_survive(
        fnp.zeros((N, N)),
        lambda z: memoryview(z).__setitem__((0, 1), 9.0),  # pyright: ignore[reportCallIssue, reportArgumentType]
    )
