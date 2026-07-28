"""``mask_indices`` must be priced like the scan it performs.

numpy's body is ``a = mask_func(ones((n, n)), k); return nonzero(a != 0)``, so
the honest cost is what ``nonzero`` costs on that n x n array. Pricing it off
the returned index count instead made it an arbitrarily cheap substitute for
``nonzero``.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope
import flopscope.numpy as fnp


def billed(fn) -> int:
    with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
        before = ctx.flops_used
        fn()
        return ctx.flops_used - before


def test_mask_indices_costs_what_nonzero_costs_on_the_same_array():
    n = 200
    sparse = np.zeros((n, n), bool)
    sparse[0, :3] = True
    via_mask = billed(lambda: fnp.mask_indices(n, lambda m, k: sparse))
    via_nonzero = billed(lambda: fnp.nonzero(fnp.asarray(sparse)))
    assert via_mask == via_nonzero


def test_mask_indices_costs_what_nonzero_costs_on_a_dense_mask():
    """A sparse mask alone does not pin the invariant: a mask that keeps just
    over half its elements (the default ``triu``/``tril`` case) is exactly
    where a returned-index-count formula and a mask-scan formula diverge.
    Assert parity for that case too, at more than one size.
    """
    for n in (8, 200):
        dense = np.triu(np.ones((n, n), int))
        via_mask = billed(lambda n=n: fnp.mask_indices(n, np.triu))
        via_nonzero = billed(lambda dense=dense: fnp.nonzero(fnp.asarray(dense)))
        assert via_mask == via_nonzero


def test_mask_indices_scales_with_the_probe_not_the_output():
    """A tiny result must not buy a large scan."""
    small = billed(lambda: fnp.mask_indices(50, lambda m, k: np.zeros((50, 50), bool)))
    large = billed(
        lambda: fnp.mask_indices(200, lambda m, k: np.zeros((200, 200), bool))
    )
    assert large > small


def test_mask_indices_floors_at_the_probe_when_mask_func_captures_it():
    """``mask_func`` receives numpy's internal ``ones((n, n), int)`` probe as
    an argument -- it can capture that reference and return something much
    smaller, but the probe was still allocated and handed over. The bill must
    not drop below what scanning that probe would honestly cost, matching
    what ``fnp.nonzero`` bills on the same probe array directly.
    """
    n = 200
    captured: dict = {}

    def harvest(m, k):
        captured["probe"] = m
        return m[:1, :1]

    via_mask = billed(lambda: fnp.mask_indices(n, harvest))
    via_probe_floor = billed(lambda: fnp.nonzero(fnp.asarray(np.ones((n, n), int))))
    assert captured["probe"].shape == (n, n), "sanity: mask_func saw the full probe"
    assert via_mask == via_probe_floor


def test_tri_indices_helpers_are_unchanged():
    """These do not route through the counted mask_indices wrapper, so
    repricing mask_indices must not move them. Pinned to the measured
    pre-change values (n*(n+1)) so a regression here is caught."""
    for n in (10, 100):
        assert billed(lambda n=n: fnp.triu_indices(n)) == n * (n + 1)
        assert billed(lambda n=n: fnp.tril_indices(n)) == n * (n + 1)


def test_n_array_protocol_second_read_does_not_shrink_the_floor():
    """``n`` must be resolved through the integer-index protocol EXACTLY
    ONCE, and that same resolved value must be what both numpy's own probe
    (``ones((n, n), int)``) is built from AND what the billing floor
    (``n*n``) uses.

    numpy's internal body resolves ``n`` via ``__index__`` to build the
    probe; the billing floor used to re-read ``n`` a SECOND time via
    ``int(n)`` -- a DIFFERENT protocol -- after that call had already run.
    An ``n`` that reports a large size to ``__index__`` (what numpy's probe
    is actually built from, and what ``mask_func`` actually receives) and a
    small one to ``__int__`` would let the floor fall far below the size of
    the probe that was genuinely allocated and exposed to ``mask_func``.
    """
    real_n = 300
    fake_n = 1

    class N_:
        def __index__(self):
            return real_n

        def __int__(self):
            return fake_n

    via_stateful = billed(
        lambda: fnp.mask_indices(N_(), lambda m, k: np.asarray(m)[:1, :1])
    )
    via_plain_floor = billed(
        lambda: fnp.mask_indices(real_n, lambda m, k: np.asarray(m)[:1, :1])
    )
    assert via_stateful == via_plain_floor
    assert via_stateful > billed(
        lambda: fnp.mask_indices(fake_n, lambda m, k: np.asarray(m)[:1, :1])
    )


def test_mask_func_ne_override_cannot_smuggle_a_bigger_scan_past_the_bill():
    """``mask_func`` can return an arbitrary ``np.ndarray`` SUBCLASS, not
    just a plain array. numpy's own body is ``nonzero(mask_func(m, k) !=
    0)`` -- the scan happens through whatever ``!=`` the RETURNED object
    implements, not through ``__array__``. A subclass whose ``__array__``
    reports a tiny array (what this op used to measure) but whose ``__ne__``
    returns something unrelated and far larger (what numpy's ``nonzero``
    actually receives and scans) would let a tiny measured mask stand in
    for a large executed one.

    ``fnp.mask_indices`` must forward the SAME array it measured -- not the
    caller's original ``mask_func`` return value -- so numpy's ``!= 0``
    cannot run against a different, unmeasured object. With that fixed, the
    override never runs at all: a plain, subclass-free array has no
    ``__ne__`` to intercept, and both the bill and the returned indices
    reflect the tiny, honestly-measured mask.
    """
    n = 4
    big = 3000

    class Sneaky(np.ndarray):
        def __ne__(self, other):  # noqa: ARG002 -- must match ndarray's signature
            return np.ones((big, big), bool)

    tiny = np.zeros((1, 1), bool).view(Sneaky)

    cost = billed(lambda: fnp.mask_indices(n, lambda m, k: tiny))
    honest_floor = billed(lambda: fnp.mask_indices(n, lambda m, k: np.zeros((n, n))))

    assert cost == honest_floor, (
        "the bill must reflect the probe floor, not the tiny __array__ value "
        "the subclass reports"
    )

    result = fnp.mask_indices(n, lambda m, k: tiny.view(Sneaky))
    total_indices = sum(int(r.size) for r in result)
    assert total_indices == 0, (
        "the executed scan must run against the same (all-zero) array that "
        "was billed, not the __ne__ override's unrelated big result"
    )


# --------------------------------------------------------------------------
# Forwarding the measured array (instead of the caller's original return
# value) must not change the bill for any of the ordinary, honest
# ``mask_func`` return forms: an fnp mask_func, a plain numpy callable, a
# bool mask, and an int mask. Each is billed identically whether the exact
# same values arrive as a plain ``np.ndarray`` or wrapped in a harmless
# (non-overriding) ``np.ndarray`` subclass -- proving the forwarding change
# is a true no-op for values that don't try to smuggle a mismatched scan.
# ``np.asarray`` on a plain ndarray is a no-op view, so this is exactly the
# invariant the fix relies on.
# --------------------------------------------------------------------------


class _PlainSubclass(np.ndarray):
    """An ``np.ndarray`` subclass with no overridden dunders -- forwarding
    ``np.asarray(x)`` instead of ``x`` must be indistinguishable from
    forwarding ``x`` itself for a subclass like this."""


@pytest.mark.parametrize(
    "make_mask",
    [
        pytest.param(lambda n: np.triu(np.ones((n, n), int)), id="int-triu"),
        pytest.param(
            lambda n: np.array([[True, False, True, False]] * n), id="bool-mask"
        ),
        pytest.param(lambda n: np.array([[1, 0, 2, 0]] * n), id="int-mask"),
    ],
)
def test_honest_mask_forms_are_unaffected_by_forwarding_the_measured_array(make_mask):
    n = 4
    mask = make_mask(n)
    plain = billed(lambda: fnp.mask_indices(n, lambda m, k: mask))
    wrapped = billed(
        lambda: fnp.mask_indices(n, lambda m, k: mask.view(_PlainSubclass))
    )
    assert plain == wrapped


def test_plain_np_triu_mask_func_is_unaffected_by_forwarding_the_measured_array():
    n = 4
    assert billed(lambda: fnp.mask_indices(n, np.triu)) == billed(
        lambda: fnp.mask_indices(n, lambda m, k: np.triu(m))
    )


def test_fnp_mask_func_is_unaffected_by_forwarding_the_measured_array():
    """An fnp ``mask_func`` returns a ``FlopscopeArray`` -- already stripped
    to a plain ndarray by ``_to_base_ndarray`` before this fix, so
    ``np.asarray`` on it was already documented as a no-op view. It still
    bills its own cost (the ``fnp.triu`` call itself) on top of the mask
    scan, exactly as before.
    """
    n = 4
    with_fnp_mask_func = billed(
        lambda: fnp.mask_indices(n, lambda m, k: fnp.triu(m, k))
    )
    scan_only = billed(lambda: fnp.mask_indices(n, np.triu))
    triu_cost_alone = billed(lambda: fnp.triu(fnp.asarray(np.ones((n, n), int))))
    assert with_fnp_mask_func == scan_only + triu_cost_alone
