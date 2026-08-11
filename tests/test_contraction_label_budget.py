"""The 52-letter subscript budget: allocation, fallback pricing, invariants."""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._pointwise import _dense_accumulation_cost
from flopscope.errors import CostFallbackWarning


def billed(fn) -> int:
    """Billed FLOPs for `fn`, warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as b:
            fn()
            return b.flops_used


# `tests/conftest.py` has an autouse fixture calling `reset_weights()`, which
# puts the suite in unit-weight mode: every op weight and dtype rate is 1.0, so
# `billed()` returns the raw FLOP cost. Do NOT introduce a dtype-weight divisor
# here — production weights get re-tuned, and a test that hardcodes one would
# break on every re-calibration. Compare raw costs.


TENSORDOT_SHAPES = [
    ((8, 6), (6, 5), ([1], [0])),
    ((4, 3, 5), (5, 7), ([2], [0])),
    ((2, 3, 4), (3, 4, 6), ([1, 2], [0, 1])),
    ((9,), (9,), ([0], [0])),
    ((5, 4), (4,), ([1], [0])),
    ((3, 3, 3), (3, 3, 3), ([2], [0])),
    ((7, 2, 3), (2, 3, 11), ([1, 2], [0, 1])),
    ((6, 1, 4), (4, 1, 6), ([2], [0])),
    ((2, 2), (2, 2), ([0, 1], [0, 1])),
    ((10, 3), (3, 10), ([1], [0])),
    ((1, 5), (5, 1), ([1], [0])),
    ((4, 5, 6), (6,), ([2], [0])),
]


def _geometry(a_shape, b_shape, axes):
    a_ax, b_ax = axes
    contracted = math.prod(a_shape[i] for i in a_ax)
    output_shape = tuple(s for i, s in enumerate(a_shape) if i not in a_ax) + tuple(
        s for j, s in enumerate(b_shape) if j not in b_ax
    )
    return contracted, output_shape


@pytest.mark.parametrize("a_shape,b_shape,axes", TENSORDOT_SHAPES)
def test_dense_cost_matches_einsum_path(a_shape, b_shape, axes):
    """The label-free formula must equal what the einsum path charges."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal(a_shape)
    b = rng.standard_normal(b_shape)
    einsum_path = billed(
        lambda: fnp.tensordot(fnp.asarray(a), fnp.asarray(b), axes=axes)
    )
    contracted, output_shape = _geometry(a_shape, b_shape, axes)
    assert (
        _dense_accumulation_cost(a.size, b.size, contracted, output_shape).total
        == einsum_path
    )


@pytest.mark.parametrize(
    "a_shape,b_shape,axes",
    [
        ((8, 0), (0, 5), ([1], [0])),  # zero-length contracted axis
        ((0, 6), (6, 5), ([1], [0])),  # zero-size output via a
        ((8, 6), (6, 0), ([1], [0])),  # zero-size output via b
    ],
)
def test_empty_domain_bills_zero_never_negative(a_shape, b_shape, axes):
    """A zero-sized contraction bills 0. It must never refund budget."""
    contracted, output_shape = _geometry(a_shape, b_shape, axes)
    a_size = math.prod(a_shape)
    b_size = math.prod(b_shape)
    cost = _dense_accumulation_cost(a_size, b_size, contracted, output_shape)
    assert cost.total == 0
    assert cost.total >= 0


def test_dense_cost_exposes_multiply_add_split():
    """mu is the multiply count, so complex billing can derive its exact ratio."""
    # (8,6)x(6,5): alpha = 8*5*6 = 240 multiplies, M = 40 cells, 200 adds.
    cost = _dense_accumulation_cost(48, 30, 6, (8, 5))
    assert cost.mu == 240
    assert cost.total == 2 * 240 - 40
    assert cost.num_terms == 2
    assert cost.fallback_used is False


from flopscope._pointwise import _contraction_subscripts


def test_subscripts_ties_contracted_pairs():
    """Contracted axis pairs share a label; free axes get distinct ones.

    b's labels start at offset a_ndim, so for two rank-2 operands b is
    initially 'cd'; tying b's axis 0 to a's axis 1 rewrites it to 'bd'.
    """
    assert _contraction_subscripts(2, 2, (1,), (0,)) == "ab,bd->ad"


def test_subscripts_handles_multiple_contracted_axes():
    assert _contraction_subscripts(3, 3, (1, 2), (0, 1)) == "abc,bcf->af"


def test_subscripts_normalises_negative_axes():
    """Negative axis indices mean the same thing as their positive form."""
    assert _contraction_subscripts(2, 2, (-1,), (-2,)) == _contraction_subscripts(
        2, 2, (1,), (0,)
    )


def test_subscripts_returns_none_above_budget():
    """52 letters exist, so a rank sum above 52 has no representation."""
    assert _contraction_subscripts(26, 26, (1,), (0,)) is not None
    assert _contraction_subscripts(27, 26, (1,), (0,)) is None
    assert _contraction_subscripts(27, 27, (1,), (0,)) is None


def test_subscripts_full_contraction_to_scalar():
    """All axes contracted on both sides gives an empty output."""
    assert _contraction_subscripts(3, 3, (0, 1, 2), (0, 1, 2)) == "abc,abc->"


def _pad_end(arr, n):
    """Append n singleton axes via basic indexing — a free, unmetered view."""
    return arr[(slice(None),) * arr.ndim + (None,) * n]


def _pad_front(arr, n):
    return arr[(None,) * n + (slice(None),) * arr.ndim]


@pytest.mark.parametrize("n_pad", [0, 10, 24, 25, 26, 30])
def test_tensordot_padding_does_not_change_bill(n_pad):
    """Singleton axes carry no arithmetic, so they must carry no price.

    Padding to 25+ per operand pushes the rank sum past 52, which is where
    the subscript budget runs out. Rank must not be a discount.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 16))
    b = rng.standard_normal((16, 8))
    baseline = billed(
        lambda: fnp.tensordot(fnp.asarray(a), fnp.asarray(b), axes=([1], [0]))
    )
    padded = billed(
        lambda: fnp.tensordot(
            _pad_end(fnp.asarray(a), n_pad),
            _pad_end(fnp.asarray(b), n_pad),
            axes=([1], [0]),
        )
    )
    assert padded == baseline


def test_tensordot_above_budget_bills_fma_not_multiplies_only():
    """Above the budget the bill is 2*alpha - M, not the multiply count."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 16))
    b = rng.standard_normal((16, 8))
    got = billed(
        lambda: fnp.tensordot(
            _pad_end(fnp.asarray(a), 25),
            _pad_end(fnp.asarray(b), 25),
            axes=([1], [0]),
        )
    )
    alpha, m = 32 * 8 * 16, 32 * 8
    assert got == 2 * alpha - m
    assert got != alpha  # the old multiply-only price


@pytest.mark.parametrize("n_pad", [0, 25])
def test_tensordot_padding_preserves_values(n_pad):
    """The fix must not change what the operation computes."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 16))
    b = rng.standard_normal((16, 8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True):
            got = fnp.tensordot(
                _pad_end(fnp.asarray(a), n_pad),
                _pad_end(fnp.asarray(b), n_pad),
                axes=([1], [0]),
            )
    assert np.allclose(np.squeeze(np.asarray(got)), a @ b)


def test_complex_tensordot_above_budget_bills_exactly():
    """Complex operands must bill, not raise fail-closed, above the budget.

    Pins exactness, not just positivity: the padded (above-budget, dense
    fallback) bill must equal the unpadded (below-budget, einsum) bill for
    the identical contraction, since padding with singleton axes carries no
    arithmetic. Below the budget `accumulation_for_billing` is always the
    einsum path's real `AccumulationCost`, so this also confirms the
    fallback's dense `_dense_accumulation_cost` reproduces the same exact
    complex ratio the einsum path would have charged.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 6)) + 1j * rng.standard_normal((8, 6))
    b = rng.standard_normal((6, 5)) + 1j * rng.standard_normal((6, 5))
    unpadded = billed(
        lambda: fnp.tensordot(fnp.asarray(a), fnp.asarray(b), axes=([1], [0]))
    )
    padded = billed(
        lambda: fnp.tensordot(
            _pad_end(fnp.asarray(a), 25),
            _pad_end(fnp.asarray(b), 25),
            axes=([1], [0]),
        )
    )
    assert unpadded > 0
    assert padded == unpadded


@pytest.mark.parametrize("n_pad", [0, 25])
def test_tensordot_negative_axes_price_same_as_positive(n_pad):
    """Negative axis specs must price identically to the positive spelling.

    Before the fix, the fallback's geometry never normalised negative axes:
    ``contracted = ...`` looped over raw ``a_contract_axes`` and silently
    *skipped* a negative entry (``if 0 <= ax < a.ndim``), collapsing
    ``contracted`` to 1 and massively OVER-billing (dividing by 1 instead of
    the true contracted size). Separately, ``b_surviving = tuple(i for i in
    range(b.ndim) if i not in b_contract_axes)`` never matched a negative
    entry either, so the contracted axis leaked into ``output_shape``,
    inflating ``M`` until ``2*alpha - M`` collapsed to exactly ``alpha`` --
    the old multiply-only price, UNDER-billing. Both directions must now
    match the positive-axis spelling exactly, at n_pad=0 (below the 52-letter
    budget, symmetry-composition path) and n_pad=25 (above it, dense
    fallback path).
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 256))
    b = rng.standard_normal((256, 8))
    a_p = _pad_end(fnp.asarray(a), n_pad)
    b_p = _pad_end(fnp.asarray(b), n_pad)
    positive = billed(lambda: fnp.tensordot(a_p, b_p, axes=([1], [0])))
    neg_b_axis = billed(lambda: fnp.tensordot(a_p, b_p, axes=([1], [-b_p.ndim])))
    neg_a_axis = billed(lambda: fnp.tensordot(a_p, b_p, axes=([-(a_p.ndim - 1)], [0])))
    assert neg_b_axis == positive
    assert neg_a_axis == positive


def test_tensordot_negative_axes_above_budget_bills_honest_fma():
    """Pins the honest above-budget total itself, not just parity.

    Reproduces the exact repro that surfaced the negative-axis bug: a=(32,
    256) and b=(256, 8), each padded to rank 27 (so 27+27=54 > 52 forces the
    dense fallback). ``axes=([1], [-27])`` used to under-bill to the old
    multiply-only price (alpha=65_536, a 0.501 ratio); ``axes=([-26], [0])``
    used to massively OVER-bill (33_488_896, a 256x ratio) because the
    negative a-axis was skipped when computing ``contracted``, dividing by 1
    instead of 256. Both must now equal the honest ``2*alpha - M``.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 256))
    b = rng.standard_normal((256, 8))
    a_p = _pad_end(fnp.asarray(a), 25)  # rank 27
    b_p = _pad_end(fnp.asarray(b), 25)  # rank 27
    alpha, m = 32 * 8 * 256, 32 * 8
    honest = 2 * alpha - m
    assert honest == 130_816  # matches the reported repro's "honest" figure

    positive = billed(lambda: fnp.tensordot(a_p, b_p, axes=([1], [0])))
    under_billed_before_fix = billed(lambda: fnp.tensordot(a_p, b_p, axes=([1], [-27])))
    over_billed_before_fix = billed(lambda: fnp.tensordot(a_p, b_p, axes=([-26], [0])))
    assert positive == honest
    assert under_billed_before_fix == honest
    assert over_billed_before_fix == honest


def test_tensordot_negative_axis_out_of_range_raises_before_charging():
    """An axis still out of range after normalisation must refuse, not charge.

    Matches what ``np.tensordot`` itself raises for the same out-of-range
    axis (it indexes ``shape[axis]`` directly and lets Python's tuple
    indexing fail) -- callers see no behavioural change beyond the timing.
    Refuse-before-charge: no budget may be spent on a call that was always
    going to fail.
    """
    a = np.zeros((3, 4))
    b = np.zeros((4, 5))
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        with pytest.raises(IndexError):
            fnp.tensordot(fnp.asarray(a), fnp.asarray(b), axes=([1], [5]))
        assert ctx.flops_used == 0
        with pytest.raises(IndexError):
            fnp.tensordot(fnp.asarray(a), fnp.asarray(b), axes=([5], [0]))
        assert ctx.flops_used == 0


def test_tensordot_zero_dim_default_axes_raises_index_error_not_zero_division():
    """A 0-d operand with the default ``axes=2`` is a known latent case.

    Real ``np.tensordot`` also raises ``IndexError`` here (``a_ndim - axes``
    goes negative, and indexing a 0-d shape with it is out of range).
    Normalising via ``ax % ndim`` unconditionally would instead raise
    ``ZeroDivisionError`` for ``ndim == 0`` -- the range check must reject
    before the modulo runs.
    """
    a = np.array(5.0)
    b = np.array(6.0)
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        with pytest.raises(IndexError):
            fnp.tensordot(fnp.asarray(a), fnp.asarray(b), axes=2)
        assert ctx.flops_used == 0


def test_tensordot_oversized_symmetry_arm_bills_fma_not_multiplies_only():
    """The oversized-symmetry branch must also charge 2*alpha - M, not alpha.

    Mutation-tested gap: reverting this arm's cost from ``accumulation.total``
    to ``accumulation.mu`` (the old multiply-only price) passed the entire
    suite before this test existed. Reachable via a genuine SymmetricTensor
    whose group is too large to enumerate (S_12, order 479_001_600 >> the
    default ``dimino_budget`` of 50_000): the branch bails on symmetry
    composition (``out_sym = None``) and prices the shape alone. The operand
    is padded to rank 57 (12 real symmetric axes + 44 singleton pad axes + 1
    contracted axis of size 5), pushing the combined rank past the
    52-letter subscript budget too -- so this lands in the oversized-symmetry
    branch's own ``else:`` (dense fallback) arm specifically, not the
    ``_subs is not None`` arm beside it.
    """
    from flopscope._perm_group import SymmetryGroup
    from flopscope._symmetry_utils import wrap_with_symmetry

    group = SymmetryGroup.symmetric(axes=tuple(range(12)))
    a = wrap_with_symmetry(np.ones((2,) * 12 + (1,) * 44 + (5,)), group)
    b = np.ones((5, 3))
    assert a.ndim + b.ndim > 52

    got = billed(lambda: fnp.tensordot(a, b, axes=([56], [0])))

    alpha = a.size * b.size // 5
    m = math.prod(a.shape[:56]) * math.prod(b.shape[1:])
    assert got == 2 * alpha - m
    assert got != alpha  # the old multiply-only (mu-only) price


def test_tensordot_accumulation_for_billing_guard_stays_none_when_symmetry_scales():
    """Mutation-tested gap: an unconditional override passed the whole suite.

    Replacing the ``cost == accumulation.total`` guard with an unconditional
    ``accumulation_for_billing = accumulation`` passed the entire suite
    before this test existed. This pins the guard's two observable
    behaviours directly: real symmetry-free complex billing is unaffected
    (covered by ``test_complex_tensordot_above_budget_bills_exactly``), and
    when a *surviving* (non-oversized) symmetry genuinely scales the cost
    down, the accumulation's mu/total split no longer describes what's
    charged, so the override must stay ``None`` and the existing fail-closed
    ``complex_factor_for`` guard must fire -- not bill a ratio derived from
    the wrong (pre-scaling) total.

    Constructed with a modest S_3 group (order 6, far under budget) on 3 of
    a complex operand's axes, padded past the 52-letter budget so the
    fallback's dense path is in play, with the contracted axis chosen so the
    S_3 symmetry fully survives into the output and actually scales the
    dense cost down (unique elements < dense output size).
    """
    from flopscope._perm_group import SymmetryGroup
    from flopscope._symmetry_utils import wrap_with_symmetry

    group = SymmetryGroup.symmetric(axes=(0, 1, 2))
    assert group.order() < 50_000  # not oversized -- takes the symmetry path
    shape = (3, 3, 3) + (1,) * 47 + (4,)
    rng = np.random.default_rng(0)
    a_vals = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    a = wrap_with_symmetry(a_vals, group)
    b = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))
    assert a.ndim + b.ndim > 52

    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        with pytest.raises(RuntimeError, match="computes its complex cost exactly"):
            fnp.tensordot(a, b, axes=([50], [0]))


@pytest.mark.parametrize("n_pad", [0, 10, 24, 25, 26, 30])
def test_full_inner_tensordot_bills_consistently(n_pad):
    """Contracting every axis: 2*numel - 1, at any rank.

    This path allocated only 26 letters, so above 26 dims it truncated the
    subscript string and leaked a numpy ValueError.
    """
    rng = np.random.default_rng(0)
    base = rng.standard_normal((16, 16))
    ndim = base.ndim + n_pad
    got = billed(
        lambda: fnp.tensordot(
            _pad_end(fnp.asarray(base), n_pad),
            _pad_end(fnp.asarray(base), n_pad),
            axes=ndim,
        )
    )
    assert got == 2 * base.size - 1


def test_full_inner_tensordot_above_budget_is_correct():
    """It must still compute the right number, not just bill one."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((16, 16))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True):
            got = fnp.tensordot(
                _pad_end(fnp.asarray(base), 30),
                _pad_end(fnp.asarray(base), 30),
                axes=32,
            )
    assert np.isclose(float(np.asarray(got)), float((base * base).sum()))


@pytest.mark.parametrize("n_pad", [0, 10, 24, 25, 26, 30])
def test_dot_padding_does_not_change_bill(n_pad):
    """dot contracts trailing axes, so pad at the FRONT to keep them in place."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 16))
    b = rng.standard_normal((16, 8))
    baseline = billed(lambda: fnp.dot(fnp.asarray(a), fnp.asarray(b)))
    padded = billed(
        lambda: fnp.dot(
            _pad_front(fnp.asarray(a), n_pad), _pad_front(fnp.asarray(b), n_pad)
        )
    )
    assert padded == baseline


@pytest.mark.parametrize("n_pad", [0, 10, 24, 25, 26, 30])
def test_inner_padding_does_not_change_bill(n_pad):
    """inner contracts the last axis of both operands; pad at the front."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 16))
    b = rng.standard_normal((8, 16))
    baseline = billed(lambda: fnp.inner(fnp.asarray(a), fnp.asarray(b)))
    padded = billed(
        lambda: fnp.inner(
            _pad_front(fnp.asarray(a), n_pad), _pad_front(fnp.asarray(b), n_pad)
        )
    )
    assert padded == baseline


@pytest.mark.parametrize("op_name", ["dot", "inner"])
def test_dot_inner_above_budget_do_not_raise_stopiteration(op_name):
    """Running out of letters must not surface as a bare StopIteration."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 6))
    b = rng.standard_normal((6, 5)) if op_name == "dot" else rng.standard_normal((5, 6))
    fn = getattr(fnp, op_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as budget:
            got = fn(_pad_front(fnp.asarray(a), 25), _pad_front(fnp.asarray(b), 25))
            assert budget.flops_used > 0
    expected = a @ b if op_name == "dot" else np.inner(a, b)
    assert np.allclose(np.squeeze(np.asarray(got)), expected)


def test_linalg_tensordot_padding_does_not_change_bill():
    """The linalg alias delegates, so it inherits the fix."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 16))
    b = rng.standard_normal((16, 8))
    baseline = billed(
        lambda: fnp.linalg.tensordot(fnp.asarray(a), fnp.asarray(b), axes=([1], [0]))
    )
    padded = billed(
        lambda: fnp.linalg.tensordot(
            _pad_end(fnp.asarray(a), 25), _pad_end(fnp.asarray(b), 25), axes=([1], [0])
        )
    )
    assert padded == baseline


# ---------------------------------------------------------------------------
# Diagnostic warning: fires only when the label-budget fallback actually
# loses precision (symmetry savings or repeated-operand savings forfeited).
# ---------------------------------------------------------------------------


def _warns_cost_fallback(fn) -> bool:
    from flopscope._pointwise import _seen_label_budget

    _seen_label_budget.cache_clear()  # dedup is per-process
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with flops.BudgetContext(flop_budget=10**16, quiet=True):
            fn()
    return any(issubclass(c.category, CostFallbackWarning) for c in caught)


def test_no_warning_when_fallback_is_exact():
    """No symmetry and no aliasing means the fallback price is exact."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 16))
    b = rng.standard_normal((16, 8))
    assert not _warns_cost_fallback(
        lambda: fnp.tensordot(
            _pad_end(fnp.asarray(a), 25), _pad_end(fnp.asarray(b), 25), axes=([1], [0])
        )
    )


def test_warns_when_operands_alias():
    """tensordot(x, x) forfeits repeated-operand savings above the budget."""
    rng = np.random.default_rng(0)
    x = fnp.asarray(rng.standard_normal((16, 16)))
    padded = _pad_end(x, 25)
    assert _warns_cost_fallback(lambda: fnp.tensordot(padded, padded, axes=([1], [0])))


def test_tensordot_warns_with_symmetric_operand_not_aliased():
    """The symmetry half of the warning predicate, not just the alias half.

    Mutation-tested gap: `_pointwise.py:5275`'s predicate is
    ``a_sym is not None or b_sym is not None or a is b``. Only the ``a is b``
    disjunct was covered above (`test_warns_when_operands_alias`); replacing
    the whole predicate with plain ``a is b`` still passed the entire suite.
    Uses a small, non-oversized S_2 symmetry (order 2, far under
    ``dimino_budget``) on axes disjoint from the contracted axis, so it
    survives the contraction and lands in the non-oversized ``else`` branch
    beside ``_symmetry_adjusted_cost`` (:5280) -- not the oversized-symmetry
    arm (:5240, already covered by
    ``test_tensordot_oversized_symmetry_arm_warns``), and not the full-inner
    fast path (:5162, covered separately below).
    """
    from flopscope._perm_group import SymmetryGroup
    from flopscope._symmetry_utils import wrap_with_symmetry

    group = SymmetryGroup.symmetric(axes=(0, 1))
    a = wrap_with_symmetry(np.ones((5, 5) + (1,) * 48 + (3,)), group)
    b = np.ones((3, 4))
    assert a is not b
    assert a.ndim != b.ndim  # not the full-inner path
    assert a.ndim + b.ndim > 52

    assert _warns_cost_fallback(lambda: fnp.tensordot(a, b, axes=([50], [0])))


def test_tensordot_oversized_symmetry_arm_warns():
    """The oversized-symmetry branch always forfeits savings, so it always warns.

    Same construction as
    ``test_tensordot_oversized_symmetry_arm_bills_fma_not_multiplies_only``:
    a genuine SymmetricTensor whose group is too large to enumerate, padded
    past the 52-letter subscript budget too, so this lands in the
    oversized-symmetry branch's dense-fallback arm. That branch only runs
    when at least one operand's symmetry is oversized (non-None), so the
    precision-loss guard is always true there.
    """
    from flopscope._perm_group import SymmetryGroup
    from flopscope._symmetry_utils import wrap_with_symmetry

    group = SymmetryGroup.symmetric(axes=tuple(range(12)))
    a = wrap_with_symmetry(np.ones((2,) * 12 + (1,) * 44 + (5,)), group)
    b = np.ones((5, 3))
    assert a.ndim + b.ndim > 52

    assert _warns_cost_fallback(lambda: fnp.tensordot(a, b, axes=([56], [0])))


def test_full_inner_tensordot_no_warning_when_fallback_is_exact():
    """Full-inner fallback (Task 4's fast path) is exact without symmetry/aliasing."""
    rng = np.random.default_rng(0)
    base_a = rng.standard_normal((16, 16))
    base_b = rng.standard_normal((16, 16))
    ndim = base_a.ndim + 30
    assert not _warns_cost_fallback(
        lambda: fnp.tensordot(
            _pad_end(fnp.asarray(base_a), 30),
            _pad_end(fnp.asarray(base_b), 30),
            axes=ndim,
        )
    )


def test_full_inner_tensordot_warns_when_operands_alias():
    """Full-inner tensordot(x, x) above budget forfeits repeated-operand savings."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((16, 16))
    padded = _pad_end(fnp.asarray(base), 30)
    ndim = base.ndim + 30
    assert _warns_cost_fallback(lambda: fnp.tensordot(padded, padded, axes=ndim))


def test_full_inner_tensordot_warns_with_symmetric_operand_not_aliased():
    """The symmetry half of the warning predicate at the full-inner site.

    Mutation-tested gap: `_pointwise.py:5162`'s predicate is
    ``_symmetry_of(a) is not None or _symmetry_of(b) is not None or a is b``.
    Only the ``a is b`` disjunct was covered above
    (`test_full_inner_tensordot_warns_when_operands_alias`); replacing the
    whole predicate with plain ``a is b`` still passed the entire suite. A
    genuinely symmetric, non-aliased operand contracted over every axis
    (full inner) above the label budget must warn too, even though a scalar
    output leaves no unique-element savings for ``_symmetry_adjusted_cost``
    to realise here -- the predicate fires on the presence of the tag, not
    on whether it changes the price.
    """
    from flopscope._perm_group import SymmetryGroup
    from flopscope._symmetry_utils import wrap_with_symmetry

    group = SymmetryGroup.symmetric(axes=(0, 1))
    shape = (5, 5) + (1,) * 25  # ndim=27; both operands fully contracted -> 54 > 52
    a = wrap_with_symmetry(np.ones(shape), group)
    b = np.ones(shape) * 2.0
    assert a is not b
    assert a.ndim == b.ndim
    assert a.ndim + b.ndim > 52

    assert _warns_cost_fallback(lambda: fnp.tensordot(a, b, axes=a.ndim))


def test_dot_no_warning_when_fallback_is_exact():
    """dot's label-budget fallback (routed through _einsum_routed_binary) is
    exact without symmetry or aliasing."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 16))
    b = rng.standard_normal((16, 16))
    assert not _warns_cost_fallback(
        lambda: fnp.dot(_pad_front(fnp.asarray(a), 25), _pad_front(fnp.asarray(b), 25))
    )


def test_dot_warns_when_operands_alias():
    """dot(x, x) above the label budget forfeits repeated-operand savings."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 16))
    padded = _pad_front(fnp.asarray(x), 25)
    assert _warns_cost_fallback(lambda: fnp.dot(padded, padded))


def test_dot_warns_with_symmetric_operand_not_aliased():
    """The symmetry half of the warning predicate in ``_einsum_routed_binary``.

    Mutation-tested gap: `_pointwise.py:4711`'s predicate is
    ``_symmetry_of(a) is not None or _symmetry_of(b) is not None or a is b``.
    Only the ``a is b`` disjunct was covered above
    (`test_dot_warns_when_operands_alias`); replacing the whole predicate
    with plain ``a is b`` still passed the entire suite. This site is shared
    by ``dot`` and ``inner`` -- exercising it through ``dot`` covers both.

    Padding is split across both operands (27 + 26, each under NumPy's
    32-dimension ceiling for a raw ``np.dot`` call) rather than concentrated
    on one -- unlike ``tensordot``, which reshapes to a 2-D matmul before
    calling into NumPy and tolerates a single operand past 32 dims (see the
    oversized-symmetry test above), ``np.dot``'s own N-D code path enforces
    ``NPY_MAXDIMS`` per operand.
    """
    from flopscope._perm_group import SymmetryGroup
    from flopscope._symmetry_utils import wrap_with_symmetry

    group = SymmetryGroup.symmetric(axes=(0, 1))
    a = wrap_with_symmetry(np.ones((5, 5) + (1,) * 24 + (3,)), group)  # ndim=27
    b = np.ones((1,) * 24 + (3, 4))  # ndim=26; contracted axis is b.shape[-2]
    assert a is not b
    assert a.ndim <= 32 and b.ndim <= 32
    assert a.ndim + b.ndim > 52

    assert _warns_cost_fallback(lambda: fnp.dot(a, b))


# ---------------------------------------------------------------------------
# Surviving output symmetry on the dot/inner label-budget fallback
# ---------------------------------------------------------------------------
#
# ``_einsum_routed_binary``'s ``subs is None`` arm used to hardcode
# ``output_symmetry = None``, so a symmetric operand paid the full dense price
# above the 52-letter budget while the identical contraction below it paid the
# symmetry-adjusted one -- rank acting as a surcharge. The arm now composes the
# surviving group the way ``tensordot``'s non-oversized arm already did.
#
# NumPy caps ``dot``/``inner`` at 32 dimensions PER OPERAND, so the budget can
# only be exceeded by splitting rank across both (27 + 27 below). That also
# rules out reaching the fallback through ``dot``'s 1-D-``b`` arm, which would
# need ``a.ndim >= 52``; that arm's axis pair is covered by construction, not
# by a test.


def _sym_ones(shape, axes):
    """Constant array tagged with a genuine symmetry (constants are symmetric)."""
    from flopscope._perm_group import SymmetryGroup
    from flopscope._symmetry_utils import wrap_with_symmetry

    return wrap_with_symmetry(np.ones(shape), SymmetryGroup.symmetric(axes=axes))


# op, big a shape, big b shape, small a shape, small b shape. The small pair is
# the same contraction with the singleton padding removed, i.e. below the
# letter budget and therefore priced by the einsum path.
SYMMETRIC_FALLBACK_CASES = [
    ("inner", (4, 4) + (1,) * 25, (4,) + (1,) * 26, (4, 4, 1), (4, 1)),
    ("dot", (4, 4) + (1,) * 25, (1,) * 26 + (4,), (4, 4, 1), (1, 4)),
]


@pytest.mark.parametrize(
    "op_name,a_big,b_big,a_small,b_small", SYMMETRIC_FALLBACK_CASES
)
def test_symmetric_operand_above_budget_bills_like_einsum_path(
    op_name, a_big, b_big, a_small, b_small
):
    """Padding a symmetric operand past the letter budget must not raise the bill.

    Singleton axes carry no arithmetic, so the padded (fallback) contraction
    and the unpadded (einsum) one are the same work and must cost the same.
    Before the fix the fallback discarded the surviving symmetry and charged
    the dense price instead.
    """
    fn = getattr(fnp, op_name)
    a_p, b_p = _sym_ones(a_big, (0, 1)), fnp.asarray(np.ones(b_big))
    assert a_p.ndim + b_p.ndim > 52
    above = billed(lambda: fn(a_p, b_p))

    a_u, b_u = _sym_ones(a_small, (0, 1)), fnp.asarray(np.ones(b_small))
    assert a_u.ndim + b_u.ndim <= 52
    below = billed(lambda: fn(a_u, b_u))

    assert above == below


@pytest.mark.parametrize(
    "op_name,a_big,b_big,a_small,b_small", SYMMETRIC_FALLBACK_CASES
)
def test_symmetric_operand_above_budget_keeps_symmetry_on_the_result(
    op_name, a_big, b_big, a_small, b_small
):
    """The surviving group is composed, so the result comes back tagged.

    The fallback previously returned a plain array, silently dropping a
    symmetry the einsum path preserves for the identical contraction.
    """
    from flopscope._symmetric import SymmetricTensor

    fn = getattr(fnp, op_name)
    a_p, b_p = _sym_ones(a_big, (0, 1)), fnp.asarray(np.ones(b_big))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True):
            got = fn(a_p, b_p)
    assert isinstance(got, SymmetricTensor)
    # The two symmetric axes of `a` lead the output, so they keep slots 0/1.
    assert got.symmetry is not None
    assert got.symmetry.axes == (0, 1)
    assert got.symmetry.order() == 2


@pytest.mark.parametrize(
    "op_name,a_big,b_big,a_small,b_small", SYMMETRIC_FALLBACK_CASES
)
def test_untagged_operand_above_budget_still_pays_the_dense_price(
    op_name, a_big, b_big, a_small, b_small
):
    """Only a real symmetry tag buys the adjustment; bare data pays 2*alpha - M.

    The guard against the fix becoming a discount: identical shapes and
    identical values, differing only in whether the symmetry was declared.
    """
    fn = getattr(fnp, op_name)
    plain_a, plain_b = fnp.asarray(np.ones(a_big)), fnp.asarray(np.ones(b_big))
    dense = billed(lambda: fn(plain_a, plain_b))

    a_size, b_size = math.prod(a_big), math.prod(b_big)
    contracted = a_big[-1]
    alpha = a_size * b_size // contracted
    m = math.prod(a_big[:-1]) * (
        math.prod(b_big[:-1]) if op_name == "inner" else math.prod(b_big) // b_big[-2]
    )
    assert dense == 2 * alpha - m

    tagged = billed(lambda: fn(_sym_ones(a_big, (0, 1)), plain_b))
    assert tagged < dense  # the symmetry genuinely scales this contraction


# ---------------------------------------------------------------------------
# Every SYMMETRIC_FALLBACK_CASES/RIGHT_OPERAND_CASES entry above puts the
# size-1 pad in contracted position, so K=1 and nothing is ever actually
# summed. The two clusters below put a genuine (size-3) extent there instead,
# pinning the properties the reviewer checked out of band: a real contraction
# still gets the composed-symmetry discount, and a tag that only covers the
# axis about to be contracted away buys nothing.
# ---------------------------------------------------------------------------

# op, big a shape, big b shape, small a shape, small b shape. The real
# contracted extent (3) sits last on both operands -- where `inner` contracts
# and where `dot` contracts a's last against b's -2 -- with K=3, not 1. The
# small pair is the same contraction with the singleton padding removed.
REAL_CONTRACTION_CASES = [
    ("inner", (4, 4) + (1,) * 24 + (3,), (5,) + (1,) * 25 + (3,), (4, 4, 3), (5, 3)),
    (
        "dot",
        (4, 4) + (1,) * 24 + (3,),
        (5,) + (1,) * 24 + (3, 1),
        (4, 4, 3),
        (5, 3, 1),
    ),
]


@pytest.mark.parametrize("op_name,a_big,b_big,a_small,b_small", REAL_CONTRACTION_CASES)
def test_symmetric_operand_above_budget_with_real_contraction_matches_einsum_path(
    op_name, a_big, b_big, a_small, b_small
):
    """Same property as ``test_symmetric_operand_above_budget_bills_like_einsum_path``,
    but with K=3 actually summed instead of K=1.

    Asserts both that the fallback matches the identical below-budget
    contraction, and that it is strictly cheaper than the same shapes with no
    symmetry tag -- so the test cannot pass if the discount silently stops
    applying (it would still equal a *different*, undiscounted, `below`).
    """
    fn = getattr(fnp, op_name)
    a_p, b_p = _sym_ones(a_big, (0, 1)), fnp.asarray(np.ones(b_big))
    assert a_p.ndim + b_p.ndim > 52
    above = billed(lambda: fn(a_p, b_p))

    a_u, b_u = _sym_ones(a_small, (0, 1)), fnp.asarray(np.ones(b_small))
    assert a_u.ndim + b_u.ndim <= 52
    below = billed(lambda: fn(a_u, b_u))
    assert above == below

    plain_a = fnp.asarray(np.ones(a_big))
    undiscounted = billed(lambda: fn(plain_a, b_p))
    assert above < undiscounted  # otherwise the discount could be gone


# op, big a shape (symmetric pair spans the contracted axis), big b shape
# with a matching contracted extent. `a`'s tag is axes (0, a.ndim - 1): the
# last axis is always the one both `dot` and `inner` contract, so once it's
# contracted away only axis 0 still carries the group -- fewer than 2
# surviving axes, the case `_surviving_symmetry_after_contraction` returns
# `None` for.
SYMMETRY_ON_CONTRACTED_AXIS_CASES = [
    ("inner", (3,) + (1,) * 25 + (3,), (5,) + (1,) * 25 + (3,)),
    ("dot", (3,) + (1,) * 25 + (3,), (5,) + (1,) * 24 + (3, 1)),
]


@pytest.mark.parametrize("op_name,a_big,b_big", SYMMETRY_ON_CONTRACTED_AXIS_CASES)
def test_symmetry_spanning_the_contracted_axis_buys_no_discount(op_name, a_big, b_big):
    """A symmetry tag that only covers the axis being contracted away is worthless.

    Once the last axis (the contracted one) is gone, the tagged pair has
    fewer than 2 surviving axes, so the composition bails to `None` and the
    fallback must fall through to the same full dense price an untagged
    operand of the identical shape pays -- this is the guard that stops an
    unearned discount, not just a case the discount happens not to reach.
    """
    fn = getattr(fnp, op_name)
    tagged = _sym_ones(a_big, (0, len(a_big) - 1))
    b_p = fnp.asarray(np.ones(b_big))
    assert tagged.ndim + b_p.ndim > 52
    with_tag = billed(lambda: fn(tagged, b_p))

    plain_a = fnp.asarray(np.ones(a_big))
    without_tag = billed(lambda: fn(plain_a, b_p))

    assert with_tag == without_tag  # the tag bought nothing


# op, plain a shape, symmetric b shape, plain small a, symmetric small b.
# `b`'s symmetric pair is axes (0, 1) in both; the contracted axis is a
# singleton so both survive into the output.
RIGHT_OPERAND_CASES = [
    ("inner", (4,) + (1,) * 26, (4, 4) + (1,) * 25, (4, 1), (4, 4, 1)),
    ("dot", (5,) + (1,) * 26, (4, 4) + (1,) * 25, (5, 1), (4, 4, 1, 1)),
]


@pytest.mark.parametrize("op_name,a_big,b_big,a_small,b_small", RIGHT_OPERAND_CASES)
def test_right_operand_symmetry_lands_in_the_offset_output_slots(
    op_name, a_big, b_big, a_small, b_small
):
    """``b``'s surviving axes are relabelled past ``a``'s, not from zero.

    Mutation-tested gap: with the symmetry only ever on ``a``, ``b_offset``
    is dead -- zeroing it passed every other test here. The output
    concatenates ``a``'s surviving axes then ``b``'s, so ``b``'s symmetric
    pair belongs at slots ``len(a_surviving) + 0/1``; landing it at 0/1
    instead would claim a symmetry between two axes that are not exchangeable
    (different operands, and here different lengths) and misprice on shapes
    where those slots differ in size.
    """
    fn = getattr(fnp, op_name)
    a_p, b_p = fnp.asarray(np.ones(a_big)), _sym_ones(b_big, (0, 1))
    assert a_p.ndim + b_p.ndim > 52
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as budget:
            result = fn(a_p, b_p)
            above = budget.flops_used

    a_u, b_u = fnp.asarray(np.ones(a_small)), _sym_ones(b_small, (0, 1))
    assert a_u.ndim + b_u.ndim <= 52
    below = billed(lambda: fn(a_u, b_u))
    assert above == below

    # a contributes len(a_big) - 1 surviving axes, so b's pair starts there.
    offset = len(a_big) - 1
    assert result.symmetry is not None
    assert result.symmetry.axes == (offset, offset + 1)


def test_oversized_symmetry_above_budget_bills_dense_without_raising():
    """An unenumerable group must degrade to the dense price, not crash.

    ``dot``/``inner`` are exactly the two ops this branch stopped crashing, so
    the symmetry composition must not reintroduce a crash class.
    ``_is_oversized_for_cost_model`` is the pre-guard: S_12 (order 479_001_600)
    is far past the default ``dimino_budget`` of 50_000. Both operands stay
    under NumPy's 32-dimension ceiling while their sum clears 52.

    The result type is what pins the pre-guard, not the price. Composing an
    oversized group anyway still lands on the dense price -- ``_symmetry_
    adjusted_cost`` -> ``unique_elements_for_shape`` has its own budget guard
    and degrades to the dense element count -- but it would stamp the result
    with a group nothing downstream can enumerate, which is the opposite of
    what ``tensordot``'s oversized arm does (``out_sym = None``). Bailing
    before composition keeps the two ops' answers to the same question the
    same, so this asserts on the wrapper, which is the only place the two
    spellings differ.
    """
    from flopscope._perm_group import SymmetryGroup
    from flopscope._symmetric import SymmetricTensor
    from flopscope._symmetry_utils import wrap_with_symmetry

    group = SymmetryGroup.symmetric(axes=tuple(range(12)))
    a = wrap_with_symmetry(np.ones((2,) * 12 + (1,) * 19 + (5,)), group)  # ndim 32
    b = np.ones((1,) * 21 + (5,))  # ndim 22
    assert a.ndim <= 32 and b.ndim <= 32
    assert a.ndim + b.ndim > 52

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as budget:
            result = fnp.inner(a, b)  # must not raise
            got = budget.flops_used

    alpha = a.size * b.size // 5
    m = math.prod(a.shape[:-1]) * math.prod(b.shape[:-1])
    assert got == 2 * alpha - m
    assert not isinstance(result, SymmetricTensor)


@pytest.mark.parametrize(
    "op_name,a_big,b_big,a_small,b_small", SYMMETRIC_FALLBACK_CASES
)
def test_dimino_budget_exceeded_mid_composition_falls_back_to_dense(
    op_name, a_big, b_big, a_small, b_small, monkeypatch
):
    """The try/except behind the oversized pre-guard, exercised directly.

    ``_is_oversized_for_cost_model`` screens each operand's group in
    isolation; restriction, remapping and the direct product can still blow
    ``dimino_budget`` mid-composition. Anything escaping there would land on
    the caller as a bare private exception, so the arm swallows it and charges
    dense. Constructing a group that trips it only during composition is
    fiddly, so the raise is injected at the first composition step.
    """
    import flopscope._pointwise as pw
    from flopscope._perm_group import _DiminoBudgetExceeded

    def _boom(group, surviving_axes):
        raise _DiminoBudgetExceeded(99_999, 50_000)

    monkeypatch.setattr(pw, "_surviving_symmetry_after_contraction", _boom)

    fn = getattr(fnp, op_name)
    a = _sym_ones(a_big, (0, 1))
    b = fnp.asarray(np.ones(b_big))

    got = billed(lambda: fn(a, b))  # must not raise

    alpha = math.prod(a_big) * math.prod(b_big) // a_big[-1]
    m = math.prod(a_big[:-1]) * (
        math.prod(b_big[:-1]) if op_name == "inner" else math.prod(b_big) // b_big[-2]
    )
    assert got == 2 * alpha - m


@pytest.mark.parametrize(
    "op_name,a_big,b_big,a_small,b_small", SYMMETRIC_FALLBACK_CASES
)
def test_complex_above_budget_bills_exactly_when_symmetry_is_a_no_op(
    op_name, a_big, b_big, a_small, b_small
):
    """No symmetry to adjust for => the exact complex override still applies.

    The `cost == accumulation.total` guard must not cost untagged complex
    operands their exact ratio: the padded bill has to match the unpadded one,
    not fail closed.
    """
    fn = getattr(fnp, op_name)
    rng = np.random.default_rng(0)

    def _cplx(shape):
        return fnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    above = billed(lambda: fn(_cplx(a_big), _cplx(b_big)))
    below = billed(lambda: fn(_cplx(a_small), _cplx(b_small)))
    assert below > 0
    assert above == below


def test_complex_above_budget_fails_closed_when_symmetry_scales_the_cost():
    """The other half of the guard: a scaled cost must not claim an exact ratio.

    Once symmetry scales `cost` below `accumulation.total`, the accumulation's
    mu/total split no longer describes what is being charged, so the override
    goes `None` and ``complex_factor_for``'s fail-closed check must fire rather
    than bill a ratio derived from the pre-scaling total.
    """
    from flopscope._perm_group import SymmetryGroup
    from flopscope._symmetry_utils import wrap_with_symmetry

    rng = np.random.default_rng(0)
    v = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    sym = v + v.T  # genuinely symmetric complex data
    a = wrap_with_symmetry(
        sym.reshape((4, 4) + (1,) * 25), SymmetryGroup.symmetric(axes=(0, 1))
    )
    b_vals = rng.standard_normal((4,) + (1,) * 26)
    b = b_vals + 1j * b_vals
    assert a.ndim + b.ndim > 52

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True):
            with pytest.raises(RuntimeError, match="computes its complex cost exactly"):
                fnp.inner(a, b)


# ---------------------------------------------------------------------------
# Mis-paired contracted axes: refuse before charging
# ---------------------------------------------------------------------------
#
# `budget.deduct` charges on entry and never rolls back when the wrapped numpy
# call raises, so a contraction numpy is going to reject must be refused BEFORE
# its cost is computed. Neither pricing path did that on its own: above the
# 52-letter budget the arithmetic fallback prices from shapes and never looks
# at the pairing at all, and below it the einsum route only appears to
# validate -- `_build_size_map` rejects two different label sizes, but einsum
# BROADCASTS an extent of 1, so `ij,jk->ik` happily priced `j=1` against `j=7`
# and left numpy to reject the call afterwards.
#
# Measured against numpy 2.2.6 (see the module note on the support matrix):
# `np.dot`, `np.inner` and `np.tensordot` all require contracted extents to
# match EXACTLY. None of the three broadcasts a size-1 contracted axis, and
# every unequal pair -- 1 against n, 0 against n -- is a ValueError. Equal
# extents are always accepted, INCLUDING 0 against 0, which yields a zero
# fill. Assertions here are on exception TYPE and on `flops_used`, never on
# message text, which differs per op and across the numpy support matrix.


def _raises_billing(fn, exc: type[Exception] = ValueError) -> int:
    """Assert `fn` raises `exc`, and return what it billed before doing so.

    `exc` is annotated rather than left to inference: the default would
    otherwise narrow the parameter to `type[ValueError]` and reject the
    `IndexError` call below.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as b:
            with pytest.raises(exc):
                fn()
            return b.flops_used


# The contracted pair has to survive the padding, so these build the padded
# shape directly instead of reusing `_pad_end` (which would push the contracted
# axis inward and leave a singleton pad axis contracted instead).
#
# `inner` and the `axes=([-1], [-1])` tensordot spec contract the last axis of
# both operands; `dot` contracts a's last with b's last-but-one, so b's padding
# goes in front. numpy caps dot/inner at 32 dimensions per operand, so `PAD=25`
# splits rank across both to reach 27 + 27 = 54 > 52.
PAD = 25
MISPAIRED_OPS = [("dot", None), ("inner", None), ("tensordot", ([-1], [-1]))]


def _operands(op_name, k, j, pad):
    """(a, b) contracting extent `k` of a against extent `j` of b."""
    a_shape = (2,) + (1,) * pad + (k,)
    b_shape = (1,) * pad + (j, 2) if op_name == "dot" else (2,) + (1,) * pad + (j,)
    return np.ones(a_shape), np.ones(b_shape)


def _honest_cost(op_name, k, pad):
    """The FMA=2 dense price `2*alpha - M` for `_operands(op_name, k, k, pad)`."""
    a, b = _operands(op_name, k, k, pad)
    if k == 0:
        return 0  # empty arithmetic domain: no multiply or accumulation events
    alpha = a.size * b.size // k
    return 2 * alpha - (a.size // k) * (b.size // k)


@pytest.mark.parametrize("op_name,axes", MISPAIRED_OPS)
@pytest.mark.parametrize("pad", [0, PAD])
def test_extent_mismatch_refuses_before_charging(op_name, axes, pad):
    """A genuine extent mismatch must raise, and bill nothing, on both sides.

    3 against 7 is unambiguous: no broadcasting rule in numpy makes this
    contraction legal. Before the fix the above-budget spelling priced it and
    entered `budget.deduct`, so an impossible call consumed budget (and could
    raise `BudgetExhaustedError` in place of numpy's shape error).
    """
    fn = getattr(fnp, op_name)
    kwargs = {"axes": axes} if axes is not None else {}
    a, b = _operands(op_name, 3, 7, pad)
    with pytest.raises(ValueError):  # ground truth: plain numpy refuses it
        getattr(np, op_name)(a, b, **kwargs)
    assert (a.ndim + b.ndim > 52) is (pad == PAD)

    a, b = fnp.asarray(a), fnp.asarray(b)
    assert _raises_billing(lambda: fn(a, b, **kwargs)) == 0


@pytest.mark.parametrize("op_name,axes", MISPAIRED_OPS)
@pytest.mark.parametrize("pad", [0, PAD])
def test_size_one_contracted_axis_refuses_before_charging(op_name, axes, pad):
    """numpy does NOT broadcast a size-1 contracted axis, so neither may we.

    This is the shape of the mismatch the einsum route used to price: einsum
    broadcasts an extent of 1, so below the budget `fnp.inner` on (2,3) against
    (2,1) billed the whole contraction and only then hit numpy's "not aligned".
    """
    fn = getattr(fnp, op_name)
    kwargs = {"axes": axes} if axes is not None else {}
    a, b = _operands(op_name, 3, 1, pad)
    with pytest.raises(ValueError):  # ground truth: plain numpy refuses it
        getattr(np, op_name)(a, b, **kwargs)

    a, b = fnp.asarray(a), fnp.asarray(b)
    assert _raises_billing(lambda: fn(a, b, **kwargs)) == 0


@pytest.mark.parametrize("op_name,axes", MISPAIRED_OPS)
def test_valid_contraction_bill_is_unchanged_either_side_of_the_budget(op_name, axes):
    """The regression an over-rejecting validator would cause: pin the price.

    Turning a working call into a hard failure is worse than the over-charge
    being fixed, so this pins both that the valid contraction still runs and
    the exact FLOP figure it costs -- 20, unchanged above and below the budget
    and equal to the honest `2*alpha - M`.
    """
    fn = getattr(fnp, op_name)
    kwargs = {"axes": axes} if axes is not None else {}

    a, b = _operands(op_name, 3, 3, 0)
    below = billed(lambda: fn(fnp.asarray(a), fnp.asarray(b), **kwargs))
    a, b = _operands(op_name, 3, 3, PAD)
    assert a.ndim + b.ndim > 52
    above = billed(lambda: fn(fnp.asarray(a), fnp.asarray(b), **kwargs))

    assert below == _honest_cost(op_name, 3, 0) == 20
    assert above == below


@pytest.mark.parametrize("op_name,axes", MISPAIRED_OPS)
@pytest.mark.parametrize("pad", [0, PAD])
def test_zero_length_contracted_axis_is_accepted_and_priced_the_same(
    op_name, axes, pad
):
    """0 against 0 is legal in numpy, so the validator must let it through.

    The empty contraction has no arithmetic domain, so it costs 0 -- the same
    figure on both sides of the letter budget (`_dense_accumulation_cost`
    already guards `2*alpha - M` from going negative here). This is the
    over-rejection guard for the zero case specifically: a validator written
    as "reject unless both extents are positive" would break it, and so would
    one that treated 0 as a wildcard the way einsum treats 1.
    """
    fn = getattr(fnp, op_name)
    kwargs = {"axes": axes} if axes is not None else {}
    a, b = _operands(op_name, 0, 0, pad)
    expected_shape = getattr(np, op_name)(a, b, **kwargs).shape  # numpy accepts it
    assert (a.ndim + b.ndim > 52) is (pad == PAD)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as budget:
            result = fn(fnp.asarray(a), fnp.asarray(b), **kwargs)
            assert budget.flops_used == _honest_cost(op_name, 0, pad) == 0
    assert np.asarray(result).shape == expected_shape


def test_mispaired_refusal_precedes_the_complex_fail_closed_guard():
    """The refusal must also beat the *other* thing `deduct` can raise.

    `complex_factor_for`'s fail-closed "exact" guard fires from inside
    `deduct`, so a check that slipped past the deduct boundary would report
    that RuntimeError for a contraction whose real problem is its shapes.
    The control here pins that these exact operands DO reach that guard when
    their extents match (symmetry scales the cost, so no exact complex ratio
    is claimable -- see
    `test_complex_above_budget_fails_closed_when_symmetry_scales_the_cost`),
    which is what makes the mis-paired assertion below meaningful rather
    than vacuous.
    """
    from flopscope._perm_group import SymmetryGroup
    from flopscope._symmetry_utils import wrap_with_symmetry

    rng = np.random.default_rng(0)
    v = rng.standard_normal((4, 4, 3)) + 1j * rng.standard_normal((4, 4, 3))
    sym = v + v.transpose(1, 0, 2)  # genuinely symmetric in axes 0 and 1
    a = wrap_with_symmetry(
        sym.reshape((4, 4) + (1,) * 24 + (3,)), SymmetryGroup.symmetric(axes=(0, 1))
    )

    def _complex_b(k):
        vals = rng.standard_normal((4,) + (1,) * 25 + (k,))
        return vals + 1j * vals

    assert a.ndim + _complex_b(3).ndim > 52
    # Control: matching extents reach the pricing guard and fail closed there.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True):
            with pytest.raises(RuntimeError, match="computes its complex cost exactly"):
                fnp.inner(a, _complex_b(3))

    assert _raises_billing(lambda: fnp.inner(a, _complex_b(7))) == 0


def test_tensordot_axis_count_mismatch_refuses_before_charging():
    """Pairing k axes of `a` with j != k of `b` is numpy's own ValueError.

    `np.tensordot` compares `len(axes_a)` to `len(axes_b)` BEFORE it indexes
    either shape, so the count check has to sit ahead of the out-of-range
    normalisation to keep the same exception for a spec that is both
    mis-paired and out of range.
    """
    a, b = fnp.asarray(np.ones((3, 4))), fnp.asarray(np.ones((4, 5)))
    assert _raises_billing(lambda: fnp.tensordot(a, b, axes=([1], [0, 1]))) == 0
    assert _raises_billing(lambda: fnp.tensordot(a, b, axes=([0, 1], [0]))) == 0
    # Mis-paired AND out of range: numpy reports the count, not the index.
    assert _raises_billing(lambda: fnp.tensordot(a, b, axes=([1], [0, 99]))) == 0
    # Same count, out of range: still the IndexError this branch already had.
    assert (
        _raises_billing(lambda: fnp.tensordot(a, b, axes=([1], [99])), IndexError) == 0
    )


def test_tensordot_full_inner_extent_mismatch_refuses_before_charging():
    """The full-inner arm prices `2*numel - 1` straight from `a.size`.

    Above the letter budget it never builds a subscript string, so nothing
    else would have caught operands of equal rank but different extents.
    """
    a = fnp.asarray(np.ones((3,) + (1,) * 26))
    b = fnp.asarray(np.ones((7,) + (1,) * 26))
    assert a.ndim + b.ndim > 52
    assert _raises_billing(lambda: fnp.tensordot(a, b, axes=a.ndim)) == 0


def test_tensordot_oversized_symmetry_arm_extent_mismatch_refuses_before_charging():
    """The third pricing arm: an operand whose group is too large to enumerate.

    Mirrors `test_tensordot_oversized_symmetry_arm_bills_fma_not_multiplies_only`
    (S_12, order 479_001_600, far past the default `dimino_budget` of 50_000,
    padded past the 52-letter budget) but with the contracted extents made
    unequal. That arm computes its own dense price and would otherwise charge
    for a contraction numpy immediately refuses.
    """
    from flopscope._perm_group import SymmetryGroup
    from flopscope._symmetry_utils import wrap_with_symmetry

    group = SymmetryGroup.symmetric(axes=tuple(range(12)))
    a = wrap_with_symmetry(np.ones((2,) * 12 + (1,) * 44 + (5,)), group)
    b = fnp.asarray(np.ones((7, 3)))
    assert a.ndim + b.ndim > 52

    assert _raises_billing(lambda: fnp.tensordot(a, b, axes=([56], [0]))) == 0


def test_dot_inner_scalar_operand_behaviour_is_untouched():
    """A 0-d operand has no contracted axis, so the validator must skip it.

    `np.dot`/`np.inner` treat a scalar operand as a plain multiply. flopscope
    has a separate, pre-existing gap there (the subscript builder divides by
    `a.ndim`), and this pins that the pairing check neither fixes nor
    aggravates it -- it must not be the thing that raises.
    """
    scalar = fnp.asarray(np.array(2.0))
    arr = fnp.asarray(np.ones((3, 4)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True):
            with pytest.raises(ZeroDivisionError):
                fnp.inner(scalar, arr)
            with pytest.raises(ZeroDivisionError):
                fnp.dot(scalar, arr)


# ---------------------------------------------------------------------------
# Zero-sized contraction under a surviving symmetry. `_dense_accumulation_cost`
# returns 0 for a zero-length contracted axis, but `_symmetry_adjusted_cost`
# used to floor its symmetry branch at 1 -- charging a call that performed no
# arithmetic, and (for complex operands) desynchronising the charge from
# `accumulation.total` so the exact complex factor was withheld and
# `complex_factor_for`'s fail-closed guard raised.
# ---------------------------------------------------------------------------


def _symmetric_contraction(op_name, dtype, pad, k):
    """A `dot`/`inner`/`tensordot` call with a surviving S_2 output symmetry.

    `k` is the contracted extent (0 for the empty-domain cases). `pad` sets
    how many leading singleton axes each operand carries: 0 keeps the
    combined rank inside the 52-letter budget (einsum path), 24 puts both
    operands at rank 27 -- past the budget when combined, so the label-free
    fallback prices it, yet still inside numpy's own 32-dimension limit on
    an *input* array, so the call really executes.

    The S_2 group sits on the two extent-5 axes, which neither op contracts,
    so it survives into the output and its unique/dense ratio is 45/75 < 1 --
    i.e. `_symmetry_adjusted_cost` genuinely scales here rather than taking
    one of its no-op early returns. That the ratio bites on exactly this
    construction is pinned by
    `test_symmetry_discount_survives_on_nonempty_contraction` below.
    """
    from flopscope._perm_group import SymmetryGroup
    from flopscope._symmetry_utils import wrap_with_symmetry

    group = SymmetryGroup.symmetric(axes=(pad, pad + 1))
    a = wrap_with_symmetry(np.ones((1,) * pad + (5, 5, k), dtype=dtype), group)
    if op_name == "inner":
        b = np.ones((1,) * (pad + 1) + (3, k), dtype=dtype)
        return a, b, lambda: fnp.inner(a, b)
    b = np.ones((1,) * (pad + 1) + (k, 3), dtype=dtype)
    if op_name == "dot":
        return a, b, lambda: fnp.dot(a, b)
    return a, b, lambda: fnp.tensordot(a, b, axes=([a.ndim - 1], [b.ndim - 2]))


@pytest.mark.parametrize("op_name", ["dot", "inner", "tensordot"])
@pytest.mark.parametrize("pad", [0, 24])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_empty_contraction_with_surviving_symmetry_bills_zero(op_name, pad, dtype):
    """K = 0 performs no arithmetic, so it costs 0 on either side of the budget.

    Below the budget the einsum path already charged 0. Above it, the
    fallback's `_dense_accumulation_cost` also returned 0, but the symmetry
    scaler floored that to 1 -- so the same valid call cost 1 purely because
    the operands were too high-rank for a subscript string. The complex case
    is the sharper half: the floored charge no longer matched
    `accumulation.total`, the call site withheld its exact
    `complex_factor_override`, and a call that worked below the budget raised
    `RuntimeError` above it.
    """
    a, b, call = _symmetric_contraction(op_name, dtype, pad, 0)
    assert (a.ndim + b.ndim > 52) is (pad == 24)
    assert billed(call) == 0


@pytest.mark.parametrize("op_name", ["dot", "inner", "tensordot"])
def test_symmetry_discount_survives_on_nonempty_contraction(op_name):
    """The zero-preservation must not disturb a real symmetric contraction.

    Same construction as the empty cases above, with the contracted extent
    at 4 instead of 0, above the letter budget so the symmetry scaler is in
    play. `alpha = 100 * 12 // 4 = 300` multiplies over `M = 5*5*3 = 75`
    output cells gives a dense `2*alpha - M = 525`; the surviving S_2 leaves
    `15 * 3 = 45` unique cells of those 75, so the charge is `525 * 45 // 75`.
    A fix that dropped the ratio (or the floor with it) would show up here.
    """
    a, b, call = _symmetric_contraction(op_name, np.float64, 24, 4)
    assert a.ndim + b.ndim > 52
    dense = _dense_accumulation_cost(a.size, b.size, 4, (5, 5, 3)).total
    assert dense == 525
    got = billed(call)
    assert got == dense * 45 // 75 == 315
    assert got < dense  # the discount is real, not a rounding artefact


def test_symmetry_adjusted_cost_preserves_zero_but_keeps_the_floor():
    """The scaler's two boundary behaviours, pinned directly.

    Zero in, zero out: no arithmetic happened, so no ratio can conjure a
    charge. Non-zero in, at least 1 out: real work whose scaled charge would
    round down to nothing still costs something. `(5, 5)` under S_2 has 15
    unique cells of 25, so `1 * 15 // 25 == 0` before the floor applies.
    """
    from flopscope._perm_group import SymmetryGroup
    from flopscope._pointwise import _symmetry_adjusted_cost

    group = SymmetryGroup.symmetric(axes=(0, 1))
    assert _symmetry_adjusted_cost(0, (5, 5), group) == 0
    assert _symmetry_adjusted_cost(0, (5, 5), None) == 0
    assert _symmetry_adjusted_cost(1, (5, 5), group) == 1
    assert _symmetry_adjusted_cost(2, (5, 5), group) == 1  # 2*15//25 == 1
    assert _symmetry_adjusted_cost(100, (5, 5), group) == 60


# --- `axes` spellings: what numpy takes, what it refuses, what each costs ----
#
# Ground truth measured against plain numpy 2.2.6 for every spelling below,
# and each test re-checks its own case against `np.tensordot` before asserting
# anything about `fnp.tensordot`, so the pin cannot drift away from numpy.
# Assertions are on exception TYPE and on `flops_used`, never on message text:
# CI spans numpy 2.0 through 2.4 and the wording differs across it.
#
# The measured gaps these cover, all of them charges for work that never ran:
#   * a duplicated contracted axis (`axes=([0, 0], [0, 0])`) was priced as if
#     the axis really did contract twice, and charged, before numpy refused it;
#   * every numpy integer scalar failed `isinstance(..., int)` and was rejected
#     with `TypeError` -- an over-rejection, in BOTH the whole-`axes` form and
#     the per-operand form;
#   * a one-shot `axes` spec was drained for flopscope's own geometry and the
#     exhausted object forwarded on, so the call was priced and charged and
#     only then refused.

AXES_PAD = 25
# `2*alpha - M` for (3,4) against (4,5) contracting the shared extent 4:
# alpha = 12*20//4 = 60, M = 3*5 = 15. Padding with singleton axes changes
# neither, so this is the price on both sides of the 52-letter budget.
AXES_HONEST_COST = 105
# `axes=0` is an outer product: nothing is contracted, so alpha = M = 240.
AXES_OUTER_COST = 240


def _axes_operands(pad, form):
    """(a, b) contracting a shared extent-4 axis, optionally past the budget.

    The whole-`axes` integer form contracts a's LAST axes against b's FIRST,
    so a's padding has to go in front and b's behind for the pairing to
    survive it; the per-operand form names axes 1 and 0 explicitly and takes
    trailing padding on both.
    """
    a, b = np.ones((3, 4)), np.ones((4, 5))
    if pad == 0:
        return a, b
    if form == "scalar":
        return _pad_front(a, pad), _pad_end(b, pad)
    return _pad_end(a, pad), _pad_end(b, pad)


def _square_axes_operands(pad):
    """Two equal (3,3) operands, so an extent check can never fire first.

    The duplicate-axis cases need every contracted pair to line up, otherwise
    they would be testing `_validate_contracted_extents` instead.
    """
    a, b = np.ones((3, 3)), np.ones((3, 3))
    return (a, b) if pad == 0 else (_pad_end(a, pad), _pad_end(b, pad))


@pytest.mark.parametrize("pad", [0, AXES_PAD])
@pytest.mark.parametrize("np_int", [np.int64, np.int32])
def test_numpy_integer_scalar_axes_is_priced_like_the_plain_int(pad, np_int):
    """`axes=np.int64(1)` is a working call in numpy, so it must work here.

    This is the whole-`axes` spelling. It and the per-operand spelling below
    are separate `isinstance(..., int)` tests in the parser, and fixing either
    one alone leaves the other rejecting a call numpy runs -- which is worse
    than the over-charge being fixed, because it turns working code into a
    hard failure. `np.int32` is in the parametrisation because numpy accepts
    the whole integer family, not just the platform-default width.
    """
    a, b = _axes_operands(pad, "scalar")
    assert (a.ndim + b.ndim > 52) is (pad == AXES_PAD)
    expected = np.tensordot(a, b, axes=np_int(1))  # ground truth: numpy runs it
    assert np.array_equal(expected, np.tensordot(a, b, axes=1))

    a, b = fnp.asarray(a), fnp.asarray(b)
    numpy_int_bill = billed(lambda: fnp.tensordot(a, b, axes=np_int(1)))
    plain_int_bill = billed(lambda: fnp.tensordot(a, b, axes=1))
    assert numpy_int_bill == plain_int_bill == AXES_HONEST_COST


@pytest.mark.parametrize("pad", [0, AXES_PAD])
@pytest.mark.parametrize("np_int", [np.int64, np.int32])
def test_numpy_integer_pair_axes_is_priced_like_the_plain_int(pad, np_int):
    """The per-operand spelling: `axes=(np.int64(1), np.int64(0))`.

    The other half of the same defect. numpy takes a numpy integer here just
    as readily -- it decides "one axis or many" with a `len()` probe, not a
    type test, so any integer scalar lands in the one-axis branch.
    """
    a, b = _axes_operands(pad, "pair")
    assert (a.ndim + b.ndim > 52) is (pad == AXES_PAD)
    expected = np.tensordot(a, b, axes=(np_int(1), np_int(0)))  # numpy runs it
    assert np.array_equal(expected, np.tensordot(a, b, axes=(1, 0)))

    a, b = fnp.asarray(a), fnp.asarray(b)
    numpy_int_bill = billed(lambda: fnp.tensordot(a, b, axes=(np_int(1), np_int(0))))
    nested_bill = billed(lambda: fnp.tensordot(a, b, axes=([np_int(1)], [np_int(0)])))
    plain_int_bill = billed(lambda: fnp.tensordot(a, b, axes=(1, 0)))
    assert numpy_int_bill == nested_bill == plain_int_bill == AXES_HONEST_COST


@pytest.mark.parametrize("pad", [0, AXES_PAD])
@pytest.mark.parametrize("np_int", [np.int64, np.int32])
def test_numpy_integer_axes_computes_the_same_answer(pad, np_int):
    """Accepting the spelling is only half of it -- the result must match."""
    a, b = _axes_operands(pad, "pair")
    expected = np.tensordot(a, b, axes=(1, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True):
            got = fnp.tensordot(
                fnp.asarray(a), fnp.asarray(b), axes=(np_int(1), np_int(0))
            )
    assert np.array_equal(np.asarray(got), expected)


@pytest.mark.parametrize("pad", [0, AXES_PAD])
@pytest.mark.parametrize(
    "axes_for",
    [
        pytest.param(lambda a: ([0, 0], [0, 1]), id="duplicate-on-a"),
        pytest.param(lambda a: ([0, 1], [0, 0]), id="duplicate-on-b"),
        pytest.param(lambda a: ([0, 0], [0, 0]), id="duplicate-on-both"),
        # Normalises to axis 0 at any rank, so it stays a duplicate of the
        # leading 0 whether or not the operands are padded. A duplicate check
        # written against the RAW axes would let this one through.
        pytest.param(lambda a: ([0, -a.ndim], [0, 1]), id="duplicate-via-negative"),
    ],
)
def test_duplicate_contracted_axis_refuses_before_charging(pad, axes_for):
    """numpy has no accepting case for a repeated contracted axis.

    It builds `newaxes_a = notin + axes_a` with `notin` excluding everything
    already named, so a repeat makes that permutation longer than the operand
    and the internal `transpose` always raises. The operands are square and
    the pairs all line up, so the extent check ahead of it cannot be what
    fires -- this really is the duplicate being refused.

    Before the fix the duplicate was priced as a genuine double contraction
    (its extent multiplied into `contracted` twice, the axis dropped from
    `a_surviving` once) and `budget.deduct` charged that on entry, on both
    sides of the letter budget.
    """
    a, b = _square_axes_operands(pad)
    axes = axes_for(a)
    assert (a.ndim + b.ndim > 52) is (pad == AXES_PAD)
    with pytest.raises(ValueError):  # ground truth: plain numpy refuses it
        np.tensordot(a, b, axes=axes)

    a, b = fnp.asarray(a), fnp.asarray(b)
    assert _raises_billing(lambda: fnp.tensordot(a, b, axes=axes)) == 0


@pytest.mark.parametrize("pad", [0, AXES_PAD])
def test_distinct_axes_on_the_same_operands_still_run(pad):
    """The over-rejection guard for the duplicate check: no repeat, no refusal.

    Same square operands, same shape of spec, every axis named once. A check
    that keyed on "two contracted axes" rather than on a repeat would break
    this, and breaking a working contraction is worse than the over-charge.
    """
    a, b = _square_axes_operands(pad)
    expected = np.tensordot(a, b, axes=([0, 1], [0, 1]))  # numpy runs it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as budget:
            got = fnp.tensordot(fnp.asarray(a), fnp.asarray(b), axes=([0, 1], [0, 1]))
            # alpha = 9*9//9 = 9 multiplies into a scalar output: 2*9 - 1.
            assert budget.flops_used == 17
    assert np.array_equal(np.asarray(got), expected)


@pytest.mark.parametrize("pad", [0, AXES_PAD])
@pytest.mark.parametrize(
    "axes_factory",
    [
        pytest.param(lambda: (iter([1]), iter([0])), id="iterators"),
        pytest.param(lambda: ((x for x in [1]), (x for x in [0])), id="generators"),
    ],
)
def test_one_shot_per_operand_axes_refuses_before_charging(pad, axes_factory):
    """A per-operand spec that cannot be re-read is numpy's own TypeError.

    numpy wraps a spec with no `__len__` in a one-element list and then uses
    it as an index into the shape tuple, which fails. flopscope used to drain
    the iterator into a tuple for its own geometry -- hiding that -- price the
    contraction, charge it, and only then hand numpy the now-empty original.
    The exception the caller saw was already right; what was wrong was that
    the budget had been spent by the time it arrived.
    """
    a, b = _axes_operands(pad, "pair")
    assert (a.ndim + b.ndim > 52) is (pad == AXES_PAD)
    with pytest.raises(TypeError):  # ground truth: plain numpy refuses it
        np.tensordot(a, b, axes=axes_factory())

    a, b = fnp.asarray(a), fnp.asarray(b)
    assert (
        _raises_billing(lambda: fnp.tensordot(a, b, axes=axes_factory()), TypeError)
        == 0
    )


@pytest.mark.parametrize("pad", [0, AXES_PAD])
@pytest.mark.parametrize(
    "axes_factory",
    [
        pytest.param(lambda: iter([[1], [0]]), id="iterator"),
        pytest.param(lambda: (spec for spec in ([1], [0])), id="generator"),
    ],
)
def test_one_shot_whole_axes_spec_still_runs(pad, axes_factory):
    """numpy DOES accept a one-shot spec at the top level, so we must too.

    It unpacks `axes_a, axes_b = axes` once and never looks at the object
    again. flopscope unpacked it too, then forwarded the drained original and
    got refused for running out of values -- after charging. Forwarding the
    normalised pairing instead is what makes this the working call numpy says
    it is; refusing every one-shot spec outright would have "fixed" the
    over-charge by breaking a call numpy runs.
    """
    a, b = _axes_operands(pad, "pair")
    expected = np.tensordot(a, b, axes=axes_factory())  # ground truth: it runs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as budget:
            got = fnp.tensordot(fnp.asarray(a), fnp.asarray(b), axes=axes_factory())
            assert budget.flops_used == AXES_HONEST_COST
    assert np.array_equal(np.asarray(got), expected)


@pytest.mark.parametrize("pad", [0, AXES_PAD])
def test_boolean_axes_follow_numpy_on_both_spellings(pad):
    """`bool` is an `int` subclass, and numpy treats the two spellings apart.

    The whole-`axes` form negates its argument, and `-True` is `-1`, so
    `axes=True` is a working one-axis contraction. Every per-operand axis, by
    contrast, ends up in `ndarray.transpose`, which refuses a boolean
    outright. Widening an integer predicate without noticing that would have
    silently changed the first; leaving the second alone would keep charging
    for a call numpy always refuses.
    """
    a, b = _axes_operands(pad, "scalar")
    expected = np.tensordot(a, b, axes=True)  # ground truth: numpy runs it
    assert np.array_equal(expected, np.tensordot(a, b, axes=1))
    fa, fb = fnp.asarray(a), fnp.asarray(b)
    assert billed(lambda: fnp.tensordot(fa, fb, axes=True)) == AXES_HONEST_COST

    a, b = _axes_operands(pad, "pair")
    with pytest.raises(TypeError):  # ground truth: plain numpy refuses it
        np.tensordot(a, b, axes=(True, False))
    fa, fb = fnp.asarray(a), fnp.asarray(b)
    assert (
        _raises_billing(lambda: fnp.tensordot(fa, fb, axes=(True, False)), TypeError)
        == 0
    )
    assert (
        _raises_billing(
            lambda: fnp.tensordot(fa, fb, axes=([True], [False])), TypeError
        )
        == 0
    )


@pytest.mark.parametrize("pad", [0, AXES_PAD])
@pytest.mark.parametrize(
    "axes_factory",
    [
        pytest.param(lambda: (1, 0), id="int-pair"),
        pytest.param(lambda: ([1], [0]), id="list-pair"),
        pytest.param(lambda: ((1,), (0,)), id="tuple-pair"),
        pytest.param(lambda: ({1}, {0}), id="set-pair"),
        pytest.param(lambda: (range(1, 2), range(0, 1)), id="range-pair"),
        pytest.param(lambda: (np.array([1]), np.array([0])), id="ndarray-pair"),
        pytest.param(lambda: (np.int64(1), np.int64(0)), id="numpy-int-pair"),
        pytest.param(lambda: ([np.int32(1)], [np.int32(0)]), id="numpy-int-in-list"),
    ],
)
def test_every_accepted_pair_spelling_bills_the_honest_price(pad, axes_factory):
    """Pin the price of every spelling numpy accepts, on both sides.

    This is the regression an over-rejecting -- or over-charging -- fix would
    cause. All of these name the same single pair of axes, so all of them must
    cost the same `2*alpha - M`, unchanged by the fixes above and unchanged by
    rank.
    """
    a, b = _axes_operands(pad, "pair")
    expected = np.tensordot(a, b, axes=axes_factory())  # numpy accepts them all
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as budget:
            got = fnp.tensordot(fnp.asarray(a), fnp.asarray(b), axes=axes_factory())
            assert budget.flops_used == AXES_HONEST_COST
    assert np.array_equal(np.asarray(got), expected)


@pytest.mark.parametrize("pad", [0, AXES_PAD])
def test_negative_and_zero_axes_spellings_are_unchanged(pad):
    """The rest of the sweep: negative axes, and `axes=0`'s outer product.

    `axes=0` contracts nothing, so it must stay an outer product rather than
    be caught by any of the new refusals -- the empty pairing has no duplicate
    and no axis to type-check. The forwarded pairing for it is `((), ())`,
    which numpy treats identically to the integer `0`.
    """
    # a's last axis carries the contracted extent in the "scalar" layout, so
    # -1 names it at either rank -- the same pairing `axes=1` expands to.
    a, b = _axes_operands(pad, "scalar")
    fa, fb = fnp.asarray(a), fnp.asarray(b)
    assert billed(lambda: fnp.tensordot(fa, fb, axes=([-1], [0]))) == AXES_HONEST_COST
    assert billed(lambda: fnp.tensordot(fa, fb, axes=(-1, 0))) == AXES_HONEST_COST
    assert billed(lambda: fnp.tensordot(fa, fb, axes=1)) == AXES_HONEST_COST

    a, b = _axes_operands(pad, "pair")
    fa, fb = fnp.asarray(a), fnp.asarray(b)
    assert billed(lambda: fnp.tensordot(fa, fb, axes=0)) == AXES_OUTER_COST
    assert billed(lambda: fnp.tensordot(fa, fb, axes=((), ()))) == AXES_OUTER_COST
    assert np.array_equal(np.tensordot(a, b, axes=((), ())), np.tensordot(a, b, axes=0))


@pytest.mark.parametrize("pad", [0, AXES_PAD])
def test_unsigned_scalar_axes_follows_numpy_rather_than_looking_plausible(pad):
    """The reason the integer arm uses numpy's `range(-axes, 0)` verbatim.

    numpy negates the whole-`axes` argument, and negating an unsigned scalar
    wraps: the a-side list comes out empty, its length never matches the
    b-side, and the call is a `ValueError`. Deriving the axes from
    `a_ndim - axes` instead looks equivalent -- and is, for every `axes` numpy
    actually runs -- but it would invent a perfectly plausible axis here,
    price the contraction, charge it, and only then let numpy refuse the call.

    `np.uint8(0)` is the control: negating it wraps to nothing, so numpy runs
    it as the outer product `axes=0` names, and so must we.
    """
    a, b = _axes_operands(pad, "scalar")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # numpy's own unsigned-negation overflow
        with pytest.raises(ValueError):  # ground truth: plain numpy refuses it
            # numpy's stubs type `axes` as int | tuple[...], not np.unsignedinteger,
            # but the runtime accepts it -- that gap is exactly what this test pins.
            np.tensordot(a, b, axes=np.uint8(1))  # type: ignore[reportCallIssue]
        expected = np.tensordot(a, b, axes=np.uint8(0))  # type: ignore[reportCallIssue]  # ground truth: it runs
    assert np.array_equal(expected, np.tensordot(a, b, axes=0))

    fa, fb = fnp.asarray(a), fnp.asarray(b)
    assert _raises_billing(lambda: fnp.tensordot(fa, fb, axes=np.uint8(1))) == 0
    assert billed(lambda: fnp.tensordot(fa, fb, axes=np.uint8(0))) == AXES_OUTER_COST


# --- the differential sweep: same call, same outcome, nothing charged --------
#
# The case-by-case pins above each cover one spelling. This drives a grid of
# them against plain `np.tensordot` on the same operands and asserts three
# things per case: identical outcome (both run, or both raise the SAME
# exception TYPE), `flops_used == 0` whenever either side raises, and identical
# shape and values whenever both run.
#
# It exists because the order of `tensordot`'s own checks is load-bearing and
# is not something a hand-picked case list keeps honest. numpy pairs the axis
# counts, then indexes each shape with the RAW (still signed, possibly `bool`)
# index, then compares the extents, and only then normalises a negative axis;
# its two internal transposes run last, after all of that. Validating in any
# other order still refuses the same calls -- and still charges nothing, which
# is what the fix was for -- but hands the caller the wrong exception type: an
# `IndexError` where numpy stops earlier at a `ValueError` extent mismatch, or
# a `TypeError` for a boolean numpy indexes happily and rejects two steps
# later. Measured against numpy 2.2.6, that ordering alone accounted for 341
# type mismatches across this grid.
#
# TYPE only, never message text: CI spans numpy 2.0 through 2.4 and the wording
# differs across it.

PARITY_SHAPES = [(2, 3), (3, 2, 3), (3,), (2, 3, 4), (4,), (3, 3), (2, 2, 2, 2)]

# Factories, not values: a one-shot spec has to be built fresh for each of the
# two calls, and a spec that numpy mutates in place must not leak between them.
PARITY_AXES_SPECS = {
    # Whole-`axes` integer form, including the counts that overrun one operand.
    **{f"int-{n}": (lambda n=n: n) for n in range(6)},
    # Every numpy integer scalar numpy itself accepts here. `uint` is not a
    # decoration: numpy negates the argument, so an unsigned one wraps.
    **{
        f"{np_int.__name__}-{n}": (lambda np_int=np_int, n=n: np_int(n))
        for np_int in (np.int64, np.int32, np.int16, np.uint8)
        for n in (0, 1, 2)
    },
    # `-True` is `-1`, so the integer arm takes a boolean; the per-operand arm
    # does not (see below). `np.bool_` has no `__neg__` at all.
    "bool-scalar-true": lambda: True,
    "bool-scalar-false": lambda: False,
    "np-bool-scalar": lambda: np.True_,
    # A 0-d array is not iterable, so numpy reads it as a count, not a pair.
    "zero-d-array-count": lambda: np.array(1),
    # Per-operand spellings: lists, tuples, bare scalars, and the containers
    # numpy's `len()` probe accepts.
    "list-pair": lambda: ([1], [0]),
    "tuple-pair": lambda: ((1,), (0,)),
    "int-pair": lambda: (1, 0),
    "set-pair": lambda: ({1}, {0}),
    "range-pair": lambda: (range(1, 2), range(0, 1)),
    "ndarray-pair": lambda: (np.array([1]), np.array([0])),
    "numpy-int-pair": lambda: (np.int64(1), np.int64(0)),
    "numpy-int-in-list": lambda: ([np.int32(1)], [np.int32(0)]),
    "zero-d-array-in-pair": lambda: (np.array(1), np.array(0)),
    "empty-pair": lambda: ([], []),
    # Negatives, which numpy normalises only AFTER the extent comparison.
    "negative-a": lambda: ([-1], [0]),
    "negative-b": lambda: ([0], [-1]),
    "negative-both": lambda: ([-1, -2], [-1, -2]),
    "negative-out-of-range": lambda: ([-3], [0]),
    # Out-of-range positives, whose exception type depends on whether an
    # earlier pair mismatches first.
    "out-of-range-a": lambda: ([5], [0]),
    "out-of-range-b": lambda: ([0], [5]),
    "out-of-range-both": lambda: ([5], [5]),
    # Duplicates, which survive to numpy's transpose and fail there.
    "duplicate-a": lambda: ([0, 0], [0, 1]),
    "duplicate-b": lambda: ([0, 1], [1, 1]),
    "duplicate-via-negative": lambda: ([0, -2], [0, 1]),
    "duplicate-unhashable": lambda: ([np.array(0), np.array(0)], [0, 1]),
    # Mis-paired counts, refused before either shape is indexed.
    "count-mismatch-short-a": lambda: ([0], [0, 1]),
    "count-mismatch-short-b": lambda: ([0, 1], [0]),
    # Longer pairings, valid only at the higher ranks in the grid.
    "pair-two": lambda: ([0, 1], [0, 1]),
    "pair-three": lambda: ([0, 1, 2], [0, 1, 2]),
    # Booleans per-operand: indexable, so they reach the transpose that
    # refuses them.
    "bool-pair": lambda: (True, False),
    "bool-pair-reversed": lambda: (False, True),
    "bool-in-list": lambda: ([True], [False]),
    "bool-list-two": lambda: ([True, False], [False, True]),
    "np-bool-pair": lambda: (np.bool_(True), np.bool_(False)),
    "bool-mixed-with-int": lambda: ([True, 0], [0, 1]),
    # Not axes at all.
    "float-in-list": lambda: ([1.0], [0]),
    "float-pair": lambda: (1.0, 0),
    "none-pair": lambda: (None, 0),
    "nested-list": lambda: ([[0]], [0]),
    "string-pair": lambda: ("a", "b"),
    # One-shot specs. numpy accepts the whole-spec form (it unpacks once) and
    # refuses the per-operand form (it indexes the iterator itself).
    "whole-spec-iterator": lambda: iter([[1], [0]]),
    "whole-spec-generator": lambda: (spec for spec in ([1], [0])),
    "whole-spec-iterator-of-ints": lambda: iter([1, 0]),
    "whole-spec-range": lambda: range(2),
    "whole-spec-ndarray": lambda: np.array([1, 0]),
    "per-operand-iterators": lambda: (iter([1]), iter([0])),
    "per-operand-generators": lambda: ((x for x in [1]), (x for x in [0])),
    "iterator-in-list": lambda: ([iter([1])], [0]),
}


class _Outcome(NamedTuple):
    """What one `tensordot` call did: raised this type, or returned this array.

    `raised` is the exception TYPE, never an instance and never a message --
    the type is the whole assertion. `billed` is 0 for plain numpy, which has
    no meter.
    """

    raised: type[BaseException] | None
    result: np.ndarray | None
    billed: int


def _numpy_outcome(a, b, axes) -> _Outcome:
    """Run plain `np.tensordot`: the ground truth this sweep compares against."""
    with warnings.catch_warnings():
        # numpy's own unsigned-negation overflow, among others.
        warnings.simplefilter("ignore")
        try:
            return _Outcome(None, np.asarray(np.tensordot(a, b, axes=axes)), 0)
        except Exception as exc:  # noqa: BLE001 -- the type IS the assertion
            return _Outcome(type(exc), None, 0)


def _flopscope_outcome(a, b, axes) -> _Outcome:
    """The same call through `fnp.tensordot`, with what it billed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as budget:
            try:
                got = fnp.tensordot(fnp.asarray(a), fnp.asarray(b), axes=axes)
                return _Outcome(None, np.asarray(got), budget.flops_used)
            except Exception as exc:  # noqa: BLE001
                return _Outcome(type(exc), None, budget.flops_used)


@pytest.mark.parametrize("spec_id", sorted(PARITY_AXES_SPECS))
def test_tensordot_axes_outcome_matches_numpy_on_every_shape_pair(spec_id):
    """One `axes` spelling against the whole shape grid, both directions.

    Every mismatch in the grid is collected rather than raised on sight, so a
    regression reports the whole family it broke instead of one arbitrary
    member of it.
    """
    make_axes = PARITY_AXES_SPECS[spec_id]
    rng = np.random.default_rng(0)
    operands = {shape: rng.standard_normal(shape) for shape in PARITY_SHAPES}
    mismatches = []
    for a_shape in PARITY_SHAPES:
        for b_shape in PARITY_SHAPES:
            a, b = operands[a_shape], operands[b_shape]
            expected = _numpy_outcome(a, b, make_axes())
            got = _flopscope_outcome(a, b, make_axes())
            where = f"a={a_shape} b={b_shape} axes={spec_id}"
            if (expected.raised is None) != (got.raised is None):
                mismatches.append(
                    f"{where}: numpy {expected.raised}, flopscope {got.raised}"
                )
                continue
            if expected.raised is not None:
                assert got.raised is not None  # the branch above settled this
                if expected.raised is not got.raised:
                    mismatches.append(
                        f"{where}: numpy {expected.raised.__name__}, "
                        f"flopscope {got.raised.__name__}"
                    )
                # Refuse before charging: a call numpy was always going to
                # refuse must leave the budget untouched.
                if got.billed != 0:
                    mismatches.append(f"{where}: refused but billed {got.billed}")
                continue
            assert expected.result is not None and got.result is not None
            if expected.result.shape != got.result.shape:
                mismatches.append(
                    f"{where}: shape {expected.result.shape} vs {got.result.shape}"
                )
            elif not np.allclose(expected.result, got.result):
                mismatches.append(f"{where}: values differ")
    assert not mismatches, "\n".join(mismatches)


def test_tensordot_axes_parity_sweep_covers_both_outcomes():
    """The guard on the sweep above: it must exercise more than one branch.

    A grid that only ever raised -- or only ever ran -- would pass the parity
    assertions while pinning nothing. This counts the outcomes to prove the
    grid straddles the boundary, and that all three exception types the
    ordering decides between actually occur in it.
    """
    rng = np.random.default_rng(0)
    operands = {shape: rng.standard_normal(shape) for shape in PARITY_SHAPES}
    ran = 0
    raised: dict[str, int] = {}
    for make_axes in PARITY_AXES_SPECS.values():
        for a_shape in PARITY_SHAPES:
            for b_shape in PARITY_SHAPES:
                outcome = _numpy_outcome(
                    operands[a_shape], operands[b_shape], make_axes()
                )
                if outcome.raised is None:
                    ran += 1
                else:
                    name = outcome.raised.__name__
                    raised[name] = raised.get(name, 0) + 1
    assert ran > 100
    assert raised["ValueError"] > 100
    assert raised["IndexError"] > 10
    assert raised["TypeError"] > 10
