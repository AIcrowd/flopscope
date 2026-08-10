"""The 52-letter subscript budget: allocation, fallback pricing, invariants."""

from __future__ import annotations

import math
import warnings

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
