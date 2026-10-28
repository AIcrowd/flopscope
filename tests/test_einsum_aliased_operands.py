"""einsum with one array passed as several operands (issue #346).

Passing the same array object twice lets the cost model treat the operands as
identical copies and discount the contraction. When a pre-reduction sums an
axis out of only one copy, the copies no longer hold equal arrays, and pricing
the step as if they did raised IndexError (two operands) or ValueError (a step
whose joint group was summed away entirely). These tests check that aliasing
never changes the value, never raises where a copy works, never bills more
than a copy, and still earns the discount where the copies stay equal.
"""

import itertools

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


@pytest.fixture
def ctx():
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as budget:
        yield budget


def _billed(ctx, eq, *ops):
    before = ctx.flops_used
    result = np.asarray(fnp.einsum(eq, *ops))
    return result, ctx.flops_used - before


MATRIX_PARTS = ["".join(p) for p in itertools.product("ijk", repeat=2)]


def _equations(n_ops, rank, letters, all_outputs, first=None):
    for parts in itertools.product(
        itertools.product(letters, repeat=rank), repeat=n_ops
    ):
        if first is not None and "".join(parts[0]) != first:
            continue
        used = sorted(set("".join("".join(p) for p in parts)))
        if all_outputs:
            outputs = [
                o for r in range(len(used) + 1) for o in itertools.permutations(used, r)
            ]
        else:
            # A scalar output sums every label, so every step pre-reduces.
            outputs = [()]
        for out in outputs:
            yield ",".join("".join(p) for p in parts) + "->" + "".join(out)


def _sweep(ctx, equations, same_ops, copy_ops, np_ops):
    """Return the equations where aliasing changes the value or bills above a copy."""
    bad = []
    for eq in equations:
        ref = np.einsum(eq, *np_ops)
        same, same_bill = _billed(ctx, eq, *same_ops)
        _, copy_bill = _billed(ctx, eq, *copy_ops)
        if same.shape != ref.shape or not np.allclose(same, ref):
            bad.append(f"{eq} value")
        elif same_bill > copy_bill:
            bad.append(f"{eq} bill {same_bill} > copy {copy_bill}")
    return bad


ISSUE_FORMS = [
    ("ij,ik->k", (5, 3)),
    ("ij,ik->j", (5, 3)),
    ("ij,ik->ij", (5, 3)),
    ("ij,jk->k", (3, 3)),
    ("ij,jk->i", (3, 3)),
]


@pytest.mark.parametrize(("eq", "shape"), ISSUE_FORMS)
def test_issue_forms_match_numpy_and_bill_like_a_copy(ctx, eq, shape):
    an = np.random.default_rng(0).standard_normal(shape)
    a = fnp.asarray(an)
    same, same_bill = _billed(ctx, eq, a, a)
    copy, copy_bill = _billed(ctx, eq, a, fnp.asarray(an.copy()))
    np.testing.assert_allclose(same, np.einsum(eq, an, an))
    np.testing.assert_allclose(copy, same)
    assert same_bill == copy_bill


def test_gram_discount_is_kept(ctx):
    xn = np.random.default_rng(1).standard_normal((5, 3))
    x = fnp.asarray(xn)
    same, same_bill = _billed(ctx, "ij,ik->jk", x, x)
    _, copy_bill = _billed(ctx, "ij,ik->jk", x, fnp.asarray(xn.copy()))
    np.testing.assert_allclose(same, xn.T @ xn)
    assert same_bill < copy_bill


def test_copies_reduced_the_same_way_keep_the_discount(ctx):
    # Both copies sum axes 1 and 2 away, so both become T.sum((1, 2)) and the
    # outer product of the two is symmetric.
    tn = np.random.default_rng(2).standard_normal((3, 3, 3))
    t = fnp.asarray(tn)
    same, same_bill = _billed(ctx, "ijk,lmn->il", t, t)
    _, copy_bill = _billed(ctx, "ijk,lmn->il", t, fnp.asarray(tn.copy()))
    np.testing.assert_allclose(same, np.einsum("ijk,lmn->il", tn, tn))
    assert same_bill < copy_bill


def test_a_trace_and_a_sum_are_not_identical_copies(ctx):
    # Both copies lose axes 1 and 2, but one by a sum and one by a trace, so
    # the reduced vectors differ and the outer product is not symmetric.
    tn = np.random.default_rng(3).standard_normal((3, 3, 3))
    t = fnp.asarray(tn)
    same, same_bill = _billed(ctx, "ijk,lmm->il", t, t)
    _, copy_bill = _billed(ctx, "ijk,lmm->il", t, fnp.asarray(tn.copy()))
    np.testing.assert_allclose(same, np.einsum("ijk,lmm->il", tn, tn))
    assert same_bill == copy_bill


def test_the_two_operand_sweep_has_every_equation_numpy_accepts():
    assert len(list(_equations(2, 2, "ijk", all_outputs=True))) == 792


@pytest.mark.parametrize("first", MATRIX_PARTS)
@pytest.mark.parametrize("kind", ["plain", "symmetric"])
def test_every_two_operand_equation_over_a_matrix(ctx, kind, first):
    rng = np.random.default_rng(4)
    if kind == "plain":
        s = fnp.asarray(rng.standard_normal((3, 3)))
    else:
        x = fnp.asarray(rng.standard_normal((5, 3)))
        s = fnp.einsum("ij,ik->jk", x, x)
    sn = np.asarray(s).copy()
    equations = _equations(2, 2, "ijk", all_outputs=True, first=first)
    assert _sweep(ctx, equations, (s, s), (s, s.copy()), (sn, sn)) == []


@pytest.mark.parametrize("first", MATRIX_PARTS)
@pytest.mark.parametrize("pattern", [(0, 0, 1), (0, 1, 0), (0, 0, 0)])
def test_three_operand_equations_with_a_repeated_matrix(ctx, pattern, first):
    rng = np.random.default_rng(5)
    sn, tn = rng.standard_normal((3, 3)), rng.standard_normal((3, 3))
    s, t = fnp.asarray(sn), fnp.asarray(tn)
    same = tuple((s, t)[p] for p in pattern)
    copies = tuple(
        t if p else (s if i == 0 else fnp.asarray(sn.copy()))
        for i, p in enumerate(pattern)
    )
    np_ops = tuple((sn, tn)[p] for p in pattern)
    equations = _equations(3, 2, "ijk", all_outputs=False, first=first)
    assert _sweep(ctx, equations, same, copies, np_ops) == []


@pytest.mark.parametrize("first", ["b", "c", "d"])
@pytest.mark.parametrize("pattern", [(0, 0, 1), (0, 0, 0)])
def test_three_operand_equations_with_a_repeated_vector(ctx, pattern, first):
    rng = np.random.default_rng(6)
    vn, wn = rng.standard_normal(3), rng.standard_normal(3)
    v, w = fnp.asarray(vn), fnp.asarray(wn)
    same = tuple((v, w)[p] for p in pattern)
    copies = tuple(
        w if p else (v if i == 0 else fnp.asarray(vn.copy()))
        for i, p in enumerate(pattern)
    )
    np_ops = tuple((vn, wn)[p] for p in pattern)
    equations = _equations(3, 1, "bcd", all_outputs=True, first=first)
    assert _sweep(ctx, equations, same, copies, np_ops) == []


def test_scalar_step_whose_joint_group_is_summed_away(ctx):
    rng = np.random.default_rng(7)
    vn, wn = rng.standard_normal(3), rng.standard_normal(3)
    v, w = fnp.asarray(vn), fnp.asarray(wn)
    same, same_bill = _billed(ctx, "b,d,c->", v, v, w)
    _, copy_bill = _billed(ctx, "b,d,c->", v, fnp.asarray(vn.copy()), w)
    np.testing.assert_allclose(same, vn.sum() ** 2 * wn.sum())
    assert same_bill == copy_bill


def test_outer_product_of_two_unreduced_copies_keeps_the_discount(ctx):
    vn = np.random.default_rng(8).standard_normal(3)
    v = fnp.asarray(vn)
    same, same_bill = _billed(ctx, "b,c,d->bc", v, v, v)
    _, copy_bill = _billed(
        ctx, "b,c,d->bc", v, fnp.asarray(vn.copy()), fnp.asarray(vn.copy())
    )
    np.testing.assert_allclose(same, np.outer(vn, vn) * vn.sum())
    assert same_bill < copy_bill


def test_reduction_signature_tells_a_trace_from_a_sum():
    from flopscope._accumulation._cost import _reduction_signature

    assert _reduction_signature("ij", {"j"}) == (-1, 1)
    assert _reduction_signature("ji", {"i"}) == (-1, 1)
    assert _reduction_signature("ij", set()) == (-1, -1)
    assert _reduction_signature("kk", {"k"}) == (0, 0)
    assert _reduction_signature("ij", {"i", "j"}) == (0, 1)


def test_identity_groups_survive_only_equal_reductions():
    from flopscope._accumulation._cost import _identity_pattern_after_pre_reduction

    same = [(-1, 1), (-1, 1)]
    differ = [(-1, 1), (-1, -1)]
    assert _identity_pattern_after_pre_reduction(((0, 1),), same) == ((0, 1),)
    assert _identity_pattern_after_pre_reduction(((0, 1),), differ) is None
    assert _identity_pattern_after_pre_reduction(None, differ) is None


def test_enumerate_wreath_rejects_copies_of_different_ranks():
    from flopscope._accumulation._wreath import enumerate_wreath

    with pytest.raises(ValueError, match="mixes operands of ranks"):
        list(
            enumerate_wreath(
                identical_groups=((0, 1),),
                per_op_symmetry=(None, None),
                axis_ranks=(1, 2),
                u_offsets=(0, 1),
            )
        )
