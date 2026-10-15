"""gufunc contractions are priced on the axes ``axis=``/``axes=`` name (#272).

``vecdot``, ``matvec`` and ``vecmat`` are priced through the einsum engine with
a ``...``-broadcasting subscript that keeps the core dimensions last. numpy's
``axis=``/``axes=`` move the core dimensions to other axes, but the bill kept
reading the default layout: ``vecdot(a, b, axis=0)`` on a ``(4096, 1)`` pair
was billed as 4096 length-1 dot products instead of one of length 4096, and
the mirror case over-billed. Each relocated call must bill exactly what the
same contraction bills when the core axes are moved last first.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp

requires_vecdot = pytest.mark.skipif(
    not hasattr(np, "vecdot"), reason="requires numpy >= 2.1"
)
requires_vecmat = pytest.mark.skipif(
    not hasattr(np, "vecmat"), reason="requires numpy >= 2.2"
)


def _billed(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
            before = int(ctx.flops_used)
            out = fn()
            return int(ctx.flops_used) - before, out


def _core_last(x, core_axes):
    """``x`` with ``core_axes`` moved last, in signature order."""
    dest = list(range(-len(core_axes), 0))
    return np.ascontiguousarray(np.moveaxis(x, list(core_axes), dest))


def _rand(shape, seed=0, dtype: Any = np.float64):
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype)


def _symmetrized(x, axes):
    perms = list(itertools.permutations(axes))
    total = np.zeros_like(x)
    for perm in perms:
        order = list(range(x.ndim))
        for src, dst in zip(axes, perm, strict=True):
            order[src] = dst
        total = total + np.transpose(x, order)
    return total / len(perms)


@requires_vecdot
def test_issue_repro_contracted_axis_first():
    a = _rand((4096, 1), 1, np.float32)
    b = _rand((4096, 1), 2, np.float32)
    cost, out = _billed(lambda: fnp.vecdot(a, b, axis=0))
    twin, _ = _billed(lambda: fnp.vecdot(a.T.copy(), b.T.copy()))
    assert cost == twin == 2 * 4096 - 1
    assert np.array_equal(np.asarray(out), np.vecdot(a, b, axis=0))  # type: ignore[reportCallIssue]


@requires_vecdot
def test_mirror_case_short_contracted_axis_is_not_over_billed():
    a = _rand((2, 4096), 1, np.float32)
    b = _rand((2, 4096), 2, np.float32)
    cost, _ = _billed(lambda: fnp.vecdot(a, b, axis=0))
    assert cost == 4096 * (2 * 2 - 1)


VECDOT_CASES = [
    ((4, 5, 6), (4, 5, 6), {"axis": 0}, (0,), (0,)),
    ((4, 5, 6), (4, 5, 6), {"axis": 1}, (1,), (1,)),
    ((4, 5, 6), (4, 5, 6), {"axis": -3}, (0,), (0,)),
    ((4, 5, 6), (4, 5, 6), {"axis": -1}, (2,), (2,)),
    ((4, 1, 6), (4, 5, 1), {"axis": 0}, (0,), (0,)),
    ((4, 3, 7), (4, 7), {"axes": [(0,), (-2,)]}, (0,), (0,)),
    ((3, 4, 7), (4, 7), {"axes": [(1,), (0,), ()]}, (1,), (0,)),
    ((4, 5, 6), (4, 5, 6), {"axes": [1, 1]}, (1,), (1,)),
    ((7,), (7, 2), {"axes": [(0,), (0,)]}, (0,), (0,)),
]


@requires_vecdot
@pytest.mark.parametrize(
    ("a_shape", "b_shape", "kwargs", "a_core", "b_core"), VECDOT_CASES
)
def test_vecdot_relocated_bills_like_core_last(
    a_shape, b_shape, kwargs, a_core, b_core
):
    a, b = _rand(a_shape, 1), _rand(b_shape, 2)
    expected = np.vecdot(a, b, **kwargs)
    cost, out = _billed(lambda: fnp.vecdot(a, b, **kwargs))
    twin, _ = _billed(lambda: fnp.vecdot(_core_last(a, a_core), _core_last(b, b_core)))
    assert cost == twin
    assert np.asarray(out).shape == expected.shape
    assert np.allclose(np.asarray(out), expected)


@requires_vecdot
def test_linalg_vecdot_axis_bills_like_core_last():
    a, b = _rand((6, 3), 1), _rand((6, 3), 2)
    cost, _ = _billed(lambda: fnp.linalg.vecdot(a, b, axis=0))
    twin, _ = _billed(lambda: fnp.vecdot(a.T.copy(), b.T.copy()))
    assert cost == twin == 3 * (2 * 6 - 1)


GEMV_CASES = [
    ("matvec", (3, 4, 2), (4, 2), [(0, 1), (0,), (0,)], (0, 1), (0,)),
    ("matvec", (3, 4, 2), (4, 2), [(0, 1), (0,), (1,)], (0, 1), (0,)),
    ("matvec", (3, 4, 2), (4, 2), [(0, 1), 0, -1], (0, 1), (0,)),
    ("matvec", (2, 4, 3), (4, 2), [(2, 1), (0,), (0,)], (2, 1), (0,)),
    ("vecmat", (4, 2), (2, 4, 3), [(0,), (1, 2), (0,)], (0,), (1, 2)),
    ("vecmat", (2, 4), (4, 2, 3), [(1,), (0, 2), (1,)], (1,), (0, 2)),
]


@requires_vecmat
@pytest.mark.parametrize(
    ("op", "a_shape", "b_shape", "axes", "a_core", "b_core"), GEMV_CASES
)
def test_matvec_vecmat_relocated_bill_like_core_last(
    op, a_shape, b_shape, axes, a_core, b_core
):
    a, b = _rand(a_shape, 1), _rand(b_shape, 2)
    expected = getattr(np, op)(a, b, axes=axes)
    cost, out = _billed(lambda: getattr(fnp, op)(a, b, axes=axes))
    twin, _ = _billed(
        lambda: getattr(fnp, op)(_core_last(a, a_core), _core_last(b, b_core))
    )
    assert cost == twin
    assert np.asarray(out).shape == expected.shape
    assert np.allclose(np.asarray(out), expected)


@requires_vecdot
@pytest.mark.parametrize(
    ("kwargs", "tag_axes"),
    [
        ({"axis": 0}, (0, 1)),
        ({"axis": 0, "keepdims": True}, (1, 2)),
        ({"axes": [(0,), (0,), (2,)], "keepdims": True}, (0, 1)),
        ({"axes": [(0,), (0,)], "keepdims": True}, (0, 1)),
    ],
)
def test_symmetric_operand_keeps_its_savings_and_tag(kwargs, tag_axes):
    k, n = 6, 5
    data = _symmetrized(_rand((k, n, n), 3), (1, 2))
    v = _rand((k,), 4)
    twin_data = _core_last(data, (0,))
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        S = flops.as_symmetric(data, symmetry=(1, 2))
        S_last = flops.as_symmetric(twin_data, symmetry=(0, 1))
        before = int(ctx.flops_used)
        out = fnp.vecdot(S, v, **kwargs)
        cost = int(ctx.flops_used) - before
        before = int(ctx.flops_used)
        fnp.vecdot(S_last, v)
        twin = int(ctx.flops_used) - before
    expected = np.vecdot(data, v, **kwargs)
    group = getattr(out, "symmetry", None)
    assert cost == twin
    assert group is not None
    assert tuple(group.axes) == tag_axes
    raw = np.asarray(out)
    assert raw.shape == expected.shape
    assert np.allclose(raw, expected)
    assert np.allclose(raw, np.swapaxes(raw, *tag_axes))


def _moved_axes(group):
    """Output axes some element of ``group`` actually moves."""
    axes = group.axes if group.axes is not None else tuple(range(group.degree))
    return {
        axes[i]
        for gen in group.generators
        for i, j in enumerate(gen.array_form)
        if i != j
    }


@requires_vecmat
@pytest.mark.parametrize(
    ("axes", "tag_axes"),
    [
        ([(2, 3), (0,), (0,)], (1, 2)),
        ([(2, 3), (0,), (1,)], (0, 2)),
        ([(2, 3), (0,), (2,)], (0, 1)),
    ],
)
def test_symmetric_loop_axes_keep_their_tag_around_the_output_core(axes, tag_axes):
    data = _symmetrized(_rand((4, 4, 3, 5), 5), (0, 1))
    w = _rand((5,), 6)
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        A = flops.as_symmetric(data, symmetry=(0, 1))
        before = int(ctx.flops_used)
        out = fnp.matvec(A, w, axes=axes)
        cost = int(ctx.flops_used) - before
        before = int(ctx.flops_used)
        fnp.matvec(A, w)
        twin = int(ctx.flops_used) - before
    expected = np.matvec(data, w, axes=axes)
    group = getattr(out, "symmetry", None)
    assert cost == twin
    assert group is not None
    assert _moved_axes(group) == set(tag_axes)
    raw = np.asarray(out)
    assert raw.shape == expected.shape
    assert np.allclose(raw, expected)
    assert np.allclose(raw, np.swapaxes(raw, *tag_axes))


@requires_vecdot
def test_relocated_contracted_extents_are_checked_before_charging():
    # The default layout pairs the two extent-4 axes and would price this;
    # numpy contracts axis 0, whose extents (2 and 1) it refuses to pair.
    a, b = np.ones((2, 4)), np.ones((1, 4))
    with pytest.raises(ValueError):
        np.vecdot(a, b, axis=0)  # type: ignore[reportCallIssue]
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        with pytest.raises(ValueError):
            fnp.vecdot(a, b, axis=0)
        assert ctx.flops_used == 0


REFUSED = [
    pytest.param("vecdot", (2, 3, 4), (2, 3, 4), {"axis": 3}, marks=requires_vecdot),
    pytest.param(
        "vecdot", (2, 3, 4), (2, 3, 4), {"axes": ((0,), (0,))}, marks=requires_vecdot
    ),
    pytest.param(
        "vecdot", (2, 3, 4), (2, 3, 4), {"axes": [(0,)]}, marks=requires_vecdot
    ),
    pytest.param(
        "vecdot",
        (2, 3, 4),
        (2, 3, 4),
        {"axis": 0, "axes": [(0,), (0,)]},
        marks=requires_vecdot,
    ),
    pytest.param(
        "vecdot",
        (2, 3, 4),
        (2, 3, 4),
        {"axes": [(0,), (0,), (0,)]},
        marks=requires_vecdot,
    ),
    pytest.param("matvec", (3, 4, 2), (4, 2), {"axis": 0}, marks=requires_vecmat),
    pytest.param(
        "matvec", (3, 4, 2), (3, 2), {"axes": [(0, 1), (0,)]}, marks=requires_vecmat
    ),
]


@pytest.mark.parametrize(("op", "a_shape", "b_shape", "kwargs"), REFUSED)
def test_spec_numpy_refuses_is_refused_before_charging(op, a_shape, b_shape, kwargs):
    a, b = np.ones(a_shape), np.ones(b_shape)
    with pytest.raises(Exception) as numpy_error:
        getattr(np, op)(a, b, **kwargs)
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        with pytest.raises(type(numpy_error.value)):
            getattr(fnp, op)(a, b, **kwargs)
        assert ctx.flops_used == 0


@requires_vecdot
def test_relocated_layout_past_the_letter_budget():
    shape = (1,) * 55 + (3, 2)
    a, b = _rand(shape, 1), _rand(shape, 2)
    cost, out = _billed(lambda: fnp.vecdot(a, b, axis=55))
    twin, _ = _billed(lambda: fnp.vecdot(_core_last(a, (55,)), _core_last(b, (55,))))
    assert cost == twin == 2 * (2 * 3 - 1)
    assert np.allclose(np.asarray(out), np.vecdot(a, b, axis=55))  # type: ignore[reportCallIssue]


@requires_vecdot
def test_symmetric_tag_survives_an_output_past_26_axes():
    # opt_einsum's symbol table runs a..z then A..Z, which is not sorted order;
    # the symmetric pair below straddles that boundary in the output.
    pad = (1,) * 25
    data = _symmetrized(_rand((4, *pad, 3, 3), 5), (26, 27))
    small = data.reshape(4, 3, 3)
    v = _rand((4,), 6)
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        S = flops.as_symmetric(data, symmetry=(26, 27))
        S_small = flops.as_symmetric(small, symmetry=(1, 2))
        before = int(ctx.flops_used)
        out = fnp.vecdot(S, v, axis=0)
        cost = int(ctx.flops_used) - before
        before = int(ctx.flops_used)
        fnp.vecdot(S_small, v, axis=0)
        small_cost = int(ctx.flops_used) - before
    group = getattr(out, "symmetry", None)
    assert group is not None
    assert _moved_axes(group) == {25, 26}
    assert np.allclose(np.asarray(out), np.vecdot(data, v, axis=0))  # type: ignore[reportCallIssue]
    assert cost == small_cost < 9 * (2 * 4 - 1)


@requires_vecdot
def test_default_layout_is_unchanged():
    a, b = _rand((4, 5, 6), 1), _rand((4, 5, 6), 2)
    cost, _ = _billed(lambda: fnp.vecdot(a, b))
    assert cost == 4 * 5 * (2 * 6 - 1)
    explicit, _ = _billed(lambda: fnp.vecdot(a, b, axis=-1))
    assert explicit == cost
