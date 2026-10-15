"""einsum tags its output symmetry on the axes that carry it (#281).

The oracle that infers an einsum step's output symmetry builds the group on
its own label frame, the step's free labels in sorted order. The output tag
was read as if the group's axes were positions in the requested output
subscript, which holds only when that subscript lists its labels in sorted
order. ``ij,k->ijk`` was tagged right, but ``ij,k->kij`` put the group on
axes that do not carry the symmetry: the result failed its own validation
(``SymmetryError``), or, with unequal extents, failed to broadcast.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


def _symmetrized(x, axes):
    perms = list(itertools.permutations(axes))
    total = np.zeros_like(x)
    for perm in perms:
        order = list(range(x.ndim))
        for src, dst in zip(axes, perm, strict=True):
            order[src] = dst
        total = total + np.transpose(x, order)
    return total / len(perms)


def _moved_axes(group):
    """Output axes some element of ``group`` actually moves."""
    axes = group.axes if group.axes is not None else tuple(range(group.degree))
    return {
        axes[i]
        for gen in group.generators
        for i, j in enumerate(gen.array_form)
        if i != j
    }


rng = np.random.default_rng(0)
S = _symmetrized(rng.standard_normal((5, 5)), (0, 1))
T = _symmetrized(rng.standard_normal((5, 5, 5)), (0, 1, 2))
V = rng.standard_normal(5)
M = rng.standard_normal((5, 4))

# spec, operands (symmetric operand first, with its tagged axes), the twin spec
# whose symmetric labels lead, and the output axes that hold the symmetric labels
CASES = [
    ("ij,k->kij", (S, (0, 1)), (V,), "ij,k->ijk", {1, 2}),
    ("ij,k->ikj", (S, (0, 1)), (V,), "ij,k->ijk", {0, 2}),
    ("ijk,il->ljk", (T, (0, 1, 2)), (M,), "ijk,il->jkl", {1, 2}),
    ("ijk,il->jlk", (T, (0, 1, 2)), (M,), "ijk,il->jkl", {0, 2}),
    ("ijk,li->lkj", (T, (0, 1, 2)), (M.T.copy(),), "ijk,li->jkl", {1, 2}),
    ("Xy,z->zXy", (S, (0, 1)), (V,), "Xy,z->Xyz", {1, 2}),
    ("ij,k,l->lkij", (S, (0, 1)), (V, V[:4].copy()), "ij,k,l->ijkl", {2, 3}),
    ("ij,k->ijk", (S, (0, 1)), (V,), "ij,k->ijk", {0, 1}),
]


@pytest.mark.parametrize(("spec", "sym", "others", "twin", "tagged"), CASES)
def test_output_symmetry_lands_on_the_symmetric_labels(spec, sym, others, twin, tagged):
    data, sym_axes = sym
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
            tagged_operand = flops.as_symmetric(data, symmetry=sym_axes)
            before = int(ctx.flops_used)
            out = fnp.einsum(spec, tagged_operand, *others)
            cost = int(ctx.flops_used) - before
            before = int(ctx.flops_used)
            fnp.einsum(twin, tagged_operand, *others)
            twin_cost = int(ctx.flops_used) - before
    expected = np.einsum(spec, data, *others)
    raw = np.asarray(out)
    group = getattr(out, "symmetry", None)
    assert raw.shape == expected.shape
    assert np.allclose(raw, expected)
    assert group is not None
    assert _moved_axes(group) == tagged
    for a, b in itertools.combinations(sorted(tagged), 2):
        assert np.allclose(raw, np.swapaxes(raw, a, b))
    assert cost == twin_cost


def test_tagged_result_feeds_a_later_contraction():
    # The issue's workaround spelling and the direct one now agree downstream.
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        tagged = flops.as_symmetric(T, symmetry=(0, 1, 2))
        direct = fnp.einsum("ijk,il->ljk", tagged, M)
        out = fnp.einsum("ljk,jk->l", direct, S)
    assert np.allclose(np.asarray(out), np.einsum("ijk,il,jk->l", T, M, S))
