"""Behavioral coverage for einsum_path / PathInfo surfaces.

Covers the public path-planning API (``fnp.einsum_path``) and the PathInfo
adapter around upstream opt_einsum (``flopscope._opt_einsum.contract_path``):
cost-field invariants, plain/rich rendering including the pre-reduction and
symmetry-savings sub-rows, and the local-resolve branch for PathOptimizer
instances.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pytest
from rich.console import Console

import flopscope as flops
import flopscope.numpy as fnp
from flopscope import SymmetryGroup
from flopscope._symmetric import as_symmetric

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _render(info, verbose: bool = False, columns: int = 200) -> str:
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        no_color=True,
        _environ={"COLUMNS": str(columns), "LINES": "60"},
    )
    if verbose:
        console.print(info._rich_renderable(verbose=True))
    else:
        console.print(info)
    return ANSI_RE.sub("", buf.getvalue())


def _collapse(rendered: str) -> str:
    """Strip box-drawing and ALL whitespace so assertions survive the rich
    table's hard column wrapping."""
    return re.sub(r"[\s│┃╭╮╰╯─━┏┓┗┛┣┫┳┻╋]+", "", rendered)


@pytest.fixture
def budget():
    with flops.BudgetContext(flop_budget=10**12, quiet=True) as b:
        yield b


def test_einsum_path_matmul_chain_invariants(budget):
    a = np.ones((8, 4))
    b = np.ones((4, 16))
    c = np.ones((16, 2))
    path, info = fnp.einsum_path("ij,jk,kl->il", a, b, c, optimize="optimal")

    # Path is executable and reproduces the plain-numpy result.
    with np.errstate(all="ignore"):
        expected = np.einsum("ij,jk,kl->il", a, b, c)
    got = fnp.einsum("ij,jk,kl->il", a, b, c, optimize=path)
    np.testing.assert_allclose(np.asarray(got), expected)

    assert info.eq == "ij,jk,kl->il"
    assert 0 < info.optimized_cost <= info.naive_cost
    assert info.speedup == pytest.approx(info.naive_cost / info.optimized_cost)
    # Intermediates for either association order: (8,16) or (4,2) matrix, plus
    # the (8,2) output; the largest is capped by the biggest of these.
    assert info.largest_intermediate in (128, 16)
    assert len(info.steps) == len(path) == 2
    assert info.optimizer_used == "optimal"


def test_einsum_path_charges_unit_probe(budget):
    a = np.ones((3, 3))
    before = budget.flops_used
    fnp.einsum_path("ij,jk->ik", a, a)
    # Documented: path planning itself is (near-)free -- one probe FLOP.
    assert budget.flops_used == before + 1


def test_pathinfo_str_and_rich_render(budget):
    a = np.ones((8, 4))
    b = np.ones((4, 16))
    c = np.ones((16, 2))
    _, info = fnp.einsum_path("ij,jk,kl->il", a, b, c, optimize="greedy")

    plain = str(info)
    assert "Complete contraction:  ij,jk,kl->il" in plain
    assert "Naive cost" in plain and "Optimized cost" in plain
    assert f"{info.largest_intermediate:,} elements" in plain
    assert "Optimizer:" in plain and "greedy" in plain

    collapsed = _collapse(_render(info))
    assert "Completecontraction:ij,jk,kl->il" in collapsed
    # Index sizes are grouped by size in the header pills.
    assert "Indexsizes:" in collapsed
    assert "Optimizer:greedy" in collapsed

    verbose = _collapse(_render(info, verbose=True))
    assert "Completecontraction" in verbose


def test_pathinfo_render_shows_pre_reductions(budget):
    # j lives only in the first operand: the step pre-reduces op0 over j and
    # the renderer prints the pre-reduce and residual-contraction sub-rows.
    a = np.ones((3, 5, 4))
    b = np.ones((4, 2))
    _, info = fnp.einsum_path("ijk,kl->il", a, b)

    collapsed = _collapse(_render(info))
    # Pre-reducing op0 over j costs 4 adds per surviving (i,k) cell:
    # 3*4*4 = 48 ops; the residual pairwise contraction gets the other 42.
    assert "pre-reduceop0{j}:48ops" in collapsed
    assert "residualcontraction:" in collapsed
    assert "42ops" in collapsed
    step = info.steps[0]
    assert step.pre_reductions
    pre = step.pre_reductions[0]
    assert pre.removed_labels == ("j",)
    assert pre.cost == 48
    assert step.flop_cost - pre.cost == 42


def test_pathinfo_render_symmetric_savings_column(budget):
    # Symmetric inputs surface per-step savings markers and the group name.
    rng = np.random.default_rng(0)
    base = rng.random((6, 6))
    s2 = SymmetryGroup.symmetric(axes=(0, 1))
    sym = as_symmetric((base + base.T) / 2, symmetry=s2)

    _, info = fnp.einsum_path("ij,jk->ik", sym, sym)
    rendered = _render(info)
    assert "S2" in rendered

    plain_info = fnp.einsum_path("ij,jk->ik", np.ones((6, 6)), np.ones((6, 6)))[1]
    assert "S2" not in _render(plain_info)


def test_einsum_explicit_path_and_cache_roundtrip(budget):
    a = np.ones((4, 5))
    b = np.ones((5, 6))
    c = np.ones((6, 3))
    with np.errstate(all="ignore"):
        expected = np.einsum("ij,jk,kl->il", a, b, c)

    for optimize in (
        [(0, 1), (0, 1)],
        ((0, 1), (0, 1)),  # tuple form is accepted at runtime
        False,
        True,
    ):
        got = fnp.einsum("ij,jk,kl->il", a, b, c, optimize=optimize)  # pyright: ignore[reportArgumentType]
        np.testing.assert_allclose(np.asarray(got), expected)

    # Same subscripts+shapes+optimizer hit the LRU path cache.
    fnp.clear_einsum_cache()
    fnp.einsum("ij,jk,kl->il", a, b, c)
    info1 = fnp.einsum_cache_info()
    fnp.einsum("ij,jk,kl->il", a, b, c)
    info2 = fnp.einsum_cache_info()
    assert info2.hits == info1.hits + 1


def test_contract_path_adapter_with_path_optimizer_instance():
    import flopscope._opt_einsum as oe

    a = np.ones((3, 4))
    b = np.ones((4, 5))
    c = np.ones((5, 2))

    optimizer = oe.BranchBound(nbranch=1)
    path, info = oe.contract_path("ij,jk,kl->il", a, b, c, optimize=optimizer)
    assert path == [(0, 1), (0, 1)] or path == [(1, 2), (0, 1)]
    assert info.optimizer_used == "BranchBound"
    assert info.optimized_cost > 0

    # shapes=True accepts bare shape tuples instead of arrays.
    path2, info2 = oe.contract_path(
        "ij,jk,kl->il",
        (3, 4),
        (4, 5),
        (5, 2),
        shapes=True,
        optimize=oe.BranchBound(nbranch=1),
    )
    assert path2 == path
    # memory_limit branches of the local resolver.
    path3, _ = oe.contract_path(
        "ij,jk,kl->il",
        a,
        b,
        c,
        optimize=oe.BranchBound(nbranch=1),
        memory_limit="max_input",
    )
    assert len(path3) == 2
    path4, _ = oe.contract_path(
        "ij,jk,kl->il",
        a,
        b,
        c,
        optimize=oe.BranchBound(nbranch=1),
        memory_limit=10**6,
    )
    assert len(path4) == 2


def test_contract_path_adapter_string_and_trivial_labels():
    import flopscope._opt_einsum as oe

    a = np.ones((3, 4))
    b = np.ones((4, 5))
    c = np.ones((5, 2))
    # Two operands: no optimizer runs; the label is 'trivial'.
    _, info2 = oe.contract_path("ij,jk->ik", a, b, optimize="greedy")
    assert info2.optimizer_used == "trivial"
    # 'auto' resolves to the inner choice for three operands (optimal).
    _, info3 = oe.contract_path("ij,jk,kl->il", a, b, c, optimize="auto")
    assert info3.optimizer_used == "optimal"
    _, info_hq = oe.contract_path("ij,jk,kl->il", a, b, c, optimize="auto-hq")
    assert info_hq.optimizer_used == "optimal"
    # opt_cost mirrors upstream's Decimal-typed legacy cost field.
    from decimal import Decimal

    assert isinstance(info3.opt_cost, Decimal)
    assert info3.opt_cost > 0
