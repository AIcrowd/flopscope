"""Deeper behavioral coverage for the vendored symmetry-aware path search.

Complements ``test_local_opt_einsum_paths_cov.py``:

* oracle-driven search — the vendored fork's distinguishing feature is that every
  optimizer accepts a ``SubgraphSymmetryOracle`` and discounts contraction FLOPs
  by the symmetry of each intermediate. We assert the discounted cost is a real
  discount (0 < symmetric cost <= dense cost) and the returned paths stay valid.
* DynamicProgramming ``minimize`` grammar ('flops'/'size'/'write'/'combo'/'limit',
  custom factor suffix, callable, invalid -> ValueError).
* memory-limited optimal/branch search (oversize-flops handling).
* the local ``_path_random`` machinery, executed with its intended sibling
  binding. ``_path_random.py`` does ``from . import _paths as paths``, but the
  package-level PEP 562 ``__getattr__`` (which maps ``_paths`` to *upstream*
  opt_einsum for the vendored test suite) answers the ``from . import`` attribute
  probe first, so a bare import silently binds upstream ``paths`` — whose
  functions reject the fork's ``symmetry_oracle``/``ssa_to_subset`` kwargs. The
  ``rewired_path_random`` fixture re-executes the module with the package
  attribute pointing at the on-disk sibling (exactly what the relative import
  names) and restores all import state afterwards.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

import flopscope._opt_einsum as _oe_pkg
from flopscope import SymmetryGroup


@pytest.fixture(autouse=True)
def _restore_upstream_shim():
    yield
    for _name in ("_paths", "_path_random"):
        _oe_pkg.__dict__.pop(_name, None)


@pytest.fixture
def paths() -> Any:
    return importlib.import_module("flopscope._opt_einsum._paths")


@pytest.fixture
def rewired_path_random(paths) -> Any:
    """The local ``_path_random`` executed with ``paths`` bound to its on-disk
    sibling (the binding its ``from . import _paths as paths`` line names)."""
    saved_mod = sys.modules.pop("flopscope._opt_einsum._path_random", None)
    saved_attr = _oe_pkg.__dict__.pop("_path_random", None)
    _oe_pkg.__dict__["_paths"] = paths
    try:
        module = importlib.import_module("flopscope._opt_einsum._path_random")
        assert module.paths is paths
        yield module
    finally:
        sys.modules.pop("flopscope._opt_einsum._path_random", None)
        _oe_pkg.__dict__.pop("_path_random", None)
        if saved_mod is not None:
            sys.modules["flopscope._opt_einsum._path_random"] = saved_mod
        if saved_attr is not None:
            _oe_pkg.__dict__["_path_random"] = saved_attr


# ── shared 4-cycle fixture: ij,jk,kl,li->  with four identical S2-symmetric ops ──

_CYCLE_TERMS = ["ij", "jk", "kl", "li"]
_CYCLE_SUBS = "ij,jk,kl,li->"
_N = 4


def _cycle_setup():
    rng = np.random.default_rng(7)
    base = rng.random((_N, _N))
    sym_op = (base + base.T) / 2.0
    arrays = [sym_op] * 4
    input_sets = [frozenset(t) for t in _CYCLE_TERMS]
    output_set: frozenset[str] = frozenset()
    size_dict = dict.fromkeys("ijkl", _N)
    return input_sets, output_set, size_dict, arrays


def _cycle_oracle():
    from flopscope._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle

    s2 = SymmetryGroup.symmetric(axes=(0, 1))
    dummy = np.empty((_N, _N))
    return SubgraphSymmetryOracle(
        operands=[dummy] * 4,
        subscript_parts=list(_CYCLE_TERMS),
        per_op_groups=[[s2], [s2], [s2], [s2]],
        output_chars="",
    )


def _is_complete(path, n_operands: int) -> bool:
    remaining = list(range(n_operands))
    for step in path:
        if not step or not all(0 <= p < len(remaining) for p in step):
            return False
        for pos in sorted(step, reverse=True):
            remaining.pop(pos)
        remaining.append(-1)
    return len(remaining) == 1


def test_oracle_greedy_search_returns_valid_path(paths):
    # greedy is the one vendored optimizer whose search accepts the oracle end
    # to end (its ssa machinery prices candidates itself). The oracle-driven
    # per-step costing used by optimal/branch/dp (calc_k12_flops with oracle)
    # dereferences the retired 'use_inner_symmetry' setting and raises KeyError
    # -- a latent defect in this dormant vendored module, deliberately not
    # asserted as behavior here.
    input_sets, output_set, size_dict, arrays = _cycle_setup()
    oracle = _cycle_oracle()

    path = paths.greedy(input_sets, output_set, size_dict, None, symmetry_oracle=oracle)

    assert _is_complete(path, 4), path
    got = np.einsum(_CYCLE_SUBS, *arrays, optimize=["einsum_path", *path])
    np.testing.assert_allclose(got, np.einsum(_CYCLE_SUBS, *arrays), rtol=1e-10)


def test_calc_k12_flops_dense_matches_flop_count(paths):
    # Plain pairwise ij,jk -> ik with all dims 3: no oracle means the fork's
    # generic flop_count formula prices the step: (2K-1)*M = (2*3-1)*9 = 45.
    from flopscope._opt_einsum._helpers import flop_count

    inputs = (frozenset("ij"), frozenset("jk"))
    output = frozenset("ik")
    size_dict = {"i": 3, "j": 3, "k": 3}
    k12, flops, sym = paths.calc_k12_flops(
        inputs, output, frozenset({0, 1}), 0, 1, size_dict
    )
    assert k12 == frozenset("ik")
    assert sym is None
    assert flops == 45
    assert flops == flop_count(
        "ijk",
        True,
        2,
        size_dict,
        input_subscripts=("ij", "jk"),
        output_subscript="ik",
        input_shapes=((3, 3), (3, 3)),
    )


def test_dp_minimize_grammar(paths):
    input_sets, output_set, size_dict, arrays = _cycle_setup()
    reference = np.einsum(_CYCLE_SUBS, *arrays)
    for minimize in ["flops", "size", "write", "combo", "limit", "combo-2"]:
        optimizer = paths.DynamicProgramming(minimize=minimize)
        path = optimizer(input_sets, output_set, size_dict, None)
        assert _is_complete(path, 4), minimize
        got = np.einsum(_CYCLE_SUBS, *arrays, optimize=["einsum_path", *path])
        np.testing.assert_allclose(got, reference, rtol=1e-10)


def test_dp_minimize_custom_callable_and_invalid(paths):
    # A callable minimize is accepted verbatim; junk strings raise ValueError.
    fn, scale = paths._parse_minimize("flops")
    assert callable(fn) and scale == 1
    custom = lambda *a, **k: None  # noqa: E731
    got_fn, got_scale = paths._parse_minimize(custom)
    assert got_fn is custom and got_scale == float("inf")
    with pytest.raises(ValueError):
        paths._parse_minimize("not-a-strategy")


def test_dp_handles_disconnected_and_single_term(paths):
    # Disconnected product graph: ab,cd->abcd has no shared indices.
    size_dict = {"a": 2, "b": 2, "c": 2, "d": 2}
    rng = np.random.default_rng(3)
    x, y = rng.random((2, 2)), rng.random((2, 2))
    path = paths.dynamic_programming(
        [frozenset("ab"), frozenset("cd")], frozenset("abcd"), size_dict, None
    )
    assert _is_complete(path, 2)
    got = np.einsum("ab,cd->abcd", x, y, optimize=["einsum_path", *path])
    np.testing.assert_allclose(got, np.einsum("ab,cd->abcd", x, y))
    # Single-operand contraction is the trivial path.
    single = paths.dynamic_programming(
        [frozenset("ab")], frozenset("a"), {"a": 2, "b": 3}, None
    )
    assert single == [(0,)]


def test_memory_limited_optimal_and_branch_all(paths):
    input_sets, output_set, size_dict, arrays = _cycle_setup()
    reference = np.einsum(_CYCLE_SUBS, *arrays)
    # memory_limit=1 forces every candidate intermediate over the cap, driving
    # the oversize/flat-contraction handling; result must stay correct.
    for fn in (paths.optimal, paths.branch_all):
        path = fn(input_sets, output_set, size_dict, 1)
        assert _is_complete(path, 4)
        got = np.einsum(_CYCLE_SUBS, *arrays, optimize=["einsum_path", *path])
        np.testing.assert_allclose(got, reference, rtol=1e-10)


def test_greedy_jitter_cost_fn(paths):
    input_sets, output_set, size_dict, arrays = _cycle_setup()
    path = paths.greedy(
        input_sets, output_set, size_dict, None, cost_fn="memory-removed-jitter"
    )
    assert _is_complete(path, 4)
    got = np.einsum(_CYCLE_SUBS, *arrays, optimize=["einsum_path", *path])
    np.testing.assert_allclose(got, np.einsum(_CYCLE_SUBS, *arrays), rtol=1e-10)


def test_package_getattr_unknown_name_raises():
    with pytest.raises(AttributeError):
        _oe_pkg.__getattr__("_no_such_submodule")


# ── local _path_random, executed with its intended sibling binding ──


def test_rewired_random_greedy_finds_valid_paths(rewired_path_random, paths):
    pr = rewired_path_random
    input_sets, output_set, size_dict, arrays = _cycle_setup()

    optimizer = pr.RandomGreedy(max_repeats=6, minimize="flops")
    path = optimizer(input_sets, output_set, size_dict, None)

    assert _is_complete(path, 4)
    assert optimizer.path == path
    got = np.einsum(_CYCLE_SUBS, *arrays, optimize=["einsum_path", *path])
    np.testing.assert_allclose(got, np.einsum(_CYCLE_SUBS, *arrays), rtol=1e-10)
    # One cost/size recorded per trial; the tracked best matches the extremes.
    assert len(optimizer.costs) == 6
    assert optimizer.best["flops"] == min(optimizer.costs)
    assert optimizer.best["size"] > 0
    # Repeated calls accumulate further trials on the same optimizer.
    optimizer.max_repeats = 2
    optimizer(input_sets, output_set, size_dict, None)
    assert len(optimizer.costs) == 8


def test_rewired_random_greedy_choosers(rewired_path_random, paths):
    pr = rewired_path_random
    input_sets, output_set, size_dict, arrays = _cycle_setup()
    reference = np.einsum(_CYCLE_SUBS, *arrays)
    # temperature=0 keeps only ties with the best candidate; nbranch=1 falls
    # back to the deterministic greedy chooser; rel_temperature=False uses the
    # absolute temperature scale. All must produce valid, correct paths.
    for kwargs in [
        {"temperature": 0.0, "max_repeats": 3},
        {"nbranch": 1, "max_repeats": 3},
        {"rel_temperature": False, "temperature": 2.0, "max_repeats": 3},
    ]:
        optimizer = pr.RandomGreedy(**kwargs)
        path = optimizer(input_sets, output_set, size_dict, None)
        assert _is_complete(path, 4), kwargs
        got = np.einsum(_CYCLE_SUBS, *arrays, optimize=["einsum_path", *path])
        np.testing.assert_allclose(got, reference, rtol=1e-10)


def test_rewired_random_greedy_function_and_128_alias(rewired_path_random, paths):
    pr = rewired_path_random
    input_sets, output_set, size_dict, arrays = _cycle_setup()
    path = pr.random_greedy(input_sets, output_set, size_dict, max_repeats=4)
    assert _is_complete(path, 4)
    got = np.einsum(_CYCLE_SUBS, *arrays, optimize=["einsum_path", *path])
    np.testing.assert_allclose(got, np.einsum(_CYCLE_SUBS, *arrays), rtol=1e-10)
    assert pr.random_greedy_128.keywords == {"max_repeats": 128}


def test_rewired_ssa_path_compute_cost_dense_exact(rewired_path_random, paths):
    # Left-to-right chain on the 4-cycle: ij,jk->ik (112 = (2*4-1)*16), then
    # ik,kl->il (112), then il,li-> (16 multiplies + 15 adds = 31). Total 255,
    # and the largest intermediate is the 4x4 = 16-cell matrix.
    pr = rewired_path_random
    input_sets, output_set, size_dict, _ = _cycle_setup()
    ssa_path = paths.linear_to_ssa([(0, 1), (0, 1), (0, 1)])

    dense_cost, dense_size = pr.ssa_path_compute_cost(
        ssa_path, input_sets, output_set, size_dict
    )
    assert dense_cost == 112 + 112 + 31
    assert dense_size == 16


def test_rewired_max_time_stops_early(rewired_path_random, paths):
    pr = rewired_path_random
    input_sets, output_set, size_dict, _ = _cycle_setup()
    optimizer = pr.RandomGreedy(max_repeats=10_000, max_time=0.0)
    path = optimizer(input_sets, output_set, size_dict, None)
    assert _is_complete(path, 4)
    # max_time=0 stops after the first assessed trial.
    assert len(optimizer.costs) < 10_000


# ── _symmetry.py: Burnside counting + symmetric flop discounts ──


def _flop_kwargs(terms, output, size_dict):
    return {
        "input_subscripts": tuple(terms),
        "output_subscript": output,
        "input_shapes": tuple(tuple(size_dict[c] for c in t) for t in terms),
    }


def test_unique_elements_burnside_counts():
    from flopscope._opt_einsum._symmetry import unique_elements

    sizes = dict.fromkeys("ijkl", 4)
    # No group: dense product. Empty index set: single scalar cell.
    assert unique_elements(frozenset("ik"), sizes) == 16
    assert unique_elements(frozenset(), sizes) == 1
    # S2 on two n=4 axes: n(n+1)/2 = 10 distinct cells.
    s2 = SymmetryGroup.symmetric(axes=(0, 1))
    assert unique_elements(frozenset("ik"), sizes, perm_group=s2) == 10
    # Labels outside the group multiply densely: 10 * 4.
    assert unique_elements(frozenset("ikl"), sizes, perm_group=s2) == 40


def test_unique_elements_oracle_inner_group_is_d4():
    # The 4-cycle's fully-contracted subset carries the dihedral D4 group on
    # (i,j,k,l); Burnside gives the classic 4-color bracelet count 55.
    from flopscope._opt_einsum._symmetry import unique_elements

    ss = _cycle_oracle().sym(frozenset({0, 1, 2, 3}))
    assert ss.inner is not None
    sizes = dict.fromkeys("ijkl", 4)
    assert unique_elements(frozenset("ijkl"), sizes, perm_group=ss.inner) == 55


def test_symmetric_flop_count_output_discount_exact():
    from flopscope._opt_einsum._helpers import flop_count
    from flopscope._opt_einsum._symmetry import symmetric_flop_count

    sizes = dict.fromkeys("ijk", 4)
    kwargs = _flop_kwargs(("ij", "jk"), "ik", sizes)
    dense = flop_count("ijk", True, 2, sizes, **kwargs)
    assert dense == (2 * 4 - 1) * 16  # (2K-1)*M matmul

    s2 = SymmetryGroup.symmetric(axes=(0, 1))
    discounted = symmetric_flop_count(
        "ijk",
        True,
        2,
        sizes,
        output_group=s2,
        output_indices=frozenset("ik"),
        **kwargs,
    )
    # Only the 10 of 16 output cells distinct under S2 are priced.
    assert discounted == dense * 10 // 16


def test_symmetric_flop_count_inner_discount_and_gates():
    from flopscope._opt_einsum._helpers import flop_count
    from flopscope._opt_einsum._symmetry import symmetric_flop_count, unique_elements

    ss = _cycle_oracle().sym(frozenset({0, 1, 2, 3}))
    sizes = dict.fromkeys("ijkl", 4)
    kwargs = _flop_kwargs(_CYCLE_TERMS, "", sizes)
    dense = flop_count("ijkl", True, 4, sizes, **kwargs)
    unique_inner = unique_elements(frozenset("ijkl"), sizes, perm_group=ss.inner)

    discounted = symmetric_flop_count(
        "ijkl",
        True,
        4,
        sizes,
        inner_group=ss.inner,
        inner_indices=frozenset("ijkl"),
        **kwargs,
    )
    assert discounted == max(dense * unique_inner // 256, 1)
    assert 0 < discounted < dense

    # use_inner_symmetry=False turns the discount off.
    assert (
        symmetric_flop_count(
            "ijkl",
            True,
            4,
            sizes,
            inner_group=ss.inner,
            inner_indices=frozenset("ijkl"),
            use_inner_symmetry=False,
            **kwargs,
        )
        == dense
    )
    # If any group label is no longer among the contracted indices, the
    # inner reduction is skipped entirely.
    assert (
        symmetric_flop_count(
            "ijkl",
            True,
            4,
            sizes,
            inner_group=ss.inner,
            inner_indices=frozenset("ijk"),
            **kwargs,
        )
        == dense
    )


# ── _helpers.py: size products, contraction finding, FMA=2 flop counts ──


def test_compute_size_by_dict():
    from flopscope._opt_einsum._helpers import compute_size_by_dict

    assert compute_size_by_dict("abbc", {"a": 2, "b": 3, "c": 5}) == 90
    assert compute_size_by_dict([], {}) == 1


def test_find_contraction():
    from flopscope._opt_einsum._helpers import find_contraction

    isets = [frozenset("ab"), frozenset("bc")]
    new_result, remaining, idx_removed, idx_contract = find_contraction(
        (0, 1), isets, frozenset("ac")
    )
    assert new_result == frozenset("ac")
    assert remaining == [frozenset("ac")]
    assert idx_removed == frozenset("b")
    assert idx_contract == frozenset("abc")

    isets3 = [frozenset("abd"), frozenset("ac"), frozenset("bdc")]
    new_result, remaining, idx_removed, idx_contract = find_contraction(
        (0, 2), isets3, frozenset("ac")
    )
    assert new_result == frozenset("ac")
    assert remaining == [frozenset("ac"), frozenset("ac")]
    assert idx_removed == frozenset("bd")
    assert idx_contract == frozenset("abcd")


def test_flop_count_fma2_values_and_required_kwargs():
    from flopscope._opt_einsum._helpers import (
        flop_count,
        has_array_interface,
    )

    # ab,bc->ac matmul (2,3)x(3,5): 2*2*3*5 - 2*5 = 50 (FMA=2, init credit).
    assert (
        flop_count(
            "abc",
            True,
            2,
            {"a": 2, "b": 3, "c": 5},
            input_subscripts=("ab", "bc"),
            output_subscript="ac",
            input_shapes=((2, 3), (3, 5)),
        )
        == 50
    )
    # Full reduction of 6 cells: 5 adds.
    assert (
        flop_count(
            "ab",
            True,
            1,
            {"a": 2, "b": 3},
            input_subscripts=("ab",),
            output_subscript="",
            input_shapes=((2, 3),),
        )
        == 5
    )
    with pytest.raises(ValueError):
        flop_count("ab", True, 1, {"a": 2, "b": 3})
    assert has_array_interface(np.zeros(2)) is True
    assert has_array_interface(object()) is False
