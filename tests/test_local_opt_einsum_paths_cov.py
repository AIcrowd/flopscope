"""Behavioral tests for flopscope's on-disk (vendored) opt_einsum path search.

``flopscope._opt_einsum._paths`` / ``._path_random`` are a symmetry-aware fork of
opt_einsum's path optimizers shipped inside the wheel. At runtime the package
``__getattr__`` hook redirects ``oe._paths`` / ``oe._path_random`` to the *upstream*
opt_einsum modules for the vendored test suite, so the on-disk copies are otherwise
only import-checked (see ``tests/accumulation/test_deletion_safety.py``). Here we
import them directly and assert they produce correct, cost-valid contraction
orderings -- i.e. every optimizer returns a path that, when handed to
``numpy.einsum``, reproduces the reference result, and the provably optimal path is
never more expensive than the greedy one.

Importing the on-disk submodules registers them on the package ``__dict__``, which
shadows the upstream redirect for the rest of the process. The autouse fixture drops
those attributes after each test so ``tests/test_opt_einsum_paths.py`` (which needs
the upstream modules) is unaffected on serial runs.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

import flopscope._opt_einsum as _oe_pkg

# subscripts -> per-index dimension sizes. Deliberately tiny and varied so the
# greedy/branch/dp/optimal search branches (chains, stars, outer products,
# Hadamard, full reductions) all get exercised without large intermediates.
_EQUATIONS: dict[str, tuple[str, dict[str, int]]] = {
    "chain3": ("ab,bc,cd->ad", {"a": 2, "b": 3, "c": 2, "d": 3}),
    "chain4": ("ab,bc,cd,de->ae", {"a": 2, "b": 2, "c": 3, "d": 2, "e": 3}),
    "chain5": ("ab,bc,cd,de,ef->af", dict.fromkeys("abcdef", 2)),
    "star": ("ax,bx,cx->abc", {"a": 2, "b": 3, "c": 2, "x": 4}),
    "scalar": ("ab,ab->", {"a": 3, "b": 4}),
    "outer": ("ab,cd->abcd", {"a": 2, "b": 3, "c": 2, "d": 2}),
    "hadamard": ("ab,ab->ab", {"a": 3, "b": 2}),
}


@pytest.fixture(autouse=True)
def _restore_upstream_shim():
    yield
    for _name in ("_paths", "_path_random"):
        _oe_pkg.__dict__.pop(_name, None)


@pytest.fixture
def paths() -> Any:
    return importlib.import_module("flopscope._opt_einsum._paths")


@pytest.fixture
def path_random() -> Any:
    return importlib.import_module("flopscope._opt_einsum._path_random")


def _setup(subscripts: str, dims: dict[str, int], seed: int = 0):
    lhs, _, rhs = subscripts.partition("->")
    terms = lhs.split(",")
    input_sets = [frozenset(t) for t in terms]
    output_set = frozenset(rhs)
    size_dict = {c: dims[c] for c in set("".join(terms)) | set(rhs)}
    rng = np.random.default_rng(seed)
    arrays = [rng.random(tuple(dims[c] for c in t)) for t in terms]
    return input_sets, output_set, size_dict, terms, arrays


def _is_complete(path, n_operands: int) -> bool:
    """A linear contraction path must consume every operand down to one tensor."""
    remaining = list(range(n_operands))
    for step in path:
        if not step or not all(0 <= p < len(remaining) for p in step):
            return False
        for pos in sorted(step, reverse=True):
            remaining.pop(pos)
        remaining.append(-1)
    return len(remaining) == 1


def _optimizer(paths_mod: Any, name: str):
    return {
        "optimal": paths_mod.optimal,
        "greedy": paths_mod.greedy,
        "branch_all": paths_mod.branch_all,
        "branch_2": paths_mod.branch_2,
        "dp": paths_mod.dynamic_programming,
        "auto": paths_mod.auto,
        "auto_hq": paths_mod.auto_hq,
    }[name]


_OPT_NAMES = ["optimal", "greedy", "branch_all", "branch_2", "dp", "auto", "auto_hq"]


@pytest.mark.parametrize("eq_key", list(_EQUATIONS))
@pytest.mark.parametrize("opt_name", _OPT_NAMES)
def test_local_optimizer_returns_correct_path(paths, opt_name, eq_key):
    subscripts, dims = _EQUATIONS[eq_key]
    input_sets, output_set, size_dict, terms, arrays = _setup(subscripts, dims)

    path = _optimizer(paths, opt_name)(input_sets, output_set, size_dict, None)

    assert _is_complete(path, len(terms)), (opt_name, eq_key, path)
    expected = np.einsum(subscripts, *arrays)
    got = np.einsum(subscripts, *arrays, optimize=["einsum_path", *path])
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_all_optimizers_agree_numerically(paths):
    subscripts, dims = _EQUATIONS["chain4"]
    input_sets, output_set, size_dict, terms, arrays = _setup(subscripts, dims)
    reference = np.einsum(subscripts, *arrays)
    for name in _OPT_NAMES:
        path = _optimizer(paths, name)(input_sets, output_set, size_dict, None)
        got = np.einsum(subscripts, *arrays, optimize=["einsum_path", *path])
        np.testing.assert_allclose(got, reference, rtol=1e-10, atol=1e-10)


def _oe_opt_cost(subscripts: str, arrays, path) -> int:
    """Independent FLOP-cost oracle: score an explicit path via upstream opt_einsum."""
    import opt_einsum as oe

    _, info = oe.contract_path(subscripts, *arrays, optimize=[tuple(s) for s in path])
    return int(info.opt_cost)


@pytest.mark.parametrize("eq_key", ["chain4", "chain5", "star"])
def test_optimal_never_worse_than_greedy(paths, eq_key):
    subscripts, dims = _EQUATIONS[eq_key]
    input_sets, output_set, size_dict, terms, arrays = _setup(subscripts, dims)

    opt_path = paths.optimal(input_sets, output_set, size_dict, None)
    grd_path = paths.greedy(input_sets, output_set, size_dict, None)

    opt_cost = _oe_opt_cost(subscripts, arrays, opt_path)
    grd_cost = _oe_opt_cost(subscripts, arrays, grd_path)
    assert opt_cost > 0
    assert opt_cost <= grd_cost


def test_ssa_linear_roundtrip(paths):
    ssa = [(0, 3), (2, 4), (1, 5)]
    assert paths.ssa_to_linear(ssa) == [(0, 3), (1, 2), (0, 1)]
    assert paths.linear_to_ssa(paths.ssa_to_linear(ssa)) == [(0, 3), (2, 4), (1, 5)]


def test_get_better_fn_semantics(paths):
    flops_better = paths.get_better_fn("flops")
    size_better = paths.get_better_fn("size")
    # (flops, size, best_flops, best_size): fewer flops wins for "flops"
    assert flops_better(10, 99, 20, 1) is True
    assert flops_better(20, 1, 10, 99) is False
    # smaller intermediate wins for "size"
    assert size_better(99, 10, 1, 20) is True
    with pytest.raises(KeyError):
        paths.get_better_fn("nonexistent")


def test_get_path_fn_known_and_unknown(paths):
    assert paths.get_path_fn("greedy") is paths.greedy
    assert paths.get_path_fn("optimal") is paths.optimal
    with pytest.raises(KeyError):
        paths.get_path_fn("does-not-exist")


def test_register_path_fn_and_duplicate_rejected(paths):
    calls: dict[str, int] = {"n": 0}

    def custom(inputs, output, size_dict, memory_limit=None, **kwargs):
        calls["n"] += 1
        return paths.greedy(inputs, output, size_dict, memory_limit)

    paths.register_path_fn("cov-custom-optimizer", custom)
    try:
        assert paths.get_path_fn("cov-custom-optimizer") is custom
        with pytest.raises(KeyError):
            paths.register_path_fn("cov-custom-optimizer", custom)
    finally:
        paths._PATH_OPTIONS.pop("cov-custom-optimizer", None)


def test_branchbound_rejects_bad_nbranch(paths):
    with pytest.raises(ValueError):
        paths.BranchBound(nbranch=0)


def test_memory_limited_search_still_correct(paths):
    subscripts, dims = _EQUATIONS["chain4"]
    input_sets, output_set, size_dict, terms, arrays = _setup(subscripts, dims)
    reference = np.einsum(subscripts, *arrays)
    # A finite memory limit routes greedy through the branch(nbranch=1) fallback
    # and constrains dynamic programming; both must still be correct.
    for fn in (paths.greedy, paths.dynamic_programming, paths.branch_2):
        path = fn(input_sets, output_set, size_dict, 64)
        assert _is_complete(path, len(terms))
        got = np.einsum(subscripts, *arrays, optimize=["einsum_path", *path])
        np.testing.assert_allclose(got, reference, rtol=1e-10, atol=1e-10)


def test_dynamic_programming_optimizer_object(paths):
    subscripts, dims = _EQUATIONS["chain5"]
    input_sets, output_set, size_dict, terms, arrays = _setup(subscripts, dims)
    optimizer = paths.DynamicProgramming(minimize="size")
    path = optimizer(input_sets, output_set, size_dict, None)
    assert _is_complete(path, len(terms))
    got = np.einsum(subscripts, *arrays, optimize=["einsum_path", *path])
    np.testing.assert_allclose(got, np.einsum(subscripts, *arrays))


def test_random_greedy_construction_and_validation(path_random):
    # __init__ wires the temperature/branch knobs and validates ``minimize``
    # without touching the (shadowed) path search.
    optimizer = path_random.RandomGreedy(
        max_repeats=8, minimize="flops", temperature=0.5, nbranch=4
    )
    assert optimizer.max_repeats == 8
    assert optimizer.minimize == "flops"
    # nbranch > 1 selects the thermal chooser; nbranch == 1 falls back to greedy.
    assert optimizer.choose_fn is not None
    assert path_random.RandomGreedy(nbranch=1).choose_fn is None
    with pytest.raises(ValueError):
        path_random.RandomGreedy(minimize="not-a-choice")
