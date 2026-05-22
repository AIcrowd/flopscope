"""Tests verifying the boundary swap: contract_path now wraps upstream.

After Task 6, flopscope._opt_einsum.contract_path() delegates to upstream
opt_einsum.contract_path() and adapts the result via build_path_info.
"""

import numpy as np

from flopscope._opt_einsum import PathInfo, contract_path


def test_contract_path_returns_flopscope_pathinfo():
    A = np.zeros((3, 4))
    B = np.zeros((4, 5))
    path, info = contract_path("ij,jk->ik", A, B, shapes=False)
    assert isinstance(info, PathInfo)


def test_contract_path_path_is_iterable_of_int_tuples():
    A = np.zeros((3, 4))
    B = np.zeros((4, 5))
    path, info = contract_path("ij,jk->ik", A, B, shapes=False)
    assert isinstance(path, list)
    for entry in path:
        assert isinstance(entry, tuple)
        assert all(isinstance(i, int) for i in entry)


def test_contract_path_three_operand_chain():
    A = np.zeros((3, 4))
    B = np.zeros((4, 5))
    C = np.zeros((5, 6))
    path, info = contract_path("ij,jk,kl->il", A, B, C, shapes=False)
    assert len(info.steps) == 2  # two pairwise contractions
    assert info.optimized_cost == sum(s.flop_count for s in info.steps)


def test_contract_path_with_shapes_true():
    """contract_path supports shapes=True for shape-only invocation."""
    path, info = contract_path(
        "ij,jk->ik",
        (3, 4),
        (4, 5),
        shapes=True,
    )
    assert isinstance(info, PathInfo)
    assert info.optimized_cost > 0
