"""Tests verifying that einsum() charges the new accumulation cost."""

import numpy as np

import flopscope as fps
import flopscope.numpy as fnp


def test_einsum_charges_accumulation_total_for_simple_matmul():
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    expected_cost = fps.einsum_accumulation_cost("ij,jk->ik", A, B).total

    with fps.BudgetContext(flop_budget=10_000, quiet=True) as ctx:
        fnp.einsum("ij,jk->ik", A, B)

    spent = ctx.flops_used
    # Allow a small einsum_path overhead (it deducts 1 in the path-only branch);
    # the einsum() call itself should be the dominant charge.
    assert spent >= expected_cost
    assert spent <= expected_cost + 10  # allow tiny overhead


def test_einsum_path_info_carries_accumulation_field():
    A = np.zeros((4, 4))

    with fps.BudgetContext(flop_budget=10**12, quiet=True):
        path, info = fnp.einsum_path("ij,jk->ik", A, A)

    assert hasattr(info, "accumulation")
    assert info.accumulation is not None
    assert info.optimized_cost == info.accumulation.total
