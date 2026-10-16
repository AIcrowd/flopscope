"""Tests for flopscope.numpy.linalg.svd."""

import numpy
import pytest

from flopscope._budget import BudgetContext
from flopscope.numpy.linalg import svd


def test_svd_full_result():
    numpy.random.seed(42)
    A = numpy.random.randn(10, 5)
    with BudgetContext(flop_budget=10**6) as budget:
        U, S, Vt = svd(A, full_matrices=False)
        U_np, S_np, Vt_np = numpy.linalg.svd(A, full_matrices=False)
        assert numpy.allclose(S, S_np)
        assert U.shape == (10, 5)
        assert Vt.shape == (5, 5)


def test_svd_full_cost():
    A = numpy.random.randn(10, 5)
    with BudgetContext(flop_budget=10**6) as budget:
        svd(A)
        # full_matrices=True (default), non-square (10,5): 4*a^2*b+22*b^3
        # a=10, b=5: 4*100*5+22*125=2000+2750=4750
        assert budget.flops_used == 4750


def test_svd_truncated_result():
    numpy.random.seed(42)
    A = numpy.random.randn(10, 5)
    with BudgetContext(flop_budget=10**6) as budget:
        U, S, Vt = svd(A, k=3)
        assert U.shape == (10, 3)
        assert S.shape == (3,)
        assert Vt.shape == (3, 5)
        _, S_full, _ = numpy.linalg.svd(A, full_matrices=False)
        assert numpy.allclose(S, S_full[:3])


def test_svd_truncated_cost():
    A = numpy.random.randn(10, 5)
    with BudgetContext(flop_budget=10**6) as budget:
        svd(A, k=3)
        # min(4mnk + 6(m+n)k² + 20k³, economy) = 1950
        assert budget.flops_used == 1950


def test_svd_truncated_cost_batched():
    A = numpy.random.randn(4, 10, 5)  # 4 matrices
    with BudgetContext(flop_budget=10**6) as budget:
        svd(A, k=3)
        # per-matrix 1950, x4 batch
        assert budget.flops_used == 7800


def test_svd_truncated_values_only_cost():
    A = numpy.random.randn(10, 5)
    with BudgetContext(flop_budget=10**6) as budget:
        svd(A, k=3, compute_uv=False)
        # min(4mnk + 2(m+n)k² + 2k³, values-only economy) = 750
        assert budget.flops_used == 750


def test_svd_not_2d():
    with BudgetContext(flop_budget=10**6):
        with pytest.raises(ValueError, match="2D"):
            svd(numpy.ones((3,)))


def test_svd_k_too_large():
    with BudgetContext(flop_budget=10**6):
        with pytest.raises(ValueError, match="k"):
            svd(numpy.ones((3, 5)), k=10)


def test_svd_rejects_nonpositive_k():
    with BudgetContext(flop_budget=10**6):
        for bad_k in (0, -1):
            with pytest.raises(ValueError, match="k"):
                svd(numpy.ones((4, 3)), k=bad_k)


def test_svd_outside_context():
    # Operations now auto-activate the global default budget instead of raising
    U, S, Vt = svd(numpy.ones((3, 3)))
    assert S.shape == (3,)


def test_svd_op_log():
    A = numpy.random.randn(8, 4)
    with BudgetContext(flop_budget=10**6) as budget:
        svd(A, k=2)
        assert budget.op_log[0].op_name == "linalg.svd"


def test_linalg_unsupported():
    from flopscope.numpy import linalg

    with pytest.raises(
        AttributeError,
        match="(does not provide|does not support|registered but not yet implemented)",
    ):
        _ = linalg.cross_decomposition
