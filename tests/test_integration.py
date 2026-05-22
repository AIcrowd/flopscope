"""End-to-end integration tests simulating participant usage."""

import numpy
import pytest

import flopscope as flops
import flopscope.numpy as fnp


def test_simple_mlp_forward_pass():
    """Simulate a participant doing a forward pass through a small MLP."""
    numpy.random.seed(42)
    width = 16
    depth = 4
    weights = [
        fnp.array(numpy.random.randn(width, width) * numpy.sqrt(2.0 / width))
        for _ in range(depth)
    ]

    with flops.BudgetContext(flop_budget=10**8) as budget:
        x = fnp.array(numpy.random.randn(100, width))
        for W in weights:
            x = fnp.einsum("bi,ji->bj", x, W)
            x = fnp.maximum(x, fnp.zeros_like(x))
        estimate = fnp.mean(x, axis=0)

        assert estimate.shape == (width,)
        assert budget.flops_used > 0
        summary = budget.summary()
        assert "einsum" in summary
        assert "maximum" in summary
        assert "mean" in summary
        einsum_ops = [r for r in budget.op_log if r.op_name == "einsum"]
        assert len(einsum_ops) == depth


def test_budget_tracking_accuracy():
    numpy.random.seed(42)
    A = fnp.array(numpy.random.randn(10, 20))
    B = fnp.array(numpy.random.randn(20, 30))

    with flops.BudgetContext(flop_budget=10**8) as budget:
        fnp.einsum("ij,jk->ik", A, B)  # new direct-event model: 12000 (was 6000)
        fnp.exp(fnp.ones((100,)))  # 100 * packaged_weight_exp
        fnp.sum(fnp.ones((50,)))  # 50
        total = budget.flops_used
        assert total > 0
        assert budget.flops_remaining == 10**8 - total


def test_flop_query_matches_execution():
    query_cost = flops.accounting.einsum_cost("ij,jk->ik", shapes=[(10, 20), (20, 30)])

    with flops.BudgetContext(flop_budget=10**8) as budget:
        A = fnp.array(numpy.random.randn(10, 20))  # 200 (array creation)
        B = fnp.array(numpy.random.randn(20, 30))  # 600 (array creation)
        fnp.einsum("ij,jk->ik", A, B)

    assert budget.flops_used == query_cost + 200 + 600


def test_mixed_free_and_counted():
    with flops.BudgetContext(flop_budget=1000) as budget:
        x = fnp.zeros((10, 10))
        x = fnp.reshape(x, (100,))
        x = fnp.ones((5, 5))
        x = fnp.transpose(x)
        assert budget.flops_used == 0
        fnp.exp(fnp.ones((10,)))
        assert budget.flops_used == 10


def test_budget_exhaustion_mid_computation():
    with pytest.raises(flops.BudgetExhaustedError) as exc_info:
        with flops.BudgetContext(flop_budget=500) as budget:
            fnp.exp(fnp.ones((100,)))  # 100
            fnp.exp(fnp.ones((100,)))  # 200
            fnp.exp(fnp.ones((100,)))  # 300
            fnp.exp(fnp.ones((100,)))  # 400
            fnp.exp(fnp.ones((100,)))  # 500
            fnp.exp(fnp.ones((100,)))  # would be 600 — exceeds!
    assert exc_info.value.flops_remaining == 0
