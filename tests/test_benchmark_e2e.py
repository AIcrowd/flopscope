# tests/test_benchmark_e2e.py
"""End-to-end test: load weights JSON and verify flopscope applies them."""

import json
import os
import tempfile

from flopscope._budget import BudgetContext
from flopscope._weights import load_weights, reset_weights


def test_e2e_weights_affect_budget():
    """Full round trip: JSON -> load_weights -> flopscope budget uses them."""
    reset_weights()

    data = {"weights": {"exp": 10.0, "add": 1.0}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    try:
        load_weights(path)

        with BudgetContext(flop_budget=10_000_000) as budget:
            budget.deduct(
                "exp", flop_cost=1000, subscripts=None, shapes=((1000,),), dtypes=()
            )
            assert budget.flops_used == 10000  # 1000 * 10.0

            budget.deduct(
                "add", flop_cost=1000, subscripts=None, shapes=((1000,),), dtypes=()
            )
            assert budget.flops_used == 11000  # 10000 + 1000 * 1.0

            budget.deduct(
                "subtract",
                flop_cost=1000,
                subscripts=None,
                shapes=((1000,),),
                dtypes=(),
            )
            assert budget.flops_used == 12000  # not in config, default 1.0
    finally:
        reset_weights()
        os.unlink(path)


def test_e2e_no_weights_backward_compat():
    """Without weights, behavior is identical to before."""
    reset_weights()
    with BudgetContext(flop_budget=10_000_000) as budget:
        budget.deduct(
            "exp", flop_cost=1000, subscripts=None, shapes=((1000,),), dtypes=()
        )
        assert budget.flops_used == 1000  # weight = 1.0 (default)
