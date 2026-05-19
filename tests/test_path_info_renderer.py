"""Tests for FlopscopePathInfo's renderer + check_consistency utility."""

import flopscope as flops
import flopscope.numpy as fnp


def test_check_consistency_returns_true_for_healthy_info():
    x = fnp.ones((4, 4))
    with flops.BudgetContext(flop_budget=10**12, quiet=True):
        _, info = fnp.einsum_path("ij,jk,kl->il", x, x, x)
    assert info.check_consistency() is True


def test_check_consistency_raises_on_forced_desync():
    """If accumulation.total is forced to disagree with sum(steps.flop_cost),
    check_consistency raises with a clear message."""
    from dataclasses import replace

    x = fnp.ones((4, 4))
    with flops.BudgetContext(flop_budget=10**12, quiet=True):
        _, info = fnp.einsum_path("ij,jk,kl->il", x, x, x)
    # Force a mismatch by replacing accumulation.total
    bad_acc = replace(info.accumulation, total=999999)
    info.accumulation = bad_acc

    import pytest
    with pytest.raises(AssertionError, match="check_consistency"):
        info.check_consistency()
