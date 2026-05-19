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


def test_step_info_populated_with_diagnostics():
    """Every step in a multi-operand path must have dense_flop_cost,
    symmetry_savings, input_groups populated (output_group/inner_group
    may be None for dense intermediates, which is OK)."""
    x = fnp.ones((10, 10))
    with flops.BudgetContext(flop_budget=10**12, quiet=True):
        _, info = fnp.einsum_path("ij,jk,kl->il", x, x, x)

    for i, step in enumerate(info.steps):
        assert step.dense_flop_cost > 0, f"step {i}: dense_flop_cost not populated"
        assert 0.0 <= step.symmetry_savings <= 1.0, (
            f"step {i}: symmetry_savings={step.symmetry_savings} out of range"
        )
        assert isinstance(step.input_groups, list), (
            f"step {i}: input_groups not a list"
        )


def test_format_table_includes_dense_flops_and_savings_columns():
    x = fnp.ones((4, 4))
    with flops.BudgetContext(flop_budget=10**12, quiet=True):
        _, info = fnp.einsum_path("ij,jk,kl->il", x, x, x)
    rendered = str(info)
    assert "dense_flops" in rendered, f"missing dense_flops column:\n{rendered}"
    assert "savings" in rendered, f"missing savings column:\n{rendered}"
    assert "symmetry" in rendered, f"missing symmetry column:\n{rendered}"


def test_rich_table_renders_without_error():
    """info.print() should not raise when Rich is installed."""
    import importlib

    if importlib.util.find_spec("rich") is None:
        import pytest

        pytest.skip("rich not installed")
    x = fnp.ones((4, 4))
    with flops.BudgetContext(flop_budget=10**12, quiet=True):
        _, info = fnp.einsum_path("ij,jk,kl->il", x, x, x)
    # Capture rich output without raising
    info.print(verbose=False)
    info.print(verbose=True)
