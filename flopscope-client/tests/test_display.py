"""Tests for the budget-display timing section and accumulator timing aggregation.

The BudgetContext timing split (backend / overhead / residual) is surfaced in the
participant-facing ``render_budget_summary()``. These tests use a fake context so
they need no live server.
"""

from __future__ import annotations

import pytest


class _FakeCtx:
    """Minimal stand-in for a closed BudgetContext (what the accumulator reads)."""

    def __init__(
        self,
        *,
        namespace=None,
        flop_budget=1_000_000,
        flops_used=400_000,
        wall_time_s=0.030,
        backend=0.001,
        overhead=0.017,
        residual=0.012,
    ):
        self.namespace = namespace
        self.flop_budget = flop_budget
        self.flops_used = flops_used
        self.wall_time_s = wall_time_s
        self.flopscope_backend_time = backend
        self.flopscope_overhead_time = overhead
        self.residual_wall_time = residual


def test_accumulator_aggregates_timing():
    from flopscope._budget import BudgetAccumulator

    acc = BudgetAccumulator()
    acc.record(
        _FakeCtx(wall_time_s=0.03, backend=0.001, overhead=0.017, residual=0.012)
    )
    acc.record(
        _FakeCtx(wall_time_s=0.05, backend=0.002, overhead=0.020, residual=0.028)
    )
    data = acc.get_data()

    assert data["wall_time_s"] == pytest.approx(0.08)
    assert data["flopscope_backend_time_s"] == pytest.approx(0.003)
    assert data["flopscope_overhead_time_s"] == pytest.approx(0.037)
    assert data["residual_wall_time_s"] == pytest.approx(0.040)


def test_accumulator_coerces_none_timing():
    """wall_time_s / residual_wall_time can be None on a never-closed context."""
    from flopscope._budget import BudgetAccumulator

    acc = BudgetAccumulator()
    acc.record(_FakeCtx(wall_time_s=None, residual=None))
    data = acc.get_data()

    assert data["wall_time_s"] == 0.0
    assert data["residual_wall_time_s"] == 0.0


def test_plain_text_summary_shows_timing_block():
    import flopscope._budget as b
    from flopscope._display import _plain_text_summary

    b._accumulator.reset()
    b._accumulator.record(
        _FakeCtx(
            flops_used=10_526_400,
            wall_time_s=0.028,
            backend=0.0,
            overhead=0.017,
            residual=0.011,
        )
    )
    try:
        text = _plain_text_summary()
        assert "Wall time" in text
        assert "backend" in text
        assert "overhead" in text
        assert "residual" in text
        assert "0.028s" in text  # wall
        assert "0.017s" in text  # overhead
        assert "0.011s" in text  # residual (billed)
    finally:
        b._accumulator.reset()
