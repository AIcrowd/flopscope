"""Tests for the regime data types in _ladder.py."""

import dataclasses

import pytest

from flopscope._accumulation._ladder import (
    AccumulationResult,
    Regime,
    RegimeContext,
    RegimeOutput,
    RegimeStep,
    Verdict,
)
from flopscope._perm_group import _Permutation as Permutation


def test_regime_context_is_frozen():
    ctx = RegimeContext(
        labels=("i", "j"),
        va=("i",),
        wa=("j",),
        elements=(Permutation.identity(2),),
        generators=(),
        sizes=(3, 3),
        visible_positions=(0,),
        partition_budget=100_000,
    )
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        ctx.labels = ("x",)  # type: ignore[misc]


def test_verdict_carries_fired_and_reason():
    v = Verdict(fired=True, reason="|V| = 1")
    assert v.fired is True
    assert v.reason == "|V| = 1"


def test_regime_output_default_sub_steps_empty():
    out = RegimeOutput(count=42)
    assert out.count == 42
    assert out.sub_steps == ()


def test_regime_step_default_sub_steps_empty():
    step = RegimeStep(regime_id="trivial", decision="fired", reason="|G|=1")
    assert step.sub_steps == ()


def test_regime_dataclass_holds_callables():
    def recognize(ctx):
        return Verdict(True, "always")

    def compute(ctx):
        return RegimeOutput(count=1)

    r = Regime(id="trivial", recognize=recognize, compute=compute)
    assert r.id == "trivial"
    assert r.recognize is recognize
    assert r.compute is compute


def test_accumulation_result_unavailable_carries_none_count():
    result = AccumulationResult(
        count=None,
        regime_id="unavailable",
        shape="mixed",
        trace=(RegimeStep("unavailable", "fired", "budget exceeded"),),
    )
    assert result.count is None
    assert result.regime_id == "unavailable"


def test_regime_id_literal_includes_all_six_regimes():
    # Must include the 5 active regimes + 'unavailable'.
    # We don't introspect Literal at runtime; this is a structural reminder.
    expected = {
        "trivial",
        "functionalProjection",
        "singleton",
        "young",
        "partitionCount",
        "unavailable",
    }
    # Ensure each is constructible as a string passed to RegimeStep.
    for rid in expected:
        step = RegimeStep(regime_id=rid, decision="fired", reason="test")  # type: ignore[arg-type]
        assert step.regime_id == rid


def test_shape_literal_includes_four_shapes():
    expected = {"trivial", "allVisible", "allSummed", "mixed"}
    for shape in expected:
        result = AccumulationResult(
            count=1,
            regime_id="trivial",
            shape=shape,  # type: ignore[arg-type]
            trace=(),
        )
        assert result.shape == shape
