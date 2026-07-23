"""Tests for namespace support on BudgetContext and OpRecord."""

import pytest

import flopscope as flops
from flopscope._budget import BudgetContext, OpRecord


def test_oprecord_has_namespace_field():
    rec = OpRecord(
        op_name="add",
        subscripts=None,
        shapes=((10,),),
        flop_cost=10,
        cumulative=10,
        namespace="training",
    )
    assert rec.namespace == "training"


def test_oprecord_namespace_defaults_none():
    rec = OpRecord(
        op_name="add",
        subscripts=None,
        shapes=((10,),),
        flop_cost=10,
        cumulative=10,
        namespace=None,
    )
    assert rec.namespace is None


def test_budget_context_accepts_namespace():
    with BudgetContext(flop_budget=1000, namespace="training") as ctx:
        assert ctx.namespace == "training"


def test_budget_context_namespace_defaults_none():
    with BudgetContext(flop_budget=1000) as ctx:
        assert ctx.namespace is None


def test_deduct_records_namespace():
    with BudgetContext(flop_budget=1000, namespace="inference") as ctx:
        ctx.deduct("add", flop_cost=10, subscripts=None, shapes=((5,),), dtypes=())
        assert ctx.op_log[0].namespace == "inference"


def test_deduct_records_none_namespace():
    with BudgetContext(flop_budget=1000) as ctx:
        ctx.deduct("add", flop_cost=10, subscripts=None, shapes=((5,),), dtypes=())
        assert ctx.op_log[0].namespace is None


def test_summary_includes_namespace_label():
    with BudgetContext(flop_budget=1000, namespace="training", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=(), dtypes=())
        s = ctx.summary()
        assert "[training]" in s


def test_summary_without_namespace():
    with BudgetContext(flop_budget=1000, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=(), dtypes=())
        s = ctx.summary()
        assert "100" in s.replace(",", "")
        assert "[" not in s.split("\n")[0]  # no bracket in header


def test_namespace_scope_builds_dotted_paths():
    with flops.BudgetContext(flop_budget=1000, namespace="predict", quiet=True) as ctx:
        with flops.namespace("precompute"):
            ctx.deduct("add", flop_cost=10, subscripts=None, shapes=(), dtypes=())
        with flops.namespace("fallback"):
            with flops.namespace("sampling"):
                ctx.deduct("mul", flop_cost=20, subscripts=None, shapes=(), dtypes=())

    assert [rec.namespace for rec in ctx.op_log] == [
        "predict.precompute",
        "predict.fallback.sampling",
    ]


def test_budget_context_preserves_literal_root_namespace():
    with flops.BudgetContext(
        flop_budget=1000, namespace="predict..raw", quiet=True
    ) as ctx:
        with flops.namespace("precompute"):
            assert ctx.namespace == "predict..raw.precompute"
            ctx.deduct("add", flop_cost=10, subscripts=None, shapes=(), dtypes=())

    assert ctx.namespace == "predict..raw"
    assert ctx.op_log[0].namespace == "predict..raw.precompute"


def test_namespace_scope_restores_previous_namespace_after_exception():
    with flops.BudgetContext(flop_budget=1000, namespace="predict", quiet=True) as ctx:
        with pytest.raises(RuntimeError, match="boom"):
            with flops.namespace("precompute"):
                assert ctx.namespace == "predict.precompute"
                raise RuntimeError("boom")

        assert ctx.namespace == "predict"
        ctx.deduct("add", flop_cost=10, subscripts=None, shapes=(), dtypes=())

    assert ctx.op_log[0].namespace == "predict"


def test_namespace_scope_requires_active_budget():
    with pytest.raises(flops.NoBudgetContextError):
        with flops.namespace("precompute"):
            pass


@pytest.mark.parametrize("name", [None, 3, "", "   ", "a.b"])
def test_namespace_scope_rejects_invalid_segment(name):
    with flops.BudgetContext(flop_budget=1000, quiet=True):
        with pytest.raises(ValueError):
            with flops.namespace(name):  # type: ignore[arg-type]
                pass
