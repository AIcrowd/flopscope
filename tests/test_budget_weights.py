"""Tests for FLOP weight integration in BudgetContext.deduct()."""

import importlib
import json
from importlib import resources

import pytest

import flopscope._weights as weights_module
from flopscope import accounting as public_flops
from flopscope._budget import BudgetContext
from flopscope._weights import load_weights, reset_weights
from flopscope.errors import BudgetExhaustedError


@pytest.fixture(autouse=True)
def _reset_weights():
    reset_weights()
    yield
    reset_weights()


def _write_weights(tmp_path, weights):
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"weights": weights}), encoding="utf-8")
    return str(path)


def _packaged_weight(op_name):
    resource = resources.files("flopscope").joinpath("data/default_weights.json")
    with resource.open("r", encoding="utf-8") as f:
        return json.load(f)["weights"][op_name]


def test_deduct_applies_weight(tmp_path):
    load_weights(_write_weights(tmp_path, {"exp": 10.0}))
    with BudgetContext(flop_budget=1_000_000) as budget:
        budget.deduct(
            "exp", flop_cost=100, subscripts=None, shapes=((100,),), dtypes=()
        )
        assert budget.flops_used == 1000
        rec = budget.op_log[0]
        assert rec.flop_cost == 1000


def test_deduct_weight_default_is_one():
    with BudgetContext(flop_budget=1_000_000) as budget:
        budget.deduct(
            "add", flop_cost=100, subscripts=None, shapes=((100,),), dtypes=()
        )
        assert budget.flops_used == 100


def test_deduct_exhaustion_accounts_for_weight(tmp_path):
    load_weights(_write_weights(tmp_path, {"exp": 10.0}))
    with pytest.raises(BudgetExhaustedError):
        with BudgetContext(flop_budget=500) as budget:
            budget.deduct(
                "exp", flop_cost=100, subscripts=None, shapes=((100,),), dtypes=()
            )


def test_deduct_uses_packaged_default_when_explicitly_loaded():
    load_weights(use_packaged_default=True)
    with BudgetContext(flop_budget=1_000_000) as budget:
        budget.deduct("exp", flop_cost=10, subscripts=None, shapes=((10,),), dtypes=())
        assert budget.flops_used == int(10 * _packaged_weight("exp"))


def test_public_helpers_match_runtime_deduction_under_default_import_config(
    monkeypatch,
):
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    monkeypatch.delenv("FLOPSCOPE_DISABLE_WEIGHTS", raising=False)
    importlib.reload(weights_module)

    expected = public_flops.pointwise_cost("exp", shape=(2, 5))

    with BudgetContext(flop_budget=1_000_000) as budget:
        budget.deduct("exp", flop_cost=10, subscripts=None, shapes=((2, 5),), dtypes=())
        assert budget.flops_used == expected


def test_deduct_disable_weights_ignores_packaged_default(monkeypatch):
    monkeypatch.setenv("FLOPSCOPE_DISABLE_WEIGHTS", "1")
    load_weights(use_packaged_default=True)
    with BudgetContext(flop_budget=1_000_000) as budget:
        budget.deduct("exp", flop_cost=10, subscripts=None, shapes=((10,),), dtypes=())
        assert budget.flops_used == 10
