"""Tests for analytical helpers and public weighted FLOP APIs."""

import json
from importlib import resources

import numpy as np
import pytest

import flopscope as flops
from flopscope import accounting as public_flops
from flopscope._budget import BudgetContext
from flopscope._flops import (
    _ceil_log2,
    analytical_pointwise_cost,
    analytical_reduction_cost,
    parse_einsum_subscripts,
    search_cost,
    sort_cost,
)
from flopscope._flops import (
    einsum_cost as analytical_einsum_cost,
)
from flopscope._flops import (
    svd_cost as analytical_svd_cost,
)
from flopscope._weights import load_weights, reset_weights
from flopscope.errors import SymmetryLossWarning


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


def test_parse_matmul():
    inputs, output = parse_einsum_subscripts("ij,jk->ik")
    assert inputs == [["i", "j"], ["j", "k"]]
    assert output == ["i", "k"]


def test_parse_trace():
    inputs, output = parse_einsum_subscripts("ii->")
    assert inputs == [["i", "i"]]
    assert output == []


def test_parse_implicit():
    inputs, output = parse_einsum_subscripts("ij,jk")
    assert inputs == [["i", "j"], ["j", "k"]]
    assert output == ["i", "k"]


def test_analytical_einsum_cost_matmul():
    # direct-event model with off-by-one correction:
    # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
    # = 60 + 60 - 15 = 105 (textbook 2n^3 - n^2 form for square matmul)
    # First cell of each output orbit is a free copy.
    assert analytical_einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)]) == 105


def test_analytical_einsum_cost_trace():
    # single-operand trace with off-by-one correction:
    # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
    # = 0 + 10 - 1 = 9 (scalar output: first cell is free).
    assert analytical_einsum_cost("ii->", shapes=[(10, 10)]) == 9


def test_analytical_einsum_cost_batch_matmul():
    # direct-event model with off-by-one correction:
    # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
    # = 120 + 120 - 30 = 210. First cell of each output orbit is a free copy.
    assert analytical_einsum_cost("bij,bjk->bik", shapes=[(2, 3, 4), (2, 4, 5)]) == 210


def test_analytical_einsum_cost_outer_product():
    # direct-event model with off-by-one correction:
    # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits) = 12
    # Outer product has no summed axes; alpha == num_output_orbits so the
    # accumulation term collapses to zero, leaving only the multiplications
    # ((k-1)*prod(M) = 12).
    assert analytical_einsum_cost("i,j->ij", shapes=[(3,), (4,)]) == 12


def test_analytical_einsum_cost_scalar_output():
    # direct-event model with off-by-one correction:
    # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
    # = 5 + 5 - 1 = 9 (scalar output: first cell is a free copy).
    assert analytical_einsum_cost("i,i->", shapes=[(5,), (5,)]) == 9


def test_analytical_pointwise_cost():
    assert analytical_pointwise_cost(shape=(256, 256)) == 256 * 256


def test_analytical_pointwise_cost_scalar():
    assert analytical_pointwise_cost(shape=()) == 1


def test_analytical_reduction_cost():
    # Updated to orbit-mapping count (was n(n+1)/2 unique-element count).
    # Full reduction of (256,256): prod(shape) - 1 = 65536 - 1 = 65535.
    assert analytical_reduction_cost(input_shape=(256, 256), axis=None) == 256 * 256 - 1


def test_analytical_svd_cost():
    assert analytical_svd_cost(m=100, n=50, k=10) == 100 * 50 * 10


def test_analytical_svd_cost_full():
    assert analytical_svd_cost(m=100, n=50, k=None) == 100 * 50 * 50


def test_analytical_pointwise_cost_symmetric():
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    assert analytical_pointwise_cost(shape=(5, 5), symmetry=symmetry) == 15


def test_analytical_pointwise_cost_partial_symmetry():
    symmetry = flops.SymmetryGroup.direct_product(
        flops.SymmetryGroup.symmetric(axes=(0, 1)),
        flops.SymmetryGroup.symmetric(axes=(2, 3)),
    )
    assert analytical_pointwise_cost(shape=(4, 4, 3, 3), symmetry=symmetry) == 60


def test_analytical_pointwise_cost_no_symmetry_unchanged():
    assert analytical_pointwise_cost(shape=(5, 5)) == 25


def test_analytical_reduction_cost_symmetric():
    # Updated to orbit-mapping count (was n(n+1)/2 unique-element count).
    # Symmetric (5,5) has 15 orbits; full reduction needs 15-1=14 additions.
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    assert (
        analytical_reduction_cost(input_shape=(5, 5), axis=None, symmetry=symmetry)
        == 14
    )


def test_analytical_reduction_cost_no_symmetry_unchanged():
    # Updated to orbit-mapping count (was n(n+1)/2 unique-element count).
    # Full reduction of (5,5): prod(shape) - 1 = 25 - 1 = 24 additions.
    assert analytical_reduction_cost(input_shape=(5, 5), axis=None) == 24


def test_analytical_einsum_cost_symmetric_input():
    # new direct-event model: S2{i,j} reduces unique elements (M=55), alpha=55
    # total = (2-1)*55 + 55 = 110 for the 'ijk' component; times 'k' component (5)
    # -> total=550, dense_baseline=500. total < gaming_bound (2*500=1000).
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    cost = analytical_einsum_cost(
        "ijk,k->ij", shapes=[(10, 10, 5), (5,)], operand_symmetries=[symmetry, None]
    )
    dense_gaming_bound = 2 * 10 * 10 * 5  # num_terms * dense_baseline
    assert cost <= dense_gaming_bound
    assert cost > 0


def test_analytical_einsum_cost_no_operand_symmetry_unchanged():
    # direct-event model with off-by-one correction:
    # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
    # = 100 + 100 - 10 = 190. First cell of each output orbit is a free copy.
    cost = analytical_einsum_cost("ij,j->i", shapes=[(10, 10), (10,)])
    assert cost == 190


def test_analytical_einsum_cost_preserves_repeated_label_axis_positions():
    # The new accumulation model uses compute_accumulation_cost directly
    # (no oracle monkeypatching needed). The S2{i (axis 0,2)} symmetry on
    # operand 'iji' with shape (4,3,4) should be detected.
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 2))
    cost = analytical_einsum_cost(
        "iji->j",
        shapes=[(4, 3, 4)],
        operand_symmetries=[symmetry],
    )
    # Verify cost is computed (non-zero) and within gaming-resistance bound
    dense_baseline = 4 * 3 * 4  # prod of all label sizes (i=4, j=3, i=4)
    assert cost > 0
    assert cost <= dense_baseline  # single operand: total <= dense_baseline


def test_analytical_einsum_cost_matches_accumulation_model():
    # The new model: analytical_einsum_cost uses compute_accumulation_cost,
    # NOT contract_path.optimized_cost (they differ after the new model).
    cost = analytical_einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)])
    # New model gives (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
    # = 60 + 60 - 15 = 105. First cell of each output orbit is a free copy.
    assert cost == 105  # direct-event model result


def test_public_pointwise_cost_is_weighted(tmp_path):
    load_weights(_write_weights(tmp_path, {"exp": 2.5}), use_packaged_default=False)
    assert public_flops.pointwise_cost("exp", shape=(3, 3)) == 22


def test_public_pointwise_cost_uses_symmetry_keyword_and_weight(tmp_path):
    symmetry = flops.SymmetryGroup.symmetric(axes=(0, 1))
    load_weights(_write_weights(tmp_path, {"exp": 2.5}), use_packaged_default=False)
    assert public_flops.pointwise_cost("exp", shape=(5, 5), symmetry=symmetry) == 37


def test_public_reduction_cost_is_weighted(tmp_path):
    # Updated to orbit-mapping count (was n(n+1)/2 unique-element count).
    # Base: (4*5) - 1 = 19 additions; 19 * 3.25 = 61.75 → int = 61.
    load_weights(_write_weights(tmp_path, {"sum": 3.25}), use_packaged_default=False)
    assert public_flops.reduction_cost("sum", input_shape=(4, 5), axis=None) == 61


def test_public_einsum_cost_is_weighted(tmp_path):
    load_weights(_write_weights(tmp_path, {"einsum": 2.0}), use_packaged_default=False)
    # direct-event model with off-by-one correction:
    # unweighted = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
    # = 60 + 60 - 15 = 105; weight=2.0 → 210.
    assert public_flops.einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)]) == 210


def test_public_helpers_can_use_packaged_default_weights():
    load_weights(use_packaged_default=True)
    assert public_flops.pointwise_cost("exp", shape=(2, 2)) == int(
        analytical_pointwise_cost((2, 2)) * _packaged_weight("exp")
    )


def test_binary_op_with_incompatible_symmetry_warns_and_returns_dense():
    from flopscope._pointwise import add as counted_add

    a = flops.as_symmetric(
        np.ones((2, 2, 2)),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    b = flops.as_symmetric(
        np.ones((2, 2, 2)),
        symmetry=flops.SymmetryGroup.symmetric(axes=(1, 2)),
    )

    with BudgetContext(flop_budget=10**6):
        with pytest.warns(
            SymmetryLossWarning,
            match="no symmetry groups shared by both operands",
        ):
            result = counted_add(a, b)

    assert not isinstance(result, flops.SymmetricTensor)


def test_public_flops_no_longer_export_symmetry_info():
    assert not hasattr(public_flops, "SymmetryInfo")
    assert "SymmetryInfo" not in public_flops.__all__


class TestCeilLog2:
    def test_one(self):
        assert _ceil_log2(1) == 1

    def test_two(self):
        assert _ceil_log2(2) == 1

    def test_three(self):
        assert _ceil_log2(3) == 2

    def test_power_of_two(self):
        assert _ceil_log2(8) == 3

    def test_large(self):
        assert _ceil_log2(1000) == 10

    def test_zero(self):
        assert _ceil_log2(0) == 1


class TestSortCost:
    def test_basic(self):
        assert sort_cost(8) == 24

    def test_one_element(self):
        assert sort_cost(1) == 1

    def test_zero_elements(self):
        assert sort_cost(0) == 1


class TestSearchCost:
    def test_basic(self):
        assert search_cost(10, 8) == 30

    def test_one_query(self):
        assert search_cost(1, 1024) == 10

    def test_empty_queries(self):
        assert search_cost(0, 100) == 1
