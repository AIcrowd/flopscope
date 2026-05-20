"""Tests for build_path_info() adapter from upstream PathInfo."""

import numpy as np
import opt_einsum

from flopscope._opt_einsum._contract import PathInfo, StepInfo, build_path_info


def test_build_path_info_returns_flopscope_pathinfo():
    A = np.zeros((3, 4))
    B = np.zeros((4, 5))
    upstream_path, upstream_info = opt_einsum.contract_path(
        "ij,jk->ik",
        A,
        B,
        shapes=False,
    )
    flop_info = build_path_info(
        upstream_path,
        upstream_info,
        size_dict=upstream_info.size_dict,
    )
    assert isinstance(flop_info, PathInfo)


def test_build_path_info_path_matches_upstream():
    A = np.zeros((3, 4))
    B = np.zeros((4, 5))
    upstream_path, upstream_info = opt_einsum.contract_path(
        "ij,jk->ik",
        A,
        B,
        shapes=False,
    )
    flop_info = build_path_info(
        upstream_path,
        upstream_info,
        size_dict=upstream_info.size_dict,
    )
    assert list(flop_info.path) == list(upstream_path)



def test_build_path_info_step_has_subscript():
    A = np.zeros((3, 4))
    B = np.zeros((4, 5))
    upstream_path, upstream_info = opt_einsum.contract_path(
        "ij,jk->ik",
        A,
        B,
        shapes=False,
    )
    flop_info = build_path_info(
        upstream_path,
        upstream_info,
        size_dict=upstream_info.size_dict,
    )
    assert flop_info.steps[0].subscript  # non-empty einsum string
    assert isinstance(flop_info.steps[0].subscript, str)


def test_build_path_info_three_operand_chain():
    """ij,jk,kl->il: 2-step path. Each step's flop_count is recomputed."""
    A = np.zeros((3, 4))
    B = np.zeros((4, 5))
    C = np.zeros((5, 6))
    upstream_path, upstream_info = opt_einsum.contract_path(
        "ij,jk,kl->il",
        A,
        B,
        C,
        shapes=False,
    )
    flop_info = build_path_info(
        upstream_path,
        upstream_info,
        size_dict=upstream_info.size_dict,
    )
    assert len(flop_info.steps) == 2
    # Each StepInfo has at least 4 fields: subscript, flop_count, input_shapes, output_shape
    for step in flop_info.steps:
        assert isinstance(step, StepInfo)
        assert step.flop_count > 0
        assert isinstance(step.input_shapes, list)
        assert step.output_shape is not None
    # optimized_cost equals sum of per-step
    assert flop_info.optimized_cost == sum(s.flop_count for s in flop_info.steps)
