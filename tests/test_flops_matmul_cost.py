"""Tests for matmul_cost helper added for issue #69."""

import numpy as np
import pytest

import flopscope.numpy as fnp
from flopscope._budget import BudgetContext
from flopscope._flops import matmul_cost


def test_matmul_cost_matches_fnp_matmul_charge_square():
    """matmul_cost(n, n, n) must equal what fnp.matmul charges on (n,n)@(n,n)."""
    n = 30
    a = fnp.asarray(np.random.default_rng(0).random((n, n)))
    b = fnp.asarray(np.random.default_rng(0).random((n, n)))
    with BudgetContext(flop_budget=10**14) as bc:
        fnp.matmul(a, b)
    assert matmul_cost(n, n, n) == bc.flops_used


def test_matmul_cost_matches_fnp_matmul_charge_rectangular():
    """matmul_cost(m, k, n) must match fnp.matmul on (m,k)@(k,n)."""
    m, k, n = 10, 20, 30
    a = fnp.asarray(np.random.default_rng(0).random((m, k)))
    b = fnp.asarray(np.random.default_rng(0).random((k, n)))
    with BudgetContext(flop_budget=10**14) as bc:
        fnp.matmul(a, b)
    assert matmul_cost(m, k, n) == bc.flops_used


def test_matmul_cost_formula():
    """Formula is 2*m*k*n - m*n (FMA=1 with accumulator off-by-one)."""
    assert matmul_cost(5, 7, 11) == 2 * 5 * 7 * 11 - 5 * 11
    assert matmul_cost(1, 1, 1) == 1  # max(1, ...) clamp


def test_matmul_cost_zero_dim_clamped_to_one():
    """Empty matmul still charges at least 1 FLOP for the budget bookkeeping."""
    assert matmul_cost(0, 5, 5) >= 1
    assert matmul_cost(5, 0, 5) >= 1
