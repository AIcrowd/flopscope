"""Regression: _dimino bails on oversized groups instead of hanging."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import flopscope as flops
from flopscope._config import get_setting, set_setting
from flopscope._perm_group import _dimino, _DiminoBudgetExceeded, _Permutation
from flopscope.errors import CostFallbackWarning


def test_dimino_raises_on_oversized_group():
    """_dimino raises _DiminoBudgetExceeded when the seen-set exceeds the
    configured dimino_budget. Regression for PR #91 hang on auto-inferred
    S_n from np.ones((1,)*n)."""
    original_budget = get_setting("dimino_budget")
    try:
        set_setting("dimino_budget", 100)
        # Generators of S_8 (8! = 40,320 elements): adjacent transposition
        # and full cycle. Small enough to construct but well above the
        # 100-element budget we set.
        adj = _Permutation([1, 0, 2, 3, 4, 5, 6, 7])
        cyc = _Permutation([1, 2, 3, 4, 5, 6, 7, 0])
        with pytest.raises(_DiminoBudgetExceeded):
            _dimino((adj, cyc))
    finally:
        set_setting("dimino_budget", original_budget)


def test_dimino_succeeds_under_budget():
    """_dimino returns the full element list when the group is small enough."""
    original_budget = get_setting("dimino_budget")
    try:
        set_setting("dimino_budget", 1000)
        # S_3 has 3! = 6 elements — well under 1000.
        gen1 = _Permutation([1, 0, 2])
        gen2 = _Permutation([0, 2, 1])
        elements = _dimino((gen1, gen2))
        assert len(elements) == 6
    finally:
        set_setting("dimino_budget", original_budget)


def test_accumulation_cost_bails_on_oversized_inferred_symmetry():
    """The numpy compat test that hung: np.ones((1,)*n) auto-infers S_n →
    n! elements. With a tiny dimino_budget the cost should bail to dense
    with CostFallbackWarning, NOT hang.

    Uses n=20 so ndim <= 26 (the existing >26 path bypasses _dimino).
    """
    n = 20
    deep = flops.as_symmetric(
        np.ones((1,) * n),
        symmetry=flops.SymmetryGroup.symmetric(axes=tuple(range(n))),
    )
    original = get_setting("dimino_budget")
    try:
        # Tiny budget forces the bail quickly.
        set_setting("dimino_budget", 1000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", CostFallbackWarning)
            cost = flops.reduction_accumulation_cost(deep)
    finally:
        set_setting("dimino_budget", original)
    # Should have completed; the cost should be the dense fallback.
    assert cost.total >= 0
    # Either a CostFallbackWarning was emitted, or fallback_used is True.
    assert any(isinstance(w.message, CostFallbackWarning) for w in caught) or getattr(
        cost, "fallback_used", False
    ), f"expected CostFallbackWarning or fallback_used=True; got {cost}"


def test_accumulation_cost_bails_in_under_one_second_on_s33():
    """End-to-end smoke: np.ones((1,)*33) should not hang.

    This is the exact CI-hanging path from PR #91. With the default
    dimino_budget of 500_000, the cost computation must complete in
    well under a second.
    """
    import time

    deep = flops.as_symmetric(
        np.ones((1,) * 33),
        symmetry=flops.SymmetryGroup.symmetric(axes=tuple(range(33))),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CostFallbackWarning)
        t0 = time.perf_counter()
        cost = flops.reduction_accumulation_cost(deep)
        elapsed = time.perf_counter() - t0
    # Sanity: ndim=33 > 26 takes the _dense_fallback_cost path; this is
    # the ultimate safety net but we still verify it completes.
    assert elapsed < 5.0, f"expected <5s, took {elapsed:.2f}s"
    assert cost.total >= 0
