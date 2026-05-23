"""Cost-parity regression tests for issue #69.

For each compound-function wrapper that had a wrong cost formula, asserts
that ``fnp.<wrapper>(...)`` charges the same number of FLOPs as a manual
composition of ``fnp.<primitive>`` calls implementing the same algorithm.

The "primitive composition" is the ground-truth oracle: each primitive
self-accounts via its own counted wrapper, so the sum is what would have
been charged if every internal ufunc fired through NEP 13.

New cases are added one per wrapper-fix task (Tasks 7-15).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

import flopscope.numpy as fnp
from flopscope._budget import BudgetContext


@dataclass
class CostParityCase:
    """One entry in the parametrized cost-parity test."""

    name: str
    setup: Callable[[np.random.Generator], tuple]  # returns (args, kwargs) for wrapper
    wrapper: Callable  # fnp.<func>
    oracle: Callable  # manual fnp.<primitive> composition with same signature as wrapper
    tolerance: float = 0.0  # relative tolerance; 0.0 means exact equality required


def _charged(callable_, *args, **kwargs) -> int:
    with BudgetContext(flop_budget=10**14) as bc:
        callable_(*args, **kwargs)
    return bc.flops_used


# ---------------------------------------------------------------------------
# Test cases — populated by Tasks 7-15. Start with diff_n1 (already matches,
# proves the harness works).
# ---------------------------------------------------------------------------

def _setup_diff_n1(rng):
    a = fnp.asarray(rng.random((1000,)))
    return (a,), {"n": 1}


def _oracle_diff_n1(a, n=1, axis=-1):
    return fnp.subtract(a[1:], a[:-1])


def _setup_diff_n3(rng):
    a = fnp.asarray(rng.random((1000,)))
    return (a,), {"n": 3}


def _oracle_diff_n3(a, n=3, axis=-1):
    out = a
    for _ in range(n):
        out = fnp.subtract(out[1:], out[:-1])
    return out


def _setup_diff_n10(rng):
    a = fnp.asarray(rng.random((1000,)))
    return (a,), {"n": 10}


def _oracle_diff_n10(a, n=10, axis=-1):
    out = a
    for _ in range(n):
        out = fnp.subtract(out[1:], out[:-1])
    return out


CASES: list[CostParityCase] = [
    CostParityCase(
        name="diff_n1",
        setup=_setup_diff_n1,
        wrapper=fnp.diff,
        oracle=_oracle_diff_n1,
        tolerance=0.0,
    ),
]

CASES.extend([
    CostParityCase(
        name="diff_n3",
        setup=_setup_diff_n3,
        wrapper=fnp.diff,
        oracle=_oracle_diff_n3,
        tolerance=0.0,
    ),
    CostParityCase(
        name="diff_n10",
        setup=_setup_diff_n10,
        wrapper=fnp.diff,
        oracle=_oracle_diff_n10,
        tolerance=0.0,
    ),
])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_cost_parity(case):
    """Wrapper-charged FLOPs must equal sum-of-fnp-primitive FLOPs for the same algo."""
    rng = np.random.default_rng(0)
    args, kwargs = case.setup(rng)
    wrapper_cost = _charged(case.wrapper, *args, **kwargs)
    oracle_cost = _charged(case.oracle, *args, **kwargs)
    if case.tolerance == 0.0:
        assert wrapper_cost == oracle_cost, (
            f"{case.name}: wrapper={wrapper_cost}, oracle={oracle_cost}, "
            f"delta={wrapper_cost - oracle_cost}"
        )
    else:
        assert wrapper_cost == pytest.approx(oracle_cost, rel=case.tolerance), (
            f"{case.name}: wrapper={wrapper_cost}, oracle={oracle_cost}, "
            f"rel_diff={(wrapper_cost-oracle_cost)/oracle_cost:.4f}, "
            f"tolerance={case.tolerance}"
        )
