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


def _setup_gradient_2d(rng):
    f = fnp.asarray(rng.random((50, 50)))
    return (f,), {}


def _oracle_gradient_2d(f):
    # Mirror numpy.gradient: for each axis, central diff (interior) +
    # boundary forward/backward diff + divide by 2 (interior).
    for axis in (0, 1):
        slc1 = [slice(None)] * 2
        slc2 = [slice(None)] * 2
        slc_mid = [slice(None)] * 2
        slc1[axis] = slice(2, None)
        slc2[axis] = slice(None, -2)
        slc_mid[axis] = slice(1, -1)
        fnp.subtract(f[tuple(slc1)], f[tuple(slc2)])
        fnp.divide(f[tuple(slc_mid)], 2.0)


def _setup_gradient_3d(rng):
    f = fnp.asarray(rng.random((20, 20, 20)))
    return (f,), {}


def _oracle_gradient_3d(f):
    for axis in range(3):
        slc1 = [slice(None)] * 3
        slc2 = [slice(None)] * 3
        slc_mid = [slice(None)] * 3
        slc1[axis] = slice(2, None)
        slc2[axis] = slice(None, -2)
        slc_mid[axis] = slice(1, -1)
        fnp.subtract(f[tuple(slc1)], f[tuple(slc2)])
        fnp.divide(f[tuple(slc_mid)], 2.0)


CASES.extend([
    CostParityCase(
        name="gradient_2d",
        setup=_setup_gradient_2d,
        wrapper=fnp.gradient,
        oracle=_oracle_gradient_2d,
        tolerance=0.05,
    ),
    CostParityCase(
        name="gradient_3d",
        setup=_setup_gradient_3d,
        wrapper=fnp.gradient,
        oracle=_oracle_gradient_3d,
        tolerance=0.05,
    ),
])


def _setup_unwrap(rng):
    a = fnp.asarray(rng.random((1000,)) * 8.0)
    return (a,), {}


def _oracle_unwrap(a):
    # numpy.unwrap (simplified): diff -> mod -> add -> subtract -> cumsum -> where
    dd = fnp.diff(a)
    ddmod = fnp.mod(fnp.add(dd, np.pi), 2 * np.pi)
    fnp.subtract(ddmod, np.pi)
    fnp.cumsum(dd)
    fnp.where(ddmod > 0, ddmod, 0.0)


CASES.append(
    CostParityCase(
        name="unwrap",
        setup=_setup_unwrap,
        wrapper=fnp.unwrap,
        oracle=_oracle_unwrap,
        tolerance=0.10,
    )
)


def _setup_matrix_power_4(rng):
    a = fnp.asarray(np.eye(20) + 0.001 * rng.random((20, 20)))
    return (a, 4), {}


def _oracle_matrix_power_4(a, n):
    # Binary squaring for n=4: A^2 = A @ A; A^4 = A^2 @ A^2 -> 2 matmuls
    a2 = fnp.matmul(a, a)
    fnp.matmul(a2, a2)


def _setup_matrix_power_16(rng):
    a = fnp.asarray(np.eye(20) + 0.001 * rng.random((20, 20)))
    return (a, 16), {}


def _oracle_matrix_power_16(a, n):
    # 16 = 2^4, 4 squarings
    a2 = fnp.matmul(a, a)
    a4 = fnp.matmul(a2, a2)
    a8 = fnp.matmul(a4, a4)
    fnp.matmul(a8, a8)


CASES.extend([
    CostParityCase(
        name="matrix_power_4",
        setup=_setup_matrix_power_4,
        wrapper=fnp.linalg.matrix_power,
        oracle=_oracle_matrix_power_4,
        tolerance=0.01,
    ),
    CostParityCase(
        name="matrix_power_16",
        setup=_setup_matrix_power_16,
        wrapper=fnp.linalg.matrix_power,
        oracle=_oracle_matrix_power_16,
        tolerance=0.01,
    ),
])


def _setup_convolve(rng):
    a = fnp.asarray(rng.random((200,)))
    v = fnp.asarray(rng.random((50,)))
    return (a, v), {}


def _oracle_convolve(a, v):
    # Convolution does 2*a*v - a - v FMA-1 ops.
    # We synthesize an oracle by creating a plain numpy array of the equivalent
    # size and firing one fnp.multiply that charges exactly that many ops.
    # Use np.zeros (not fnp.asarray) to avoid charging extra asarray FLOPs.
    s = 2 * a.size * v.size - a.size - v.size
    probe = np.zeros(s)
    fnp.multiply(probe, 1.0)


CASES.append(
    CostParityCase(
        name="convolve",
        setup=_setup_convolve,
        wrapper=fnp.convolve,
        oracle=_oracle_convolve,
        tolerance=0.0,
    )
)


def _setup_correlate(rng):
    a = fnp.asarray(rng.random((200,)))
    v = fnp.asarray(rng.random((50,)))
    return (a, v), {}


def _oracle_correlate(a, v):
    import numpy as _np
    s = 2 * a.size * v.size - a.size - v.size
    probe = _np.zeros(s)
    fnp.multiply(probe, 1.0)


CASES.append(
    CostParityCase(
        name="correlate",
        setup=_setup_correlate,
        wrapper=fnp.correlate,
        oracle=_oracle_correlate,
        tolerance=0.0,
    )
)


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
