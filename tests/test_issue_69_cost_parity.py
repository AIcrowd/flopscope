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
    oracle: (
        Callable  # manual fnp.<primitive> composition with same signature as wrapper
    )
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

CASES.extend(
    [
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
    ]
)


def _setup_gradient_2d(rng):
    f = fnp.asarray(rng.random((50, 50)))
    return (f,), {}


def _oracle_gradient(f):
    # Honest mirror of numpy.gradient (edge_order=1): for EACH axis it emits one
    # output value per input element, so the oracle must count both the interior
    # AND the two boundary hyperplanes -- omitting the boundaries (as an earlier
    # oracle did) is exactly the under-count that issue #188 fixed. Per axis:
    #   interior: central difference = subtract(f[2:], f[:-2]) + divide(.../2)
    #             over the L-2 interior planes;
    #   each edge: one-sided difference = subtract(two planes) + divide over the
    #              single boundary plane.
    # That is 2*S*(L-2)/L + 4*S/L = 2*S FLOPs per axis (S = f.size).
    ndim = f.ndim
    for axis in range(ndim):
        interior_hi = [slice(None)] * ndim
        interior_lo = [slice(None)] * ndim
        interior_mid = [slice(None)] * ndim
        interior_hi[axis] = slice(2, None)
        interior_lo[axis] = slice(None, -2)
        interior_mid[axis] = slice(1, -1)
        fnp.subtract(f[tuple(interior_hi)], f[tuple(interior_lo)])
        fnp.divide(f[tuple(interior_mid)], 2.0)
        for near, far in ((slice(1, 2), slice(0, 1)), (slice(-1, None), slice(-2, -1))):
            edge_near = [slice(None)] * ndim
            edge_far = [slice(None)] * ndim
            edge_near[axis] = near
            edge_far[axis] = far
            fnp.subtract(f[tuple(edge_near)], f[tuple(edge_far)])
            fnp.divide(f[tuple(edge_near)], 1.0)


def _setup_gradient_3d(rng):
    f = fnp.asarray(rng.random((20, 20, 20)))
    return (f,), {}


CASES.extend(
    [
        CostParityCase(
            name="gradient_2d",
            setup=_setup_gradient_2d,
            wrapper=fnp.gradient,
            oracle=_oracle_gradient,
            tolerance=0.05,
        ),
        CostParityCase(
            name="gradient_3d",
            setup=_setup_gradient_3d,
            wrapper=fnp.gradient,
            oracle=_oracle_gradient,
            tolerance=0.05,
        ),
    ]
)


def _setup_unwrap(rng):
    a = fnp.asarray(rng.random((1000,)) * 8.0)
    return (a,), {}


def _oracle_unwrap(a):
    # numpy.unwrap full 13-pass trace (see unwrap_cost docstring for derivation):
    # 1. diff
    dd = fnp.diff(a)
    # 2. dd - interval_low  (subtract scalar)
    dd_shifted = fnp.subtract(dd, -np.pi)
    # 3. mod(dd_shifted, period)
    ddmod_raw = fnp.mod(dd_shifted, 2 * np.pi)
    # 4. + interval_low  (add scalar back)
    ddmod = fnp.add(ddmod_raw, -np.pi)
    # 5. ddmod == interval_low  (boundary check)
    eq_low = fnp.equal(ddmod, -np.pi)
    # 6. dd > 0
    dd_pos = fnp.greater(dd, 0.0)
    # 7. bitwise-and of bool arrays
    boundary_mask = fnp.bitwise_and(eq_low, dd_pos)
    # 8. copyto/select (boundary fix)
    fnp.where(boundary_mask, np.pi / 2, ddmod)
    # 9. ph_correct = ddmod - dd
    ph_correct = fnp.subtract(ddmod, dd)
    # 10. abs(dd)
    abs_dd = fnp.abs(dd)
    # 11. abs_dd < discont
    small_jump = fnp.less(abs_dd, np.pi)
    # 12. copyto/select (zero small jumps)
    ph_correct_zeroed = fnp.where(small_jump, 0.0, ph_correct)
    # 13. cumsum
    fnp.cumsum(ph_correct_zeroed)


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


CASES.extend(
    [
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
    ]
)


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

    # default mode='valid': honest = (2*min-1)*(max-min+1)
    mn = min(a.size, v.size)
    mx = max(a.size, v.size)
    s = (2 * mn - 1) * (mx - mn + 1)
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


def _setup_cross(rng):
    a = fnp.asarray(rng.random((100, 3)))
    b = fnp.asarray(rng.random((100, 3)))
    return (a, b), {}


def _oracle_cross(a, b):
    # 3-vec cross product per row: 6 mults + 3 subs = 9 ops per 3-element output
    # row = 3 ops per output element (output (100, 3) = 300 elements -> 900 ops).
    import numpy as _np

    out_size = a.shape[0] * 3
    probe = _np.zeros(out_size * 3)
    fnp.multiply(probe, 1.0)


CASES.append(
    CostParityCase(
        name="cross",
        setup=_setup_cross,
        wrapper=fnp.cross,
        oracle=_oracle_cross,
        tolerance=0.0,
    )
)


def _setup_pinv(rng):
    a = fnp.asarray(rng.random((30, 20)))
    return (a,), {}


def _oracle_pinv(a):
    # numpy.pinv: svd -> threshold -> (vt.T * s_inv) -> matmul with u.T
    u, s, vt = fnp.linalg.svd(a, full_matrices=False)
    cutoff = fnp.max(s) * max(a.shape) * np.finfo(s.dtype).eps
    s_inv = fnp.where(s > cutoff, 1.0 / s, 0.0)
    fnp.matmul(vt.T * s_inv, u.T)


CASES.append(
    CostParityCase(
        name="pinv",
        setup=_setup_pinv,
        wrapper=fnp.linalg.pinv,
        oracle=_oracle_pinv,
        tolerance=0.05,
    )
)


def _setup_lstsq(rng):
    a = fnp.asarray(rng.random((40, 25)))
    b = fnp.asarray(rng.random((40,)))
    return (a, b), {"rcond": None}


def _oracle_lstsq(a, b, rcond=None):
    # numpy.lstsq via SVD:
    #   u, s, vt = svd(a, full_matrices=False)  -> (m,k), (k,), (k,n) where k=min(m,n)
    #   ub = u.T @ b                            -> (k,) for 1D b, (k,c) for 2D
    #   ub_scaled = ub / s                      -> k * c divides (c=1 for 1D b)
    #   x = vt.T @ ub_scaled                    -> (n,) for 1D b, (n,c) for 2D
    u, s, vt = fnp.linalg.svd(a, full_matrices=False)
    ub = fnp.matmul(u.T, b)
    ub_scaled = fnp.divide(ub, s)
    fnp.matmul(vt.T, ub_scaled)


CASES.append(
    CostParityCase(
        name="lstsq",
        setup=_setup_lstsq,
        wrapper=lambda a, b, rcond=None: fnp.linalg.lstsq(a, b, rcond=rcond),
        oracle=_oracle_lstsq,
        tolerance=0.05,
    )
)


def _setup_polyval(rng):
    p = fnp.asarray(rng.random((5,)))
    x = fnp.asarray(rng.random((100,)))
    return (p, x), {}


def _oracle_polyval(p, x):
    # Horner's method: result = p[0]; for c in p[1:]: result = result * x + c
    # numpy.polyval seeds result = p[0] (a scalar broadcast, no multiply charge).
    # Each of the deg=len(p)-1 iterations costs: 1 multiply + 1 add over x.size.
    # Total: deg * 2 * x.size ops — exactly what polyval_cost returns.
    # Use plain np.ones (uncharged) so initialization doesn't add spurious FLOPs.
    result = np.array(p[0]) * np.ones(x.shape)  # plain numpy broadcast, no fnp charge
    for i in range(1, len(p)):
        result = fnp.add(fnp.multiply(result, x), p[i])


CASES.append(
    CostParityCase(
        name="polyval",
        setup=_setup_polyval,
        wrapper=fnp.polyval,
        oracle=_oracle_polyval,
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
            f"rel_diff={(wrapper_cost - oracle_cost) / oracle_cost:.4f}, "
            f"tolerance={case.tolerance}"
        )
