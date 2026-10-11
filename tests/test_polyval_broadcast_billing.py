"""``polyval`` bills every Horner step over the broadcast output (#270).

numpy evaluates ``y = y * x + p[i]`` for each coefficient slice, so when ``p``
has trailing dimensions (or ``x`` is smaller than ``p[i]``, down to a 0-d
point) every step runs over ``broadcast_shapes(p.shape[1:], x.shape)``. The
bill is ``2 * deg * numel(output)``; for a 1-D ``p`` that is the old
``2 * deg * x.size``.
"""

from __future__ import annotations

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp


def _bill(fn):
    """FLOPs billed for ``fn()`` alone, plus its result."""
    with flops.BudgetContext(flop_budget=10**16, quiet=True) as bc:
        out = fn()
    return int(bc.flops_used), out


def test_trailing_coefficient_dims_bill_the_broadcast_output():
    # The issue's case (a): p (8, 512) against x (4096, 1) -> (4096, 512).
    rng = np.random.default_rng(0)
    p = rng.standard_normal((8, 512))
    x = rng.standard_normal((4096, 1))
    fp, fx = fnp.asarray(p), fnp.asarray(x)
    billed, got = _bill(lambda: fnp.polyval(fp, fx))
    assert got.shape == (4096, 512)
    assert billed == 2 * 7 * 4096 * 512
    np.testing.assert_array_equal(np.asarray(got), np.polyval(p, x))


def test_zero_d_point_against_3d_coefficients_bills_every_cell():
    # The issue's case (b): P (2, 1024, 1024) at a 0-d x was billed 2 FLOPs.
    rng = np.random.default_rng(1)
    P = rng.standard_normal((2, 1024, 1024))
    fP, t = fnp.asarray(P), fnp.asarray(0.5)
    billed, got = _bill(lambda: fnp.polyval(fP, t))
    assert got.shape == (1024, 1024)
    assert billed == 2 * 1 * 1024 * 1024
    np.testing.assert_array_equal(np.asarray(got), np.polyval(P, 0.5))


def test_one_d_coefficients_are_unchanged():
    # The issue's control (c): 1-D p bills 2 * deg * x.size, as before.
    rng = np.random.default_rng(2)
    q = fnp.asarray(rng.standard_normal(8))
    z = fnp.asarray(rng.standard_normal(2097152))
    billed, _ = _bill(lambda: fnp.polyval(q, z))
    assert billed == 2 * 7 * 2097152
    # A 0-d x with 1-D p is one point.
    billed_point, _ = _bill(lambda: fnp.polyval(q, 0.25))
    assert billed_point == 2 * 7


def test_x_already_covering_the_coefficient_slice_is_unchanged():
    # p[i] of shape (3,) broadcasts against x (5, 3) without widening it:
    # the output is x.shape, so the bill is the old 2 * deg * x.size.
    rng = np.random.default_rng(3)
    p = fnp.asarray(rng.standard_normal((4, 3)))
    x = fnp.asarray(rng.standard_normal((5, 3)))
    billed, got = _bill(lambda: fnp.polyval(p, x))
    assert got.shape == (5, 3)
    assert billed == 2 * 3 * 15
