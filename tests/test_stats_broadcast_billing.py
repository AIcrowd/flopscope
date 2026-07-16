"""Tests that stats billing covers the broadcast of x with distribution params.

Regression tests for the broadcast undercount: every ``ContinuousDistribution``
method used to charge on ``numel(x)`` alone, but array-valued ``loc``/``scale``
(or ``a``/``b``/``s``) numpy-broadcast the output larger than ``x``, so e.g.
``norm.pdf(0.5, loc=np.zeros(1000))`` returned 1000 elements while billing for
one. The charge is now ``cost_per_elem * numel(broadcast(x, params))``.
"""

from __future__ import annotations

import numpy as np
import pytest

from flopscope._budget import BudgetContext
from flopscope.stats._norm import norm
from flopscope.stats._truncnorm import truncnorm

# ---------------------------------------------------------------------------
# TestBroadcastBilling
# ---------------------------------------------------------------------------


class TestBroadcastBilling:
    """Array-valued distribution parameters must be billed, not just x."""

    def test_scalar_x_array_loc(self):
        """Scalar x + 1000-element loc: output has 1000 elements; bill all."""
        loc = np.zeros(1000)
        with BudgetContext(10**9, quiet=True) as ctx:
            result = norm.pdf(0.5, loc=loc)  # pyright: ignore[reportArgumentType]
        assert np.asarray(result).shape == (1000,)
        assert ctx.flops_used == 27 * 1000

    def test_array_x_array_scale_broadcast_wider(self):
        """(1, 10) x with (100, 1) scale broadcasts to (100, 10) = 1000 elems."""
        x = np.zeros((1, 10))
        scale = np.ones((100, 1))
        with BudgetContext(10**9, quiet=True) as ctx:
            result = norm.pdf(x, scale=scale)  # pyright: ignore[reportArgumentType]
        assert np.asarray(result).shape == (100, 10)
        assert ctx.flops_used == 27 * 1000

    def test_truncnorm_array_bounds(self):
        """truncnorm with (100, 10) bounds bills the full broadcast output."""
        x = np.zeros(10)
        a = np.full((100, 10), -1.0)
        with BudgetContext(10**9, quiet=True) as ctx:
            result = truncnorm.pdf(x, a=a, b=2.0)
        assert np.asarray(result).shape == (100, 10)
        assert ctx.flops_used == 28 * 1000


# ---------------------------------------------------------------------------
# TestNoBroadcastRegression
# ---------------------------------------------------------------------------


class TestNoBroadcastRegression:
    """Scalar-parameter calls keep the exact pre-fix charge."""

    def test_array_x_scalar_params(self):
        xs = np.zeros(1000)
        with BudgetContext(10**9, quiet=True) as ctx:
            norm.pdf(xs)
        assert ctx.flops_used == 27 * 1000


# ---------------------------------------------------------------------------
# TestIncompatibleShapes
# ---------------------------------------------------------------------------


class TestIncompatibleShapes:
    """Billing must not mask numpy's error for non-broadcastable shapes."""

    def test_numpy_error_propagates(self):
        x = np.zeros(3)
        loc = np.zeros(4)
        with BudgetContext(10**9, quiet=True) as ctx:
            with pytest.raises(ValueError, match="broadcast"):
                norm.pdf(x, loc=loc)  # pyright: ignore[reportArgumentType]
        # The fail-closed charge falls back to numel(x), as before the fix.
        assert ctx.flops_used == 27 * 3
