"""Performance smoke tests for SymmetryGroup.order() closed-form (#71).

Marked ``slow`` so CI doesn't fail on machine-speed flakes. Run locally:

    uv run pytest tests/test_perm_group_perf.py -v --no-header
"""

from __future__ import annotations

import time

import pytest

from flopscope._perm_group import SymmetryGroup


@pytest.mark.slow
class TestOrderPerformance:
    def test_s_20_order_under_one_millisecond(self):
        g = SymmetryGroup.symmetric(axes=tuple(range(20)))
        t = time.perf_counter()
        result = g.order()
        elapsed_ms = (time.perf_counter() - t) * 1000.0
        # |S_20| = 2_432_902_008_176_640_000
        assert result == 2_432_902_008_176_640_000
        assert elapsed_ms < 1.0, f"S_20.order() took {elapsed_ms:.3f}ms (expected <1ms)"

    def test_direct_product_s10_x_s10_under_one_millisecond(self):
        a = SymmetryGroup.symmetric(axes=tuple(range(10)))
        b = SymmetryGroup.symmetric(axes=tuple(range(10, 20)))
        g = SymmetryGroup.direct_product(a, b)
        t = time.perf_counter()
        result = g.order()
        elapsed_ms = (time.perf_counter() - t) * 1000.0
        # |S_10| * |S_10| = 3_628_800 * 3_628_800
        assert result == 3_628_800 * 3_628_800
        assert elapsed_ms < 1.0, (
            f"direct_product(S_10, S_10).order() took {elapsed_ms:.3f}ms"
        )

    def test_high_degree_cyclic_is_fast(self):
        # C_100: degree 100, |G| = 100. Old degree-cap would reject; new |G|-cap accepts.
        g = SymmetryGroup.cyclic(axes=tuple(range(100)))
        t = time.perf_counter()
        result = g.order()
        elapsed_ms = (time.perf_counter() - t) * 1000.0
        assert result == 100
        assert elapsed_ms < 1.0
