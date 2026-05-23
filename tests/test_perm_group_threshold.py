"""Pin the dimino_budget-driven threshold migration (issue #71)."""

from __future__ import annotations

import pytest

import flopscope as flops
from flopscope._config import get_setting as _get_setting
from flopscope._perm_group import SymmetryGroup
from flopscope._pointwise import _is_oversized_for_cost_model


@pytest.fixture(autouse=True)
def _reset_dimino_budget():
    # Snapshot and restore so individual tests can tweak the budget.
    saved = _get_setting("dimino_budget")
    yield
    flops.configure(dimino_budget=saved)


class TestIsOversizedForCostModel:
    def test_none_group_is_not_oversized(self):
        assert _is_oversized_for_cost_model(None) is False

    def test_small_group_is_not_oversized(self):
        g = SymmetryGroup.symmetric(axes=(0, 1, 2))
        assert _is_oversized_for_cost_model(g) is False

    def test_large_known_kind_group_is_oversized_above_budget(self):
        # |S_12| = 479_001_600 > default budget 500_000
        g = SymmetryGroup.symmetric(axes=tuple(range(12)))
        assert _is_oversized_for_cost_model(g) is True

    def test_high_degree_but_small_order_is_not_oversized(self):
        # C_50 has degree 50 but |G| = 50. The old degree-based cap
        # would (wrongly) reject this; the new |G|-based cap accepts it.
        g = SymmetryGroup.cyclic(axes=tuple(range(50)))
        assert _is_oversized_for_cost_model(g) is False

    def test_configurable_budget_can_make_small_groups_oversized(self):
        flops.configure(dimino_budget=1)
        g = SymmetryGroup.symmetric(axes=(0, 1, 2))  # |G| = 6 > 1
        assert _is_oversized_for_cost_model(g) is True


class TestRemovedSymbols:
    def test_max_symmetry_degree_for_cost_no_longer_imports(self):
        with pytest.raises(ImportError):
            from flopscope._pointwise import _MAX_SYMMETRY_DEGREE_FOR_COST  # noqa: F401


class TestDiminoBudgetExceededDoesNotEscape:
    """Regression test: unknown-kind oversized groups must not leak the
    internal :class:`_DiminoBudgetExceeded` exception to user code (issue
    found in final review of the #71/#73 branch)."""

    def test_outer_does_not_raise_on_unknown_oversized_group(self):
        import warnings as _warnings

        import numpy as np

        from flopscope._symmetry_utils import wrap_with_symmetry
        from flopscope.errors import CostFallbackWarning

        flops.configure(dimino_budget=5)
        # S_4 generators — |G| = 24 > 5, will exceed budget.
        # from_generators produces an unknown-kind group, so order()
        # routes through _dimino.
        g = SymmetryGroup.from_generators(
            [[1, 2, 3, 0], [1, 0, 2, 3]], axes=tuple(range(4))
        )
        a = wrap_with_symmetry(np.ones((2, 2, 2, 2)), g)
        with flops.BudgetContext(flop_budget=int(1e10)):
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                # Must not raise _DiminoBudgetExceeded.
                np.multiply.outer(a, np.array([1.0]))
        cost_warnings = [w for w in caught if issubclass(w.category, CostFallbackWarning)]
        assert len(cost_warnings) >= 1, "expected a CostFallbackWarning"

    def test_tensordot_does_not_raise_on_unknown_oversized_group(self):
        import warnings as _warnings

        import numpy as np

        from flopscope._symmetry_utils import wrap_with_symmetry
        from flopscope.errors import CostFallbackWarning

        flops.configure(dimino_budget=5)
        g = SymmetryGroup.from_generators(
            [[1, 2, 3, 0], [1, 0, 2, 3]], axes=tuple(range(4))
        )
        a = wrap_with_symmetry(np.ones((2, 2, 2, 2)), g)
        b = wrap_with_symmetry(np.ones((2, 2, 2, 2)), g)
        with flops.BudgetContext(flop_budget=int(1e10)):
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                np.tensordot(a, b, axes=([0], [0]))
        cost_warnings = [w for w in caught if issubclass(w.category, CostFallbackWarning)]
        assert len(cost_warnings) >= 1, "expected a CostFallbackWarning"
