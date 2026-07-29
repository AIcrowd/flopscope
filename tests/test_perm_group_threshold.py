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
        # |S_12| = 479_001_600 >> default budget 50_000
        g = SymmetryGroup.symmetric(axes=tuple(range(12)))
        assert _is_oversized_for_cost_model(g) is True

    def test_s_8_fits_under_default_budget(self):
        # |S_8| = 40_320 < 50_000 → fits with margin.
        # Confirms the shipped default accepts the S_8-class cold-cost
        # tier deliberately (see _config.py docstring).
        g = SymmetryGroup.symmetric(axes=tuple(range(8)))
        assert _is_oversized_for_cost_model(g) is False

    def test_s_9_exceeds_default_budget(self):
        # |S_9| = 362_880 > 50_000 → bails to dense cost.
        g = SymmetryGroup.symmetric(axes=tuple(range(9)))
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


class TestShippedDefaultBudget:
    def test_default_dimino_budget_is_50000(self):
        # Pinned: |S_8| = 40_320 fits with margin; |S_9| = 362_880 doesn't.
        # See benchmarks/_perm_group_calibration.py for the empirical
        # justification. Tune via flops.configure(dimino_budget=...).
        assert _get_setting("dimino_budget") == 50_000


class TestRemovedSymbols:
    def test_max_symmetry_degree_for_cost_no_longer_imports(self):
        import importlib

        module = importlib.import_module("flopscope._pointwise")
        with pytest.raises(AttributeError):
            module._MAX_SYMMETRY_DEGREE_FOR_COST  # noqa: B018


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
        cost_warnings = [
            w for w in caught if issubclass(w.category, CostFallbackWarning)
        ]
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
        cost_warnings = [
            w for w in caught if issubclass(w.category, CostFallbackWarning)
        ]
        assert len(cost_warnings) >= 1, "expected a CostFallbackWarning"


class TestUnknownKindOversizedGroupDegradesToDense:
    """The unique-element counter must degrade, never leak _DiminoBudgetExceeded.

    ``fnp.full((2,)*9, 0)`` auto-infers the full S_9 exchange symmetry as a
    RAW generator set (unknown kind): ``order()`` has no closed form there, so
    ``burnside_unique_count`` enumerates and blows ``dimino_budget``. Before
    the fix, the private ``_DiminoBudgetExceeded`` escaped through any billed
    op on such a tensor -- numpy 2.4's ``TestCreationFuncs::test_full`` (the
    first genuine 2.4 run of the borrowed suites) found exactly this.
    """

    def test_billing_on_oversized_inferred_symmetry_charges_dense(self):
        import warnings

        import flopscope.numpy as fnp
        from flopscope.errors import CostFallbackWarning

        with flops.BudgetContext(flop_budget=10**12, quiet=True) as b:
            sym = fnp.full((2,) * 9, 0)
            before = b.flops_used
            with pytest.warns(CostFallbackWarning, match="dense"):
                _ = sym == 0
            assert b.flops_used - before == 512  # dense numel(2**9), no discount
            # Unary ops route through the same counter; must also survive.
            before = b.flops_used
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fnp.abs(sym)
            assert b.flops_used - before == 512

    def test_small_inferred_symmetry_keeps_the_discount(self):
        import flopscope.numpy as fnp

        with flops.BudgetContext(flop_budget=10**12, quiet=True) as b:
            sym = fnp.full((2, 2, 2), 0)  # inferred S_3, |G| = 6: enumerable
            before = b.flops_used
            _ = sym == 0
            # multiset count C(2+3-1, 3) = 4 unique elements, not dense 8
            assert b.flops_used - before == 4
