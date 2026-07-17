"""Regression tests pinned to flopscope#18.

Each test snapshots a specific bypass that was present before the fix.
If any test here regresses, the FLOP-accounting hole has reopened.
"""

import pickle

import numpy
import pytest

from flopscope import BudgetContext
from flopscope._ndarray import FlopscopeArray
from flopscope.numpy import random as merandom


class TestDefaultRngHatch:
    def test_default_rng_charges_flops(self):
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng = merandom.default_rng(42)
            rng.standard_normal(100)
        assert budget.flops_used == 100

    def test_default_rng_returns_flopscope_array(self):
        with BudgetContext(flop_budget=10**6, quiet=True):
            rng = merandom.default_rng(42)
            x = rng.standard_normal(10)
        assert isinstance(x, FlopscopeArray)

    def test_default_rng_uniform_charges(self):
        # Replicate whestbench/starterkit canonical idiom.
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng = merandom.default_rng(42)
            rng.uniform(0.0, 1.0, size=(8, 8))
        assert budget.flops_used == 3 * 64


class TestRandomStateHatch:
    def test_RandomState_charges_flops(self):
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rs = merandom.RandomState(42)
            rs.randn(10)
        assert budget.flops_used == 10

    def test_RandomState_returns_flopscope_array(self):
        with BudgetContext(flop_budget=10**6, quiet=True):
            rs = merandom.RandomState(42)
            z = rs.randn(10)
        assert isinstance(z, FlopscopeArray)


class TestGetattrHatch:
    def test_unknown_attribute_raises(self):
        # Was: silent forward to numpy.random with no counting.
        with pytest.raises(AttributeError, match="default_rng"):
            _ = merandom.completely_made_up_attribute

    def test_random_integers_raises(self):
        # numpy.random.random_integers exists but is not in our explicit list.
        with pytest.raises(AttributeError):
            _ = merandom.random_integers


class TestModuleLevelUnchanged:
    def test_randn_no_warning(self):
        # The fix must not introduce a DeprecationWarning on module-level samplers.
        # Use pytest.warns(None) by treating any warning as a failure via -W error
        # (project conftest doesn't enforce this; check explicitly).
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn any warning into an error
            with BudgetContext(flop_budget=10**6, quiet=True):
                merandom.randn(5)  # must not raise

    def test_randn_still_counts(self):
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            merandom.randn(10, 20)
        assert budget.flops_used == 200


class TestPickleRoundtrip:
    def test_pickle_default_rng_preserves_counting(self):
        rng = merandom.default_rng(42)
        revived = pickle.loads(pickle.dumps(rng))
        # Type identity preserved
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        assert isinstance(revived, _CountedGenerator)
        # Counting still works
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            revived.standard_normal(7)
        assert budget.flops_used == 7

    def test_pickle_RandomState_preserves_counted_class(self):
        rs = merandom.RandomState(42)
        revived = pickle.loads(pickle.dumps(rs))
        from flopscope.numpy.random._counted_classes import _CountedRandomState

        assert isinstance(revived, _CountedRandomState)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            revived.randn(11)
        assert budget.flops_used == 11

    def test_pickle_RandomState_preserves_state(self):
        # Pickle round-trip must preserve full state for bit-identical streams.
        rs1 = merandom.RandomState(42)
        rs1.randn(5)  # advance state, populate Box-Muller cache
        rs2 = pickle.loads(pickle.dumps(rs1))
        with BudgetContext(flop_budget=10**6, quiet=True):
            a = rs1.randn(7)
            b = rs2.randn(7)
        assert (a == b).all(), f"streams diverge after pickle round-trip: {a} vs {b}"

    def test_copy_copy_default_rng_preserves_counted_class(self):
        import copy as _copy

        from flopscope.numpy.random._counted_classes import _CountedGenerator

        rng = merandom.default_rng(42)
        clone = _copy.copy(rng)
        assert isinstance(clone, _CountedGenerator)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            clone.standard_normal(9)
        assert budget.flops_used == 9

    def test_copy_deepcopy_RandomState_preserves_counted_class(self):
        import copy as _copy

        from flopscope.numpy.random._counted_classes import _CountedRandomState

        rs = merandom.RandomState(42)
        clone = _copy.deepcopy(rs)
        assert isinstance(clone, _CountedRandomState)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            clone.randn(13)
        assert budget.flops_used == 13


class TestSpawn:
    def test_spawned_children_are_counted(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        rng = merandom.default_rng(42)
        children = rng.spawn(3)
        assert all(isinstance(c, _CountedGenerator) for c in children)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            children[0].standard_normal(11)
        assert budget.flops_used == 11


class TestBitGeneratorPassthrough:
    def test_passthrough_classes_resolve(self):
        for name in (
            "BitGenerator",
            "MT19937",
            "PCG64",
            "PCG64DXSM",
            "Philox",
            "SFC64",
        ):
            cls = getattr(merandom, name)
            assert cls is getattr(numpy.random, name)

    def test_construction_via_explicit_bit_generator(self):
        # Generator(BitGenerator(seed)) is a documented numpy idiom.
        rng = merandom.Generator(merandom.PCG64(42))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng.normal(0, 1, size=10)
        assert budget.flops_used == 10


class TestCrossApiFlopParity:
    """Same physical op charges the same FLOPs regardless of which API was called.

    These tests load empirical weights (load_weights()) so the alias fallback
    is exercised — without it, module-level would charge 16.0 × n and
    method-level would charge 1.0 × n, asymmetric.
    """

    def test_randn_parity_module_vs_default_rng(self):
        from flopscope._weights import load_weights

        load_weights()
        with BudgetContext(flop_budget=10**6, quiet=True) as b1:
            merandom.randn(10)
        with BudgetContext(flop_budget=10**6, quiet=True) as b2:
            rng = merandom.default_rng(42)
            rng.standard_normal(10)
        assert b1.flops_used == b2.flops_used, (
            f"FLOP asymmetry: randn={b1.flops_used}, "
            f"default_rng().standard_normal={b2.flops_used}"
        )

    def test_normal_parity_all_three_apis(self):
        from flopscope._weights import load_weights

        load_weights()
        with BudgetContext(flop_budget=10**6, quiet=True) as b1:
            merandom.normal(0, 1, size=10)
        with BudgetContext(flop_budget=10**6, quiet=True) as b2:
            merandom.default_rng(42).normal(0, 1, size=10)
        with BudgetContext(flop_budget=10**6, quiet=True) as b3:
            merandom.RandomState(42).normal(0, 1, size=10)
        assert b1.flops_used == b2.flops_used == b3.flops_used, (
            f"normal asymmetry: module={b1.flops_used}, "
            f"default_rng={b2.flops_used}, RandomState={b3.flops_used}"
        )

    def test_2d_shuffle_charges_shape_zero_across_apis(self):
        from flopscope._weights import load_weights

        load_weights()
        a = numpy.arange(50).reshape(5, 10)
        with BudgetContext(flop_budget=10**6, quiet=True) as b1:
            merandom.shuffle(a.copy())
        with BudgetContext(flop_budget=10**6, quiet=True) as b2:
            merandom.default_rng(42).shuffle(a.copy())
        with BudgetContext(flop_budget=10**6, quiet=True) as b3:
            merandom.RandomState(42).shuffle(a.copy())
        # All charge shape[0]=5 × weight 4.0 (access tier) = 20
        assert b1.flops_used == b2.flops_used == b3.flops_used == 20, (
            f"shuffle 2D asymmetry: module={b1.flops_used}, "
            f"default_rng={b2.flops_used}, RandomState={b3.flops_used}"
        )

    def test_uniform_parity_all_three_apis(self):
        """Uniform sampler — common participant code path (whestbench/starterkit)."""
        from flopscope._weights import load_weights

        load_weights()
        with BudgetContext(flop_budget=10**6, quiet=True) as b1:
            merandom.uniform(0, 1, size=(8, 8))
        with BudgetContext(flop_budget=10**6, quiet=True) as b2:
            merandom.default_rng(42).uniform(0, 1, size=(8, 8))
        with BudgetContext(flop_budget=10**6, quiet=True) as b3:
            merandom.RandomState(42).uniform(0, 1, size=(8, 8))
        assert b1.flops_used == b2.flops_used == b3.flops_used

    def test_shuffle_positional_axis_parity_across_apis(self):
        """rng.shuffle(arr, 1) charges the same across APIs."""
        from flopscope._weights import load_weights

        load_weights()
        a = numpy.arange(50).reshape(5, 10)
        with BudgetContext(flop_budget=10**6, quiet=True) as b1:
            merandom.default_rng(42).shuffle(a.copy(), 1)
        with BudgetContext(flop_budget=10**6, quiet=True) as b2:
            merandom.default_rng(42).shuffle(a.copy(), axis=1)
        # Positional and keyword forms charge identically; both should be
        # shape[1]=10 × weight 4.0 (access tier) = 40
        assert b1.flops_used == b2.flops_used == 40
