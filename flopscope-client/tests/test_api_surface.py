"""Tests for the flopscope client API surface.

These tests verify that the public API is correctly wired up without
requiring a running server.  They test:

- Registry correctness (entry count, lookups, categories)
- Module-level constants and dtype exports
- Class exports (ndarray, BudgetContext, errors)
- Auto-generated proxy functions exist and are callable
- __getattr__ error messages for blacklisted and unknown names
- Submodule structure (fft, linalg, random)
- flops local cost functions
"""

from __future__ import annotations

import math

import pytest
from flopscope._remote_array import RemoteArray

import flopscope as we
from flopscope._registry import (
    BLACKLISTED,
    COUNTED_BINARY,
    COUNTED_CUSTOM,
    COUNTED_REDUCTION,
    COUNTED_UNARY,
    FREE,
    FUNCTION_CATEGORIES,
    get_category,
    is_blacklisted,
    is_valid_op,
    iter_proxyable,
)

# ===================================================================
# Registry tests
# ===================================================================


class TestRegistry:
    """Tests for _registry.py and _registry_data.py."""

    def test_registry_has_482_plus_entries(self):
        assert len(FUNCTION_CATEGORIES) >= 400, (
            f"Registry suspiciously small: {len(FUNCTION_CATEGORIES)} entries. "
            f"Run: uv run scripts/sync_client.py"
        )

    @pytest.mark.parametrize("op", ["add", "exp", "zeros", "einsum", "dot", "matmul"])
    def test_is_valid_op_known(self, op):
        assert is_valid_op(op) is True

    def test_is_valid_op_unknown(self):
        assert is_valid_op("totally_fake_function_xyz") is False

    def test_get_category_add(self):
        assert get_category("add") == COUNTED_BINARY

    def test_get_category_exp(self):
        assert get_category("exp") == COUNTED_UNARY

    def test_get_category_sum(self):
        assert get_category("sum") == COUNTED_REDUCTION

    def test_get_category_einsum(self):
        assert get_category("einsum") == COUNTED_CUSTOM

    def test_get_category_zeros(self):
        assert get_category("zeros") == FREE

    def test_get_category_unknown(self):
        assert get_category("totally_fake_xyz") is None

    @pytest.mark.parametrize("op", ["savetxt", "loadtxt"])
    def test_blacklisted_category(self, op):
        assert get_category(op) == BLACKLISTED

    def test_is_blacklisted(self):
        assert is_blacklisted("savetxt") is True
        assert is_blacklisted("save") is False  # now a supported pickle-free I/O op
        assert is_blacklisted("add") is False
        assert is_blacklisted("unknown_xyz") is False

    def test_iter_proxyable_nonempty(self):
        ops = iter_proxyable()
        assert len(ops) > 100
        # Should not contain any blacklisted operations
        for op in ops:
            assert get_category(op) != BLACKLISTED

    def test_iter_proxyable_prefix_filter(self):
        random_ops = iter_proxyable(prefix="random.")
        assert len(random_ops) > 10
        for op in random_ops:
            assert op.startswith("random.")

    def test_fft_ops_are_registered(self):
        fft_ops = [k for k in FUNCTION_CATEGORIES if k.startswith("fft.")]
        assert len(fft_ops) > 0

    def test_linalg_svd_is_counted_custom(self):
        assert get_category("linalg.svd") == COUNTED_CUSTOM


# ===================================================================
# Constants and dtype exports
# ===================================================================


class TestConstants:
    """Tests for constants exported by the top-level package."""

    def test_pi(self):
        assert we.pi == pytest.approx(math.pi)

    def test_e(self):
        assert we.e == pytest.approx(math.e)

    def test_inf(self):
        assert we.inf == math.inf

    def test_nan(self):
        assert math.isnan(we.nan)

    def test_newaxis(self):
        assert we.newaxis is None


class TestDtypes:
    """Dtype names are dual-purpose objects, not strings (matches numpy)."""

    def test_float32_is_not_a_string(self):
        assert not isinstance(we.float32, str)
        assert (we.float32 == "float32") is False

    def test_float32_is_callable(self):
        assert callable(we.float32)

    def test_dtype_object_equals_string(self):
        assert we.dtype("float32") == "float32"

    def test_uint16_uint32_uint64_present(self):
        for n in ("uint16", "uint32", "uint64"):
            assert callable(getattr(we, n))

    def test_str_of_label_is_wire_name(self):
        assert str(we.bool_) == "bool"
        assert str(we.float64) == "float64"


# ===================================================================
# Class exports
# ===================================================================


class TestClassExports:
    """Tests for class/type exports."""

    def test_ndarray_is_RemoteArray(self):
        assert we.ndarray is RemoteArray

    def test_BudgetContext_is_class(self):
        from flopscope._budget import BudgetContext

        assert we.BudgetContext is BudgetContext

    def test_OpRecord_is_class(self):
        from flopscope._budget import OpRecord

        assert we.OpRecord is OpRecord

    def test_errors_exported(self):
        assert issubclass(we.BudgetExhaustedError, we.FlopscopeError)
        assert issubclass(we.NoBudgetContextError, we.FlopscopeError)
        assert issubclass(we.SymmetryError, we.FlopscopeError)
        assert issubclass(we.FlopscopeServerError, we.FlopscopeError)
        assert issubclass(we.FlopscopeWarning, UserWarning)


# ===================================================================
# Proxy function existence
# ===================================================================


class TestProxyFunctions:
    """Tests that auto-generated proxy functions exist and are callable."""

    @pytest.mark.parametrize(
        "name",
        [
            "add",
            "subtract",
            "multiply",
            "divide",
            "exp",
            "log",
            "sqrt",
            "abs",
            "zeros",
            "ones",
            "eye",
            "arange",
            "dot",
            "matmul",
            "sum",
            "mean",
            "reshape",
            "transpose",
            "concatenate",
            "stack",
            "clip",
        ],
    )
    def test_proxy_exists_and_callable(self, name):
        fn = getattr(we, name)
        assert callable(fn), f"we.{name} should be callable"

    def test_array_is_callable(self):
        assert callable(we.array)

    def test_einsum_is_callable(self):
        assert callable(we.einsum)

    def test_proxy_has_correct_name(self):
        assert we.add.__name__ == "add"
        assert we.exp.__name__ == "exp"
        assert we.zeros.__name__ == "zeros"


# ===================================================================
# __getattr__ error messages
# ===================================================================


class TestGetattr:
    """Tests for __getattr__ error handling on blacklisted/unknown names."""

    def test_io_ops_are_callable(self):
        # save/load/savez/savez_compressed are now supported (pickle-free I/O).
        assert callable(we.save)
        assert callable(we.load)
        assert callable(we.savez)
        assert callable(we.savez_compressed)

    def test_blacklisted_top_level_still_raises(self):
        with pytest.raises(AttributeError, match="intentionally excluded"):
            _ = we.savetxt

    def test_unknown_top_level(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = we.completely_nonexistent_function_xyz

    def test_fft_registered_not_blacklisted(self):
        # fft.fft is now counted_custom, not blacklisted
        with pytest.raises(AttributeError, match="registered but not yet implemented"):
            _ = we.fft.fft

    def test_fft_rfft_registered_not_blacklisted(self):
        # fft.rfft is now counted_custom, not blacklisted
        with pytest.raises(AttributeError, match="registered but not yet implemented"):
            _ = we.fft.rfft

    def test_unknown_fft(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = we.fft.nonexistent_xyz

    def test_linalg_eig_is_proxy(self):
        # linalg.eig is now a counted_custom proxy, not blacklisted
        assert callable(we.linalg.eig)

    def test_unknown_linalg(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = we.linalg.nonexistent_xyz

    def test_unknown_random(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = we.random.nonexistent_xyz


# ===================================================================
# Submodule structure
# ===================================================================


class TestSubmodules:
    """Tests for submodule accessibility."""

    def test_fft_is_module(self):
        import types

        assert isinstance(we.fft, types.ModuleType)

    def test_linalg_is_module(self):
        import types

        assert isinstance(we.linalg, types.ModuleType)

    def test_random_is_module(self):
        import types

        assert isinstance(we.random, types.ModuleType)

    def test_flops_is_module(self):
        import types

        assert isinstance(we.flops, types.ModuleType)

    def test_linalg_svd_callable(self):
        assert callable(we.linalg.svd)

    def test_random_functions_callable(self):
        for name in [
            "randn",
            "normal",
            "uniform",
            "rand",
            "seed",
            "choice",
            "permutation",
            "shuffle",
        ]:
            fn = getattr(we.random, name)
            assert callable(fn), f"we.random.{name} should be callable"


# ===================================================================
# flops local cost functions
# ===================================================================


class TestFlopsLocal:
    """Tests for locally computable flops cost functions."""

    def test_pointwise_cost(self):
        assert we.flops.pointwise_cost("exp", shape=(3, 4)) == 12
        assert we.flops.pointwise_cost("add", shape=(10,)) == 10
        assert we.flops.pointwise_cost("sqrt", shape=()) == 1

    def test_reduction_cost(self):
        assert we.flops.reduction_cost("sum", input_shape=(3, 4), axis=0) == 12
        assert we.flops.reduction_cost("mean", input_shape=(3, 4), axis=None) == 12
        assert we.flops.reduction_cost("max", input_shape=(10,)) == 10

    def test_einsum_cost_is_callable(self):
        assert callable(we.flops.einsum_cost)

    def test_svd_cost_is_callable(self):
        assert callable(we.flops.svd_cost)


# ===================================================================
# array() and einsum() special cases (structure only, no server)
# ===================================================================


class TestArraySpecialCase:
    """Tests for the array() function structure (no server needed)."""

    def test_array_returns_existing_remote_array(self):
        ra = RemoteArray(handle_id="test-id", shape=(2, 3), dtype="float64")
        assert we.array(ra) is ra

    def test_array_rejects_unsupported_type(self):
        with pytest.raises(TypeError, match="Cannot create array"):
            we.array("not a valid input")

    def test_flatten_helper(self):
        from flopscope import _flatten

        flat, shape = _flatten([[1, 2], [3, 4]])
        assert flat == [1, 2, 3, 4]
        assert shape == (2, 2)

    def test_flatten_1d(self):
        from flopscope import _flatten

        flat, shape = _flatten([1, 2, 3])
        assert flat == [1, 2, 3]
        assert shape == (3,)

    def test_flatten_3d(self):
        from flopscope import _flatten

        flat, shape = _flatten([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert flat == [1, 2, 3, 4, 5, 6, 7, 8]
        assert shape == (2, 2, 2)

    def test_flatten_inhomogeneous_raises(self):
        from flopscope import _flatten

        with pytest.raises(ValueError, match="Inhomogeneous"):
            _flatten([[1, 2], [3]])


# ===================================================================
# __all__ coverage sanity check
# ===================================================================


class TestModuleCompleteness:
    """Sanity checks for module completeness."""

    def test_no_submodule_ops_in_top_level(self):
        """Submodule ops (random.*, linalg.*, fft.*) should NOT be
        directly in the top-level namespace."""
        # random.randn should not be accessible as we.randn
        # (it is accessed as we.random.randn)
        for prefix in ("random.", "linalg.", "fft."):
            for key in FUNCTION_CATEGORIES:
                if key.startswith(prefix):
                    short = key.split(".", 1)[1]
                    # If a top-level function or submodule exists with the
                    # same name, skip the check (e.g. "random" is both a
                    # submodule and random.random).
                    if short in FUNCTION_CATEGORIES or short in (
                        "fft",
                        "linalg",
                        "random",
                    ):
                        continue
                    # Should NOT be in top-level namespace
                    assert not hasattr(we, short) or short in dir(we.__class__), (
                        f"{key} should not be accessible as we.{short}"
                    )
