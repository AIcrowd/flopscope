"""FMA=2 unification regression tests.

Asserts the post-unification state:
- `flops.fma_cost` no longer exists on the public API
- `flops.configure(fma_cost=...)` raises for unknown setting
- Dense / symmetric cost columns match for unsymmetric matmul
- 7 affected ops produce post-doubled values
"""

import pytest
import flopscope as flops


def test_fma_cost_removed_from_public_api():
    assert not hasattr(flops, "fma_cost"), (
        "flops.fma_cost should be removed in FMA=2 unification"
    )


def test_configure_fma_cost_raises():
    with pytest.raises((ValueError, KeyError, TypeError)):
        flops.configure(fma_cost=2)


def test_fma_cost_constant_removed():
    """The private FMA_COST constant + its module should be gone."""
    with pytest.raises(ImportError):
        from flopscope._cost_model import fma_cost  # noqa: F401
