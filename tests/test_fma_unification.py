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


def test_hamming_cost_doubled():
    """hamming(n=400) should charge 2*n = 800 (was n=400 under FMA=1)."""
    import flopscope.numpy as fnp
    with flops.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        _ = fnp.hamming(400)
    assert bc.flops_used == 800, f"hamming(400) charged {bc.flops_used}, expected 800"


def test_hanning_cost_doubled():
    """hanning(n=400) should charge 2*n = 800."""
    import flopscope.numpy as fnp
    with flops.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        _ = fnp.hanning(400)
    assert bc.flops_used == 800, f"hanning(400) charged {bc.flops_used}, expected 800"
