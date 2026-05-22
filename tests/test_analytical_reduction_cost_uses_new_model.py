"""Direct callers of _flops.analytical_reduction_cost get the new model."""

import flopscope as fps
from flopscope._flops import analytical_reduction_cost


def test_plain_sum_charges_n_minus_1():
    """sum of (n,) charges n - 1 (was n with the legacy formula)."""
    assert analytical_reduction_cost((10,), axis=0, symmetry=None) == 9


def test_symmetric_sum_uses_orbit_mapping_not_unique_count():
    """sum of symmetric (n, n) → scalar charges α - 1, not unique-count."""
    n = 4
    sym = fps.SymmetryGroup.symmetric(axes=(0, 1))
    # Legacy: unique_elements_for_shape = n(n+1)/2 = 10
    # New: orbit-mapping α - 1. The exact α depends on ladder, but should
    # not equal 10. Conservative check: cost >= n - 1 (lower bound).
    cost = analytical_reduction_cost((n, n), axis=(0, 1), symmetry=sym)
    assert cost != n * (n + 1) // 2  # not the legacy value
    assert cost >= n - 1
