"""Verify that _get_path_info no longer threads a symmetry oracle."""

import inspect

import flopscope._einsum as einsum_module


def test_symmetry_fingerprint_helper_removed():
    """_symmetry_fingerprint should be gone after Task 26."""
    assert not hasattr(einsum_module, "_symmetry_fingerprint")


def test_path_cache_signature_no_oracle_args():
    """The cached compute function no longer takes symmetry_fingerprint or use_inner_symmetry.

    Note: SubgraphSymmetryOracle IS now used in _einsum.py (Bug B fix) but ONLY
    in the post-cache path-group population step, not inside the cached function
    itself.  The old assertion that banned SubgraphSymmetryOracle from _einsum.py
    was guarding against the Task-26 entanglement where the oracle was invoked
    inside the LRU cache keying logic; that concern no longer applies because the
    oracle call is gated on `any(s is not None for s in per_op_symmetries)` and
    runs after the cache returns.
    """
    src = inspect.getsource(einsum_module)
    assert "symmetry_fingerprint" not in src
    assert "use_inner_symmetry" not in src
    # SubgraphSymmetryOracle is intentionally present in _einsum.py after the
    # Bug B renderer fix (operand symmetries propagated to path-walker steps).
    # The assertion below is removed; see comment above.
