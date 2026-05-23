"""Guards against silent regressions of the numpy-compat monkeypatch.

The patch loop in conftest._patch_numpy() resolves flopscope functions
via getattr on the module returned by _current_flopscope_numpy(). If
that module doesn't expose the names in REGISTRY, every lookup
silently AttributeErrors and the loop becomes a no-op — exercising
native numpy instead of flopscope. This file asserts the harness is
actually patching.
"""

import sys

import numpy as np


def test_patched_dict_is_populated():
    cf = sys.modules["tests.numpy_compat.conftest"]
    assert len(cf._PATCHED) > 100, (
        f"Compat harness patched only {len(cf._PATCHED)} entries; "
        "expected >100. _current_flopscope_numpy() likely points at the "
        "wrong module."
    )


def test_known_sentinels_patched():
    cf = sys.modules["tests.numpy_compat.conftest"]
    for name in ("zeros_like", "ones_like", "einsum", "sum"):
        assert name in cf._PATCHED, (
            f"Expected {name!r} to be patched onto numpy by the compat "
            f"harness; not present in _PATCHED."
        )


def test_numpy_attribute_actually_replaced():
    cf = sys.modules["tests.numpy_compat.conftest"]
    assert np.zeros_like is not cf._ORIGINAL_NUMPY.zeros_like, (
        "np.zeros_like is still the original numpy function; the "
        "monkeypatch did not take effect."
    )
