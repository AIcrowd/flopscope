"""Shared pytest fixtures for the flopscope-client test suite.

Mirrors the in-process ``tests/conftest.py`` weight isolation: FLOP weights are
reset to unit (1.0) around every test so that local cost-query assertions
(e.g. ``flops.pointwise_cost``) check the analytical FLOP count independent of
the packaged empirical weights — which are applied in production and may be
recalibrated. Tests that exercise weighting (``test_weights.py``) load weights
explicitly in their bodies, so this reset does not interfere with them.
"""

import pytest

import flopscope._weights as weights_module
from flopscope._weights import reset_weights


@pytest.fixture(autouse=True)
def _reset_weights():
    reset_weights()
    weights_module._WARNED_MESSAGES.clear()
    yield
    reset_weights()
    weights_module._WARNED_MESSAGES.clear()


@pytest.fixture(autouse=True)
def _reset_pending_handles():
    """Isolate the module-global free queue: RemoteArray GC enqueues handle ids
    into flopscope._handles, and tests asserting exact send sequences must start
    from an empty queue. Drains before AND after each test."""
    from flopscope import _handles

    _handles.drain_pending()
    yield
    _handles.drain_pending()
