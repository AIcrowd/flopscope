"""Shared pytest fixtures for flopscope-server tests."""

import pytest


@pytest.fixture(autouse=True)
def _reset_handle_counter():
    """Reset the process-global array + generator handle counters before each test.

    This ensures tests that assert exact handle names (e.g. "a0", "g0") remain
    deterministic.  Production NEVER calls these — the counters are only reset
    here for test isolation.
    """
    import flopscope_server._array_store as mod
    import flopscope_server._generator_store as gmod

    mod._reset_handle_counter()
    gmod._reset_gen_counter()
    yield


@pytest.fixture(autouse=True)
def _reset_core_budget_summary_state():
    """Isolate the authoritative summary epoch between server tests."""
    from flopscope._budget import budget_reset, get_active_budget

    active = get_active_budget()
    if active is not None:
        active.__exit__(None, None, None)
    budget_reset()
    yield
    active = get_active_budget()
    if active is not None:
        active.__exit__(None, None, None)
    budget_reset()
