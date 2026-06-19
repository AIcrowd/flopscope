"""Shared pytest fixtures for flopscope-server tests."""

import pytest


@pytest.fixture(autouse=True)
def _reset_handle_counter():
    """Reset the process-global handle counter before each test.

    This ensures tests that assert exact handle names (e.g. "a0") remain
    deterministic.  Production NEVER calls this — the counter is only reset
    here for test isolation.
    """
    import flopscope_server._array_store as mod

    mod._reset_handle_counter()
    yield
