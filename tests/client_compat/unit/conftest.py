"""Client-side unit tests that need the client on sys.path but NO server.

Lives under ``tests/client_compat/`` so the parent conftest still puts the CLIENT
flopscope ahead of the in-process ``src/`` package — that path setup is exactly
what these tests need. What they do NOT need is the parent's autouse
``_fresh_connection_and_budget`` fixture, which opens an ambient
``BudgetContext`` against a live flopscope-server; these tests assert on pure
client-side accounting logic and would otherwise fail at setup with
"session already open" / no server. Overriding the fixture by name here is the
standard pytest mechanism for that.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fresh_connection_and_budget():
    """No-op override: these tests never touch the wire."""
    yield
