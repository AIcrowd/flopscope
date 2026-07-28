"""Client-side unit tests that need the client on sys.path but NO server.

Lives under ``tests/client_compat/`` so the parent conftest still puts the CLIENT
flopscope ahead of the in-process ``src/`` package — that path setup is exactly
what these tests need. What they do not need is either of the parent's autouse
fixtures:

* ``_server`` (session-scoped) spawns a real flopscope-server subprocess, which
  requires the server virtualenv and makes a client-only checkout unable to run
  this directory;
* ``_fresh_connection_and_budget`` opens an ambient ``BudgetContext`` against
  that server, so without it these tests fail at setup with "session already
  open" / no server.

These tests assert on pure client-side accounting logic and never touch the
wire, so both are overridden by name — the standard pytest mechanism. Keep both
overrides: dropping the ``_server`` one silently reintroduces the subprocess and
the serverless claim above stops being true.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _server():
    """No-op override: do not spawn flopscope-server for this directory."""
    yield None


@pytest.fixture(autouse=True)
def _fresh_connection_and_budget():
    """No-op override: these tests never open a budget or touch the wire."""
    yield
