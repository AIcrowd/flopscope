"""Client-parity harness: run NumPy's suite against the flopscope CLIENT."""
from __future__ import annotations

import sys

import pytest

from ._server_fixture import ensure_client_on_path, start_server, stop_server

# MUST happen before any `import flopscope`, so the client wins over native src/.
# Also purge any already-cached native flopscope from sys.modules so the client
# package wins when this conftest is imported by an xdist worker that already
# has the native src/ on sys.path.
ensure_client_on_path()
for _mod_name in list(sys.modules.keys()):
    if _mod_name == "flopscope" or _mod_name.startswith("flopscope."):
        del sys.modules[_mod_name]


@pytest.fixture(scope="session", autouse=True)
def _server():
    proc = start_server()
    yield proc
    stop_server(proc)


@pytest.fixture(autouse=True)
def _fresh_connection_and_budget():
    """Reset client connection + budget state around every test.

    Does NOT open a BudgetContext — each test is responsible for that.
    Mirrors the ``_reset_client`` fixture in flopscope-client/tests/test_full_integration.py.
    """
    from flopscope._connection import reset_connection
    from flopscope._budget import _reset_global_default

    reset_connection()
    _reset_global_default()
    yield
    reset_connection()
    _reset_global_default()
