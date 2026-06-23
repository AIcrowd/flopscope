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
    """Reset client connection + budget state and open an ambient budget per test.

    The ambient ``BudgetContext`` is REQUIRED, not optional: unlike native
    flopscope (which lazily uses a global-default budget for unbudgeted ops), the
    CLIENT raises ``NoBudgetContextError: no active session`` if an op runs with
    no active budget. NumPy's own test suite (which this harness runs against the
    client) never opens a budget, so without this ambient context every NumPy
    test would fail spuriously. ``10**15`` FLOPs is effectively unbounded for
    NumPy's tiny test arrays; ``quiet=True`` suppresses per-test output.

    Hand-written harness tests therefore must NOT open their own BudgetContext
    (the client rejects nested contexts) — they rely on this ambient one.
    """
    import flopscope
    from flopscope._connection import reset_connection
    from flopscope._budget import _reset_global_default

    reset_connection()
    _reset_global_default()
    ctx = flopscope.BudgetContext(flop_budget=10**15, quiet=True)
    ctx.__enter__()
    try:
        yield
    finally:
        ctx.__exit__(None, None, None)
        reset_connection()
        _reset_global_default()
