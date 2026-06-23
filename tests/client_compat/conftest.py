"""Client-parity harness: run NumPy's suite against the flopscope CLIENT."""

from __future__ import annotations

import fnmatch
import sys

import pytest

from ._server_fixture import ensure_client_on_path, start_server, stop_server
from .xfails_client import XFAIL_PATTERNS

# MUST happen before any `import flopscope`, so the client wins over native src/.
# Also purge any already-cached native flopscope from sys.modules so the client
# package wins when this conftest is imported by an xdist worker that already
# has the native src/ on sys.path.
ensure_client_on_path()
for _mod_name in list(sys.modules.keys()):
    if _mod_name == "flopscope" or _mod_name.startswith("flopscope."):
        del sys.modules[_mod_name]

# Force NumPy's lazy RNG entropy init NOW, while numpy is still unpatched, so it
# can never trigger a patched op (-> client dispatch) later. numpy.random's
# SeedSequence.get_assembled_entropy calls functions the patch replaces; if that
# init first runs after patching (e.g. mid-test with an open budget), it would
# dispatch to the server. Doing it here caches the init under native numpy.
import numpy as _np_warmup  # noqa: E402

_np_warmup.random.default_rng()
del _np_warmup


def pytest_configure(config):
    # Import _coerce FIRST so it snapshots the genuine numpy constructors before
    # patch() runs. Then patch numpy -> client. Then install the RemoteArray ->
    # numpy coercion AFTER patch() so it owns array/asarray/asanyarray (those do
    # output coercion for asserts, not client routing).
    from . import _coerce
    from ._patch_client import patch

    patch()
    _coerce.install()


def pytest_unconfigure(config):
    from . import _coerce
    from ._patch_client import unpatch

    unpatch()
    _coerce.uninstall()


@pytest.fixture()
def _patch_active():
    # patch() ran at configure; this fixture documents the dependency for tests
    # that assert the swap is active.
    yield


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
    from flopscope._connection import reset_connection

    import flopscope
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


def pytest_collection_modifyitems(config, items):
    """Mark known by-design client divergences as (non-strict) xfail."""
    for item in items:
        for pattern, reason in XFAIL_PATTERNS.items():
            if fnmatch.fnmatch(item.nodeid, pattern) or pattern in item.nodeid:
                item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
                break
