"""Methods mode: run numpy's ndarray method/operator tests against the CLIENT.

This dir lives UNDER tests/client_compat/, so the parent (function-mode) conftest
also loads here — fine for a methods-mode run (its function patches are harmless;
the sessionstart re-patch below wins the np.array conflict). The REVERSE is unsafe:
a function-mode run that collects this dir would globally construction-patch
np.array and break the function suite. The `test-client-parity` Makefile target
therefore passes `--ignore=tests/client_compat/methods`; run this dir only via
`make test-client-parity-methods` (or an explicit path to it).
"""

from __future__ import annotations

import fnmatch
import sys

import pytest

from .._server_fixture import ensure_client_on_path, start_server, stop_server
from ..xfails_client import XFAIL_PATTERNS

ensure_client_on_path()
for _mod_name in list(sys.modules.keys()):
    if _mod_name == "flopscope" or _mod_name.startswith("flopscope."):
        del sys.modules[_mod_name]

import numpy as _np_warmup  # noqa: E402

_np_warmup.random.default_rng()
del _np_warmup


def pytest_configure(config):
    from . import _patch_constructors

    _patch_constructors.patch()


def pytest_sessionstart(session):
    # Re-apply after all pytest_configure hooks have run (parent conftest's
    # pytest_configure fires AFTER ours, since conftest hooks call child-first;
    # parent's _coerce.install() overwrites our construction patch). Re-patching
    # here — after all configure hooks — ensures we win.
    from . import _patch_constructors

    _patch_constructors.patch()


def pytest_unconfigure(config):
    from . import _patch_constructors

    _patch_constructors.unpatch()


@pytest.fixture(scope="session", autouse=True)
def _server():
    proc = start_server()
    yield proc
    stop_server(proc)


@pytest.fixture(autouse=True)
def _fresh_connection_and_budget():
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
    for item in items:
        for pattern, reason in XFAIL_PATTERNS.items():
            if fnmatch.fnmatch(item.nodeid, pattern) or pattern in item.nodeid:
                item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
                break
