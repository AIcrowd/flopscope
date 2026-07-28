"""Every op that runs a participant callback must be registered as one.

``REGISTRY[name]["local_callback"]`` is the single source of truth: it feeds
both the client's ``LOCAL_CALLBACK_OPS`` and the public
``remote_unsupported_ops()``. An op missing the flag is reported as remotely
supported when it is not, and is easy to overlook when the callback family is
updated.
"""

from __future__ import annotations

import flopscope
from flopscope._registry import REGISTRY

# Ops whose public signature takes a participant-supplied callable.
CALLBACK_OPS = {
    "apply_along_axis",
    "apply_over_axes",
    "fromfunction",
    "fromiter",
    "piecewise",
    "mask_indices",
}


def test_callback_ops_carry_the_local_callback_flag():
    missing = sorted(
        name
        for name in CALLBACK_OPS
        if name in REGISTRY and not REGISTRY[name].get("local_callback")
    )
    assert not missing, f"callback ops missing local_callback: {missing}"


def test_remote_unsupported_ops_reports_every_callback_op():
    assert CALLBACK_OPS <= set(flopscope.remote_unsupported_ops())


def test_client_registry_data_is_in_sync():
    """_registry_data.py is generated; regenerate with scripts/sync_client.py."""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parent.parent
        / "flopscope-client"
        / "src"
        / "flopscope"
        / "_registry_data.py"
    )
    spec = importlib.util.spec_from_file_location("_registry_data", path)
    # spec_from_file_location is typed as returning ModuleSpec | None, but for
    # a file path that exists on disk it always resolves to a real spec with a
    # loader; assert the runtime invariant instead of relaxing the null checks.
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.LOCAL_CALLBACK_OPS) == set(flopscope.remote_unsupported_ops())
