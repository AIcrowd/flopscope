"""Make NumPy's assert/array helpers understand client RemoteArray results.

We do NOT modify client code (Phase 1 is measurement only). Instead we wrap the
numpy entry points that asserts route through (``asarray``/``asanyarray``/
``array``) so a ``RemoteArray``/``RemoteScalar`` is materialized to a real
ndarray via ``.tolist()``. ``np.testing.assert_*`` call ``asanyarray``/
``asarray`` on their inputs, so wrapping those lets value/shape assertions work
against remote handles.

This is the OUTPUT side; the INPUT side (coercing ndarray args INTO the client)
lives in ``_patch_client.py``. ``install()`` runs AFTER ``patch()`` so these
wrappers own ``array``/``asarray``/``asanyarray`` (they do coercion, not client
routing). ``_ORIG`` is snapshotted at import — before ``patch()`` — so the
wrappers call the genuine numpy constructors and never recurse.
"""
from __future__ import annotations

import numpy as np

_ORIG = {name: getattr(np, name) for name in ("asarray", "asanyarray", "array")}


def _is_remote(x) -> bool:
    return type(x).__name__ in ("RemoteArray", "RemoteScalar")


def _materialize(x):
    # RemoteArray -> nested lists; RemoteScalar -> python scalar.
    return x.tolist() if hasattr(x, "tolist") else float(x)


def install() -> None:
    for name, orig in _ORIG.items():
        def make(orig=orig):
            def wrapper(obj, *args, **kwargs):
                if _is_remote(obj):
                    obj = _materialize(obj)
                return orig(obj, *args, **kwargs)
            return wrapper

        setattr(np, name, make())


def uninstall() -> None:
    for name, orig in _ORIG.items():
        setattr(np, name, orig)
