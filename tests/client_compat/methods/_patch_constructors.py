"""Patch numpy array CONSTRUCTORS -> client RemoteArray for the methods mode.

The function-mode harness keeps constructors native (so the function suite runs
on real ndarrays). This mode does the opposite: it makes np.array/zeros/ones/...
return client RemoteArrays, so numpy's own ndarray METHOD test classes exercise
RemoteArray's surface. asarray/asanyarray are instead made *coercing* (the assert
path goes through them), so numpy.testing.assert_* still works. Budget-guarded:
outside an active BudgetContext we behave as native numpy (collection/import/xdist
controller) so nothing dispatches to an absent server.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np

from .._coerce import _CoercingConstructor  # reuse the non-descriptor coercer

# Constructors numpy's method test classes use to BUILD arrays -> return RemoteArray.
_CONSTRUCTORS = (
    "array",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "empty",
    "empty_like",
    "arange",
    "full",
    "eye",
    "diag",
)
# Entry points numpy's assert helpers use to NORMALIZE -> coerce RemoteArray->ndarray.
_COERCERS = ("asarray", "asanyarray")

_PATCHED: dict[str, object] = {}


def _to_client_arg(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return type(x)(_to_client_arg(e) for e in x)
    return x


class _ConstructorToRemoteArray:
    """Route a numpy constructor to the client (returning a RemoteArray) when a
    budget is active, else native numpy. A CLASS INSTANCE (non-descriptor) so
    numpy tests doing ``self.array = np.array`` don't auto-bind ``self``."""

    def __init__(self, client_fn, original_fn):
        self._client_fn = client_fn
        self._orig = original_fn
        self.__name__ = getattr(original_fn, "__name__", "")
        self.__qualname__ = getattr(original_fn, "__qualname__", self.__name__)
        self.__doc__ = getattr(original_fn, "__doc__", None)

    def __call__(self, *args, **kwargs):
        import flopscope._budget as _b

        if _b._active_context is None:
            return self._orig(*args, **kwargs)
        return self._client_fn(
            *[_to_client_arg(a) for a in args],
            **{k: _to_client_arg(v) for k, v in kwargs.items()},
        )


def _client_numpy():
    mod = sys.modules.get("flopscope.numpy")
    if mod is None:
        mod = importlib.import_module("flopscope.numpy")
    return mod


def patch() -> None:
    """Install the construction-swap patch.  Idempotent: safe to call twice.

    Called once from pytest_configure (early, before parent conftest runs) and
    once from pytest_sessionstart (after all configure hooks, so we win over the
    parent conftest's _coerce.install()).  On the second call _PATCHED already
    holds the pre-patch originals, so we only re-setattr without re-snapshotting.
    """
    fnp = _client_numpy()
    for name in _CONSTRUCTORS:
        client_fn = getattr(fnp, name, None)
        if client_fn is None or not hasattr(np, name):
            continue
        current = getattr(np, name)
        if isinstance(current, _ConstructorToRemoteArray):
            continue  # already ours; nothing to do
        # Snapshot the pre-patch original only on the first install.
        if name not in _PATCHED:
            _PATCHED[name] = current
        original = _PATCHED[name]
        setattr(np, name, _ConstructorToRemoteArray(client_fn, original))
    for name in _COERCERS:
        if not hasattr(np, name):
            continue
        current = getattr(np, name)
        if isinstance(current, _CoercingConstructor):
            continue  # already wrapped
        if name not in _PATCHED:
            _PATCHED[name] = current
        setattr(np, name, _CoercingConstructor(_PATCHED[name]))


def unpatch() -> None:
    for name, original in _PATCHED.items():
        setattr(np, name, original)
    _PATCHED.clear()
