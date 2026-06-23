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


class _CoercingConstructor:
    """Materialize a leading RemoteArray, then delegate to a numpy constructor.

    Implemented as a CLASS INSTANCE, not a function, on purpose: numpy's tests
    store constructors as class attributes (e.g. ``self.array = np.array`` then
    ``self.array([[1, 3]], dtype=...)``). A plain function is a descriptor, so
    Python would auto-bind ``self`` as the first positional argument — numpy then
    sees ``array(self, data, dtype=...)`` and raises "dtype given by name and
    position". Instances have no ``__get__``, so they are returned unbound and
    receive only the caller's arguments. (Same rationale as the native harness's
    ``_NonDescriptor``.)
    """

    def __init__(self, orig):
        self._orig = orig
        self.__name__ = getattr(orig, "__name__", "")
        self.__qualname__ = getattr(orig, "__qualname__", self.__name__)
        self.__doc__ = getattr(orig, "__doc__", None)

    def __call__(self, obj, *args, **kwargs):
        if _is_remote(obj):
            obj = _materialize(obj)
        return self._orig(obj, *args, **kwargs)


def install() -> None:
    for name, orig in _ORIG.items():
        setattr(np, name, _CoercingConstructor(orig))


def uninstall() -> None:
    for name, orig in _ORIG.items():
        setattr(np, name, orig)
