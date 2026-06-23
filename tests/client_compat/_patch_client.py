"""Patch ``numpy.<fn>`` -> CLIENT ``flopscope.numpy.<fn>`` for the parity harness.

Adapted from ``tests/numpy_compat/conftest.py::_patch_numpy``, but:

1. The source is the **client** ``flopscope.numpy`` (a ZMQ proxy), not native
   flopscope. There is NO ``_np`` freeze/rebind — the client has no internal
   numpy to recurse into.
2. Each patched op is wrapped with an **input coercer**. The client rejects real
   ``numpy.ndarray`` arguments (``RemoteSerializationError`` — "pass a
   materialized array"); native flopscope accepts them. NumPy's own tests build
   inputs via ``np.array(...)`` (kept native — ``array`` is in ``_SKIP``), so
   those inputs arrive as ``ndarray``. The wrapper converts ``ndarray`` ->
   ``list`` / numpy-scalar -> python-scalar so the client accepts them, exactly
   as a participant calling ``fnp.array(data)`` would. ``RemoteArray`` args pass
   through untouched (the client accepts those).

ufunc-typed names (e.g. ``np.add``) are skipped — their replacements are plain
functions lacking ``.reduce``/``.outer``/``.nargs`` which numpy probes at
collection time (same rationale as the native harness).
"""

from __future__ import annotations

import functools
import importlib
import sys

import numpy as np
from flopscope._registry_data import FUNCTION_CATEGORIES

# Originals we replaced, for unpatch().
_PATCHED: dict[str, object] = {}

# Names left for native numpy (descriptor/auto-bind issues at collection time)
# or owned by _coerce (array/asarray/asanyarray do output coercion, not routing).
_SKIP = {
    "linalg.outer",
    "array",
    "arange",
    "asarray",
    "random.randint",
    "random.shuffle",
}


def _to_client_arg(x):
    """Coerce a numpy value to something the client backend accepts.

    ndarray -> nested list; numpy scalar -> python scalar; list/tuple ->
    same container with elements coerced (handles ``concatenate([a, b])``).
    RemoteArray / RemoteScalar / python scalars pass through unchanged.
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):  # np.float64, np.int64, np.bool_, ...
        return x.item()
    if isinstance(x, (list, tuple)):
        return type(x)(_to_client_arg(e) for e in x)
    return x


def _input_coercing(client_fn, original_fn):
    """Wrap a client op for global numpy patching.

    Routes to the client ONLY when a BudgetContext is active (i.e. inside a
    test body, where the harness has a live server + open budget). Outside a
    test — during collection, module imports, numpy's own lazy init
    (e.g. ``numpy.random.mtrand``), or the xdist controller process — there is
    no active budget and no guaranteed server, so the wrapper behaves as the
    ORIGINAL numpy function. Without this guard, numpy/pytest internals that
    call a patched function at import time dispatch to a server that may not
    exist and the whole session dies with a handshake timeout.

    The wrapper introspects as the ORIGINAL numpy function
    (``wraps(original_fn)``, not ``client_fn``): numpy submodules read function
    metadata at import time — e.g. ``numpy.ma.extras`` does
    ``np.apply_over_axes.__doc__.find('Notes')`` — and the client proxies carry
    ``__doc__ is None``, which would crash that import. Copying numpy's
    ``__doc__``/``__name__``/``__module__`` keeps that introspection working
    while behavior still routes to the client.
    """

    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        import flopscope._budget as _b

        if _b._active_context is None:
            return original_fn(*args, **kwargs)
        return client_fn(
            *[_to_client_arg(a) for a in args],
            **{k: _to_client_arg(v) for k, v in kwargs.items()},
        )

    return wrapper


def _client_numpy():
    mod = sys.modules.get("flopscope.numpy")
    if mod is None:
        mod = importlib.import_module("flopscope.numpy")
    return mod


def patch() -> None:
    fnp = _client_numpy()
    for name, cat in FUNCTION_CATEGORIES.items():
        if cat == "blacklisted" or name in _SKIP:
            continue
        parts = name.split(".")
        if len(parts) > 2:
            continue

        # Resolve the client function (attribute access; flopscope.numpy.<sub>
        # is reachable via getattr even though it is not an importable package).
        try:
            if len(parts) == 1:
                we_fn = getattr(fnp, name)
            else:
                we_fn = getattr(getattr(fnp, parts[0]), parts[1])
        except AttributeError:
            continue

        # Skip ufuncs: replacements lack .reduce/.outer/.nargs probed at
        # collection time.
        try:
            if len(parts) == 1:
                np_obj = getattr(np, name, None)
            else:
                np_obj = getattr(getattr(np, parts[0], None), parts[1], None)
            if isinstance(np_obj, np.ufunc):
                continue
        except (AttributeError, TypeError):
            pass

        try:
            if len(parts) == 1:
                if hasattr(np, name):
                    original = getattr(np, name)
                    _PATCHED[name] = original
                    setattr(np, name, _input_coercing(we_fn, original))
            else:
                sub = getattr(np, parts[0])
                if hasattr(sub, parts[1]):
                    original = getattr(sub, parts[1])
                    _PATCHED[name] = original
                    setattr(sub, parts[1], _input_coercing(we_fn, original))
        except (AttributeError, TypeError):
            continue


def unpatch() -> None:
    for name, original in _PATCHED.items():
        parts = name.split(".")
        if len(parts) == 1:
            setattr(np, name, original)
        elif len(parts) == 2:
            setattr(getattr(np, parts[0]), parts[1], original)
    _PATCHED.clear()
