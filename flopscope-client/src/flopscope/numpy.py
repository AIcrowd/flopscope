"""flopscope.numpy — client-side numpy-compatible surface.

The lightweight client exposes the numpy-compatible API directly on the
top-level ``flopscope`` module; this submodule re-exports it under the
canonical ``flopscope.numpy`` path so the starter-kit idiom
``import flopscope.numpy as fnp`` works without the evaluator's build-time
shim. (The evaluator's Dockerfile writes byte-identical content, so the two
are equivalent.) The full in-process flopscope distribution ships its own
real ``flopscope/numpy/`` package; this is the client-only equivalent.
"""

from flopscope import *  # noqa: F401,F403,I001

import flopscope as _flopscope  # noqa: E402

__all__ = list(_flopscope.__all__)

from flopscope._getattr import make_module_getattr as _make_module_getattr  # noqa: E402,I001

__getattr__ = _make_module_getattr("", "flopscope.numpy")
