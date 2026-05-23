# src/flopscope/linalg/_aliases.py
"""Linalg namespace aliases that delegate to top-level flopscope operations.

These functions exist in numpy.linalg as convenience aliases.
FLOP costs are handled by the delegated top-level implementations.
"""

from __future__ import annotations

from typing import Any

import numpy as _np
from numpy.typing import ArrayLike

import flopscope.numpy as _me
from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._docstrings import attach_docstring
from flopscope._ndarray import FlopscopeArray


def matmul(x1: ArrayLike, x2: ArrayLike, /) -> FlopscopeArray:
    """Matrix multiply (linalg namespace). Delegates to flopscope.matmul."""
    return _me.matmul(x1, x2)  # type: ignore[reportReturnType]


attach_docstring(
    matmul, _np.linalg.matmul, "linalg", "0 FLOPs (delegates to flopscope.matmul)"
)


@_counted_wrapper
def cross(x1: ArrayLike, x2: ArrayLike, /, *, axis: int = -1) -> FlopscopeArray:
    """Cross product (linalg namespace). Uses np.linalg.cross for strict validation."""
    import builtins as _builtins

    from flopscope._ndarray import FlopscopeArray, _asflopscope, _to_base_ndarray
    from flopscope._validation import require_budget

    budget = require_budget()
    inputs_were_whest = isinstance(x1, FlopscopeArray) or isinstance(x2, FlopscopeArray)
    x1_arr = _np.asarray(x1)
    x2_arr = _np.asarray(x2)
    out_shape = _np.broadcast_shapes(x1_arr.shape, x2_arr.shape)
    out_size = 1
    for d in out_shape:
        out_size *= d
    with budget.deduct(
        "linalg.cross",
        flop_cost=_builtins.max(out_size * 5, 1),
        subscripts=None,
        shapes=(x1_arr.shape, x2_arr.shape),
    ):
        result = _call_numpy(
            _np.linalg.cross, _to_base_ndarray(x1), _to_base_ndarray(x2), axis=axis
        )  # type: ignore[reportCallIssue]
    if isinstance(result, _np.ndarray) and inputs_were_whest:
        return _asflopscope(result)  # type: ignore[reportReturnType]
    return result  # type: ignore[reportReturnType]


attach_docstring(
    cross, _np.linalg.cross, "linalg", "0 FLOPs (delegates to flopscope.cross)"
)


def outer(x1: ArrayLike, x2: ArrayLike, /) -> FlopscopeArray:
    """Outer product (linalg namespace). Delegates to flopscope.outer."""
    return _me.outer(x1, x2)  # type: ignore[reportReturnType]


attach_docstring(
    outer, _np.linalg.outer, "linalg", "0 FLOPs (delegates to flopscope.outer)"
)


def tensordot(x1: ArrayLike, x2: ArrayLike, /, *, axes: Any = 2) -> FlopscopeArray:
    """Tensor dot product (linalg namespace). Delegates to flopscope.tensordot."""
    return _me.tensordot(x1, x2, axes=axes)  # type: ignore[reportReturnType]


attach_docstring(
    tensordot,
    _np.linalg.tensordot,
    "linalg",
    "0 FLOPs (delegates to flopscope.tensordot)",
)


if hasattr(_np.linalg, "vecdot"):

    def vecdot(  # type: ignore[reportRedeclaration]
        x1: ArrayLike, x2: ArrayLike, /, *, axis: int = -1
    ) -> FlopscopeArray:
        """Vector dot product (linalg namespace). Delegates to flopscope.vecdot."""
        return _me.vecdot(x1, x2, axis=axis)  # type: ignore[reportReturnType]

    attach_docstring(
        vecdot,
        _np.linalg.vecdot,
        "linalg",
        "0 FLOPs (delegates to flopscope.vecdot)",
    )

else:
    from flopscope.errors import UnsupportedFunctionError

    def vecdot(*args: Any, **kwargs: Any) -> FlopscopeArray:
        raise UnsupportedFunctionError("linalg.vecdot", min_version="2.1")


def diagonal(x: ArrayLike, /, *, offset: int = 0) -> FlopscopeArray:
    """Diagonal (linalg namespace). Delegates to flopscope.diagonal. Cost: 0 FLOPs."""
    return _me.diagonal(x, offset=offset, axis1=-2, axis2=-1)  # type: ignore[reportReturnType]


attach_docstring(diagonal, _np.linalg.diagonal, "linalg", "0 FLOPs (free)")


def matrix_transpose(x: ArrayLike, /) -> FlopscopeArray:
    """Transpose (linalg namespace). Delegates to flopscope.matrix_transpose. Cost: 0 FLOPs."""
    return _me.matrix_transpose(x)  # type: ignore[reportReturnType]


attach_docstring(
    matrix_transpose, _np.linalg.matrix_transpose, "linalg", "0 FLOPs (free)"
)
