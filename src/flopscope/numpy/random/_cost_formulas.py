"""Named cost-formula vocabulary for fnp.random method-level entries.

Each formula resolves to a callable ``(args, kwargs, result) -> int`` that
computes the FLOP cost from the call arguments and the numpy result.
The registry's ``cost_formula`` field names which formula a method uses.
"""

from __future__ import annotations

import builtins as _builtins
from collections.abc import Callable
from typing import Any

import numpy as _np

from flopscope._flops import _ceil_log2 as _ceil_log2
from flopscope._flops import sort_cost as _sort_cost
from flopscope._flops import svd_cost as _svd_cost


def _numel_output(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> int:
    if isinstance(result, _np.ndarray):
        return _builtins.max(int(result.size), 1)
    return 1


def _numel_input(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> int:
    a = args[0] if args else kwargs.get("x")
    if a is None:
        return 1
    if isinstance(a, _np.ndarray):
        return _builtins.max(int(a.size), 1)
    if hasattr(a, "__len__"):
        return _builtins.max(len(a), 1)
    return 1


def _length(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> int:
    if args:
        n = int(args[0])
    elif "length" in kwargs:
        n = int(kwargs["length"])
    else:
        n = 1
    return _builtins.max(n, 1)


def _sort_cost_formula(
    args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
) -> int:
    a = args[0] if args else kwargs.get("a")
    if a is None:
        return _sort_cost(1)
    if isinstance(a, (int, _np.integer)):
        n = int(a)
    elif isinstance(a, _np.ndarray):
        n = int(a.shape[0]) if a.ndim > 0 else int(a)
    elif hasattr(a, "__len__"):
        n = len(a)
    else:
        n = 1
    return _sort_cost(n)


def _shape_axis(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> int:
    """Cost = shape along the axis being permuted (defaults to axis=0).

    Used by shuffle/permutation: the algorithm is O(shape[axis]) RNG draws
    regardless of how wide each slice is. For integer input (the
    ``permutation(int_n)`` case), cost = ``int(n)``. For ``axis=None`` —
    which numpy interprets as "flatten then operate" — cost = numel.
    """
    a = args[0] if args else kwargs.get("x")
    if a is None:
        return 1
    if isinstance(a, (int, _np.integer)):
        return _builtins.max(int(a), 1)

    axis = args[1] if len(args) >= 2 else kwargs.get("axis", 0)
    if axis is None:
        if isinstance(a, _np.ndarray):
            return _builtins.max(int(a.size), 1)
        if hasattr(a, "__len__"):
            return _builtins.max(len(a), 1)
        return 1

    if isinstance(a, _np.ndarray):
        if a.ndim == 0:
            # 0-D scalar array; numpy choice/permutation treats as int(a)
            return _builtins.max(int(a), 1)
        return _builtins.max(int(a.shape[int(axis)]), 1)
    if hasattr(a, "__len__"):
        return _builtins.max(len(a), 1)
    return 1


def _choice_pool_size(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Pool length n for choice: the extent of the dimension being sampled.

    ``Generator.choice(a, size=None, replace=True, p=None, axis=0,
    shuffle=True)`` samples along ``axis`` (5th positional), so n must be
    the pool's extent along that axis, not ``shape[0]``. The legacy
    signatures sharing this formula (``RandomState.choice`` and module-level
    ``random.choice``) have no ``axis`` argument and require 1-D pools, so
    the ``axis=0`` default keeps their billing exactly as before.

    Mirrors ``_shape_axis``: ``int`` pool -> ``int(a)``; 0-D ndarray ->
    ``int(a)``; ndarray -> ``shape[axis]``; ``__len__`` fallback for
    non-ndarray sequences (axis-0 semantics -- numpy converts them anyway);
    ``axis=None`` (not valid for numpy choice, handled defensively) -> numel.
    Floors at 1.
    """
    a = args[0] if args else kwargs.get("a")
    if a is None:
        return 1
    if isinstance(a, (int, _np.integer)):
        return _builtins.max(int(a), 1)

    axis = args[4] if len(args) >= 5 else kwargs.get("axis", 0)
    if axis is None:
        if isinstance(a, _np.ndarray):
            return _builtins.max(int(a.size), 1)
        if hasattr(a, "__len__"):
            return _builtins.max(len(a), 1)
        return 1

    if isinstance(a, _np.ndarray):
        if a.ndim == 0:
            # 0-D scalar array; numpy choice treats as int(a)
            return _builtins.max(int(a), 1)
        return _builtins.max(int(a.shape[int(axis)]), 1)
    if hasattr(a, "__len__"):
        return _builtins.max(len(a), 1)
    return 1


def _choice_cost(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> int:
    # Generator.choice:    choice(a, size=None, replace=True, p=None, axis=0, shuffle=True)
    # RandomState.choice:  choice(a, size=None, replace=True, p=None)
    # `replace` is the 3rd positional or the `replace` kwarg.
    # `p`       is the 4th positional or the `p` kwarg.
    if len(args) >= 3:
        replace = bool(args[2])
    else:
        replace = bool(kwargs.get("replace", True))
    if len(args) >= 4:
        p = args[3]
    else:
        p = kwargs.get("p", None)
    if replace:
        base = _numel_output(args, kwargs, result)
        if p is not None:
            # numpy builds a CDF over the n-element pool (n = extent along the
            # sampled axis): cumsum + normalise + final pass (3*n) then
            # binary-searches each draw (size * ceil(log2(n))).
            n = _choice_pool_size(args, kwargs)
            draws = _builtins.max(base, 1)
            base += 3 * n + draws * _ceil_log2(n)
        return base
    # replace=False: pool size n along the sampled axis
    n = _choice_pool_size(args, kwargs)
    if p is None:
        # Fisher-Yates O(n): legacy RandomState.choice is permutation(pop)[:size];
        # Generator uses Floyd's/tail-shuffle (<= O(n)); n is a conservative ceiling.
        return n
    # Data-dependent rejection loop with weights: sort_cost(n) conservative floor.
    return _sort_cost(n)


def multivariate_normal_flops(N: int, d: int) -> int:
    """Composite mvn cost: covariance factorization (SVD of the d×d covariance,
    matching numpy's default method='svd' which calls np.linalg.svd(cov) with
    full U/V) + affine transform (2*N*d^2) + N*d standard-normal draws at the
    transcendental rate (16/draw). Tier folded into flop_cost; weight 1.0.

    Factorization: svd_cost(d, d, with_vectors=True) = 6*d*d^2 + 20*d^3 = 26*d^3
    (thin SVD of a square d×d matrix; LAPACK dgesdd path, G&VL 4e §8.6).
    numpy.random.multivariate_normal (Generator default method='svd',
    RandomState always SVD, module-level np.random.multivariate_normal) calls
    np.linalg.svd(cov) on the symmetric d×d covariance matrix.
    """
    return _builtins.max(
        _svd_cost(d, d, with_vectors=True) + 2 * N * d * d + 16 * N * d, 1
    )


def _multivariate_normal_cost(
    args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
) -> int:
    # result has shape (..., d); d from the trailing axis, N = leading numel.
    shape = getattr(result, "shape", ())
    d = int(shape[-1]) if shape else 1
    n = int(result.size // d) if d else 1
    return multivariate_normal_flops(n, d)


def _uniform_cost(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> int:
    """uniform draws + affine map ``low + (high-low)*U`` (mul+add) = 3*numel."""
    return 3 * _numel_output(args, kwargs, result)


COST_FORMULAS: dict[str, Callable[[tuple[Any, ...], dict[str, Any], Any], int]] = {
    "numel(output)": _numel_output,
    "uniform": _uniform_cost,
    "numel(input)": _numel_input,
    "shape[axis]": _shape_axis,
    "length": _length,
    "sort_cost(n)": _sort_cost_formula,
    "choice_cost": _choice_cost,
    "multivariate_normal": _multivariate_normal_cost,
}
