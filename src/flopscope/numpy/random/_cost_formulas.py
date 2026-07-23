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
from flopscope._ndarray import _to_base_ndarray


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
        # Non-ndarray array-likes (e.g. a raw Python list) reach here
        # uncoerced: every cost formula is invoked with the wrapper's
        # original args tuple, never the ndarray-stripped call_args that
        # _make_counted_method's movement-method branch builds for the real
        # numpy call (that stripping exists so a FlopscopeArray operand
        # doesn't re-enter numpy's C dispatch; it never touches what a
        # formula sees). So nothing coerces a list before it gets here.
        # len() on a nested list counts only the outer dimension;
        # asarray(a).size counts every element numpy actually shuffles,
        # matching the true work for any nesting depth.
        return _builtins.max(int(_np.asarray(a).size), 1)
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
    Non-ndarray array-likes (e.g. a raw Python list) are routed through
    ``asarray()`` first so they are billed by the same shape/axis logic as
    an equivalent ndarray, rather than by axis-blind ``len(a)``.
    """
    a = args[0] if args else kwargs.get("x")
    if a is None:
        return 1
    if isinstance(a, (int, _np.integer)):
        return _builtins.max(int(a), 1)

    if not isinstance(a, _np.ndarray):
        if not hasattr(a, "__len__"):
            return 1
        # Cost formulas always see the wrapper's original, uncoerced args
        # (see _numel_input's comment for the full explanation) -- so a raw
        # Python list reaches here exactly as the caller wrote it. Route it
        # through asarray() so it gets real shape/axis semantics instead of
        # len(a), which only ever reports the outer dimension no matter
        # what axis was requested.
        a = _np.asarray(a)

    axis = args[1] if len(args) >= 2 else kwargs.get("axis", 0)
    if axis is None:
        return _builtins.max(int(a.size), 1)

    if a.ndim == 0:
        # 0-D scalar array; numpy choice/permutation treats as int(a)
        return _builtins.max(int(a), 1)
    return _builtins.max(int(a.shape[int(axis)]), 1)


def _choice_pool_size(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Pool length n for choice: the extent of the dimension being sampled.

    ``Generator.choice(a, size=None, replace=True, p=None, axis=0,
    shuffle=True)`` samples along ``axis`` (5th positional), so n must be
    the pool's extent along that axis, not ``shape[0]``. The legacy
    signatures sharing this formula (``RandomState.choice`` and module-level
    ``random.choice``) have no ``axis`` argument and require 1-D pools, so
    the ``axis=0`` default keeps their billing exactly as before.

    Mirrors ``_shape_axis``: ``int`` pool -> ``int(a)``; 0-D ndarray ->
    ``int(a)``; ndarray -> ``shape[axis]``; ``axis=None`` (not valid for
    numpy choice, handled defensively) -> numel. Non-ndarray array-likes
    (e.g. a raw Python list) are routed through ``asarray()`` first so they
    are billed by the same shape/axis logic as an equivalent ndarray, rather
    than by axis-blind ``len(a)`` (which only ever reports the outer
    dimension no matter what axis was requested -- the same bug ``_shape_axis``
    fixes for shuffle/permutation). Floors at 1.
    """
    a = args[0] if args else kwargs.get("a")
    if a is None:
        return 1
    if isinstance(a, (int, _np.integer)):
        return _builtins.max(int(a), 1)

    if not isinstance(a, _np.ndarray):
        if not hasattr(a, "__len__"):
            return 1
        a = _np.asarray(a)

    axis = args[4] if len(args) >= 5 else kwargs.get("axis", 0)
    if axis is None:
        return _builtins.max(int(a.size), 1)

    if a.ndim == 0:
        # 0-D scalar array; numpy choice treats as int(a)
        return _builtins.max(int(a), 1)
    return _builtins.max(int(a.shape[int(axis)]), 1)


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
        # Partial shuffle O(n): legacy RandomState.choice is permutation(pop)[:size];
        # Generator uses Floyd's/tail-shuffle (<= O(n)); n is a conservative ceiling.
        return n
    # Data-dependent rejection loop with weights: sort_cost(n) conservative floor.
    return _sort_cost(n)


def _mvhg_cost(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> int:
    """multivariate_hypergeometric(colors, nsample, size=None, method='marginals').

    Base cost is numel(output). ``method="count"`` builds a temporary array
    of integers with length ``sum(colors)`` (numpy's own docstring for the
    method), then for each of the ``num_variates`` output vectors does a
    partial shuffle over the first ``min(nsample, sum(colors) -
    nsample)`` entries of that buffer, followed by a separate pass counting
    the colors of those same shuffled entries -- numpy's C implementation
    (``random_multivariate_hypergeometric_count``) samples whichever of "the
    nsample chosen" / "the sum(colors)-nsample excluded" is smaller and
    inverts the count if it took the exclusion path, so the shuffle length is
    never more than half of ``sum(colors)``. Two full passes (shuffle, then
    count) over that length is the real per-variate cost, so
    ``method="count"`` bills ``sum(colors) + 2*num_variates*min(nsample,
    sum(colors)-nsample) + numel(output)``. That draw term is a safe ceiling
    -- it now scales correctly in both ``nsample`` and ``size``, the two
    dimensions the old flat ``numel(output)`` bill was completely blind to,
    with a 2x coefficient covering both passes over the shuffled entries
    instead of just the first. ``method="marginals"`` (the default) never
    allocates that buffer or shuffles, and stays at numel(output).

    ``method`` accepts the call's positional-or-keyword form, matching
    ``_choice_cost``'s handling of ``replace``/``p``/``axis``: it is the 4th
    positional argument if given positionally, else the ``method`` kwarg.
    ``nsample`` is the 2nd positional argument or the ``nsample`` kwarg --
    it is a required parameter, so by the time this formula runs numpy's own
    call has already succeeded and nsample is never actually missing.
    ``num_variates`` is read off ``result.shape`` (trailing axis =
    num_colors, the same convention ``_multivariate_normal_cost`` uses)
    instead of re-parsing the ``size`` argument, so it stays correct whether
    the caller passed ``size`` as ``None``, an int, or a tuple.
    """
    colors = args[0] if args else kwargs.get("colors")
    method = args[3] if len(args) >= 4 else kwargs.get("method", "marginals")
    out = _builtins.max(int(getattr(result, "size", 1)), 1)
    if method == "count" and colors is not None:
        # Strip to a base ndarray first: colors may be a caller-supplied
        # FlopscopeArray, and summing it directly would re-enter NumPy's
        # __array_function__ dispatch from inside this wrapper's call frame
        # (the _called_from_wrapper() tripwire raises on exactly that).
        colors = _to_base_ndarray(colors)
        total = int(_np.sum(colors))
        nsample = int(args[1]) if len(args) >= 2 else int(kwargs.get("nsample", 0))
        shape = getattr(result, "shape", ())
        num_colors = int(shape[-1]) if shape else 1
        num_variates = out // num_colors if num_colors else 1
        draws = (
            2 * num_variates * _builtins.min(nsample, _builtins.max(total - nsample, 0))
        )
        return _builtins.max(total + draws + out, out)
    return out


def multivariate_normal_flops(N: int, d: int) -> int:
    """Composite mvn cost: covariance factorization (SVD of the d×d covariance,
    matching numpy's default method='svd' which calls np.linalg.svd(cov) with
    full U/V) + affine transform (2*N*d^2) + N*d standard-normal draws at the
    transcendental rate (16/draw). Tier folded into flop_cost; weight 1.0.

    Factorization: svd_cost(d, d, with_vectors=True) = 6*d*d^2 + 20*d^3 = 26*d^3
    (thin SVD of a square d×d matrix; LAPACK dgesdd path).
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
    "mvhg_cost": _mvhg_cost,
    "multivariate_normal": _multivariate_normal_cost,
}
