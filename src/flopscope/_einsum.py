"""Einsum with analytical FLOP counting, symmetry detection, and path optimization."""

from __future__ import annotations

import functools
from typing import Any, cast

import numpy as _np

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._config import get_setting
from flopscope._ndarray import FlopscopeArray, _to_base_ndarray
from flopscope._perm_group import SymmetryGroup
from flopscope._pointwise import _prepare_symmetric_out, _validate_result_symmetry
from flopscope._symmetric import SymmetricTensor
from flopscope._symmetry_utils import normalize_symmetry_input, validate_symmetry_group
from flopscope._validation import maybe_check_nan_inf, require_budget


def _identity_pattern(operands):
    """Build a hashable pattern of which operands are the same Python object.

    Returns None if all operands are distinct objects (common case).
    Otherwise returns a tuple of tuples, where each inner tuple lists
    positions sharing the same object identity (only groups of size >= 2).

    This mirrors the identical_operand_groups logic in _build_bipartite.
    """
    id_to_positions: dict[int, list[int]] = {}
    for idx, op in enumerate(operands):
        id_to_positions.setdefault(id(op), []).append(idx)
    groups = tuple(
        tuple(positions)
        for positions in id_to_positions.values()
        if len(positions) >= 2
    )
    return groups if groups else None


def _make_path_cache(maxsize):
    """Create a new lru_cache-wrapped path computation function.

    The cache key includes subscripts, shapes, optimizer, per_op_symmetries,
    and identity_pattern. Re-runs with the same inputs return the cached path.

    The key also includes ``per_op_symmetries`` (a tuple of per-operand
    SymmetryGroup-or-None, canonicalized as a hashable fingerprint) and
    ``identity_pattern`` so that symmetric operands produce a distinct cache
    entry from dense operands with the same subscripts and shapes.  When
    symmetry-aware path search is enabled the chosen path may differ; without
    this the dense-optimal path would silently be reused for symmetric inputs.
    """

    @functools.lru_cache(maxsize=maxsize)
    def _compute(
        subscripts,
        shapes,
        optimize,
        per_op_symmetries,
        identity_pattern,
    ):
        from flopscope._opt_einsum import contract_path as _contract_path

        _path, path_info = _contract_path(
            subscripts,
            *shapes,
            shapes=True,
            optimize=optimize if not isinstance(optimize, tuple) else list(optimize),
        )
        return path_info

    return _compute


_path_cache = _make_path_cache(4096)


def _rebuild_einsum_cache():
    """Rebuild the path cache with the current configured maxsize."""
    global _path_cache
    _path_cache = _make_path_cache(int(get_setting("einsum_path_cache_size")))  # type: ignore[arg-type]


def clear_einsum_cache():
    """Clear the einsum path cache.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Discards all cached contraction paths.

    Notes
    -----
    Discards all cached contraction paths. Subsequent ``einsum()`` and
    ``einsum_path()`` calls will recompute paths from scratch.

    Examples
    --------
    >>> import flopscope.numpy as fnp
    >>> fnp.clear_einsum_cache()
    """
    _path_cache.cache_clear()


def einsum_cache_info():
    """Return einsum path cache statistics.

    Parameters
    ----------
    None

    Returns
    -------
    object
        The standard ``functools.lru_cache`` statistics tuple with ``hits``,
        ``misses``, ``maxsize``, and ``currsize`` fields.

    Examples
    --------
    >>> import flopscope.numpy as fnp
    >>> info = fnp.einsum_cache_info()
    >>> total = info.hits + info.misses
    >>> rate = info.hits / max(total, 1)
    """
    return _path_cache.cache_info()


def _execute_pairwise(path_info, operands: list):
    """Execute pairwise contractions according to the optimized path."""
    ops = list(operands)
    for contract_inds, step in zip(path_info.path, path_info.steps, strict=False):
        # Pop operands in reverse sorted order (same as opt_einsum convention)
        inds = sorted(contract_inds, reverse=True)
        tensors = [ops.pop(i) for i in inds]
        result = _call_numpy(
            _np.einsum, step.subscript, *[_to_base_ndarray(t) for t in tensors]
        )
        ops.append(result)
    return ops[0]


_LARGE_K_THRESHOLD = 8


def _resolve_optimize_for_k(optimize, k: int):
    """Auto-downgrade 'auto' to 'greedy' for k >= 8 to avoid optimal/B&B
    cold-call latency on large operand counts. Explicit user choices
    (optimal/branch/dp/etc.) are honored verbatim. See spec §10.
    """
    if optimize == "auto" and k >= _LARGE_K_THRESHOLD:
        return "greedy"
    return optimize


def _normalize_optimize(optimize):
    if optimize is False:
        return "auto"
    if isinstance(optimize, list):
        return tuple(tuple(t) for t in optimize)
    return optimize


def _parse_einsum_parts(subscripts: str, operands):
    from flopscope._opt_einsum import parse_einsum_input

    input_subscripts, output_subscript, _ = parse_einsum_input((subscripts, *operands))
    canonical_subscripts = f"{input_subscripts}->{output_subscript}"
    return canonical_subscripts, input_subscripts.split(","), output_subscript


def _get_path_info(
    subscripts: str,
    operands,
    optimize,
    *,
    per_op_symmetries=None,
    identity_pattern=None,
):
    canonical_subscripts, input_parts, output_subscript = _parse_einsum_parts(
        subscripts,
        operands,
    )
    shapes = tuple(tuple(op.shape) for op in operands)

    # Build a hashable symmetry key for the cache.  Each entry is either None
    # (dense operand) or the canonical fingerprint of a SymmetryGroup so that
    # symmetric and dense operands with identical subscripts/shapes get distinct
    # cache slots.  This prevents a dense-optimal path from being silently
    # reused when symmetry-aware path search is later enabled.
    if per_op_symmetries is None:
        from flopscope._accumulation._public import _per_op_symmetries as _extract_syms

        per_op_symmetries = _extract_syms(operands)
    syms_key = tuple(per_op_symmetries)

    if identity_pattern is None:
        from flopscope._accumulation._public import _identity_pattern as _extract_id

        identity_pattern = _extract_id(operands)

    effective_optimize = _resolve_optimize_for_k(optimize, k=len(operands))
    path_info = _path_cache(
        canonical_subscripts,
        shapes,
        _normalize_optimize(effective_optimize),
        syms_key,
        identity_pattern,
    )

    # Bug B fix: if any operand has declared symmetry OR multiple operand
    # positions alias to the same array (identity_pattern), rebuild path_info
    # through the SubgraphSymmetryOracle so that per-step input_groups /
    # output_group / inner_group reflect the true residual symmetry of each
    # intermediate.  Without this rebuild, Source-A (declared groups),
    # Source-B (identical-operand swap), and Source-C (coordinated relabel)
    # π-generators never reach the renderer.
    #
    # _path_cache returns a shared cached object that must not be mutated;
    # build_path_info returns a fresh PathInfo each time.  The rebuild is
    # skipped when there's no symmetry signal at all (the common case) to
    # keep the fast path.
    _has_identity_alias = bool(identity_pattern) and any(
        len(group) > 1 for group in identity_pattern
    )
    if any(s is not None for s in per_op_symmetries) or _has_identity_alias:
        import numpy as _np_tmp
        import opt_einsum as _oe

        from flopscope._opt_einsum._contract import build_path_info as _bpi

        # Build dummy operands with the correct shapes, then alias positions
        # listed in the same identity-group to share object identity — this
        # is the signal the oracle uses to fire Source-B (identical-operand
        # swap) and Source-C (coordinated axis relabel) generators.
        _dummy_ops: list = [_np_tmp.empty(sh) for sh in shapes]
        if identity_pattern is not None:
            for group in identity_pattern:
                canonical = _dummy_ops[group[0]]
                for pos in group[1:]:
                    _dummy_ops[pos] = canonical

        _norm_optimize = _normalize_optimize(effective_optimize)
        if isinstance(_norm_optimize, tuple):
            _norm_optimize = list(_norm_optimize)
        _upstream_path, _upstream_info = _oe.contract_path(
            canonical_subscripts,
            *_dummy_ops,
            optimize=_norm_optimize,  # type: ignore[arg-type]
        )
        # Carry the optimizer label through the rebuild so the renderer's
        # "Optimizer:" pill stays populated.  effective_optimize is whatever
        # was actually used for path search; coerce to a string label.
        if isinstance(effective_optimize, str):
            _optimizer_label = effective_optimize
        else:
            _optimizer_label = getattr(_upstream_info, "_path_type", "") or ""
        path_info = _bpi(
            _upstream_path,
            _upstream_info,
            size_dict=_upstream_info.size_dict,
            optimizer_used=_optimizer_label,
            per_op_symmetries=per_op_symmetries,
            identity_pattern=identity_pattern,
        )

    return canonical_subscripts, input_parts, output_subscript, shapes, path_info


def _relabel_group_to_output(
    group, source_labels: tuple[str, ...], output_subscript: str
):
    if group is None or not source_labels or not output_subscript:
        return None
    output_positions = {label: idx for idx, label in enumerate(output_subscript)}
    try:
        source_positions = tuple(output_positions[label] for label in source_labels)
    except KeyError:
        return None
    if len(set(source_positions)) != len(source_positions):
        return None

    order = tuple(
        sorted(range(len(source_positions)), key=source_positions.__getitem__)
    )
    axes = tuple(source_positions[idx] for idx in order)
    source_to_sorted = {
        source_idx: sorted_idx for sorted_idx, source_idx in enumerate(order)
    }

    from flopscope._perm_group import _PermutationCompat as Permutation

    generators = []
    for gen in group.generators:
        generators.append(
            Permutation(
                [source_to_sorted[gen.array_form[source_idx]] for source_idx in order]
            )
        )

    remapped = SymmetryGroup(*generators, axes=axes)
    return validate_symmetry_group(remapped, ndim=len(output_subscript))


def _infer_pathless_output_symmetry(operands, input_parts, output_subscript: str):
    if len(operands) != 1:
        return None
    operand = operands[0]
    if not isinstance(operand, SymmetricTensor) or operand.symmetry is None:
        return None
    group = operand.symmetry
    axes = group.axes if group.axes is not None else tuple(range(group.degree))
    source_labels = tuple(input_parts[0][axis] for axis in axes)
    return _relabel_group_to_output(group, source_labels, output_subscript)


def _resolve_output_symmetry(
    *,
    symmetry,
    operands,
    input_parts,
    output_subscript: str,
):
    if symmetry is not None:
        return normalize_symmetry_input(symmetry, ndim=len(output_subscript))
    return _infer_pathless_output_symmetry(operands, input_parts, output_subscript)


@_counted_wrapper
def einsum(
    subscripts: str,
    *operands: _np.ndarray,
    out: Any = None,
    optimize: str | bool | list[Any] = "auto",
    symmetry: Any = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Evaluate Einstein summation with FLOP counting and optional path optimization.

    Wraps ``numpy.einsum`` with analytical FLOP cost computation and
    optional symmetry savings. If any input is a ``SymmetricTensor``,
    the cost is automatically reduced. If ``symmetry`` is provided and the output passes validation, a ``SymmetricTensor`` is returned.

    All contractions go through opt_einsum's ``contract_path`` to find an
    optimal pairwise decomposition. The charged FLOP cost comes from the
    path-independent symmetry-aware accumulation total
    (``path_info.accumulation.total``); per-step ``flop_count`` values on
    each ``StepInfo`` use flopscope's FMA=2 textbook convention throughout.

    Contraction paths are cached in a module-level LRU cache keyed on
    (subscripts, shapes, optimizer, per_op_symmetries, identity_pattern).
    Repeated calls with the same inputs skip path recomputation entirely.
    See ``clear_einsum_cache()`` and ``einsum_cache_info()``.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string (e.g., ``'ij,jk->ik'``).
    *operands : numpy.ndarray
        Input arrays. ``SymmetricTensor`` inputs are detected automatically
        for cost savings.
    optimize : str, bool, or list of tuple, optional
        Contraction path strategy. Default ``'auto'``.

        - ``'auto'``, ``'greedy'``, ``'optimal'``, ``'dp'``, etc.:
          Use the named algorithm to find the best path.
        - A list of int-tuples (e.g. ``[(1, 2), (0, 1)]``): use this
          explicit contraction path. Obtain one from ``fnp.einsum_path()``
          or construct manually. Each tuple names the operand positions
          to contract at that step; the result is appended to the end.
        - ``False``: treated as ``'auto'``.
    symmetry : SymmetryGroup or symmetry shorthand, optional
        Declares output symmetry and wraps the validated result as a
        ``SymmetricTensor``. This does NOT declare input symmetry; use
        ``flops.as_symmetric()`` for that.

    Returns
    -------
    numpy.ndarray or SymmetricTensor
        The result of the einsum.

    Raises
    ------
    BudgetExhaustedError
        If the operation would exceed the FLOP budget.
    NoBudgetContextError
        If called outside a ``BudgetContext``.
    SymmetryError
        If ``symmetry`` is provided but the result
        does not satisfy the declared symmetry. Validation checks the
        data against each generator of the group.
    """
    budget = require_budget()
    canonical_subscripts, input_parts, output_subscript, shapes, path_info = (
        _get_path_info(
            subscripts,
            operands,
            optimize,
        )
    )

    accumulation_cost = _get_accumulation_cost(
        canonical_subscripts=canonical_subscripts,
        input_parts=tuple(input_parts),
        output_subscript=output_subscript,
        shapes=shapes,
        operands=tuple(operands),
    )

    from flopscope._accumulation._path_info import FlopscopePathInfo

    path_info = FlopscopePathInfo.from_inner(
        inner=path_info,
        accumulation=accumulation_cost,
    )

    target_symmetry = _resolve_output_symmetry(
        symmetry=symmetry,
        operands=operands,
        input_parts=input_parts,
        output_subscript=output_subscript,
    )
    effective_out_symmetry = target_symmetry
    if effective_out_symmetry is None and isinstance(out, SymmetricTensor):
        effective_out_symmetry = out.symmetry
    target_symmetry = _prepare_symmetric_out(out, effective_out_symmetry)

    with budget.deduct(
        "einsum",
        flop_cost=accumulation_cost.total,
        subscripts=canonical_subscripts,
        shapes=tuple(shapes),
    ):
        if path_info.steps:
            result = _execute_pairwise(path_info, list(operands))
        else:
            result = _call_numpy(
                _np.einsum,
                canonical_subscripts,
                *[_to_base_ndarray(o) for o in operands],
            )

    if out is not None:
        _validate_result_symmetry(result, target_symmetry)
        _np.copyto(_np.asarray(out), _np.asarray(result), casting="unsafe")
        maybe_check_nan_inf(out, "einsum")
        return out

    if target_symmetry is not None:
        _validate_result_symmetry(result, target_symmetry)
        result = SymmetricTensor(_np.asarray(result), symmetry=target_symmetry)
    else:
        result = _asflopscope(_np.asarray(result))

    maybe_check_nan_inf(result, "einsum")
    return result  # type: ignore[return-value]


@_counted_wrapper
def einsum_path(
    subscripts: str,
    *operands: _np.ndarray,
    optimize: str | bool | list[Any] = "auto",
) -> tuple[list[Any], Any]:
    """Compute the optimal contraction path without executing.

    Returns ``(path, PathInfo)`` with zero budget cost. The returned
    ``path`` can be passed back to ``fnp.einsum(..., optimize=path)``
    to execute with that exact contraction order.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string.
    *operands : numpy.ndarray
        Input arrays.
    optimize : str, bool, or list of tuple, optional
        Path optimization strategy. Default ``'auto'``.

    Returns
    -------
    path : list of tuple of int
        The contraction path. Pass to ``fnp.einsum(..., optimize=path)``.
    info : PathInfo
        Diagnostics including per-step costs and symmetry savings.
    """
    budget = require_budget()
    with budget.deduct("einsum_path", flop_cost=1, subscripts=None, shapes=()):
        pass
    canonical_subscripts, input_parts, output_subscript, shapes, path_info = (
        _get_path_info(
            subscripts,
            operands,
            optimize,
        )
    )

    accumulation_cost = _get_accumulation_cost(
        canonical_subscripts=canonical_subscripts,
        input_parts=tuple(input_parts),
        output_subscript=output_subscript,
        shapes=shapes,
        operands=tuple(operands),
    )

    from flopscope._accumulation._path_info import FlopscopePathInfo

    path_info = FlopscopePathInfo.from_inner(
        inner=path_info,
        accumulation=accumulation_cost,
    )

    return list(path_info.path), path_info


# ── Accumulation cost helper + cache ─────────────────────────────────


from flopscope._accumulation._cache import (  # noqa: E402, F401
    _accumulation_cache,
    get_accumulation_cost_cached,
)
from flopscope._accumulation._cache import (  # noqa: E402
    rebuild_accumulation_cache as _rebuild_accumulation_cache_fn,
)
from flopscope._accumulation._public import (  # noqa: E402
    _accumulation_fingerprint,
    _identity_pattern,
)


def _get_accumulation_cost(
    *,
    canonical_subscripts: str,
    input_parts: tuple,
    output_subscript: str,
    shapes: tuple,
    operands: tuple,
):
    """Cached accumulation-cost lookup for einsum() / einsum_path()."""
    # Resolve partition_budget to the active setting BEFORE cache lookup so
    # the cache key reflects the budget used; otherwise stale entries from a
    # prior setting value can leak across calls.
    partition_budget = cast(int, get_setting("partition_budget"))
    return get_accumulation_cost_cached(
        canonical_subscripts=canonical_subscripts,
        input_parts=tuple(input_parts),
        output_subscript=output_subscript,
        shapes=shapes,
        sym_fingerprint=_accumulation_fingerprint(operands),
        identity_pattern=_identity_pattern(operands),
        partition_budget=partition_budget,
    )


def _rebuild_accumulation_cache():
    """Rebuild the accumulation cache with the current configured maxsize."""
    _rebuild_accumulation_cache_fn(cast(int, get_setting("einsum_path_cache_size")))


import sys as _sys  # noqa: E402

from flopscope._ndarray import _asflopscope  # noqa: E402
from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__], skip_names={"einsum", "einsum_path"})
