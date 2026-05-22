"""Public einsum_accumulation_cost — pure inspection function."""

from __future__ import annotations

from typing import cast

import numpy as _np

from flopscope._config import get_setting
from flopscope._opt_einsum import parse_einsum_input
from flopscope._symmetric import SymmetricTensor

from ._cache import get_accumulation_cost_cached
from ._cost import AccumulationCost


def _per_op_symmetries(operands):
    """Extract per-operand SymmetryGroup from SymmetricTensor inputs."""
    out = []
    for op in operands:
        if isinstance(op, SymmetricTensor) and op.symmetry is not None:
            out.append(op.symmetry)
        else:
            out.append(None)
    return tuple(out)


def _identity_pattern(operands):
    """Group operand positions sharing object id (for repeated-operand wreath)."""
    id_to_positions: dict[int, list[int]] = {}
    for idx, op in enumerate(operands):
        id_to_positions.setdefault(id(op), []).append(idx)
    groups = tuple(
        tuple(positions)
        for positions in id_to_positions.values()
        if len(positions) >= 2
    )
    return groups if groups else None


def _accumulation_fingerprint(operands):
    """Hashable per-operand symmetry fingerprint for the accumulation cache key."""
    parts = []
    for sym in _per_op_symmetries(operands):
        if sym is None:
            parts.append(None)
            continue
        axes = sym.axes
        gens = tuple(tuple(gen.array_form) for gen in sym.generators)
        parts.append((axes, gens))
    return tuple(parts)


def einsum_accumulation_cost(
    subscripts: str,
    *operands,
    partition_budget: int | None = None,
) -> AccumulationCost:
    """Compute the symmetry-aware direct-event cost of an einsum expression.

    Returns the path-independent total = (k-1)·∏ M_a + ∏ α_a, decomposed
    per component, with the regime that fired and the full ladder trace.

    This function is for inspection and debugging — it does not execute the
    einsum. To execute, use ``flopscope.einsum()``, which charges this same
    cost against the active BudgetContext.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string (e.g. ``'ij,jk->ik'``).
    *operands : numpy.ndarray or SymmetricTensor
        Input arrays. SymmetricTensor inputs contribute their declared symmetry
        to the detected pointwise group G_pt.
    partition_budget : int, optional
        Maximum number of typed partitions per component before the
        partition-count regime refuses. Defaults to
        ``flopscope.get_setting('partition_budget')`` (typically 100 000).

    Returns
    -------
    AccumulationCost
        Whole-einsum cost decomposition.
    """
    input_subscripts, output_subscript, _ = parse_einsum_input((subscripts, *operands))
    canonical_subscripts = f"{input_subscripts}->{output_subscript}"
    input_parts = tuple(input_subscripts.split(","))
    shapes = tuple(tuple(_np.asarray(op).shape) for op in operands)

    # Resolve partition_budget to the active setting BEFORE cache lookup. If we
    # pass None, the cache key collapses across all settings values and stale
    # entries computed under a different budget can leak across tests/configs.
    if partition_budget is None:
        partition_budget = cast(int, get_setting("partition_budget"))

    return get_accumulation_cost_cached(
        canonical_subscripts=canonical_subscripts,
        input_parts=input_parts,
        output_subscript=output_subscript,
        shapes=shapes,
        sym_fingerprint=_accumulation_fingerprint(operands),
        identity_pattern=_identity_pattern(operands),
        partition_budget=partition_budget,
    )


def _per_op_sym_fingerprint(sym) -> tuple:
    """Single-symmetry fingerprint for the reduction cache key."""
    axes = sym.axes if sym.axes is not None else tuple(range(sym.degree))
    gens = tuple(tuple(g.array_form) for g in sym.generators)
    return (axes, gens)


def reduction_accumulation_cost(
    a,
    axis=None,
    *,
    op_factor: int = 1,
    extra_ops: int = 0,
    partition_budget: int | None = None,
) -> AccumulationCost:
    """Public inspection function for reduction cost.

    Extracts symmetry from `a`, normalizes axis, routes to the cached
    compute_reduction_accumulation_cost.

    Parameters
    ----------
    a : numpy.ndarray or SymmetricTensor
        Input array. SymmetricTensor inputs contribute their declared symmetry.
    axis : int, tuple of int, or None
        Axes to reduce along. None reduces all axes.
    op_factor : int, default 1
        FLOPs per binary accumulation event.
    extra_ops : int, default 0
        Output-side dense additions (e.g. num_output_orbits for mean's divide).
    partition_budget : int, optional
        Override the partition counter budget. Defaults to global setting.

    Returns
    -------
    AccumulationCost
        Whole-reduction cost decomposition.
    """
    from ._cache import get_reduction_cost_cached
    from ._reduction import _normalize_axis

    arr = _np.asarray(a) if not isinstance(a, SymmetricTensor) else a
    sym = arr.symmetry if isinstance(arr, SymmetricTensor) else None
    shape = tuple(arr.shape)
    ndim = len(shape)
    axes_summed = _normalize_axis(axis, ndim)

    sym_fp = _per_op_sym_fingerprint(sym) if sym is not None else (None,)

    # Resolve partition_budget to the active setting BEFORE cache lookup. If we
    # pass None, the cache key collapses across all settings values and stale
    # entries computed under a different budget can leak across tests/configs.
    if partition_budget is None:
        partition_budget = cast(int, get_setting("partition_budget"))

    return get_reduction_cost_cached(
        input_shape=shape,
        axes_summed=axes_summed,
        sym_fingerprint=sym_fp,
        op_factor=op_factor,
        extra_ops=extra_ops,
        partition_budget=partition_budget,
    )
