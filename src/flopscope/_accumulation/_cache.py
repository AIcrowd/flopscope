"""LRU cache for compute_accumulation_cost.

Lives in _accumulation rather than _einsum.py so that both the public inspection
function (_public.einsum_accumulation_cost) and the einsum path (_einsum._get_accumulation_cost)
can share it without circular imports.
"""

from __future__ import annotations

import functools
from typing import Any

from ._cost import AccumulationCost, compute_accumulation_cost


def _make_accumulation_cache(maxsize: int):
    @functools.lru_cache(maxsize=maxsize)
    def _compute(
        canonical_subscripts: str,
        input_parts: tuple,
        output_subscript: str,
        shapes: tuple,
        sym_fingerprint: tuple,
        identity_pattern: tuple | None,
        partition_budget: int | None,
        fma_cost: int,  # included in key so configure(fma_cost=N) invalidates
    ) -> AccumulationCost:
        # Reconstruct per-op symmetries from the fingerprint.
        from flopscope._perm_group import SymmetryGroup
        from flopscope._perm_group import _PermutationCompat as Permutation

        per_op_symmetries: list[Any] = []
        for fp_entry in sym_fingerprint:
            if fp_entry is None:
                per_op_symmetries.append(None)
                continue
            axes, gen_arrays = fp_entry
            gens = [Permutation(list(g)) for g in gen_arrays]
            group = SymmetryGroup(*gens, axes=axes) if gens else None
            per_op_symmetries.append(group)

        return compute_accumulation_cost(
            canonical_subscripts=canonical_subscripts,
            input_parts=input_parts,
            output_subscript=output_subscript,
            shapes=shapes,
            per_op_symmetries=tuple(per_op_symmetries),
            identity_pattern=identity_pattern,
            partition_budget=partition_budget,
        )

    return _compute


_accumulation_cache = _make_accumulation_cache(4096)


def get_accumulation_cost_cached(
    *,
    canonical_subscripts: str,
    input_parts: tuple,
    output_subscript: str,
    shapes: tuple,
    sym_fingerprint: tuple,
    identity_pattern: tuple | None,
    partition_budget: int | None,
) -> AccumulationCost:
    """Cached entry point. Routed through by both public and einsum-internal callers."""
    return _accumulation_cache(
        canonical_subscripts,
        tuple(input_parts),
        output_subscript,
        shapes,
        sym_fingerprint,
        identity_pattern,
        partition_budget,
        2,  # FMA=2 fixed convention; hard-coded so cache key is stable
    )


def rebuild_accumulation_cache(maxsize: int) -> None:
    """Rebuild the cache with a new maxsize."""
    global _accumulation_cache
    _accumulation_cache = _make_accumulation_cache(maxsize)


def _make_reduction_cache(maxsize: int):
    @functools.lru_cache(maxsize=maxsize)
    def _compute(
        input_shape: tuple,
        axes_summed: tuple,
        sym_fingerprint: tuple,
        op_factor: int,
        extra_ops: int,
        partition_budget: int | None,
    ) -> AccumulationCost:
        from flopscope._perm_group import SymmetryGroup
        from flopscope._perm_group import _PermutationCompat as Permutation

        # Reconstruct symmetry from fingerprint.
        if sym_fingerprint == (None,) or sym_fingerprint == ():
            symmetry = None
        else:
            axes, gen_arrays = sym_fingerprint
            gens = [Permutation(list(g)) for g in gen_arrays]
            symmetry = SymmetryGroup(*gens, axes=axes) if gens else None

        from ._reduction import compute_reduction_accumulation_cost

        return compute_reduction_accumulation_cost(
            input_shape=input_shape,
            axes_summed=axes_summed,
            symmetry=symmetry,
            op_factor=op_factor,
            extra_ops=extra_ops,
            partition_budget=partition_budget,
        )

    return _compute


_reduction_cache = _make_reduction_cache(4096)


def get_reduction_cost_cached(
    *,
    input_shape: tuple,
    axes_summed: tuple,
    sym_fingerprint: tuple,
    op_factor: int,
    extra_ops: int,
    partition_budget: int | None,
) -> AccumulationCost:
    """Cached entry point for reduction-cost lookups."""
    return _reduction_cache(
        input_shape,
        axes_summed,
        sym_fingerprint,
        op_factor,
        extra_ops,
        partition_budget,
    )


def rebuild_reduction_cache(maxsize: int) -> None:
    """Rebuild the cache with a new maxsize."""
    global _reduction_cache
    _reduction_cache = _make_reduction_cache(maxsize)
