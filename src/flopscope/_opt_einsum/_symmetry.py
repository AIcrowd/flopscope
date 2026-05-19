"""Symmetry-aware cost helpers for opt_einsum contraction paths.

This module provides:
- unique_elements / compute_unique_size: count distinct elements under symmetry.
- symmetric_flop_count: FLOP estimate reduced by output symmetry.

Detection of symmetries is handled by ``_subgraph_symmetry.SubgraphSymmetryOracle``.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from flopscope._perm_group import SymmetryGroup

from ._helpers import flop_count


@dataclass(frozen=True)
class SubsetSymmetry:
    """Symmetry info for one contraction subset, split by V/W.

    Attributes
    ----------
    output : SymmetryGroup or None
        Exact permutation group for the output (V) labels.
    inner : SymmetryGroup or None
        Exact permutation group for the inner (W) labels.
    """

    output: SymmetryGroup | None
    inner: SymmetryGroup | None


def unique_elements(
    indices: frozenset[str],
    size_dict: dict[str, int],
    perm_group: SymmetryGroup | None = None,
) -> int:
    """Count distinct elements of a tensor with the given symmetry.

    When perm_group is provided, uses Burnside's lemma for exact counting.
    When None, returns the dense count (product of all sizes).
    """
    if not indices:
        return 1

    if perm_group is not None:
        if perm_group._labels is not None:
            label_list = list(perm_group._labels)
        else:
            label_list = sorted(indices)[: perm_group.degree]
        pg_size_dict: dict[int, int] = {}
        accounted: set[str] = set()
        for i, lbl in enumerate(label_list):
            pg_size_dict[i] = size_dict[lbl]
            accounted.add(lbl)
        count = perm_group.burnside_unique_count(pg_size_dict)
        for idx in indices:
            if idx not in accounted:
                count *= size_dict[idx]
        return count

    # No symmetry -- dense count.
    count = 1
    for idx in indices:
        count *= size_dict[idx]
    return count


compute_unique_size = unique_elements


def symmetric_flop_count(
    idx_contraction: Collection[str],
    inner: bool,
    num_terms: int,
    size_dictionary: dict[str, int],
    *,
    output_group: SymmetryGroup | None = None,
    output_indices: frozenset[str] | None = None,
    inner_group: SymmetryGroup | None = None,
    inner_indices: frozenset[str] | None = None,
    use_inner_symmetry: bool = True,
    per_operand_free_counts: tuple[int, ...] | None = None,
) -> int:
    r"""FLOP count for a symmetric tensor contraction.

    Computes the direct-evaluation cost, reduced by the output and inner
    symmetry ratios when the corresponding groups are provided.

    Inner symmetry reduction is applied only when ``use_inner_symmetry``
    is True and **all** labels in ``inner_group`` are present as
    contracted indices in this specific step. If any W-group labels were
    contracted at earlier steps and are no longer present, the inner
    reduction is skipped.

    Parameters
    ----------
    use_inner_symmetry : bool, optional
        Whether to apply the inner (W-side) symmetry reduction.
        Default ``True``.
    per_operand_free_counts : tuple of int, optional
        Number of free (non-contracted) indices contributed by each
        operand.  For a pairwise contraction this is ``(s, t)``.
        Reserved for future cost models; not currently used.
    """
    from ._helpers import compute_size_by_dict

    # --- Direct-evaluation estimate ---
    cost = flop_count(idx_contraction, inner, num_terms, size_dictionary)

    if output_group is not None and output_indices is not None:
        total = compute_size_by_dict(output_indices, size_dictionary)
        unique = unique_elements(
            output_indices, size_dictionary, perm_group=output_group
        )
        cost = cost * unique // total

    # Inner symmetry: only apply when ALL group labels are being contracted
    # at this step.  The oracle's W-group may span labels from prior steps
    # that no longer exist in the current pairwise contraction.
    if use_inner_symmetry and inner_group is not None and inner_indices is not None:
        group_labels = set(inner_group._labels) if inner_group._labels else set()
        if group_labels and group_labels <= set(inner_indices):
            total_inner = compute_size_by_dict(inner_indices, size_dictionary)
            if total_inner > 0:
                unique_inner = unique_elements(
                    inner_indices, size_dictionary, perm_group=inner_group
                )
                cost = cost * unique_inner // total_inner

    return max(cost, 1)
