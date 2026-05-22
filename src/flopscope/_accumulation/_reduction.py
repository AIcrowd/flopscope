"""Reduction-cost orchestrator + helpers.

Implements the two-tier reduction cost model:
- Tier 1 (ufunc.reduce): orbit-mapping α via compute_accumulation_cost
- Tier 2 (non-ufunc):   dense × (num_output_orbits / num_output_elems)

See .aicrowd/superpowers/specs/2026-05-13-symmetry-aware-reduction-cost-design.md.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flopscope._perm_group import SymmetryGroup


def _normalize_axis(
    axis: int | tuple[int, ...] | list[int] | None,
    ndim: int,
) -> tuple[int, ...]:
    """Normalize an axis specifier to a sorted tuple of non-negative ints.

    - None → all axes (full reduction)
    - int → singleton, with negative-index wrapping
    - tuple/list → sorted unique, with negative-index wrapping
    """
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        return (axis % ndim,)
    return tuple(sorted({a % ndim for a in axis}))


def _num_output_orbits(
    input_shape: tuple[int, ...],
    axes_summed: tuple[int, ...],
    symmetry: SymmetryGroup | None,
) -> int:
    """Count orbits of the output multi-index under the output stabilizer.

    Output shape = input_shape with axes_summed removed.
    Output stabilizer = reduce_group(symmetry, ndim=ndim, axis=axes_summed).
    Returns the size-aware Burnside orbit count on the output shape.
    """
    ndim = len(input_shape)
    axes_set = frozenset(axes_summed)
    output_shape = tuple(input_shape[i] for i in range(ndim) if i not in axes_set)

    if not output_shape:
        return 1

    if symmetry is None:
        return math.prod(output_shape)

    from flopscope._symmetry_utils import reduce_group

    from ._burnside import size_aware_burnside

    output_group = reduce_group(
        symmetry,
        ndim=ndim,
        axis=axes_summed,
        keepdims=False,
    )

    if output_group is None:
        return math.prod(output_shape)

    # The group's permutation indices are LOCAL (0..degree-1).
    # Map them to the axes the group actually operates on in output_shape,
    # then extract only those sizes so Burnside sees the right array shape.
    group_output_axes = (
        output_group.axes
        if output_group.axes is not None
        else tuple(range(output_group.degree))
    )
    group_sizes = tuple(output_shape[ax] for ax in group_output_axes)

    elements = output_group.elements()
    orbits_on_group_dims = size_aware_burnside(elements, group_sizes)

    # Remaining output dims (not in the group) contribute independently.
    group_axes_set = frozenset(group_output_axes)
    remaining = math.prod(
        output_shape[i] for i in range(len(output_shape)) if i not in group_axes_set
    )
    return orbits_on_group_dims * max(remaining, 1)


def output_discounted_reduction_cost(
    input_shape: tuple[int, ...],
    axes_summed: tuple[int, ...],
    symmetry: SymmetryGroup | None,
    dense_per_output_cost: int,
) -> int:
    """Tier-2 reduction cost: num_output_orbits × dense_per_output_cost.

    For non-ufunc reductions (median, percentile, quantile) where we can't
    decompose the operation internally. The discount uses only the output
    stabilizer; input symmetry doesn't help unless the operation is
    decomposable.

    Equivalent to: dense_total × (num_output_orbits / num_output_elems).
    """
    num_orbits = _num_output_orbits(input_shape, axes_summed, symmetry)
    return num_orbits * dense_per_output_cost


def compute_reduction_accumulation_cost(
    input_shape: tuple[int, ...],
    axes_summed: tuple[int, ...],
    symmetry: SymmetryGroup | None = None,
    *,
    op_factor: int = 1,
    extra_ops: int = 0,
    partition_budget: int | None = None,
):
    """Tier-1 reduction cost: orbit-mapping α via the existing ladder.

    Constructs a single-operand 'fake einsum' (input subscripts = full label
    set, output subscripts = labels not in axes_summed), runs the existing
    component decomposition + ladder, then aggregates via aggregate_reduction.

    Off-by-one fix (#56): cost = op_factor × (α - num_output_orbits) + extra_ops.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input array.
    axes_summed : tuple of int
        Axes to reduce. Must be non-negative, sorted, unique.
        Use _normalize_axis to prepare.
    symmetry : SymmetryGroup, optional
        Pointwise symmetry of the input. None for dense inputs.
    op_factor : int, default 1
        FLOPs per binary accumulation event.
    extra_ops : int, default 0
        Output-side dense additions (e.g. num_output_orbits for mean's divide).
    partition_budget : int, optional
        Override the partition counter budget. Defaults to global setting.

    Returns
    -------
    AccumulationCost
        Same shape as einsum_accumulation_cost — fields total/mu/alpha/m_total/...
    """
    from ._cost import aggregate_reduction, compute_accumulation_cost

    ndim = len(input_shape)
    if not axes_summed:
        # No reduction at all — cost is 0 (just identity).
        return _trivial_zero_cost(input_shape, op_factor, extra_ops)

    # Build a fake einsum: 'abc...' input, output is labels of kept axes.
    if ndim > 26:
        # Fall back to dense — labelers can't go beyond a-z cleanly. Unusual.
        return _dense_fallback_cost(
            input_shape,
            axes_summed,
            symmetry,
            op_factor,
            extra_ops,
        )

    axes_set = frozenset(axes_summed)
    labels = [chr(ord("a") + i) for i in range(ndim)]
    input_subs = "".join(labels)
    output_subs = "".join(labels[i] for i in range(ndim) if i not in axes_set)
    canonical_subscripts = f"{input_subs}->{output_subs}"

    # Single operand, per-op symmetry = the input symmetry.
    per_op_symmetries = (symmetry,)

    from flopscope._perm_group import _DiminoBudgetExceeded

    try:
        einsum_cost = compute_accumulation_cost(
            canonical_subscripts=canonical_subscripts,
            input_parts=(input_subs,),
            output_subscript=output_subs,
            shapes=(input_shape,),
            per_op_symmetries=per_op_symmetries,
            identity_pattern=None,  # single operand, no identity grouping
            partition_budget=partition_budget,
        )

        # `output_dense` carries num_output_orbits — orbit-aware, computed
        # directly via _num_output_orbits. For dense inputs this equals
        # prod(output_shape) so the locked-stub docstring's intent is preserved.
        num_output_orbits = _num_output_orbits(
            input_shape=input_shape,
            axes_summed=axes_summed,
            symmetry=symmetry,
        )
    except _DiminoBudgetExceeded as exc:
        # _num_output_orbits' element enumeration can also exceed the budget
        # on an oversized output stabilizer. Charge dense and warn.
        # (compute_accumulation_cost emits its own CostFallbackWarning when it
        # bails before this; if it succeeded but _num_output_orbits bailed,
        # this is the only warning the caller sees.)
        import warnings as _warnings

        from flopscope.errors import CostFallbackWarning

        _warnings.warn(
            CostFallbackWarning(
                f"reduction: dimino_budget exceeded ({exc.seen_count} > {exc.budget}) "
                f"while computing output orbit count — charging dense cost. "
                f"Raise via flopscope.configure(dimino_budget=...) to attempt "
                f"exact counting."
            ),
            stacklevel=3,
        )
        return _dense_fallback_cost(
            input_shape, axes_summed, symmetry, op_factor, extra_ops
        )
    dense_baseline = math.prod(input_shape) if input_shape else 1

    # If compute_accumulation_cost already bailed (e.g. dimino_budget exceeded
    # on an auto-inferred S_n), it returns an AccumulationCost with
    # fallback_used=True and an empty per_component tuple. aggregate_reduction
    # would then mis-interpret the empty tuple as "no failures"; re-shape the
    # cost under the reduction-fallback formula instead.
    if einsum_cost.fallback_used and not einsum_cost.per_component:
        from ._cost import AccumulationCost

        input_axis_size = (
            dense_baseline // num_output_orbits if num_output_orbits else 0
        )
        fallback_total = (
            num_output_orbits * max(0, input_axis_size - 1) * op_factor + extra_ops
        )
        return AccumulationCost(
            total=fallback_total,
            mu=None,
            alpha=None,
            m_total=1,
            dense_baseline=dense_baseline,
            num_terms=1,
            per_component=(),
            fallback_used=True,
            unavailable_components=(),
            unavailable_reason=einsum_cost.unavailable_reason,
        )

    # Re-aggregate the per-component costs under the reduction formula.
    return aggregate_reduction(
        einsum_cost.per_component,
        op_factor=op_factor,
        dense_baseline=dense_baseline,
        output_dense=num_output_orbits,
        extra_ops=extra_ops,
    )


def _trivial_zero_cost(
    input_shape: tuple[int, ...],
    op_factor: int,
    extra_ops: int,
):
    """Cost of a 'no axes reduced' degenerate case."""
    from ._cost import AccumulationCost

    return AccumulationCost(
        total=extra_ops,
        mu=0,
        alpha=0,
        m_total=1,
        dense_baseline=math.prod(input_shape) if input_shape else 1,
        num_terms=1,
        per_component=(),
        fallback_used=False,
    )


def _dense_fallback_cost(
    input_shape: tuple[int, ...],
    axes_summed: tuple[int, ...],
    symmetry: SymmetryGroup | None,
    op_factor: int,
    extra_ops: int,
):
    """ndim > 26 fallback: charge dense without symmetry."""
    from ._cost import AccumulationCost

    axes_set = frozenset(axes_summed)
    dense = math.prod(input_shape) if input_shape else 1
    output_dense = (
        math.prod(input_shape[i] for i in range(len(input_shape)) if i not in axes_set)
        if any(i not in axes_set for i in range(len(input_shape)))
        else 1
    )
    input_axis_size = dense // output_dense if output_dense else 0
    total = output_dense * max(0, input_axis_size - 1) * op_factor + extra_ops
    return AccumulationCost(
        total=total,
        mu=None,
        alpha=None,
        m_total=1,
        dense_baseline=dense,
        num_terms=1,
        per_component=(),
        fallback_used=True,
    )
