"""Per-op symmetry transport rules for NumPy-protocol shape ops.

Each `transport_<op>(group_or_groups, *, ...)` function returns either a new
`SymmetryGroup` (the output's surviving symmetry) or `None` (no non-trivial
group survives — caller emits `SymmetryLossWarning`).

See `.aicrowd/superpowers/specs/2026-05-22-shape-op-symmetry-transport-design.md`
for the per-op rules and rationale.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from flopscope._perm_group import SymmetryGroup
from flopscope._symmetry_utils import (
    broadcast_group,
    group_orbits_on_axes,
    intersect_groups,
    _normalize_reps_for_output,
    remap_group_axes,
    remap_group_for_expand_dims,
    restrict_group_to_axes,
    setwise_stabilizer,
)


def _stub(name: str):
    def _impl(*args, **kwargs):  # pragma: no cover - replaced per task
        raise NotImplementedError(f"transport_{name} not yet implemented")
    _impl.__name__ = f"transport_{name}"
    return _impl


def transport_reshape(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> SymmetryGroup | None:
    if group is None:
        return None
    A = group.axes or tuple(range(group.degree))
    m = input_shape[A[0]]  # block axis size (must be equal for all a in A by group validity)

    A_sorted = sorted(A)
    n = len(A_sorted)

    # Skeleton segment products from input (n+1 segments interleaved with n block axes).
    seg_input = []
    prev = -1
    for a in A_sorted:
        seg_input.append(math.prod(input_shape[prev + 1 : a]))
        prev = a
    seg_input.append(math.prod(input_shape[prev + 1 :]))

    # Walk through output, matching one segment then one block axis at a time.
    out_positions: list[int] = []
    pos = 0
    for k in range(n):
        target = seg_input[k]
        accum = 1
        while accum < target and pos < len(output_shape):
            accum *= output_shape[pos]
            pos += 1
        if accum != target:
            return None
        # Skip length-1 padding axes between segment and block axis (unless m == 1).
        if m != 1:
            while pos < len(output_shape) and output_shape[pos] == 1:
                pos += 1
        if pos >= len(output_shape) or output_shape[pos] != m:
            return None
        out_positions.append(pos)
        pos += 1

    if math.prod(output_shape[pos:]) != seg_input[-1]:
        return None

    axis_map = {a: out_positions[A_sorted.index(a)] for a in A}
    return remap_group_axes(group, axis_map)


def transport_ravel(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
) -> SymmetryGroup | None:
    return transport_reshape(
        group, input_shape=input_shape, output_shape=(math.prod(input_shape),),
    )
def transport_concatenate(
    groups: Sequence[SymmetryGroup | None],
    *,
    output_ndim: int,
    axis: int | None,
) -> SymmetryGroup | None:
    if axis is None:
        return None
    if any(g is None for g in groups):
        return None
    # Restrict each input's group to axes != axis.
    restricted = []
    for g in groups:
        A = g.axes or tuple(range(g.degree))
        a = axis % (max(output_ndim, len(A) + 1))
        keep = tuple(x for x in A if x != a)
        if len(keep) < 2:
            return None
        r = restrict_group_to_axes(g, keep)
        if r is None:
            return None
        restricted.append(r)
    # Intersect across all restricted groups.
    result = restricted[0]
    for g in restricted[1:]:
        result = intersect_groups(result, g, ndim=output_ndim)
        if result is None:
            return None
    return result
def transport_stack(
    groups: Sequence[SymmetryGroup | None],
    *,
    output_ndim: int,
    axis: int = 0,
) -> SymmetryGroup | None:
    if any(g is None for g in groups):
        return None
    k = axis % output_ndim
    # Shift each input's block axes >= k by +1.
    shifted = []
    for g in groups:
        A = g.axes or tuple(range(g.degree))
        axis_map = {a: (a if a < k else a + 1) for a in A}
        r = remap_group_axes(g, axis_map)
        if r is None:
            return None
        shifted.append(r)
    # Intersect.
    result = shifted[0]
    for g in shifted[1:]:
        result = intersect_groups(result, g, ndim=output_ndim)
        if result is None:
            return None
    return result
def transport_vstack(
    groups: Sequence[SymmetryGroup | None],
    *,
    output_ndim: int,
    input_ndims: Sequence[int],
) -> SymmetryGroup | None:
    # vstack: atleast_2d each input, then concat axis=0.
    # For any input that's 1-D, the promoted form has the data axis at position 1,
    # and 1-D inputs carry no multi-axis group anyway -> None for them.
    promoted = []
    for g, nd in zip(groups, input_ndims):
        if nd >= 2:
            promoted.append(g)
        else:
            # 1-D input promoted to (1, N) -- no multi-axis group.
            promoted.append(None)
    return transport_concatenate(promoted, output_ndim=output_ndim, axis=0)


def transport_hstack(
    groups: Sequence[SymmetryGroup | None],
    *,
    output_ndim: int,
    input_ndims: Sequence[int],
) -> SymmetryGroup | None:
    # hstack: concat axis=0 if all 1-D, else axis=1.
    if all(nd == 1 for nd in input_ndims):
        return transport_concatenate(groups, output_ndim=output_ndim, axis=0)
    return transport_concatenate(groups, output_ndim=output_ndim, axis=1)


def transport_column_stack(
    groups: Sequence[SymmetryGroup | None],
    *,
    output_ndim: int,
    input_ndims: Sequence[int],
) -> SymmetryGroup | None:
    # column_stack: promote 1-D (N,) -> (N, 1), then concat axis=1.
    # 1-D inputs become column vectors with no multi-axis group.
    promoted = []
    for g, nd in zip(groups, input_ndims):
        if nd >= 2:
            promoted.append(g)
        else:
            promoted.append(None)
    return transport_concatenate(promoted, output_ndim=output_ndim, axis=1)
def transport_split(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
    axis: int = 0,
) -> SymmetryGroup | None:
    if group is None:
        return None
    a = axis % len(input_shape)
    A = group.axes or tuple(range(group.degree))
    keep = tuple(x for x in A if x != a)
    if len(keep) < 2:
        return None
    return restrict_group_to_axes(group, keep)


def transport_hsplit(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
) -> SymmetryGroup | None:
    # hsplit uses axis=0 for 1-D, axis=1 otherwise.
    if len(input_shape) == 1:
        return transport_split(group, input_shape=input_shape, axis=0)
    return transport_split(group, input_shape=input_shape, axis=1)


def transport_vsplit(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
) -> SymmetryGroup | None:
    return transport_split(group, input_shape=input_shape, axis=0)


def transport_dsplit(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
) -> SymmetryGroup | None:
    return transport_split(group, input_shape=input_shape, axis=2)
def transport_atleast_1d(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
) -> SymmetryGroup | None:
    if group is None:
        return None
    # For any input with rank >= 1, atleast_1d is identity.
    # (Rank 0 -> rank 1, but rank 0 can't carry a multi-axis group.)
    return group


def transport_atleast_2d(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
) -> SymmetryGroup | None:
    if group is None:
        return None
    # For any input with rank >= 2 (which is the only case that can carry
    # a non-trivial multi-axis group), atleast_2d is identity.
    return group


def transport_atleast_3d(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
) -> SymmetryGroup | None:
    if group is None:
        return None
    if len(input_shape) >= 3:
        # No-op.
        return group
    if len(input_shape) == 2:
        # NumPy appends a trailing length-1 axis: (M, N) -> (M, N, 1).
        # Block axes don't shift.
        return group
    # len(input_shape) <= 1: cannot carry a multi-axis group; defensive None.
    return None
def transport_broadcast_to(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> SymmetryGroup | None:
    if group is None:
        return None
    return broadcast_group(
        group, input_shape=input_shape, output_shape=output_shape,
    )


def transport_expand_dims(
    group: SymmetryGroup | None,
    *,
    input_ndim: int,
    axis,
) -> SymmetryGroup | None:
    return remap_group_for_expand_dims(group, ndim=input_ndim, axis=axis)
def transport_squeeze(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None,
) -> SymmetryGroup | None:
    if group is None:
        return None
    A = group.axes or tuple(range(group.degree))
    # Resolve axis to a sorted tuple of axes being squeezed.
    if axis is None:
        squeezed = tuple(i for i, s in enumerate(input_shape) if s == 1)
    elif isinstance(axis, int):
        squeezed = (axis % len(input_shape),)
    else:
        squeezed = tuple(sorted(a % len(input_shape) for a in axis))
    # Rule (b): if any squeezed axis is in the block, drop.
    if any(a in A for a in squeezed):
        return None
    # Rule (a): shift block axes > each squeezed axis down by 1, preserving order.
    surviving = sorted(i for i in range(len(input_shape)) if i not in squeezed)
    new_index = {old: new for new, old in enumerate(surviving)}
    axis_map = {a: new_index[a] for a in A}
    return remap_group_axes(group, axis_map)
def transport_flip(
    group: SymmetryGroup | None,
    *,
    ndim: int,
    axes_flipped,
) -> SymmetryGroup | None:
    if group is None:
        return None
    # Normalize axes_flipped: None -> all, int -> tuple, negative wrap.
    if axes_flipped is None:
        F = set(range(ndim))
    elif isinstance(axes_flipped, int):
        F = {axes_flipped % ndim}
    else:
        F = {a % ndim for a in axes_flipped}
    A = set(group.axes or range(group.degree))
    F_A = F & A
    if not F_A or F_A == A:
        return group
    return setwise_stabilizer(group, fixed_set=F_A)
def transport_tile(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    reps,
) -> SymmetryGroup | None:
    if group is None:
        return None
    A = group.axes or tuple(range(group.degree))
    shift = len(output_shape) - len(input_shape)
    reps_norm = _normalize_reps_for_output(reps, output_ndim=len(output_shape))
    # Orbit-constancy check on output-positioned reps.
    for orbit in group_orbits_on_axes(group, A):
        reps_in_orbit = {reps_norm[a + shift] for a in orbit}
        if len(reps_in_orbit) > 1:
            return None
    if shift == 0:
        return group
    # Block axes shift in the output; relabel the group.
    axis_map = {a: a + shift for a in A}
    return remap_group_axes(group, axis_map)


def transport_repeat(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
    axis: int | None,
) -> SymmetryGroup | None:
    if group is None:
        return None
    if axis is None:
        # repeat(axis=None) ravels first -> never preserves multi-axis sym.
        return None
    a = axis % len(input_shape)
    A = set(group.axes or range(group.degree))
    if a in A:
        return None
    return group


def transport_roll(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None,
) -> SymmetryGroup | None:
    if group is None:
        return None
    if axis is None:
        # roll(axis=None) flattens first -> always drops.
        return None
    if isinstance(axis, int):
        rolled = {axis % len(input_shape)}
    else:
        rolled = {a % len(input_shape) for a in axis}
    A = set(group.axes or range(group.degree))
    if rolled & A:
        return None
    return group
def transport_transpose(
    group: SymmetryGroup | None,
    *,
    ndim: int,
    axes: Sequence[int] | None = None,
) -> SymmetryGroup | None:
    if group is None:
        return None
    if axes is None:
        order = tuple(reversed(range(ndim)))
    else:
        order = tuple(a % ndim for a in axes)
    mapping = {old: new for new, old in enumerate(order)}
    return remap_group_axes(group, mapping)


def transport_swapaxes(
    group: SymmetryGroup | None,
    *,
    ndim: int,
    axis1: int,
    axis2: int,
) -> SymmetryGroup | None:
    if group is None:
        return None
    a1 = axis1 % ndim
    a2 = axis2 % ndim
    order = list(range(ndim))
    order[a1], order[a2] = order[a2], order[a1]
    mapping = {old: new for new, old in enumerate(order)}
    return remap_group_axes(group, mapping)


def transport_moveaxis(
    group: SymmetryGroup | None,
    *,
    ndim: int,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> SymmetryGroup | None:
    if group is None:
        return None
    src = (source,) if isinstance(source, int) else tuple(source)
    dst = (destination,) if isinstance(destination, int) else tuple(destination)
    src = tuple(s % ndim for s in src)
    dst = tuple(d % ndim for d in dst)
    order = [i for i in range(ndim) if i not in src]
    for d, s in sorted(zip(dst, src)):
        order.insert(d, s)
    mapping = {old: new for new, old in enumerate(order)}
    return remap_group_axes(group, mapping)


def transport_matrix_transpose(
    group: SymmetryGroup | None,
    *,
    ndim: int,
) -> SymmetryGroup | None:
    if ndim < 2:
        return None
    return transport_swapaxes(group, ndim=ndim, axis1=ndim - 2, axis2=ndim - 1)
