"""Per-op symmetry transport rules for NumPy-protocol shape ops.

Each `transport_<op>(group_or_groups, *, ...)` function returns either a new
`SymmetryGroup` (the output's surviving symmetry) or `None` (no non-trivial
group survives — caller emits `SymmetryLossWarning`).

See `.aicrowd/superpowers/specs/2026-05-22-shape-op-symmetry-transport-design.md`
for the per-op rules and rationale.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from flopscope._perm_group import SymmetryGroup
from flopscope._symmetry_utils import remap_group_axes


def _stub(name: str):
    def _impl(*args, **kwargs):  # pragma: no cover - replaced per task
        raise NotImplementedError(f"transport_{name} not yet implemented")
    _impl.__name__ = f"transport_{name}"
    return _impl


# Stubs — replaced in subsequent tasks.
transport_reshape = _stub("reshape")
transport_ravel = _stub("ravel")
transport_concatenate = _stub("concatenate")
transport_stack = _stub("stack")
transport_vstack = _stub("vstack")
transport_hstack = _stub("hstack")
transport_column_stack = _stub("column_stack")
transport_split = _stub("split")
transport_hsplit = _stub("hsplit")
transport_vsplit = _stub("vsplit")
transport_dsplit = _stub("dsplit")
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
transport_broadcast_to = _stub("broadcast_to")
transport_expand_dims = _stub("expand_dims")
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
transport_flip = _stub("flip")
transport_roll = _stub("roll")
transport_tile = _stub("tile")
transport_repeat = _stub("repeat")
transport_transpose = _stub("transpose")
transport_swapaxes = _stub("swapaxes")
transport_moveaxis = _stub("moveaxis")
transport_matrix_transpose = _stub("matrix_transpose")
