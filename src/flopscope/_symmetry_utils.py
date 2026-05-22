"""Helper primitives for exact tensor symmetry groups."""

from __future__ import annotations

import functools
import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from flopscope._perm_group import SymmetryGroup
from flopscope.errors import SymmetryError


def _normalize_axis_tuple(
    axes: Iterable[Any],
    *,
    ndim: int | None = None,
    what: str = "axes",
) -> tuple[int, ...]:
    norm_axes = tuple(axes)
    if not norm_axes:
        raise ValueError(f"{what} must be non-empty")
    if not all(isinstance(axis, int) for axis in norm_axes):
        raise TypeError(f"{what} must contain only integers")
    if len(set(norm_axes)) != len(norm_axes):
        raise ValueError(f"{what} contain duplicate entries")
    if ndim is not None and any(axis < 0 or axis >= ndim for axis in norm_axes):
        raise ValueError(f"{what} are out of range for ndim={ndim}")
    return norm_axes


def normalize_symmetry_input(obj, *, ndim: int | None = None):
    """Normalize supported symmetry shorthands to a single SymmetryGroup."""
    if obj is None:
        return None
    if isinstance(obj, SymmetryGroup):
        return validate_symmetry_group(obj, ndim=ndim)
    if (
        isinstance(obj, list)
        and obj
        and all(isinstance(group, SymmetryGroup) for group in obj)
    ):
        raise TypeError("symmetry must be a single SymmetryGroup, not a list of groups")
    if isinstance(obj, (tuple, list)) and obj:
        first = obj[0]
        if isinstance(first, int):
            axes = _normalize_axis_tuple(obj, ndim=ndim, what="symmetry axes")
            return SymmetryGroup.symmetric(axes=axes)
        if isinstance(first, (tuple, list)):
            blocks = []
            seen: set[int] = set()
            for block in obj:
                norm_block = _normalize_axis_tuple(
                    block, ndim=ndim, what="symmetry partition block"
                )
                overlap = seen & set(norm_block)
                if overlap:
                    raise ValueError(
                        "symmetry partition blocks overlap on axes "
                        f"{tuple(sorted(overlap))}"
                    )
                seen.update(norm_block)
                blocks.append(norm_block)
            return SymmetryGroup.young(blocks=tuple(blocks))
    raise TypeError(
        "symmetry must be a SymmetryGroup or an approved axis/partition shorthand"
    )


def validate_symmetry_group(
    group: SymmetryGroup,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> SymmetryGroup:
    """Validate tensor-facing properties of a symmetry group."""
    if not isinstance(group, SymmetryGroup):
        raise TypeError("symmetry must be a SymmetryGroup")
    axes = group.axes
    if axes is None:
        if ndim is not None and group.degree > ndim:
            raise ValueError(
                f"SymmetryGroup degree {group.degree} exceeds tensor rank {ndim}"
            )
        return group
    norm_axes = _normalize_axis_tuple(axes, ndim=ndim, what="SymmetryGroup axes")
    if norm_axes != axes:
        raise ValueError("SymmetryGroup axes must already be normalized")
    if shape is not None:
        for orbit in group.orbits():
            sizes = {shape[axes[i]] for i in orbit}
            if len(sizes) > 1:
                raise SymmetryError(
                    axes=tuple(axes[i] for i in orbit), max_deviation=float("inf")
                )
    return group


def unique_elements_for_shape(
    group: SymmetryGroup | None,
    shape: tuple[int, ...],
) -> int:
    """Return the number of unique tensor elements implied by symmetry."""
    if group is None:
        return math.prod(shape)
    return _unique_elements_for_shape_cached(group, tuple(shape))


@functools.cache
def _unique_elements_for_shape_cached(
    group: SymmetryGroup,
    shape: tuple[int, ...],
) -> int:
    validate_symmetry_group(group, ndim=len(shape), shape=shape)
    axes = group.axes
    if axes is None:
        axes = tuple(range(group.degree))
    size_dict = {local_idx: shape[axis] for local_idx, axis in enumerate(axes)}
    result = group.burnside_unique_count(size_dict)
    accounted = set(axes)
    for axis, size in enumerate(shape):
        if axis not in accounted:
            result *= size
    return result


def embed_group(group: SymmetryGroup | None, ndim: int) -> SymmetryGroup | None:
    """Embed a group acting on selected tensor axes into full rank ``ndim``."""
    if group is None:
        return None
    validate_symmetry_group(group, ndim=ndim)
    axes = group.axes
    if axes is None:
        axes = tuple(range(group.degree))
    if axes == tuple(range(ndim)) and group.degree == ndim:
        return group
    generators = []
    for generator in group.generators:
        arr = list(range(ndim))
        for local_idx, axis in enumerate(axes):
            arr[axis] = axes[generator.array_form[local_idx]]
        generators.append(arr)
    if not generators:
        generators.append(list(range(ndim)))
    return SymmetryGroup.from_generators(generators, axes=tuple(range(ndim)))


def restrict_group_to_axes(
    group: SymmetryGroup | None,
    axes: Iterable[int],
) -> SymmetryGroup | None:
    """Restrict a group to a specific ordered subset of its tensor axes."""
    if group is None:
        return None
    validate_symmetry_group(group)
    group_axes = group.axes
    if group_axes is None:
        group_axes = tuple(range(group.degree))
    wanted_axes = _normalize_axis_tuple(axes, what="restricted axes")
    local_indices = []
    for axis in wanted_axes:
        if axis not in group_axes:
            raise ValueError(
                f"restricted axes {wanted_axes} are not a subset of {group_axes}"
            )
        local_indices.append(group_axes.index(axis))
    if len(local_indices) < 2:
        return None
    restricted = group.restrict(tuple(local_indices))
    if restricted.order() <= 1:
        return None
    return restricted


def remap_group_axes(
    group: SymmetryGroup | None,
    axis_map: Mapping[int, int],
) -> SymmetryGroup | None:
    """Rename tensor axes while preserving the group's local action."""
    if group is None:
        return None
    validate_symmetry_group(group)
    axes = group.axes
    if axes is None:
        axes = tuple(range(group.degree))
    remapped_axes = []
    for axis in axes:
        if axis not in axis_map:
            raise ValueError(f"missing remap for axis {axis}")
        remapped_axes.append(axis_map[axis])
    _normalize_axis_tuple(remapped_axes, what="remapped axes")
    return SymmetryGroup.from_generators(
        group.generator_literals,  # pyright: ignore[reportArgumentType]
        axes=tuple(remapped_axes),  # type: ignore[arg-type]
    )


def remap_group_for_expand_dims(
    group: SymmetryGroup | None,
    *,
    ndim: int,
    axis,
) -> SymmetryGroup | None:
    """Remap tensor-axis support after ``numpy.expand_dims`` axis insertion."""
    if group is not None:
        validate_symmetry_group(group, ndim=ndim)
    probe_shape = tuple(range(2, 2 + ndim))
    probe = np.empty(probe_shape)
    expanded_shape = np.expand_dims(probe, axis=axis).shape
    remapped = None
    if group is not None:
        axis_map = {
            old_axis: expanded_shape.index(size)
            for old_axis, size in enumerate(probe_shape)
        }
        remapped = remap_group_axes(group, axis_map)
    inserted_axes = tuple(
        axis_idx for axis_idx, size in enumerate(expanded_shape) if size == 1
    )
    inserted = inserted_axes_symmetry(inserted_axes)
    return direct_product_groups(remapped, inserted)


def inserted_axes_symmetry(
    inserted_positions: Sequence[int],
) -> SymmetryGroup | None:
    """Symmetry of N freshly-inserted size-1 axes at the given output positions.

    Used by axis-inserting operations (``expand_dims``, ``__getitem__`` with
    ``None``/``np.newaxis``). Returns ``None`` for fewer than 2 positions
    (no non-trivial group). For 2+, returns
    ``SymmetryGroup.symmetric(axes=tuple(inserted_positions))``.
    """
    if len(inserted_positions) < 2:
        return None
    return SymmetryGroup.symmetric(axes=tuple(inserted_positions))


def intersect_groups(
    a: SymmetryGroup | None,
    b: SymmetryGroup | None,
    *,
    ndim: int,
) -> SymmetryGroup | None:
    """Intersect two groups after embedding them into the same tensor rank."""
    if a is None or b is None:
        return None
    if a.axes is not None and b.axes is not None and a.axes == b.axes:
        common = sorted(
            set(a.elements()) & set(b.elements()),
            key=lambda perm: tuple(perm.array_form),
        )
        if len(common) <= 1:
            return None
        return SymmetryGroup(*common, axes=a.axes)
    embedded_a = embed_group(a, ndim)
    embedded_b = embed_group(b, ndim)
    assert embedded_a is not None
    assert embedded_b is not None
    common = sorted(
        set(embedded_a.elements()) & set(embedded_b.elements()),
        key=lambda perm: tuple(perm.array_form),
    )
    if len(common) <= 1:
        return None
    return SymmetryGroup(*common, axes=tuple(range(ndim)))


def direct_product_groups(*groups: SymmetryGroup | None) -> SymmetryGroup | None:
    """Compose disjoint groups, dropping trivial and absent factors."""
    factors = []
    for group in groups:
        if group is None:
            continue
        validate_symmetry_group(group)
        if group.order() > 1:
            factors.append(group)
    if not factors:
        return None
    if len(factors) == 1:
        return factors[0]
    product = SymmetryGroup.direct_product(*factors)
    return product if product.order() > 1 else None


def broadcast_group(
    group: SymmetryGroup | None,
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> SymmetryGroup | None:
    """Broadcast a single input symmetry group onto an output shape."""
    if len(input_shape) > len(output_shape):
        raise ValueError("input rank cannot exceed output rank")

    factors: list[SymmetryGroup] = []
    offset = len(output_shape) - len(input_shape)

    created_by_size: OrderedDict[int, list[int]] = OrderedDict()
    for axis in range(offset):
        created_by_size.setdefault(output_shape[axis], []).append(axis)
    for block in created_by_size.values():
        if len(block) >= 2:
            factors.append(SymmetryGroup.symmetric(axes=tuple(block)))

    if group is not None:
        validate_symmetry_group(group, ndim=len(input_shape), shape=input_shape)
        axes = group.axes
        if axes is None:
            axes = tuple(range(group.degree))
        kept_local = []
        for local_idx, axis in enumerate(axes):
            out_axis = axis + offset
            if input_shape[axis] == 1 and output_shape[out_axis] > 1:
                continue
            kept_local.append(local_idx)
        if len(kept_local) >= 2:
            restricted = (
                group
                if len(kept_local) == group.degree
                else group.restrict(tuple(kept_local))
            )
            restricted_axes = (
                restricted.axes
                if restricted.axes is not None
                else tuple(range(restricted.degree))
            )
            remapped = remap_group_axes(
                restricted,
                {
                    restricted_axes[new_local_idx]: axes[old_local_idx] + offset
                    for new_local_idx, old_local_idx in enumerate(kept_local)
                },
            )
            if remapped is not None and remapped.order() > 1:
                factors.append(remapped)

    return direct_product_groups(*factors)


def reduce_group(
    group: SymmetryGroup | None,
    *,
    ndim: int,
    axis: int | tuple[int, ...] | None,
    keepdims: bool = False,
) -> SymmetryGroup | None:
    """Propagate a single symmetry group through a reduction."""
    if group is None or axis is None:
        return None
    validate_symmetry_group(group, ndim=ndim)
    axes_set = {axis % ndim} if isinstance(axis, int) else {a % ndim for a in axis}
    old_to_new: dict[int, int] = {}
    if keepdims:
        old_to_new = {dim: dim for dim in range(ndim)}
    else:
        new_idx = 0
        for dim in range(ndim):
            if dim not in axes_set:
                old_to_new[dim] = new_idx
                new_idx += 1

    group_axes = group.axes
    if group_axes is None:
        group_axes = tuple(range(group.degree))
    local_reduced = {
        i for i, tensor_axis in enumerate(group_axes) if tensor_axis in axes_set
    }
    local_kept = [
        i for i, tensor_axis in enumerate(group_axes) if tensor_axis not in axes_set
    ]

    if not local_reduced:
        remapped = remap_group_axes(
            group,
            {tensor_axis: old_to_new[tensor_axis] for tensor_axis in group_axes},
        )
        return remapped if remapped is not None and remapped.order() > 1 else None
    if not local_kept:
        return None

    stabilized = group.setwise_stabilizer(local_reduced)
    restricted = stabilized.restrict(tuple(local_kept))
    if restricted.order() <= 1:
        return None
    restricted_axes = (
        restricted.axes
        if restricted.axes is not None
        else tuple(range(restricted.degree))
    )
    remapped = remap_group_axes(
        restricted,
        {
            restricted_axes[new_local_idx]: old_to_new[group_axes[old_local_idx]]
            for new_local_idx, old_local_idx in enumerate(local_kept)
        },
    )
    return remapped if remapped is not None and remapped.order() > 1 else None


def wrap_with_symmetry(data, symmetry: SymmetryGroup | None):
    """Wrap ndarray-like data with symmetry metadata when a group is present."""
    array = np.asarray(data)
    if symmetry is None:
        return array
    validate_symmetry_group(symmetry, ndim=array.ndim)
    from flopscope._symmetric import SymmetricTensor

    return SymmetricTensor(array, symmetry=symmetry)


def wrap_with_trusted_symmetry(data, symmetry: SymmetryGroup | None):
    """Wrap data with already-proven symmetry metadata without re-validating.

    This helper is for internal call sites only, where the symmetry was
    generated or revalidated by trusted constructor logic. Avoiding the
    redundant validation call keeps constructor hot paths fast while leaving
    public/user-facing symmetry paths fully validated.
    """
    array = np.asarray(data)
    if symmetry is None:
        return array
    from flopscope._symmetric import SymmetricTensor

    return SymmetricTensor(array, symmetry=symmetry)
