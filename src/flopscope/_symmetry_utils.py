"""Helper primitives for exact tensor symmetry groups."""

from __future__ import annotations

import functools
import math
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple as _np_normalize_axis_tuple

from flopscope._config import get_setting as _get_setting
from flopscope._perm_group import SymmetryGroup, _DiminoBudgetExceeded
from flopscope.errors import CostFallbackWarning, SymmetryError


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
    """Return the number of unique tensor elements implied by symmetry.

    Degrades to the dense element count when the group cannot be enumerated
    within ``dimino_budget``: unknown-kind groups (raw generators -- e.g. the
    auto-inferred full-exchange symmetry of a constant array) have no closed
    form for either the Burnside count or the ``__hash__`` canonicalization
    that keying the cache requires, so both can raise
    ``_DiminoBudgetExceeded`` mid-enumeration. Charging dense is the same
    never-under-bill fallback the oversized-group guards in ``_pointwise``
    use; leaking the private exception through whatever billed op asked for
    the count is what numpy 2.4's ``TestCreationFuncs::test_full`` first
    surfaced.
    """
    if group is None:
        return math.prod(shape)
    try:
        return _unique_elements_for_shape_cached(group, tuple(shape))
    except _DiminoBudgetExceeded:
        _warn_dense_fallback_once(group.degree, tuple(shape))
        return math.prod(shape)


@functools.cache
def _dense_fallback_seen(degree: int, shape: tuple[int, ...]) -> None:
    return None


def _warn_dense_fallback_once(degree: int, shape: tuple[int, ...]) -> None:
    """Emit :class:`CostFallbackWarning` once per ``(degree, shape)``.

    Mirrors ``_pointwise._warn_oversized_once``: the setting check stays
    OUTSIDE the seen-cache so re-enabling ``symmetry_warnings`` later still
    warns, and hot paths (compat suites doing thousands of ops on the same
    inferred symmetry) do not flood the log. Keyed by degree, not the group
    object -- hashing the group is exactly what can blow the budget.
    """
    if not _get_setting("symmetry_warnings"):
        return
    info_before = _dense_fallback_seen.cache_info()
    _dense_fallback_seen(degree, shape)
    if _dense_fallback_seen.cache_info().hits > info_before.hits:
        return  # already warned for this (degree, shape) pair
    budget = int(_get_setting("dimino_budget"))  # type: ignore[arg-type]
    warnings.warn(
        f"unique-element count for a degree-{degree} SymmetryGroup exceeded "
        f"the enumeration budget ({budget}); charging the dense element "
        "count. Suppress with flops.configure(symmetry_warnings=False).",
        CostFallbackWarning,
        stacklevel=3,
    )


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


def _build_from_kind(kind: tuple) -> SymmetryGroup | None:
    """Construct an interned SymmetryGroup from a ``_known_kind`` tag.

    Routes through the public factory matching the kind name. Returns
    ``None`` for trivial (identity) kinds, since callers expect ``None``
    to represent "no non-trivial symmetry."
    """
    name = kind[0]
    if name == "identity":
        return None
    if name == "symmetric":
        return SymmetryGroup.symmetric(axes=kind[1])
    if name == "cyclic":
        return SymmetryGroup.cyclic(axes=kind[1])
    if name == "dihedral":
        return SymmetryGroup.dihedral(axes=kind[1])
    if name == "direct_product":
        children = [_build_from_kind(child) for child in kind[1]]
        non_trivial = [c for c in children if c is not None]
        if not non_trivial:
            return None
        if len(non_trivial) == 1:
            return non_trivial[0]
        return SymmetryGroup.direct_product(*non_trivial)
    raise AssertionError(f"unknown kind {kind!r}")


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
    """Restrict a group to a specific ordered subset of its tensor axes.

    The helper composes :meth:`SymmetryGroup.setwise_stabilizer` and
    :meth:`SymmetryGroup.restrict` so strict subsets of free-permuting groups
    (e.g. ``symmetric(A)``) project cleanly to a sub-action. Provenance is
    preserved only in the no-op case (``axes == group.axes``); strict-subset
    results carry ``_known_kind=None`` — the "sub-symmetric is still symmetric"
    rule lives in ``reduce_group``, not here.
    """
    if group is None:
        return None
    validate_symmetry_group(group)
    group_axes = group.axes
    if group_axes is None:
        group_axes = tuple(range(group.degree))
    wanted_axes = _normalize_axis_tuple(axes, what="restricted axes")
    if wanted_axes == group_axes:
        # No-op: kind passes through via the interned original.
        return group
    local_indices = []
    for axis in wanted_axes:
        if axis not in group_axes:
            raise ValueError(
                f"restricted axes {wanted_axes} are not a subset of {group_axes}"
            )
        local_indices.append(group_axes.index(axis))
    if len(local_indices) < 2:
        return None
    kept = tuple(local_indices)
    # First compute the setwise stabilizer so that restrict() only sees
    # permutations that map the kept set to itself.
    stabilized = group.setwise_stabilizer(set(kept))
    restricted = stabilized.restrict(kept)
    if restricted.order() <= 1:
        return None
    return restricted


def _remap_kind(kind: tuple | None, axis_map: Mapping[Any, Any]) -> tuple | None:
    """Apply ``axis_map`` to the axes inside a ``_known_kind`` tag.

    Returns ``None`` if any leaf axis is missing from the map (caller's
    responsibility to ensure full coverage).
    """
    if kind is None:
        return None
    name = kind[0]
    if name in ("identity", "symmetric", "cyclic", "dihedral"):
        try:
            return (name, tuple(axis_map[a] for a in kind[1]))
        except KeyError:
            return None
    if name == "direct_product":
        children = tuple(_remap_kind(child, axis_map) for child in kind[1])
        if any(child is None for child in children):
            return None
        return ("direct_product", tuple(sorted(children, key=repr)))
    return None


def _reduced_kind(
    kind: tuple | None,
    *,
    reduced_axes: set[int],
    axis_map: Mapping[Any, Any],
) -> tuple | None:
    """Compute the reduced kind tag for a known-kind group.

    ``reduced_axes`` is the set of tensor axes being reduced over (in the
    parent group's axis space). ``axis_map`` is the surviving-axes mapping
    from old tensor axes to new tensor positions (post-reduction layout).

    Returns ``None`` if the result is trivial or can't be expressed in
    closed form.
    """
    if kind is None:
        return None
    name = kind[0]
    if name == "identity":
        kept = tuple(axis_map[a] for a in kind[1] if a not in reduced_axes)
        if not kept:
            return None
        return ("identity", kept)
    if name == "symmetric":
        kept = tuple(axis_map[a] for a in kind[1] if a not in reduced_axes)
        if len(kept) < 2:
            return None
        return ("symmetric", kept)
    if name == "cyclic":
        # Cyclic only survives if NONE of its axes are reduced (the cycle
        # would otherwise no longer be a closed orbit on the kept axes).
        if any(a in reduced_axes for a in kind[1]):
            return None
        kept = tuple(axis_map[a] for a in kind[1])
        return ("cyclic", kept)
    if name == "dihedral":
        if any(a in reduced_axes for a in kind[1]):
            return None
        kept = tuple(axis_map[a] for a in kind[1])
        return ("dihedral", kept)
    if name == "direct_product":
        new_children = []
        for child in kind[1]:
            new_child = _reduced_kind(
                child, reduced_axes=reduced_axes, axis_map=axis_map
            )
            if new_child is not None:
                new_children.append(new_child)
        if not new_children:
            return None
        if len(new_children) == 1:
            return new_children[0]
        return ("direct_product", tuple(sorted(new_children, key=repr)))
    return None


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
    remapped = SymmetryGroup.from_generators(
        group.generator_literals,  # pyright: ignore[reportArgumentType]
        axes=tuple(remapped_axes),  # type: ignore[arg-type]
    )
    new_kind = _remap_kind(group._known_kind, axis_map)
    if new_kind is not None:
        remapped._known_kind = new_kind
        remapped = SymmetryGroup._intern(remapped)
    return remapped


def remap_group_for_expand_dims(
    group: SymmetryGroup | None,
    *,
    ndim: int,
    axis,
) -> SymmetryGroup | None:
    """Remap tensor-axis support after ``numpy.expand_dims`` axis insertion.

    Pure shape arithmetic: no array is allocated. This used to build
    ``probe_shape = tuple(range(2, 2 + ndim))`` and allocate
    ``np.empty(probe_shape)`` -- ``(ndim + 1)!`` float64 elements, 152 TiB at
    rank 15 -- solely to read one ``.shape``, and did it unconditionally, even
    when ``group is None``. ``expand_dims`` therefore died around rank 14 where
    numpy itself handles 41 (#173).

    The probe's distinct sizes served a second purpose: ``axis_map`` was built
    by ``expanded_shape.index(size)``, which needed every original axis to have
    a unique length. The arithmetic below reproduces that mapping exactly --
    original axes fill the output positions the inserted axes did not take, in
    order -- without needing distinct sizes.

    Axis normalization is delegated to numpy's own ``normalize_axis_tuple``
    rather than reimplemented, so negative, tuple, repeated and out-of-range
    axis forms raise exactly what ``numpy.expand_dims`` raises on every numpy
    in the supported range.
    """
    if group is not None:
        validate_symmetry_group(group, ndim=ndim)
    # Mirrors numpy.expand_dims: a non-tuple/list axis is wrapped, the output
    # rank is ndim + len(axis), and normalization is relative to that rank.
    axes = axis if type(axis) in (tuple, list) else (axis,)
    out_ndim = len(axes) + ndim
    inserted_axes = tuple(sorted(_np_normalize_axis_tuple(axes, out_ndim)))
    remapped = None
    if group is not None:
        inserted_set = set(inserted_axes)
        original_positions = [
            position for position in range(out_ndim) if position not in inserted_set
        ]
        axis_map = dict(enumerate(original_positions))
        remapped = remap_group_axes(group, axis_map)
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
    # Easy known-kind case — same group ∩ itself = itself. Preserve
    # provenance without enumeration. Skip trivial groups (order <= 1)
    # so the existing "None means no symmetry" convention holds.
    if a._known_kind is not None and a._known_kind == b._known_kind:
        if a.order() <= 1:
            return None
        return a
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


def setwise_stabilizer(
    group: SymmetryGroup | None,
    fixed_set: Iterable[int],
) -> SymmetryGroup | None:
    """Return the subgroup G' = {π ∈ G : π(fixed_set) = fixed_set}.

    `fixed_set` is interpreted as tensor-axis indices; elements not in
    ``group.axes`` are silently filtered out. Returns ``None`` if the
    stabilizer is trivial (order ≤ 1).
    """
    if group is None:
        return None
    axes = group.axes
    if axes is None:
        axes = tuple(range(group.degree))
    # Translate tensor-axis indices to internal degree indices.
    internal = {axes.index(a) for a in fixed_set if a in axes}
    result = group.setwise_stabilizer(internal)
    return result if result.order() > 1 else None


def group_orbits_on_axes(
    group: SymmetryGroup,
    axes: Sequence[int],
) -> list[set[int]]:
    """Return the orbits of `group`'s action on the given tensor `axes`.

    Axes not acted on by `group` are returned as singleton orbits. Output
    order is deterministic (axes appear in their first-encounter order).
    """
    axis_list = list(axes)
    group_axes = group.axes
    if group_axes is None:
        group_axes = tuple(range(group.degree))
    # Map: tensor-axis -> set of tensor-axes reachable by any generator.
    # For axes outside group_axes, the orbit is just itself.
    seen: set[int] = set()
    orbits: list[set[int]] = []
    for a in axis_list:
        if a in seen:
            continue
        if a not in group_axes:
            orbits.append({a})
            seen.add(a)
            continue
        orbit: set[int] = set()
        frontier = {a}
        while frontier:
            x = frontier.pop()
            if x in orbit:
                continue
            orbit.add(x)
            local_x = group_axes.index(x)
            for generator in group.generators:
                local_y = generator.array_form[local_x]
                y = group_axes[local_y]
                if y not in orbit:
                    frontier.add(y)
        orbits.append(orbit)
        seen |= orbit
    return orbits


def _normalize_reps_for_output(reps, *, output_ndim: int) -> tuple[int, ...]:
    """Normalize `reps` arg to a tuple of length `output_ndim`.

    Matches NumPy.tile's right-alignment rule: if `reps` is shorter, it's
    prepended with 1s. If `reps` is a scalar, treat as `(reps,)`.
    """
    if isinstance(reps, int):
        reps_tup = (reps,)
    else:
        reps_tup = tuple(reps)
    if len(reps_tup) < output_ndim:
        reps_tup = (1,) * (output_ndim - len(reps_tup)) + reps_tup
    return reps_tup


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

    # Fast path for known-kind groups: compute the reduced kind directly
    # and route through the appropriate factory. Avoids _dimino entirely.
    if group._known_kind is not None:
        reduced_kind = _reduced_kind(
            group._known_kind, reduced_axes=axes_set, axis_map=old_to_new
        )
        if reduced_kind is not None:
            return _build_from_kind(reduced_kind)
        # Fall through to the generic path if the kind can't be reduced
        # in closed form (e.g. partial reduction of a direct_product child
        # whose own kind doesn't survive).

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
    """Attach a symmetry claim to data whose contents have NOT been checked.

    This is the untrusted spelling: it verifies only that the group's axes
    fit the array's rank, then hands the buffer to the validating
    constructor, which checks the contents and bills for doing so. Nothing
    in this package calls it -- internal transforms use
    :func:`wrap_with_derived_symmetry` -- so a caller reaching it is making
    a fresh claim about data the library has never inspected, and pays the
    same price as :func:`flopscope.as_symmetric` for the privilege.
    """
    array = np.asarray(data)
    if symmetry is None:
        return array
    validate_symmetry_group(symmetry, ndim=array.ndim)
    from flopscope._symmetric import SymmetricTensor

    return SymmetricTensor(array, symmetry=symmetry)


def wrap_with_trusted_symmetry(data, symmetry: SymmetryGroup | None):
    """Attach symmetry metadata without validating or charging.

    The single trusted attachment point in the package: this function's code
    object is the one ``SymmetricTensor.__new__`` recognizes, so trust is
    anchored to something a caller cannot fabricate (a copy of this function
    defined elsewhere gets a different code object and is not trusted). The
    two wrappers below route through it rather than constructing directly,
    which is what lets them inherit that trust without widening it.

    Only call this where the symmetry is already established: derived by an
    algebraic transform of an already-tagged tensor, or correct by
    construction. It never looks at the buffer.
    """
    array = np.asarray(data)
    if symmetry is None:
        return array
    from flopscope._symmetric import SymmetricTensor

    return SymmetricTensor(array, symmetry=symmetry)


def wrap_with_derived_symmetry(data, symmetry: SymmetryGroup | None):
    """Attach symmetry carried over from an already-tagged input.

    For array transforms -- reshape, transpose, split, concatenate and the
    rest -- whose output symmetry a ``_symmetry_transport`` helper computed
    from the input's own validated group. The claim is inherited rather than
    fresh, so it is not re-checked or re-billed; the structural check below
    only confirms the transported group still fits the new rank.
    """
    array = np.asarray(data)
    if symmetry is None:
        return array
    validate_symmetry_group(symmetry, ndim=array.ndim)
    return wrap_with_trusted_symmetry(array, symmetry)


def wrap_with_inferred_symmetry(data, symmetry: SymmetryGroup | None):
    """Wrap data with auto-inferred symmetry metadata.

    Identical to :func:`wrap_with_trusted_symmetry` except the resulting
    array carries ``_symmetry_inferred = True``. Read by
    ``_prepare_symmetric_out`` to decide whether a non-symmetric ``out=``
    write should silently downgrade the target (inferred) or raise
    (explicit). Internal call sites only — never expose to user code.
    """
    array = np.asarray(data)
    if symmetry is None:
        return array
    obj = wrap_with_trusted_symmetry(array, symmetry)
    obj._symmetry_inferred = True
    return obj
