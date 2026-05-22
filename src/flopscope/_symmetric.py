"""Symmetric tensor support: SymmetricTensor, as_symmetric, and helpers."""

from __future__ import annotations

import warnings

import numpy as np

from flopscope._config import get_setting
from flopscope._ndarray import FlopscopeArray, _asplainflopscope
from flopscope._perm_group import SymmetryGroup
from flopscope._symmetry_utils import (
    broadcast_group,
    inserted_axes_symmetry,
    intersect_groups,
    normalize_symmetry_input,
    reduce_group,
    remap_group_axes,
    restrict_group_to_axes,
    validate_symmetry_group,
)
from flopscope.errors import SymmetryError, SymmetryLossWarning

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_symmetry(
    data: np.ndarray,
    axis_groups: list[tuple[int, ...]],
) -> None:
    """Validate that *data* has the claimed symmetry.

    For each group, checks that all dims have equal sizes and that all
    pairwise transpositions are satisfied within tolerance.

    Raises
    ------
    SymmetryError
        If the data is not symmetric along the claimed axes.
    """
    for group in axis_groups:
        if len(group) < 2:
            continue
        # Check equal sizes.
        sizes = [data.shape[d] for d in group]
        if len(set(sizes)) != 1:
            raise SymmetryError(axes=group, max_deviation=float("inf"))
        # Check pairwise transpositions.
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                axes = list(range(data.ndim))
                axes[group[i]], axes[group[j]] = axes[group[j]], axes[group[i]]
                transposed = data.transpose(axes)
                if not np.allclose(data, transposed, atol=1e-6, rtol=1e-5):
                    max_dev = float(np.max(np.abs(data - transposed)))
                    raise SymmetryError(axes=group, max_deviation=max_dev)


def symmetrize(
    data: np.ndarray,
    *,
    symmetry,
) -> SymmetricTensor:
    """Project an array onto the invariant subspace of a permutation group.

    This applies Reynolds symmetrization:

    ``R_G(T) = (1 / |G|) * sum_{g in G} g · T``

    Parameters
    ----------
    data : array_like
        Input array to project.
    symmetry : SymmetryGroup
        Symmetry group to average over. If ``symmetry.axes`` is ``None``, axes are
        interpreted as ``tuple(range(symmetry.degree))``.

    Returns
    -------
    SymmetricTensor
        The projected tensor, validated and wrapped as a :class:`SymmetricTensor`.

    Raises
    ------
    SymmetryError
        If ``data`` has incompatible dimensions for ``group`` axes or if the
        projected result cannot be validated as symmetric for ``group``.

    Notes
    -----
    ``symmetrize`` performs exact Reynolds averaging internally and then delegates
    to :func:`as_symmetric` so the result participates in downstream symmetry
    tracking and validation in the usual way.

    Estimated FLOP cost is approximately:

    ``|G| * n_elem + n_elem`` for projection, plus validation from
    ``as_symmetric``:

    - ``|G|`` transposed add passes over ``n_elem`` elements
    - one final scaling pass
    - symmetry checks inside validation

    where ``|G|`` is the group order and ``n_elem = data.size``.

    The canonical pattern for generating random data with symmetry is:

    ``fnp.random.symmetric(shape, symmetry_group, distribution=...)``.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>> data = fnp.random.randn(4, 4)
    >>> S = flops.symmetrize(data, flops.SymmetryGroup.symmetric(axes=(0, 1)))
    >>> S.is_symmetric((0, 1))
    True
    """
    array = np.asarray(data)
    group = _resolve_symmetry_argument(
        array,
        symmetry=symmetry,
    )
    assert group is not None  # required=True raises if symmetry is None
    validate_symmetry_group(group, ndim=array.ndim, shape=array.shape)
    group_axes = group.axes if group.axes is not None else tuple(range(group.degree))
    symmetrized = np.zeros_like(array, dtype=np.result_type(array, np.float64))

    for g in group.elements():
        perm = list(range(array.ndim))
        for local_idx, tensor_axis in enumerate(group_axes):
            perm[tensor_axis] = group_axes[g.array_form[local_idx]]
        symmetrized = symmetrized + np.transpose(array, perm)

    return as_symmetric(symmetrized / group.order(), symmetry=group)


def validate_symmetry_groups(data: np.ndarray, groups: list) -> None:
    """Validate that *data* is symmetric under the given SymmetryGroups.

    Raises
    ------
    ValueError
        If a group has no axes set.
    SymmetryError
        If the data is not symmetric under the claimed group.
    """
    for group in groups:
        axes = group.axes
        if axes is None:
            axes = tuple(range(group.degree))
            group._axes = axes
        validate_symmetry_group(group, ndim=data.ndim, shape=data.shape)
        for orbit in group.orbits():
            sizes = {data.shape[axes[i]] for i in orbit}
            if len(sizes) != 1:
                raise SymmetryError(
                    axes=tuple(axes[i] for i in orbit), max_deviation=float("inf")
                )
        for gen in group.generators:
            if gen.is_identity:
                continue
            perm = list(range(data.ndim))
            for i in range(group.degree):
                perm[axes[i]] = axes[gen._array_form[i]]
            transposed = data.transpose(perm)
            if not np.allclose(data, transposed, atol=1e-6, rtol=1e-5):
                max_dev = float(np.max(np.abs(data - transposed)))
                raise SymmetryError(axes=tuple(axes), max_deviation=max_dev)


def _resolve_symmetry_argument(
    data: np.ndarray,
    *,
    symmetry,
    required: bool = True,
):
    if symmetry is None:
        if required:
            raise ValueError("symmetry must be provided")
        return None
    return normalize_symmetry_input(symmetry, ndim=np.asarray(data).ndim)


def is_symmetric(
    data: np.ndarray,
    *,
    symmetry,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> bool:
    """Check whether *data* is invariant under the given symmetry.

    Parameters
    ----------
    data : numpy.ndarray
        The array to test.
    symmetry : SymmetryGroup or array-like specification
        Symmetry to verify, normalized via :func:`normalize_symmetry_input`.
    atol : float, optional
        Absolute tolerance used by :func:`numpy.allclose`. Default ``1e-6``.
    rtol : float, optional
        Relative tolerance used by :func:`numpy.allclose`. Default ``1e-5``.

    Returns
    -------
    bool
        ``True`` if *data* is invariant under every group element, otherwise
        ``False``.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>> matrix = fnp.array([[1.0, 2.0], [2.0, 3.0]])
    >>> flops.is_symmetric(
    ...     matrix, symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1))
    ... )
    True
    """
    group = _resolve_symmetry_argument(
        data,
        symmetry=symmetry,
        required=False,
    )
    if group is None:
        return False

    array = np.asarray(data)
    group_axes = group.axes if group.axes is not None else tuple(range(group.degree))
    validate_symmetry_group(group, ndim=array.ndim, shape=array.shape)

    for elem in group.elements():
        axes = list(range(array.ndim))
        for src_local, dst_local in enumerate(elem.array_form):
            axes[group_axes[src_local]] = group_axes[dst_local]
        if not np.allclose(array, array.transpose(axes), atol=atol, rtol=rtol):
            return False
    return True


# ---------------------------------------------------------------------------
# Symmetry-loss warning helper
# ---------------------------------------------------------------------------


def _warn_symmetry_loss(
    lost_dims: list[tuple[int, ...]],
    reason: str,
    stacklevel: int = 3,
) -> None:
    """Emit a :class:`SymmetryLossWarning` if warnings are enabled."""
    if not get_setting("symmetry_warnings"):
        return
    dim_str = ", ".join(str(g) for g in lost_dims)
    warnings.warn(
        f"Symmetry lost along dims {dim_str}: {reason}. "
        "Use as_symmetric() to re-tag if you know the result is symmetric. "
        "Suppress with flops.configure(symmetry_warnings=False).",
        SymmetryLossWarning,
        stacklevel=stacklevel,
    )


# ---------------------------------------------------------------------------
# Symmetry propagation helpers
# ---------------------------------------------------------------------------


def propagate_symmetry_slice(
    groups: list[SymmetryGroup],
    shape: tuple[int, ...],
    key,
) -> list[SymmetryGroup] | None:
    """Compute new symmetry groups after ``__getitem__(key)``.

    Parameters
    ----------
    groups : list of SymmetryGroup
        Each group has ``axes`` indicating which tensor dimensions it acts on.
    shape : tuple of int
        Original tensor shape.
    key : indexing key
        The slicing/indexing key.

    Returns
    -------
    list of SymmetryGroup or None
        Surviving groups, or ``None`` if no symmetry survives.
    """
    ndim = len(shape)

    if not isinstance(key, tuple):
        key = (key,)

    for k in key:
        # Bool indices are boolean masks in numpy (add a size-1 batch axis for
        # True / size-0 for False), not integer scalars. `isinstance(True, int)`
        # is also True in Python, so we must check bool BEFORE the int branch
        # below would silently misclassify it as an integer scalar index.
        if isinstance(k, (np.ndarray, list, bool, np.bool_)):
            return None

    # Expand Ellipsis.
    expanded: list = []
    ellipsis_seen = False
    for k in key:
        if k is Ellipsis:
            if ellipsis_seen:
                raise IndexError("only one Ellipsis allowed")
            ellipsis_seen = True
            n_newaxis_in_key = sum(1 for kk in key if kk is None)
            n_explicit = len(key) - 1 - n_newaxis_in_key
            n_fill = ndim - n_explicit
            expanded.extend([slice(None)] * n_fill)
        else:
            expanded.append(k)
    if not ellipsis_seen:
        n_newaxis = sum(1 for k in expanded if k is None)
        while len(expanded) - n_newaxis < ndim:
            expanded.append(slice(None))
    key_expanded = expanded

    # Classify each original dim.
    old_dim_idx = 0
    dim_actions: dict[int, str | tuple] = {}

    for k in key_expanded:
        if k is None:
            continue
        if old_dim_idx >= ndim:
            break
        if isinstance(k, (int, np.integer)):
            dim_actions[old_dim_idx] = "removed"
            old_dim_idx += 1
        elif isinstance(k, slice):
            start, stop, step = k.indices(shape[old_dim_idx])
            if start == 0 and stop == shape[old_dim_idx] and step == 1:
                dim_actions[old_dim_idx] = "untouched"
            else:
                new_size = max(
                    0, (stop - start + (step - (1 if step > 0 else -1))) // step
                )
                dim_actions[old_dim_idx] = ("resized", new_size)
            old_dim_idx += 1
        else:
            return None

    while old_dim_idx < ndim:
        dim_actions[old_dim_idx] = "untouched"
        old_dim_idx += 1

    # Build old→new dim mapping.
    removed_dims = {d for d, a in dim_actions.items() if a == "removed"}
    old_to_new: dict[int, int | None] = {}
    newaxis_positions: list[int] = []
    orig_idx = 0
    for k in key_expanded:
        if k is None:
            newaxis_positions.append(orig_idx)
        else:
            orig_idx += 1

    new_idx = 0
    for d in range(ndim):
        while newaxis_positions and newaxis_positions[0] <= d:
            newaxis_positions.pop(0)
            new_idx += 1
        if d in removed_dims:
            old_to_new[d] = None
        else:
            old_to_new[d] = new_idx
            new_idx += 1

    # Process each group.
    new_groups: list[SymmetryGroup] = []
    for group in groups:
        axes = group.axes
        if axes is None:
            continue

        # Map tensor axes to group-local indices.
        local_removed: set[int] = set()
        local_kept: list[int] = []
        for local_idx, tensor_dim in enumerate(axes):
            action = dim_actions.get(tensor_dim, "untouched")
            if action == "removed":
                local_removed.add(local_idx)
            else:
                local_kept.append(local_idx)

        if not local_kept:
            continue

        if any(
            dim_actions.get(axes[local_idx], "untouched") != "untouched"
            for local_idx in local_kept
        ):
            continue

        # Pointwise stabilizer: each removed axis must map to itself.
        # (Setwise would only be valid when all removed axes share the
        # same slice value, which we can't determine in general.)
        stab = group.pointwise_stabilizer(local_removed)

        # Restrict to kept local indices.
        kept_tuple = tuple(local_kept)
        if len(kept_tuple) < 2:
            continue

        restricted = restrict_group_to_axes(stab, tuple(axes[k] for k in kept_tuple))
        if restricted is None:
            continue

        final = remap_group_axes(
            restricted,
            {axes[k]: old_to_new[axes[k]] for k in kept_tuple},  # type: ignore[arg-type]
        )
        if final is None:
            continue
        new_groups.append(final)

    # Build the inserted-axis group from freshly-inserted None positions.
    inserted_output_positions: list[int] = []
    out_idx = 0
    for k in key_expanded:
        if k is None:
            inserted_output_positions.append(out_idx)
            out_idx += 1
        elif isinstance(k, (int, np.integer)):
            pass  # axis removed; no output slot
        else:
            out_idx += 1

    inserted_group = inserted_axes_symmetry(inserted_output_positions)
    if inserted_group is not None:
        new_groups.append(inserted_group)

    return new_groups if new_groups else None


def propagate_symmetry_reduce(
    groups: list[SymmetryGroup],
    ndim: int,
    axis: int | tuple[int, ...] | None,
    keepdims: bool = False,
) -> list[SymmetryGroup] | None:
    """Compute new symmetry groups after a reduction.

    Parameters
    ----------
    groups : list of SymmetryGroup
        Each group has ``axes`` indicating which tensor dimensions it acts on.
    ndim : int
        Original tensor rank.
    axis : int, tuple of int, or None
        Axes being reduced.
    keepdims : bool
        Whether reduced dims are kept at size 1.

    Returns
    -------
    list of SymmetryGroup or None
        Surviving groups, or ``None`` if no symmetry survives.
    """
    new_groups: list[SymmetryGroup] = []
    for group in groups:
        reduced = reduce_group(group, ndim=ndim, axis=axis, keepdims=keepdims)
        if reduced is not None:
            new_groups.append(reduced)

    return new_groups if new_groups else None


def intersect_symmetry(
    groups_a: list[SymmetryGroup] | None,
    groups_b: list[SymmetryGroup] | None,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> list[SymmetryGroup] | None:
    """Intersect symmetry groups for binary ops, accounting for broadcasting.

    For groups acting on the same output axes, computes the element-set
    intersection.  Broadcast-stretched dimensions (size 1 → larger) are
    removed from groups before intersecting.

    Parameters
    ----------
    groups_a, groups_b : list of SymmetryGroup or None
        Symmetry groups for each operand.
    shape_a, shape_b : tuple of int
        Input shapes (before broadcasting).
    output_shape : tuple of int
        Broadcast output shape.

    Returns
    -------
    list of SymmetryGroup or None
        Groups present in both operands, or *None* if no shared symmetry.
    """
    if groups_a is None or groups_b is None:
        return None

    ndim_out = len(output_shape)

    aligned_a = [
        aligned
        for group in groups_a
        if (
            aligned := broadcast_group(
                group, input_shape=shape_a, output_shape=output_shape
            )
        )
        is not None
    ]
    aligned_b = [
        aligned
        for group in groups_b
        if (
            aligned := broadcast_group(
                group, input_shape=shape_b, output_shape=output_shape
            )
        )
        is not None
    ]

    # Intersect: for groups acting on the same output axes, compute element intersection.
    b_by_axes: dict[tuple[int, ...], SymmetryGroup] = {}
    for g in aligned_b:
        if g.axes is not None:
            b_by_axes[g.axes] = g

    intersection: list[SymmetryGroup] = []
    for ga in aligned_a:
        if ga.axes is None:
            continue
        gb = b_by_axes.get(ga.axes)
        if gb is None:
            continue
        common = intersect_groups(ga, gb, ndim=ndim_out)
        if common is not None:
            intersection.append(common)

    return intersection if intersection else None


# ---------------------------------------------------------------------------
# SymmetricTensor  (np.ndarray subclass)
# ---------------------------------------------------------------------------


def _merge_symmetry_groups(groups) -> SymmetryGroup | None:
    groups = [group for group in groups if group is not None]
    if not groups:
        return None
    if len(groups) == 1:
        return groups[0]
    return SymmetryGroup.direct_product(*groups)


def _wrap_tensor_result(data: np.ndarray, symmetry: SymmetryGroup | None):
    if symmetry is None:
        return _asplainflopscope(data)
    return SymmetricTensor(data, symmetry=symmetry)


class SymmetricTensor(FlopscopeArray):
    """An ndarray that carries symmetry metadata.

    Do not instantiate directly; use :func:`as_symmetric`.
    """

    __slots__ = ("_symmetry",)

    def __new__(
        cls,
        input_array: np.ndarray,
        *,
        symmetry: SymmetryGroup,
    ) -> SymmetricTensor:
        obj = np.asarray(input_array).view(cls)
        obj._symmetry = symmetry
        return obj

    def __array_finalize__(self, obj: object) -> None:
        self._symmetry = None

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        result = super().__array_wrap__(out_arr, context, return_scalar)
        if return_scalar:
            return result
        if isinstance(result, SymmetricTensor) and result._symmetry is None:
            return _asplainflopscope(np.asarray(result))
        return result

    # -- public API --

    @property
    def symmetry(self) -> SymmetryGroup:
        """Exact symmetry group carried by this tensor."""
        return self._symmetry  # type: ignore[return-value]

    def is_symmetric(
        self,
        *,
        symmetry=None,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> bool:
        """Check whether the data satisfies the given (or carried) symmetry."""
        group = (
            self._symmetry
            if symmetry is None
            else _resolve_symmetry_argument(
                self,
                symmetry=symmetry,
                required=False,
            )
        )
        if group is None:
            return False
        return is_symmetric(np.asarray(self), symmetry=group, atol=atol, rtol=rtol)

    # -- slicing with symmetry propagation --

    def __getitem__(self, key):  # type: ignore[override]
        """Index with symmetry propagation.

        Computes the pointwise-stabilizer subgroup for axes removed by
        integer indexing, then restricts surviving groups to the output
        axes.  Returns a plain ``ndarray`` when no symmetry survives.
        Emits :class:`~flopscope.errors.SymmetryLossWarning` on partial or
        total symmetry loss.
        """
        result = super().__getitem__(key)
        if not isinstance(result, np.ndarray) or result.ndim == 0:
            return result if not isinstance(result, np.ndarray) else np.asarray(result)

        if self._symmetry is None:
            # Even with no input symmetry, multiple inserted None axes form a
            # free symmetric group on those axes. Run the propagator with an
            # empty groups list so the inserted-axis logic still fires.
            new_groups = propagate_symmetry_slice([], self.shape, key)
            if new_groups:
                return _wrap_tensor_result(
                    np.asarray(result), _merge_symmetry_groups(new_groups)
                )
            return _asplainflopscope(np.asarray(result))

        new_groups = propagate_symmetry_slice([self._symmetry], self.shape, key)
        new_symmetry = _merge_symmetry_groups(new_groups or [])
        if new_groups is not None:
            # Fire only on real structural reduction: the new group's order is
            # strictly less than the original's. This silences false alarms on
            # operations that gain symmetry (e.g. `a[None, :, None, :]`
            # produces a richer Young group) or merely relabel axes without
            # losing structure (e.g. `a[None, :, :]` shifts axes, preserves
            # order). It under-fires on the rare case where original sym is
            # broken and a same-order new group is gained on different axes;
            # the gained group is still attached to the result so a careful
            # user can inspect `.symmetry` directly.
            if (
                new_symmetry is not None
                and self._symmetry.axes is not None
                and new_symmetry.order() < self._symmetry.order()
            ):
                _warn_symmetry_loss(
                    [self._symmetry.axes],
                    "slicing reduced symmetric group structure",
                )
            return _wrap_tensor_result(np.asarray(result), new_symmetry)

        if self._symmetry.axes is not None:
            _warn_symmetry_loss(
                [self._symmetry.axes],
                "slicing removed all symmetric dim groups",
            )
        return _asplainflopscope(np.asarray(result))

    # -- copy preserves metadata --

    def copy(self, order: str = "C") -> SymmetricTensor:  # type: ignore[override]
        out = super().copy(order=order).view(type(self))  # type: ignore[arg-type]
        out._symmetry = self._symmetry
        return out

    def reshape(self, *shape, **kwargs):  # type: ignore[override]
        return _asplainflopscope(np.reshape(np.asarray(self), *shape, **kwargs))

    def ravel(self, order: str = "C"):  # type: ignore[override]
        return _asplainflopscope(np.ravel(np.asarray(self), order=order))  # type: ignore[arg-type]

    def flatten(self, order: str = "C"):  # type: ignore[override]
        return _asplainflopscope(np.asarray(self).flatten(order))  # type: ignore[arg-type]

    def squeeze(self, axis=None):  # type: ignore[override]
        return _asplainflopscope(np.squeeze(np.asarray(self), axis=axis))

    def astype(  # type: ignore[override]
        self,
        dtype,
        order: str = "K",
        casting: str = "unsafe",
        subok: bool = False,
        copy: bool = True,
    ):
        return _asplainflopscope(
            np.asarray(self).astype(
                dtype,
                order=order,  # type: ignore[arg-type]
                casting=casting,  # type: ignore[arg-type]
                subok=False,
                copy=copy,
            )
        )

    def transpose(self, *axes):  # type: ignore[override]
        if not axes or axes == (None,):
            order = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            order = tuple(axes[0])
        else:
            order = tuple(axes)
        result = np.transpose(np.asarray(self), axes=order)
        mapping = {old: new for new, old in enumerate(order)}
        return _wrap_tensor_result(
            result,
            remap_group_axes(self._symmetry, mapping),
        )

    def swapaxes(self, axis1: int, axis2: int):  # type: ignore[override]
        order = list(range(self.ndim))
        axis1 %= self.ndim
        axis2 %= self.ndim
        order[axis1], order[axis2] = order[axis2], order[axis1]
        return self.transpose(tuple(order))

    @property
    def T(self):
        return self.transpose()

    # -- pickling --

    def __reduce__(self):
        pickled_state = super().__reduce__()
        return (
            pickled_state[0],
            pickled_state[1],
            pickled_state[2] + (self._symmetry,),  # type: ignore[operator]
        )

    def __setstate__(self, state):
        if len(state) < 2 or not isinstance(state[-1], SymmetryGroup):
            raise ValueError("legacy symmetry payloads are not supported")
        super().__setstate__(state[:-1])
        self._symmetry = state[-1]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def as_symmetric(
    data: np.ndarray,
    *,
    symmetry,
) -> SymmetricTensor:
    """Wrap *data* as a :class:`SymmetricTensor` after validating symmetry.

    Parameters
    ----------
    data : numpy.ndarray
        The tensor data.
    symmetry : SymmetryGroup or shorthand
        Exact symmetry input accepted by :func:`normalize_symmetry_input`.

    Returns
    -------
    SymmetricTensor
        View of ``data`` carrying validated symmetry metadata.

    Raises
    ------
    SymmetryError
        If the data does not satisfy the claimed symmetry.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>>
    >>> matrix = fnp.array([[1.0, 2.0], [2.0, 3.0]])
    >>> tagged = flops.as_symmetric(matrix, (0, 1))
    >>> tagged.symmetric_axes
    [(0, 1)]
    """
    group = _resolve_symmetry_argument(
        data,
        symmetry=symmetry,
    )
    assert group is not None  # required=True raises if symmetry is None
    array = np.asarray(data)
    validate_symmetry_groups(array, [group])
    return SymmetricTensor(array, symmetry=group)
