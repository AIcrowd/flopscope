"""Symmetric tensor support: SymmetricTensor, as_symmetric, and helpers."""

from __future__ import annotations

import sys as _sys

import numpy as np

from flopscope._budget import _counted_wrapper
from flopscope._canonical_symmetry import canonical_copy, canonicalize
from flopscope._dtype_billing import integer_to_float64_min_dtype
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
    wrap_with_trusted_symmetry,
)
from flopscope._validation import require_budget
from flopscope._write_epoch import epoch_of
from flopscope.errors import (
    _SYMMETRY_DOCS_PATH,
    SymmetryError,
    _docs_url,
)

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


def _validation_compute_dtype(dtype: np.dtype) -> np.dtype:
    """Compute dtype of a symmetry check: the tolerance comparison's own width.

    Both billed validation sites charge ``k * (7n - 1)`` -- allclose's cost
    formula, once per non-identity generator -- because the work they do is
    ``np.allclose(array, array.transpose(perm))``. numpy's allclose casts its
    reference operand to ``result_type(y, 1.)`` before comparing, so an
    integer or bool array is compared in float64 however narrow it is. Billing
    the array's own dtype charged bool and int8..uint32 half of what the
    comparison costs. The result dtype cannot catch this: is_symmetric returns
    a Python bool, and as_symmetric returns a tensor of the operand dtype.
    """
    return integer_to_float64_min_dtype(np.dtype(dtype))


def _nonidentity_generator_count(group) -> int:
    """Number of non-identity generators (the count is what validation iterates)."""
    return sum(1 for gen in group.generators if not gen.is_identity)


#: Symmetrization strategies accepted by :func:`symmetrize`'s ``mode``.
_SYMMETRIZE_MODES = frozenset({"reynolds-projection", "canonical-copy"})


def _require_enumerable_for_reynolds(group) -> None:
    """Refuse a Reynolds projection over a group that cannot be enumerated.

    Every other consumer of the enumeration budget can degrade to a dense
    cost and carry on, because for them the group is only an accounting
    detail. Reynolds averaging is the one place where enumerating the group
    IS the computation, so there is nothing to degrade to -- and the
    ``canonical-copy`` mode, which reads the generators alone, is the way
    through.
    """
    from flopscope._config import get_setting
    from flopscope._perm_group import _DiminoBudgetExceeded

    budget = int(get_setting("dimino_budget"))  # type: ignore[arg-type]
    try:
        order = group.order()
    except _DiminoBudgetExceeded as exc:
        seen, budget = exc.seen_count, exc.budget
    else:
        if order <= budget:
            return
        seen = order
    raise ValueError(
        f"symmetrize(mode='reynolds-projection') averages over every element "
        f"of this symmetry group, and enumerating it exceeded dimino_budget "
        f"({seen} > {budget}). Use mode='canonical-copy', which reads the "
        f"group's generators alone and never enumerates it, billing "
        f"numel(data) instead of (|G| + 1) * numel(data). It gives a "
        f"different result, not a cheaper route to the same one: each orbit "
        f"keeps its lexicographically first entry instead of averaging the "
        f"orbit. (flops.configure(dimino_budget=...) raises the limit for "
        f"in-process runs only.) "
        f"See: {_docs_url(_SYMMETRY_DOCS_PATH)}"
    ) from None


def _project_core(array, group):
    """Raw Reynolds projection. UNCOUNTED. Returns an ndarray.

    R_G(T) = (1/|G|) * sum_{g in G} g·T  — |G| transposed adds + one scaling pass.
    """
    array = np.asarray(array)
    group_axes = group.axes if group.axes is not None else tuple(range(group.degree))
    symmetrized = np.zeros_like(array, dtype=np.result_type(array, np.float64))
    for g in group.elements():
        perm = list(range(array.ndim))
        for local_idx, tensor_axis in enumerate(group_axes):
            perm[tensor_axis] = group_axes[g.array_form[local_idx]]
        symmetrized = symmetrized + np.transpose(array, perm)
    return symmetrized / group.order()


def _check_generators(array, group, *, atol: float = 1e-6, rtol: float = 1e-5) -> bool:
    """Return True iff *array* is invariant under every non-identity generator.

    Checking generators is sufficient for whole-group invariance. UNCOUNTED.
    Mirrors the generator loop in ``validate_symmetry_groups``.
    """
    array = np.asarray(array)
    axes = group.axes if group.axes is not None else tuple(range(group.degree))
    for gen in group.generators:
        if gen.is_identity:
            continue
        perm = list(range(array.ndim))
        for i in range(group.degree):
            perm[axes[i]] = axes[gen._array_form[i]]
        if not np.allclose(array, array.transpose(perm), atol=atol, rtol=rtol):
            return False
    return True


@_counted_wrapper
def symmetrize(
    data: np.ndarray,
    *,
    symmetry,
    mode: str = "reynolds-projection",
) -> SymmetricTensor:
    """Make an array invariant under a permutation group.

    Two modes, differing in whether the discarded entries get a vote.
    ``"reynolds-projection"`` (the default) averages each orbit:

    ``R_G(T) = (1 / |G|) * sum_{g in G} g · T``

    ``"canonical-copy"`` instead keeps one entry per orbit -- the one at the
    lexicographically smallest index -- and copies it over the rest. Every
    other value in the orbit is discarded rather than mixed in, which is what
    makes it the right choice when the input is not trusted: nothing a caller
    hid in the redundant positions can reach the result. It is also the
    cheaper of the two, being one copy pass rather than ``|G|`` transposed
    adds, and it never enumerates the group.

    Parameters
    ----------
    data : array_like
        Input array to symmetrize.
    symmetry : SymmetryGroup
        Symmetry group to average over. If ``symmetry.axes`` is ``None``, axes are
        interpreted as ``tuple(range(symmetry.degree))``.
    mode : {"reynolds-projection", "canonical-copy"}, optional
        Which symmetrization to apply. Defaults to ``"reynolds-projection"``.

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
    ``"reynolds-projection"`` performs exact Reynolds averaging internally,
    billing ``(|G| + 1) * numel(data)`` FLOPs:

    - ``|G|`` transposed add passes over ``numel`` elements
    - one final scaling pass (divide by ``|G|``)

    Internal validation runs but is NOT billed (decision D1).

    where ``|G|`` is the group order and ``numel = data.size``.

    ``"canonical-copy"`` bills ``numel(data)`` -- one write per output
    element, the same rate as every other materializing copy (``copy``,
    ``take``, ``repeat``). It needs no validation pass because its output is
    invariant by construction, and it never enumerates ``|G|``: the orbit map
    is built from the group's generators and cached per
    ``(shape, group action)``.

    The two modes also differ in dtype. Averaging must divide, so
    ``"reynolds-projection"`` accumulates in
    ``result_type(data, float64)`` -- a ``float32`` input comes back
    ``float64``. ``"canonical-copy"`` only moves values, so it preserves the
    input dtype exactly, including integer, boolean and complex types.

    The canonical pattern for generating random data with symmetry is:

    ``fnp.random.symmetric(shape, symmetry_group, distribution=...)``.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>> data = fnp.random.randn(4, 4)
    >>> S = flops.symmetrize(data, symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)))
    >>> S.is_symmetric((0, 1))
    True
    """
    if mode not in _SYMMETRIZE_MODES:
        raise ValueError(
            f"unknown symmetrize mode {mode!r}; "
            f"expected one of {', '.join(map(repr, sorted(_SYMMETRIZE_MODES)))}"
        )
    array = np.asarray(data)
    group = _resolve_symmetry_argument(array, symmetry=symmetry)
    assert group is not None  # required=True raises if symmetry is None
    validate_symmetry_group(group, ndim=array.ndim, shape=array.shape)
    n = array.size
    if mode == "canonical-copy":
        # One write per output element -- the rate every other materializing
        # copy pays (copy/take/repeat). Deliberately does NOT consult
        # group.order(): enumerating the group is the cost this mode exists
        # to avoid, and the orbit map only ever needs the generators.
        cost = max(n, 1)
        # A gather moves values without computing any, so the output dtype is
        # the input's -- no float64 sentinel, unlike the averaging branch.
        dtypes = (array.dtype,)
    else:
        # Averaging visits every group element, so a group too large to
        # enumerate cannot be projected at all. Refuse here, above the
        # deduct, so a call that cannot finish is not charged for trying:
        # for a group whose order is known in closed form the cost is
        # computable, and without this check the caller would be billed the
        # full Reynolds price and only then hit the enumeration limit.
        _require_enumerable_for_reynolds(group)
        cost = max((group.order() + 1) * n, 1)
        # _project_core always accumulates in np.result_type(array, float64) --
        # the "/ group.order()" scaling pass needs float precision even from
        # float32 input (verified: symmetrize(float32).dtype == float64,
        # symmetrize(complex64).dtype == complex128) -- so the float64 sentinel
        # must join the resolve rather than replace it (result_type preserves
        # kind: result_type(complex64, float64) == complex128).
        dtypes = (array.dtype, np.dtype(np.float64))
    budget = require_budget()
    with budget.deduct(
        "symmetrize",
        flop_cost=cost,
        subscripts=None,
        shapes=(array.shape,),
        dtypes=dtypes,
    ):
        if mode == "canonical-copy":
            # Exactly invariant by construction, so unlike the averaging
            # branch there is no residual rounding to validate away.
            return SymmetricTensor._construct_trusted(
                canonical_copy(array, group), symmetry=group
            )
        projected = _project_core(array, group)
        # D1: internal validation runs but is NOT billed — build the tensor
        # directly rather than calling the (later-counted) as_symmetric.
        # Uses the trusted, non-revalidating constructor so this validated-
        # but-uncharged pass isn't repeated (and charged for) by
        # SymmetricTensor's public constructor.
        validate_symmetry_groups(projected, [group])
        return SymmetricTensor._construct_trusted(projected, symmetry=group)


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


def _validate_and_charge_symmetry(
    array: np.ndarray,
    group: SymmetryGroup,
    *,
    op_name: str = "as_symmetric",
) -> None:
    """Validate *array* against *group* and bill the same rate as
    :func:`as_symmetric` (``k * (7n - 1)``, ``k`` = non-identity generators).

    A symmetry tag is a billing claim about buffer CONTENTS: whoever calls
    this helper pays to have that claim checked. Shared by
    :func:`as_symmetric` and :class:`SymmetricTensor`'s public constructor
    (``__new__``, for a bare top-level call) so those two paths can never
    disagree about what counts as symmetric -- but NOT every path that can
    end up attaching a tag calls this helper; see the "WHAT THIS DOES NOT
    CLOSE" comment on ``SymmetricTensor.__new__`` for the two that don't.
    Raises :class:`SymmetryError` (via :func:`validate_symmetry_groups`) if
    the claim is false.
    """
    n = array.size
    k = _nonidentity_generator_count(group)
    cost = max(k * (7 * n - 1), 1)
    budget = require_budget()
    with budget.deduct(
        op_name,
        flop_cost=cost,
        subscripts=None,
        shapes=(array.shape,),
        dtypes=(_validation_compute_dtype(array.dtype),),
    ):
        validate_symmetry_groups(array, [group])


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


@_counted_wrapper
def is_symmetric(
    data: np.ndarray,
    *,
    symmetry,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> bool:
    """Check whether *data* is invariant under the given symmetry.

    Checks the group's generators rather than all elements; this is
    mathematically equivalent for any well-formed group.

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
        ``True`` if *data* is invariant under every non-identity generator,
        otherwise ``False``.

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
    group = _resolve_symmetry_argument(data, symmetry=symmetry, required=False)
    if group is None:
        return False
    array = np.asarray(data)
    validate_symmetry_group(group, ndim=array.ndim, shape=array.shape)
    n = array.size
    k = _nonidentity_generator_count(group)
    cost = max(k * (7 * n - 1), 1)
    budget = require_budget()
    with budget.deduct(
        "is_symmetric",
        flop_cost=cost,
        subscripts=None,
        shapes=(array.shape,),
        dtypes=(_validation_compute_dtype(array.dtype),),
    ):
        return _check_generators(array, group, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Symmetry-loss warning helper
# ---------------------------------------------------------------------------

from flopscope.errors import (  # noqa: E402
    _warn_symmetry_loss,
)  # re-exported for back-compat

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
    # symmetry here is DERIVED (propagated from an already-validated tensor
    # via slicing/transpose/etc.), not a caller-supplied claim over fresh
    # data, so it goes through the trusted, non-revalidating constructor --
    # matching the rule that views must not be charged.
    return SymmetricTensor._construct_trusted(data, symmetry=symmetry)


# Trust is anchored to ONE code object: `wrap_with_trusted_symmetry`
# (flopscope._symmetry_utils). A caller cannot fabricate it -- a copy of that
# function defined elsewhere compiles to a different code object, because
# `co_filename` participates in equality -- which is why trust is keyed on
# identity here rather than on an argument the caller could set.
#
# The package's 39 internal attachment sites all reach it: the 27 array
# transforms in _array_ops.py go through `wrap_with_derived_symmetry`, the
# constant fills through `wrap_with_inferred_symmetry`, and both delegate to
# the trusted wrapper rather than constructing directly, so they inherit
# that trust without widening the set. `wrap_with_symmetry` is deliberately
# NOT in here: nothing internal calls it, so a caller reaching it is making
# a fresh claim about unexamined data and routes through the validating,
# charging constructor like any other untrusted ingress.
#
# The set still exists for exactly one site: `_build_symmetric_proxy`
# (_accumulation/_cost.py), which tags an uninitialized `np.empty` scratch
# buffer that the cost model only ever reads for shape and symmetry. It is
# the sole in-package caller that can run outside a `@_counted_wrapper`
# frame, so it is the only thing keeping this mechanism alive -- worth
# knowing before anyone deletes it as dead weight.
_TRUSTED_SYMMETRY_WRAPPER_CODES = frozenset({wrap_with_trusted_symmetry.__code__})


class SymmetricTensor(FlopscopeArray):
    """An ndarray that carries symmetry metadata.

    Do not instantiate directly; use :func:`as_symmetric`.
    """

    __slots__ = (
        "_symmetry_raw",
        "_symmetry_inferred",
        "_symmetry_epoch",
    )

    def __new__(
        cls,
        input_array: np.ndarray,
        *,
        symmetry: SymmetryGroup,
    ) -> SymmetricTensor:
        # A tag is a billing claim about buffer CONTENTS, so a bare,
        # top-level `SymmetricTensor(data, symmetry=...)` is checked and
        # charged exactly like `as_symmetric` -- same validator, same price,
        # same `SymmetryError` on a false claim -- and then canonicalized, so
        # the tag it mints asserts no more than the data supports.
        #
        # Validation is skipped on two trusted routes. First, an immediate
        # caller of `wrap_with_trusted_symmetry`, the package's single
        # trusted attachment point (see `_TRUSTED_SYMMETRY_WRAPPER_CODES`
        # above). Second, construction from inside another flopscope op's
        # `@_counted_wrapper` frame: many sites in this package (pointwise,
        # einsum, solvers, random.symmetric, accumulation) tag a result whose
        # symmetry they have already established mathematically -- exp() of a
        # symmetric input really is symmetric -- and a cost-estimation proxy
        # over uninitialized memory needs the metadata rather than a real
        # check (see `_accumulation/_cost.py`'s `_build_symmetric_proxy`).
        #
        # KNOWN GAP (pinned as an expected failure in
        # `tests/test_symmetric_tensor_new_validation.py`): `_called_from_wrapper`
        # walks the ENTIRE call stack for a `@_counted_wrapper` frame, so a
        # tensor constructed inside a participant callback that a counted host
        # op invokes (`fnp.apply_along_axis`, `fnp.piecewise`) inherits the
        # host's trust. Closing it needs the walk to stop at the callback
        # boundary rather than pass through it. Not reachable on the graded
        # backend, where those ops refuse a callback over the wire.
        from flopscope._budget import _called_from_wrapper

        array = np.asarray(input_array)
        trusted = (
            _sys._getframe(1).f_code in _TRUSTED_SYMMETRY_WRAPPER_CODES
            or _called_from_wrapper()
        )
        if not trusted:
            _validate_and_charge_symmetry(array, symmetry, op_name="as_symmetric")
            # Same trust boundary as `as_symmetric`, so the same rule: the
            # tolerant check authorizes a tag the cost model reads as exact,
            # and canonicalizing here is what makes those two agree. Honest
            # data is returned untouched, so this stays a view.
            array = canonicalize(array, symmetry)
        obj = array.view(cls)
        obj._symmetry = symmetry
        obj._symmetry_inferred = False
        return obj

    @classmethod
    def _construct_trusted(
        cls,
        input_array: np.ndarray,
        *,
        symmetry: SymmetryGroup | None,
    ) -> SymmetricTensor:
        """Wrap data as a SymmetricTensor without validating or charging.

        Internal-only bypass around the public, validating ``__new__``
        above. Used exclusively by code within this module that has
        already established the symmetry claim itself: ``as_symmetric``
        and ``symmetrize``'s Reynolds-projected output (both validated
        inline just above their own call sites, per decision D1 for the
        latter) and ``_wrap_tensor_result``, which propagates symmetry
        DERIVED from an already-validated tensor via slicing/transpose --
        never a fresh, caller-supplied claim. Do not call this from
        outside the module: it is exactly the bypass the public
        constructor exists to close.
        """
        obj = np.asarray(input_array).view(cls)
        obj._symmetry = symmetry
        obj._symmetry_inferred = False
        return obj

    def __array_finalize__(self, obj: object) -> None:
        self._symmetry = None
        self._symmetry_inferred = False

    # The tag is a billing claim about buffer contents, so it is stamped with
    # the buffer's write count and voided once that count moves. Gating the
    # storage rather than the public ``symmetry`` property covers the internal
    # ``self._symmetry`` reads too, so a voided tag cannot be laundered back
    # out through ``copy()``, ``.T``, slicing or a pickle round-trip.
    @property
    def _symmetry(self):
        raw = self._symmetry_raw
        if raw is None:
            return None
        if epoch_of(self) != self._symmetry_epoch:
            self._symmetry_raw = None  # latch: a voided claim never returns
            return None
        return raw

    @_symmetry.setter
    def _symmetry(self, value) -> None:
        self._symmetry_raw = value
        self._symmetry_epoch = 0 if value is None else epoch_of(self)

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
        from flopscope._array_ops import reshape as _reshape

        return _reshape(self, *shape, **kwargs)

    def ravel(self, order: str = "C"):  # type: ignore[override]
        from flopscope._array_ops import ravel as _ravel

        return _ravel(self, order=order)

    def flatten(self, order: str = "C"):  # type: ignore[override]
        from flopscope._array_ops import ravel as _ravel
        from flopscope._symmetry_transport import transport_ravel
        from flopscope.errors import _warn_symmetry_loss

        in_group = self._symmetry
        if in_group is not None:
            out_group = transport_ravel(in_group, input_shape=np.asarray(self).shape)
            if out_group is None:
                _warn_symmetry_loss(
                    lost_dims=[in_group.axes or tuple(range(in_group.degree))],
                    reason="flatten collapses to a single axis; block cannot fit",
                )
        # Pass the raw ndarray view so _ravel does not emit a second warning.
        out = _ravel(np.asarray(self), order=order)
        return np.array(out, copy=True)

    def squeeze(self, axis=None):  # type: ignore[override]
        from flopscope._array_ops import squeeze as _squeeze

        return _squeeze(self, axis=axis)  # type: ignore[arg-type]

    def astype(  # type: ignore[override]
        self,
        dtype,
        order: str = "K",
        casting: str = "unsafe",
        subok: bool = False,
        copy: bool = True,
    ):
        # Route through the counted ndarray method (FlopscopeArray.astype ->
        # _astype_counted) so this bills like every other astype call --
        # unlike copy() above, astype intentionally does NOT reattach
        # symmetry (a cast is not guaranteed to preserve the symmetric
        # structure), so no `.view(type(self))` here; the counted backend
        # already returns a plain (non-Symmetric) FlopscopeArray.
        return super().astype(
            dtype,
            order=order,
            casting=casting,
            subok=subok,
            copy=copy,
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


@_counted_wrapper
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
    >>> tagged = flops.as_symmetric(matrix, symmetry=(0, 1))
    >>> tagged.symmetric_axes
    [(0, 1)]
    """
    group = _resolve_symmetry_argument(data, symmetry=symmetry)
    assert group is not None  # required=True raises if symmetry is None
    array = np.asarray(data)
    _validate_and_charge_symmetry(array, group, op_name="as_symmetric")
    # Validation is tolerant, but the tag it authorizes is read as exact: the
    # cost model prices every orbit position after the first as redundant and
    # never re-reads the buffer. Copy one representative across each orbit so
    # that reading is true. Values that differed only within tolerance do not
    # survive, which is what stops a caller from scaling an asymmetric tensor
    # under atol, collecting the tag, and scaling back up with the independent
    # values -- and their discount -- intact. Data that is already exactly
    # invariant is passed through untouched, so the honest case keeps this
    # function's zero-copy view semantics.
    array = canonicalize(array, group)
    # Already validated and charged above -- construct via the trusted,
    # non-revalidating path so this doesn't pay (or re-check) twice through
    # SymmetricTensor's public, validating constructor. The canonical copy is
    # exactly invariant by construction, so there is nothing left to re-check.
    return SymmetricTensor._construct_trusted(array, symmetry=group)
