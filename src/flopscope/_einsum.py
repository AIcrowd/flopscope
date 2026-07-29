"""Einsum with analytical FLOP counting, symmetry detection, and path optimization."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as _np

from flopscope._budget import _call_numpy, _counted_wrapper
from flopscope._config import get_setting
from flopscope._dtype_billing import resolve_billing_dtype, store_billing_dtypes
from flopscope._ndarray import FlopscopeArray, _to_base_ndarray
from flopscope._perm_group import SymmetryGroup
from flopscope._pointwise import (
    _prepare_symmetric_out,
    _validate_result_symmetry,
)
from flopscope._symmetric import SymmetricTensor
from flopscope._symmetry_utils import normalize_symmetry_input, validate_symmetry_group
from flopscope._validation import _normalize_out, maybe_check_nan_inf, require_budget


def _identity_pattern(operands):
    """Build a hashable pattern of which operands are the same Python object.

    Returns None if all operands are distinct objects (common case).
    Otherwise returns a tuple of tuples, where each inner tuple lists
    positions sharing the same object identity (only groups of size >= 2).

    This mirrors the identical_operand_groups logic in _build_bipartite.
    """
    id_to_positions: dict[int, list[int]] = {}
    for idx, op in enumerate(operands):
        id_to_positions.setdefault(id(op), []).append(idx)
    groups = tuple(
        tuple(positions)
        for positions in id_to_positions.values()
        if len(positions) >= 2
    )
    return groups if groups else None


def _make_path_cache(maxsize):
    """Create a new lru_cache-wrapped path computation function.

    The cache key includes subscripts, shapes, optimizer, per_op_symmetries,
    and identity_pattern. Re-runs with the same inputs return the cached path.

    The key also includes ``per_op_symmetries`` (a tuple of per-operand
    SymmetryGroup-or-None, canonicalized as a hashable fingerprint) and
    ``identity_pattern`` so that symmetric operands produce a distinct cache
    entry from dense operands with the same subscripts and shapes.  When
    symmetry-aware path search is enabled the chosen path may differ; without
    this the dense-optimal path would silently be reused for symmetric inputs.
    """

    @functools.lru_cache(maxsize=maxsize)
    def _compute(
        subscripts,
        shapes,
        optimize,
        per_op_symmetries,
        identity_pattern,
    ):
        from flopscope._opt_einsum import contract_path as _contract_path

        _path, path_info = _contract_path(
            subscripts,
            *shapes,
            shapes=True,
            optimize=optimize if not isinstance(optimize, tuple) else list(optimize),
        )
        return path_info

    return _compute


_path_cache = _make_path_cache(4096)


def _rebuild_einsum_cache():
    """Rebuild the path cache with the current configured maxsize."""
    global _path_cache
    _path_cache = _make_path_cache(int(get_setting("einsum_path_cache_size")))  # type: ignore[arg-type]


def clear_einsum_cache():
    """Clear the einsum path cache.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Discards all cached contraction paths.

    Notes
    -----
    Discards all cached contraction paths. Subsequent ``einsum()`` and
    ``einsum_path()`` calls will recompute paths from scratch.

    Examples
    --------
    >>> import flopscope.numpy as fnp
    >>> fnp.clear_einsum_cache()
    """
    _path_cache.cache_clear()


def einsum_cache_info():
    """Return einsum path cache statistics.

    Parameters
    ----------
    None

    Returns
    -------
    object
        The standard ``functools.lru_cache`` statistics tuple with ``hits``,
        ``misses``, ``maxsize``, and ``currsize`` fields.

    Examples
    --------
    >>> import flopscope.numpy as fnp
    >>> info = fnp.einsum_cache_info()
    >>> total = info.hits + info.misses
    >>> rate = info.hits / max(total, 1)
    """
    return _path_cache.cache_info()


@functools.cache
def _einsum_supports_dtype(dtype) -> bool:
    """Can ``np.einsum`` contract in ``dtype`` at all? Asked by asking numpy.

    einsum has no inner loop for the string/bytes/void kinds. numpy surfaces
    that badly -- ``np.einsum("ij,jk->ik", f64, f64, out=np.empty(..., "U64"))``
    raises ``SystemError("<built-in function c_einsum> returned a result with
    an exception set")`` whose ``__cause__`` is the real refusal,
    ``TypeError("invalid data type for einsum")``. The refusal is the part
    worth reproducing; the SystemError wrapper is a numpy reporting bug.

    A one-element contraction is the cheapest way to have numpy answer for
    itself rather than us keeping a hand-written list of dtype kinds in sync
    with it, and the answer is cached per dtype so it costs one tiny numpy
    call per distinct dtype per process. ``zeros`` and not ``empty``:
    uninitialised object cells are ``None``, and ``None * None`` would raise
    ``TypeError`` and misreport object -- which einsum does support -- as
    unsupported.
    """
    probe = _np.zeros(1, dtype=dtype)
    try:
        _np.einsum("i,i->i", probe, probe)
    except TypeError:
        return False
    return True


_CastingKind = Literal["no", "equiv", "safe", "same_kind", "unsafe"]


def _resolve_out_compute_dtype(
    operand_dtypes: tuple,
    out_dtype,
    n_operands: int,
    casting: _CastingKind = "safe",
):
    """Reproduce numpy's ``einsum(..., out=)`` casting decision exactly.

    numpy does not contract in the operands' own promoted dtype and then cast
    the answer into ``out``. It resolves ONE computation dtype for the whole
    iterator with the destination participating in the promotion, and refuses
    the call unless that dtype casts into the destination under ``casting``::

        op_dtype = np.result_type(*operand_dtypes, out.dtype)
        accept  iff np.can_cast(op_dtype, out.dtype, casting)

    Both halves matter, and getting either wrong is a shipped bug:

    * Dropping the check is what this function fixes. ``copyto(...,
      casting="unsafe")`` let two float64 operands land in an int64
      destination (numpy raises ``TypeError``) and let a complex result be
      truncated into a float64 one with nothing but a ``ComplexWarning``.
    * Dropping ``out.dtype`` from the promotion is what killed a previous
      attempt at this fix: it computed ``result_type(*operand_dtypes)`` and
      asked whether THAT cast into the destination, which refused 138 calls
      plain numpy accepts. ``np.result_type`` is a lattice minimum, not a
      left-fold, so ``result_type(int8, uint8)`` is ``int16`` while
      ``result_type(int8, uint8, float16)`` is ``float16`` -- float16 holds
      every int8 and every uint8 exactly, so it is a legal common type and
      the lattice picks it. Being stricter than numpy is a regression, not a
      conservative choice.

    Returns the resolved computation dtype, which the caller must contract in
    -- ``int8 x int8 -> int16`` destination yields 300 in numpy, not the -56
    an int8 contraction would overflow to.

    Raises
    ------
    TypeError
        With numpy's own wording, when the computation dtype cannot cast into
        the destination. Raised before any FLOP is charged.
    numpy.exceptions.DTypePromotionError
        Propagated unwrapped from ``np.result_type`` when no common dtype
        exists at all (a ``datetime64`` destination, say); it is a
        ``TypeError`` subclass and numpy surfaces it the same way.
    """
    op_dtype = _np.result_type(*operand_dtypes, out_dtype)
    # The rule governs the INPUT cast too, not only the store. Under 'safe'
    # and 'same_kind' this is implied -- an operand always casts into a
    # promotion that includes it -- which is why it only shows up under the
    # strict kinds: `bool x bool` into an int8 destination promotes to int8,
    # and int8 stores into int8 under any rule, but numpy still refuses the
    # call under 'equiv'/'no' because the bool OPERANDS do not cast to int8
    # under those. Checking only the store made flopscope looser than numpy
    # in 856 measured cells, all of them here.
    for operand_dtype in operand_dtypes:
        if not _np.can_cast(operand_dtype, op_dtype, casting=casting):
            raise TypeError(
                f"Iterator operand required copying or buffering, but "
                f"neither copying nor buffering was enabled, according to "
                f"the rule '{casting}'"
            )
    if not _np.can_cast(op_dtype, out_dtype, casting=casting):
        # numpy's nditer wording, verbatim: the destination is operand number
        # `n_operands` in the iterator (0-based, after the inputs). `!r` and
        # not `dtype('{...}')` because repr is what numpy formats with, and
        # str disagrees with it on exactly the dtypes where it matters --
        # `str(np.dtype('S32'))` is '|S32' but numpy's message says 'S32',
        # and object prints as 'object' rather than numpy's 'O'.
        raise TypeError(
            f"Iterator requested dtype could not be cast from "
            f"{op_dtype!r} to {out_dtype!r}, the operand "
            f"{n_operands} dtype, according to the rule '{casting}'"
        )
    if not _einsum_supports_dtype(op_dtype):
        # numpy's own refusal reason, taken from the `__cause__` it loses on
        # the way out. Checked AFTER the cast rule because that is numpy's
        # order: a `U32` destination fails the cast (op_dtype `U64` does not
        # fit) while a `U64` one gets all the way to the missing inner loop.
        # Zero-cost either way -- the alternative is billing a contraction
        # that then dies inside numpy, and nothing here is ever refunded.
        raise TypeError("invalid data type for einsum")
    return op_dtype


def _expands_when_materialized(array) -> bool:
    """Would copying *array* cost far more than the memory it occupies?

    True for a broadcast view (a zero stride repeats one element across a
    whole axis) and for any view whose logical size dwarfs the buffer it
    looks at. Casting such an operand allocates its logical shape, which is
    unbounded relative to what the caller actually handed over.
    """
    if 0 in array.strides and array.size > 1:
        return True
    base = array.base
    return base is not None and getattr(base, "nbytes", array.nbytes) * 4 < array.nbytes


def _execute_pairwise(path_info, operands: list):
    """Execute pairwise contractions according to the optimized path."""
    ops = list(operands)
    for contract_inds, step in zip(path_info.path, path_info.steps, strict=False):
        # Pop operands in reverse sorted order (same as opt_einsum convention)
        inds = sorted(contract_inds, reverse=True)
        tensors = [ops.pop(i) for i in inds]
        bases = [_to_base_ndarray(t) for t in tensors]
        # The path was chosen (and billed) before execution, so optimized
        # dispatch may only pick the step's kernel (tensordot/BLAS vs the
        # non-dispatching sum-of-products loop), never re-plan it. That
        # holds only for two-operand steps — numpy would decompose an
        # n-ary step (from an explicit user path) into its own pairwise
        # path — and only for non-object dtypes: the tensordot route
        # changes object-element semantics (multiplication order and
        # c_einsum's `0 + first_term` accumulator seeding).
        use_optimize = len(bases) == 2 and all(
            base.dtype != _np.object_ for base in bases
        )
        result = _call_numpy(_np.einsum, step.subscript, *bases, optimize=use_optimize)
        ops.append(result)
    return ops[0]


_LARGE_K_THRESHOLD = 8


def _resolve_optimize_for_k(optimize, k: int):
    """Auto-downgrade 'auto' to 'greedy' for k >= 8 to avoid optimal/B&B
    cold-call latency on large operand counts. Explicit user choices
    (optimal/branch/dp/etc.) are honored verbatim.
    """
    if optimize == "auto" and k >= _LARGE_K_THRESHOLD:
        return "greedy"
    return optimize


def _normalize_optimize(optimize):
    if optimize is False:
        return "auto"
    if isinstance(optimize, list):
        return tuple(tuple(t) for t in optimize)
    return optimize


def _parse_einsum_parts(subscripts: str, operands):
    from flopscope._opt_einsum import parse_einsum_input

    input_subscripts, output_subscript, _ = parse_einsum_input((subscripts, *operands))
    canonical_subscripts = f"{input_subscripts}->{output_subscript}"
    return canonical_subscripts, input_subscripts.split(","), output_subscript


def _get_path_info(
    subscripts: str,
    operands,
    optimize,
    *,
    per_op_symmetries=None,
    identity_pattern=None,
):
    canonical_subscripts, input_parts, output_subscript = _parse_einsum_parts(
        subscripts,
        operands,
    )
    shapes = tuple(tuple(op.shape) for op in operands)

    # Build a hashable symmetry key for the cache.  Each entry is either None
    # (dense operand) or the canonical fingerprint of a SymmetryGroup so that
    # symmetric and dense operands with identical subscripts/shapes get distinct
    # cache slots.  This prevents a dense-optimal path from being silently
    # reused when symmetry-aware path search is later enabled.
    if per_op_symmetries is None:
        from flopscope._accumulation._public import _per_op_symmetries as _extract_syms

        per_op_symmetries = _extract_syms(operands)
    syms_key = tuple(per_op_symmetries)

    if identity_pattern is None:
        from flopscope._accumulation._public import _identity_pattern as _extract_id

        identity_pattern = _extract_id(operands)

    effective_optimize = _resolve_optimize_for_k(optimize, k=len(operands))
    path_info = _path_cache(
        canonical_subscripts,
        shapes,
        _normalize_optimize(effective_optimize),
        syms_key,
        identity_pattern,
    )

    # Bug B fix: if any operand has declared symmetry OR multiple operand
    # positions alias to the same array (identity_pattern), rebuild path_info
    # through the SubgraphSymmetryOracle so that per-step input_groups /
    # output_group / inner_group reflect the true residual symmetry of each
    # intermediate.  Without this rebuild, Source-A (declared groups),
    # Source-B (identical-operand swap), and Source-C (coordinated relabel)
    # π-generators never reach the renderer.
    #
    # _path_cache returns a shared cached object that must not be mutated;
    # build_path_info returns a fresh PathInfo each time.  The rebuild is
    # skipped when there's no symmetry signal at all (the common case) to
    # keep the fast path.
    _has_identity_alias = bool(identity_pattern) and any(
        len(group) > 1 for group in identity_pattern
    )
    if any(s is not None for s in per_op_symmetries) or _has_identity_alias:
        import numpy as _np_tmp
        import opt_einsum as _oe

        from flopscope._opt_einsum._contract import build_path_info as _bpi

        # Build dummy operands with the correct shapes, then alias positions
        # listed in the same identity-group to share object identity — this
        # is the signal the oracle uses to fire Source-B (identical-operand
        # swap) and Source-C (coordinated axis relabel) generators.
        _dummy_ops: list = [_np_tmp.empty(sh) for sh in shapes]
        if identity_pattern is not None:
            for group in identity_pattern:
                canonical = _dummy_ops[group[0]]
                for pos in group[1:]:
                    _dummy_ops[pos] = canonical

        _norm_optimize = _normalize_optimize(effective_optimize)
        if isinstance(_norm_optimize, tuple):
            _norm_optimize = list(_norm_optimize)
        _upstream_path, _upstream_info = _oe.contract_path(
            canonical_subscripts,
            *_dummy_ops,
            optimize=_norm_optimize,  # type: ignore[arg-type]
        )
        # Carry the optimizer label through the rebuild so the renderer's
        # "Optimizer:" pill stays populated.  effective_optimize is whatever
        # was actually used for path search; coerce to a string label.
        if isinstance(effective_optimize, str):
            _optimizer_label = effective_optimize
        else:
            _optimizer_label = getattr(_upstream_info, "_path_type", "") or ""
        path_info = _bpi(
            _upstream_path,
            _upstream_info,
            size_dict=_upstream_info.size_dict,
            optimizer_used=_optimizer_label,
            per_op_symmetries=per_op_symmetries,
            identity_pattern=identity_pattern,
        )

    return canonical_subscripts, input_parts, output_subscript, shapes, path_info


def _relabel_group_to_output(
    group, source_labels: tuple[str, ...], output_subscript: str
):
    if group is None or not source_labels or not output_subscript:
        return None
    output_positions = {label: idx for idx, label in enumerate(output_subscript)}
    try:
        source_positions = tuple(output_positions[label] for label in source_labels)
    except KeyError:
        return None
    if len(set(source_positions)) != len(source_positions):
        return None

    order = tuple(
        sorted(range(len(source_positions)), key=source_positions.__getitem__)
    )
    axes = tuple(source_positions[idx] for idx in order)
    source_to_sorted = {
        source_idx: sorted_idx for sorted_idx, source_idx in enumerate(order)
    }

    from flopscope._perm_group import _PermutationCompat as Permutation

    generators = []
    for gen in group.generators:
        generators.append(
            Permutation(
                [source_to_sorted[gen.array_form[source_idx]] for source_idx in order]
            )
        )

    remapped = SymmetryGroup(*generators, axes=axes)
    return validate_symmetry_group(remapped, ndim=len(output_subscript))


def _infer_pathless_output_symmetry(operands, input_parts, output_subscript: str):
    if len(operands) != 1:
        return None
    operand = operands[0]
    if not isinstance(operand, SymmetricTensor) or operand.symmetry is None:
        return None
    group = operand.symmetry
    operand_subscript = input_parts[0]
    operand_rank = len(operand_subscript)

    # Detect axes that get summed out (label appears in operand but not output).
    summed_axes = tuple(
        i for i, label in enumerate(operand_subscript) if label not in output_subscript
    )

    if summed_axes:
        # Compute the setwise-stabilizer of the summed axes inside the group,
        # then project onto the surviving axes via the existing reduce_group
        # helper (which composes setwise_stabilizer + restrict + axis remap
        # into one numpy-reduction-style call).  This is the
        # stabilizer-restriction operation Wilson's review asked for.
        from flopscope._symmetry_utils import reduce_group

        reduced_group = reduce_group(group, ndim=operand_rank, axis=summed_axes)
        if reduced_group is None:
            return None
        # reduce_group's keepdims=False shifts surviving operand axes to
        # contiguous 0..k-1 positions in the reduced-tensor frame.  Recover
        # operand-subscript labels for the reduced group's axes so that the
        # subsequent _relabel_group_to_output call can map them to the
        # einsum's output_subscript positions (which may further reorder).
        kept_operand_axes = [
            i for i in range(operand_rank) if i not in set(summed_axes)
        ]
        new_to_operand_axis = dict(enumerate(kept_operand_axes))
        reduced_axes = (
            reduced_group.axes
            if reduced_group.axes is not None
            else tuple(range(reduced_group.degree))
        )
        source_labels = tuple(
            operand_subscript[new_to_operand_axis[ax]] for ax in reduced_axes
        )
        return _relabel_group_to_output(reduced_group, source_labels, output_subscript)

    # No reduction — surviving labels all appear in output; existing direct path.
    axes = group.axes if group.axes is not None else tuple(range(group.degree))
    source_labels = tuple(operand_subscript[axis] for axis in axes)
    return _relabel_group_to_output(group, source_labels, output_subscript)


def _infer_multi_operand_output_symmetry(path_info, output_subscript: str):
    """Infer the output tensor's symmetry from the path walker's last step.

    Returns the SymmetryGroup acting on the einsum's output_subscript labels,
    or None if no symmetry was derived or relabel fails.

    The path walker's oracle stores `output_group` on each StepInfo with axes
    indexing the *step's* output subscript.  We must relabel those axes to
    positions in the *einsum's* output_subscript, which may differ (opt_einsum
    can permute labels for BLAS-friendly orientation).
    """
    if path_info is None:
        return None
    steps = getattr(path_info, "steps", None)
    if not steps:
        return None
    last = steps[-1]
    group = getattr(last, "output_group", None)
    if group is None:
        return None
    if not output_subscript:
        return None
    # Derive the step's output labels from its subscript string ("lhs->rhs").
    step_subscript = getattr(last, "subscript", "")
    if "->" not in step_subscript:
        return None
    _, step_out = step_subscript.split("->", 1)
    if not step_out:
        return None
    # group.axes are positions in step_out; map each to its label, then relabel
    # to positions in output_subscript.
    axes = group.axes if group.axes is not None else tuple(range(group.degree))
    try:
        source_labels = tuple(step_out[ax] for ax in axes)
    except IndexError:
        return None
    return _relabel_group_to_output(group, source_labels, output_subscript)


def _resolve_output_symmetry(
    *,
    symmetry,
    operands,
    input_parts,
    output_subscript: str,
    path_info=None,
):
    if symmetry is not None:
        return normalize_symmetry_input(symmetry, ndim=len(output_subscript))
    if len(operands) == 1:
        return _infer_pathless_output_symmetry(operands, input_parts, output_subscript)
    return _infer_multi_operand_output_symmetry(path_info, output_subscript)


@dataclass(frozen=True, slots=True)
class _CostInfo:
    """Output of :func:`_resolve_cost_and_output_symmetry`.

    Carries everything a bilinear wrapper needs to charge budget and wrap
    its result: the symmetry-aware accumulation cost, the inferred output
    symmetry (or None), the canonical einsum subscript string, the shapes
    tuple, and the full path info (reserved for future use).
    """

    accumulation: Any  # flopscope._accumulation._cost.AccumulationCost
    output_symmetry: SymmetryGroup | None
    canonical_subscripts: str
    input_parts: tuple[str, ...]
    output_subscript: str
    shapes: tuple[tuple[int, ...], ...]
    path_info: Any  # FlopscopePathInfo


def _resolve_cost_and_output_symmetry(
    subscripts: str,
    *operands: Any,
    optimize: str | bool | list[Any] = "auto",
) -> _CostInfo:
    """Run path-find + accumulation-cost + output-symmetry inference.

    Does NOT execute compute; does NOT charge budget. Used by the bilinear
    wrappers (matmul/dot/outer/inner/tensordot/vdot) to share einsum's
    cost+symmetry-inference machinery while keeping their native BLAS-fast
    compute paths and friendly op-names.

    Parameters
    ----------
    subscripts : str
        Einsum subscript string (e.g. ``"ij,jk->ik"``).
    *operands
        The operands as the caller sees them (raw ndarray, FlopscopeArray,
        or SymmetricTensor — all handled).
    optimize : str | bool | list, optional
        Path optimizer; defaults to ``"auto"``.

    Returns
    -------
    _CostInfo
        Dataclass with ``accumulation``, ``output_symmetry``,
        ``canonical_subscripts``, ``input_parts``, ``output_subscript``,
        ``shapes``, ``path_info``.
    """
    canonical_subscripts, input_parts, output_subscript, shapes, path_info = (
        _get_path_info(subscripts, operands, optimize)
    )
    accumulation_cost = _get_accumulation_cost(
        canonical_subscripts=canonical_subscripts,
        input_parts=tuple(input_parts),
        output_subscript=output_subscript,
        shapes=shapes,
        operands=tuple(operands),
    )
    from flopscope._accumulation._path_info import FlopscopePathInfo

    path_info = FlopscopePathInfo.from_inner(
        inner=path_info,
        accumulation=accumulation_cost,
    )
    output_symmetry = _resolve_output_symmetry(
        symmetry=None,
        operands=operands,
        input_parts=input_parts,
        output_subscript=output_subscript,
        path_info=path_info,
    )
    return _CostInfo(
        accumulation=accumulation_cost,
        output_symmetry=output_symmetry,
        canonical_subscripts=canonical_subscripts,
        input_parts=tuple(input_parts),
        output_subscript=output_subscript,
        shapes=tuple(shapes),
        path_info=path_info,
    )


@_counted_wrapper
def einsum(
    subscripts: str,
    *operands: _np.ndarray,
    out: Any = None,
    optimize: str | bool | list[Any] = "auto",
    symmetry: Any = None,
    **kwargs: Any,
) -> FlopscopeArray:
    """Evaluate Einstein summation with FLOP counting and optional path optimization.

    Wraps ``numpy.einsum`` with analytical FLOP cost computation and
    optional symmetry savings. If any input is a ``SymmetricTensor``,
    the cost is automatically reduced. If ``symmetry`` is provided and the output passes validation, a ``SymmetricTensor`` is returned.

    All contractions go through opt_einsum's ``contract_path`` to find an
    optimal pairwise decomposition. The charged FLOP cost comes from the
    path-independent symmetry-aware accumulation total
    (``path_info.accumulation.total``); per-step ``flop_count`` values on
    each ``StepInfo`` use flopscope's FMA=2 textbook convention throughout.

    Contraction paths are cached in a module-level LRU cache keyed on
    (subscripts, shapes, optimizer, per_op_symmetries, identity_pattern).
    Repeated calls with the same inputs skip path recomputation entirely.
    See ``clear_einsum_cache()`` and ``einsum_cache_info()``.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string (e.g., ``'ij,jk->ik'``).
    *operands : numpy.ndarray
        Input arrays. ``SymmetricTensor`` inputs are detected automatically
        for cost savings.
    optimize : str, bool, or list of tuple, optional
        Contraction path strategy. Default ``'auto'``.

        - ``'auto'``, ``'greedy'``, ``'optimal'``, ``'dp'``, etc.:
          Use the named algorithm to find the best path.
        - A list of int-tuples (e.g. ``[(1, 2), (0, 1)]``): use this
          explicit contraction path. Obtain one from ``fnp.einsum_path()``
          or construct manually. Each tuple names the operand positions
          to contract at that step; the result is appended to the end.
        - ``False``: treated as ``'auto'``.
    symmetry : SymmetryGroup or symmetry shorthand, optional
        Declares output symmetry and wraps the validated result as a
        ``SymmetricTensor``. This does NOT declare input symmetry; use
        ``flops.as_symmetric()`` for that.

    Returns
    -------
    numpy.ndarray or SymmetricTensor
        The result of the einsum.

    Raises
    ------
    BudgetExhaustedError
        If the operation would exceed the FLOP budget.
    NoBudgetContextError
        If called outside a ``BudgetContext``.
    SymmetryError
        If ``symmetry`` is provided but the result
        does not satisfy the declared symmetry. Validation checks the
        data against each generator of the group.
    """
    budget = require_budget()
    info = _resolve_cost_and_output_symmetry(subscripts, *operands, optimize=optimize)
    canonical_subscripts = info.canonical_subscripts
    accumulation_cost = info.accumulation
    path_info = info.path_info
    shapes = info.shapes
    output_subscript = info.output_subscript

    # User-declared symmetry overrides the helper's inferred symmetry;
    # otherwise honor an existing SymmetricTensor `out=` operand.
    if symmetry is not None:
        target_symmetry = normalize_symmetry_input(symmetry, ndim=len(output_subscript))
    else:
        target_symmetry = info.output_symmetry
    # Before anything is billed: a wrapped destination (``out=[dest]``) would
    # otherwise reach the copy below, where ``_np.asarray`` builds a new array
    # from the container -- the result lands in that temporary, ``dest`` keeps
    # its old contents, and the caller gets the untouched wrapper back having
    # paid for the contraction.
    out = _normalize_out(out, "einsum")

    effective_out_symmetry = target_symmetry
    if (
        effective_out_symmetry is None
        and isinstance(out, SymmetricTensor)
        and not getattr(out, "_symmetry_inferred", False)
    ):
        # Only a tag the caller validated becomes a requirement on the result.
        # Lifting an *inferred* one made an ordinary scratch arena raise: an
        # `fnp.zeros((n, n))` destination is auto-tagged symmetric, so writing
        # any asymmetric contraction into it failed, for a legal numpy call
        # against metadata the caller never asked for. The pointwise factories
        # drop an inferred tag quietly; this brings einsum in line.
        effective_out_symmetry = out.symmetry
    target_symmetry = _prepare_symmetric_out(out, effective_out_symmetry)

    operand_arrays = [
        o if isinstance(o, _np.ndarray) else _np.asarray(o) for o in operands
    ]
    billing_dtypes = tuple(a.dtype for a in operand_arrays)
    billing_dtypes += store_billing_dtypes(out)
    resolved = resolve_billing_dtype(billing_dtypes)
    complex_override = contraction_complex_override(accumulation_cost, resolved)

    # Settle the destination's dtype question BEFORE the deduct below, because
    # a refusal has to cost nothing: flops_used never decreases and nothing is
    # ever refunded, so a call numpy would have rejected outright must not be
    # able to bill a contraction on its way to raising.
    compute_dtype = None
    out_dtype = (
        getattr(_to_base_ndarray(out), "dtype", None) if out is not None else None
    )
    # The caller's own ``casting=`` governs, exactly as it does in numpy.
    # Hardcoding "safe" here would refuse the whole point of passing
    # ``casting="unsafe"`` -- and would report "the rule 'safe'" while doing
    # it -- making flopscope stricter than numpy on a call numpy accepts,
    # which is the regression this change exists to avoid.
    requested_casting = kwargs.get("casting", "safe")
    if out_dtype is not None:
        compute_dtype = _resolve_out_compute_dtype(
            tuple(a.dtype for a in operand_arrays),
            out_dtype,
            len(operand_arrays),
            casting=requested_casting,
        )

    with budget.deduct(
        "einsum",
        flop_cost=accumulation_cost.total,
        subscripts=canonical_subscripts,
        shapes=tuple(shapes),
        dtypes=billing_dtypes,
        complex_factor_override=complex_override,
    ):
        # Contract in the dtype numpy would have contracted in. Casting the
        # ANSWER into `out` is not the same operation as computing in the
        # destination's dtype: `int8 x int8` into an int16 destination is 300
        # in numpy and -56 if the contraction runs in int8 first, and
        # `bool x bool` into any numeric destination counts the matches
        # (3) where a bool contraction only ors them (1).
        needs_promotion = compute_dtype is not None and any(
            a.dtype != compute_dtype for a in operand_arrays
        )
        # Casting an operand materializes its LOGICAL shape. That is fine for
        # a dense array -- the copy is the same size as the original -- but
        # ruinous for an operand whose logical size far exceeds its storage:
        # a broadcast view has O(1) bytes and O(numel) logical size, so
        # promoting one turned a 4-byte view into an allocation the size of
        # the contraction (~80GB for a 100000^2 view; measured 128MB against
        # numpy's 0.1MB at 4000^2).
        expansive = needs_promotion and any(
            _expands_when_materialized(_to_base_ndarray(a)) for a in operand_arrays
        )
        if expansive:
            # ``dtype=`` casts inside numpy's iterator, per element, so the
            # view is never materialized. This gives up the planned pairwise
            # path for these operands, which is the right trade only BECAUSE
            # they are the expansive ones -- the plan is not worth an
            # allocation proportional to a logical shape. Dense operands keep
            # their path below.
            result = _call_numpy(
                _np.einsum,
                canonical_subscripts,
                *[_to_base_ndarray(o) for o in operands],
                dtype=compute_dtype,
                casting=requested_casting,
            )
        else:
            exec_operands = operands
            if needs_promotion:
                exec_operands = [
                    _to_base_ndarray(a).astype(compute_dtype, copy=False)
                    for a in operand_arrays
                ]
            if path_info.steps:
                result = _execute_pairwise(path_info, list(exec_operands))
            else:
                result = _call_numpy(
                    _np.einsum,
                    canonical_subscripts,
                    *[_to_base_ndarray(o) for o in exec_operands],
                )

    if out is not None:
        _validate_result_symmetry(result, target_symmetry)
        # ``_to_base_ndarray``, never ``_np.asarray``: asarray on anything that
        # is not already an array builds a NEW buffer, and the copy below then
        # fills that temporary while the caller's destination keeps its old
        # contents. Guarding the argument stops a container getting here, but
        # taking the materialising call out is what makes the whole class of
        # silent mis-write structurally impossible rather than merely gated.
        dest = _to_base_ndarray(out)
        source = _np.asarray(result)
        # einsum is the one contraction that does not forward ``out=``, so by
        # here numpy has already written the entire result into a buffer of its
        # own and this copies that buffer into the caller's destination. That is
        # a second full materialising pass over the data, not the destination
        # numpy would have allocated anyway, and the cost model prices exactly
        # that at one unit per element written (see docs/reference/cost-model.md,
        # "every byte written is metered", decision-procedure step 2). Charging
        # it as ``copyto``, with ``copyto``'s own cost and dtypes rather than the
        # contraction's, makes ``einsum(out=d)`` cost precisely what ``einsum()``
        # followed by ``fnp.copyto(d, result)`` costs -- including on complex
        # operands, where a contraction's exact complex factor is not a copy's.
        # Until this the pass was free, so ``einsum("ij->ji", src, out=dest)``
        # materialised a full transpose of any size at no charge, which is the
        # copy-chain laundering the write-metered model exists to close.
        with budget.deduct(
            "copyto",
            flop_cost=dest.size,
            subscripts=None,
            shapes=(),
            dtypes=(source.dtype, dest.dtype),
        ):
            # Through ``_call_numpy``, which is what puts the copy's wall time in
            # backend rather than leaving it to the enclosing
            # ``@_counted_wrapper``'s overhead remainder -- being outside the
            # contraction's deduct never made it residual, because the wrapper
            # bills its whole non-backend remainder to overhead. ``_call_numpy``
            # also records the write itself, ``np.copyto`` being in
            # ``_MUTATES_FIRST_ARG``, so a symmetry tag observing ``out``'s buffer
            # (or an alias of it) is still voided -- that is what the hand-rolled
            # ``note_write`` this replaces was for.
            #
            # The casting rule is the one settled in #168: ``safe`` once the
            # dtype has been resolved above, because the contraction already ran
            # in a dtype that casts into the destination, so this copy is a
            # narrowing-free store and numpy will say so if it ever is not.
            _call_numpy(
                _np.copyto,
                dest,
                source,
                casting=requested_casting if compute_dtype is not None else "unsafe",
            )
        maybe_check_nan_inf(out, "einsum")
        return out

    if target_symmetry is not None and _validate_result_symmetry(
        result, target_symmetry
    ):
        result = SymmetricTensor(_np.asarray(result), symmetry=target_symmetry)
    else:
        # An unverified claim is not stamped. Validation is skipped for
        # non-finite results, so tagging regardless would let a caller mint a
        # symmetry claim on asymmetric data by poisoning one entry and then
        # cleaning it up downstream.
        result = _asflopscope(_np.asarray(result))

    maybe_check_nan_inf(result, "einsum")
    return result  # type: ignore[return-value]


@_counted_wrapper
def einsum_path(
    subscripts: str,
    *operands: _np.ndarray,
    optimize: str | bool | list[Any] = "auto",
) -> tuple[list[Any], Any]:
    """Compute the optimal contraction path without executing.

    Returns ``(path, PathInfo)`` with zero budget cost. The returned
    ``path`` can be passed back to ``fnp.einsum(..., optimize=path)``
    to execute with that exact contraction order.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string.
    *operands : numpy.ndarray
        Input arrays.
    optimize : str, bool, or list of tuple, optional
        Path optimization strategy. Default ``'auto'``.

    Returns
    -------
    path : list of tuple of int
        The contraction path. Pass to ``fnp.einsum(..., optimize=path)``.
    info : PathInfo
        Diagnostics including per-step costs and symmetry savings.
    """
    budget = require_budget()
    with budget.deduct(
        "einsum_path", flop_cost=1, subscripts=None, shapes=(), dtypes=()
    ):
        pass
    canonical_subscripts, input_parts, output_subscript, shapes, path_info = (
        _get_path_info(
            subscripts,
            operands,
            optimize,
        )
    )

    accumulation_cost = _get_accumulation_cost(
        canonical_subscripts=canonical_subscripts,
        input_parts=tuple(input_parts),
        output_subscript=output_subscript,
        shapes=shapes,
        operands=tuple(operands),
    )

    from flopscope._accumulation._path_info import FlopscopePathInfo

    path_info = FlopscopePathInfo.from_inner(
        inner=path_info,
        accumulation=accumulation_cost,
    )

    return list(path_info.path), path_info


# ── Accumulation cost helper + cache ─────────────────────────────────


from flopscope._accumulation._cache import (  # noqa: E402, F401
    _accumulation_cache,
    get_accumulation_cost_cached,
)
from flopscope._accumulation._cache import (  # noqa: E402
    rebuild_accumulation_cache as _rebuild_accumulation_cache_fn,
)
from flopscope._accumulation._cost import contraction_complex_override  # noqa: E402
from flopscope._accumulation._public import (  # noqa: E402
    _accumulation_fingerprint,
    _identity_pattern,
)


def _get_accumulation_cost(
    *,
    canonical_subscripts: str,
    input_parts: tuple,
    output_subscript: str,
    shapes: tuple,
    operands: tuple,
):
    """Cached accumulation-cost lookup for einsum() / einsum_path()."""
    # Resolve partition_budget to the active setting BEFORE cache lookup so
    # the cache key reflects the budget used; otherwise stale entries from a
    # prior setting value can leak across calls.
    partition_budget = cast(int, get_setting("partition_budget"))
    return get_accumulation_cost_cached(
        canonical_subscripts=canonical_subscripts,
        input_parts=tuple(input_parts),
        output_subscript=output_subscript,
        shapes=shapes,
        sym_fingerprint=_accumulation_fingerprint(operands),
        identity_pattern=_identity_pattern(operands),
        partition_budget=partition_budget,
    )


def _rebuild_accumulation_cache():
    """Rebuild the accumulation cache with the current configured maxsize."""
    _rebuild_accumulation_cache_fn(cast(int, get_setting("einsum_path_cache_size")))


import sys as _sys  # noqa: E402

from flopscope._ndarray import _asflopscope  # noqa: E402
from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__], skip_names={"einsum", "einsum_path"})
