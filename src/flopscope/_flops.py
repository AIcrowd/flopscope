"""FLOP cost calculators for flopscope operations."""

from __future__ import annotations

import math
from collections import Counter

from flopscope._perm_group import SymmetryGroup
from flopscope._symmetry_utils import unique_elements_for_shape


def parse_einsum_subscripts(subscripts: str) -> tuple[list[list[str]], list[str]]:
    """Parse an einsum subscript string into input and output index lists.

    Parameters
    ----------
    subscripts : str
        Einsum subscript string (e.g., ``'ij,jk->ik'``).

    Returns
    -------
    inputs : list of list of str
        Index labels for each input operand.
    output : list of str
        Index labels for the output.
    """
    subscripts = subscripts.replace(" ", "")
    if "->" in subscripts:
        input_part, output_part = subscripts.split("->")
        output = list(output_part)
    else:
        input_part = subscripts
        all_labels: list[str] = []
        for part in input_part.split(","):
            all_labels.extend(list(part))
        counts = Counter(all_labels)
        output = sorted(lbl for lbl, c in counts.items() if c == 1)
    inputs = [list(part) for part in input_part.split(",")]
    return inputs, output


def einsum_cost(
    subscripts: str,
    shapes: list[tuple[int, ...]],
    operand_symmetries: list[SymmetryGroup | None] | None = None,
    identity_pattern: tuple[tuple[int, ...], ...] | None = None,
) -> int:
    """FLOP cost of an einsum operation.

    Uses the whole-expression direct-event accumulation model:
    total = (k-1) * prod(M) + prod(alpha), where M is the number of
    unique output elements and alpha is the number of unique output+
    reduction-axis combinations. FMA = 2 (textbook): the α/M formula counts
    multiplies and adds separately by construction.

    Parameters
    ----------
    subscripts : str
        Einsum subscript string.
    shapes : list of tuple of int
        Shapes of the input operands.
    operand_symmetries : list of SymmetryGroup or None, optional
        Exact symmetry group for each input operand.
    identity_pattern : tuple of tuple of int, optional
        Groups of operand positions that alias to the same underlying
        array. Each inner tuple lists ≥2 positions sharing identity. When
        provided, the cost path applies repeated-operand savings (e.g.,
        the joint symmetry of ``A @ A``).

    Returns
    -------
    int
        Estimated FLOP count.
    """
    import numpy as _np

    from flopscope._accumulation._cost import compute_accumulation_cost
    from flopscope._opt_einsum import parse_einsum_input

    dummy_operands = [_np.empty(s) for s in shapes]
    input_subscripts, output_subscript, _ = parse_einsum_input(
        (subscripts, *dummy_operands)
    )
    canonical_subscripts = f"{input_subscripts}->{output_subscript}"
    input_parts = tuple(input_subscripts.split(","))

    per_op_syms: tuple[SymmetryGroup | None, ...] = (
        tuple(operand_symmetries)
        if operand_symmetries is not None
        else (None,) * len(shapes)
    )

    cost = compute_accumulation_cost(
        canonical_subscripts=canonical_subscripts,
        input_parts=input_parts,
        output_subscript=output_subscript,
        shapes=shapes,
        per_op_symmetries=per_op_syms,
        identity_pattern=identity_pattern,
    )
    return cost.total


def analytical_pointwise_cost(
    shape: tuple[int, ...], symmetry: SymmetryGroup | None = None
) -> int:
    """FLOP cost of a pointwise (element-wise) operation.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array.
    symmetry : SymmetryGroup or None, optional
        If provided, only unique elements are counted.

    Returns
    -------
    int
        Estimated FLOP count (one per element, or one per unique element).
    """
    if symmetry is not None:
        return max(unique_elements_for_shape(symmetry, shape), 1)
    result = 1
    for dim in shape:
        result *= dim
    return max(result, 1)


def analytical_reduction_cost(
    input_shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
    symmetry: SymmetryGroup | None = None,
) -> int:
    """FLOP cost of a reduction operation (orbit-mapping model).

    Delegates to compute_reduction_accumulation_cost from the
    _accumulation subpackage. The legacy `unique_elements_for_shape`-based
    formula is removed in favor of the path-independent orbit-mapping count
    (with #56's off-by-one corrected).

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input array.
    axis : int, tuple of int, or None, optional
        Axis or axes to reduce. None means full reduction.
    symmetry : SymmetryGroup, optional
        Pointwise symmetry of the input.

    Returns
    -------
    int
        FLOP count via the Tier-1 orbit-mapping model.
    """
    from flopscope._accumulation._reduction import (
        _normalize_axis,
        compute_reduction_accumulation_cost,
    )

    ndim = len(input_shape)
    axes_summed = _normalize_axis(axis, ndim)
    cost = compute_reduction_accumulation_cost(
        input_shape=tuple(input_shape),
        axes_summed=axes_summed,
        symmetry=symmetry,
        op_factor=1,
        extra_ops=0,
    )
    return max(cost.total, 1)


# Backward-compatible internal aliases. The public weighted API lives in
# ``flopscope.flops`` and wraps these analytical formulas.
pointwise_cost = analytical_pointwise_cost
reduction_cost = analytical_reduction_cost


def svd_cost(m: int, n: int, k: int | None = None) -> int:
    """FLOP cost of a (truncated) SVD.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    k : int or None, optional
        Number of singular values/vectors to compute. Defaults to min(m, n).

    Returns
    -------
    int
        Estimated FLOP count: m * n * k.

    Notes
    -----
    Based on Golub-Reinsch bidiagonalization.
    """
    if k is None:
        k = min(m, n)
    return m * n * k


def _ceil_log2(n: int) -> int:
    """Return ceil(log2(n)), minimum 1.

    Parameters
    ----------
    n : int
        Input value.

    Returns
    -------
    int
        ceil(log2(n)), with a floor of 1.
    """
    if n <= 1:
        return 1
    return max(math.ceil(math.log2(n)), 1)


def sort_cost(n: int) -> int:
    """FLOP cost of comparison-based sort.

    Parameters
    ----------
    n : int
        Number of elements to sort.

    Returns
    -------
    int
        Estimated FLOP count: n * ceil(log2(n)).
    """
    if n <= 0:
        return 1
    return max(n * _ceil_log2(n), 1)


def search_cost(queries: int, sorted_size: int) -> int:
    """FLOP cost of binary search.

    Parameters
    ----------
    queries : int
        Number of search queries.
    sorted_size : int
        Size of the sorted array being searched.

    Returns
    -------
    int
        Estimated FLOP count: queries * ceil(log2(sorted_size)).
    """
    if queries <= 0:
        return 1
    return max(queries * _ceil_log2(sorted_size), 1)
