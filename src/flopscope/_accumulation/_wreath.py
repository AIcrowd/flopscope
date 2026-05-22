"""Direct enumerator for the wreath product ∏_i (H_i ≀ S_{m_i}).

Port of website/components/symmetry-aware-einsum-contractions/engine/wreath.js.

`i` indexes identical-operand groups (operands sharing the same name).
`H_i` is each operand's declared axis symmetry on its own axes.
`m_i` is the number of copies of operand i.

Each wreath element is a row permutation σ on the U-vertices, paired with
factorization metadata that names the (outer-S_{m_i}, base-H_i) decomposition
for diagnostic display.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from flopscope._perm_group import SymmetryGroup, _dimino
from flopscope._perm_group import _Permutation as Permutation


@dataclass(frozen=True)
class WreathElement:
    """One element of ∏_i (H_i ≀ S_{m_i}). Carries the row permutation and provenance."""

    row_perm: Permutation
    factorization: dict[str, Any]


def enumerate_h(sym: Any, rank: int) -> Iterator[Permutation]:
    """Enumerate every element of the rank-`rank` group described by `sym`.

    `sym` may be:
      - None or 'none' → identity only
      - 'symmetric' → S_rank
      - 'cyclic' → C_rank
      - 'dihedral' → D_rank
      - a SymmetryGroup object → use its elements directly
      - a dict with {'type': 'custom', 'generators': [...]} → custom group

    Returns rank-`rank` permutations (operate on positions 0..rank-1).
    """
    identity = Permutation.identity(rank)
    if sym is None or sym == "none":
        yield identity
        return

    if isinstance(sym, SymmetryGroup):
        axes = getattr(sym, "axes", None)
        for el in sym.elements():
            arr = list(range(rank))
            if axes is not None:
                # elements() yields permutations on positions 0..len(axes)-1;
                # map them to the actual tensor axis positions via sym.axes.
                for local_i, local_j in enumerate(el.array_form):
                    if local_i < len(axes) and local_j < len(axes):
                        from_axis = axes[local_i]
                        to_axis = axes[local_j]
                        if from_axis < rank and to_axis < rank:
                            arr[from_axis] = to_axis
            else:
                # No axes annotation; embed at zero offset (legacy path).
                for i, j in enumerate(el.array_form):
                    if i < rank:
                        arr[i] = j if j < rank else i
            yield Permutation(arr)
        return

    if sym == "symmetric" or (isinstance(sym, dict) and sym.get("type") == "symmetric"):
        gens = []
        for k in range(rank - 1):
            arr = list(range(rank))
            arr[k], arr[k + 1] = arr[k + 1], arr[k]
            gens.append(Permutation(arr))
        if not gens:
            yield identity
            return
        for el in _dimino(tuple(gens)):
            yield el
        return

    if sym == "cyclic" or (isinstance(sym, dict) and sym.get("type") == "cyclic"):
        if rank <= 1:
            yield identity
            return
        rotation = list(range(1, rank)) + [0]
        for el in _dimino((Permutation(rotation),)):
            yield el
        return

    if sym == "dihedral" or (isinstance(sym, dict) and sym.get("type") == "dihedral"):
        if rank <= 2:
            for el in enumerate_h("symmetric", rank):
                yield el
            return
        rot = list(range(1, rank)) + [0]
        ref = list(range(rank))
        for k in range(rank // 2):
            ref[k], ref[rank - 1 - k] = ref[rank - 1 - k], ref[k]
        for el in _dimino((Permutation(rot), Permutation(ref))):
            yield el
        return

    raise ValueError(f"unsupported symmetry declaration: {sym!r}")


def _outer_permutations(m: int) -> Iterator[list[int]]:
    """Yield every permutation array of length m (S_m). Used for the outer factor."""
    for perm_tuple in itertools.permutations(range(m)):
        yield list(perm_tuple)


def _flatten_factor_to_row_perm(
    group: Sequence[int],
    base_tuple: Sequence[Permutation],
    top_perm: Sequence[int],
    u_offsets: Sequence[int],
    axis_ranks: Sequence[int],
    n_u: int,
) -> list[int]:
    """Build row-perm contribution for one identical-group factor.

    Mirrors JS flattenFactorToRowPerm: arr[to] = from (inverse representation
    consistent with the JS engine).
      top_perm[j] = new position of copy j within the group.
      base_tuple[j] = axis permutation applied to copy j's axes before relocation.
    """
    arr = list(range(n_u))
    for j in range(len(group)):
        p = group[j]
        rank = axis_ranks[p]
        new_j = top_perm[j]
        new_p = group[new_j]
        h = base_tuple[j]
        for a in range(rank):
            from_idx = u_offsets[p] + a
            to_idx = u_offsets[new_p] + h.array_form[a]
            arr[to_idx] = from_idx
    return arr


def enumerate_wreath(
    *,
    identical_groups: Sequence[Sequence[int]],
    per_op_symmetry: Sequence[Any],
    axis_ranks: Sequence[int],
    u_offsets: Sequence[int],
) -> Iterator[WreathElement]:
    """Iterate ∏_i (H_i ≀ S_{m_i}) and yield row permutations on the U-vertices.

    `identical_groups`: tuple of operand-index tuples, each grouping copies of
        the same operand.
    `per_op_symmetry`: parallel to operand index — declared H_i for each operand.
    `axis_ranks`: parallel — number of axes per operand.
    `u_offsets`: parallel — starting U-vertex index for each operand.
    """
    total_u = sum(axis_ranks)

    # For each identical-group, build a list of (arr, factor_meta) pairs.
    per_group_options: list[list[tuple[list[int], dict]]] = []
    for grp in identical_groups:
        m = len(grp)
        # Base H_i — all copies in the group share the same declared symmetry.
        base_sym = per_op_symmetry[grp[0]]
        base_rank = axis_ranks[grp[0]]
        h_elements = list(enumerate_h(base_sym, base_rank))

        group_options: list[tuple[list[int], dict]] = []
        for top_perm in _outer_permutations(m):
            for base_tuple in itertools.product(h_elements, repeat=m):
                arr = _flatten_factor_to_row_perm(
                    grp, base_tuple, top_perm, u_offsets, axis_ranks, total_u
                )
                group_options.append(
                    (
                        arr,
                        {
                            "group": tuple(grp),
                            "outer": tuple(top_perm),
                            "base_arrs": tuple(tuple(h.array_form) for h in base_tuple),
                        },
                    )
                )
        per_group_options.append(group_options)

    if not per_group_options:
        # No operands: just identity.
        yield WreathElement(
            row_perm=Permutation.identity(total_u),
            factorization={"groups": ()},
        )
        return

    # Cartesian product across groups: merge contributions into a single row perm.
    # Different groups touch disjoint U-vertex ranges, so merge = overwrite non-identity.
    for combo in itertools.product(*per_group_options):
        row_perm_arr = list(range(total_u))
        for arr, _factor in combo:
            for i in range(total_u):
                if arr[i] != i:
                    row_perm_arr[i] = arr[i]
        factorization = {
            "groups": tuple(factor for _, factor in combo),
        }
        yield WreathElement(
            row_perm=Permutation(row_perm_arr),
            factorization=factorization,
        )
