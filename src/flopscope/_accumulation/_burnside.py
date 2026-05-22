"""Size-aware Burnside lemma for orbit counting under heterogeneous label dimensions.

Port of website/components/symmetry-aware-einsum-contractions/engine/sizeAware/burnside.js.

M = (1 / |G|) · Σ_{g ∈ G} ∏_{c ∈ cycles(g)} n_c

where n_c is the common size of the labels in cycle c. Within any cycle of a valid
symmetry, all labels must share a size (asserted at group-construction time elsewhere;
re-asserted here for safety).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from flopscope._perm_group import _Permutation as Permutation


def _common_size_or_throw(cycle: Sequence[int], sizes: Sequence[int]) -> int:
    n0 = sizes[cycle[0]]
    for idx in cycle:
        if sizes[idx] != n0:
            cycle_str = ",".join(str(i) for i in cycle)
            cycle_sizes = ",".join(str(sizes[i]) for i in cycle)
            raise ValueError(
                f"cycle size mismatch: labels {cycle_str} have sizes {cycle_sizes} — "
                f"a permutation can only mix labels of equal size."
            )
    return n0


def size_aware_burnside(elements: Iterable[Permutation], sizes: Sequence[int]) -> int:
    """Count orbits of `elements` acting on the assignment grid ∏ [sizes].

    Returns ``M = |X / G|`` where ``X = ∏_ℓ [sizes[ℓ]]``.
    """
    elements_tuple = tuple(elements)
    if not elements_tuple:
        raise ValueError("size_aware_burnside requires at least one group element")

    total = 0
    for g in elements_tuple:
        contribution = 1
        for cycle in g.full_cyclic_form:
            contribution *= _common_size_or_throw(cycle, sizes)
        total += contribution

    if total % len(elements_tuple) != 0:
        raise ValueError(
            f"Burnside sum {total} not divisible by |G|={len(elements_tuple)} — "
            f"group elements probably incomplete or inconsistent."
        )
    return total // len(elements_tuple)
