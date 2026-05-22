# tests/accumulation/_sympy_oracle.py

from __future__ import annotations

import itertools

MAX_PAIR_TOUCHES = 100_000  # |X| · |G| budget


def _restrict_stabilizer(elements, visible_positions):
    """Restrict elements that preserve V to local-V coords."""
    visible_set = set(visible_positions)
    by_key: dict[tuple[int, ...], tuple[int, ...]] = {}
    for g in elements:
        preserves = all(g.array_form[p] in visible_set for p in visible_positions)
        if not preserves:
            continue
        local_index = {gp: lp for lp, gp in enumerate(visible_positions)}
        local_arr = tuple(local_index[g.array_form[p]] for p in visible_positions)
        by_key[local_arr] = local_arr
    return tuple(by_key.values())


def _apply_perm_to_tuple(tup, perm_array_form):
    out = [0] * len(tup)
    for src in range(len(tup)):
        out[perm_array_form[src]] = tup[src]
    return tuple(out)


def _canonical_under(tup, h_local_arrays):
    if not h_local_arrays:
        return tup
    best = None
    for h in h_local_arrays:
        moved = _apply_perm_to_tuple(tup, h)
        if best is None or moved < best:
            best = moved
    return best


def sympy_brute_force_alpha(*, elements, sizes, visible_positions):
    """Brute-force α via explicit orbit enumeration. Bounded to |X|·|G| ≤ 100k."""
    x_size = 1
    for s in sizes:
        x_size *= s
    pair_touches = x_size * len(elements)
    if pair_touches > MAX_PAIR_TOUCHES:
        raise ValueError(
            f"sympy_brute_force_alpha: input too large "
            f"(|X|·|G| = {pair_touches} > budget {MAX_PAIR_TOUCHES})"
        )

    h_local = _restrict_stabilizer(elements, tuple(visible_positions))
    all_assignments = list(itertools.product(*[range(s) for s in sizes]))

    remaining = set(all_assignments)
    total = 0
    while remaining:
        rep = next(iter(remaining))
        orbit = set()
        for g in elements:
            moved = _apply_perm_to_tuple(rep, g.array_form)
            orbit.add(moved)
        for tup in orbit:
            remaining.discard(tup)
        projected_canonical = set()
        for tup in orbit:
            visible = tuple(tup[p] for p in visible_positions)
            projected_canonical.add(_canonical_under(visible, h_local))
        total += len(projected_canonical)

    return total


def sympy_brute_force_alpha_for_reduction(input_shape, axes_summed, symmetry):
    """Enumerate input → output orbit mappings via flopscope's own permutations.

    For each output orbit Q (under H = stabilizer of visible axes in G):
      α += #{ input orbit O : π_V(O) ∩ Q ≠ ∅ }

    Uses flopscope's internal _Permutation / SymmetryGroup rather than SymPy,
    to avoid index-mapping complexity between SymPy's API and tensor axes.
    """
    import itertools

    ndim = len(input_shape)
    axes_set = frozenset(axes_summed)
    visible_axes = tuple(i for i in range(ndim) if i not in axes_set)

    sym_axes = (
        symmetry.axes if symmetry.axes is not None else tuple(range(symmetry.degree))
    )

    # G elements: each is a _Permutation over local indices 0..degree-1.
    # To act on a full ndim-tuple, we lift: tensor_axis → tensor_axis unless
    # it's in sym_axes, in which case we follow the local permutation.
    local_to_tensor = sym_axes  # local_index → tensor_axis
    tensor_to_local = {ax: li for li, ax in enumerate(sym_axes)}

    def apply_g_to_point(pt, g):
        """Apply g (local _Permutation) to a full-ndim tuple `pt`."""
        arr = list(pt)
        for li, tensor_ax in enumerate(local_to_tensor):
            target_li = g.array_form[li]
            arr[local_to_tensor[target_li]] = pt[tensor_ax]
        return tuple(arr)

    g_elements = symmetry.elements()

    # Enumerate full input space and group into input orbits.
    space = list(itertools.product(*[range(d) for d in input_shape]))
    remaining = set(space)
    input_orbits = []
    while remaining:
        rep = next(iter(remaining))
        orbit = set()
        for g in g_elements:
            orbit.add(apply_g_to_point(rep, g))
        for pt in orbit:
            remaining.discard(pt)
        input_orbits.append(frozenset(orbit))

    # Build output orbits: H = elements of G that map visible_axes → visible_axes.
    # A g in G preserves visible_axes setwise iff for every visible tensor_ax,
    # g maps it to another visible tensor_ax (i.e. g(tensor_ax) ∈ visible_axes).
    visible_set = set(visible_axes)
    h_elements = [
        g
        for g in g_elements
        if all(
            local_to_tensor[g.array_form[tensor_to_local[ax]]] in visible_set
            for ax in visible_axes
            if ax in tensor_to_local
        )
    ]

    def project(pt):
        """Project full-ndim tuple to visible_axes coordinates."""
        return tuple(pt[ax] for ax in visible_axes)

    def apply_h_to_visible(vis_pt, g):
        """Apply g's action restricted to visible_axes.

        vis_pt[i] is the value at visible_axes[i].
        Under g, axis visible_axes[i] maps to local_to_tensor[g(local_of(visible_axes[i]))].
        Build result by reading where each source visible axis ends up.
        """
        vis_index = {ax: i for i, ax in enumerate(visible_axes)}
        result = list(vis_pt)
        for i, ax in enumerate(visible_axes):
            if ax in tensor_to_local:
                li = tensor_to_local[ax]
                target_li = g.array_form[li]
                target_ax = local_to_tensor[target_li]
                result[vis_index[target_ax]] = vis_pt[i]
        return tuple(result)

    output_space = list(
        itertools.product(*[range(input_shape[ax]) for ax in visible_axes])
    )
    remaining_out = set(output_space)
    output_orbits = []
    while remaining_out:
        rep = next(iter(remaining_out))
        orbit = set()
        for g in h_elements:
            orbit.add(apply_h_to_visible(rep, g))
        if not orbit:
            orbit = {rep}
        for pt in orbit:
            remaining_out.discard(pt)
        output_orbits.append(frozenset(orbit))

    # Count: for each pair (input_orbit, output_orbit), does any projected
    # input point land in the output orbit?
    alpha = 0
    for input_orbit in input_orbits:
        for output_orbit in output_orbits:
            for input_pt in input_orbit:
                if project(input_pt) in output_orbit:
                    alpha += 1
                    break

    return alpha
