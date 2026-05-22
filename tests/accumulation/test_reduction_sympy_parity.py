"""SymPy brute-force orbit-mapping vs flopscope's compute_reduction_accumulation_cost."""

from __future__ import annotations

import pytest

import flopscope as fps
from flopscope._accumulation._reduction import (
    _num_output_orbits,
    compute_reduction_accumulation_cost,
)
from tests.accumulation._sympy_oracle import sympy_brute_force_alpha_for_reduction

REDUCTION_CASES = [
    # (input_shape, axes_summed, sym_spec)
    ((4, 4), (1,), ("symmetric", (0, 1))),
    ((4, 4), (0, 1), ("symmetric", (0, 1))),
    ((4, 4, 4), (2,), ("symmetric", (0, 1, 2))),
    ((4, 4, 4), (1, 2), ("symmetric", (0, 1, 2))),
    ((4, 4, 4), (2,), ("cyclic", (0, 1, 2))),
    ((3, 3, 3, 3), (2, 3), ("symmetric", (0, 1, 2, 3))),
]


def _build_sym(spec):
    kind, axes = spec
    if kind == "symmetric":
        return fps.SymmetryGroup.symmetric(axes=axes)
    if kind == "cyclic":
        return fps.SymmetryGroup.cyclic(axes=axes)
    raise ValueError(kind)


@pytest.mark.parametrize(
    "shape,axes,sym_spec",
    REDUCTION_CASES,
    ids=[f"{s}-{a}-{k}" for (s, a, (k, _)) in REDUCTION_CASES],
)
def test_python_matches_sympy_oracle(shape, axes, sym_spec):
    sym = _build_sym(sym_spec)
    flop = compute_reduction_accumulation_cost(
        input_shape=shape,
        axes_summed=axes,
        symmetry=sym,
    )
    num_orbits = _num_output_orbits(shape, axes, sym)
    # Un-apply off-by-one to recover raw α.
    raw_alpha = flop.total + num_orbits
    oracle_alpha = sympy_brute_force_alpha_for_reduction(shape, axes, sym)
    assert raw_alpha == oracle_alpha, (
        f"shape={shape}, axes={axes}, sym={sym_spec}: "
        f"python={raw_alpha} oracle={oracle_alpha}"
    )
