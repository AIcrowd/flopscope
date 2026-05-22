"""Per-op symmetry transport rules for NumPy-protocol shape ops.

Each `transport_<op>(group_or_groups, *, ...)` function returns either a new
`SymmetryGroup` (the output's surviving symmetry) or `None` (no non-trivial
group survives — caller emits `SymmetryLossWarning`).

See `.aicrowd/superpowers/specs/2026-05-22-shape-op-symmetry-transport-design.md`
for the per-op rules and rationale.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from flopscope._perm_group import SymmetryGroup


def _stub(name: str):
    def _impl(*args, **kwargs):  # pragma: no cover - replaced per task
        raise NotImplementedError(f"transport_{name} not yet implemented")
    _impl.__name__ = f"transport_{name}"
    return _impl


# Stubs — replaced in subsequent tasks.
transport_reshape = _stub("reshape")
transport_ravel = _stub("ravel")
transport_concatenate = _stub("concatenate")
transport_stack = _stub("stack")
transport_vstack = _stub("vstack")
transport_hstack = _stub("hstack")
transport_column_stack = _stub("column_stack")
transport_split = _stub("split")
transport_hsplit = _stub("hsplit")
transport_vsplit = _stub("vsplit")
transport_dsplit = _stub("dsplit")
transport_atleast_1d = _stub("atleast_1d")
transport_atleast_2d = _stub("atleast_2d")
transport_atleast_3d = _stub("atleast_3d")
transport_broadcast_to = _stub("broadcast_to")
transport_expand_dims = _stub("expand_dims")
transport_squeeze = _stub("squeeze")
transport_flip = _stub("flip")
transport_roll = _stub("roll")
transport_tile = _stub("tile")
transport_repeat = _stub("repeat")
transport_transpose = _stub("transpose")
transport_swapaxes = _stub("swapaxes")
transport_moveaxis = _stub("moveaxis")
transport_matrix_transpose = _stub("matrix_transpose")
