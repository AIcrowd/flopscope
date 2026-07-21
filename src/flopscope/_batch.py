"""Shared gufunc batch-count primitive.

The batch (loop) multiplier for a gufunc-style op is the product of the
broadcast of every operand's leading dims — the dims LEFT of its core.
Deriving it from one operand alone under-bills the batch/broadcast family.
"""

from __future__ import annotations

import math

import numpy as _np


def _broadcast_batch(*shapes: tuple[int, ...], core_ranks: tuple[int, ...]) -> int:
    """Product of the broadcast loop dims across gufunc operands.

    Each ``shapes[i]``'s last ``core_ranks[i]`` dims are its core; the
    remaining leading dims broadcast together (numpy gufunc semantics).
    Raises ``ValueError`` on incompatible loop dims (mirrors numpy).
    """
    if not shapes:
        return 1
    loops = [tuple(s[: len(s) - r]) for s, r in zip(shapes, core_ranks, strict=True)]
    return int(math.prod(_np.broadcast_shapes(*loops)))
