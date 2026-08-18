"""Path-search wall time must not land in the free overhead bucket."""

import string

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp


def _chain(k, dim=4):
    L = string.ascii_lowercase
    subs = ",".join(L[i] + L[i + 1] for i in range(k)) + "->" + L[0] + L[k]
    ops = [fnp.array(np.ones((dim, dim))) for _ in range(k)]
    return subs, ops


def test_large_k_optimal_does_not_buy_free_wall_time():
    """k=10 'optimal' cost ~2.8s of free overhead for 2,016 billed FLOPs."""
    subs, ops = _chain(10)
    with flops.budget(10**15, quiet=True) as b:
        fnp.einsum(subs, *ops, optimize="optimal")
        overhead = b.flopscope_overhead_time_s
    assert overhead < 0.5, f"path search parked {overhead:.3f}s in the free bucket"


def test_small_k_optimal_still_allowed():
    """The corpus never exceeds 4 operands; those must keep working."""
    subs, ops = _chain(4)
    with flops.budget(10**15, quiet=True) as b:
        fnp.einsum(subs, *ops, optimize="optimal")
    assert b.flops_used > 0
