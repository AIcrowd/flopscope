"""SymmetricTensor.__new__ must not mint a free, unvalidated symmetry tag."""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope import SymmetricTensor
from flopscope import SymmetryGroup
from flopscope._perm_group import _Permutation as P


def billed(fn) -> int:
    with flops.budget(10**15, quiet=True) as b:
        fn()
        return b.flops_used


def test_constructing_over_asymmetric_data_is_refused_or_charged():
    raw = np.random.default_rng(0).random((6, 6))
    sym = SymmetryGroup(P([1, 0]), axes=(0, 1))
    with flops.budget(10**12, quiet=True) as b:
        try:
            SymmetricTensor(raw, symmetry=sym)
        except flops.SymmetryError:
            return                      # refusing is an acceptable outcome
        assert b.flops_used > 0, "validated a tag without charging for it"


def test_forged_tag_does_not_discount_downstream_pointwise():
    raw = np.random.default_rng(0).random((6, 6))
    sym = SymmetryGroup(P([1, 0]), axes=(0, 1))
    honest = billed(lambda: fnp.sin(fnp.array(raw)))
    try:
        t = SymmetricTensor(raw, symmetry=sym)
    except flops.SymmetryError:
        pytest.skip("construction refused outright, which also closes it")
    assert billed(lambda: fnp.sin(t)) == honest


def test_genuinely_symmetric_data_still_works():
    raw = np.random.default_rng(1).random((6, 6))
    raw = raw + raw.T                    # actually symmetric
    with flops.budget(10**12, quiet=True):
        t = SymmetricTensor(raw, symmetry=SymmetryGroup(P([1, 0]), axes=(0, 1)))
    assert t.symmetry is not None
