"""SymmetricTensor.__new__ must not mint a free, unvalidated symmetry tag.

Two routes still reach an unvalidated tag despite this fix -- both
in-process-only, neither remotely reachable (see the module-level comments
on `SymmetricTensor.__new__` and `_TRUSTED_SYMMETRY_WRAPPER_CODES` in
src/flopscope/_symmetric.py for the full rationale). They are pinned below
as `strict=True` xfail tests: if a future change closes either one, the
xfail turns into a failure, which is the signal to update this pin (not to
silently let it pass).
"""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope import SymmetricTensor, SymmetryGroup
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
            return  # refusing is an acceptable outcome
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
    raw = raw + raw.T  # actually symmetric
    with flops.budget(10**12, quiet=True):
        t = SymmetricTensor(raw, symmetry=SymmetryGroup(P([1, 0]), axes=(0, 1)))
    assert t.symmetry is not None


@pytest.mark.xfail(
    reason="known in-process bypass: wrap_with_symmetry runs only a structural "
    "check, and __new__ trusts it by code-object identity. Not remotely "
    "reachable (absent from REGISTRY and from the flopscope/fnp namespaces). "
    "Closing it needs a capability token threaded through 34 internal call "
    "sites, or a content check inside wrap_with_symmetry itself.",
    strict=True,
)
def test_wrap_with_symmetry_does_not_mint_an_unvalidated_tag():
    from flopscope._symmetry_utils import wrap_with_symmetry

    raw = np.random.default_rng(0).random((6, 6))
    sym = SymmetryGroup(P([1, 0]), axes=(0, 1))
    honest = billed(lambda: fnp.sin(fnp.array(raw)))
    with flops.budget(10**12, quiet=True) as b:
        try:
            t = wrap_with_symmetry(raw, sym)
        except flops.SymmetryError:
            return  # refusing is an acceptable outcome
        assert b.flops_used > 0, "validated a tag without charging for it"
    assert billed(lambda: fnp.sin(t)) == honest, "forged tag discounted a downstream op"


@pytest.mark.xfail(
    reason="known in-process bypass: _called_from_wrapper walks the ENTIRE "
    "call stack for any @_counted_wrapper frame, so a SymmetricTensor "
    "constructed inside a participant callback (fnp.apply_along_axis, "
    "fnp.piecewise, ...) inherits the host op's trust and is never "
    "validated. Not remotely reachable (these callback ops raise "
    "RemoteCallbackError on the server backend). Closing it needs the trust "
    "check to stop at the callback boundary, not walk through it.",
    strict=True,
)
def test_symmetric_tensor_constructed_inside_a_host_op_callback_is_validated():
    raw = np.random.default_rng(0).random((6, 6))
    sym = SymmetryGroup(P([1, 0]), axes=(0, 1))
    honest = billed(lambda: fnp.sin(fnp.array(raw)))

    box: dict = {}

    def callback(_row):
        try:
            box["tensor"] = SymmetricTensor(raw, symmetry=sym)
        except flops.SymmetryError as exc:
            box["error"] = exc
        return 0.0

    with flops.budget(10**12, quiet=True):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fnp.apply_along_axis(callback, 1, fnp.array(np.zeros((1, 3))))

    if "error" in box:
        return  # refusing is an acceptable outcome
    t = box["tensor"]
    assert billed(lambda: fnp.sin(t)) == honest, (
        "forged tag (minted inside a host-op callback) discounted a downstream op"
    )
