"""Tests for the current symmetry API surface."""

import pickle

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._budget import BudgetContext
from flopscope._ndarray import FlopscopeArray
from flopscope._symmetric import SymmetricTensor, as_symmetric, is_symmetric, symmetrize
from flopscope._weights import load_weights, reset_weights
from flopscope.errors import SymmetryError, SymmetryLossWarning


def _s2(*axes):
    return flops.SymmetryGroup.symmetric(axes=axes)


def test_as_symmetric_exposes_only_symmetry_object():
    group = _s2(0, 1)
    tensor = as_symmetric(np.eye(3), symmetry=group)
    assert tensor.symmetry == group
    assert not hasattr(tensor, "symmetry_info")
    assert not hasattr(tensor, "symmetric_axes")


def test_array_finalize_is_conservative():
    tensor = as_symmetric(np.eye(3), symmetry=_s2(0, 1))
    finalized = np.asarray(tensor).view(SymmetricTensor)
    assert finalized.symmetry is None


def test_copy_preserves_symmetry_and_shape_methods_now_transport(recwarn):
    """After issue #68, shape methods either transport symmetry or warn on drop."""
    tensor = as_symmetric(np.eye(3), symmetry=_s2(0, 1))

    copied = tensor.copy()
    # reshape((-1,)) collapses S_2 on (0,1) of (3,3) into a 1-D shape — drops with warning.
    reshaped = tensor.reshape(-1)
    raveled = tensor.ravel()
    flattened = tensor.flatten()
    cast = tensor.astype(np.float32)

    assert isinstance(copied, SymmetricTensor)
    assert copied.symmetry == tensor.symmetry
    # reshape/ravel/flatten to 1-D drop the multi-axis group.
    assert not isinstance(reshaped, SymmetricTensor)
    assert not isinstance(raveled, SymmetricTensor)
    assert not isinstance(flattened, SymmetricTensor)
    # astype is not in the shape-op transport scope; remains as-is.
    assert not isinstance(cast, SymmetricTensor)
    # The three shape ops should have emitted SymmetryLossWarning.
    from flopscope.errors import SymmetryLossWarning

    sym_warnings = [
        w for w in recwarn.list if issubclass(w.category, SymmetryLossWarning)
    ]
    assert len(sym_warnings) == 3  # reshape, ravel, flatten each warn once


def test_astype_bills_like_copy_and_stays_free_for_the_true_noop():
    """SymmetricTensor.astype must route through the counted backend (the
    astype/asarray Option B billing fix) -- before that fix it bypassed
    billing entirely via raw numpy (``np.asarray(self).astype(...)``), so it
    stayed silently free regardless of the shared astype weight.
    """
    tensor = as_symmetric(np.eye(20, dtype=np.float64), symmetry=_s2(0, 1))
    n = np.asarray(tensor).size  # 400

    load_weights()
    try:
        # Real cast (default copy=True): heavier(float64, float32) = float64,
        # rate 2.0 -> n * 2.0. Matches the top-level astype/copy formula
        # exactly -- confirmed against SymmetricTensor.copy() below.
        with BudgetContext(flop_budget=10**12, quiet=True) as b:
            cast = tensor.astype(np.float32)
        assert not isinstance(cast, SymmetricTensor)
        assert b.flops_used == n * 2

        # Same-dtype astype with the default copy=True is still a real copy
        # and must bill exactly what .copy() bills.
        with BudgetContext(flop_budget=10**12, quiet=True) as b:
            same_dtype_copy = tensor.astype(tensor.dtype, copy=True)
        with BudgetContext(flop_budget=10**12, quiet=True) as b2:
            plain_copy = tensor.copy()
        assert b.flops_used == b2.flops_used == n * 2
        assert isinstance(plain_copy, SymmetricTensor)  # copy() preserves symmetry
        assert not isinstance(same_dtype_copy, SymmetricTensor)  # astype does not

        # The one true no-op -- copy=False with an unchanged dtype -- is the
        # only case that stays free.
        with BudgetContext(flop_budget=10**12, quiet=True) as b:
            noop = tensor.astype(tensor.dtype, copy=False)
        assert b.flops_used == 0
        assert not isinstance(noop, SymmetricTensor)
        assert np.shares_memory(np.asarray(noop), np.asarray(tensor))
    finally:
        reset_weights()


def test_symmetric_tensor_squeeze_preserves_block(recwarn):
    """Squeezing an axis outside the block now preserves symmetry."""
    data = np.eye(3).reshape(1, 3, 3)
    tensor = as_symmetric(data, symmetry=_s2(1, 2))
    squeezed = tensor.squeeze(axis=0)
    assert isinstance(squeezed, SymmetricTensor)
    assert set(squeezed.symmetry.axes or ()) == {0, 1}


def test_transpose_remaps_symmetry():
    tensor = symmetrize(np.arange(27.0).reshape(3, 3, 3), symmetry=_s2(0, 2))
    out = tensor.transpose((2, 1, 0))
    assert isinstance(out, SymmetricTensor)
    assert out.symmetry == _s2(0, 2)


def test_swapaxes_remaps_symmetry():
    tensor = symmetrize(np.arange(27.0).reshape(3, 3, 3), symmetry=_s2(0, 2))
    out = tensor.swapaxes(0, 1)
    assert isinstance(out, SymmetricTensor)
    assert out.symmetry == _s2(1, 2)


def test_plain_slices_drop_symmetry():
    tensor = as_symmetric(np.eye(4), symmetry=_s2(0, 1))
    with pytest.warns(SymmetryLossWarning):
        sliced_fwd = tensor[1:, 1:]
    assert isinstance(sliced_fwd, FlopscopeArray)
    assert not isinstance(sliced_fwd, SymmetricTensor)
    with pytest.warns(SymmetryLossWarning):
        sliced_rev = tensor[::-1, ::-1]
    assert isinstance(sliced_rev, FlopscopeArray)
    assert not isinstance(sliced_rev, SymmetricTensor)


def test_is_symmetric_checks_declared_group():
    assert is_symmetric(np.eye(3), symmetry=_s2(0, 1))
    assert not is_symmetric(np.array([[1, 2], [3, 4]]), symmetry=_s2(0, 1))


def test_as_symmetric_accepts_exact_group_and_young_group():
    rng = np.random.default_rng(7)
    data = rng.standard_normal((4, 4, 3, 3))
    data = (data + data.transpose(1, 0, 2, 3)) / 2
    data = (data + data.transpose(0, 1, 3, 2)) / 2
    tensor = as_symmetric(data, symmetry=((0, 1), (2, 3)))
    assert tensor.symmetry == flops.SymmetryGroup.young(blocks=((0, 1), (2, 3)))


def test_rejects_non_symmetric_data():
    rng = np.random.default_rng(99)
    data = rng.standard_normal((5, 5))
    with pytest.raises(SymmetryError):
        as_symmetric(data, symmetry=_s2(0, 1))


def test_pickle_roundtrip_keeps_symmetry():
    tensor = as_symmetric(np.eye(3), symmetry=_s2(0, 1))
    loaded = pickle.loads(pickle.dumps(tensor))
    assert isinstance(loaded, SymmetricTensor)
    assert loaded.symmetry == tensor.symmetry


def test_legacy_pickle_payload_is_rejected():
    tensor = as_symmetric(np.eye(3), symmetry=_s2(0, 1))
    payload = tensor.__reduce__()
    legacy_state = payload[2] + ([(0, 1)],)
    rebuilt = SymmetricTensor(np.zeros((3, 3)), symmetry=_s2(0, 1))
    with pytest.raises(ValueError, match="legacy symmetry payload"):
        rebuilt.__setstate__(legacy_state)


def test_public_exports_only_current_surface():
    # Symmetry primitives live at the top-level ``flopscope`` package.
    assert hasattr(flops, "SymmetryGroup")
    assert hasattr(flops, "SymmetricTensor")
    assert hasattr(flops, "as_symmetric")
    assert hasattr(flops, "symmetrize")
    assert not hasattr(flops, "PermutationGroup")
    assert not hasattr(flops, "Permutation")
    assert not hasattr(flops, "Cycle")
    assert not hasattr(flops, "SymmetryInfo")


def test_symmetrize_uses_symmetry_keyword():
    group = _s2(0, 1)
    base = np.arange(16.0).reshape(4, 4)
    result = symmetrize(base, symmetry=group)
    assert isinstance(result, SymmetricTensor)
    assert result.symmetry == group
    assert result.is_symmetric()


def test_random_symmetric_uses_group_object():
    tensor = fnp.random.symmetric((4, 4), _s2(0, 1))
    assert isinstance(tensor, SymmetricTensor)
    assert tensor.symmetry == _s2(0, 1)
    assert tensor.is_symmetric()


def test_flops_module_no_longer_exports_symmetry_info():
    assert not hasattr(flops.accounting, "SymmetryInfo")
    assert "SymmetryInfo" not in flops.accounting.__all__


def test_einsum_output_uses_symmetry_keyword():
    x = np.ones((5, 3))
    with BudgetContext(flop_budget=10**8, quiet=True) as budget:
        cov = fnp.einsum("ki,kj->ij", x, x, symmetry=_s2(0, 1))
        cost = budget.flops_used

    with BudgetContext(flop_budget=10**8, quiet=True) as budget:
        dense = fnp.einsum("ki,kj->ij", x, np.ones((5, 3)), symmetry=_s2(0, 1))
        dense_cost = budget.flops_used

    assert isinstance(cov, SymmetricTensor)
    assert isinstance(dense, SymmetricTensor)
    assert cov.symmetry == _s2(0, 1)
    assert dense.symmetry == _s2(0, 1)
    assert cost < dense_cost
