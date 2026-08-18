"""Tests for symmetry-aware einsum."""

import numpy

import flopscope as flops
from flopscope._budget import BudgetContext
from flopscope._einsum import einsum
from flopscope._symmetric import SymmetricTensor, as_symmetric


class TestEinsumSymmetricInput:
    def test_symmetric_input_reduces_cost(self):
        S = as_symmetric(
            numpy.ones((10, 10, 5)),
            symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
        )
        v = numpy.ones(5)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            einsum("ijk,k->ij", S, v)
            # new direct-event model: total=550, gaming bound=num_terms*dense=2*500=1000
            dense_gaming_bound = 2 * 10 * 10 * 5 * 2  # 2 operands * dense_baseline
            assert budget.flops_used < dense_gaming_bound
            assert budget.flops_used > 0

    def test_plain_input_unchanged(self):
        A = numpy.eye(10)
        v = numpy.ones(10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            einsum("ij,j->i", A, v)
            # direct-event model with off-by-one correction:
            # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
            # = 100 + 100 - 10 = 190. First cell of each output orbit is free.
            assert budget.flops_used == 190


class TestEinsumSymmetricOutput:
    def test_symmetry_returns_symmetric_tensor(self):
        X = numpy.ones((5, 10))
        target = flops.SymmetryGroup.symmetric(axes=(0, 1))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetry=target)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == target

    def test_without_symmetry_returns_plain(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij,jk->ik", A, B)
            assert not isinstance(result, SymmetricTensor)


class TestEinsumSymmetryParam:
    def test_symmetry_param_returns_symmetric_tensor(self):
        X = numpy.ones((5, 10))
        g = flops.SymmetryGroup.symmetric(axes=(0, 1))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetry=g)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == g

    def test_symmetry_accepts_exact_group_shorthand(self):
        X = numpy.ones((5, 10))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetry=(0, 1))
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == flops.SymmetryGroup.symmetric(axes=(0, 1))


def test_total_never_exceeds_k_times_dense_baseline():
    """Even with declared symmetries, total <= k * dense_baseline always (gaming-resistance)."""
    import numpy as np

    import flopscope as fps

    A = np.zeros((4, 4, 4))
    A_sym = fps.as_symmetric(A, symmetry=(0, 1, 2))
    cost = fps.einsum_accumulation_cost("ijk,abc->ic", A_sym, A_sym)
    upper_bound = cost.num_terms * cost.dense_baseline
    assert cost.total <= upper_bound


def test_accumulation_cost_ignores_uninitialized_proxy_buffer(monkeypatch):
    """einsum_accumulation_cost must not validate its internal np.empty proxy.

    Regression for a spurious ``SymmetryError`` on a *direct*
    ``einsum_accumulation_cost`` call: ``_build_symmetric_proxy`` allocates an
    uninitialized ``np.empty`` buffer purely for its shape+symmetry metadata,
    but used to wrap it through the *validating* public ``SymmetricTensor``
    constructor. When the reused heap page held leftover non-zero floats
    (e.g. after any prior (4,4,4) float allocation in the same xdist worker),
    validation of the non-symmetric garbage raised ``SymmetryError``.

    Here we deterministically force ``np.empty`` to hand back non-symmetric
    data and assert the cost query still succeeds and is unchanged.
    """
    import numpy as np

    import flopscope as fps
    from flopscope._accumulation import _cache

    A_sym = fps.as_symmetric(np.zeros((4, 4, 4)), symmetry=(0, 1, 2))
    subs = "ijk,abc->ic"

    # Baseline on clean memory.
    _cache._accumulation_cache.cache_clear()
    expected = fps.einsum_accumulation_cost(subs, A_sym, A_sym)

    # Poison np.empty so the internal proxy buffer is non-symmetric garbage.
    real_empty = np.empty
    garbage = np.random.default_rng(0).standard_normal((4, 4, 4)).copy()
    guard = {"filling": False}

    def dirty_empty(shape, *args, **kwargs):
        buf = real_empty(shape, *args, **kwargs)
        if not guard["filling"]:
            arr = np.asarray(buf)
            if arr.shape == (4, 4, 4) and arr.dtype == np.float64:
                guard["filling"] = True
                arr[...] = garbage
                guard["filling"] = False
        return buf

    monkeypatch.setattr(np, "empty", dirty_empty)
    # Force a cache miss so the proxy is rebuilt under the poisoned allocator.
    _cache._accumulation_cache.cache_clear()

    got = fps.einsum_accumulation_cost(subs, A_sym, A_sym)  # must not raise

    assert got.total == expected.total
    assert got.num_terms == expected.num_terms
    assert got.dense_baseline == expected.dense_baseline
