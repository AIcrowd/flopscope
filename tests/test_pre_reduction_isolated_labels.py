"""Regression tests for per-binary-step pre-reduction of isolated summed labels.

Closes AIcrowd/flopscope#55.  When a label is summed in a binary step AND
appears in only one of the two inputs, that operand is pre-reduced along the
isolated axes before the main contraction.  Mirrors PyTorch's sumproduct_pair.
"""

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp


def _einsum_path_info(subscripts, *operands):
    with flops.BudgetContext(10**12, quiet=True):
        _, info = fnp.einsum_path(subscripts, *operands)
    return info


def test_wilson_bilinear_trace_drops_cost():
    """Wilson #55 canonical: einsum('ik,jl->ij', A, B) for dense A, B, n=10.

    With pre-reduction: pre-reduce A on {k}, pre-reduce B on {l}, then
    outer product i,j->ij.  Total cost should be dramatically lower than
    the dense ~20,000 baseline.
    """
    n = 10
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    info = _einsum_path_info("ik,jl->ij", A, B)
    assert len(info.steps) == 1, info.steps
    pre = info.steps[0].pre_reductions
    assert len(pre) == 2, f"expected 2 pre-reductions, got {len(pre)}: {pre}"
    removed = {(p.operand_index, p.removed_labels) for p in pre}
    assert removed == {(0, ("k",)), (1, ("l",))}, removed
    # Cost should be dramatically below dense ~20,000
    assert info.optimized_cost < 1000, info.optimized_cost
    # Sanity: each pre-reduction cost is positive and below 2*n*n
    for p in pre:
        assert 0 < p.cost <= 2 * n * n, p


def test_cross_s2_isolated_label():
    """einsum('ij,k->ik', A, v) with A having S_2 on (0, 1).

    j is summed and appears only in A.  Pre-reduce A on axis 1 (label j).
    """
    n = 6
    rng = np.random.default_rng(0)
    A = flops.symmetrize(rng.standard_normal((n, n)), symmetry=(0, 1))
    v = rng.standard_normal((n,))
    info = _einsum_path_info("ij,k->ik", A, v)
    pre = info.steps[0].pre_reductions
    assert len(pre) == 1, pre
    assert pre[0].operand_index == 0
    assert pre[0].removed_labels == ("j",)
    assert pre[0].surviving_subscript == "i"


def test_multi_step_bilinear_trace_3():
    """einsum('ik,jl,mn->ijm', A, B, C) dense, n=5.

    Verify at least 2 pre-reductions fire across the path's binary steps.
    """
    n = 5
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    C = rng.standard_normal((n, n))
    info = _einsum_path_info("ik,jl,mn->ijm", A, B, C)
    total_pre_reductions = sum(len(s.pre_reductions) for s in info.steps)
    assert total_pre_reductions >= 2, (
        f"expected >=2 pre-reductions across steps, got {total_pre_reductions}; "
        f"steps: {[s.pre_reductions for s in info.steps]}"
    )


def test_no_isolation_unchanged():
    """einsum('ij,jk->ik', A, B): no isolation (j is shared).

    Step's pre_reductions must remain empty tuple.
    """
    n = 4
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    info = _einsum_path_info("ij,jk->ik", A, B)
    for s in info.steps:
        assert s.pre_reductions == (), s.pre_reductions


def test_pre_reduction_with_declared_symmetry():
    """A is S_2-symmetric on (0, 1); einsum('ij,k->ik', A, v).

    Pre-reduction of A on axis 1 should use S_2-aware cost
    (reduction_accumulation_cost honors declared symmetry).
    Cost should be strictly less than dense n*(n-1).
    """
    n = 6
    rng = np.random.default_rng(0)
    A = flops.symmetrize(rng.standard_normal((n, n)), symmetry=(0, 1))
    v = rng.standard_normal((n,))
    info = _einsum_path_info("ij,k->ik", A, v)
    pre = info.steps[0].pre_reductions
    assert len(pre) == 1
    dense_reduce_cost = n * (n - 1)
    assert pre[0].cost < dense_reduce_cost, (
        f"pre-reduction cost {pre[0].cost} not lower than dense {dense_reduce_cost}"
    )


def test_pre_reduced_operand_symmetry_propagates():
    """T is S_3 on (0,1,2); einsum('abc,d->abd', T, v).

    c isolated to T, summed.  Pre-reduce c: surviving symmetry should be
    reduce_group(S_3, ndim=3, axis=2) = S_2{a,b}.

    Verify the PreReduction.reduced_symmetry_fingerprint encodes a
    non-trivial group (S_2 has order 2).
    """
    n = 4
    rng = np.random.default_rng(0)
    T = flops.symmetrize(rng.standard_normal((n, n, n)), symmetry=(0, 1, 2))
    v = rng.standard_normal((n,))
    info = _einsum_path_info("abc,d->abd", T, v)
    pre = info.steps[0].pre_reductions
    assert len(pre) == 1
    assert pre[0].removed_labels == ("c",)
    assert pre[0].reduced_symmetry_fingerprint is not None
    assert pre[0].reduced_symmetry_fingerprint != (None,)


def test_renderer_shows_pre_reduction_subrows():
    """str(info) for bilinear-trace must contain 'pre-reduce' substring."""
    n = 5
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    info = _einsum_path_info("ik,jl->ij", A, B)
    rendered = str(info)
    assert "pre-reduce" in rendered, (
        f"rendered table missing 'pre-reduce' sub-row:\n{rendered}"
    )
