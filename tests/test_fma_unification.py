"""FMA=2 unification regression tests.

Asserts the post-unification state:
- `flops.fma_cost` no longer exists on the public API
- `flops.configure(fma_cost=...)` raises for unknown setting
- Dense / symmetric cost columns match for unsymmetric matmul
- 7 affected ops produce post-doubled values
"""

import pytest

import flopscope as flops


def test_fma_cost_removed_from_public_api():
    assert not hasattr(flops, "fma_cost"), (
        "flops.fma_cost should be removed in FMA=2 unification"
    )


def test_configure_fma_cost_raises():
    with pytest.raises((ValueError, KeyError, TypeError)):
        flops.configure(fma_cost=2)


def test_fma_cost_constant_removed():
    """The private FMA_COST constant + its module should be gone.

    The pyright/ruff suppressions are necessary because we deliberately
    import a deleted module to verify the deletion.
    """
    import importlib

    with pytest.raises(ImportError):
        importlib.import_module("flopscope._cost_model")


def test_hamming_cost_doubled():
    """hamming(n=400) should charge 2*n = 800 (was n=400 under FMA=1)."""
    import flopscope.numpy as fnp

    with flops.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        _ = fnp.hamming(400)
    assert bc.flops_used == 800, f"hamming(400) charged {bc.flops_used}, expected 800"


def test_hanning_cost_doubled():
    """hanning(n=400) should charge 2*n = 800."""
    import flopscope.numpy as fnp

    with flops.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        _ = fnp.hanning(400)
    assert bc.flops_used == 800, f"hanning(400) charged {bc.flops_used}, expected 800"


def test_polyval_cost_doubled():
    """polyval(c, x) with len(c)=4, x.shape=(10,): m=10, deg=3.
    Was m*deg = 30, now 2*m*deg = 60.
    """
    import numpy as np

    import flopscope.numpy as fnp

    with flops.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        _ = fnp.polyval(np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(10))
    assert bc.flops_used == 60, f"polyval charged {bc.flops_used}, expected 60"


def test_multi_dot_cost_doubled():
    """multi_dot([A,B,C]) for (M,K)x(K,N)x(N,P) with M=4,K=3,N=2,P=5:
    optimal order pairs (A@B) first → 4*3*2 = 24, then (AB)@C → 4*2*5 = 40.
    Under FMA=1: 24 + 40 = 64. Under FMA=2: 2*(64) = 128.
    """
    import numpy as np

    import flopscope.numpy as fnp

    A = np.zeros((4, 3))
    B = np.zeros((3, 2))
    C = np.zeros((2, 5))
    with flops.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        _ = fnp.linalg.multi_dot([A, B, C])
    assert bc.flops_used == 128, f"multi_dot charged {bc.flops_used}, expected 128"


def test_norm_cost_doubled():
    """linalg.norm(x) for x.shape=(8,): numel=8. Was 8, now 2*8 = 16."""
    import numpy as np

    import flopscope.numpy as fnp

    with flops.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        _ = fnp.linalg.norm(np.zeros(8))
    assert bc.flops_used == 16, f"linalg.norm charged {bc.flops_used}, expected 16"


def test_vector_norm_cost_doubled():
    """linalg.vector_norm(x) for x.shape=(8,): was 8, now 16."""
    import numpy as np

    import flopscope.numpy as fnp

    with flops.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        _ = fnp.linalg.vector_norm(np.zeros(8))
    assert bc.flops_used == 16, (
        f"linalg.vector_norm charged {bc.flops_used}, expected 16"
    )


def test_matrix_norm_cost_doubled():
    """linalg.matrix_norm(x) for x.shape=(4,4): numel=16. Was 16, now 2*16 = 32."""
    import numpy as np

    import flopscope.numpy as fnp

    with flops.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        _ = fnp.linalg.matrix_norm(np.zeros((4, 4)))
    assert bc.flops_used == 32, (
        f"linalg.matrix_norm charged {bc.flops_used}, expected 32"
    )


def test_dense_flops_equals_flops_for_unsymmetric_matmul():
    """For unsymmetric matmul, dense_flop_cost should equal flop_cost
    (both produced by α/M model with no symmetry).
    """
    import numpy as np

    import flopscope.numpy as fnp

    for n in [2, 4, 10]:
        A = np.zeros((n, n))
        B = np.zeros((n, n))
        with flops.BudgetContext(flop_budget=10**12, quiet=True):
            _, info = fnp.einsum_path("ij,jk->ik", A, B)
        for step in info.steps:
            assert step.dense_flop_cost == step.flop_cost, (
                f"n={n}: dense={step.dense_flop_cost} != flop_cost={step.flop_cost}"
            )


def test_dense_flops_exceeds_flops_for_symmetric_matmul():
    """For symmetric matmul, dense_flop_cost >= flop_cost (savings >= 0)."""
    import numpy as np

    import flopscope.numpy as fnp

    n = 4
    A = flops.as_symmetric(np.zeros((n, n)), symmetry=(0, 1))
    B = np.zeros((n, n))
    with flops.BudgetContext(flop_budget=10**12, quiet=True):
        _, info = fnp.einsum_path("ij,jk->ik", A, B)
    for step in info.steps:
        assert step.dense_flop_cost >= step.flop_cost, (
            f"dense={step.dense_flop_cost} < flop_cost={step.flop_cost}; impossible"
        )


# ── Bug A / B regression tests (renderer fix) ──────────────────────────


def test_bug_a_unsymmetric_chain_savings_header_zero():
    """Bug A regression: for unsymmetric 3-matrix chain, header Savings must be 0.0%.

    Before the fix, naive_cost was computed via the legacy helpers.flop_count over
    ALL labels (single-shot k-way contraction) which gave a different model than
    the per-step dense_flop_cost.  The header showed e.g. "56.2% saved" while
    per-step columns both showed 0.0%, because naive_cost and dense_flop_cost came
    from different models.  After the fix, naive_cost = sum(dense_flop_cost), so
    both columns agree at 0.0%.
    """
    import numpy as np

    import flopscope.numpy as fnp

    A = np.ones((4, 4))
    B = np.ones((4, 4))
    C = np.ones((4, 4))
    with flops.BudgetContext(flop_budget=10**12, quiet=True):
        _, info = fnp.einsum_path("ij,jk,kl->il", A, B, C)
    rendered = str(info)
    assert "Savings:  0.0%" in rendered or "Savings: 0.0%" in rendered, (
        f"Bug A: header Savings should be 0.0% for unsymmetric chain.\nrendered:\n{rendered}"
    )


def test_bug_b_symmetric_frobenius_cost_matches_direct():
    """Bug B regression: einsum_path for S3-symmetric Frobenius inner matches
    einsum_accumulation_cost on all three cost surfaces.

    Before the fix, FlopscopePathInfo.from_inner() only synced step.flop_cost
    when accumulation.per_step was non-empty (populated by _walk_path_and_aggregate
    for 3+-op paths).  For 2-op binary einsums, aggregate_einsum() returns a flat
    AccumulationCost with per_step=(), so step.flop_cost stayed at the dense 127
    even though info.optimized_cost correctly returned 39 via accumulation.total.
    The renderer header and per-step "flops" column both showed the wrong 127.

    After the fix, from_inner() handles the single-step case by setting
    steps[0].flop_cost = accumulation.total, and also updates inner.optimized_cost
    and inner.speedup so the renderer is consistent.
    """
    import numpy as np

    import flopscope.numpy as fnp

    T = flops.as_symmetric(np.zeros((4, 4, 4)), symmetry=(0, 1, 2))
    direct = flops.einsum_accumulation_cost("ijk,ijk->", T, T).total
    with flops.BudgetContext(flop_budget=10**12, quiet=True):
        _, info = fnp.einsum_path("ijk,ijk->", T, T)
    # All three cost surfaces must agree.
    assert info.optimized_cost == direct, (
        f"Bug B: info.optimized_cost={info.optimized_cost} != direct={direct}"
    )
    step_sum = sum(s.flop_cost for s in info.steps)
    assert step_sum == direct, (
        f"Bug B: sum(steps.flop_cost)={step_sum} != direct={direct}"
    )
    assert info.accumulation.total == direct, (
        f"Bug B: info.accumulation.total={info.accumulation.total} != direct={direct}"
    )
    # check_consistency must pass (all three surfaces in agreement).
    assert info.check_consistency()
    # Renderer must show the symmetric cost, not the dense fallback.
    rendered = str(info)
    assert f"Optimized cost (flopscope):  {direct}" in rendered, (
        f"Bug B: renderer header shows wrong cost:\n{rendered}"
    )
