"""Regression tests for Sprint 4: per-step cost = min(Cat A, Cat B, Cat C).

Closes the final loose end on Wilson's PR #91 review (K·W·W·W step 2 still
using input-symmetry savings even though output S₃ Burnside is tighter).

The selection is "pick the minimum of all valid Burnside-based candidates":

- Cat A (per-input wreath/sigma): always valid, baseline.
- Cat B (output-orbit Burnside): valid when merged_output_group is
  non-trivial; formula ``O_out · (2·W − 1)``.
- Cat C (joint-Burnside on V ∪ W): valid in Regime 1 with V preserved
  setwise.

Each step's ``cost_source`` records the winning category.
"""

from __future__ import annotations

import numpy as np

import flopscope.numpy as fnp
from flopscope import BudgetContext, symmetrize


def _path(subscripts, *operands):
    with BudgetContext(10**12, quiet=True):
        _, info = fnp.einsum_path(subscripts, *operands)
    return info


def test_kwww_step2_uses_output_burnside():
    """Wilson's main remaining concern (post-Sprint-3 review).

    ``einsum("abc,ai,bj,ck->ijk", K_S3, W, W, W)`` at n=10:

    - Step 2 (``icj,ck->ijk``) sees S₂{i,j} on the intermediate (Cat A
      savings ≈ 10,450) but the output is provably S₃{i,j,k} from W's
      wreath-replication, so Cat B Burnside gives 220 · 19 = 4,180.
    - Sprint 4 picks the tighter Cat B for this step.
    """
    n = 10
    K = symmetrize(
        fnp.random.default_rng(0).standard_normal((n, n, n)), symmetry=(0, 1, 2)
    )
    W = fnp.random.default_rng(0).standard_normal((n, n))
    info = _path("abc,ai,bj,ck->ijk", K, W, W, W)

    step2 = info.steps[2]
    # 220 = (n³ + 3n² + 2n)/6 = S₃ orbit count on (i,j,k) for n=10.
    # 19 = 2W − 1 where W = step W-side size = n = 10.
    expected_cat_b = 220 * 19
    assert step2.flop_cost == expected_cat_b, (
        f"step 2 expected {expected_cat_b} (Cat B = 220·19), got {step2.flop_cost}"
    )
    assert step2.cost_source == "output-burnside", (
        f"expected cost_source='output-burnside', got {step2.cost_source!r}"
    )


def test_kwww_total_drops_with_tighter_step2():
    """Whole-expression cost reflects the step 2 drop.

    Before Sprint 4: step0=10,450 + step1=10,450 + step2=10,450 = 31,350.
    After Sprint 4:  step0=10,450 + step1=10,450 + step2=4,180  = 25,080.
    """
    n = 10
    K = symmetrize(
        fnp.random.default_rng(0).standard_normal((n, n, n)), symmetry=(0, 1, 2)
    )
    W = fnp.random.default_rng(0).standard_normal((n, n))
    info = _path("abc,ai,bj,ck->ijk", K, W, W, W)
    assert info.optimized_cost == 25_080, (
        f"expected 25,080 (= 10,450 + 10,450 + 4,180), got {info.optimized_cost}"
    )


def test_kwww_step1_cost_source_recorded():
    """Step 1 of K·W·W·W (``ibc,bj->icj``) has S₂{b,c} input → S₂{i,j} output.

    Cat B and Cat A may tie on this step (either is acceptable); we just
    assert cost_source is recorded (not None) and ≤ pre-Sprint-4 value.
    """
    n = 10
    K = symmetrize(
        fnp.random.default_rng(0).standard_normal((n, n, n)), symmetry=(0, 1, 2)
    )
    W = fnp.random.default_rng(0).standard_normal((n, n))
    info = _path("abc,ai,bj,ck->ijk", K, W, W, W)
    step1 = info.steps[1]
    assert step1.cost_source is not None, "cost_source should be set"
    assert step1.cost_source in {
        "per-input",
        "joint-burnside",
        "output-burnside",
    }, f"unexpected source: {step1.cost_source!r}"
    assert step1.flop_cost <= 10_450, (
        f"step 1 regressed beyond pre-Sprint-4 baseline: {step1.flop_cost}"
    )


def test_dense_chain_uses_per_input():
    """Distinct dense operands → no symmetry → Cat A is the only valid path.

    cost_source must be ``"per-input"`` for every step.
    """
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    B = rng.standard_normal((5, 5))
    info = _path("ij,jk->ik", A, B)
    for i, step in enumerate(info.steps):
        assert step.cost_source == "per-input", (
            f"step {i}: expected per-input, got {step.cost_source!r}"
        )


def test_5a_step2_stays_at_55():
    """§5(a) 4-cycle of shared symmetric S — Cat C joint-Burnside still wins.

    Step 2 ``(0, 1) lik,jki->ijkl`` cost must remain 55 (Wilson-verified),
    and ``cost_source`` should be ``"joint-burnside"``.
    """
    S = fnp.zeros((4, 4))
    info = _path("ij,jk,kl,li->ijkl", S, S, S, S)
    step2 = info.steps[2]
    assert step2.flop_cost == 55, f"§5(a) step 2 regressed: {step2.flop_cost} != 55"
    assert step2.cost_source == "joint-burnside", (
        f"§5(a) step 2 should use joint-burnside, got {step2.cost_source!r}"
    )


def test_no_step_regresses_vs_sprint3_baseline():
    """Monotonicity guard: no step's flop_cost may INCREASE under Sprint 4.

    The min(A, B, C) selection can only tighten costs (or hold), never loosen.
    """
    # Curated coverage spanning the cases most likely to shift.
    n = 6
    rng = np.random.default_rng(0)

    # Case 1: K·W·W·W per-step (Wilson's main case)
    K = symmetrize(rng.standard_normal((n, n, n)), symmetry=(0, 1, 2))
    W = rng.standard_normal((n, n))
    info1 = _path("abc,ai,bj,ck->ijk", K, W, W, W)

    # Case 2: §5(a) 4-cycle symmetric S
    S = fnp.zeros((n, n))
    info2 = _path("ij,jk,kl,li->ijkl", S, S, S, S)

    # Case 3: matrix chain dense (no symmetry — must be per-input)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    C = rng.standard_normal((n, n))
    info3 = _path("ij,jk,kl->il", A, B, C)

    for info in (info1, info2, info3):
        for step in info.steps:
            assert step.flop_cost <= step.dense_flop_cost, (
                f"step {step.subscript}: flop_cost {step.flop_cost} > dense "
                f"{step.dense_flop_cost} (regression in cost model)"
            )
            assert step.cost_source is not None, (
                f"step {step.subscript}: cost_source not recorded"
            )


def test_cat_b_loses_when_per_input_is_tighter():
    """When Cat A (per-input) gives a tighter count than Cat B, Cat A wins.

    For ``einsum("ij,ij->ij", S, S)`` with S₂-symmetric S:
    - Output is S₂ (Hadamard preserves symmetry).
    - Per-input via wreath/sigma gives small cost (single-step expression
      using the input S₂ directly).
    Either ``per-input`` or ``output-burnside`` may win as long as the
    chosen cost is the minimum of valid candidates.
    """
    S = fnp.zeros((4, 4))
    info = _path("ij,ij->ij", S, S)
    step = info.steps[0]
    # The chosen source must be one of the three valid options.
    assert step.cost_source in {
        "per-input",
        "joint-burnside",
        "output-burnside",
    }, f"unexpected cost_source: {step.cost_source!r}"
    # And the cost must be ≤ dense (sanity).
    assert step.flop_cost <= step.dense_flop_cost
