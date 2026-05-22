# Migration: Symmetry-Aware Einsum Cost (JS-mirrored)

This is a behavior change in the FLOP cost charged for einsum operations
involving SymmetricTensor inputs.

## What changed

The einsum cost model was rewritten to match the canonical specification
described in the [Symmetry Detection Deep Dive](/docs/understanding/symmetry-detection/).
The new model:

- Computes a path-independent direct-event count: `(k-1)·∏ M_a + ∏ α_a`
  per independent component.
- Uses a 5-regime classification ladder (`trivial`, `functionalProjection`,
  `singleton`, `young`, `partitionCount`) plus an explicit `unavailable` state.
- Charges `k · ∏ n_ℓ` (dense) when the typed-partition budget is exceeded —
  conservative and gaming-resistant.

The previous model charged a per-step `cost · unique / total` ratio per pairwise
contraction, which depended on the contraction path opt_einsum picked.

## How to migrate

If your code reads `path_info.optimized_cost` or relies on `BudgetContext.spent`
matching specific FLOP integers, expect numbers to shift for any expression with
declared symmetry. Trivial-symmetry expressions are unchanged.

To inspect the new cost decomposition:

```python
import flopscope as fps
import numpy as np

A = fps.as_symmetric(np.zeros((4, 4, 4)), symmetry=(0, 1, 2))  # S_3
cost = fps.einsum_accumulation_cost('ijk,abc->ic', A, A)

print(f'total = {cost.total}')
print(f'mu = (k-1) * prod(M) = {cost.mu}')
print(f'alpha = prod(alpha_a) = {cost.alpha}')
for component in cost.per_component:
    print(f'  {component.labels}: M={component.m}, '
          f'alpha={component.alpha}, regime={component.regime_id}')
```

## What didn't change

- Path optimization picks the same paths it did with the dense (no-symmetry)
  cost model. Execution is unchanged.
- `SymmetricTensor` class, `as_symmetric`, declared symmetry validation — all
  unchanged.
- Non-einsum operations (sum, mean, etc.) still use `_symmetry_adjusted_cost`
  with the existing `unique / dense` ratio.

## Future work

The reduction-cost API hooks (`aggregate_reduction`) are committed as stubs
raising `NotImplementedError`. A follow-up sprint implements ufunc.reduce-aware
cost calculation reusing the same per-component machinery.

---

## Reduction cost rewrite

PR #91 also rewrites the cost surface for `np.ufunc.reduce`-shaped operations (`sum`, `prod`, `max`, `min`, `all`, `any`, `mean`) and for the partition-style reductions (`median`, `percentile`, `quantile`). They now use the same orbit-aware model as einsum.

**What this means concretely (closes [#56](https://github.com/AIcrowd/flopscope/issues/56)):**

```python
import numpy as np
import flopscope as flops

# Before PR #91:  sum on (10,) charged n flops = 10
# After PR #91:   sum on (10,) charges (n - 1) = 9
print(flops.reduction_accumulation_cost(np.zeros(10)).total)  # → 9

# mean on (10,) adds one divide per output orbit
print(flops.reduction_accumulation_cost(np.zeros(10), extra_ops=1).total)  # → 10
```

For inputs with declared symmetry, both terms drop — see the new [Symmetry-aware FLOP counting](/docs/understanding/symmetry-detection/) page for worked examples.

`fnp.median`, `fnp.percentile`, `fnp.quantile` use a separate Tier-2 selection-style cost; inspect with `flops.tier2_reduction_cost(...)`.

---

## FMA convention flip (2026-05-20)

Flopscope previously counted a fused multiply-add (FMA) as **1 operation** by
default and exposed a `fma_cost` setting toggling 1 or 2. As of this version:

- **The `fma_cost` setting and `FMA_COST` constant are removed.** Flopscope
  now uses the FMA=2 textbook convention (multiplies and adds counted
  separately) everywhere. There is no longer a knob.
- **Seven ops doubled their analytical FLOP count:** `hamming`, `hanning`,
  `polyval`, `linalg.multi_dot`, `linalg.norm`, `linalg.vector_norm`,
  `linalg.matrix_norm`.
- **For `hamming` and `hanning`, the empirical weight halved (16 → 8)**
  so runtime predictions are invariant. For the other five, weights stay
  at 1.0 and runtime predictions shift ~2× higher.
- **The α/M formula was already FMA=2-textbook by construction.** Any
  einsum cost (via `einsum_accumulation_cost`, `fnp.einsum`, etc.) is
  numerically unchanged.
- **`info.steps[i].dense_flop_cost` semantics changed.** Previously the
  upstream opt_einsum FMA-fused count; now the α/M-no-symmetry count.
  For unsymmetric matmul, `dense_flop_cost == flop_cost` now.

If you depended on `flops.fma_cost()` or `flops.configure(fma_cost=...)`,
remove the call. There is no replacement — flopscope is FMA=2-only.

---

## New public inspection and cache API

PR #91 promotes several utilities to the top-level `flopscope.*` surface so participant code doesn't need private imports.

| Function | What it does |
|---|---|
| `flops.einsum_accumulation_cost(subs, *operands)` | Returns an `AccumulationCost` for an einsum expression. Path-independent. |
| `flops.reduction_accumulation_cost(a, axis=None, ...)` | Returns an `AccumulationCost` for an additive reduction. |
| `flops.tier2_reduction_cost(a, axis=None, *, dense_per_output_cost=None)` | Returns the FLOP total for a selection-style reduction (`median` / `percentile` / `quantile`). |
| `flops.einsum_clear_caches()` | Clears the einsum path and accumulation-cost caches. |
| `flops.einsum_cache_info()` | Returns `{"path": CacheInfo, "accumulation": CacheInfo}`. |
| `flops.reduction_clear_cache()` | Clears the reduction accumulation-cost cache. |
| `flops.reduction_cache_info()` | Returns the CacheInfo for the reduction cache. |
| `flops.clear_cache()` | Clears all flopscope caches (einsum + reduction) in one call. |
| `flops.fma_cost()` | Returns the current FMA-convention setting (1 or 2). |

See the new [Symmetry-aware FLOP counting](/docs/understanding/symmetry-detection/) page for usage examples.
