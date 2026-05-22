# Changelog

## Unreleased

### Changed (BREAKING)

- **Bilinear wrappers propagate symmetry and use joint-operand cost
  savings (closes #59).** `matmul`, `dot`, `outer`, `inner`, `vdot`,
  and `tensordot` (for full-inner-contraction `axes=N` cases) now route
  through the same cost+symmetry-inference path as `fnp.einsum`.
  - `A @ A` with symmetric `A` now returns a `SymmetricTensor` and is
    charged the symmetry-aware FLOP cost — empirically ~52% of dense
    (the output-symmetry savings), matching `fnp.einsum('ij,jk->ik', A, A)`.
  - `outer(v, v)` returns a `SymmetricTensor` and is charged
    `n(n+1)/2` unique-product FLOPs instead of dense `n²`.
  - Analogous parity for `dot`, `inner`, `vdot`, and `tensordot(A, A, axes=N)`.
  - The detector keys on Python `id()`, so the documented limitation
    `A @ A.T` (with `A` non-symmetric) is NOT detected — `.T` returns
    a view with a different `id()`. Do not "fix" this with
    `np.shares_memory`; that would produce numerically wrong results
    for unrelated slice views sharing a buffer.
  - `kron` is intentionally left alone (not an einsum contraction).
  - The public `flopscope.accounting.einsum_cost` introspection helper
    now forwards `identity_pattern` so its numbers agree with the
    actual einsum cost for repeated-operand cases.

- **Vendored opt_einsum replaced with runtime dependency.** flopscope now
  depends on `opt_einsum>=3.3.0,<4.0.0` instead of vendoring its source.
  The remaining `flopscope._opt_einsum` is a slim ~830-line shim that
  adapts upstream's `PathInfo` to flopscope's expected shape and recomputes
  per-step FLOP costs using flopscope's FMA convention.
  - `flopscope._opt_einsum.contract_path` still returns a flopscope
    `PathInfo` with the same essential fields (`path`, `steps`,
    `optimized_cost`, etc.).
  - `StepInfo` no longer carries the dead symmetry-related fields
    (`input_groups`, `output_group`, `inner_group`, `inner_applied`,
    `dense_flop_cost`, `symmetry_savings`). It now has 4-5 fields:
    `subscript`, `flop_count`, `input_shapes`, `output_shape` (plus
    `merged_subset` if still used by display).

- **Einsum cost model rewritten** to mirror the JS Symmetry-Aware Einsum
  Contractions explorer's α/M direct-event model. The charged FLOP cost is
  now path-independent: `(k - 1) · ∏ M_a + ∏ α_a` summed across components.
  - `path_info.optimized_cost` returns the new whole-expression cost. For
    expressions with declared symmetry, this number differs from the old
    per-step `cost · unique/total` formula. See migration notes below.
  - Path optimization no longer uses symmetry; `opt_einsum.contract_path`
    behaves like upstream stock opt_einsum.
  - Per-step `path_info.steps[i].flop_count` reverts to dense (no symmetry
    adjustment per step).

- **Reduction cost model rewritten** to the orbit-mapping model. Tier 1
  (`np.ufunc.reduce` ops — sum, prod, max, min, all, any, bitwise_or/and/xor,
  logical_or/and):
  ```
  cost = op_factor × (α - num_output_orbits) + extra_ops
  ```
  where α is the per-output-orbit input-orbit count summed across output
  orbits. The `α - num_output_orbits` correction fixes #56's off-by-one
  (dense `sum(n)` charges `n - 1`, was `n`). For symmetric inputs the new
  model charges more than the legacy `unique_elements_for_shape` formula —
  see #56 for the architectural shift.
- **`np.median`, `np.percentile`, `np.quantile`** now use a Tier-2
  output-discounted formula:
  ```
  cost = num_output_orbits × dense_per_output_cost
  ```
  For median/percentile/quantile, `dense_per_output_cost = axis_dim` (one
  partition pass per output cell).
- **`np.mean`** charges `sum_cost + num_output_orbits` (one divide per
  output orbit; orbit-shared output values share the divisor).
- `flopscope._flops.analytical_reduction_cost` body replaced with a
  delegating call to `compute_reduction_accumulation_cost`. Signature
  unchanged; numbers change.
- `flopscope.accounting.reduction_cost` returns different numbers for both
  dense and symmetric inputs (via the body change above).

### Fixed

- `flopscope.numpy.einsum_path` cache now keys on `fma_cost()` in addition
  to `(subscripts, shapes, optimize)`. Previously, toggling
  `flopscope.configure(fma_cost=2)` after a path was cached would return a
  stale `PathInfo` whose per-step `flop_count` values were computed under
  the old FMA convention.

### Added

- `flopscope.fma_cost()` — top-level re-export of the FMA-convention reader
  (was `flopscope._cost_model.fma_cost`). Returns the current value of the
  `fma_cost` setting (1 or 2).
- `flopscope.einsum_clear_caches()` — clears the einsum path cache and the
  einsum accumulation-cost cache together. Useful for cold-call benchmarks.
  (The narrower `fnp.clear_einsum_cache()` continues to clear only the path
  cache.)
- `flopscope.einsum_cache_info()` — returns
  `{"path": CacheInfo, "accumulation": CacheInfo}` so callers can inspect
  both einsum caches in one call.
- `flopscope.reduction_clear_cache()` and `flopscope.reduction_cache_info()`
  — same pattern for the reduction accumulation-cost cache used by
  `fnp.sum` / `fnp.mean` / `fnp.median` / etc.
- `flopscope.clear_cache()` — convenience aggregate that clears both
  einsum and reduction caches in one call.
- `flopscope.tier2_reduction_cost(a, axis=None, *, dense_per_output_cost=None)`
  — public inspection function for selection-style reductions
  (`np.median` / `np.percentile` / `np.quantile`). Hides the
  `op_factor=0, extra_ops=…` invocation pattern; `dense_per_output_cost`
  defaults to the product of the reduced axes' lengths.
- `flopscope.einsum_accumulation_cost(subscripts, *operands, partition_budget=None)`
  — public inspection function returning the new `AccumulationCost` decomposition
  (path-independent, per-component breakdown, regime trace).
- `flopscope.AccumulationCost`, `flopscope.ComponentCost`, `flopscope.RegimeStep`
  — public dataclasses.
- New settings:
  - `partition_budget` (default 100 000): per-component typed-partition cap.
  - `dimino_budget` (default 500 000): whole-expression `G_pt` closure cap.
- `CostFallbackWarning` now also fires when a partition counter exceeds its
  budget; total falls back to `k · dense_baseline` (the no-symmetry direct-
  event count).
- New configurable setting `fma_cost` (default 1). Counts a fused
  multiply-add as 1 op (hardware convention). Set to 2 to get the
  textbook / opt_einsum convention.
- `flopscope._cost_model.fma_cost()` function replaces the
  `FMA_COST` constant. The constant is removed.
- `flopscope.reduction_accumulation_cost(a, axis=None, *, op_factor=1, extra_ops=0)`
  — public inspection function returning an `AccumulationCost` for a
  reduction. Parallel to `einsum_accumulation_cost`.
- Internal `_accumulation/_reduction.py`: `compute_reduction_accumulation_cost`
  orchestrator, `output_discounted_reduction_cost` (Tier 2), and
  `_normalize_axis` / `_num_output_orbits` helpers.
- `_accumulation/_cache.py`: `_reduction_cache` + `get_reduction_cost_cached`
  (LRU 4,096).
- `_accumulation/_cost.py:aggregate_reduction` body implemented (was a
  signature-locked `NotImplementedError`).

### Removed

- `flopscope._opt_einsum._paths` — now upstream
- `flopscope._opt_einsum._path_random` — now upstream
- `flopscope._opt_einsum._parser` — now upstream (re-exported via shim)
- `flopscope._opt_einsum._blas` — was unused dead code
- `flopscope._opt_einsum._testing` — was unused dead code
- `flopscope._opt_einsum._typing` — now upstream
- `flopscope._cost_model.FMA_COST` constant — replaced by `fma_cost()`
- `flopscope._opt_einsum._subgraph_symmetry` — internal module deleted.
- `flopscope._opt_einsum._symmetry` — internal module deleted (was mostly
  `symmetric_flop_count`, `unique_elements`, `SubsetSymmetry` — all only used
  by the deleted oracle).
- `use_inner_symmetry` setting — was a knob on the deleted oracle.

---

### BREAKING

- `BudgetContext.untracked_time` and `summary_dict()["untracked_time_s"]` now
  subtract the new `flopscope_overhead_time_s` bucket. Consumers that previously
  read `untracked_time` as a proxy for "all unattributed time" should switch to
  `flopscope_overhead_time + untracked_time` for the old value, or stay on
  `untracked_time` for the new (more meaningful) "genuinely neither" residual.

- `BudgetContext.tracked_time` semantics tightened: now reflects ONLY the wall
  time of the inner numpy call, not the entire `with budget.deduct(...):`
  block. Time spent in flopscope's pre/post-numpy work inside the block (view-
  casts, copyto fallbacks, dispatch logic) is now attributed to
  `flopscope_overhead_time`.

- `BudgetContext.wall_time_s` now spans `__init__` start through the end of
  `__exit__` (was `__enter__` start through start of `__exit__` body). For an
  empty `with BudgetContext(): pass` the value grows by ~3 µs (was ~0.4 µs,
  now ~2.7 µs); real workloads see noise-level changes. The pre-`__enter__`
  slice (`__init__` body + banner print) and the post-`__exit__` body slice
  (accumulator-record, active-budget restore) are now attributed to
  `flopscope_overhead_time` rather than falling out of any bucket. Issue #82.

### Added

- `BudgetContext.flopscope_overhead_time` property — measured wall-clock time
  spent inside flopscope's own dispatch code.
- `summary_dict()["flopscope_overhead_time_s"]` and the same key per namespace
  bucket in `summary_dict(by_namespace=True)`.
- `OpRecord.flopscope_overhead` — per-op overhead in seconds.

### Changed

- `flopscope.configure(check_nan_inf=True)` — opt-in scan time is now attributed
  to `flopscope_overhead_time` instead of `untracked_time`.

- `_NamespaceScope.__enter__/__exit__` brackets now use `try/finally` so that
  push/pop time is billed to `flopscope_overhead_time` even when the underlying
  `_push_namespace`/`_pop_namespace` raises. Issue #82.
