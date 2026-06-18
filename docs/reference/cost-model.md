# Cost model reference

> **Start here.** This is the cost model's conceptual and audit reference. Read it
> to understand *how* billing works and to satisfy yourself that it is correct and
> non-gameable — you do **not** need to read every operation. The exhaustive,
> generated per-op list (every op with its `cost_formula` and `weight`) lives in
> [`ops.json`](#exhaustive-per-op-reference) and the website API pages; this doc
> explains the model by **family rule** so you can reason about a whole class at once.

flopscope bills compute as:

```
charged = int(flop_cost × weight)
```

## How to read this

1. **[Billing model & design principles](#billing-model--design-principles)** — the one equation and *why* it is split into `flop_cost` and `weight`.
2. **[Non-exploitability](#non-exploitability)** — the invariants that keep billing sound, and the test that enforces each.
3. **[Cost by family](#cost-by-family)** — the rule + evidence + representative ops for the family you care about.
4. **[Exhaustive per-op reference](#exhaustive-per-op-reference)** — drill into `ops.json` for one op's exact formula.

**Completeness guarantee:** every billed operation is classified in the registry and
appears in `ops.json` with a `cost_formula`; `tests/test_cost_model_coverage.py`
enforces both that, and that every op-class (`ops.json` *area*) is covered by a family
rule below. So nothing billed is undocumented, even where this doc summarizes by rule.

---

## Billing model & design principles

Every operation is charged `charged = int(flop_cost × weight)`.

**Two layers, on purpose.** `flop_cost` is the operation count: every shape- and
algorithm-dependent term lives here, so anyone — us or a participant — can read it off
the formula and audit it function by function. `weight` is a separate per-element factor
that captures how much more one element of an operation costs on real hardware: a plain
add is `1`, while a transcendental element (`sin`, `exp`, …) does many times more
floating-point work. Rather than bill each op its exact measured ratio — noisy,
machine-specific, and hard to audit — we group operations into a small fixed ladder of
tiers (`{0, 1, 8, 16}`); that grouping is a deliberate competition-design choice. The
rule that keeps the split honest — and the model non-gameable — is that **a shape or
algorithm constant never lives in a weight**: anything depending on a matrix dimension
or loop length belongs in `flop_cost`. (Enforced by `tests/test_weight_tier_policy.py`.)

**We bill the textbook standard-algorithm cost, not literal BLAS/LAPACK.**
`linalg.inv` is billed `2n³` (the standard LU-based `dgetrf`+`dgetri` operation
count) regardless of what the underlying library does; top-k SVD is billed as the
standard truncated-algorithm cost. This keeps billing deterministic,
hardware-independent, and composable. 

The rest of this section defines the conventions these principles rest on.

### FMA=2

Each floating-point multiply, add, subtract, divide, or square root counts
as 1 FLOP.  A fused multiply-add (FMA) therefore counts as 2.  This matches
the standard textbook convention.  All formulas in this document are stated
in FMA=2 units unless noted.

### Comparison and select

A single comparison (`>`, `==`, `!=`, …) or conditional-select (`where`,
`choose`) counts as 1 FLOP.  Sorting, partition, and percentile operations
use this convention when counting per-element work.

### Transcendental tier (weight 16.0)

Operations whose per-element cost is dominated by a libm minimax polynomial
evaluation (sin, cos, tan, exp, log, arcsin, arccos, arctan, arcsinh,
arccosh, arctanh, power, and their NumPy 2.x aliases) are billed at weight
16.0.  The `flop_cost` formula is `numel(output)` (1 per element); the 16×
factor is supplied entirely by the weight.

A subset of moderate-cost binary ops (floor_divide, mod/remainder, fmod,
arctan2, hypot, logaddexp, logaddexp2) is grouped into the same tier
(weight 16.0).

### Half-tier transcendentals (weight 8.0)

Ops whose per-element work is a single cosine evaluation amortized over a
cheap window formula (`hamming`, `hanning`) are billed at weight 8.0.

### The unifying philosophy — compute, not logistics

> **flopscope meters _computation on values_, not _data logistics_.**
> An operation is **charged** for the floating-point arithmetic and value-comparisons
> it performs to produce its output.  It is **free** (weight 0) if it only relocates,
> replicates, selects-by-a-given-selector, or constant-fills values that already exist.

**The decision procedure** — apply these three steps in order to any op:

1. **View / metadata only** (returns a view, inspects shape/dtype, no new buffer)?
   → **Free (0).**
2. **Does it produce output values by doing floating-point arithmetic, *or* by
   comparing element values?** → **Charged.** `flop_cost` = standard-algorithm op
   count; `weight` = hardware tier.  This includes elementwise math, transcendentals,
   reductions, contraction (matmul/einsum), FFT, polynomial, random generation, and
   ops that *derive* a result by *testing values*: `sort`/`argsort`/`partition`/
   `searchsorted`/`unique*`, `nonzero`/`argwhere`/`flatnonzero`/`count_nonzero`/
   `where(1-arg)`, `clip`/`minimum`/`maximum`, set-ops, and *computed creators*
   (`arange`/`linspace`/`geomspace`/`logspace`/`vander`).
3. **Otherwise** it only relocates / replicates / selects-by-a-given-selector /
   constant-fills existing values → **Free (0).**  This covers copy/concat/roll/
   repeat/tile, gather/scatter & mask-select with a *given* selector, and constant init.

**Key invariant:** any predicate or index feeding a step-3 op was itself produced by a
step-2 op and charged there.  A free-tier op may **never bundle** value-arithmetic or
value-comparison into its own cost.

After removing the gather tier, the only active weights are `{0, 1, 8, 16}`.
Data-movement, selection-by-given-selector, and constant-init all carry weight 0.
The only residual `4.0` entries are the submission-blocked callback ops
(`piecewise`/`apply_along_axis`/`apply_over_axes`); they raise `RemoteCallbackError`
on the grading backend and are left untouched.

### Views and metadata (weight 0.0)

Weight 0 now covers four categories:

1. **Views / metadata** — operations that return a view of existing memory or inspect
   metadata without touching element values: `reshape`, `ravel`, `flatten`,
   `transpose`, `diagonal` (as a view), `squeeze`, `broadcast_to`, `astype` (no copy),
   `fftshift`/`ifftshift`, `linalg.diagonal`, `linalg.matrix_transpose`, and all
   other shape/stride/dtype introspection ops.
2. **Copy / materialize** — data-movement ops that copy or rearrange existing values
   into a new buffer: `concatenate`, `stack`, `hstack`, `vstack`, `column_stack`,
   `dstack`, `tile`, `repeat`, `roll`, `tril`, `triu`, `copy`, and kin.
3. **Gather / scatter & mask-select with a given selector** — ops whose mask or
   index is an *input*: `take`, `take_along_axis`, `put`, `put_along_axis`, `choose`,
   `where(cond, x, y)` (3-arg), `select`, `compress(mask, a)`, `extract(mask, a)`,
   `place`, `putmask`.
4. **Constant init** — ops that fill a new array with a fixed value (no per-element
   arithmetic): `zeros`, `ones`, `empty`, `full`, `eye`, `identity`, `tri`,
   `zeros_like`, `ones_like`, `empty_like`, `full_like`, `meshgrid`.

**Refinement A — selection** (resolves `where`/`compress`/`extract`/`choose`/`select`):

> **Selector given ⇒ free; selector derived by testing values ⇒ charged.**

Free: `where(cond, x, y)`, `choose`, `select`, `compress(mask, a)`, `extract(mask, a)`,
`take`, `take_along_axis`.  The mask/index is an **input**; any predicate that built it
(e.g. `a > 0.5` → `greater`) is a separate, separately-charged op.

Charged: `where(cond)` (1-arg, ≡ `nonzero`), `nonzero`, `argwhere`, `flatnonzero`,
`count_nonzero`.  These **derive** the selector by testing values (`!= 0`), so the test
is their compute — they are charged `numel(input)` at weight 1.0.

Value-changing `astype` (to-bool `!=0`, float→int truncation, float-narrowing rounding)
is also charged `numel` (weight 1.0) — a per-element value test.  Lossless width casts
(e.g. `float32→float64`) stay free.  The method `a.nonzero()` is charged identically to
`fnp.nonzero(a)`.

**Refinement B — creation** (resolves init vs computed generators):

> **Constant-fill / replicate ⇒ free; compute-a-value-per-element ⇒ charged.**

Free: `zeros`, `ones`, `empty`, `full`, `eye`, `identity`, `tri`, `*_like`, and
`meshgrid` (pure replication of coordinate vectors — no per-element arithmetic).

Charged: `arange`, `linspace` (`2×numel`), `geomspace`, `logspace` (`16×numel`),
`vander` (`N(N-2)`).  If these were free a participant could synthesize an affine/
log-spaced ramp for free while the equivalent explicit `x*step+start` is charged — the
substitution arbitrage the non-exploitability section forbids.  Constant-fill has no
such arithmetic equivalent, so it is free.

### Composite ops (weight 1.0 with heterogeneous flop_cost)

When an operation mixes sub-tiers internally (e.g. random samplers, stats
kernels, norms with SVD), all per-element factors are folded into `flop_cost`
and the active weight is set to 1.0.  This avoids double-counting with the
tier factor.

### NumPy 2.x ufunc aliases

NumPy 2.x introduced `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`,
`atan2`, `pow`, and `divmod` as canonical aliases for their `arc*` /
`power` / `floor_divide` counterparts (identical ufunc objects).  flopscope
resolves these via `_UFUNC_ALIAS_RENAMES` in `_weights.py` so each alias
charges the same weight as its canonical twin.  

---

## Non-exploitability

The cost model meters compute so a participant cannot do expensive real work while
being billed cheaply. The two threats are **under-count** (an op billed below its
true cost) and **substitution arbitrage** (routing the same work through a
cheaper-billed but equivalent op). The model defends against both with invariants,
each backed by a CI-enforced test you can open and read:

| Invariant | What it guarantees | Enforced by |
|---|---|---|
| **Faithful cost** | each `flop_cost` is the real standard-algorithm op count, with every shape/algorithm constant inside `flop_cost` | per-op evidence in [§Cost by family](#cost-by-family); `test_cost_constant_unification.py`, `test_cost_formula_vs_code.py` |
| **Weight-tier policy** | every active weight ∈ `{0, 1, 8, 16}`; arithmetic ops are 0 or 1; **no algorithm constant in a weight** | `test_weight_tier_policy.py` |
| **No substitution arbitrage** | a bit-identical alias cannot bill cheaper than its canonical (e.g. `acos` *is* `arccos` — the 16× ufunc-alias fix); equivalent contractions (`dot`/`inner`/`matmul`/`einsum`) share one cost engine | `test_ufunc_alias_parity.py`, `test_random_weight_aliasing.py`; the shared einsum engine ([§Contraction](#contraction-einsum-family)) |
| **No cheap in-op path** | top-k `svd(k=)` cannot yield a *full* decomposition below full price (the `min(4mnk, economy)` cap + `k ≥ min → full` guard); invalid `k` (`< 1` or `> min(m, n)`) is rejected before any billing | `test_svd_topk_cost.py` (cap / guard / monotonicity); `test_linalg.py` (invalid-`k` `ValueError`) |
| **Free-tier discipline** | only ops that perform no value arithmetic/comparison carry weight 0; a value-test is charged wherever it hides — including `a.nonzero()` (method), value-changing `astype`, `where(1-arg)`, `argwhere`, `flatnonzero`, and `count_nonzero` | `test_weight_tier_policy.py`; `test_data_movement_free_tier.py` (free-labels consistency guard) |
| **Memoization accepted** | free gather makes look-up-table reuse (precompute once with a charged op, then `take` for free) cheaper — this is deliberate: memoization is a legitimate optimization under a pure-compute metric | documented here; `test_data_movement_free_tier.py` |
| **End-to-end billing** | production `flop_cost × weight` is pinned per tier `{0,1,8,16}` (catches a silent weight regression) | `test_production_weight_billing.py` |

An auditor can read this table top-to-bottom and, for each claim, open the named test
to see exactly what guarantees it. The first two rows are the load-bearing ones: an exact
`flop_cost` defeats under-count, and the weight-tier policy (no constant in a weight)
defeats the family of arbitrage exploits where a high-constant op is re-tiered cheaply.

---

## Cost by family

Each family below is one **rule** + its **evidence/citation** + **representative ops**.
The rule is the part to audit; the per-op tables are kept where each op carries a
*distinct* cited constant (linalg, FFT, polynomial, stats, window, random) because
those constants are the evidence — and because `ops.json`'s generated `cost_formula`
is coarse for many composite ops (it records `per-operation` where the real formula is
shape-dependent). For families whose members all share one rule (copy/gather, views),
only representatives are listed and the full set is a filter in
[`ops.json`](#exhaustive-per-op-reference).

---

### Elementwise (pointwise unary and binary)

**Family rule**: `flop_cost = numel(output)`.

**Baseline tier (weight 1.0)**: arithmetic (+, −, ×, ÷, √), rounding
(ceil, floor, trunc, rint, around/round), sign/abs, logical (not, and, or,
xor), bitwise (and, or, xor, invert, left_shift, right_shift), comparisons
(equal, not_equal, greater, less, greater_equal, less_equal), copies
(positive, negative, real, imag, conj/conjugate, fabs, modf, frexp, spacing,
nan_to_num, isclose, isneginf, isposinf, deg2rad/degrees, rad2deg/radians,
ldexp, nextafter, copysign, heaviside, signbit), and their NumPy aliases.

**Transcendental tier (weight 16.0)**: exp, exp2, expm1, log, log2, log10,
log1p, cbrt, sin, cos, tan, sinh, cosh, tanh, arcsin, arccos, arctan,
arcsinh, arccosh, arctanh, sinc, i0, power, angle, and their NumPy 2.x
aliases (asin, acos, atan, asinh, acosh, atanh, atan2, pow).

**Moderate binary tier (weight 16.0)**: arctan2/atan2, hypot, logaddexp,
logaddexp2, floor_divide, mod/remainder, fmod, float_power.

**Basis**: DECLARED per-element FMA=2 convention and empirical calibration.
Source: `src/flopscope/_pointwise.py`.

---

### Reduction

**Family rule**: `flop_cost = numel(input) − numel(output)` (orbit-mapping
model; one add or compare per element consumed by the reduction).

Ops that do more than one accumulation pass carry the extra passes in
`flop_cost` (never in the weight column): the variance family makes four
passes (mean-sum, centre, square, variance-sum), `ptp` makes two (max + min)
plus the per-output subtract, and `mean`/`average` add the per-output divide.

| Op | flop_cost | weight | basis |
|---|---|---|---|
| `sum`, `prod`, `max`, `min`, `any`, `all`, `nansum`, `nanmax`, `nanmin`, `nanprod` | numel(input) − numel(output) | 1.0 | DECLARED reduction skeleton (one add or compare per consumed element) |
| `cumsum`, `cumprod`, `nancumsum`, `nancumprod`, `cumulative_sum`, `cumulative_prod` | numel(input) − num_output_slices (= n−1 for a full 1-D scan; product of non-reduced dims otherwise) | 1.0 | DECLARED: scan accumulation; output shape = input shape so the generic `numel(in)−numel(out)` formula evaluates to 0 — these use the correct per-slice count instead |
| `mean`, `average` (unweighted) | numel(input) | 1.0 | DERIVED: reduction (numel−M) + M divides |
| `average(weights=)` | `3·numel − M`, M = num output slices (1 for full reduction) | 1.0 | DERIVED: a·w multiply pass (numel) + a·w sum (numel−M) + weight sum (numel−M) + M divides |
| `std`, `var`, `nanstd`, `nanvar` | ≈ 4 × numel(input) (std: + M sqrt) | 1.0 | DERIVED four-pass: mean-sum, centre, square, var-sum (exact: 2·numel + 2·(numel−M) + 2M) |
| `argmax`, `argmin` | numel(input) − num_output_slices (= n−1 for full 1-D; reduction_cost model) | 1.0 | DECLARED scan: same orbit model as reduction family |
| `median`, `nanmedian` | axis length per output slice | 1.0 | DECLARED; partition (introselect) per output |
| `percentile`, `nanpercentile`, `quantile`, `nanquantile` | axis length per output slice | 1.0 | DECLARED; partition (introselect) per output |
| `ptp` | 2 × numel(input) − numel(output) | 1.0 | DERIVED: max pass + min pass + M subtracts (2·(numel−M)+M) |
| `count_nonzero` | numel(input) | 1.0 | DECLARED comparison scan (every element tested regardless of axis) |
| `nanmean` | numel(input) | 1.0 | DERIVED: reduction (numel−M) + M divides; billed identically to mean |

Source: `src/flopscope/_pointwise.py`; reduction accumulation model in
`src/flopscope/_accumulation/`.

---

### Contraction (einsum family)

Every op in this family is billed by **one shared, symmetry-aware engine**
(`_resolve_cost_and_output_symmetry` → `einsum_cost`); the closed forms below are
that engine's output specialised to each op's shapes, not separately maintained
constants.

**Family rule:**

```
flop_cost = (2K − 1) × M
```

- `(2K − 1)` is one length-`K` dot product: `K` multiplies + `K − 1` adds (FMA=2).
- `K` = product of the **contracted** (summed) axis dimensions.
- `M` = number of output cells the engine computes. This is `prod(output dims)`
  for a generic contraction, but the engine **reduces it to the unique-orbit
  count when it can prove the output is symmetric** — when operands alias the
  same array (`outer(v, v)`, `inner(A, A)`) or carry an `as_symmetric` tag. It
  never invents savings: `A @ A` for a general `A` still costs the full
  `2n³ − n²`, because `A @ A` is not symmetric.

| Op | Contraction (`k` = contracted dim) | flop_cost `= (2K − 1) × M` |
|---|---|---|
| `matmul`, `linalg.matmul` | `(m,k) · (k,n) → (m,n)` | `2mkn − mn` |
| `dot` | matrix `(m,k)·(k,n) → (m,n)`; matrix–vector `(m,k)·(k,) → (m,)` | `2mkn − mn`; `m(2k − 1)` |
| `inner` | `(m,k) · (n,k) → (m,n)` — contracts the **last** axes | `2mkn − mn` |
| `tensordot`, `linalg.tensordot` | contracts the chosen axes | `(2K − 1) × M` |
| `outer`, `linalg.outer` | `(m,) · (n,) → (m,n)` — nothing summed, `K = 1` | `mn` |
| `vdot`, `vecdot`, `linalg.vecdot` | `(N,) · (N,) → scalar` — `M = 1` | `2N − 1` |
| `matvec`, `vecmat` | matrix·vector / vector·matrix, contracting `k` → length-`m` | `m(2k − 1)` |
| `kron` | `(a,) ⊗ (b,)` of flattened operands — nothing summed, `K = 1` | `a.size × b.size` |
| `einsum` | any subscripts | whole-expression accumulation (below) |

**Symmetry savings** make `M` drop below `prod(output)` (here `v` is length `n`,
`A` is `n × n`):

| Expression | generic `M` | symmetric `M` | flop_cost |
|---|---|---|---|
| `outer(v, v)` | `n²` | `n(n+1)/2` | `n(n+1)/2` |
| `inner(A, A)` | `n²` | `n(n+1)/2` | `(2n − 1) · n(n+1)/2` |

`einsum` runs the accumulation directly as `(K − 1)·M + α`, where `α` is the
number of unique (output + contracted) index combinations — equal to `K·M` for a
single clean contraction, but more general for multi-index or broadcast
subscripts. A multi-operand einsum (`≥ 3` operands) walks the `opt_einsum`
optimal binary path and sums per-step costs. Batched/stacked variants of any row
above multiply the closed form by the batch size.

**Compound linalg** ops are *chains* of matmuls, billed as the sum of their steps
through the `matmul_cost(m, k, n)` helper — which itself delegates to
`einsum_cost('ij,jk->ik', …)`, so each step equals a 2-D matmul by construction
(no duplicated `2mkn − mn` constant to drift). `linalg.pinv` and `linalg.lstsq`
build on the same helper.

| Op | flop_cost | basis |
|---|---|---|
| `linalg.matrix_power` | `(⌊log₂ k⌋ + popcount(k) − 1) × matmul_cost(n, n, n)` | repeated squaring |
| `linalg.multi_dot` | sum of optimal-chain matmul costs; each step `2mkn − mn` | optimal chain order |

All contraction ops use **weight 1.0** — the shape formulas already carry the
full FMA=2 cost. Source: `_pointwise.py` (op wrappers), `_einsum.py`
(`_resolve_cost_and_output_symmetry`), `_flops.py` (`einsum_cost`,
`matmul_cost`), `_accumulation/` (accumulation model).

---

### Generator (linspace, arange, and kin)

| Op | flop_cost | basis | source |
|---|---|---|---|
| `arange` | `2 × numel(output)` | DERIVED: `start + i×step` per element = 1 mul + 1 add (FMA=2) | `_array_ops.py`; numpy arraytypes.c.src |
| `linspace` | `2 × numel(output)` (handles broadcast start/stop and `retstep=True`) | DERIVED: same affine model as arange | `_array_ops.py`; commit 790d19af + retstep fix |
| `geomspace` | `numel(output)` (weight **16.0**) → billed `16 × numel(output)` | DERIVED: flop_cost = numel(output); transcendental weight 16.0 (log + exp path) | `_array_ops.py` |
| `logspace` | `numel(output)` (weight **16.0**) → billed `16 × numel(output)` | DERIVED: same transcendental path as geomspace | `_array_ops.py` |
| `zeros`, `ones`, `full`, `zeros_like`, `ones_like`, `full_like`, `eye`, `identity`, `empty`, `empty_like`, `tri` | 0 (allocation, no arithmetic) | DECLARED free: constant-fill / replicate (Refinement B) | `_array_ops.py` |
| `meshgrid` | 0 (free) | DECLARED free: pure replication of coordinate vectors; no per-element arithmetic (Refinement B) | `_array_ops.py` |

Weight: **1.0** for `arange` and `linspace`; **16.0** for `geomspace` and
`logspace` (transcendental path).  Source: `src/flopscope/_array_ops.py`.

---

### Sort and select

**Family rule** (DECLARED):

| Op | flop_cost | basis |
|---|---|---|
| `sort`, `argsort` | `num_slices × n × ⌈log₂ n⌉` | DECLARED comparison sort (n = axis length) |
| `unique`, `unique_counts`, `unique_inverse`, `unique_values`, `unique_all` | `n × ⌈log₂ n⌉` (axis=None); `num_slices × shape[axis] × ⌈log₂ shape[axis]⌉` (axis=k) | DECLARED sort-based; axis-aware per-slice |
| `lexsort` | `k × n × ⌈log₂ n⌉` (k = number of keys, n = sequence length) | DECLARED |
| `partition`, `argpartition` | `num_slices × n × len(kth)` | DECLARED quickselect O(n) expected |
| `searchsorted` | `m × ⌈log₂ n⌉` (m = queries, n = sorted size) | DECLARED binary search |
| `sort_complex` | `num_slices × n × ⌈log₂ n⌉`, `n = a.shape[-1]`, `num_slices = a.size // n` (sorts last axis; equals flat formula only for 1-D) | DECLARED |
| `in1d`, `isin` | `(n + m) × ⌈log₂(n + m)⌉` (sort path); `max(sort_cost(n+m), 2nm)` when numpy's masked-loop path triggers (small integer ar2) | DECLARED algo-aware |
| `intersect1d` | `sort_cost(n) + sort_cost(m) + sort_cost(n+m)` (default `assume_unique=False`); `sort_cost(n+m)` when `assume_unique=True` | DECLARED: numpy calls `unique()` on both inputs when `assume_unique` is falsy |
| `setdiff1d`, `setxor1d`, `union1d` | `(n + m) × ⌈log₂(n + m)⌉` | DECLARED |

All sort/select ops use **weight 1.0**; comparison = 1 FLOP convention.
Source: `src/flopscope/_sorting_ops.py`, `src/flopscope/_flops.py` (`sort_cost`, `search_cost`).

---

### Linalg direct (non-iterative)

All ops use **weight 1.0** with all shape constants in `flop_cost`.  Per-matrix
cost is multiplied by the batch dimension product for stacked inputs.  Zero-dim
matrices charge 0.

| Op | flop_cost (per matrix) | basis | source |
|---|---|---|---|
| `linalg.cholesky` | `n³/3` | DERIVED: Cholesky factorization (dpotrf) | `_decompositions.py:cholesky_cost` |
| `linalg.qr` (reduced/complete) | `2(2mnk − 2k³/3)`, `k = min(m,n)` | DERIVED: factorization (dgeqrf) + Q-formation (dorgqr) ≈ same count | `_decompositions.py:qr_cost` |
| `linalg.qr` (r/raw) | `2mnk − 2k³/3` | DERIVED: factorization only | `_decompositions.py:qr_cost` |
| `linalg.solve` | `2n³/3 + 2n²×nrhs` | DERIVED: LU solve (dgesv = dgetrf + dgetrs) | `_solvers.py:solve_cost` |
| `linalg.inv` | `2n³` | DERIVED: LU factorization + inversion (dgetrf + dgetri ≈ 2n³) | `_solvers.py:inv_cost` |
| `linalg.det` | `2n³/3 + n` | DERIVED: LU factorization (dgetrf) + diagonal product | `_properties.py:det_cost` |
| `linalg.slogdet` | `2n³/3 + 18n` | DERIVED: LU (dgetrf) + sum of log\|diag\| (abs + 16/elem log + reduce) | `_properties.py:slogdet_cost` |
| `linalg.norm` (fro/L1/Linf) | `2 × numel(effective_shape) × n_groups` | DERIVED: FMA=2 square+accumulate or abs+accumulate | `_properties.py:norm_cost` |
| `linalg.norm` (ord=2, nuc) | `(2ab² + 2b³) × n_groups`, `a=max(m,n)`, `b=min(m,n)` | DERIVED: values-only SVD cost per group | `_properties.py:norm_cost` |
| `linalg.vector_norm` | `2 × numel(effective_shape) × n_groups` (standard ord); `(18 × numel + 16) × n_groups` (general fractional p-norm: abs + pow per element) | DERIVED: FMA=2 | `_properties.py:vector_norm_cost` |
| `linalg.matrix_norm` | same as `linalg.norm` | DERIVED | `_properties.py` |
| `linalg.trace` | `min(m,n) × batch` | DERIVED: n−1 diagonal adds, batch-multiplied | `_properties.py:trace_cost` |
| `linalg.tensorinv` | `2n³`, `n = prod(shape[:ind])` | DERIVED: via inv | `_solvers.py:tensorinv_cost` |
| `linalg.tensorsolve` | `2n³/3 + 2n²`, `n = prod(shape[ind:])` | DERIVED: via solve | `_solvers.py:tensorsolve_cost` |
| `linalg.matrix_rank` | `2ab² + 2b³ + min(m,n)`, `a=max(m,n)`, `b=min(m,n)` | DERIVED: values-only SVD + `min(m,n)` threshold comparisons | `_properties.py:matrix_rank_cost` |
| `linalg.cond` | `2ab² + 2b³ + 1` for `ord∈{None,2,−2}` (values-only SVD + 1 divide); `2k³ + 4mn + 1`, `k=min(m,n)` for other ords (inv-based) | DERIVED | `_properties.py:cond_cost` |
| `linalg.pinv` | `6ab² + 20b³ + min(m,n) + n·min(m,n) + matmul\_cost(n, min(m,n), m)`, `a=max(m,n)`, `b=min(m,n)` | DERIVED: thin SVD (with vectors) + threshold + diagonal scale + reconstruction matmul | `_solvers.py:pinv_cost` |
| `linalg.lstsq` | `6ab² + 20b³ + matmul\_cost(k,m,c) + k·c + matmul\_cost(n,k,c)`, `k=min(m,n)`, `c=#rhs cols` | DERIVED: thin SVD (with vectors) + U^T b + divide by s + reconstruction | `_solvers.py:lstsq_cost` |
| `linalg.cross` | `3 × numel(output)` (delegates to `fnp.cross`) | DERIVED | `_aliases.py` |
| `linalg.multi_dot` | optimal chain matmul cost; each step uses `matmul_cost(m,k,n)` = `2mkn − mn` | DERIVED | `_compound.py:multi_dot_cost` |
| `linalg.outer`, `linalg.tensordot`, `linalg.vecdot`, `linalg.matmul`, `linalg.matrix_power` | delegates to `fnp.*` | DERIVED | `_compound.py`, `_aliases.py` |
| `linalg.diagonal`, `linalg.matrix_transpose` | 0 (view) | DECLARED free | `_aliases.py` |

---

### Linalg iterative (eigen / SVD)

These ops use LAPACK drivers that iterate until convergence; counts are
leading-order estimates of the standard operation count.  All use
**weight 1.0**.

| Op | flop_cost (per matrix) | basis | source |
|---|---|---|---|
| `linalg.eig` | `25n³` | DERIVED: dense eigendecomposition with eigenvectors — Hessenberg reduction + QR iteration + back-transform (dgeev) | `_decompositions.py:eig_cost` |
| `linalg.eigvals` | `10n³` | DERIVED: dense eigenvalues only, no vectors (dgeev) | `_decompositions.py:eigvals_cost` |
| `linalg.eigh` | `9n³` | DERIVED: symmetric tridiagonalization + divide-and-conquer with eigenvectors (dsyevd) | `_decompositions.py:eigh_cost` |
| `linalg.eigvalsh` | `4n³/3` | DERIVED: symmetric tridiagonalization only, no vectors (dsyevd) | `_decompositions.py:eigvalsh_cost` |
| `linalg.svd` (thin, full_matrices=False or square) | `6ab² + 20b³`, `a=max(m,n)`, `b=min(m,n)` | DERIVED: thin SVD — Σ + U₁ + V (dgesdd thin path) | `_svd.py:svd_cost` |
| `linalg.svd` (full, full_matrices=True and m≠n) | `4a²b + 22b³` | DERIVED: full SVD — forming the full m×m U dominates (dgesdd) | `_svd.py:svd_cost` |
| `linalg.svdvals` | `2ab² + 2b³` | DERIVED: SVD values only, no vectors (dgesdd) | `_decompositions.py:svdvals_cost` |
| `roots` | `10n³`, `n` = stripped companion dimension (leading and trailing zero coefficients removed before companion matrix is built) | DERIVED: companion-matrix eigvals (delegates to eigvals_cost on trimmed degree) | `_polynomial.py`; consistent with polynomial-table `roots` row |

#### Top-k (truncated) SVD

`linalg.svd(..., k=)` and `linalg.svdvals(..., k=)` accept a top-k parameter.
For `1 ≤ k < min(m, n)` the billed cost is

    min(4·m·n·k, economy)

where `economy` is the full thin/values-only cost above. `4·m·n·k` is the
leading-order cost (FMA=2, Θ(mnk)) of a rank-k truncated SVD (two
unavoidable passes over A). It is billed as the
**standard truncated-algorithm cost of the operation** — consistent with how
this model bills direct-linalg ops at their textbook standard-algorithm count
rather than literal BLAS/LAPACK work — even though the reference implementation
computes the full economy SVD
and slices (results stay exact). Unlike the full case, **values-only is not
leading-order cheaper** for top-k. `k = min(m, n)` (all components) bills the
full economy cost, and the `full_matrices` full-U premium applies only to the
full decomposition (`k is None`); so a complete decomposition can never be
obtained below full price. Invalid `k` (`< 1` or `> min(m, n)`) raises
`ValueError`.

**Accepted residual:** because `4mnk < 6ab²+20b³` for all `k ≤ min(m, n)`, the
truncated rate applies up to `k = min(m, n) − 1`, so a caller can obtain up to
`min(m, n) − 1` exact singular vectors at the truncated rate. The guard ensures
they can never obtain **all** `min(m, n)` components below full price.

Per-matrix cost is multiplied by the batch dimension product.  Constants
marked "provisional": iteration counts are input-dependent and the cubic
constant is the standard textbook estimate.

---

### FFT

**Family rule** (DERIVED, radix-2 FFT — 5 real ops per butterfly):

| Op | flop_cost | basis |
|---|---|---|
| `fft.fft`, `fft.ifft` | `5 × N × ⌈log₂ N⌉`, `N` = transform length | DERIVED: 5 real ops per butterfly |
| `fft.fft2`, `fft.ifft2`, `fft.fftn`, `fft.ifftn` | `5 × N × Σᵢ⌈log₂ dᵢ⌉`, `N = prod(transform dims)`, `dᵢ` = individual axis lengths | DERIVED: sum of per-axis log₂ terms (coincides with `5N⌈log₂N⌉` only when all axes are the same power of 2) |
| `fft.rfft`, `fft.irfft` | `5 × (N/2) × ⌈log₂ N⌉` | DERIVED: real-input / real-output half-spectrum |
| `fft.rfft2`, `fft.irfft2`, `fft.rfftn`, `fft.irfftn` | `5 × (N/2) × Σᵢ⌈log₂ dᵢ⌉` (real half-spectrum) | DERIVED: half-spectrum with per-axis log₂ sum |
| `fft.hfft` | `5 × (n_out/2) × ⌈log₂ n_out⌉` | DERIVED: hfft = irfft(conj(a)) — conjugate-symmetry halves the work |
| `fft.ihfft` | `5 × (n/2) × ⌈log₂ n⌉` | DERIVED: same `hfft_cost(n)` formula |
| `fft.fftfreq` | `n` (index grid scaled by `1/(n*d)` — one divide per output element) | DECLARED: `n` divides |
| `fft.rfftfreq` | `n//2 + 1` (real-spectrum grid has `n//2 + 1` elements) | DECLARED: `n//2 + 1` divides |
| `fft.fftshift`, `fft.ifftshift` | 0 | DECLARED free/metadata |

All counted FFT ops use **weight 1.0**.  Source: `src/flopscope/numpy/fft/_transforms.py`.

---

### Polynomial

| Op | flop_cost | basis | source |
|---|---|---|---|
| `polyval` | `2 × deg × points` (Horner: 1 mul + 1 add per coefficient per point, FMA=2) | DERIVED | `_polynomial.py` |
| `polyfit` | `2 × m × (deg+1)²` (Vandermonde least-squares estimate) | DERIVED: Vandermonde matrix construction + normal-equations cost; NOT an SVD path | `_polynomial.py` |
| `polyadd`, `polysub` | `max(len_a, len_b)` (= `max(n1, n2, 1)`) | DERIVED: output length equals the longer polynomial | `_polynomial.py` |
| `polymul` | `2nm − n − m` (direct conv, FMA=2) | DERIVED | `_polynomial.py` |
| `convolve` | `full`: `2nm − n − m`; `valid`: `(2·min−1)·(max−min+1)`; `same`: exact dot-length sum per numpy C layout | DERIVED per-mode | `_pointwise.py:convolve` |
| `poly` (1-D, build from roots) | `(3n² + n) // 2`, `n = len(roots)` (iterative convolution with length-2 kernel per root; FMA=2) | DERIVED | `_polynomial.py:poly_cost` |
| `polyder` | `t × n − t(t+1)/2`, `t = min(m, n−1)` (order-aware; one multiply per surviving coefficient per derivative step) | DERIVED | `_polynomial.py:polyder_cost` |
| `polyint` | `m × n + m(m−1)/2` (order-aware; m passes each dividing n+j coefficients) | DERIVED | `_polynomial.py:polyint_cost` |
| `roots` | `10n³`, `n = stripped companion dimension` (zero-leading/trailing coefficients stripped before companion matrix is built) | DERIVED: delegates to `eigvals_cost` on trimmed degree | `_polynomial.py:roots_cost` |

Source: `src/flopscope/_polynomial.py`.

---

### Random (module-level, Generator, RandomState)

Random ops are composite: the generation kernel cost and any setup cost
(PRNG state update, rejection sampling) are folded into `flop_cost`; the
weight tier **varies** by distribution family.  Billed cost = `flop_cost ×
weight`.

Weight tiers:

- **weight 1.0** — uniform/integer/structural draws: `rand`, `random`,
  `random_sample`, `ranf`, `sample`, `uniform`, `randint`, `integers`,
  `choice`, `shuffle`, `permutation`, `multivariate_normal`.
- **weight 16.0** — transcendental samplers (every continuous/transformed
  distribution): `normal`, `standard_normal`, `randn`, `exponential`,
  `standard_exponential`, `poisson`, `binomial`, `geometric`,
  `hypergeometric`, `negative_binomial`, `multinomial`, `beta`, `dirichlet`,
  `f`, `gamma`, `gumbel`, `laplace`, `logistic`, `lognormal`, `logseries`,
  `pareto`, `power`, `rayleigh`, `standard_cauchy`, `standard_gamma`,
  `standard_t`, `triangular`, `vonmises`, `wald`, `weibull`, `zipf`, and all
  their Generator / RandomState counterparts.

| Op / family | flop_cost | basis | source |
|---|---|---|---|
| `random.rand`, `random.random`, `random.random_sample`, `random.ranf`, `random.sample` | `numel(output)` | DECLARED: 1 FLOP per uniform draw | `_cost_formulas.py` |
| `random.uniform` | `3 × numel(output)` | DERIVED: affine map `low + (high − low) × U` = 1 sub + 1 mul + 1 add per element (FMA=2, three ops) | `_cost_formulas.py` |
| `random.randn`, `random.standard_normal`, `random.normal` | `numel(output)` (weight **16.0**) → billed `16 × numel` | DECLARED: flop_cost = numel(output); transcendental weight 16.0 from `default_weights.json` | `_cost_formulas.py` |
| `random.randint`, `random.integers` | `numel(output)` | DECLARED | `_cost_formulas.py` |
| `random.choice` (replace=True, p=None) | `numel(output)` | DECLARED | `_cost_formulas.py` |
| `random.choice` (replace=True, p≠None) | `numel(output) + 3n + m×⌈log₂ n⌉` (n=population, m=size) | DERIVED: cumsum + normalize + searchsorted | `_cost_formulas.py` |
| `random.choice` (replace=False, p=None) | `n` (O(n) shuffle-based sampling: conservative ceiling on tail-shuffle) | DECLARED | `_cost_formulas.py` |
| `random.choice` (replace=False, p≠None) | `sort_cost(n) = n × ⌈log₂ n⌉` (data-dependent rejection loop with weights) | DECLARED | `_cost_formulas.py` |
| `random.shuffle`, `random.permutation` | `numel(input)` | DECLARED: O(n) in-place shuffle | `_cost_formulas.py` |
| `random.exponential` | `numel(output)` (weight **16.0**) → billed `16 × numel` | DECLARED: transcendental weight 16.0 | `_cost_formulas.py` |
| `random.poisson`, `random.binomial`, `random.geometric`, `random.hypergeometric`, `random.negative_binomial`, `random.multinomial` | `numel(output)` (weight **16.0**) → billed `16 × numel` | DECLARED: transcendental weight 16.0 | `_cost_formulas.py` |
| `random.multivariate_normal` | `26d³ + 2Nd² + 16Nd` (d=dims, N=size) | DERIVED composite: SVD factorization of covariance (`svd_cost(d,d,with_vectors=True)` = `6d·d² + 20d³` = `26d³`) + affine transform (`2Nd²`) + N·d transcendental normal draws (`16Nd`) | `_cost_formulas.py` |
| `random.beta`, `random.dirichlet`, `random.f`, `random.gamma`, `random.gumbel`, `random.laplace`, `random.logistic`, `random.lognormal`, `random.logseries`, `random.pareto`, `random.power`, `random.rayleigh`, `random.standard_cauchy`, `random.standard_exponential`, `random.standard_gamma`, `random.standard_t`, `random.triangular`, `random.vonmises`, `random.wald`, `random.weibull`, `random.zipf` | `numel(output)` (weight **16.0**) → `16 × numel` | DECLARED: flop_cost = numel(output); transcendental weight 16.0 for all continuous/transformed distributions | `_cost_formulas.py` |

Source: `src/flopscope/numpy/random/_cost_formulas.py`.

---

### Stats

Stats ops are composite (weight 1.0; all per-element factors in `flop_cost`).

| Op | flop_cost (per element) | basis |
|---|---|---|
| `stats.norm.pdf` | 27 | DERIVED: exp(17) + affine normalization(10); composite, weight 1.0 |
| `stats.norm.cdf` | 48 | DERIVED: erf rational approx(45) + affine(3); composite, weight 1.0 |
| `stats.norm.ppf` | 83 | DERIVED composite: degree-5 rational approximation + Newton step (erf + pdf + correction) + affine |
| `stats.expon.pdf` | 22 | DERIVED: z=(x−loc)/scale(2) + exp(−z)(17) + /scale(1) + where(2); weight 1.0 |
| `stats.expon.cdf` | 22 | DERIVED: z(2) + exp(−z)(17) + 1−exp(1) + where(2); weight 1.0 |
| `stats.expon.ppf` | 27 | DERIVED: loc−scale·log1p(−q)(19) + 3 where/cmp/and(8); weight 1.0 |
| `stats.cauchy.pdf` | 6 | DERIVED pure-arithmetic: z=(x−loc)/scale; 1/(π·scale·(1+z²)) = 6 FLOPs/elem; weight 1.0 |
| `stats.cauchy.cdf` | 20 | DERIVED: z(2) + arctan(16) + /π(1) + 0.5+(1); weight 1.0 |
| `stats.cauchy.ppf` | 28 | DERIVED: q−0.5(1) + π·(1) + tan(16) + loc+scale·(2) + 3 where(8); weight 1.0 |
| `stats.logistic.pdf` | 23 | DERIVED: z(2) + exp(−z)(17) + (1+ez)(1) + sq(1) + scale·(1) + div(1); weight 1.0 |
| `stats.logistic.cdf` | 21 | DERIVED: z(2) + exp(−z)(17) + 1+ez(1) + 1/denom(1); weight 1.0 |
| `stats.logistic.ppf` | 28 | DERIVED: 1−q(1) + q/(1−q)(1) + log(16) + loc+scale·(2) + 3 where(8); weight 1.0 |
| `stats.laplace.pdf` | 22 | DERIVED: \|x−loc\|(3) + exp(−z)(17) + /(2·scale)(2); weight 1.0 |
| `stats.laplace.cdf` | 40 | DERIVED composite: two eager exp branches + arithmetic/select; weight 1.0 |
| `stats.laplace.ppf` | 51 | DERIVED composite: two eager log branches + edge selects; weight 1.0 |
| `stats.truncnorm.pdf` | 28 | DERIVED composite: norm.pdf + cdf normalization; weight 1.0 |
| `stats.truncnorm.cdf` | 51 | DERIVED composite: affine + norm.cdf + boundary selects; weight 1.0 |
| `stats.truncnorm.ppf` | 81 | DERIVED composite: affine + rational + Newton with erf+exp; weight 1.0 |
| `stats.lognorm.pdf` | 62 | DERIVED composite: log + exp + arithmetic per element; weight 1.0 |
| `stats.lognorm.cdf` | 70 | DERIVED composite: log + erf rational approx + arithmetic; weight 1.0 |
| `stats.lognorm.ppf` | 106 | DERIVED composite: ndtri + exp; weight 1.0 |
| `stats.uniform.pdf` | 1 | DECLARED: 1 FLOP/elem |
| `stats.uniform.cdf` | 4 | DERIVED: sub + div + 2 clip compare/selects; weight 1.0 |

Source: `src/flopscope/stats/`.

---

### Window

| Op | flop_cost | basis | source |
|---|---|---|---|
| `bartlett` | `4n` (weight 1.0) | DERIVED: compare + divide + add + select per sample (FMA=2, 4 ops/sample) | `_window.py:bartlett_cost` |
| `blackman` | `40n` (weight 1.0) | DERIVED composite: 2 cosine evals at transcendental rate (16/elem each) + 8 mul/div/add per sample; all folded into flop_cost | `_window.py:blackman_cost` |
| `hamming` | `2n` (weight 8.0) | DECLARED: cosine eval per sample at the half-transcendental tier | `_window.py:hamming_cost` |
| `hanning` | `2n` (weight 8.0) | DECLARED: cosine eval per sample at the half-transcendental tier | `_window.py:hanning_cost` |
| `kaiser` | `23n` (weight 1.0) | DERIVED composite: 1 Bessel I₀ eval at transcendental tier (16/elem) + 7 scalar FLOPs per sample; folded into flop_cost | `_window.py:kaiser_cost` |

Source: `src/flopscope/_window.py`.

---

### Interp and histogram

| Op | flop_cost | basis | source |
|---|---|---|---|
| `interp` | `3m + m × ⌈log₂(numel(xp))⌉`, `m = numel(x)` (interpolation arithmetic + binary search per query) | DERIVED | `_counting_ops.py` |
| `histogram` (integer bins) | `n × ⌈log₂(bins)⌉` (binary-search binning pass only) | DERIVED | `_counting_ops.py` |
| `histogram` (string bins, e.g. `'auto'`) | `n × (2 + estimator_cost + ⌈log₂ resolved_bins⌉)` (deferred: resolved after the call; estimator costs: sturges/sqrt/rice=0, fd/auto=+1n, scott=+4n, doane=+6n, stone=+max(100,√n)n) | DERIVED | `_counting_ops.py` |
| `histogram2d`, `histogramdd` | same as `histogram` per axis | DERIVED | `_counting_ops.py` |
| `histogram_bin_edges` | `n` (= `max(n, 1)`) for integer bins; string estimator bins: same formula as `histogram` string path | DECLARED: integer bins charge one comparison per element (no log₂ factor); estimator resolves bin count at call time | `_counting_ops.py` |
| `trapezoid`, `trapz` | `4 × numel(y)` | DERIVED: `(d·(y₁+y₂)/2).sum()` ≈ 3 elementwise ops + sum-reduce per point, charged as a clean 4/point upper bound | `_pointwise.py`; fixed in this branch |

Source: `src/flopscope/_counting_ops.py`, `src/flopscope/_array_ops.py`.

---

### Set ops

| Op | flop_cost | basis |
|---|---|---|
| `unique`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values` | `n × ⌈log₂ n⌉` | DECLARED sort-based |
| `in1d`, `isin` | `(n+m) × ⌈log₂(n+m)⌉` | DECLARED sort-based |
| `intersect1d` | `sort_cost(n) + sort_cost(m) + sort_cost(n+m)` (default); `sort_cost(n+m)` when `assume_unique=True` | DECLARED: pre-sorts both inputs when `assume_unique` is falsy |
| `setdiff1d`, `setxor1d`, `union1d` | `(n+m) × ⌈log₂(n+m)⌉` | DECLARED sort-based |
| `searchsorted` | `m × ⌈log₂ n⌉` | DECLARED binary search |

Comparison = 1 FLOP convention; weight 1.0.

---

### Counting (diff, ediff1d, clip, allclose, isclose, count_nonzero, trace)

| Op | flop_cost | basis | source |
|---|---|---|---|
| `clip` | `max(n_bounds, 1) × numel(output)` (1 compare-select per bound; n_bounds=0,1,2; floor of 1 ensures materialising copy is not free) | DERIVED | `_pointwise.py` |
| `count_nonzero` | `numel(input)` (every element tested regardless of axis; comparison-scan model) | DECLARED | `_pointwise.py` |
| `diff` | `prod(a.shape[:ax]) × (n×L − n×(n+1)/2) × prod(a.shape[ax+1:])`, `L = a.shape[ax]` | DERIVED: `n` passes of `L−k` subtractions | `_pointwise.py` |
| `ediff1d` | `ary.size − 1 + size(to_begin) + size(to_end)` | DECLARED | `_pointwise.py` |
| `gradient` | base: `sum_ax 2·S·(L−2)/L`; each coord-array axis adds a spacing surcharge (uniform: `+3(L−1)`; non-uniform: `+3S(L−2)/L + 10(L−2) + 3(L−1) + 4S/L`) | DERIVED | `_pointwise.py:gradient` |
| `allclose` | `7·numel(broadcast) − 1` (6 FLOPs/elem tolerance core + numel−1 all-reduce) | DERIVED | `_counting_ops.py` |
| `isclose` | `6·numel(broadcast)` (sub + 2·abs + mul + add + cmp per element) | DECLARED | `_pointwise.py` |
| `trace` (numpy.trace) | `min(ax1, ax2) × n_traces` where `n_traces = size / (shape[ax1] × shape[ax2])` (batch-multiplied) | DERIVED | `_counting_ops.py:trace` |
| `correlate` | mode-aware: `full` = `2nm−n−m+1`; `valid` = `(2·min−1)·(max−min+1)`; `same` = exact dot-length sum per numpy C layout | DERIVED per-mode | `_pointwise.py:_correlate_cost` |
| `cross` | `3 × numel(output)` (2 muls + 1 sub per output scalar; 3-vec path preserves last dim, 2-D z-only drops last dim) | DERIVED: FMA=2, 3 FLOPs per output element | `_pointwise.py:cross` |
| `cov` | `2f²s + 2fs` (f = features, s = samples) | DERIVED: Gram term `f²` dot products of length `s` (2f²s) + centering pass `fs` elements × 2 FLOPs | `_pointwise.py:_cov_cost` |
| `corrcoef` | `2f²s + 2fs + 2f² + f` | DERIVED: cov_cost + normalization (f² divides at weight 2.0 + f sqrts) | `_pointwise.py:_corrcoef_cost` |
| `unwrap` | `11 × numel(input)` | DERIVED: 11 charged passes (diff, mod, cmp×2, bitwise, sub, abs, cmp, cumsum); 2 select passes (steps 8/12) are 3-arg where = free; prior value was 13 | `_unwrap.py:unwrap_cost` |

---

### Copy and gather

**Family rule: free — pure relocation/selection.**

Data-movement ops that copy, rearrange, or select-by-a-given-selector carry **weight 0**
and bill `flop_cost = 0`.  They produce no per-element arithmetic and derive no selector
by testing values — they only move existing values into a new buffer or layout.  This
covers: `concatenate`, `stack`, `hstack`, `vstack`, `column_stack`, `dstack`, `block`,
`bmat`, `tile`, `repeat`, `resize`, `roll`, `tril`, `triu`, `insert`, `append`,
`delete`, `diag` (both extract and construct), `diagflat`, `fill_diagonal`,
`take`, `take_along_axis`, `put`, `put_along_axis`, `choose`, `compress`,
`extract`, `select`, `place`, `putmask`, `where(cond, x, y)` (3-arg), `unstack`, and
all other ops from the copy/materialize/gather/scatter families.  (`pad`, `copyto`, and
`trim_zeros` are **not** unconditionally free — see
[§Boundary ops](#boundary-ops-free-behavior--a-value-computing-path).)

**Selector-deriving siblings are charged** (they test values to produce the selector):

| Op | flop_cost | basis |
|---|---|---|
| `nonzero`, `where(cond)` (1-arg) | `numel(input)` (weight 1.0) | DECLARED: implicit `!= 0` scan per element |
| `argwhere` | `numel(input)` (weight 1.0) | DECLARED: ≡ `transpose(nonzero(a))` |
| `flatnonzero` | `numel(input)` (weight 1.0) | DECLARED: ≡ `nonzero(a.ravel())` |
| `count_nonzero` | `numel(input)` (weight 1.0) | DECLARED: comparison scan every element |

These ops derive a selector by testing element values (`!= 0`), so the test is their
compute cost.  The predicate and the selection are the *same* step here — unlike the
3-arg `where(cond, x, y)` where the predicate (a separate charged op) is an *input*.

**Worked examples**:

| Expression | Charge | Reasoning |
|---|---|---|
| `where(a > 0.5, x, y)` | pay `greater` = `numel(a)` for the predicate; the `where` (select) is free | predicate tests values (charged separately); selection by given mask is logistics |
| `nonzero(a)` | charged `numel(a)` | derives the selector by testing `!=0` — value-test is its compute |
| `arange(n)` | charged `2×numel` | computes `start + i·step` per element (1 mul + 1 add) |
| `meshgrid(x, y)` | free | replicates `x`,`y` into grids; no per-element arithmetic |
| `take(a, idx)` | free | index given; pure gather |
| `hstack([a, b])` | free | copies existing values into a new buffer |
| `sort(a)` | charged `n·⌈log₂ n⌉` | output order derived by comparing values |
| `a.astype(float64)` | free | width cast = representation only (no value change) |
| `a.astype(bool)` | charged `numel(a)` | per-element `!=0` test = value-comparison |

Source: `src/flopscope/_array_ops.py`.

---

#### Copy-and-gather: ops with distinct charged siblings

The table below lists ops whose cost formula differs from 0 because they contain
value-arithmetic or perform I/O work beyond pure relocation:

| Op | flop_cost | basis | source |
|---|---|---|---|
| `diag` (extract, 2-D) | 0 (free — pure gather of diagonal elements) | DECLARED: no arithmetic | `_array_ops.py` |
| `diag` (construct, 1-D) | 0 (free — copy into diagonal of new matrix) | DECLARED: no arithmetic | `_array_ops.py` |
| `diagonal` | 0 (view) | DECLARED: `numpy.diagonal` returns a read-only view | `_array_ops.py` |
| `copyto` | 0 for same-dtype / `where`-mask copy / lossless widening; `numel(dst)` (or popcount(`where`)) for a value-changing (lossy) cast | DERIVED: path-aware — pure scatter-write and lossless width casts are free, a value-changing (lossy) cast is charged (see [§Boundary ops](#boundary-ops-free-behavior--a-value-computing-path)) | `_array_ops.py` |
| `packbits` | `numel(input)` (weight 1.0) | DECLARED: per-bit test+shift; value-test per element | `_array_ops.py` |
| `unpackbits` | `numel(output)` (weight 1.0) | DECLARED: unpacks 8 bits per input byte; proportional to output | `_array_ops.py` |
| `mask_indices` | `2n² + 8k` (weight 1.0, `k` = selected pairs) | DECLARED: n² mask scan (value test) + gather of 2k index values | `_array_ops.py` |

---

### Boundary ops (free behavior + a value-computing path)

A free (weight-0) classification covers only an op's **pure data-movement / structural**
behavior. Any parameter, mode, or path that **computes or inspects values** is **charged**
with a reliable cost reusing the convention for that work; a path we cannot reliably bill
is **rejected with a clear error**. These four ops carry weight **1.0** with a path-aware
`flop_cost`:

| Op | free path (`flop_cost = 0`) | charged / rejected path |
|---|---|---|
| `pad` | `constant`, `edge`, `empty`, `wrap`, `reflect`/`symmetric` (`reflect_type='even'`) | stat modes `maximum`/`minimum`/`mean`/`median`: `Σᵢ stats_i·stat_len_i·cross_i` (lanes from the input cross-section); `linear_ramp` and `reflect_type='odd'`: `2·(numel_out − numel_in)`; **`mode=<callable>` raises** |
| `ravel_multi_index` | — | `2·(ndim − 1)·N` (one unit stride), `+N` for `mode='clip'/'wrap'` |
| `trim_zeros` | — | `numel(input)` (value scan for the nonzero boundary) |
| `copyto` | same-dtype copy, `where`-mask copy, lossless widening | `numel(dst)` (or popcount(`where`)) for a value-changing (lossy) cast; lossless width casts are free |

For `pad` stat modes: `cross_i = numel_in // in_shape[i]`, `stat_len_i = min(stat_length_i,
in_shape[i])` (default = full axis), summed over padded axes only; a full-axis stat serves
both sides (one reduction). `mean` adds one divide per stat output cell.

---

### Functional / higher-order

Operations that apply a user-supplied callable across an array. flopscope bills the
result the wrapper materializes (numpy runs the callback itself).

> **Submission caveat:** these run a Python callback *in-process* and raise
> `RemoteCallbackError` on the client/server backend used for AIcrowd submissions, so
> they cannot appear in submitted code — their cost matters only for local runs.

| Op | flop_cost | source |
|---|---|---|
| `apply_along_axis`, `apply_over_axes` | `numel(output)` | `_counting_ops.py` |
| `fromfunction` | `numel(output)` | `_array_ops.py` |
| `piecewise` | `numel(output)` (the op bills its assembled result; each condition you pass in `condlist` is billed separately as its own comparison op) | `_counting_ops.py` |

---

### View / free (weight 0.0)

**Family rule**: operations that return a view, re-interpret memory, or
inspect metadata without touching element values charge 0 FLOPs.

Weight 0 now covers *four* sub-families (see [§The unifying philosophy](#the-unifying-philosophy--compute-not-logistics)
in the Billing model section for the full rule and both refinements):

- **Views / metadata**: `reshape`, `ravel`, `flatten`, `transpose`, `squeeze`,
  `expand_dims`, `broadcast_to`, `atleast_1d/2d/3d`, `asarray` (no copy),
  `asfortranarray`, `ascontiguousarray`, `astype` (no copy / lossless-width),
  `view`, `diagonal` (view), `moveaxis`, `swapaxes`,
  `ndim`, `shape`, `size`, `nbytes`, `itemsize`, `dtype`, `flags`, `base`,
  `data`, `ctypes`, `strides`, `T`, `linalg.diagonal`, `linalg.matrix_transpose`,
  `fft.fftshift`, `fft.ifftshift`, `isscalar`, `isfortran`.
- **Copy / materialize**: `concatenate`, `stack`, `hstack`, `vstack`,
  `column_stack`, `dstack`, `block`, `bmat`, `tile`, `repeat`, `resize`,
  `roll`, `tril`, `triu`, `copy`, `insert`, `append`, `delete`, `diagflat`,
  `fill_diagonal`, `unstack`, and kin.
- **Gather / scatter & mask-select (selector given)**: `take`, `take_along_axis`,
  `put`, `put_along_axis`, `choose`, `where(cond, x, y)` (3-arg), `select`,
  `compress(mask, a)`, `extract(mask, a)`, `place`, `putmask`.
- **Constant init**: `zeros`, `ones`, `empty`, `full`, `eye`, `identity`, `tri`,
  `zeros_like`, `ones_like`, `empty_like`, `full_like`, `meshgrid`.

Source: `src/flopscope/_array_ops.py`.

---

## Exhaustive per-op reference

The complete, per-op cost data lives in **`website/public/ops.json`** — one record per
operation with `name`, `module`, `area`, `category`, `weight`, `cost_formula`,
`cost_formula_latex`, `notes`, and `summary`. It is **generated** from the registry +
weight tables by `scripts/generate_api_docs.py` and powers the website's API pages.

- **Find an op:** filter `ops.json` by `name`, or browse the website API pages.
- **Filter a family:** by `area` (`core` / `fft` / `linalg` / `random` / `stats`) or `module`.
- **It can't drift:** CI runs `scripts/generate_api_docs.py --check`, which regenerates
  `ops.json` to a temp dir and fails if the committed file's **cost-model fields**
  differ (`weight`, `cost_formula`, `category`, `notes`, …). The `summary` field is
  sourced from the installed numpy's docstrings and is allowed to vary across the
  numpy-version matrix, so it is excluded from the check — which means the gate also
  proves the cost model is numpy-version-independent. Every billed op is present
  (aliases resolve transitively to their canonical), enforced by
  `tests/test_cost_model_coverage.py`.

> **Granularity note.** `ops.json` is exhaustive in *coverage* — every op, with its
> weight and a formula string — but its `cost_formula` is **coarse for many composite
> `counted_custom` ops**, recording `per-operation` / `varies` where the real cost is
> shape-dependent. For those, the closed form and its derivation live in the family
> tables above. Treat `ops.json` as the complete index and this document as the precise
> reference; the completeness test ties them together.

