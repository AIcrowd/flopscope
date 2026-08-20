# Cost model reference

> **Start here.** This is the cost model's conceptual and audit reference. Read it
> to understand *how* billing works and to satisfy yourself that it is correct and
> non-gameable — you do **not** need to read every operation. The exhaustive,
> generated per-op list (every op with its `cost_formula` and `weight`) lives in
> [`ops.json`](#exhaustive-per-op-reference) and the website API pages; this doc
> explains the model by **family rule** so you can reason about a whole class at once.

flopscope bills compute as:

```
charged = int(flop_cost × dtype_rate × complex_factor × weight)
```

## How to read this

1. **[Billing model & design principles](#billing-model--design-principles)** — the one equation and *why* it is split into four factors.
2. **[Dtype and precision](#dtype-and-precision)** — how width (`dtype_rate`) and complex structure (`complex_factor`) are priced, and why.
3. **[Non-exploitability](#non-exploitability)** — the invariants that keep billing sound, and the test that enforces each.
4. **[Cost by family](#cost-by-family)** — the rule + evidence + representative ops for the family you care about.
5. **[Exhaustive per-op reference](#exhaustive-per-op-reference)** — drill into `ops.json` for one op's exact formula.

**Completeness guarantee:** every billed operation is classified in the registry and
appears in `ops.json` with a `cost_formula`; `tests/test_cost_model_coverage.py`
enforces both that, and that every op-class (`ops.json` *area*) is covered by a family
rule below. So nothing billed is undocumented, even where this doc summarizes by rule.

---

## Billing model & design principles

Every operation is charged `charged = int(flop_cost × dtype_rate × complex_factor × weight)`.

**Four factors, on purpose.** Each factor isolates one kind of reasoning so it can be
read off and audited on its own:

- **`flop_cost`** — the operation count. Every shape- and algorithm-dependent term lives
  here, so anyone — us or a participant — can read it off the formula and audit it
  function by function.
- **`dtype_rate`** — the width factor. One billed FLOP is one 32-bit-class real
  operation; a 64-bit-class operand bills `2×`. Like the weight ladder this is a *policy*
  table (subject to tuning), not shape math — defined in
  [Dtype and precision](#dtype-and-precision).
- **`complex_factor`** — the complex-structure factor. It is `1` for real dtypes and, for
  complex dtypes, the number of real (component-precision) operations one billed unit
  expands into (a complex multiply is 6 real FLOPs, a complex add 2). It is *math*, not
  policy — a textbook decomposition fixed per op, and computed per call for the
  contraction family from the same accumulation that produces `flop_cost`.
- **`weight`** — the hardware tier. A separate per-element factor for how much more one
  element costs on real hardware: a plain sequential add or write is `1`, a
  non-sequential memory access (a sort's comparison order, a computed-index gather, a
  random-reorder draw) is `4`, and a transcendental element (`sin`, `exp`, …) does many
  times more floating-point work at `16`. Rather than bill each op its exact measured
  ratio — noisy, machine-specific, and hard to audit — we group operations into a small
  fixed ladder of tiers (`{0, 1, 4, 16}`); that grouping is a deliberate
  competition-design choice.

For a real 32-bit workload `dtype_rate` and `complex_factor` are both `1`, so the bill is
exactly `int(flop_cost × weight)` and nothing here changes.

The rule that keeps the split honest — and the model non-gameable — is one **separation
invariant**: *a shape or algorithm constant never lives in a weight* (anything depending
on a matrix dimension or loop length belongs in `flop_cost`); *width policy never lives in
`flop_cost` or `weight`* (it lives in `dtype_rate`); and *complex structure lives in
`complex_factor`* — computed per call for the contraction family from the same
decomposition that produces `flop_cost`. (The weight half is enforced by
`tests/test_weight_tier_policy.py`; the complex-classification half by
`tests/test_complex_factor_completeness.py`.)

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

All formulas count *operations*; the billing **unit** (one 32-bit-class real
op) and the per-width and complex-structure pricing are defined once in
[Dtype and precision](#dtype-and-precision).

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

### Access tier (weight 4.0)

Operations whose per-element cost is dominated by a **non-sequential memory access** —
comparison-sort order, binary search, a computed-index gather, or a random-reorder draw
— are billed at weight 4.0. `flop_cost` stays the plain algorithmic count (comparisons,
index dereferences, or draws); the 4× comes entirely from the weight, the same
one-factor-does-one-job split the transcendental tier uses.

This tier covers four families, each detailed in its own section below:

- **Sort / search** — `sort`, `argsort`, `lexsort`, `partition`, `argpartition`,
  `searchsorted` (see [Sort and select](#sort-and-select)).
- **Set / histogram** — `unique` and its NumPy 2.x siblings, `union1d`, `intersect1d`,
  `setdiff1d`, `setxor1d`, `histogram`, `histogram2d`, `histogramdd`,
  `histogram_bin_edges`, `bincount`, `digitize` (see [Set ops](#set-ops) and [Interp and
  histogram](#interp-and-histogram)).
- **Gather** — `take`, `take_along_axis`, `choose`: a computed-index read is a
  non-sequential memory access, unlike a materializing copy's sequential write (see
  [Copy and gather](#copy-and-gather)).
- **Random reorder** — `random.permutation`, `random.shuffle`, `random.choice`, and the
  `Generator`/`RandomState` `.choice` / `.permutation` / `.shuffle` (plus `Generator.permuted`)
  methods (see [Random](#random-module-level-generator-randomstate)). Note
  `random.sample` is *not* in this tier — despite the name it is numpy's alias for
  `random_sample`, a plain uniform draw.

A materializing copy or write (`concatenate`, `ones`, a scatter write) stays at weight
1.0 — it touches memory sequentially, one write per element. The access tier prices the
*pattern* of access, not merely the presence of a write.

### The unifying philosophy — every byte written is metered

> **flopscope meters both computation on values and the memory traffic that produces
> them.** A **view** is free — it creates no new buffer and touches no element. Any op
> that **writes a new buffer** is charged at least `1` per element written, whether the
> values it writes are computed (`sin(x)`), copied (`concatenate`), replicated
> (`tile`), or a repeated constant (`ones`). Ops whose access pattern is
> **non-sequential** — sorting, a computed-index gather, a random-reorder draw — are
> charged more per element (`4`) than a straight sequential write (`1`), because that
> is where real hardware actually spends more per element: cache-friendly sequential
> writes are cheap, branchy/random-order accesses are not.

**The decision procedure** — apply these steps in order to any op:

1. **View / metadata only** (returns a view or read-only reinterpretation of existing
   memory, inspects shape/dtype, allocates no new buffer)? → **Free (0).** Reshape,
   ravel, and `copy` do **not** qualify even when NumPy happens to return a view —
   flopscope bills them as if they always materialize, so the price cannot depend on a
   layout coincidence (see [Views and metadata](#views-and-metadata-weight-00)).
2. **Does it write a new buffer sequentially** — a materializing copy (`concatenate`,
   `tile`, `roll`), a constant fill that isn't the zero-page default (`ones`, `full`,
   `eye`'s diagonal), a scatter write at a *given* index (`put`, `putmask`,
   `fill_diagonal`), or output-shaped selection (`select`, `compress`, `extract`)? →
   **Charged at weight 1.0.** `flop_cost` is the count of elements actually written
   (not the whole array when only part of it changes, e.g. `eye`'s off-diagonal zeros
   are free).
3. **Does it read via a *non-sequential* access pattern** — a computed-index gather
   (`take`, `take_along_axis`, `choose`), a comparison-order derivation (`sort`,
   `unique`, `searchsorted`), a per-element conditional select (`where(cond, x, y)`,
   which dereferences a different source per output element and bills
   `4 × numel(broadcast output)`), or a random-reorder draw (`shuffle`, `permutation`,
   `choice`)? → **Charged at weight 4.0** (see [Access tier](#access-tier-weight-40)).
4. **Does it produce output values by doing floating-point arithmetic, *or* by
   comparing element values?** → **Charged.** `flop_cost` = standard-algorithm op
   count; `weight` = hardware tier. This includes elementwise math, transcendentals,
   reductions, contraction (matmul/einsum), FFT, polynomial, random generation, and
   ops that *derive* a result by *testing values*: `nonzero`/`argwhere`/
   `flatnonzero`/`count_nonzero`/`where(1-arg)`, `clip`/`minimum`/`maximum`, and
   *computed creators* (`arange`/`linspace`/`geomspace`/`logspace`/`vander`).
5. **Otherwise** — allocates a new buffer but writes nothing into it (`zeros`,
   `empty`) → **Free (0).** The OS zero-page (or uninitialized allocation) costs
   nothing to hand out; a participant only pays once real values land in it.

**Key invariant:** any predicate or index feeding a step-2/3 op was itself produced by
an earlier charged step. A free-tier op may **never bundle** value-arithmetic,
value-comparison, or a non-zero write into its own cost.

The active weights are `{0, 1, 4, 16}`. Weight 0 is now a narrow band — genuine views,
metadata, and all-zero/uninitialized allocation. Almost everything that writes a new
buffer bills at least weight 1.0; the historical "free data movement" framing (copy,
gather, scatter, and select-with-a-given-selector all billed 0) has been replaced by
this write-metered model, since a participant who can move arbitrary amounts of data
for free can launder real compute through a materializing copy chain.

### Views and metadata (weight 0.0)

Weight 0 is now a narrow band — a new buffer is never allocated, or it is allocated
but nothing is written into it:

1. **Views / metadata** — operations that return a view of existing memory or inspect
   metadata without touching element values: `transpose`, `swapaxes`,
   `moveaxis`, `squeeze`, `expand_dims`, `flip`/`fliplr`/`flipud`, `rot90`,
   `atleast_1d`/`atleast_2d`/`atleast_3d`, `broadcast_to`, `view`,
   `real`/`imag` (component extraction), `split`, `hsplit`, `vsplit`,
   `array_split`, `unstack`, `diagonal` (the 2-D view path — see
   [Copy-and-gather](#copy-and-gather) for the
   1-D *construct* path, which writes), `linalg.diagonal`, `linalg.matrix_transpose`,
   `from_dlpack` (zero-copy ingest), and all other shape/stride/dtype introspection
   (`ndim`, `shape`, `size`, `nbytes`, `itemsize`, `dtype`, `flags`, `base`, `data`,
   `ctypes`, `strides`, `T`, `isscalar`, `isfortran`).
   **`reshape`, `ravel`, `copy`, `flatten`, and `require` do *not* belong here** —
   flopscope bills them `numel(input)` at weight 1.0 unconditionally, even on the
   (common) call pattern where NumPy itself returns a view. Billing the cautious,
   always-charged price avoids a participant relying on a layout coincidence
   (contiguity, stride pattern) to get a real copy for free. `flatten` is not even a
   coincidence: `ndarray.flatten` is documented by NumPy to always allocate a new
   buffer, never a view.
2. **All-zero / uninitialized allocation** — a new buffer whose contents are the
   platform zero-page default, so nothing is actually written: `zeros`, `zeros_like`,
   `empty`, `empty_like`. Any *other* constant fill (`ones`, `full`, `eye`'s diagonal,
   `identity`, `tri`) writes real, non-zero values and is charged — see
   [Copy and gather](#copy-and-gather).

**Refinement — representation vs. value change** (resolves `astype`/`asarray`):

> **A representation change that performs no work is free; every cast or copy that
> actually runs is charged the same as `copy`.**

`astype` and `asarray` bill `numel(input)` at the heavier of the source/destination
dtype rate — the same formula `copy` bills (see [Which dtype prices a
call](#which-dtype-prices-a-call)) — for every call that does real work: narrowing,
widening, float→int truncation, bool coercion, a same-dtype `astype` called with the
default `copy=True`, and any `asarray(x, dtype=other)` that actually converts the
buffer. The one free case left is the genuine no-op, where NumPy itself performs zero
work: `astype(x, dtype, copy=False)` when `dtype` already equals `x`'s dtype returns
the *identical object*, and `asarray(x)` with no `dtype=` (or a `dtype=` that already
matches `x`'s dtype) performs no conversion — both bill `0`, same as the views/metadata
ops above. Any other combination — `copy=False` across an actual dtype change (NumPy
cannot honor the request and copies anyway), or any other `astype`/`asarray` call —
performs a real `numel`-element write and is charged.

This closes a loophole where `x.astype(x.dtype)` (default `copy=True`) was a free
substitute for `x.copy()`, and `x.astype(bool)` / `x.astype(int)` were free substitutes
for a billed `!= 0` test or a `trunc`-then-cast. It also means choosing a narrower
working dtype now costs a one-time `copy`-priced toll to get there; the arithmetic
performed *afterward* on the narrower array is still cheaper per the dtype-rate table
(see [Which dtype prices a call](#which-dtype-prices-a-call)), so that saving is not
eliminated — only no longer free to acquire up front. Contrast the charged cases with
`ones`/`full`, which also touch every element but write an actual value into memory —
those are charged the same way (see [Views and metadata](#views-and-metadata-weight-00)
category 2 above and [Copy and gather](#copy-and-gather)).

The method `a.nonzero()` is charged identically to `fnp.nonzero(a)`; `nonzero`,
`argwhere`, `flatnonzero`, `count_nonzero`, and `where` (1-arg) remain charged
`numel(input)` at weight 1.0 — they derive their result by testing element values
(`!= 0`), and that test is real compute regardless of how "small" the eventual output
is (see [Copy and gather](#copy-and-gather) for the full selector-family accounting).

### Composite ops (weight 1.0 with heterogeneous flop_cost)

When an operation mixes sub-tiers internally (e.g. random samplers, stats
kernels, norms with SVD), all per-element factors are folded into `flop_cost`
and the active weight is set to 1.0.  This avoids double-counting with the
tier factor.

### NumPy 2.x ufunc aliases

NumPy 2.x introduced `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `atan2`, and
`pow` as canonical aliases for their `arc*` / `power` counterparts — literally the same
ufunc object (`np.acos is np.arccos`). flopscope resolves these via
`_UFUNC_ALIAS_RENAMES` in `_weights.py` so each alias charges the same **weight** as its
canonical twin.

`divmod` rides the same rename map but is not a true 1:1 alias: `np.divmod` is a distinct
two-output ufunc (`nin=2, nout=2`) from `np.floor_divide` (`nin=2, nout=1`) —
`np.divmod is np.floor_divide` is `False`. `_UFUNC_ALIAS_RENAMES` only borrows
`floor_divide`'s **weight** (16.0) for `divmod` as a conservative floor; `divmod`'s
`flop_cost` is its own `nout=2 × numel(output)` (see
[Elementwise](#elementwise-pointwise-unary-and-binary)), so it bills 2× `floor_divide`'s
total, not the same total.

---

## Dtype and precision

`flop_cost` counts operations, but two operations of the same shape can do very different
amounts of real floating-point work depending on their **dtype**. Two factors price that:
`dtype_rate` (how wide each scalar is) and `complex_factor` (how much a complex operation
expands into real ones). Both sit in the billing product
`int(flop_cost × dtype_rate × complex_factor × weight)`, and both are `1.0` for the
32-bit real baseline — so a real-fp32 workload bills exactly its `flop_cost × weight` and
nothing in this section changes it.

### Why width and structure are priced

We identified two ways a dtype-blind meter under-counts:

**Complex packing.** A complex multiply does the work of four real multiplies and two
real adds, but a dtype-blind meter would bill `x * z` (with `z` complex) as one real
multiply. A participant can then carry two independent real payloads in the real and
imaginary lanes of one complex array and recover both products from a single "one-FLOP"
multiply — buying two real results for less than the price of one. The same trick scales
through every complex-valued op, most sharply through matmul, where one complex matmul
stands in for two real ones.

**Width (bit) packing.** A float64 lane is twice as wide as a float32 lane. A dtype-blind
meter bills a float64 multiply the same as a float32 multiply, so a participant can pack
two independent lower-precision payloads into the two halves of a 64-bit lane (or a narrow
integer into a wide one) and process both under a single op's charge.

The answer to both is the same, and it is **not** to ban wide or complex dtypes — it is
to **price them**. `dtype_rate` charges a 64-bit-class op `2×` a 32-bit-class op;
`complex_factor` charges a complex op the real-operation count it actually expands into.
Under that pricing neither pack pays off (see [On packing](#on-packing)).

### The billing unit and rate table

> **One billed FLOP is one 32-bit-class real scalar operation** — a float32 or int32 add,
> multiply, compare, and so on.

Everything is priced relative to that anchor. `dtype_rate` is the width multiplier for the
call's resolved dtype:

| dtype | rate | | dtype | rate |
|---|---|---|---|---|
| `bool` | 1.0 | | `int32` | 1.0 |
| `int8` | 1.0 | | `uint32` | 1.0 |
| `uint8` | 1.0 | | `float32` | 1.0 |
| `int16` | 1.0 | | `int64` | 2.0 |
| `uint16` | 1.0 | | `uint64` | 2.0 |
| `float16` | 1.0 | | `float64` | 2.0 |
| `float96` | 3.0 | | `complex64` | 1.0 |
| `float128` | 4.0 | | `complex128` | 2.0 |
| `complex192` | 3.0 | | `complex256` | 4.0 |

- **32-bit-class and narrower → `1.0`.** float16/int16/int8/bool all bill at the baseline
  width.
- **64-bit-class → `2.0`.** float64, int64, uint64 do twice the scalar work of the
  baseline width.
- **Extended precision → width class.** `float96` (3.0) and `float128` (4.0) are the
  platform `longdouble` types (Linux only; on macOS/Windows `longdouble` aliases
  `float64`). At these rates, packing narrower payloads through their wider mantissas
  still loses: two float32-payload products in one `float128` multiply cost `4.0`
  against the honest `2 × 1.0`. Note these dtypes cannot cross the evaluation wire
  (the transfer codec accepts the 14 standard dtypes) — they are priced for in-process
  compute only.
- **complex → its component width's rate.** `complex64` is two float32 components → `1.0`;
  `complex128` is two float64 components → `2.0`; `complex192`/`complex256` follow their
  extended components (3.0 / 4.0). The *structure* cost of being complex (that a complex
  multiply is several real ones) is carried separately by `complex_factor`, not folded
  in here.

These are **current policy values, subject to tuning** — the same status as the weight
ladder, and configured in the same file (`src/flopscope/data/default_weights.json`, under
`"dtype_rates"`). They are a deliberate competition-design choice, not a hardware
measurement.

**Fail-closed for unknown numeric dtypes.** Any *numeric* dtype not in the table (a
future type numpy or an extension package might introduce) has no defined rate, so
billing **raises `UnsupportedDtypeError` before charging any FLOPs** rather than
guessing — a numeric dtype cannot slip through unpriced; supporting a new one is a
deliberate one-line policy addition to the table. *Non-numeric* dtypes (`object`,
`str_`, `bytes_`, `datetime64`, `timedelta64`, structured/void, object-free or not)
are refused outright rather than priced at any rate — see the note directly below.

> **Non-numeric dtypes are not supported.** flopscope bills only the numeric
> kinds — bool, signed/unsigned integer, float, complex (`dtype.kind in
> "biufc"`) — a NUMERIC ALLOWLIST rather than a denylist of the non-numeric
> kinds, so a dtype kind flopscope has never seen (a future numpy or
> extension-package type) is refused by default instead of silently priced.
> Two distinct dtype families fall outside it. `object` (and any
> structured/void dtype embedding an object field — this subsumes the
> `numpy.dtype.hasobject` check) can hold unbounded per-element computation
> that no rate expresses. `str_`/`bytes_`/`datetime64`/`timedelta64`/
> structured-void — object-free included — are bounded, but their real
> per-element cost is not the fixed unit a flat rate would have to assume: a
> wide string or record does more work than a narrow one, and
> datetime64/timedelta64 are integers underneath, at whatever the platform's
> integer rate is, not a dtype-blind flat one. Both are refused the same
> way, whenever the dtype reaches a counted operation as an operand (an
> ndarray, or a Python sequence NumPy would coerce into one), an explicit
> `dtype=`, a fill value or distribution parameter about to be cast the same
> way, or an `out=` destination — flopscope raises `UnsupportedDtypeError`
> instead of billing it, including flopscope's own conversion ops
> (`array`/`asarray`/`astype`/`fromiter`/`require`/`full`/`full_like`) and
> every random sampler, which refuse non-numeric input rather than convert
> or relocate it. A raw Python sequence is judged by its leaves: `str`,
> `bytes`, a NumPy scalar whose own dtype is non-numeric, and a stdlib
> `date`/`time`/`timedelta` are refused wherever they appear inside a
> list/tuple operand, so `fnp.random.choice(['a','b','c','d'], 2)` is refused
> exactly like the `np.array(['a','b','c','d'])` spelling of it. A bare
> string ARGUMENT is untouched — `ord='fro'`, an einsum subscript, a
> `casting=` mode and a `requirements=`/`order=`/`signature=`/`optimize=`
> specifier are options, not payloads. One exception on the dtype side: a dtype that is
> still zero-itemsize once NumPy *materialises* it (an empty structured spec,
> or `'V0'`) is let through regardless of kind — zero bytes per element
> cannot embed an object field or carry any itemsize-dependent cost, and
> NumPy's own internals allocate one as a zero-byte shape-computation
> placeholder. A zero-length *string* dtype is not in that exception: NumPy
> promotes `'U0'`/`'S0'` to `'U1'`/`'S1'` on allocation, so it is refused
> like any other string dtype. To work with mixed, ragged, or
> non-numeric data, convert with plain NumPy *before* it reaches flopscope —
> `clean = np.array(x, dtype=np.float64)`, **not** `fnp.array(...)`, since
> flopscope's own conversion ops refuse non-numeric input too — or hold it
> in a Python list of numeric arrays and process each one separately.

### Complex arithmetic from first principles

`complex_factor` answers one question per op: *when the resolved dtype is complex, how many
real (component-precision) FLOPs does one billed unit of this op actually perform?* It is
`1.0` for real dtypes. For complex dtypes it is a fixed decomposition of the op into real
arithmetic (`z = a + bi`, FMA=2 convention):

| atom | real FLOPs | decomposition |
|---|---|---|
| add / subtract / negate | 2 | two real adds |
| multiply | 6 | 4 real mul + 2 real add |
| divide | 11 | numerator 4 mul + 2 add; denominator 2 mul + 1 add; 2 divides |
| reciprocal | 6 | special case of divide |
| fused multiply-add | 4 | 8 real FLOPs per FMA=2 unit (1 complex mul + 1 complex add) |
| absolute | 4 | abs(z) = sqrt(re² + im²): 2 mul + 1 add + 1 sqrt |
| sqrt | 10 | complex square root |
| ordering compare | 2 | lexicographic: compare real parts, tie-break on imaginary |

Every charged op is classified by the atom its **billed unit** reduces to:

| class | factor | representative ops |
|---|---|---|
| add / compare / sort / set / reduce-sum | **2** | two real adds, two lexicographic compares, or two component relocations — a complex value is two real components, and every op prices at least one unit per component: `add`, `subtract`, `sum`, `mean`, `cumsum`, `sort`, `unique`, `where`, comparisons (`less`/`greater`/`equal`), set ops, `conj`, `angle`, `concatenate`, `take`, `random.shuffle`, and the wider copy/gather/creation family |
| multiply / pure product | **6** | `multiply`, `prod` (`square` = 5) |
| divide | **11** | `divide` (`reciprocal` = 6) |
| variance family | **2.5** | `var`, `std` (square-and-sum: mostly adds, some multiplies) |
| absolute / magnitude | **4** | `abs` (`hypot` itself is complex-illegal: numpy raises) |
| transcendental | **per op** | `log` 2.25, `exp` 3.1, `sin`/`cos` 3.4, `tan` 3.6 — each from its complex closed form |
| dense linalg | **4** | `inv`, `solve`, `svd`, `det`, `qr`, `eig`, … (complex factorization ≈ 4× the real arithmetic) |
| contraction (FMA) | **exact** | `einsum`, `matmul`, `dot`, `inner`, `tensordot`, `vdot`, … — computed per call |
| FFT | **1** | no separate factor — the FFT formulas (`5N·log₂N`) already count the complex real-FLOPs: `fft.fft`, `fft.ifft`, `fft.rfft` |
| complex-illegal | **raises** | `floor`/`ceil`/`trunc`, bitwise & shift ops, `mod`/`fmod`/`floor_divide`, real-only math (`cbrt`, `i0`, `unwrap`, …) and the real-only random samplers — anything numpy rejects on complex input |

Composite `counted_custom` ops carry a per-op derived factor of the same kind (for example
`polyval` 4, `cross` 4.7, `interp` 3, `cov`/`corrcoef` 4). Two subtleties are worth
stating explicitly.

**Contraction is billed exactly, not with a flat factor.** A length-`K` dot product is `K`
multiplies + `K − 1` adds (real `flop_cost = 2K − 1`). In complex arithmetic that is `K`
complex multiplies + `K − 1` complex adds = `6K + 2(K − 1) = 8K − 2` real FLOPs. The ratio
`(8K − 2) / (2K − 1)` is `6` at `K = 1`, `≈ 4.13` at `K = 8`, and approaches `4` as
`K → ∞` — it is **not** a constant. So the contraction family carries the sentinel
`"exact"`, and the engine computes the complex total per call directly from the same
accumulation that produced `flop_cost`: `6·(multiplies) + 2·(adds)`, with the multiply and
add counts read straight off the einsum cost object. A single flat factor would invite
**alias-shopping** — route a multiply-heavy computation through the contraction whose flat
factor is cheapest. Exact per-call billing removes the arbitrage: `einsum('i,i->i', z, z)`
(a pure elementwise product, `K = 1`, all multiplies) bills factor 6, identical to
`multiply(z, z)`, not a discounted 4.

**FFT is priced in at factor 1.** The FFT `flop_cost` of `5N·log₂N` is *already* a count
of real FLOPs, so its `complex_factor` is `1.0`; applying a second complex factor would
double-charge. Derivation: a radix-2 FFT does `(N/2)·log₂N` butterflies; each butterfly is
one complex multiply (6 real FLOPs) + two complex adds (2 × 2 = 4 real FLOPs) = 10 real
FLOPs; `10 × (N/2)·log₂N = 5N·log₂N`. The complex structure is baked into `flop_cost`, so
`complex_factor` stays `1`.

**Guaranteed coverage.** Every charged op carries an explicit `complex_factor` (a number,
`"exact"`, or `"illegal"`); `tests/test_complex_factor_completeness.py` fails the build if
any charged op is unclassified. An op with no classification is necessarily
free / movement / blacklisted and bills factor `2` — a complex value is two real
components, and relocating one still prices one unit per component even though no
arithmetic happens.

### Which dtype prices a call

The billing dtype follows all work the call performs: the selected compute loop and any
destination buffer it materializes. Each op declares those participants at its deduct
site. Ordinary operands promote with `np.result_type`; Python scalars follow NumPy's
NEP 50 weak promotion, so `f32_array * 2.0` stays float32 (the scalar does not up-promote
the array). Single-output binary pointwise ufuncs — and their `ufunc.outer` spellings —
derive their compute price from NumPy's complete resolved loop signature, including every
arithmetic input slot and every output slot. That complete signature also determines
whether the registry's complex-arithmetic factor applies. The promoted operands provide an
input-rate floor: it can raise only `dtype_rate`, and does not invent complex arithmetic
when NumPy resolves a non-complex numeric loop (for example, a predicate's boolean
output). `out=` then joins as a separate participant under the
widest-participating-buffer doctrine.

**A control operand is not an arithmetic operand.** One ufunc slot in NumPy's table names
a number that *steers* the arithmetic instead of being arithmetic, and it prices neither
the loop nor the floor. `ldexp(x1, x2)` computes `x1 * 2**x2`: `x2` is the exponent, and
NumPy ships `ei->e` / `fi->f` / `di->d` (and their `l` variants) precisely so a wide
exponent does not drag the mantissa up a precision tier. `ldexp(f32, int32_array)`
resolves `fi->f`, computes `float32`, and bills at the float32 rate (`1000` for a
1,000-element call) — identical to `ldexp(f32, int8_array)`, because it is identical
mantissa work producing identical values. Membership is derived from the running NumPy's
loop table rather than a hard-coded list (a slot qualifies iff NumPy has a loop producing
an inexact result from an integral value there, and no loop accepting an inexact value
there); swept across NumPy's entire ufunc namespace that predicate selects exactly one
slot, `ldexp`'s exponent. This is the input-side mirror of the price-neutral `frexp`
exponent *output* described below, and of what `ufunc.at` has always billed for this
family.

Worked consequences:

- **Mixed kinds promote up.** `matmul(int32_array, float32_array)` resolves to `float64`
  (numpy's promotion for that mix) and bills at the float64 rate — the same as a genuine
  float64 matmul. There is no discount for feeding narrow-looking operands.
- **For these binary spellings, `dtype=` and `signature=` select computation, while
  `out=` prices materialization.** An explicit `dtype=` constrains NumPy's output DType;
  billing still resolves every arithmetic input slot and output slot of the resulting
  loop. For example, `ldexp(int8, int64, dtype=float16)` runs the
  `float16,int64->float16` loop and therefore bills at the float16 rate — the forced
  mantissa width, not the exponent's. For symmetric loops,
  `multiply(f64, f64, dtype=float32)` casts both operands on read and runs the float32
  loop, so a 1000-element call bills at the float32 rate (`1000`), not at the operands'
  float64 rate. A forced `signature=` (or its `sig=` alias) likewise prices the exact
  loop NumPy executes, including a forced complex loop's registry complex factor; an
  explicit `casting=` mode participates in that same resolution. `out=` is a separate
  participant: it cannot narrow compute, but a wider destination can raise the bill. Thus
  `multiply(f64, f64, out=float32_arr)` still computes and bills float64 (`2000`), while
  `multiply(f32, f32, out=float64_arr)` also bills `2000` because it materializes a
  genuine float64 buffer. This keeps `out=` casting at `astype` parity: a wider `out=`
  costs what the equivalent cast costs, and a narrower `out=` never discounts the loop.
- **`dtype=` and `out=` compose for these binary spellings.**
  `add(int8, int8, dtype=float16, out=float64_arr)` computes with the float16 loop but
  bills at the float64 rate under the widest-participating-buffer doctrine; without
  `out=`, the same explicit loop bills at the float16 rate.
- **For these binary spellings, a complex destination does not turn a non-complex loop
  into complex arithmetic.** When a loop with no complex signature slot stores into a
  complex `out=`, the destination still participates in width billing, but computation
  keeps its resolved classification. The call uses a store-only complex factor of `2.0`
  instead of the ufunc's complex-arithmetic factor.
- **A destination NumPy would have allocated anyway is price-neutral.** The rule above
  prices a destination as a buffer the call materializes — but two kinds of destination
  are not stores of the computed value at all, and naming them must cost exactly what
  letting NumPy allocate them costs. A **multi-output** op can have an output whose dtype
  belongs to the op signature rather than to its arithmetic: `frexp`'s second output is
  always `int32`, whatever precision the mantissa runs at, so `frexp(a_f32, out=(m_f32,
  e_i32))` bills what `frexp(a_f32)` bills (`20000` for a 10,000-element call — `frexp` is
  `nout=2`, so its own `flop_cost` is already `2 × numel(output)`, see
  [Elementwise](#elementwise-pointwise-unary-and-binary)), not the float64 rate that
  `result_type(float32, float32, int32)` would have promoted to. An
  **index reduction**'s destination holds positions, not values: `argmin`/`argmax`/
  `nanargmin`/`nanargmax` return `intp` regardless of the input's precision, so
  `argmin(a_f32, out=intp_arr)` bills what `argmin(a_f32)` bills (`9999`). Widening past
  NumPy's own choice still widens the rate in both families — a float64 mantissa or an
  int64 exponent on `frexp` bills `40000` (double the natural `20000`, from the rate, not
  from `nout` a second time). Index reductions additionally constrain `out=`
  by **kind**: NumPy accepts any integer or boolean buffer at any width and refuses every
  float and complex one, and that refusal is decided before the reduction runs, so it
  costs `0`.
- **A call NumPy has no loop for costs `0`.** Operand dtypes NumPy cannot run the
  operation on — `bitwise_and(f32, f32)`, `ldexp(f32, uint64)`, `invert(f32)`,
  `subtract(bool_, bool_)`, and every `reduce`/`accumulate`/`reduceat` of a ufunc with no
  same-dtype loop, such as `ldexp` — are refused above the deduct site rather than billed
  and then raised. The question is settled with `ufunc.resolve_dtypes`, which is the same
  resolution NumPy's own dispatcher performs, so the exception a caller sees is
  byte-identical in type and message to the one the unwrapped call raises. This is the
  same reject-before-billing rule as the `out=` kind constraint above and `svd(k=)`
  below; before it applied to dtypes, `gcd.reduce` on a float array deducted 1584 FLOPs
  for a call that never ran. Where flopscope has a refusal of its own for the same
  operands — a non-numeric dtype, or a complex operand to a complex-illegal op — that one
  still takes precedence, so those keep raising `UnsupportedDtypeError` with flopscope's
  diagnostic rather than NumPy's bare "no loop matched".
- **A non-numeric destination is refused, not priced.** A `str_`/`bytes_`/`datetime64`/
  `timedelta64`/structured-void `out=` — object-free included — describes where the
  result is stored, not the loop that produced it, and its real per-element cost is not
  the fixed unit a flat rate would have to assume. Rather than dropping it from the
  resolution and letting the call bill at the operands' own rate (which would launder a
  wide or width-mismatched destination down to whatever the arithmetic already costs),
  it is kept in the resolution precisely so it is refused —
  `add(i64_arr, i64_arr, out=m8_arr, casting='unsafe')` raises `UnsupportedDtypeError`
  rather than billing `10` or `20`. An `object`-carrying destination is refused the same
  way (see [Dtype and precision](#the-billing-unit-and-rate-table)).
- **Casting/converting bills like `copy`.** `astype` and `asarray` bill `numel(input)`
  at the heavier of source/destination rate — via `heavier_billing_dtype`, not
  `result_type`, which would over-promote a cross-kind cast such as `float32 → int32`
  to float64 — whenever they perform real work: narrowing, widening, a value-changing
  cast (to-bool, float→int, complex→real), a same-dtype `astype` with the default
  `copy=True`, or any `asarray(x, dtype=other)` that actually converts the buffer. The
  one free case is the genuine no-op: `astype(x, dtype, copy=False)` when `dtype`
  already matches `x`'s dtype returns the identical object, and a dtype-free or
  dtype-matching `asarray(x)` performs no conversion — both bill `0` (see
  [Views and metadata](#views-and-metadata-weight-00) for the full rule). `copyto`, the
  in-place write primitive, is priced one unit per element written (per selected
  element under `where=`), same-dtype or not.
- **`real` / `imag` are free.** Extracting a component of a complex value is a
  view / constant-fill, not arithmetic — `flop_cost = 0`.
- **Dtype-neutral bookkeeping declares `dtypes=()`.** Ops that carry no value dtype at
  all (for example `einsum_path`) resolve to rate `1.0`, factor `1.0`, so the rate table
  cannot perturb them. Ops with a *fixed* output dtype but no array operand — window
  functions, `fft.fftfreq`, the random *distribution* samplers — are **not** neutral: they
  declare their output dtype, whatever it is (float64 for the continuous draws, int64 for
  the discrete ones), so their real arithmetic bills at that dtype's rate like everything
  else. This does **not** cover the data-movement random ops (`choice`, `shuffle`,
  `permutation`, `Generator.permuted`), which reorder caller-supplied values of any
  numeric dtype and *are* billed neutrally — their selection work is width-independent.
  See [Random](#random-module-level-generator-randomstate).

### Reduction accumulators

A reduction is billed at the dtype numpy actually accumulates in. The accumulator
resolves in numpy's own order: an explicit `dtype=` argument (positional or keyword) if
given — billed exactly as requested, wider or narrower, because that request *is* the
accumulator numpy runs; else `out`'s dtype if an output array is given — this widens the
accumulator when `out` is wider than the input, but a narrower `out` only casts the
final store, so the loop keeps the input's own width; else the family default — integer
and boolean inputs widen to the platform default integer for `sum`/`prod`/`cumsum`/
`cumprod` (and their `nan`-prefixed variants), and to `float64` for `mean`/`var`/`std`,
never billed below the input's own rate. Both `numpy.trace` and `linalg.trace` widen the
same way `sum` does — a batched diagonal sum is still a sum — and the generic
`ufunc.reduce` / `accumulate` / `reduceat` paths resolve their accumulator through the
identical rule: `np.add.reduce` is the same machinery `sum` runs on, so
`np.add.reduce(int32_arr, dtype=int32)` bills exactly like `sum(int32_arr, dtype=int32)`.

`reduceat`'s base cost follows numpy's own per-segment semantics rather than a flat
per-element charge: for segment `i`, `indices[i] < indices[i+1]` reduces the elements
between them (a length-`L` segment costs `L-1` applications, the same `n-1` convention
`reduce` uses), while a non-monotonic pair (`indices[i] >= indices[i+1]`) is a plain
element copy with no arithmetic — the segment costs 0. The final segment always runs to
the end of the axis. The exact call-level base formula is
`max(lanes × applications_per_lane, lanes × produced_cells_per_lane, 1)`. The first two
entries are the variable-work terms: actual ufunc applications across segments and cells
produced per lane. The cell term accounts for singleton/copy branches that emit without
invoking the ufunc; the final term retains a call-level minimum of one. The resulting
base cost is priced through the standard resolved accumulator dtype rate, complex
factor, and inherited base-ufunc weight. Ordinary long segments remain
arithmetic-dominated; a single whole-axis segment of 1000 elements bills 999
applications. When the reduced axis has length one, the produced-cell term can make
`reduceat(a, [0])` bill more than `reduce(a)`; this is intentional because `reduceat`
retains its per-produced-cell floor even when the segment performs no arithmetic.
`ufunc.outer` is not itself an accumulating reduction, but an explicit
`dtype=` on it resolves through the same request-is-the-loop rule: the dtype names the
loop numpy actually runs, not a discount, so
`np.multiply.outer(int32_arr, int32_arr, dtype=float64)` bills the float64 rate over the
full outer-product grid — double the default int32-loop total.

Worked examples (production rates, 1000-element input, `sum`):

- `sum(int32_data)` — implicit int64 accumulator → 999 × 2.0 = **1998**.
- `sum(int32_data, dtype=int32)` — explicit 32-bit accumulation is real 32-bit work →
  999 × 1.0 = **999**.
- `sum(float32_data, None, float64)` — a positional wide accumulator bills exactly like
  the keyword spelling `sum(float32_data, dtype=float64)` → 999 × 2.0 = **1998**.
- `sum(float64_data, dtype=float32)` — numpy genuinely accumulates in float32 (operands
  cast to float32 on read), so the bill follows that narrower loop →
  999 × 1.0 = **999**.

`trace` follows the identical rule at its own, much smaller element count:
`linalg.trace` on an `8×8` int32 matrix bills its 8-element diagonal sum at the int64
rate (**16**); the same shape in float32 stays at float32 (**8** — floats never widen).
The explicit accumulator carries through the top-level wrapper too:
`trace(int32_matrix, dtype=int32)` bills **8**, at int32 — the same rate as the input,
because an explicit accumulator request is honored exactly.

Two spellings resolve differently because they compute differently. Pointwise `dtype=`
computes narrow: operands are cast to the requested dtype before the elementwise loop
runs, so the bill is narrow too. Pointwise `out=` computes at the operand-promoted width
and only casts the result on the way into `out`, so the bill stays wide. Reductions
follow the same split one level up: `dtype=` sets the accumulator in both directions —
narrower or wider, exactly as requested; `out=` widens the accumulator when `out` is
wider than the input, while a narrower `out` only casts the final store, so it never
bills below the input's own rate.

### Ops that compute wider than their inputs

Outside the accumulator machinery above, several numpy kernels simply run their
arithmetic at a wider dtype than the input's own — the bill follows where the
computation actually happens, not the input's label:

| family | integer / bool inputs compute in | float inputs |
|---|---|---|
| unary float-only ufuncs (`exp`, `sin`, `sqrt`, …) | the same-size float (`int8`/`uint8`/`bool` → `float16`; `int16`/`uint16` → `float32`; `int32`/`int64`/`uint32`/`uint64` → `float64`) — `angle` is the one exception, flooring `bool` alone at `float64` (see below) | unchanged |
| binary float-only ufuncs except division (`arctan2`, `hypot`, `logaddexp`, …) | NumPy-selected float loop (`bool`/`int8`/`uint8` → `float16`; `int16`/`uint16` → `float32`; wider integers → `float64`) | unchanged — this replaces the old blanket-float64 rule for narrow integers |
| `divide` / `true_divide` | integer pairs resolve to `float64` | unchanged |
| `float_power` | NumPy's selected loop has a `float64` real minimum | `float64` real minimum — unchanged in effect |
| FFT family | `complex128` (no size-mapping — even `int8` runs the `complex128` path) | `float16`/`float32` → `complex64`; `float64` → `complex128` |
| LAPACK-backed `linalg.*` | `float64` | unchanged — single-precision drivers stay single |
| mean-shaped composites (`average`, `median`, `gradient`, `percentile`, `nanmedian`, `nanpercentile`, …) | `float64`, regardless of the integer's own width (no size-mapping, unlike the unary-ufunc row above) | unchanged |
| always-float64 composites (`polyfit`, `polyint`, …) | `float64` | `float64` — forced even from `float32` |

The binary ufunc rows above report NumPy loop-table results, not operation-name floors
imposed by flopscope. In particular, `float_power`'s narrowest real and complex
signatures are `dd->d` and `DD->D`, respectively, so `complex64` promotes to
`complex128` — unchanged in effect.

`exp` prices its size-mapped loop directly: on a 1000-element input, `int8` and `int16`
both still bill the `1.0` rate (**16000** — float16 and float32 share the baseline rate),
while `int32`/`int64` bill the `2.0` float64 rate (**32000**).

`angle` follows that same size-mapped loop for every actual integer/unsigned width, but
`bool` is a documented exception: on a 1000-element input, `angle(bool_data)` bills the
`float64` rate (**32000**) while `angle(int8_data)` and `sin(bool_data)` both bill the
`float16` rate (**16000**).

For a worked binary example, `hypot(int8, int8)` selects a float16 loop and
`hypot(int16, int16)` selects a float32 loop; both bill the baseline dtype rate `1.0`.
By contrast, `divide(int16, int16)` resolves to and bills float64 at rate `2.0`. Billing
inspects the selected **complete** loop signature and then floors it by the promoted raw
input rate. That prevents a narrow output loop — or any other narrow slot — from
discounting wider inputs; it does not regress to an output-slot-only rule.

`fft.fft` of a length-1024 (`N`) signal has a `5N·log₂N` = 51200 real-FLOP count before
the dtype rate; an int32 signal bills that count at the complex128 rate (**102400**),
while a float32 signal keeps the complex64 rate and bills the raw **51200** unscaled.

A single non-inexact operand forces `linalg.*`'s promotion, regardless of how wide the
*other* operand is: an `8×8` `linalg.solve(int32_matrix, int32_vector)` bills float64
(**938**), and so does mixing kinds — `solve(bool_matrix, float32_vector)` on the same
shape bills the identical **938**, exactly as if the matrix were genuinely float64. The
all-float32 system keeps the single-precision driver and bills half that (**469**).

Integer contractions keep their input's own rate: `matmul`, `dot`, `einsum`, and
`matrix_power` with a non-negative exponent genuinely run integer arithmetic
(`matmul(int32, int32)` stays int32). `matrix_power` with a *negative* exponent is the
exception — numpy inverts the matrix first, which routes through the same LAPACK
float64 promotion as `linalg.inv`: a `4×4` `matrix_power(int32_matrix, -1)` bills
float64 (**224**) while `matrix_power(int32_matrix, 2)` stays int32 (**112**), and
`matrix_power(float32_matrix, -1)` keeps the single-precision driver (**112**) even
while inverting.

Not every widening composite forces float64 unconditionally, and not every one lands on
float64 at all — the billed dtype is always whatever `resolved_dtype` records for that
specific call, not one fixed target per op family. `roots` builds a companion matrix and
finds its eigenvalues, so it inherits `linalg.eig`'s dtype rule directly: an integer
input computes float64, but a float32 input keeps its own single-precision driver.
`poly` (rebuilding coefficients from roots) runs a different algorithm — iterative
convolution, not eigendecomposition — but resolves its dtype the same preserve-if-inexact
way. Neither belongs in the always-float64 row above. `vander` promotes through
`numpy.promote_types(x.dtype, int)`, the same kind of rule a reduction accumulator uses:
a narrow integer lands on the platform `int64` (`vander(int32_x)` bills its int64 rate,
not float64), while float and complex inputs still land on `float64` / `complex128`.

The generic `ufunc.outer` and `ufunc.at` methods apply this same per-element promotion
to their base ufunc, so there is no method-shaped way around it:
`np.hypot.outer(int32_arr, int32_arr)` bills the float64 rate over its full
outer-product grid, and `np.exp.at(int32_arr, ...)` bills the float64 rate over the
indices it touches — identical to calling `hypot` or `exp` directly.

Zero-work contractions charge **0** regardless of dtype. `einsum('ij->ji', z)` is a pure
transpose — no multiply-adds happen — and a `matmul` with an empty operand has no terms
to sum, so both bill 0 even when `z` is complex128; a same-shape *non-empty* complex128
matmul still bills its exact contraction total (see [Contraction is billed
exactly](#complex-arithmetic-from-first-principles)).

Zero *arithmetic* is not zero *work* once a destination is named. `einsum` is the one
contraction that does not forward `out=` to NumPy — NumPy writes its own buffer and the
wrapper then copies that buffer into the caller's — so `einsum(subs, …, out=dest)` bills
its contraction total **plus** a `copyto` of `numel(dest)`, at the same price
`fnp.copyto(dest, result)` charges for the identical write. The step-2 rule applies to it
like any other materializing copy; without the charge, `einsum('ij->ji', z, out=dest)`
would be a full transpose of any size for nothing. The viewing form is unaffected: with no
destination NumPy returns a view, nothing is written, and 0 stays correct.

**Guaranteed coverage.** `tests/test_compute_dtype_conformance.py` probes every charged
registry op with a discriminating int32 input (or records why it is exempt) and asserts
the billed rate is at least the rate of both the NEP-50-promoted input dtype and numpy's
actual result dtype — an op that quietly starts computing in a wider dtype than it bills
fails the build instead of shipping an undercount. The sweep covers charged *registry*
ops; the dynamic ufunc-method surface (`.outer`, `.reduceat`, `.at`, `.reduce`,
`.accumulate`, which bill under per-call op names like `hypot.outer` rather than registry
keys) is out of its reach and is locked by targeted tests in `tests/test_dtype_cost.py`
instead. There is a second, narrower carve-out inside the sweep: the index-output ops
(`argmax`/`argmin`, `argsort`, `nonzero`/`flatnonzero`/`argwhere`, `searchsorted`,
`count_nonzero`, `digitize`, `lexsort`, the `unique_*` tuple forms, and kin) are exempt
from the **result**-dtype half of the assertion, because they return `int64` index arrays
whatever the operand's width — their result dtype says nothing about the arithmetic they
did. They are held to an **operand**-dtype floor instead, probed at `float64` so the
assertion can actually be violated: the comparison work is done at the operand's width
and must be billed there. A second pass narrows the same probes to `float32` and requires
the billed rate to follow the operand down, which is what separates billing the operand
from billing the `int64` index result — those rate the same, so a floor alone cannot tell
them apart. `searchsorted` and `digitize` are held to over-resolving instead: their bin
edges and needle promote the call, so they bill above the operand, which over-bills
rather than under-bills. `tests/test_binary_ufunc_spelling_billing.py` pins
complete-signature billing, narrow binary loops, the promoted-input floor,
direct/`outer` parity, explicit `dtype=`/`signature=` constraints plus wider-`out=`
composition, descriptor-safe operand metadata, and store-only complex billing for the
single-output binary paths.

Reproduce any of these yourself: run the call inside a `BudgetContext` and read
`op_log[-1].resolved_dtype` alongside `flops_used`.

### Worked examples

Billed FLOPs for an `N = 100` elementwise call under production rates (`multiply`: weight
1, factor 6; `add`: weight 1, factor 2; `sin`: weight 16, factor 3.4):

| op | float32 | float64 | complex64 | complex128 |
|---|---|---|---|---|
| `multiply` | 100 | 200 | 600 | 1200 |
| `add` | 100 | 200 | 200 | 400 |
| `sin` | 1600 | 3200 | 5440 | 10880 |

Reading one cell: `multiply` on `complex128` = `int(100 × 2.0 × 6.0 × 1.0) = 1200` —
`flop_cost` 100, float64-component rate 2.0, complex-multiply factor 6.0, baseline weight
1.0. `add` on `complex64` = `int(100 × 1.0 × 2.0 × 1.0) = 200`: a complex add's factor is
only 2, so it costs far less than the complex multiply. `sin` carries its weight-16
transcendental tier and a 3.4 complex factor, so `complex128` =
`int(100 × 2.0 × 3.4 × 16) = 10880`.

Contraction is exact rather than a flat factor: a `matmul` of two `(8, 8)` matrices bills
`960` real (`2·8³ − 8²`), `1920` at float64, and `3968` at complex64 — the engine's exact
`6·512 + 2·448` (512 scalar multiplies, 448 adds), a ratio of `≈ 4.13`, not `4`. At
complex128 it is `7936` (that exact total × the 2.0 rate). These values are pinned in
`tests/test_dtype_cost.py`.

### On packing

Packing is **not banned and not judged** — it is priced so it does not pay:

- **Complex packing loses.** Two honest real multiplies of 100-element float32 arrays bill
  `200`; folding the two payloads into one complex64 multiply bills `600` (factor 6) — a
  `3×` loss. At matmul scale the packed complex matmul bills `3968` against the `1920` of
  the two honest real matmuls — roughly `2×` worse, and that is before any pack/unpack
  overhead.
- **Width packing breaks even at best.** Two float32 multiplies bill `2 × 100 = 200`; one
  float64 multiply carrying both payloads bills `100 × 2.0 = 200` — identical, and that is
  *before* the arithmetic to pack the lanes in and unpack the results out, which is itself
  charged. So 32-into-64 packing is break-even-or-losing.
- **Sub-32-bit lane tricks** (packing several int8 or int16 payloads into a wider lane) can
  in principle recover a small constant-factor advantage, since everything at or below
  32-bit shares the `1.0` rate. That gain is bounded and small, is considered in-bounds,
  and the rate table deliberately does not chase it.

Billing follows the loop numpy actually runs, in both directions: an explicit narrow
`dtype=` — pointwise, or a reduction's accumulator — bills narrow because the arithmetic
genuinely happens at that width, trading precision for a cheaper bill exactly as
requested. `out=` alone never narrows a loop, so it cannot be used to buy a narrow bill
for wide compute. The width-rate and complex-factor pricing above is what makes packing
a loss or a break-even; there is no separate dtype-request rule needed to close it.

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
| **Weight-tier policy** | every active weight ∈ `{0, 1, 4, 16}`; arithmetic ops are 0 or 1; **no algorithm constant in a weight** | `test_weight_tier_policy.py` |
| **No substitution arbitrage** | a bit-identical alias cannot bill cheaper than its canonical (e.g. `acos` *is* `arccos` — the 16× ufunc-alias fix); equivalent contractions (`dot`/`inner`/`matmul`/`einsum`) share one cost engine | `test_ufunc_alias_parity.py`, `test_random_weight_aliasing.py`; the shared einsum engine ([§Contraction](#contraction-einsum-family)) |
| **No unpriceable or mispriceable dtype** | a non-numeric dtype (`dtype.kind` outside the allowlist `"biufc"` — object, string, bytes, structured/void, datetime64, timedelta64) is refused before any charge — as an operand, an explicit `dtype=`, or an `out=` destination — because it either has unbounded per-element cost (object) or a real per-element cost no flat rate captures (the rest); a dtype that is still zero-itemsize once NumPy materialises it is the one exception, since it carries no data either way — `'U0'`/`'S0'` are not, because NumPy promotes them to `'U1'`/`'S1'` on allocation | `tests/test_object_dtype_ban.py` |
| **No cheap in-op path** | top-k `svd(k=)` cannot yield a *full* decomposition below full price (the `min(4mnk, economy)` cap + `k ≥ min → full` guard); invalid `k` (`< 1` or `> min(m, n)`) is rejected before any billing | `test_svd_topk_cost.py` (cap / guard / monotonicity); `test_linalg.py` (invalid-`k` `ValueError`) |
| **No free symmetry tag over a bare top-level constructor call** | a *bare, top-level* `SymmetricTensor(data, symmetry=…)` call cannot mint a tag over data that doesn't actually satisfy the claimed group: it now validates the claim through the same check `as_symmetric(data, symmetry=…)` already used, raising `SymmetryError` on a mismatch and charging `k·(7n − 1)` (`k` = non-identity generators, `n = data.size`) for a genuine one — and symmetry a flopscope op derives *internally* (e.g. `exp` propagating the tag of an already-validated operand, or a slice/transpose view) is exempt from re-paying, since it never reaches an unvalidated, caller-supplied claim. **Two narrower, in-process-only routes remain open and are NOT closed by this**: calling `flopscope._symmetry_utils.wrap_with_symmetry` directly (it only checks that the group's axes fit `ndim`, never buffer contents — trusted by this constructor for its ~30 legitimate internal uses in `_array_ops.py`), and constructing inside a participant callback a counted host op invokes (`fnp.apply_along_axis`, `fnp.piecewise`, …), which inherits the host op's trust. Neither is remotely reachable — `wrap_with_symmetry` is outside the server's op REGISTRY and unexported, and callback ops raise `RemoteCallbackError` on the server backend — so both are pinned as `strict=True` `xfail` regression tests rather than fixed, to avoid repricing the ~34 internal call sites that legitimately rely on the same trust | `tests/test_symmetric_tensor_new_validation.py` (incl. the two pinned `xfail`s); `tests/test_symmetric_cost.py` (`k·(7n−1)` rate) |
| **Free-tier discipline** | weight 0 is limited to views/metadata, untouched (zero-page or uninitialized) allocation, and the narrow `astype`/`asarray` no-op (`copy=False` with an already-matching dtype; a dtype-free or dtype-matching `asarray`) — every other cast or copy, including a same-dtype `astype(copy=True)`, bills `numel` like `copy`. Any **metered** op that writes a new buffer — copied, replicated, constant-filled, gathered, or scattered — carries weight ≥ 1 (ndarray methods inherited from numpy are outside the meter by design — see [§The meter boundary](#the-meter-boundary)). Every value-test is charged wherever it hides: `a.nonzero()` (method), `where(1-arg)`, `argwhere`, `flatnonzero`, `count_nonzero` | `test_weight_tier_policy.py`; `test_data_movement_free_tier.py` (free-labels consistency guard) |
| **No free-gather discount** | a computed-index gather (`take`, `take_along_axis`, `choose`) is metered at the access tier (weight 4.0) like any other non-sequential read, so precomputing a look-up table and then gathering from it no longer buys a categorical discount; only genuine view-indexing (a static/basic index, `arr[i]`) stays free | `test_data_movement_free_tier.py`; [§Copy and gather](#copy-and-gather) |
| **Complex packing non-profitable** | folding two real payloads into one complex op's real/imag lanes bills the op's true complex structure (`multiply` factor 6, matmul exact `≈4.13×`), so the pack costs more than the honest real work it replaces | `tests/test_dtype_cost.py` (packing tests) |
| **Width packing break-even-or-losing** | a 64-bit op bills `2×` a 32-bit op (`dtype_rate`), so packing two 32-bit payloads into one 64-bit lane is break-even before pack/unpack overhead; billing follows the loop numpy actually runs, so an explicit narrow `dtype=` only bills narrow when the compute is genuinely that narrow, and `out=` alone never shrinks the loop | `tests/test_dtype_cost.py` (width-packing tests); `tests/test_unary_out_never_narrows.py` (the unary float-loop family's whole `out=` grid) |
| **End-to-end billing** | production billing is pinned per weight tier `{0,1,4,16}` (catches a silent weight regression); the retired `8.0` tier is documented as retired rather than silently dropped | `test_production_weight_billing.py` |
| **No free path-search wall time** | `opt_einsum.contract_path` runs as pure Python inside the counted wrapper, never through `_call_user_code`, so its wall time books to the unbilled `flopscope_overhead_time_s` bucket rather than to backend FLOPs or residual. Once operand count `k ≥ _LARGE_K_THRESHOLD` (8), `optimize=` is resolved through an **allowlist**, not a denylist: only strategies known to run in linear time (`'greedy'`, `'eager'`, `'opportunistic'`, `'random-greedy'`, `'random-greedy-128'`), `False` (no search), and an explicit user-supplied path (a `list`/`tuple` — the order is already decided, so there is no search to bound) pass through verbatim; everything else — every current `opt_einsum` alias for an exhaustive or superlinear search (`'auto'`, `'auto-hq'`, `'optimal'`, `'branch-all'`, `'branch-2'`, `'branch-1'`, `'dp'`, `'dynamic-programming'`), `True`, `None`, a custom `PathOptimizer`, and any future `opt_einsum` string this list doesn't recognize — downgrades to `'greedy'`. The allowlist replaced an earlier denylist that missed the `'dynamic-programming'` alias (the exact same function object as `'dp'`, registered under a second name) plus `'auto-hq'`/`'branch-1'`; a live repro parked 45.75s of free wall time via `optimize='dynamic-programming'` at k=20 before the allowlist closed it | `_resolve_optimize_for_k` in `_einsum.py`; `tests/test_einsum_path_search_billing.py`, `tests/accumulation/test_path_aware_cost.py::test_large_k_auto_fallback_to_greedy` and `::test_large_k_covers_every_opt_einsum_path_alias` (the latter iterates `opt_einsum.paths._PATH_OPTIONS` directly, so a future `opt_einsum` release adding a new exhaustive-search alias fails CI instead of silently reopening the hole) |

An auditor can read this table top-to-bottom and, for each claim, open the named test
to see exactly what guarantees it. The first two rows are the load-bearing ones: an exact
`flop_cost` defeats under-count, and the weight-tier policy (no constant in a weight)
defeats the family of arbitrage exploits where a high-constant op is re-tiered cheaply.

### The meter boundary

Every invariant above scopes to **metered** ops — the `fnp.*` surface that routes
through flopscope. `FlopscopeArray` is an `np.ndarray` subclass that intercepts the
ufunc/operator protocol, so arithmetic is counted; methods it inherits unchanged from
numpy are not. `.tobytes()`, `.tolist()`, `.data` and `__array__` therefore write a new
buffer for 0 billed FLOPs, and `fnp.frombuffer(a.tobytes())` reproduces `fnp.copy(a)`
without the `numel` charge.

This is deliberate, and it is contained by **residual billing rather than by weight**.
Whestbench scores effective compute as `C = F + λ·R`, where `R` is participant wall time
outside metered dispatch and `λ = 1e11` FLOP/s. Anything materialised in participant
Python is paid for in `R`, and λ sits far above the rate at which data movement converts
wall time into saved FLOPs:

| achievable memory bandwidth | billed FLOP/s saved | residual FLOP/s charged | net |
|---|---|---|---|
| 1 GB/s | 2.50e8 | 1.00e11 | 400× loss |
| 5 GB/s | 1.25e9 | 1.00e11 | 80× loss |
| 10 GB/s | 2.50e9 | 1.00e11 | 40× loss |
| 20 GB/s | 5.00e9 | 1.00e11 | 20× loss |

So the free-copy channel is real but strictly losing: at 1 billed FLOP per float32
element, substituting `fnp.frombuffer(a.tobytes() + b.tobytes())` for
`fnp.concatenate([a, b])` saves 2e6 FLOPs and costs ~1.8e8 residual FLOP-equivalents —
about 92× worse. Overlapping the copy with an in-flight metered op does not rescue it:
`tobytes` holds the GIL, and breaking even requires hiding ~99% of the copy.

Metering these methods would also be actively harmful, not merely conservative.
`tobytes()` is on flopscope's own wire path (`_request_handler.py` serialises arrays with
`arr.tobytes()`), so billing it would make the transport bill itself on every array
transfer, and `.tolist()` is the documented way to read outputs back at the end of
`predict()`. The boundary is drawn where it is so that getting data *out* is never a
charged operation.

The place the cost model is genuinely sensitive is the narrower one where work happens
*inside* the meter for a mispriced amount — which is what every invariant in the table
above is defending.

**Timing attribution inside a metered call** has three buckets, not two:
`wall_time_s == flopscope_backend_time_s + flopscope_overhead_time_s +
residual_wall_time_s`, and `residual_wall_time_s` is defined as `wall − backend −
overhead` — so `flopscope_overhead_time_s` (flopscope's own dispatch code: wrapper
preambles, bookkeeping, path search) sits outside *both* billed FLOPs and the
λ-rated residual `R`. It is meant to be small and bounded (dispatch bookkeeping),
not a channel a caller can inflate; see the **No free path-search wall time**
invariant above for the one case where an unbounded pure-Python computation
(`opt_einsum.contract_path`) used to run inside it.

**NEP 18 dispatch coverage.** A second, narrower boundary gap sits in-process only (the
graded sandbox has no local `numpy` for a participant to import and mix in). `np.<func>(a,
b)` where `a` is a `FlopscopeArray` and `b` a plain `ndarray` falls through to NumPy's own
`__array_function__` protocol whenever `func` is not in
`FlopscopeArray._ARRAY_FUNCTION_DISPATCH`, and NumPy runs its default implementation
directly on the raw buffers, so that work is not counted. The same call with every argument
a `FlopscopeArray` fails **closed** instead, with a `TypeError` ("no implementation found"). The following are
explicitly bound so a mixed call routes through the metered implementation the same as an
all-`FlopscopeArray` call already does:

| `np.<func>` | routes to |
|---|---|
| `copyto` | `fnp.copyto` |
| `putmask` | `fnp.putmask` |
| `place` | `fnp.place` |
| `select` | `fnp.select` |
| `cov` | `fnp.cov` |
| `corrcoef` | `fnp.corrcoef` |
| `linalg.tensorsolve` | `fnp.linalg.tensorsolve` |

`tests/test_array_function_dispatch_coverage.py` pins this floor — it forces the lazy
dispatch-map build via `FlopscopeArray._get_array_function_dispatch()` and asserts each of
the seven above is present. Coverage is a floor, not a completeness guarantee; the bound set is the one the test pins.

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

**Family rule**: `flop_cost = numel(output)` for a single-output op. A **multi-output
ufunc** (`divmod`, `frexp`, `modf` — the only NumPy ufuncs with `nout > 1` flopscope
counts) is priced as `nout` independent applications of the reference algorithm, so its
`flop_cost = nout × numel(output)`: `divmod` bills what running `floor_divide` and `mod`
separately would bill, combined; `modf`/`frexp` bill twice a one-output unary of the same
shape. The justification is **not** the write-metering principle above — arithmetic
weight, not buffer traffic, dominates these ops, so that clause does not govern them.
The justification is the model's standing rule that pricing tracks a **standardized
reference algorithm**, not whatever compute-sharing a specific backend implementation
happens to do — stated earlier as "we bill the textbook standard-algorithm cost, not
literal BLAS/LAPACK" (see [Billing model & design
principles](#billing-model--design-principles)) and already the basis for the FMA=2
convention and the `trapz`/`nanstd`/`linspace` fixes. A backend that computes several
outputs more cheaply than the sum of independent calls does not earn its caller a
discount; without this rule, an efficient combined op becomes a way to obtain two
deliverables for the price of one. Measured backend timing makes this explicit rather
than contradicting it: at N=500,000, `np.divmod` runs in ~0.98x a single
`np.floor_divide` (~0.49x `floor_divide`+`mod` combined) — NumPy's `divmod` genuinely
shares the division and derives the remainder almost for free — yet flopscope prices it
at `2×` `floor_divide` anyway, by the reference-algorithm rule above; the bill is a
deliberate ceiling on what a participant could otherwise obtain via an efficient
combined op, not a measurement of `divmod`'s own backend cost. `out=` does not multiply
this a second time — the `nout` scaling lives in `flop_cost` (the cell-count axis),
which is independent of the `out=` destination-dtype-rate axis described in [Which
dtype prices a call](#which-dtype-prices-a-call); supplying `out=` widens the rate
exactly as it would for any other op, never the cell count.

**Baseline tier (weight 1.0)**: arithmetic (+, −, ×, ÷, √), rounding
(ceil, floor, trunc, rint, around/round), sign/abs, logical (not, and, or,
xor), bitwise (and, or, xor, invert, left_shift, right_shift), comparisons
(equal, not_equal, greater, less, greater_equal, less_equal), copies
(positive, negative, conj/conjugate, fabs, spacing,
nan_to_num, isclose, isneginf, isposinf, deg2rad/degrees, rad2deg/radians,
ldexp, nextafter, copysign, heaviside, signbit), and their NumPy aliases. `modf` and
`frexp` are also weight 1.0, but — being `nout=2` — bill `2 × numel(output)`, not
`numel(output)`; see the family rule above.
(`real`/`imag` are **not** here — component extraction is free; see
[View / free](#view--free-weight-00).)

**Transcendental tier (weight 16.0)**: exp, exp2, expm1, log, log2, log10,
log1p, cbrt, sin, cos, tan, sinh, cosh, tanh, arcsin, arccos, arctan,
arcsinh, arccosh, arctanh, sinc, i0, power, angle, and their NumPy 2.x
aliases (asin, acos, atan, asinh, acosh, atanh, atan2, pow).

**Moderate binary tier (weight 16.0)**: arctan2/atan2, hypot, logaddexp,
logaddexp2, floor_divide, mod/remainder, fmod, float_power. NumPy's selected
`float_power` loop has no real signature narrower than `dd->d` and no complex signature
narrower than `DD->D`, so float32 operands bill as float64 and complex64 operands bill
as complex128. This comes from NumPy's loop table, not a hand-maintained family mapping.
`divmod` sits in this tier too, at `floor_divide`'s weight (see [NumPy 2.x ufunc
aliases](#numpy-2x-ufunc-aliases)) — but being `nout=2`, its `flop_cost` is `2 ×
numel(output)`, so a bare `divmod` call bills exactly what running `floor_divide` and
`mod` separately would bill, combined.

**Iterative integer tier (weight 16.0)**: `gcd`, `lcm`. These sit at the weight-16 tier
without being transcendental functions, and the reason is the same one the tier exists
for: the per-element kernel is a bounded multi-step loop, not a single instruction.
NumPy computes `gcd` by the Euclidean algorithm — a remainder loop whose trip count
grows with the operand magnitude — and `lcm` runs that same loop and then a multiply and
a divide. Pricing them at the arithmetic tier would sell a genuinely iterative
per-element computation at the price of an `add`. Both are integer-only
(`complex_factor: illegal` — NumPy raises `TypeError` on complex input), so the complex
table below never applies to them; an `int64` operand pair bills
`numel × 16 × 2` (weight 16.0 at the int64 rate 2.0).

**Basis**: DECLARED per-element FMA=2 convention and empirical calibration.

**Complex dtypes**: each op bills its per-op `complex_factor` from the class table in
[Dtype and precision](#complex-arithmetic-from-first-principles) — `add`/comparisons 2,
`multiply` 6, `divide` 11, `abs` 4, transcendentals per op (`sin` 3.4, `log` 2.25); `conj`
and `angle` price the same component floor (factor 2); rounding, bitwise, and
`mod`-family ops are complex-illegal and raise.

Source: `src/flopscope/_pointwise.py`.

---

### Reduction

**Family rule**: `flop_cost = numel(input) − numel(output)` (orbit-mapping
model; one add or compare per element consumed by the reduction).

Ops that do more than one accumulation pass carry the extra passes in
`flop_cost` (never in the weight column): the variance family makes four
passes (mean-sum, centre, square, variance-sum), `ptp` makes two (max + min)
plus the per-output subtract, and `mean`/`average` add the per-output divide.
A `nan*` variant adds a further `numel(input)` on top of its plain sibling's
formula, for the `isnan` test pass the `nan*` form runs before reducing that
its plain sibling does not — **but only where NumPy actually runs that pass**.
Nine of the factory-built `nan*` reductions go through NumPy's
`_replace_nan`, which returns no mask for a non-inexact dtype, so an
**integer or bool input runs no `isnan` pass and bills exactly like its plain
sibling**; only float/complex input carries the surcharge. `nanmedian`,
`nanpercentile`, and `nanquantile` instead go through `_remove_nan_1d`, which
calls `np.isnan` whatever the dtype, so those three carry it for every dtype.
**`nanmax` and `nanmin` carry it at no dtype at all**: for any plain
non-object ndarray — which is every operand flopscope hands them — NumPy
takes a fast path that reduces with `fmax`/`fmin` and then tests the *reduced
output*, never the input, so there is no input-sized pass to charge and both
bill exactly like `max`/`min`. Like every other
pass in these ops, the surcharge is **orbit-mapped**: on a symmetric operand it
counts unique elements, not dense `numel`. The pass is folded into the same
`flop_cost` as the rest of the reduction, so it is charged as one additional
pass over the input priced at the op's own rate/factor — not a separate flat
rate: it inherits the host op's complex factor (e.g. 6.0 on `nanprod`) and
its resolved accumulator dtype (e.g. the widened int64 rate on an int32
input), the same way the variance family and `ptp` already blend multiple
passes under one op-level factor.

| Op | flop_cost | weight | basis |
|---|---|---|---|
| `sum`, `prod`, `max`, `min`, `any`, `all`, `nansum`, `nanmax`, `nanmin`, `nanprod` | numel(input) − numel(output) (+numel(input) for `nansum`/`nanprod` on float/complex input only; `nanmax`/`nanmin` add nothing at any dtype — see the family rule) | 1.0 | DECLARED reduction skeleton (one add or compare per consumed element) |
| `cumsum`, `cumprod`, `nancumsum`, `nancumprod`, `cumulative_sum`, `cumulative_prod` | numel(input) − num_output_slices (= n−1 for a full 1-D scan; product of non-reduced dims otherwise) (+numel(input) for the `nan*` forms on float/complex input only) | 1.0 | DECLARED: scan accumulation; output shape = input shape so the generic `numel(in)−numel(out)` formula evaluates to 0 — these use the correct per-slice count instead |
| `mean`, `average` (unweighted) | numel(input) | 1.0 | DERIVED: reduction (numel−M) + M divides |
| `average(weights=)` | `3·numel − M`, M = num output slices (1 for full reduction) | 1.0 | DERIVED: a·w multiply pass (numel) + a·w sum (numel−M) + weight sum (numel−M) + M divides |
| `std`, `var`, `nanstd`, `nanvar` | ≈ 4 × numel(input) (std: + M sqrt); `nanstd`/`nanvar`: + numel(input) isnan pass on float/complex input only ⇒ ≈ 5 × numel(input) there, ≈ 4 × on integer/bool | 1.0 | DERIVED four-pass: mean-sum, centre, square, var-sum (exact: 2·numel + 2·(numel−M) + 2M) |
| `argmax`, `argmin` | numel(input) − num_output_slices (= n−1 for full 1-D; reduction_cost model) | 1.0 | DECLARED scan: same orbit model as reduction family |
| `median`, `nanmedian` | axis length per output slice (+numel(input) for `nanmedian`, every dtype) | 1.0 | DECLARED; partition (introselect) per output |
| `percentile`, `nanpercentile`, `quantile`, `nanquantile` | per output slice, piecewise in axis length `n` and `q.size` `k` — unweighted: `n·min(k, 1 + 4⌈log₂ min(k,n)⌉) + 4k`; with `weights=`: `4n⌈log₂ n⌉ + 3n + k(⌈log₂ n⌉ + 4)` (+numel(input) for the `nan*` forms, every dtype) | 1.0 | DECLARED; unweighted bills the cheaper of `k` partition passes or one shared sort-parity pass (a dense `q` returns the sorted input); the weighted branch sorts internally, so it is priced at sort parity plus a per-`q` lookup |
| `ptp` | 2 × numel(input) − numel(output) | 1.0 | DERIVED: max pass + min pass + M subtracts (2·(numel−M)+M) |
| `count_nonzero` | numel(input) | 1.0 | DECLARED comparison scan (every element tested regardless of axis) |
| `nanmean` | numel(input) (+numel(input) for the isnan pass on float/complex input only ⇒ 2 × numel(input) there, numel(input) on integer/bool) | 1.0 | DERIVED: reduction (numel−M) + M divides, + the `nan*` isnan pass where NumPy runs it |

**Complex dtypes**: sum-type reductions (`sum`, `mean`, `cumsum`, `nansum`, …) bill factor
2 (complex add); product reductions (`prod`, `cumprod`, `nanprod`) factor 6 (complex
multiply); the variance family (`var`, `std`, `nanvar`, `nanstd`) factor 2.5.

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
above multiply the closed form by the batch size. The path search that picks
this binary order runs unbilled (see the **No free path-search wall time**
invariant in [§Non-exploitability](#non-exploitability)), so at `k ≥ 8`
operands only an allowlisted, linear-time `optimize=` choice runs verbatim —
every other choice, including every `opt_einsum` alias for an exhaustive or
superlinear search, is downgraded to `'greedy'` before the search runs.

**Compound linalg** ops are *chains* of matmuls, billed as the sum of their steps
through the `matmul_cost(m, k, n)` helper — which itself delegates to
`einsum_cost('ij,jk->ik', …)`, so each step equals a 2-D matmul by construction
(no duplicated `2mkn − mn` constant to drift). `linalg.pinv` and `linalg.lstsq`
build on the same helper.

| Op | flop_cost | basis |
|---|---|---|
| `linalg.matrix_power` | `(⌊log₂ k⌋ + popcount(k) − 1) × matmul_cost(n, n, n)` | repeated squaring |
| `linalg.multi_dot` | sum of optimal-chain matmul costs, each step `2mkn − mn`; the two-array form with a 0-d operand instead bills `numel(other)`, the scalar multiply NumPy performs there | optimal chain order |

**Complex dtypes**: the contraction family is billed **exactly**, not with a flat factor.
The engine expands each call's `flop_cost` into `6·(multiplies) + 2·(adds)` from the same
accumulation decomposition (see
[the exact-contraction rule](#complex-arithmetic-from-first-principles)). For a length-`K`
dot product the complex/real ratio is `(8K − 2)/(2K − 1)` (6 at `K=1`, `≈4.13` at `K=8`),
so `einsum('i,i->i', z, z)` bills exactly like a complex `multiply`.

All contraction ops use **weight 1.0** — the shape formulas already carry the
full FMA=2 cost. Source: `_pointwise.py` (op wrappers), `_einsum.py`
(`_resolve_cost_and_output_symmetry`), `_flops.py` (`einsum_cost`,
`matmul_cost`), `_accumulation/` (accumulation model).

---

### Generator (linspace, arange, and kin)

| Op | flop_cost | basis | source |
|---|---|---|---|
| `arange` | `2 × numel(output)` | DERIVED: `start + i×step` per element = 1 mul + 1 add (FMA=2) | `_array_ops.py`; numpy arraytypes.c.src |
| `linspace` | `2 × numel(output)` (handles broadcast start/stop and `retstep=True`) | DERIVED: same affine model as arange | `_array_ops.py` |
| `geomspace` | `numel(output)` (weight **16.0**); output is float64 by default → billed `32 × numel(output)` (`16 ×` only with a 32-bit `dtype=`) | DERIVED: flop_cost = numel(output); transcendental weight 16.0 (log + exp path) | `_array_ops.py` |
| `logspace` | `numel(output)` (weight **16.0**); output is float64 by default → billed `32 × numel(output)` (`16 ×` only with a 32-bit `dtype=`) | DERIVED: same transcendental path as geomspace | `_array_ops.py` |
| `zeros`, `zeros_like`, `empty`, `empty_like` | 0 (allocation; the zero-page / uninitialized default — nothing is written) | DECLARED free: untouched allocation | `_array_ops.py` |
| `ones`, `ones_like`, `full`, `full_like` | `numel(output)` | DECLARED: every cell is written a real, non-zero value | `_array_ops.py` |
| `eye`, `identity` | diagonal length written (`max(0, min(N, M−k))` for `k≥0`, `max(0, min(N+k, M))` for `k<0`) | DECLARED: only the diagonal 1s are written; the off-diagonal zero background is free, same as `zeros` | `_array_ops.py` |
| `tri` | `numel(output)` | DECLARED: bills the full output (unlike `eye`, the zero region above the diagonal is not billed separately as free) | `_array_ops.py` |
| `meshgrid` | `numel(output)`, summed across the returned grids (dense default; `sparse=True`/`copy=False` are separate argument-conditional branches of the same formula) | DECLARED: replicates coordinate vectors into dense grids — a materializing copy, not arithmetic | `_array_ops.py` |

Weight: **1.0** for `arange`, `linspace`, `ones`/`ones_like`/`full`/`full_like`,
`eye`/`identity`, `tri`, and `meshgrid`; **16.0** for `geomspace` and `logspace`
(transcendental path); **0.0** for `zeros`/`zeros_like`/`empty`/`empty_like`.
Source: `src/flopscope/_array_ops.py`.

---

### Sort and select

**Family rule** (DECLARED). Comparison-order derivation and binary search are
**non-sequential access** — see [Access tier](#access-tier-weight-40) — so most of
this family is billed at weight **4.0**; the two members that are not genuinely
sort-based (`sort_complex` sorts one axis with a fixed comparator, `in1d`/`isin` keep
their pre-existing weight) stay at weight **1.0**:

| Op | flop_cost | weight | basis |
|---|---|---|---|
| `sort`, `argsort` | `num_slices × n × ⌈log₂ n⌉` | 4.0 | DECLARED comparison sort (n = axis length) |
| `unique`, `unique_counts`, `unique_inverse`, `unique_values`, `unique_all` | `n × ⌈log₂ n⌉` (axis=None); `num_slices × shape[axis] × ⌈log₂ shape[axis]⌉` (axis=k) | 4.0 | DECLARED sort-based; axis-aware per-slice |
| `lexsort` | `k × n × ⌈log₂ n⌉` (k = number of keys, n = sequence length) | 4.0 | DECLARED |
| `partition`, `argpartition` | `num_slices × n × len(kth)` | 4.0 | DECLARED quickselect O(n) expected |
| `searchsorted` | `m × ⌈log₂ n⌉` (m = queries, n = sorted size) | 4.0 | DECLARED binary search |
| `intersect1d` | `sort_cost(n) + sort_cost(m) + sort_cost(n+m)` (default `assume_unique=False`); `sort_cost(n+m)` when `assume_unique=True` | 4.0 | DECLARED: numpy calls `unique()` on both inputs when `assume_unique` is falsy |
| `setdiff1d`, `setxor1d`, `union1d` | `(n + m) × ⌈log₂(n + m)⌉` | 4.0 | DECLARED |
| `sort_complex` | `num_slices × n × ⌈log₂ n⌉`, `n = a.shape[-1]`, `num_slices = a.size // n` (sorts last axis; equals flat formula only for 1-D) | 1.0 | DECLARED |
| `in1d`, `isin` | `(n + m) × ⌈log₂(n + m)⌉` (sort path); `max(sort_cost(n+m), 2nm)` when numpy's masked-loop path triggers (small integer ar2) | 1.0 | DECLARED algo-aware |

None of the `flop_cost` formulas changed — only the weight moved for the rows marked
`4.0`, from the same DECLARED comparison-sort / binary-search counts as before.

**Complex dtypes**: comparisons are lexicographic (compare real parts, tie-break on
imaginary), so the sort/select family bills factor 2.

Source: `src/flopscope/_sorting_ops.py`, `src/flopscope/_flops.py` (`sort_cost`, `search_cost`).

---

### Linalg direct (non-iterative)

All ops use **weight 1.0** with all shape constants in `flop_cost`.  Per-matrix
cost is multiplied by the batch dimension product for stacked inputs.  Zero-dim
matrices charge 0.

**Complex dtypes**: most of the dense linalg family bills `complex_factor = 4` — a
complex factorization does roughly `4×` the real arithmetic of its real counterpart —
and `linalg.matrix_power` and `linalg.multi_dot` bill this flat `4` as well. The
exceptions are the ops that mirror a bare `fnp.*` twin and take *its* complex factor
instead of this dense rule: `linalg.outer` bills factor 6 like `outer`; `linalg.matmul`,
`linalg.tensordot`, and `linalg.vecdot` bill the same **exact** per-call contraction
factor as `matmul`/`tensordot`/`vecdot`; `linalg.trace` bills 2 like `trace` (a diagonal
sum, not a factorization); and `linalg.cross` bills 4.7 like `cross`. (Where such a row
still carries `complex_factor = 4.0` in the registry it is inert metadata — the
delegation or the op-specific factor applies instead.)

| Op | flop_cost (per matrix) | basis | source |
|---|---|---|---|
| `linalg.cholesky` | `n³/3` | DERIVED: Cholesky factorization (dpotrf) | `_decompositions.py:cholesky_cost` |
| `linalg.qr` (reduced/complete) | `2(2mnk − 2k³/3)`, `k = min(m,n)` | DERIVED: factorization (dgeqrf) + Q-formation (dorgqr) ≈ same count | `_decompositions.py:qr_cost` |
| `linalg.qr` (r/raw) | `2mnk − 2k³/3` | DERIVED: factorization only | `_decompositions.py:qr_cost` |
| `linalg.solve` | `2n³/3 + 2n²×nrhs` | DERIVED: LU solve (dgesv = dgetrf + dgetrs); batch is the numpy-broadcast of `a`'s and `b`'s leading dims, not just `a`'s | `_solvers.py:solve_cost` |
| `linalg.inv` | `2n³` | DERIVED: LU factorization + inversion (dgetrf + dgetri ≈ 2n³) | `_solvers.py:inv_cost` |
| `linalg.det` | `2n³/3 + n` | DERIVED: LU factorization (dgetrf) + diagonal product | `_properties.py:det_cost` |
| `linalg.slogdet` | `2n³/3 + 18n` | DERIVED: LU (dgetrf) + sum of log\|diag\| (abs + 16/elem log + reduce) | `_properties.py:slogdet_cost` |
| `linalg.norm` (fro/L1/Linf) | `2 × numel(effective_shape) × n_groups` | DERIVED: FMA=2 square+accumulate or abs+accumulate | `_properties.py:norm_cost` |
| `linalg.norm` (ord=2, nuc) | `(2ab² + 2b³) × n_groups`, `a=max(m,n)`, `b=min(m,n)` | DERIVED: values-only SVD cost per group | `_properties.py:norm_cost` |
| `linalg.vector_norm` | `2 × numel(effective_shape) × n_groups` (standard ord); `(18 × numel + 16) × n_groups` (general fractional p-norm: abs + pow per element) | DERIVED: FMA=2 | `_properties.py:vector_norm_cost` |
| `linalg.matrix_norm` | same as `linalg.norm` | DERIVED | `_properties.py` |
| `linalg.trace` | `diagonal_len × batch`, `diagonal_len = min(m,n)` reduced by `|offset|` (floored at 1) | DERIVED: diagonal adds, batch-multiplied | `_properties.py:trace_cost` |
| `linalg.tensorinv` | `2n³`, `n = prod(shape[:ind])` | DERIVED: via inv | `_solvers.py:tensorinv_cost` |
| `linalg.tensorsolve` | `2n³/3 + 2n²`, `n = isqrt(prod(shape))` | DERIVED: via solve; n is the solved system's true dimension, independent of `axes` reordering | `_solvers.py:tensorsolve_cost` |
| `linalg.matrix_rank` | `2ab² + 2b³ + min(m,n)`, `a=max(m,n)`, `b=min(m,n)` | DERIVED: values-only SVD + `min(m,n)` threshold comparisons | `_properties.py:matrix_rank_cost` |
| `linalg.cond` | `2ab² + 2b³ + 1` for `ord∈{None,2,−2}` (values-only SVD + 1 divide); `2k³ + 4mn + 1`, `k=min(m,n)` for other ords (inv-based) | DERIVED | `_properties.py:cond_cost` |
| `linalg.pinv` | `6ab² + 20b³ + min(m,n) + n·min(m,n) + matmul\_cost(n, min(m,n), m)`, `a=max(m,n)`, `b=min(m,n)` | DERIVED: thin SVD (with vectors) + threshold + diagonal scale + reconstruction matmul | `_solvers.py:pinv_cost` |
| `linalg.lstsq` | `6ab² + 20b³ + matmul\_cost(k,m,c) + k·c + matmul\_cost(n,k,c)`, `k=min(m,n)`, `c=#rhs cols` | DERIVED: thin SVD (with vectors) + U^T b + divide by s + reconstruction | `_solvers.py:lstsq_cost` |
| `linalg.cross` | `3 × numel(output)` (delegates to `fnp.cross`) | DERIVED | `_aliases.py` |
| `linalg.multi_dot` | optimal chain matmul cost, each step `matmul_cost(m,k,n)` = `2mkn − mn`; two arrays with a 0-d operand: `numel(other)` (NumPy delegates that form to `dot`, a scalar multiply) | DERIVED | `_compound.py:multi_dot_cost` |
| `linalg.outer`, `linalg.tensordot`, `linalg.vecdot`, `linalg.matmul` | delegates to `fnp.*` (inherits the bare op's cost *and* complex factor — see the complex-dtype note above) | DERIVED | `_compound.py`, `_aliases.py` |
| `linalg.diagonal`, `linalg.matrix_transpose` | 0 (view) | DECLARED free | `_aliases.py` |

---

### Linalg iterative (eigen / SVD)

These ops use LAPACK drivers that iterate until convergence; counts are
leading-order estimates of the standard operation count.  All use
**weight 1.0**, and bill `complex_factor = 4` on complex input (as for the
direct linalg family).

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
| `fft.fft2`, `fft.ifft2`, `fft.fftn`, `fft.ifftn` | staged: `Σ` of the per-axis 1-D `fft` costs over numpy's execution cascade (reverse axis order), each `5 × batchᵢ × dᵢ × ⌈log₂ dᵢ⌉` at that axis's *current* intermediate shape; reduces to `5 × N × Σᵢ⌈log₂ dᵢ⌉`, `N = prod(transform dims)`, when `s` resizes no axis | DERIVED: sum of the 1-D cascade; the final-shape product holds only when no axis is resized by `s` |
| `fft.rfft`, `fft.irfft` | `5 × (N/2) × ⌈log₂ N⌉` | DERIVED: real-input / real-output half-spectrum |
| `fft.rfft2`, `fft.irfft2`, `fft.rfftn`, `fft.irfftn` | staged: a real 1-D FFT on the last axis (`5 × (d_last/2) × ⌈log₂ d_last⌉`) plus complex 1-D FFTs on the remaining axes over the half-spectrum intermediate, summed in numpy's order (r2c: real axis first; c2r: real axis last) | DERIVED: per-stage cascade; the half-spectrum shrinks the downstream batch, so it differs from `5 × (N/2) × Σᵢ⌈log₂ dᵢ⌉` except in the no-resize two-axis case |
| `fft.hfft` | `5 × (n_out/2) × ⌈log₂ n_out⌉` | DERIVED: hfft = irfft(conj(a)) — conjugate-symmetry halves the work |
| `fft.ihfft` | `5 × (n/2) × ⌈log₂ n⌉` | DERIVED: same `hfft_cost(n)` formula |
| `fft.fftfreq` | `n` (index grid scaled by `1/(n*d)` — one divide per output element) | DECLARED: `n` divides |
| `fft.rfftfreq` | `n//2 + 1` (real-spectrum grid has `n//2 + 1` elements) | DECLARED: `n//2 + 1` divides |
| `fft.fftshift`, `fft.ifftshift` | `numel(output)` | DERIVED: `numpy.roll`-based data-movement reindex, billed at weight 1.0 like a materializing copy (not part of the FFT priced-in family) — see [Copy and gather](#copy-and-gather) |

The table above gives the **per-transform** cost; every transform op also multiplies by
however many independent transforms the call performs. `fft`/`ifft`/`rfft`/`irfft`
transform a single `axis` and batch over every other axis (`numel(input) / N`).
`fft2`/`ifft2`/`rfft2`/`irfft2`/`fftn`/`ifftn`/`rfftn`/`irfftn` transform the given
`axes` — or, when `axes` is omitted but `s` is given, the trailing `len(s)` axes, not
every axis — and batch over whatever axes remain.

**Complex dtypes**: `complex_factor` is **1** (priced in) — `5N·log₂N` already counts the
complex transform's real FLOPs (10 per radix-2 butterfly), so a complex128 transform bills
only the `2.0` dtype rate over its float32 counterpart, with no extra structural factor
(see [FFT is priced in](#complex-arithmetic-from-first-principles)).

All counted FFT ops use **weight 1.0**.  Source: `src/flopscope/numpy/fft/_transforms.py`.

---

### Polynomial

| Op | flop_cost | basis | source |
|---|---|---|---|
| `polyval` | `2 × deg × points` (Horner: 1 mul + 1 add per coefficient per point, FMA=2) | DERIVED | `_polynomial.py` |
| `polyfit` | `m×deg` (Vandermonde build) `+ lstsq_cost(m, deg+1, ncols)` (`ncols = y.shape[1]` for 2-D `y`, else 1) | DERIVED: builds an `(m, deg+1)` Vandermonde matrix and solves it via `lstsq` (SVD least-squares) — prices the identical solve `linalg.lstsq` charges for the same shape, rather than a cheaper standalone normal-equations estimate; the SVD factorization is shared across `ncols` RHS columns, so cost grows sublinearly in `ncols`, not linearly | `_polynomial.py:polyfit_cost` |
| `polyadd`, `polysub` | size of the axis-0 zero-padded, broadcast-aligned result (= `numpy.polyadd(a1, a2).size`; reduces to `max(n1, n2, 1)` for 1-D inputs) | DERIVED: mirrors numpy's own zero-pad-then-broadcast algorithm on shapes, including higher-rank/broadcasting operands | `_polynomial.py` |
| `polymul` | `2nm − n − m` (direct conv, FMA=2) | DERIVED | `_polynomial.py` |
| `polydiv` | `1 + Q·(2·n2 + 1)`, `Q = max(n1 − n2 + 1, 0)` (`n1` = dividend length, `n2` = divisor length; each of the `Q` quotient terms costs 1 scale-divide + `n2` mul + `n2` sub — not `polymul`'s `2nm − n − m`, though the two happen to coincide at some shapes) | DERIVED | `_polynomial.py:polydiv_cost` |
| `convolve` | `full`: `2nm − n − m`; `valid`: `(2·min−1)·(max−min+1)`; `same`: exact dot-length sum per numpy C layout | DERIVED per-mode | `_pointwise.py:convolve` |
| `poly` (1-D, build from roots) | `(3n² + n) // 2`, `n = len(roots)` (iterative convolution with length-2 kernel per root; FMA=2) | DERIVED | `_polynomial.py:poly_cost` |
| `polyder` | `t × n − t(t+1)/2`, `t = min(m, n−1)` (order-aware; one multiply per surviving coefficient per derivative step) | DERIVED | `_polynomial.py:polyder_cost` |
| `polyint` | `m × n + m(m−1)/2` (order-aware; m passes each dividing n+j coefficients) | DERIVED | `_polynomial.py:polyint_cost` |
| `roots` | `10n³`, `n = stripped companion dimension` (zero-leading/trailing coefficients stripped before companion matrix is built) | DERIVED: delegates to `eigvals_cost` on trimmed degree | `_polynomial.py:roots_cost` |

`polyfit`, `polyder`, and `polyint` return float64 regardless of the input dtype, so
their real bill is `2×` the `flop_cost` above; `polyval` preserves the input dtype and
bills at its own rate. `polydiv` preserves a float or complex input dtype but floors an
integer/bool input to float64 (numpy runs its scale-divide there), the same
kind-conditional promotion an `eigvals`-backed linalg op uses.

Source: `src/flopscope/_polynomial.py`.

---

### Random (module-level, Generator, RandomState)

Random ops are composite: the generation kernel cost and any setup cost
(PRNG state update, rejection sampling) are folded into `flop_cost`; the
weight tier **varies** by distribution family.  The full bill follows the
four-factor formula — draws bill their output dtype's rate (numpy's samplers
default to float64, rate 2.0); complex dtypes are covered below.

Weight tiers:

- **weight 1.0** — plain uniform/integer/structural draws: `rand`, `random`,
  `random_sample`, `ranf`, `sample` (a documented numpy alias of
  `random_sample` — the same uniform draw, so it prices at the same tier as its
  twins, not at the reorder tier), `uniform`, `randint`, `integers`,
  `multivariate_normal`.
- **weight 4.0** — reorder / resample ops, the same non-sequential [access
  tier](#access-tier-weight-40) as sort and gather: `random.permutation`,
  `random.shuffle` and `random.choice` at module level, and every
  `Generator`/`RandomState`
  `.choice` / `.permutation` / `.shuffle` method (plus `Generator.permuted`). The module-level
  surface delegates to the legacy `RandomState` singleton, so it runs the same
  selection machinery as the method surface and prices identically — a
  module-level entry at weight 1.0 would be a cheaper alias route around this
  tier. A Fisher-Yates
  reorder and a rejection-sampled choice both touch memory non-sequentially, the
  same reason a sort does.
- **weight 16.0** — transcendental samplers (every continuous/transformed
  distribution): `normal`, `standard_normal`, `randn`, `exponential`,
  `standard_exponential`, `poisson`, `binomial`, `geometric`,
  `hypergeometric`, `multivariate_hypergeometric`, `negative_binomial`,
  `multinomial`, `beta`, `chisquare`, `noncentral_chisquare`, `dirichlet`,
  `f`, `noncentral_f`, `gamma`, `gumbel`, `laplace`,
  `logistic`, `lognormal`, `logseries`, `pareto`, `power`, `rayleigh`,
  `standard_cauchy`, `standard_gamma`, `standard_t`, `triangular`,
  `vonmises`, `wald`, `weibull`, `zipf`, and all their Generator /
  RandomState counterparts.

| Op / family | flop_cost | weight | basis | source |
|---|---|---|---|---|
| `random.rand`, `random.random`, `random.random_sample`, `random.ranf` | `numel(output)` | 1.0 | DECLARED: 1 FLOP per uniform draw | `_cost_formulas.py` |
| `random.sample` | `numel(output)` | 1.0 | DECLARED: a documented numpy alias of `random_sample` — the same uniform draw, so it prices at the plain-draw tier like its twins. It has no `dtype=` parameter and always resolves float64, so the bill is `2 × numel` | `_cost_formulas.py` |
| `random.uniform` | `3 × numel(output)` | 1.0 | DERIVED: affine map `low + (high − low) × U` = 1 sub + 1 mul + 1 add per element (FMA=2, three ops) | `_cost_formulas.py` |
| `random.randn`, `random.standard_normal`, `random.normal` | `numel(output)` | 16.0 | DECLARED: flop_cost = numel(output); transcendental weight 16.0 from `default_weights.json`; draws default to float64 (see the preamble above), so the full bill is `32 × numel` unless a narrower `dtype=` is passed | `_cost_formulas.py` |
| `random.randint`, `random.integers` | `numel(output)` | 1.0 | DECLARED | `_cost_formulas.py` |
| `random.choice` (module-level; replace=True, p=None) | `numel(output)` | 4.0 | DECLARED | `_cost_formulas.py` |
| `random.choice` (module-level; replace=False, p=None) | `n` (Fisher-Yates, matches `permutation`) | 4.0 | DECLARED | `_cost_formulas.py` |
| `random.choice` (module-level; replace=False, p≠None) | `sort_cost(n) = n × ⌈log₂ n⌉` (conservative floor for the data-dependent rejection loop) | 4.0 | DECLARED | `_cost_formulas.py` |
| `Generator.choice`, `RandomState.choice` | same formula shape **and weight** as module-level `random.choice` above (`Generator.choice` additionally adds `3n + m×⌈log₂ n⌉` — CDF build + binary search — when `p` is given and `replace=True`) | 4.0 | DECLARED/DERIVED | `_cost_formulas.py` |
| `random.shuffle`, `random.permutation` (module-level) | `max(n, 1)`, `n = x.shape[0]` for array input or the int argument itself | 4.0 | DECLARED: in-place Fisher-Yates draws, dtype-neutral | `_cost_formulas.py` |
| `Generator.permutation`, `Generator.shuffle`, `RandomState.permutation`, `RandomState.shuffle` | `max(shape[axis], 1)` (`axis` defaults to 0; `RandomState` has no `axis` kwarg) | 4.0 | DECLARED: Fisher-Yates draws | `_cost_formulas.py` |
| `Generator.permuted` | `numel(input)` (every element is reordered within its axis-slice, not just `shape[axis]` slices as `shuffle`/`permutation` bill; a nested Python-list input is routed through `asarray` first, so it bills every element regardless of nesting depth, not just the outer dimension) | 4.0 | DECLARED | `_cost_formulas.py` |
| `random.exponential` | `numel(output)` | 16.0 | DECLARED: transcendental weight 16.0; float64 by default (see the preamble above) → `32 × numel` in practice | `_cost_formulas.py` |
| `random.poisson`, `random.binomial`, `random.geometric`, `random.hypergeometric`, `random.negative_binomial`, `random.multinomial` | `numel(output)` | 16.0 | DECLARED: transcendental weight 16.0; these draws resolve to int64 (rate 2.0, same as the float64 case above) → `32 × numel` in practice | `_cost_formulas.py` |
| `random.multivariate_normal` | `26d³ + 2Nd² + 16Nd` (d=dims, N=size) | 1.0 | DERIVED composite: SVD factorization of covariance (`svd_cost(d,d,with_vectors=True)` = `6d·d² + 20d³` = `26d³`) + affine transform (`2Nd²`) + N·d transcendental normal draws (`16Nd`) | `_cost_formulas.py` |
| `Generator.multivariate_hypergeometric` (`method='marginals'`, the default) | `numel(output)` | 16.0 | DECLARED | `_cost_formulas.py` |
| `Generator.multivariate_hypergeometric` (`method='count'`) | `sum(colors) + 2 × num_variates × min(nsample, sum(colors) − nsample) + numel(output)` | 16.0 | DERIVED composite: numpy builds a temporary counting buffer of length `sum(colors)`, then for each of the `num_variates` output draws does a partial Fisher-Yates shuffle over `min(nsample, sum(colors) − nsample)` buffer entries (whichever of the sampled/excluded side is smaller), followed by a separate pass counting the colors of those same shuffled entries — two full passes over that length | `_cost_formulas.py` |
| `random.beta`, `random.chisquare`, `random.noncentral_chisquare`, `random.dirichlet`, `random.f`, `random.noncentral_f`, `random.gamma`, `random.gumbel`, `random.laplace`, `random.logistic`, `random.lognormal`, `random.logseries`, `random.pareto`, `random.power`, `random.rayleigh`, `random.standard_cauchy`, `random.standard_exponential`, `random.standard_gamma`, `random.standard_t`, `random.triangular`, `random.vonmises`, `random.wald`, `random.weibull`, `random.zipf` | `numel(output)` | 16.0 | DECLARED: flop_cost = numel(output); transcendental weight 16.0 for all continuous/transformed distributions; each draw resolves to its actual output dtype — float64 (rate 2.0) for the continuous members, int64 (rate 2.0) for the discrete `logseries`/`zipf` — so the full bill is `32 × numel` in practice | `_cost_formulas.py` |

**Complex dtypes**: the distribution samplers produce **real-only** outputs, so a complex
resolved dtype is `complex_factor = "illegal"` and raises. A draw bills at its **output
dtype's** rate — `standard_normal(dtype=float64)` bills `2×` the float32 draw. The
data-movement random ops (`shuffle`, `permutation`, `choice`, `Generator.permuted`) permute
caller-supplied values of any *numeric* dtype and are billed dtype-neutrally — their cost
counts the selection work, which is width-independent — so complex input does not raise. A
non-numeric pool (object included) is refused by the dtype ban before the selection work
ever runs, see [Dtype and precision](#the-billing-unit-and-rate-table).

Source: `src/flopscope/numpy/random/_cost_formulas.py`.

---

### Stats

Stats ops are composite (weight 1.0; all per-element factors in `flop_cost`).
The billed element count is the numel of the broadcast of the input with the
distribution parameters — array-valued `loc`/`scale` (or `a`/`b`/`s`) enlarge
the output beyond `x` and are charged accordingly. Every `stats.*` kernel computes
in float64 regardless of the input's own dtype, so the real bill is `2×` the
per-element counts below even for a float32 input (e.g. `stats.uniform.pdf` on a
10-element float32 array bills `20`, not `10`).

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
| `hamming` | `18n` (weight 1.0) | DERIVED composite: 1 cosine eval at transcendental rate (16/elem) + multiply + subtract per sample; folded into flop_cost, kaiser-family convention | `_window.py:hamming_cost` |
| `hanning` | `18n` (weight 1.0) | DERIVED composite: 1 cosine eval at transcendental rate (16/elem) + multiply + subtract per sample; folded into flop_cost, kaiser-family convention | `_window.py:hanning_cost` |
| `kaiser` | `23n` (weight 1.0) | DERIVED composite: 1 Bessel I₀ eval at transcendental tier (16/elem) + 7 scalar FLOPs per sample; folded into flop_cost | `_window.py:kaiser_cost` |

`hamming`/`hanning` used to be a separate half-transcendental weight-8.0 tier
(`2n` flop_cost). That tier is retired: both now fold their cosine evaluation into
`flop_cost` at weight 1.0, the same [composite-ops](#composite-ops-weight-10-with-heterogeneous-flop_cost)
pattern `blackman`/`kaiser` already used — bringing the whole window family onto one
consistent weight.

Source: `src/flopscope/_window.py`.

---

### Interp and histogram

The binning family (`histogram*`, `bincount`, `digitize`) derives its result by
searching/counting against bin edges — a non-sequential access, so it sits in the
[access tier](#access-tier-weight-40) at weight 4.0. `interp` and `trapezoid`/`trapz`
are not bin-search ops and stay at weight 1.0.

| Op | flop_cost | weight | basis | source |
|---|---|---|---|---|
| `interp` | `3m + m × ⌈log₂(numel(xp))⌉`, `m = numel(x)` (interpolation arithmetic + binary search per query) | 1.0 | DERIVED | `_counting_ops.py` |
| `histogram` (integer bins) | `n × ⌈log₂(bins)⌉` (binary-search binning pass only) | 4.0 | DERIVED | `_counting_ops.py` |
| `histogram` (string bins, e.g. `'auto'`) | `n × (2 + estimator_cost + ⌈log₂ resolved_bins⌉)` (deferred: resolved after the call; estimator costs: sturges/sqrt/rice=0, fd/auto=+1n, scott=+4n, doane=+6n, stone=+max(100,√n)n) | 4.0 | DERIVED | `_counting_ops.py` |
| `histogram2d`, `histogramdd` | same as `histogram` per axis | 4.0 | DERIVED | `_counting_ops.py` |
| `histogram_bin_edges` | `n` (= `max(n, 1)`) for integer bins; string estimator bins: same formula as `histogram` string path | 4.0 | DECLARED: integer bins charge one comparison per element (no log₂ factor); estimator resolves bin count at call time | `_counting_ops.py` |
| `bincount` | `numel(x)` (floor 1) | 4.0 | DECLARED: one bucket increment per element | `_counting_ops.py` |
| `digitize` | `numel(x) × ⌈log₂(len(bins))⌉` (floor 1) | 4.0 | DECLARED: binary search per element | `_counting_ops.py` |
| `trapezoid`, `trapz` | `4 × numel(y)` | 1.0 | DERIVED: `(d·(y₁+y₂)/2).sum()` ≈ 3 elementwise ops + sum-reduce per point, charged as a clean 4/point upper bound | `_pointwise.py` |

`histogram`, `histogram2d`, `histogramdd`, and `bincount` produce `int64` counts
regardless of the input dtype, so the dtype rate adds a further `2×` on top of the
formulas above (e.g. `histogram` with `n=100, bins=8` bills `2400`, not `1200`).
`digitize` and `histogram_bin_edges` return the query's own index/edge dtype and do
**not** carry this overlay. `interp` also always returns float64, so it too bills `2×`
its `flop_cost` above regardless of the input dtype.

Source: `src/flopscope/_counting_ops.py`, `src/flopscope/_array_ops.py`.

---

### Set ops

| Op | flop_cost | weight | basis |
|---|---|---|---|
| `unique`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values` | `n × ⌈log₂ n⌉` | 4.0 | DECLARED sort-based |
| `in1d`, `isin` | `(n+m) × ⌈log₂(n+m)⌉` | 1.0 | DECLARED sort-based |
| `intersect1d` | `sort_cost(n) + sort_cost(m) + sort_cost(n+m)` (default); `sort_cost(n+m)` when `assume_unique=True` | 4.0 | DECLARED: pre-sorts both inputs when `assume_unique` is falsy |
| `setdiff1d`, `setxor1d`, `union1d` | `(n+m) × ⌈log₂(n+m)⌉` | 4.0 | DECLARED sort-based |
| `searchsorted` | `m × ⌈log₂ n⌉` | 4.0 | DECLARED binary search |

Comparison = 1 FLOP convention. Weight is 4.0 (the [access tier](#access-tier-weight-40))
for every genuinely sort-based row above; `in1d`/`isin` are the one pair that kept their
pre-existing weight 1.0 (see [Sort and select](#sort-and-select) for why).

---

### Counting (diff, ediff1d, clip, allclose, isclose, count_nonzero, trace)

| Op | flop_cost | basis | source |
|---|---|---|---|
| `clip` | `max(n_bounds, 1) × numel(output)` (1 compare-select per bound; n_bounds=0,1,2; floor of 1 ensures materialising copy is not free) | DERIVED | `_pointwise.py` |
| `count_nonzero` | `numel(input)` (every element tested regardless of axis; comparison-scan model) | DECLARED | `_pointwise.py` |
| `diff` | `prod(a.shape[:ax]) × (n×L − n×(n+1)/2) × prod(a.shape[ax+1:])`, `L = a.shape[ax]` | DERIVED: `n` passes of `L−k` subtractions | `_pointwise.py` |
| `ediff1d` | `ary.size − 1 + size(to_begin) + size(to_end)` | DECLARED | `_pointwise.py` |
| `gradient` | base: `sum over the axes axis= selects (all axes when axis=None) of 2·S` (`S = f.size`; one output value per element along each axis — interior differences and both boundaries — at ~2 FLOPs each, so at `edge_order=1` the cost is independent of axis length). Any other accepted `edge_order` (0, 2, or a non-integer in (1, 2]; > 2 is rejected) runs the 5-FLOP second-order boundary stencil on each of the two boundary hyperplanes instead of the 2-FLOP one-sided difference, adding `+6·S/L` per selected axis (`L = f.shape[axis]`). Each coord-array axis adds a spacing surcharge (uniform: `+3(L−1)`; non-uniform: `+3S(L−2)/L + 10(L−2) + 3(L−1) + 4S/L`) | DERIVED | `_pointwise.py:gradient` |
| `allclose` | `7·numel(broadcast) − 1` (6 FLOPs/elem tolerance core + numel−1 all-reduce) | DERIVED | `_counting_ops.py` |
| `isclose` | `6·numel(broadcast)` (sub + 2·abs + mul + add + cmp per element) | DECLARED | `_pointwise.py` |
| `trace` (numpy.trace) | `diagonal_len × n_traces` where `diagonal_len = min(shape[ax1], shape[ax2])` reduced by `|offset|` (floored at 1) and `n_traces = size / (shape[ax1] × shape[ax2])` (batch-multiplied); bills identically to `linalg.trace` on matched axes | DERIVED | `_counting_ops.py:trace` |
| `correlate` | mode-aware: `full` = `2nm−n−m+1`; `valid` = `(2·min−1)·(max−min+1)`; `same` = exact dot-length sum per numpy C layout | DERIVED per-mode | `_pointwise.py:_correlate_cost` |
| `cross` | `3 × numel(output)` (2 muls + 1 sub per output scalar; 3-vec path preserves last dim, 2-D z-only drops last dim) | DERIVED: FMA=2, 3 FLOPs per output element | `_pointwise.py:cross` |
| `cov` | `2f²s + 2fs` (f = features, s = samples) | DERIVED: Gram term `f²` dot products of length `s` (2f²s) + centering pass `fs` elements × 2 FLOPs | `_pointwise.py:_cov_cost` |
| `corrcoef` | `2f²s + 2fs + 2f² + f` | DERIVED: cov_cost + normalization (f² divides, 2 FLOPs each, + f sqrts) | `_pointwise.py:_corrcoef_cost` |
| `unwrap` | `13 × numel(input)` | DERIVED: 13 charged one-FLOP passes (diff, sub×2, mod, add, cmp×3, bitwise-and, select×2, abs, cumsum); the 2 select passes (3-arg `where`) no longer discount to free now that 3-arg `where` itself bills, so all 13 passes are charged — prior value was 11 | `_unwrap.py:unwrap_cost` |

`cov` and `corrcoef` compute in float64 regardless of the input dtype, so their real
bill is `2×` the formulas above.

---

### Copy and gather

**Family rule: every write is metered; the access *pattern* sets the weight.**

This family used to be uniformly free (pure relocation/selection bought a 0 bill). It
is not any more: a materializing copy writes real memory and is charged weight 1.0; a
computed-index gather reads via a non-sequential pattern and is charged weight 4.0
(the [access tier](#access-tier-weight-40)); only genuine views stay free. `flop_cost`
is the count of elements actually **written** — for the triangular/diagonal family
that is fewer than the whole array (e.g. `eye`'s off-diagonal zeros are free), not the
whole buffer indiscriminately.

**Materializing copy (weight 1.0, `flop_cost = numel(output)`)** — assembles a new
buffer by copying or rearranging existing values sequentially: `concatenate`, `stack`,
`hstack`, `vstack`, `column_stack`, `dstack`, `block`, `bmat`, `tile`, `repeat`,
`resize`, `roll`, `insert`, `append`, `delete`, `concat`, `fromiter` (materializes an
iterable into an array, one write per element). `row_stack` bills exactly `vstack`'s
cost — it is a bare `return vstack(tup)` with no `deduct()` of its own, so its own
weight entry is inert by construction. `reshape`, `ravel`, `copy`, and `require` are
billed `numel(input)` regardless of whether NumPy itself returns a view or a copy for
that call — see [Views and metadata](#views-and-metadata-weight-00) for why the
cautious, always-charged price was chosen over trying to detect the view case.
`flatten` (the ndarray method — there is no free-standing `fnp.flatten` function) is
always a genuine copy, never a view, and is billed the same `numel(input)` under the
`copy` op name. `fft.fftshift`/`fft.ifftshift` are `numpy.roll` under the hood — a
data-movement reindex, not part of the FFT priced-in family (see [FFT](#fft)).

**Triangular / diagonal writes (weight 1.0)** — bill only the cells that survive, not
the whole array:

| Op | flop_cost | basis |
|---|---|---|
| `triu` | elements at/above the kth diagonal (batch leading dims multiply in; floored at 1) | DECLARED |
| `tril` | elements at/below the kth diagonal (batch leading dims multiply in; floored at 1) | DECLARED |
| `diag` (2-D input, extract) | `0` — a view, no write at all | DECLARED |
| `diag` (1-D input, construct) | `numel(v)` — the zero background is free | DECLARED |
| `diagflat` | `numel(v)` — the zero background is free | DECLARED |
| `fill_diagonal` | `min(m, n)`, or `ceil(m·n / (n+1))` when `wrap=True` and the array is taller than it is wide (`m > n`) — the count NumPy's flat-slice write (`step = n+1` over the full buffer) actually touches; converges to ~half of `numel(a)`, not all of it | DECLARED |

**Gather — computed-index read (weight 4.0, `flop_cost = numel(output)`)** — a
non-sequential memory access, the [access tier](#access-tier-weight-40): `take`,
`take_along_axis`, `choose`.

**Scatter / select-with-a-given-selector (weight 1.0)** — the mask or index is an
*input*, not derived by testing values, but writing the selected cells is still a
real write:

| Op | flop_cost | basis |
|---|---|---|
| `put` | `numel(indices)` | DECLARED |
| `put_along_axis` | `(numel(arr) / arr.shape[axis]) × indices.shape[axis]` (`indices.size` when `axis=None`) | DECLARED: elements actually scattered |
| `putmask`, `place` | `numel(input)` | DECLARED |
| `select` | `numel(output) × len(condlist)` | DECLARED: one scan per condition, applied across the whole broadcast output |
| `where(cond, x, y)` (3-arg) | `4 × numel(broadcast output)` | DECLARED: gather-rate select — it dereferences a *different* source (`x` or `y`) per output element, the same access pattern as `take` |
| `compress` | `len(condition) + 4 × numel(output)` | DECLARED: scan the condition (1/elem) + gather-rate copy of the kept slices |
| `extract` | `numel(arr)` | DECLARED: scan every element of `arr` against the (already-built) mask |

`where`'s 3-arg form used to be free; it is now the one member of this bucket billed
at the gather rate (`4×`) rather than the plain write rate (`1×`) that `put`/`putmask`/
`place` use, precisely because it *reads* non-sequentially (from whichever of `x`/`y`
the mask picks) rather than writing a single given input through unchanged.

**Selector-deriving ops are charged** (they *test* values to produce the selector,
unlike the ops above whose selector is a given input):

| Op | flop_cost | basis |
|---|---|---|
| `nonzero`, `where(cond)` (1-arg) | `numel(input)` (weight 1.0) | DECLARED: implicit `!= 0` scan per element |
| `argwhere` | `numel(input)` (weight 1.0) | DECLARED: ≡ `transpose(nonzero(a))` |
| `flatnonzero` | `numel(input)` (weight 1.0) | DECLARED: ≡ `nonzero(a.ravel())` |
| `count_nonzero` | `numel(input)` (weight 1.0) | DECLARED: comparison scan every element |

These ops derive a selector by testing element values (`!= 0`), so the test is their
compute cost. The predicate and the selection are the *same* step here — unlike the
3-arg `where(cond, x, y)` above, where the predicate (a separate charged op) is an
*input*.

**Worked examples**:

| Expression | Charge | Reasoning |
|---|---|---|
| `where(a > 0.5, x, y)` | `greater` = `numel(a)`, plus `where` itself = `4 × numel(a)` | the predicate is charged separately; the select is now also charged, at the gather rate |
| `nonzero(a)` | charged `numel(a)` | derives the selector by testing `!=0` — value-test is its compute |
| `arange(n)` | charged `2×numel` | computes `start + i·step` per element (1 mul + 1 add) |
| `meshgrid(x, y)` | charged `numel(output)` | replicates `x`,`y` into grids — a materializing copy, not a view |
| `take(a, idx)` | charged `4×numel(output)` | index given, but the read pattern is non-sequential — access-tier gather |
| `hstack([a, b])` | charged `numel(output)` | copies existing values into a new buffer — a real, sequential write |
| `sort(a)` | charged `4×n·⌈log₂ n⌉` | output order derived by comparing values — access-tier |
| `a.astype(float64)` | charged `numel(a)` | default `copy=True` performs a real write, billed like `copy`, even for a pure width change |
| `a.astype(bool)` | charged `numel(a)` | bills like `copy` — no longer a value-test exception, same formula as any other cast |
| `a.astype(a.dtype, copy=False)` | free | the one true no-op — NumPy returns the identical object, so nothing is written |

**Complex dtypes**: the movement, gather, and scatter ops in this family carry the
component `complex_factor = 2` (the same floor as everywhere else) and never raise on
complex input; the charged selector-deriving siblings (`nonzero`, `argwhere`,
`flatnonzero`) test `!= 0`, a value comparison, so they also bill factor 2 on complex.

Source: `src/flopscope/_array_ops.py`.

---

#### Copy-and-gather: ops with distinct charged siblings

The table below lists ops whose cost formula sits outside the family rules above
because they perform per-bit/index work or I/O beyond pure relocation:

| Op | flop_cost | basis | source |
|---|---|---|---|
| `copyto` | `numel(dst)` (or popcount(`where`) when masked) | DECLARED: priced per element written, unconditionally — same-dtype or not (see [§Boundary ops](#boundary-ops-free-behavior--a-value-computing-path)) | `_array_ops.py` |
| `packbits` | `numel(input)` (weight 1.0) | DECLARED: per-bit test+shift; value-test per element | `_array_ops.py` |
| `unpackbits` | `numel(output)` (weight 1.0) | DECLARED: unpacks 8 bits per input byte; proportional to output | `_array_ops.py` |
| `mask_indices` | `max(numel(mask), n²)` (weight 1.0, priced at the mask's own dtype) | DECLARED: scans the array `mask_func` produces, matching the `nonzero`/`flatnonzero`/`argwhere`/`count_nonzero` convention of billing `numel(input)`; floored at n² (the numel of the `ones((n, n), int)` probe `mask_func` receives as an argument, and can capture) so a `mask_func` that returns something smaller cannot buy a cheaper scan; the returned index arrays are not charged | `_array_ops.py` |
| `getitem` (`arr[key]`) | basic indexing (int/slice/newaxis/Ellipsis, or a tuple thereof): `0` (view); advanced (fancy/boolean) indexing: `4·numel(output)` + `numel(mask)` per boolean-mask part (weight 1.0) | DECLARED: fancy gather billed at the `take` rate; boolean-mask parts additionally scan like `compress` | `_ndarray.py` |

`getitem` is the one op in this table with no module-level `fnp.<name>` call form — it
bills `FlopscopeArray.__getitem__`, i.e. `arr[key]` syntax, not a function call. See
[the unifying philosophy](#the-unifying-philosophy--every-byte-written-is-metered) for
the basic-vs-advanced-indexing distinction this formula rests on.

`mask_indices` prices only the mask that `mask_func` produces, not the index arrays it
returns, so — unlike the dtype-neutral [index generators](#index-generators) below —
its charge scales with the mask's dtype. The charge is also floored at n² (int dtype):
`mask_func` receives numpy's internal `ones((n, n), int)` probe as an argument, so it
can capture that reference and return an arbitrarily small result while still having
had the full probe handed to it. `triu_indices`/`tril_indices` are separate ops with
their own dtype-neutral formula and are unaffected by this.

---

#### Index generators

These ops return coordinate/index arrays rather than values, and price
**dtype-neutrally** — the billed count is `numel` of the returned index array(s)
regardless of the array's actual integer dtype, since the work (computing positions)
does not depend on how wide the position values happen to be stored:

| Op | flop_cost | basis |
|---|---|---|
| `unravel_index` | `numel` of the returned index arrays | DECLARED |
| `ravel_multi_index` | `numel(output)` (= `N`, the number of index tuples) | DECLARED |
| `diag_indices`, `diag_indices_from` | `numel` of the returned index arrays | DECLARED |
| `tril_indices`, `tril_indices_from` | `numel` of the returned index arrays | DECLARED |
| `triu_indices`, `triu_indices_from` | `numel` of the returned index arrays | DECLARED |
| `tri` | `numel(output)` — **not** dtype-neutral; bills the actual output dtype like a value array (see [Generator](#generator-linspace-arange-and-kin)) | DECLARED |
| `broadcast_shapes` | sum of `len(shape)` across the input shapes (floor 1) | DECLARED |
| `indices` | `numel(output)` at the actual output dtype (dense: `len(dims)·prod(dims)`; sparse: `sum(dims)`) | DECLARED |
| `ix_` | `sum(numel(outputs)) + sum(numel(Boolean inputs))` — **not** dtype-neutral: billed at the returned index arrays' actual dtype (`intp`), like `tri` above | DECLARED |

All weight 1.0. Source: `src/flopscope/_array_ops.py`.

`ix_`'s second term is the load-bearing one. Every Boolean argument pays
`numel(arg)`, the full mask, because NumPy runs an internal `nonzero` scan over all of
it; the charge therefore does not fall with the mask's popcount. Without that term a
one-`True` mask over a large array would produce a one-element output and bill almost
nothing, making an arbitrarily large scan free. `ix_` bills weight 1.0 — it was
previously in the free tier and is not any more.

---

### Boundary ops (free behavior + a value-computing path)

A free (weight-0) classification covers only an op's **pure data-movement / structural**
behavior. Any parameter, mode, or path that **computes or inspects values** is **charged**
with a reliable cost reusing the convention for that work; a path we cannot reliably bill
is **rejected with a clear error**. These three ops carry weight **1.0** with a path-aware
`flop_cost`:

| Op | free path (`flop_cost = 0`) | charged / rejected path |
|---|---|---|
| `pad` | — (no free path — every mode writes a fresh output buffer) | `numel(output)` base for **every** mode; movement modes (`constant`, `edge`, `empty`, `wrap`, `reflect`/`symmetric` with `reflect_type='even'`) add nothing further on top of the base; `linear_ramp` and `reflect_type='odd'` add `(numel_out − numel_in)` on top of the base; stat modes `maximum`/`minimum`/`mean`/`median` reduce **every** axis — not only the ones actually padded — over a cross-section that **grows** as earlier axes get padded, adding `Σᵢ stats_i·stat_len_i·cross_i` on top of the base; **`mode=<callable>` raises** |
| `trim_zeros` | — | `numel(input)` (value scan for the nonzero boundary) |
| `copyto` | — | `numel(dst)` (or popcount(`where`) when masked) — every write is priced, same-dtype or not |

For `pad` stat modes: numpy pads axes in order `0..ndim-1`, mutating one shared output
buffer in place, so axis `i`'s reduction reads a cross-section built from axes `< i` at
their *already-padded* size and axes `> i` at their original size:
`cross_i = ∏_{j<i} grown_j · ∏_{j>i} in_shape[j]`, where `grown_j = in_shape[j] +
before_j + after_j` is axis `j`'s final padded length. `stat_len_i = min(stat_length_i,
in_shape[i])` (default = full axis). This reduction happens on **every** axis,
including one with pad width `(0, 0)` — its result is simply discarded into a width-0
output region, but the FLOPs are real; only a wholly **empty** input (`numel_in == 0`,
any axis length 0) skips the reduction loop entirely, mirroring numpy's own early-out.
A full-axis stat (`stat_len` covering the whole axis on both sides) serves both sides
from one reduction; otherwise numpy computes each side separately, even when one side's
width is 0. `mean` adds one divide per stat output cell.

Worked example (`a = arange(5, dtype=int32)`, `pad_width=2` ⇒ `numel_out=9`,
`numel_in=5`): `constant`, `edge`, `wrap`, `empty`, `reflect`, and `symmetric` each
bill **9** (the `numel_out` base, no surcharge); `linear_ramp` and `reflect_type='odd'`
bill **13** (`9 + (9−5)`); `maximum` bills **14** (`9 + 5`); `mean` bills **15**
(`9 + (5 + 1 divide)`). This single-axis case can't show the cross-section growing, so
a second example: `a.shape == (3, 4)`, `pad_width=((1, 1), (0, 0))` (axis 0 padded on
both sides, axis 1 left completely unpadded) ⇒ `numel_out = 5·4 = 20`. `maximum` bills
**52**: axis 0's reduction has `cross_0 = in_shape[1] = 4` and `stat_len_0 = 3`, costing
`4·3 = 12`; axis 1's reduction — even though its pad width is `(0, 0)` — has
`cross_1 = grown_0 = 5` (axis 0's *padded* length) and `stat_len_1 = 4`, costing
`5·4 = 20`; total `20 + 12 + 20 = 52`. `mean` bills **61** (the same `12 + 20 = 32`
reduction cost plus one divide per axis: `cross_0 + cross_1 = 4 + 5 = 9`;
`20 + 32 + 9 = 61`).

(`ravel_multi_index` — an index-math op with no free path — lives in the [Index
generators](#index-generators) table, billed `numel(output)` and dtype-neutral.)

**Complex dtypes**: `pad` bills factor 2 on every mode, including the movement modes —
they now write the same `numel(output)` base as every other mode. The charged
`trim_zeros` bills factor 2 too (value scan / reduction). `copyto` resolves its billed
dtype from its source operand together with its destination via the `out=` doctrine
(`store_billing_dtypes`), not a bare `np.result_type` over both — see [Which dtype
prices a call](#which-dtype-prices-a-call): a numeric destination joins the resolution
and can only widen the bill, same-dtype or not; a non-numeric destination (`str_`,
`bytes_`, `datetime64`, `timedelta64`, structured/void, object-free included) is refused
outright rather than priced, the same as an `object`-carrying one.

---

### I/O (save / load)

Writing an array to disk is a real, billable side effect — even though it produces no
new in-memory array — while reading one back is not: the values already existed
somewhere (the caller's own prior compute paid for them, or they were supplied as
input data), so ingesting them is a free, view-like operation, the same treatment
`from_dlpack` gets.

| Op | flop_cost | weight | basis |
|---|---|---|---|
| `save` | `4 × (numel(array) + ndim(array) × 8)` | 1.0 | DECLARED: I/O write, priced per element serialized plus the array's shape header |
| `savez`, `savez_compressed` | `4 × (Σ numel(array_i) + Σ ndim(array_i) × 8 + Σ len(member name bytes) + 8)` (summed over every array passed, plus the byte length of a `__meta__` blob when present, plus the UTF-8 byte length of every archive member name including `"__meta__"`, plus one more 8-byte shape header for the names blob itself, present whenever the archive has at least one non-empty member name) | 1.0 | DECLARED: same per-element I/O price as `save`, one archive |
| `load` | `0` | — | DECLARED free: ingesting previously-computed values is not new compute |
| `from_dlpack` | `0` | — | DECLARED free: zero-copy ingest from another array library |

The `4×` constant is a flat write-amplification price or format overhead, not a literal
byte count — it multiplies with the normal `dtype_rate`/`complex_factor`/`weight` factors
like any other `flop_cost`, so a `save` of a float64 array bills `2×` the same call on
float32 (`4 × numel × 2.0 × 1.0`), not a fixed byte size. `load` and `from_dlpack` have
no cost path in code and are unconditionally free; `save`/`savez`/`savez_compressed`
are the charged member of this family — writing an array to disk is metered, reading
one back is not.

Each array's `.npy`/`.npz` header also encodes its shape as one 8-byte integer per
dimension — a channel a participant fully controls independent of the element count
(`zeros((0, K))` has 0 elements but an arbitrary `K`, so a bare `4 × numel` price floors
at a near-zero cost no matter how large `K` is). Every billed array's shape header
(`ndim(array) × 8` bytes) is billed alongside its element data. `savez`/`savez_compressed`
bill one further 8-byte shape header for the archive's member-names blob itself: the
concatenated member names are ingested server-side as one synthetic 1-D `uint8` array, so
they carry a shape header of their own, billed in-process too so the two paths match.

The optional `__meta__` dict `savez`/`savez_compressed` accept is JSON-serialized and
written to the archive as a `uint8` byte array, exactly like any named array — so it
bills the same per-byte egress cost (`4 × len(json-encoded blob)`, at the `uint8` dtype
rate) and counts toward the `Σ numel(array_i)` sum above. It is not a free side channel:
billing it separately from the named arrays would let a participant round-trip arbitrary
data through `__meta__` at a flat, size-independent cost.

The archive's MEMBER NAMES — the `savez`/`savez_compressed` keyword-argument names,
plus the auto-generated `arr_0`, `arr_1`, ... names for arrays passed positionally,
plus the literal `"__meta__"` member when a meta block is present — are
written into the archive and read back verbatim by `load`, exactly like array data, so
their UTF-8 byte length is billed too, folded into the same total above. Without this, a
participant could smuggle data through many tiny arrays given very large names (an
archive member name can be tens of thousands of bytes) instead of through the array
values, at a near-zero, name-independent cost.

`savez`/`savez_compressed` accept arrays positionally as well as by keyword, matching
`numpy.savez`: a positional array is stored under its auto-generated `arr_N` name and
bills exactly the same as the identical array passed under that name as a keyword (the
member-name byte cost above is identical either way). A keyword name that collides with
a positional array's auto-generated name raises the same error numpy raises, before any
billing happens.

Source: `src/flopscope/_io.py`.

---

### Functional / higher-order

Operations that apply a user-supplied callable across an array. flopscope bills the
result the wrapper materializes (numpy runs the callback itself).

> **Submission caveat:** these run a Python callback *in-process* and raise
> `RemoteCallbackError` on the client/server backend used for AIcrowd submissions, so
> they cannot appear in submitted code — their cost matters only for local runs.

| Op | flop_cost | weight | source |
|---|---|---|---|
| `apply_along_axis` | `numel(output)` | 1.0 | `_counting_ops.py` |
| `apply_over_axes` | `numel(output)` | 1.0 | `_counting_ops.py` |
| `fromfunction` | `numel(output)` | 1.0 | `_array_ops.py` |
| `fromiter` | `numel(output)` | 1.0 | `_array_ops.py` |
| `mask_indices` | `max(numel(mask), n²)` | 1.0 | `_array_ops.py` |
| `piecewise` | `numel(input) × len(condlist)` | 1.0 | `_counting_ops.py` |

`apply_along_axis` was price-cut from weight 4.0 to 1.0 (the wrapper's own
`numel(output)` formula is unchanged; only the callback's *own* `fnp` calls, billed
separately, ever carried the higher tiers). `piecewise` moved from a flat
`numel(input)` at weight 4.0 to `numel(input) × len(condlist)` at weight 1.0 — every
condition you pass rescans the input once, folded directly into `flop_cost` rather
than relying on the weight to do that work.

---

### View / free (weight 0.0)

**Family rule**: operations that return a view, re-interpret memory, or leave a newly
allocated buffer untouched (no per-element write) charge 0 FLOPs. See [The unifying
philosophy](#the-unifying-philosophy--every-byte-written-is-metered) and [Views and
metadata](#views-and-metadata-weight-00) for the full decision procedure and both
refinements — this entry is the family-table lookup, not a restatement of the rule.

Weight 0 now covers two sub-families:

- **Views / metadata** — no new buffer, or a read-only reinterpretation of an
  existing one: `transpose`, `swapaxes`, `moveaxis`, `squeeze`,
  `expand_dims`, `flip`/`fliplr`/`flipud`, `rot90`, `atleast_1d`/`atleast_2d`/
  `atleast_3d`, `broadcast_to`,
  `view`, `real`/`imag` (component extraction — a strided view or constant-fill, no
  arithmetic), `split`, `hsplit`, `vsplit`, `array_split`, `unstack`, `diagonal`
  (2-D view path only — the 1-D *construct* path writes, see [Copy and
  gather](#copy-and-gather)), `linalg.diagonal`, `linalg.matrix_transpose`,
  `from_dlpack`, `load`, and all other shape/stride/dtype introspection (`ndim`,
  `shape`, `size`, `nbytes`, `itemsize`, `dtype`, `flags`, `base`, `data`, `ctypes`,
  `strides`, `T`, `isscalar`, `isfortran`). `astype`/`asarray` are **not** in this
  list any more — they moved to the charged paragraph below; see [Views and
  metadata](#views-and-metadata-weight-00) for their remaining narrow no-op
  exception.
- **Untouched allocation** — a new buffer whose contents are the platform zero-page
  default, so nothing is actually written: `zeros`, `zeros_like`, `empty`,
  `empty_like`.

**Everything else that used to live in this family now writes real memory and is
charged.** `reshape`/`ravel`/`copy`/`flatten`/`require`, every materializing-copy op
(`concatenate`/`stack`/`tile`/…), `astype`/`asarray` (any real cast or copy — see
[Views and metadata](#views-and-metadata-weight-00) for the narrow
`copy=False`/no-conversion no-op that stays free), and `fft.fftshift`/`fft.ifftshift`
are weight 1.0; `take`/`take_along_axis`/`choose` (gather) are weight 4.0;
`put`/`putmask`/`place`/`compress`/`extract`/`select`/3-arg `where`/`fill_diagonal`
(scatter/select) are weight 1.0 (`where` gathers, so it is weight 4.0) — see [Copy and
gather](#copy-and-gather) for the full breakdown. `ones`/`full`/`eye`/`identity`/
`tri`/`meshgrid` moved the same way — see
[Generator](#generator-linspace-arange-and-kin) for the constant-fill split that
keeps only `zeros`/`empty` (and their `_like` forms) free.

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
