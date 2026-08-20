# Changelog

## Unreleased

### Billing impact

- Six ops that answer with a boolean but compute in a promoted float dtype
  billed their raw integer operand, undercharging by exactly 2x wherever the
  promoted loop rates 2.0 and the operand's own dtype rates 1.0. `signbit`
  publishes no integer loop at all (`np.signbit.types` is
  `['e->?', 'f->?', 'd->?', 'g->?']`), so an integer operand is promoted to
  the same-size float loop before the sign test runs — `int32`/`uint32`
  compute in `float64`; `isneginf`/`isposinf` are
  `logical_and(isinf(x), signbit(x))` and inherit that promotion exactly.
  `isclose`/`allclose` promote harder and for a different reason: NumPy's
  `isclose` casts its reference operand with `result_type(y, 1.)` before
  running `|x−y| <= atol + rtol·|y|`, unconditionally (an explicit
  `rtol=0, atol=0` promotes just the same), so EVERY integer and boolean
  width computes in `float64`. `is_symmetric` and `as_symmetric` (and the
  `SymmetricTensor` constructor, which shares as_symmetric's charge) bill
  `k·(7n − 1)` — allclose's own cost formula, once per non-identity
  generator — because the work they do is
  `np.allclose(array, array.transpose(perm))`; they inherit allclose's
  `float64` floor with it.

  Measured on 1000 elements (a 32x32 matrix for the symmetry pair) under the
  shipped weights, every affected cell moves by exactly 2.00x and nothing
  else moves: `signbit`/`isneginf`/`isposinf` 1000 → 2000 at `int32` and
  `uint32` only (`bool`/`int8`/`int16` promote to `float16`/`float32`, which
  rate 1.0, so they are unchanged); `isclose` 6000 → 12000, `allclose`
  6999 → 13998 and `is_symmetric`/`as_symmetric` 7167 → 14334 at `bool`,
  `int8`, `int16`, `int32` and `uint32`. `int64`, `uint64`, `float32` and
  `float64` operands are unchanged throughout — they already billed the
  dtype the work runs at. This raises what the grader charges, but ships in a
  new version opening a new phase, so no submission is re-evaluated against
  it: nothing already scored is repriced.

  The weights are untouched; only the dtype rate follows the promotion.

### Fixed

- The compute-dtype conformance sweep floored a billed rate at the rate of
  NumPy's RESULT dtypes. For a predicate that result is `bool`, which rates
  1.0 — the lowest rate any dtype resolves to — so the floor was not a
  relaxed oracle for the whole predicate family, it was a switched-off one,
  and every underbill above passed it. Ops returning a PYTHON bool
  (`allclose`, `array_equal`, `is_symmetric`) contributed no result dtype at
  all, leaving the floor on its 1.0 default. The sweep now adds a
  COMPUTE-LOOP floor for every bool-result probe: the widest loop-input
  dtype NumPy itself resolves while running the same call, observed by
  replaying it on plain NumPy against an operand that records each
  resolution — never through flopscope, so the oracle cannot inherit the
  promotion model it audits. An accounting test discovers the bool-result
  probes by running them, so a newly charged predicate op fails until its
  replay exists instead of quietly inheriting the 1.0 floor.

## v0.11.0 (2026-08-17)

### BREAKING CHANGE

- **billing**: every non-numeric dtype — `object`, `str_`, `bytes_`,
  `datetime64`, `timedelta64`, and structured/void (object-free included) —
  now raises `UnsupportedDtypeError` wherever it reaches a registered
  operation, including a free, 0-FLOP one, as an operand (an ndarray, or a
  Python sequence that coerces to one), an explicit `dtype=`, a fill value or
  distribution parameter about to be cast the same way, or an `out=`
  destination. The predicate is a NUMERIC ALLOWLIST (`dtype.kind in
  "biufc"`: bool, signed/unsigned integer, float, complex), not a denylist,
  so a dtype kind flopscope has never seen is refused by default. Object
  carries unbounded per-element Python cost that no rate expresses; the
  other kinds are bounded, but their real per-element cost (itemsize for
  string/bytes/structured, the underlying integer rate for
  datetime64/timedelta64) is not the fixed unit a flat rate assumed. One
  exception: a dtype that is still zero-itemsize once NumPy MATERIALISES it
  (an empty structured spec, or `'V0'`) is let through regardless of kind,
  since it carries no data either way — NumPy's own internals allocate one as
  a zero-byte shape-computation placeholder. A zero-length *string* dtype
  (`'U0'`/`'S0'`) is not in that exception: NumPy promotes it to `'U1'`/`'S1'`
  on allocation, so it is refused like any other string dtype. This includes
  flopscope's own conversion ops
  (`array`/`asarray`/`astype`/`fromiter`/`require`/`full`/
  `full_like`) and every random sampler (module-level, `Generator`, and
  `RandomState`) and `flopscope.stats` distribution, all of which refuse
  non-numeric input rather than convert or relocate through it. Calls that
  previously succeeded now raise, e.g. `fnp.array([1.0, None])`,
  `fnp.zeros(3, dtype=object)`, `fnp.multiply(object_arr, object_arr)`,
  `fnp.random.normal(loc=[obj, obj], scale=1.0)`,
  `fnp.reshape(str_arr, str_arr.shape)`, and `fnp.add(m8_arr, m8_arr)`.
  Convert with plain NumPy before passing data to flopscope
  (`clean = np.array(x, dtype=np.float64)` — not `fnp.array(...)`, which
  refuses non-numeric input too), or hold ragged/mixed data in a Python list
  of numeric arrays. This supersedes part of 0.10.0: #159 stopped an object
  `out=` from discounting the arithmetic's rate to 1.0; such destinations,
  and every other non-numeric one, are now refused outright.

### Billing impact

- A contraction whose combined operand rank exceeds the 52-letter subscript
  budget previously fell back to a price that counted multiplies only, not the
  honest FMA count. Padding both operands with singleton axes until the combined
  rank crossed that budget therefore bought the same arithmetic at roughly half
  price. Padded and unpadded forms now bill identically, so an estimator built
  around the padded form bills close to twice what it did before: measured on
  real estimators at production geometry (width 256, depth 32), one MLP each,
  totals moved by 1.89x to 1.94x. This is the only change in this release that
  moves what the grader charges for calls that already ran, and the only one
  that requires re-evaluation.

- A counted op on a non-numeric-dtype array or destination previously billed
  at a flat rate that did not track the real per-element cost — unbounded
  for `object`, itemsize- or representation-blind for the rest; that surface
  is now refused rather than mis-priced. `copyto`'s non-numeric destination
  now routes through the same `store_billing_dtypes` doctrine every other
  `out=` path uses instead of resolving as a plain operand, so it is refused
  the same way. The same under-bill existed on the sibling routes named
  above (a source or parameter cast to a numeric dtype while only the output
  element count was billed); it is closed the same way.
- `fnp.ix_` now bills `sum(numel(outputs))` for index-array construction plus
  `numel(arg)` for every Boolean argument scanned internally by NumPy's
  `nonzero`; it was previously multiplied by a shipped weight of `0.0`.
- `fnp.ix_` continues to accept plain NumPy arrays, list-like inputs, and
  `FlopscopeArray` inputs, but now rejects foreign NumPy `ndarray` subclasses,
  including `MaskedArray` and `memmap`, because their hooks cannot be safely
  billed.
- **No billed amount changes for the metered-wrapper return-type fix below.**
  It raises the LOCAL, in-process estimate only. Measured on a
  contraction-heavy workload doing arithmetic downstream of the affected ops:
  local `flops_used` 2,230,766 → 2,326,380 (+95,614, +4.29%), while the same
  workload driven over a real client/server round trip bills 2,327,658 grader
  FLOPs on both the old and the new code — byte-identical. The local estimate
  moves toward the amount the grader was already charging; no re-evaluation
  follows from it.
- **No billed amount changes for the linalg return-type fix below either.**
  Same shape, measured the same way. A decomposition-heavy workload built from
  PLAIN numpy operands (the only case that moves — every linalg op already
  wrapped its result when an operand was a `FlopscopeArray`): local
  `flops_used` 8,283,360 → 8,352,480 (+69,120, +0.83%). The same workload built
  from `FlopscopeArray` operands reads 8,352,480 on both old and new code, so
  what the change does is make the plain-operand path agree with the one the
  grader already charged. **That percentage is workload-specific, not a
  characteristic figure.** The newly-billed work is the downstream elementwise
  arithmetic, which is O(n²), while the decompositions that dominate the total
  are O(n³) — so the rise falls as matrices get bigger. The same
  cholesky/inv/qr/eigh/solve loop measured across sizes moves +11.1% at n=4,
  +5.3% at n=8, +2.6% at n=16, +1.3% at n=32 and +0.64% at n=64; a submission
  doing heavier algebra per decomposition sits higher still. Read it as "single
  digit percent on typical geometry, more on small matrices", not as 0.83%.
  On the grader side, 64 measurements — 31 linalg ops
  plus `multi_dot(out=)`, each on both operand kinds — driven through the
  server's real `RequestHandler.handle` → `_pack_result` produce a
  byte-identical wire form, op-call cost and downstream-multiply cost on both
  trees, and a real client/server round trip over the same 28 ops bills 10,112
  grader FLOPs either way with every `RemoteArray`/`RemoteScalar` split
  unchanged. The reason is structural: `Session.store_array` already view-casts
  every stored ndarray to `FlopscopeArray`, so the server never saw the
  difference.
- **No billed amount changes for the cross/outer/contraction return-type fix
  below either.** Local only, measured the same way. The workload, stated
  precisely enough to re-run: 40 iterations at n=32 of `matmul(a, b)`,
  `dot(a, b)`, `outer(u, w)`, `cross(v3, w3)`, `matvec(a, u)` and a scalar
  `inner(u, w)`, with one elementwise expression on each array result
  (`m*0.5 + m`, `d - d*0.25`, `o + o`, `c*2.0`, `mv + mv`) and a `.sum()` over
  each: local `flops_used` 10,570,880 → 11,231,760 on plain-numpy
  operands (+660,880, +6.25%) and 11,067,600 → 11,231,760 on `FlopscopeArray`
  operands (+164,160, +1.48%). The two operand kinds now read the SAME total,
  which is the point — `cross` and `outer` were raw on both kinds, the
  contraction helper only on plain numpy, and the remaining spread was the
  local estimate disagreeing with itself. **Those percentages are
  workload-specific, not characteristic figures:** the newly-billed work is
  O(n²) elementwise arithmetic while the contractions that dominate are O(n³),
  so the rise shrinks as arrays grow. On the grader side, 146 measurements —
  73 call shapes across `cross`, `outer`, every contraction op and their
  unaffected neighbours, each on both operand kinds — driven through the
  server's real `RequestHandler._pack_result` give a byte-identical wire form
  (handle vs by-value, shape, dtype), a byte-identical op-call cost, and
  unchanged `out=`/operand identity on both trees: 0 differences in all three.
- **No billed amount changes for the fft free-op return-type fix below
  either.** Local only, same shape, measured the same way. A realistic spectral
  workload — 20 iterations of `fft2` → `fftshift` → an `abs()²` power spectrum
  weighted by an `fftfreq` grid → `ifftshift` → `ifft2` — moves local
  `flops_used` by +2.86% at n=16 (1,075,840 → 1,106,560), +1.60% at n=32
  (5,121,280 → 5,203,200), +1.03% at n=64 (23,759,360 → 24,005,120) and +0.76%
  at n=128 (108,139,520 → 108,958,720). The rise shrinks with n for the usual
  reason: the newly-billed work is O(n²) elementwise arithmetic while the
  transforms that dominate are O(n² log n). A workload made of nothing but the
  four helpers and arithmetic on their results is the other end of the range
  and roughly doubles — 990,720 → 1,661,600, +67.7% — so read the figure as
  "low single-digit percent on spectral code, much more if the helpers are the
  whole workload". Both operand kinds read identically before and after, since
  these four returned raw on plain numpy and `FlopscopeArray` input alike. On
  the grader side, 134 measurements — 67 call shapes covering all four ops at
  ranks 0–3, int/bool/float32/complex operands, every `axes` form, list and
  Python-float inputs, the eight fft transforms and the ops the earlier stages
  touched, each on both operand kinds — driven through the server's real
  `RequestHandler._pack_result` give a byte-identical wire form, op-call cost
  and argument identity on both trees: 0 differences in all three. A real
  client/server round trip over the same probes is byte-identical too, for a
  second reason worth stating plainly: **the client exposes no `fft` proxy at
  all** — `flopscope/fft/__init__.py` is a stub and every registered `fft.*`
  name raises `AttributeError` — so none of these ops can reach the grader
  today in the first place.
- **No billed amount changes for the `bmat` alignment below either**, and here
  the local estimate does not merely move toward the grader's — it lands on it
  exactly. A workload that is nothing but `bmat` and arithmetic on its result
  (20 iterations of a 2×2 nest of an n×n block, then `m*m`, `+`, `.sum()`)
  reads local `flops_used` 2,560 → 10,200 at n=4, 10,240 → 40,920 at n=8,
  40,960 → 163,800 at n=16 and 163,840 → 655,320 at n=32. Every one of those
  new totals is the number the same workload bills over a real client/server
  round trip, on the old tree and the new tree alike — the grader column is
  10,200 / 40,920 / 163,800 / 655,320 either way. **That ~4× is the extreme
  end of the range, not a characteristic figure:** it is what happens when
  every operation after the `bmat` was billing 0 locally and nothing else is
  in the workload. `bmat`'s own call cost is untouched (8 FLOPs for a 2×2
  float64 probe under the packaged table, before and after, in-process and
  remote), and the packed wire form is unchanged — same array handle, same
  shape, same dtype, stored as the same `FlopscopeArray` — because
  `Session.store_array` was already view-casting the `numpy.matrix` away on
  receipt.

### Feat

- **cost-model**: warn when a label-budget fallback loses precision
- **cost-model**: add label-free contraction cost helper

### Fix

- **client**: rebuild namedtuple results from the wire's container type
- **server**: carry the namedtuple container type on multi results
- **docs**: stop bmat's reference promising the numpy.matrix it no longer returns
- **bmat**: align the in-process answer with the remote backend's
- **client**: accept n-D buffers and ndarray input in array and asarray
- **client**: compute wall and residual timing live inside the context
- **billing**: wrap the four fft free-op returns so downstream arithmetic is billed
- **billing**: wrap contraction results only on the branch that ships raw
- **billing**: wrap outer's plain result, leaving out= identity untouched
- **billing**: wrap cross's array return so downstream arithmetic is billed
- **billing**: wrap linalg's array returns so downstream arithmetic is billed
- **budget**: keep accepting int budgets past float range in the finite guard
- **billing**: wrap metered ARRAY results, keep scalar results by value
- **cost-model**: pin the 0-d y guard to numpy's actual refusal set
- **symmetry**: compute expand_dims axis remap by arithmetic, not a probe array
- **cost-model**: raise ValueError, not IndexError, for 0-d y in cov/corrcoef
- **budget**: compute wall and residual timing live inside the context
- **budget**: reject non-finite flop_budget instead of failing open
- **billing**: classify a bytearray leaf as the numeric buffer it realises as
- **cost-model**: refuse mis-paired matmul contractions before charging
- **billing**: refuse non-numeric Python sequences and zero-length string dtypes
- **cost-model**: pin tensordot axes refusals, not numpy's version-dependent types
- **cost-model**: mirror numpy's tensordot axes check ORDER, not just its refusals
- **cost-model**: parse tensordot axes as numpy does, and refuse before pricing
- **cost-model**: keep an empty contraction free when output symmetry survives
- **cost-model**: refuse mis-paired contractions before dot/inner/tensordot price
- **cost-model**: keep operand symmetry on the dot/inner label-budget fallback
- **cost-model**: stop enumerating branches in label-budget warning text
- **cost-model**: give dot and inner an honest out-of-letters branch
- **cost-model**: route tensordot full-inner path through shared allocator
- **cost-model**: normalise negative tensordot axes before fallback pricing
- **cost-model**: bill tensordot fallback at FMA=2, not multiplies only
- include close bookkeeping in timing
- **billing**: charge ix boolean mask scans
- exclude backend registration from timing
- avoid nested backend timing overlap
- bound backend timing to reset epochs
- rebase live timers at budget reset
- **ci**: align local coverage threshold
- align remote budget summary formatting
- align remote budget context summaries
- query authoritative global budget summaries
- attribute remote summary overhead authoritatively
- harden budget summary reset authorization
- add trusted budget summary epoch reset
- ignore stale closed summary sessions
- serve authoritative budget summaries
- harden server handshake validation
- negotiate authoritative budget summaries
- attribute budget inspection overhead
- **protocol**: restore single-output tuple identity
- **protocol**: preserve nonforeign output semantics
- **protocol**: preserve aliased callback outputs
- **protocol**: preserve callback output layout
- **protocol**: preserve symmetric callback outputs
- **stats**: warn on narrow-float promotion (#201)
- **timing**: attribute numeric protocol callbacks to residual (#199)
- **billing**: meter numeric dtypes only, refuse the rest (#196)
- **billing**: floor reduceat by produced cells

### Refactor

- **cost-model**: generalise tensordot subscript allocator
- centralize budget record updates
- add incremental budget rollups

### Perf

- remove history scans from budget summaries
- aggregate closed budget contexts incrementally
- materialize context summaries from rollups

## v0.10.0 (2026-07-31)

### BREAKING CHANGE

- **einsum**: `out=` now follows the caller's casting rule, exactly as numpy does (#168).
  Calls that previously succeeded by silently truncating — two float64 operands
  contracted into an int64 destination, a complex result stored into a float64
  destination — now raise `TypeError`, matching plain numpy. Accepted calls can also
  change value: the contraction now runs in the dtype numpy contracts in, so
  `einsum("ij,jk->ik", int8_a, int8_b, out=int16_dest)` returns the true 300 where it
  previously returned the overflowed -56. Verified differentially against plain numpy
  (8,640 casting-rule cells plus 20,992 default-rule cells, zero disagreements).

### Billing impact

- Billed FLOP totals change in both directions for `out=` and symmetry-tag patterns:
  destination writes are now priced (einsum #169, fft #163, wider-buffer rates
  #156/#159/#162), symmetry-tag discounts are voided when the tagged buffer is
  rewritten (#157, #165), frexp and the argmin/argmax family no longer over-charge
  (#165), and refused operations are uniformly free (#153, #165, #168). Graded totals
  for submissions that relied on these patterns will differ from v0.9.1.

### Feat

- **matmul**: accept a destination, and settle the tagged-destination rule (#164)

### Fix

- **compat**: make the CI numpy matrix real and fix everything it hid (numpy 2.0-2.4) (#171)
- **billing**: price einsum's destination write and put its wall in backend (#169)
- **einsum**: cast inside numpy's iterator instead of materializing operands (#170)
- **einsum**: apply the caller's casting rule to out=, as numpy does (#168)
- **server**: stop charging for results the server cannot deliver (#153)
- **client**: keep BudgetContext.flops_used live between operations (#155)
- **server**: reject unbound method ops that segfault on dispatch (#154)
- **billing**: meter ufunc.at, reduceat and mask_indices by their real work
- **client**: scope the dispatch accumulator to flopscope's own code (#161)
- **billing**: close the out= follow-ups left open by #156, #162 and #163 (#165)
- **billing**: price the fft destination and stop returning buffers numpy never wrote (#163)
- **billing**: unwrap a one-tuple out= and stop under-billing the destination (#156)
- **billing**: stop a narrow out= discounting a wider accumulator (#162)
- **billing**: stop a non-numeric out= from laundering the arithmetic's rate (#159)
- **billing**: void symmetry tags when the buffer they describe is written (#157)

### Perf

- **einsum**: dispatch pairwise contraction steps through numpy's optimized path (#160)

## v0.9.1 (2026-07-23)

### Fix

- **billing**: cost-model accuracy follow-ups from the #150 review (#151)

## v0.9.0 (2026-07-23)

### Feat

- **billing**: dtype-aware four-factor cost model + reviewer-driven re-tiering (#150)

### Fix

- **client,server**: tag list index keys on the wire to disambiguate from tuples (#148)
- **cost**: charge zero for empty contractions and reductions (#146)

## v0.8.0rc5 (2026-06-25)

### Fix

- **client,server**: decode wire with raw=False; drop bytes-vs-str heuristic (#143)

## v0.8.0rc4 (2026-06-24)

### Fix

- **client**: client-parity rc4 — recover prod submission failures (#141)

## v0.8.0rc3 (2026-06-24)

### Feat

- **client**: client/native parity harness, RemoteArray surface, immutability (#140)

## v0.8.0rc2 (2026-06-22)

### Fix

- **server**: connection-lifetime handle store for warm-child handle aliasing (#139)
- free server array handles on GC; never reuse handle ids (#138)

## v0.8.0rc1 (2026-06-19)

### Fix

- **client**: numpy-free callable dtype objects + parity guard (#137)
- **ci**: restore the GitHub Pages deploy step dropped in the CI refactor (#135)

## v0.8.0rc0 (2026-06-16)

### Feat

- **cost-model**: charge value-changing astype casts (to-bool/float->int/narrowing)
- **cost-model**: charge 1-arg where (nonzero), free the 3-arg select
- **cost-model**: make data-movement and gather ops free (weight 0)
- **client**: clear errors for fnp.<blacklisted>/<server-only> via numpy __getattr__
- mark flops.* cost-introspection helpers as SERVER_ONLY
- **client**: clear server-only errors for top-level + flops.* names
- add SERVER_ONLY declaration synced to client
- **registry**: blacklist numpy iterator/state/dtype-info utilities
- **client**: expose random.Generator/RandomState/SeedSequence
- **client**: RemoteRandomState + RemoteSeedSequence proxies + wire codec
- **server**: dispatch RandomState.<method> and SeedSequence.generate_state
- **server**: pack/resolve RandomState + SeedSequence handles
- **registry**: register symmetric ops; regenerate client registry
- **random**: random.symmetric bills sample + symmetrize ((|G|+2)*numel)
- **symmetric**: bill as_symmetric/is_symmetric; is_symmetric checks generators
- **symmetric**: bill symmetrize at (|G|+1)*numel
- **docs**: generate_api_docs --check gate for ops.json drift
- **cost**: top-k SVD bills verified 4mnk truncated cost (capped at full)
- **client**: make the immutable-array assignment error actionable
- **server**: token-gate budget_open/budget_close via --token-fd
- **errors**: add UnauthorizedControlError (core + generated client)

### Fix

- **release**: make version-sync and the version handshake prerelease-robust (#133)
- **cost-model**: copyto charges only value-changing (lossy) casts, mirroring astype
- **cost-model**: charge copyto value-changing cast only
- **cost-model**: charge trim_zeros value scan
- **cost-model**: charge ravel_multi_index linear-index computation
- **cost-model**: make pad mode-aware (charge value modes, reject callable)
- **cost-model**: concat and ix_ are free data-movement (set weight 0, revert label)
- astype method must honor casting/order params (was silently dropped)
- charge a.nonzero() method (was bypassing accounting)
- **test**: update unwrap pins to 11x and where weight in empirical weights.json
- make unwrap cost consistent at 11 (label + formula pin missed in Task 4)
- **symmetry**: empty/empty_like/tri must not infer constant-fill symmetry
- **types**: use _np.shape(base) so pyright accepts *_like shapes arg
- **#126**: route constant-init ops through deduct so time is accounted
- **#126**: route free view ops through deduct so time is accounted
- **ci**: ops.json drift gate ignores numpy-version-dependent summary
- **poly**: polyfit strips FlopscopeArray inputs (x/y/w) before numpy.polyfit
- **cost**: reject k<1 in svd (close negative-k undercount); refresh wrapper docstring
- **client**: re-sync generated _registry_data.py after random_integers blacklist
- **cost**: cross bills 3*numel(actual result) — robust to axis kwargs (review fix)
- **cost**: intersect1d sorts both inputs; mvn factorization bills SVD
- **cost**: cross/convolve/cov/corrcoef/unwrap/poly honest costs
- **cost**: diag/diagonal view-vs-copy + gather-tier consistency
- **cost**: fft freq grids bill n; random.uniform 3x affine; random_integers blacklisted
- **cost**: stats norm/expon/cauchy/logistic/laplace/truncnorm composite kernels
- **cost**: drop low-value 8-op blacklist reclassification; keep gap fixes
- **cost**: linalg trace/slogdet/multi_dot, random.choice (audit gaps)
- **cost**: sort crash + isin/unique/poly/roots cost fixes (audit gaps)
- **cost**: trace batch, window/fft/histogram/allclose (audit gaps)
- **cost**: _free_ops copy/gather/stack ops bill materialized output (audit gaps)
- **cost**: _pointwise clip/count_nonzero/correlate/gradient/nan costs
- **cost**: stats laplace/lognorm/uniform/cauchy composite kernels
- **client**: self-time send_recv transport so no caller leaks to residual
- **client**: bill flops.load ingress to overhead, add send_recv span guard
- **cost**: ptp 2-pass, average divides, nan-quantile wrappers, free dtype checks
- **cost**: stats norm/truncnorm/lognorm composites bill real kernels
- **cost**: weighted choice bills cdf build; diff bills and accepts pads
- **cost**: lexsort all slices; sort_complex per-slice; select bills output
- **cost**: svd bills full_matrices honestly; general-p norms bill pow
- **cost**: linspace(retstep)/arange/indices bill materialized output (audit-2 verified)
- **cost**: numpy 2.x ufunc aliases bill canonical weight (16x exploit)
- **cost**: norm family bills batch dims (was 1-slice)
- **sort**: forward kind/order to numpy (results diverged for structured/stable sorts)
- **cost**: Generator/RandomState multivariate_normal composite formula
- **cost**: multivariate_normal bills factorization+transform+draws
- **cost**: eigen-family provisional constants; roots composes eigvals
- **cost**: cholesky/qr/det/slogdet textbook constants, mode-aware qr, de-weighted
- **cost**: solve/inv/tensor solvers honest LU constants, nrhs-aware
- **cost**: svd family real FMA=2 constants; de-weight composers
- **cost**: cross parity oracle charges 3/output (matches the cross fix)
- **cost**: poly strips input (no crash), bills 2*n^2 + eigvals on 2-D
- **cost**: vander charges n*(N-2) (seeded x^1 column is free)
- **cost**: cross charges 3*output.size (was 5)
- **cost**: interp adds the search-locate term, not multiplies by it
- **cost**: polydiv scales with quotient length, not dividend*divisor
- **cost**: geomspace/logspace cost broadcast output x transcendental weight
- **cost**: linspace costs 2*numel(output), broadcast-aware
- **cost**: trapezoid/trapz charge 4*numel (FMA=2 averaging pass)
- **cost**: average via _call_numpy; oversized tensordot via einsum_cost
- **cost**: var/std/nanvar/nanstd bill 4 passes; weight 2.0->1.0
- **cost**: average charges the a*w multiply pass when weighted
- **cost**: polymul uses convolve FMA=2 formula
- **cost**: multi_dot promotes 1-D operands (no matvec overcharge)
- **cost**: route tensordot partial contraction through einsum (FMA=2)
- **docs-gen**: preserve ufunc wrapper signatures; sanitize volatile reprs
- **server**: ignore client flop_multiplier; cost is flop_cost*weight only

### Refactor

- **weights**: drop duplicate weights dict; delete generate_default_weights.py
- **weights**: empirical-docs read applied weight from default_weights.json
- **weights**: ops.json + coverage read billed default_weights.json
- retire leftover 'free ops' section labels after rename
- rename _free_ops.py to _array_ops.py (it holds charged ops too)
- **symmetric**: extract uncounted _project_core/_check_generators
- **cost**: matmul_cost delegates to einsum_cost (single source of truth)
- **cost**: tensorsolve/tensorinv delegate to solve/inv costs
- **client**: drop flop_multiplier; BudgetContext stays functional
- **core**: remove vestigial flop_multiplier from BudgetContext

## v0.7.0 (2026-06-09)

### Feat

- **warn**: warn that flops.configure() is a no-op on flopscope-client / eval servers
- **client**: re-export participant-facing error classes at top level
- **client**: raise RemoteSerializationError for non-serializable args
- **warn**: warn in-process when callback ops are used (RemoteCallbackWarning)
- **api**: add remote_unsupported_ops() to enumerate callback ops
- **client**: raise RemoteCallbackError for callback ops instead of opaque msgpack error
- **client**: add local_callback flag and RemoteCallbackError codegen
- **budget**: add deduct_after deferred-cost timer (records backend, charges at exit)
- **budget**: add _call_user_code carve-out so user-code time bills to residual
- **io**: pickle-free savez/load + flops.Module (#116)

### Fix

- **budget**: re-sort unique compat shim inside its deduct block
- **budget**: route bmat/concat/dstack data-movement through deduct_after
- **budget**: record data-movement numpy time as backend via deduct_after
- **budget**: satisfy pyright for _DeferredOpTimer timer-union and test budget narrowing
- **budget**: bill callback wall time to residual for callback ops
- **client**: rehabilitate test suite + ship flopscope.numpy (#118)

### Refactor

- **budget**: extract _charge_op shared by deduct and deduct_after

## v0.6.0 (2026-06-08)

### BREAKING CHANGE

- consumers reading these attributes (e.g.
ctx.residual_wall_time) must update to the _s names; there are no aliases.

### Refactor

- **budget**: rename BudgetContext timing props to _s suffix (#117)

## v0.5.0 (2026-06-06)

### BREAKING CHANGE

- multi-operand einsum path selection and billed totals may
change where FMA=2 vs FMA=1 flips the cheapest order.
- FLOP costs change for dot/inner with >2-D operands.
- FLOP costs change for vecmat, matvec, vecdot, and N-D/mixed
matmul. Consumers that pin or budget on absolute FLOP counts should re-baseline.

### Feat

- **timing**: precise client/server timing split (#115)

### Fix

- **cost**: broadcast size-1 axes in the accumulation cost model
- **opt-einsum**: FMA=2 accumulation cost in contraction-path search
- **cost**: route dot/inner N-D through einsum (outer-product subscripts)
- **linalg**: lstsq uses matmul_cost now that matmul 2-D×1-D is exact
- **cost**: count batch/broadcast axes in vecmat/matvec/vecdot + matmul N-D

### Refactor

- **pointwise**: extract _einsum_routed_binary contraction-cost helper

## v0.4.3 (2026-06-02)

### Fix

- **server**: raise UnsupportedReturnType for unpackable results

## v0.4.2 (2026-06-01)

### Feat

- **ci**: gate numpy compat checks

### Fix

- support fnp.random.default_rng() across the client/server boundary

## v0.4.1 (2026-05-26)

Bug-fix release for the broken `flopscope[server]` extra in v0.4.0.

### Fixed

- The `flopscope[server]` extra now correctly pins
  `flopscope-server==0.4.1` (matching the rest of the release). In
  v0.4.0 the extra was stuck at `flopscope-server==0.3.0` because the
  pin location was not tracked by commitizen's `version_files`, so
  `pip install "flopscope[server]==0.4.0"` from PyPI was
  **unresolvable** (it pulled flopscope-server 0.3.0, which in turn
  requires flopscope==0.3.0, conflicting with the 0.4.0 root).
- `pip install "flopscope[server]==0.4.1"` resolves cleanly.

### Tooling

- `commitizen.version_files` now includes
  `pyproject.toml:flopscope-server==` so the `[server]` extra pin
  follows future bumps automatically.
- `scripts/check_version_sync.py` now compares 8 version locations
  (added the `[server]` extra pin) and would catch this regression
  in CI. `tests/test_check_version_sync.py` includes a corresponding
  guard test (`test_server_extra_pin_drift_detected`).
- Drift-detection tests in `tests/test_check_version_sync.py` are
  now version-agnostic (they read the current X.Y.Z from
  `pyproject.toml` at test time instead of hardcoding it). v0.4.0's
  main CI failed after the bump because hardcoded `"0.3.0"` strings
  no longer matched.

## v0.4.0 (2026-05-26)

Follow-up to v0.3.0 that completes the multi-package PyPI release. All
three packages — `flopscope`, `flopscope-server`, `flopscope-client` —
are now published in lockstep, each with a polished README rendering
on its PyPI project page.

### Added

- `flopscope-client` first PyPI release. The Trusted Publisher block
  on PyPI's side that deferred this package from v0.3.0 was resolved.
  The package is now in both `build` and `publish-pypi` matrices in
  `.github/workflows/pypi-publish.yml`, treated identically to
  `flopscope-server`.
- Dedicated `README.md` for `flopscope-server` and `flopscope-client`
  (the root `flopscope` README was already present in-tree but was
  not wired into PyPI metadata).
- `license = "MIT"` field added to the server and client pyprojects
  (only the root previously declared it).

### Fixed

- `[project].readme = "README.md"` added to all three pyproject.toml
  files. v0.3.0 had published flopscope and flopscope-server with
  empty descriptions because no readme was configured; v0.4.0
  backfills them.

### Tooling

- The PyPI publish workflow's environment-approval gate now covers all
  three matrix entries with a single click.

## v0.3.0 (2026-05-26)

Synchronized multi-package release. The `flopscope-server` package is
published to PyPI for the first time, versioned in lockstep with
`flopscope`. The `flopscope-client` package is built and tested in this
release but its PyPI publish is deferred to a follow-up release pending
resolution of a PyPI Trusted Publisher bug (the publisher-create form
returns 500 for the `flopscope-client` project name despite the
identical request succeeding for `flopscope-server`).

### Added

- `flopscope[server]` extra: `pip install "flopscope[server]"` installs
  both flopscope and flopscope-server, exact-pinned to the same version.
- `flopscope-server` first PyPI release. Server-side runtime for the
  client/server architecture; pulls in flopscope as a dependency.
- Runtime version handshake between client and server: the first
  request from a flopscope-client to a flopscope-server compares
  versions and raises `ConnectionError` with both versions on mismatch.
  Code lives in both packages so the contract is in place for the
  follow-up flopscope-client PyPI release.

### Changed

- `flopscope.__version__` now reflects the synchronized release line
  (still suffixed `+np<numpy_version>`).
- `flopscope-server`'s `flopscope` dependency is now an exact pin
  (`flopscope==0.3.0`) so server and library always travel together.

### Tooling

- Commitizen `version_files` is configured to update all version
  strings across the three packages in one `cz bump` invocation,
  including the cross-package pin.
- New `scripts/check_version_sync.py` and `make check-sync-versions`
  catch drift in CI before merge.
- `.github/workflows/pypi-publish.yml` is now a matrix workflow:
  one `v*` tag triggers three parallel builds, three parallel
  publishes (gated by a single `pypi` environment approval), and one
  GitHub Release.

## v0.2.0 (2026-05-26)

First PyPI release.

Flopscope is a NumPy-compatible math library that counts every FLOP
analytically, so compute budgets stop being guesswork.

### What's included

- 508 NumPy-compatible operations with analytical FLOP cost formulas
- Symmetry-aware einsum cost model (direct-event α/M)
- Orbit-mapping cost model for reductions (`sum`, `mean`, `median`, …)
- Configurable FMA cost convention (1 op vs 2 op)
- Budget tracking via `flopscope.BudgetContext` with namespaces and
  per-operation breakdowns
- Symmetric tensor support via `flopscope.as_symmetric`
- Bilinear-wrapper symmetry propagation (`matmul`, `dot`, `outer`,
  `inner`, `vdot`, `tensordot`)
- Public inspection helpers: `einsum_accumulation_cost`,
  `reduction_accumulation_cost`, `tier2_reduction_cost`

### Release tooling

- Commitizen for version bumps + CHANGELOG management
- Conventional-commits enforcement via a `gitlint` `commit-msg` hook
- PyPI publishing via Trusted Publishing (OIDC, no API tokens stored)
- Auto-created GitHub Release on every tag push

See the [README](README.md) for the API overview and the
[docs site](https://aicrowd.github.io/flopscope/) for guides and the
full API reference.
