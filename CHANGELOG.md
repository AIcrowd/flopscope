# Changelog

## Unreleased

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
  exception: a zero-itemsize dtype (an empty structured spec, or a
  zero-length string/bytes dtype) is let through regardless of kind, since it
  carries no data either way — NumPy's own internals allocate one as a
  zero-byte shape-computation placeholder. This includes flopscope's own
  conversion ops (`array`/`asarray`/`astype`/`fromiter`/`require`/`full`/
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

### Fix

- **billing**: `tensordot`, `kron`, `diff`, `gradient`, `ediff1d`, `convolve`,
  `correlate`, `cov`, `sort_complex`, and the array-returning shapes of
  `corrcoef`, `interp`, `trapezoid`, and `trapz` now return flopscope types
  instead of the raw `numpy.ndarray` their inner numpy call produced.
  Arithmetic performed on one of those results billed **0** FLOPs in-process
  while the grader billed it: the server stores an ndarray result as a handle,
  so the participant gets a `RemoteArray` whose arithmetic is dispatched and
  charged. Local estimates therefore read low against what was actually
  charged, and they now rise to meet it (figures under *Billing impact*
  above). `gradient`'s multi-axis tuple return is preserved.
  **A numpy SCALAR result is deliberately left unwrapped** — `vdot`, and
  `trapezoid`/`trapz`/`interp`/`corrcoef` on the argument shapes where numpy
  returns a scalar. The server packs a numpy scalar by value, so the
  participant gets a `RemoteScalar` whose arithmetic runs locally and costs
  nothing; the in-process path already agreed at 0. A 0-d `FlopscopeArray` is
  an `ndarray`, so wrapping those results would have flipped them to handles
  and started charging downstream arithmetic that is free today — a change to
  grader billing, which this release deliberately does not make.
  The 14 sites carried a `# wrapped at fnp.<name> import time` comment
  describing a wrapping mechanism that does not exist; the comments are
  removed. A registry-driven sweep now probes every metered op, in every
  argument form it accepts, and pins the set that still returns a raw
  `numpy.ndarray` as a shrink-only inventory, so a new wrapper cannot regress
  silently. That sweep also found 36 further ops with the same defect,
  concentrated in `linalg` and `fft` and including structured returns
  (`EigResult`, `QRResult`, `SVDResult`), plain tuples (`histogramdd`,
  `linalg.lstsq`) and a `numpy.matrix` (`bmat`). The 19 `linalg.*` entries are
  closed by the next item, 8 contraction/product ops by the one after it, and
  the 4 `fft` index/frequency helpers by the one after that; the remaining 5
  (`bmat`, `histogramdd`, `isfinite`, `isinf`, `isnan`) are recorded, not
  changed. #193 is therefore referenced, not closed.
- **billing**: every array-returning op in `flopscope.numpy.linalg` now returns
  a flopscope type on plain-numpy operands too, closing 19 of the 36 raw
  returns the sweep above found. These ops already wrapped their result when
  an operand was a `FlopscopeArray`, so the gap was the plain-numpy case: a
  participant holding `numpy` arrays got a raw ndarray back from `cholesky`,
  `eig`, `inv`, `qr`, `solve`, `svd`, `lstsq` and the rest, and the arithmetic
  that followed billed 0 in-process while the grader billed it. The module
  gained the same `wrap_module_returns` call nine other modules already carry,
  which wraps ndarray returns only: `det`, `norm`, `cond`, `matrix_rank`,
  `slogdet`'s `sign`/`logabsdet` and `lstsq`'s `rank` stay numpy scalars, so
  they still reach the participant as a `RemoteScalar` packed by value and
  their downstream arithmetic stays free. `EigResult`, `EighResult`,
  `QRResult`, `SVDResult` and `SlogdetResult` keep their named type — the wrap
  rebuilds them rather than flattening them to tuples, so `.eigenvalues`,
  `.U` and `.sign` still resolve. `linalg.multi_dot` is deliberately excluded:
  it is the one op here that takes `out=` and returns that buffer by identity,
  and wrapping the return turned `multi_dot([m, n], out=dest) is dest` from
  True to False for a plain-ndarray destination. That exclusion has a price,
  stated rather than glossed: `multi_dot`'s ordinary array-returning shape
  still hands back a raw ndarray on plain-numpy operands, so the honest count
  across this release is 31 closed, 5 still inventoried, and 1
  closed-off-by-exclusion. It cannot
  go in `KNOWN_RAW_RETURN_OPS` — that inventory is exact in both directions and
  the sweep classifies `multi_dot` as clean, because the only probe form that
  fits it returns a scalar — so it is pinned by a dedicated test instead.
  `linalg.tensorsolve`, which
  no sweep probe form fits, is fixed by the same call and carries its own
  assertion. Retracts a documented promise: `solve` and `lstsq` claimed
  numpy's subclass-return policy (a plain `b` yields a plain-ndarray solution
  even when `a` is a `FlopscopeArray`). That no longer holds and the
  docstrings saying so are deleted. `linalg` was the only module honouring it
  — `sort`, `cumsum`, `tensordot` and `fft.fft` on plain numpy input all
  already return `FlopscopeArray`.
- **billing**: `cross`, `outer`, `linalg.outer`, and the array-returning shapes
  of `dot`, `inner`, `matmul`, `vecdot`, `matvec`, `vecmat`, `linalg.matmul`
  and `linalg.vecdot` now return flopscope types, closing 8 more of the 36 raw
  returns and leaving 9. Same defect as the two items above — arithmetic on the
  result billed 0 in-process while the grader billed it through the client's
  `RemoteArray` — and the same shape of fix, but deliberately applied one
  `return` statement at a time rather than module-wide, because all three sites
  share their return path with cases that must not be wrapped. `outer`'s
  destination early exit still hands back the caller's own buffer (`out is
  result` stays True for a plain and a `FlopscopeArray` destination alike), and
  `outer(v, v)` still returns a `SymmetricTensor` carrying its S2 group, which
  is what halves a symmetric outer's price.
  **The contraction site is asymmetric on purpose.** `dot`, `inner`, `matmul`,
  `vecdot`, `matvec`, `vecmat`, `linalg.matmul` and `linalg.vecdot` all return
  through one helper, and that helper already wrapped its result whenever an
  operand was a `FlopscopeArray` — through `_asflopscope`, which converts even
  numpy's SCALAR (the 1-D × 1-D shape) into a 0-d array. Those calls therefore
  reach the grader as an array handle today and their downstream arithmetic is
  charged. Only the plain-numpy branch changes. Rewriting the whole return to
  the scalar-preserving wrap — the obvious one-liner — would have REMOVED that
  existing wrap and cut grader billing on 1-D × 1-D contractions from billed to
  0; measured through the server's own `_pack_result`, it moves 7 wire forms
  from a handle to by-value. Wrapping unconditionally instead moves 15 the
  other way. The form shipped here moves none of them, and both failure
  directions are now pinned by tests rather than by review attention.
  One residual is recorded rather than closed: a **complex** scalar
  contraction result is still a #193 divergence. Scalars are passed through on
  the reasoning that they are free in-process and free on the grader, but
  `_pack_result` only packs by value when `.item()` is msgpack-native, and a
  `complex128` scalar is not — it falls back to a handle the grader bills while
  in-process billing stays 0. It packs identically on the pre-fix tree, so
  nothing regressed; it is now written down and pinned instead of implied.
- **billing**: `fft.fftfreq`, `fft.rfftfreq`, `fft.fftshift` and
  `fft.ifftshift` now return flopscope types, closing 4 more of the 36 raw
  returns and leaving 5. Same defect as the three items above: all four ended
  in a bare `return result` with no wrapping of any kind, so arithmetic on a
  frequency grid or a shifted array billed 0 in-process while the grader would
  bill it. Closed module-wide — the same `wrap_module_returns` call `linalg`
  and nine other modules already carry — rather than one `return` at a time,
  because `flopscope/numpy/fft/_free.py` contains nothing but those four
  metered wrappers: its only other public names are imports, which the default
  `check_module=True` filter excludes. That was enumerated before the form was
  chosen, and a test now pins both halves of it — the set of names the pass may
  reach, and the identity of the imported helpers it must leave alone.
  None of the four takes `out=`, so there is no caller-buffer identity to
  break, and no `skip_names` is needed.
  **There is no scalar shape to trap on here**, which is what makes the
  module-wide form safe: wrapping a numpy scalar would flip a by-value
  `RemoteScalar` into an array handle and start charging arithmetic that is
  free today. `fftfreq`/`rfftfreq` take an int and always build a rank-1 grid,
  including at `n=1`; `fftshift`/`ifftshift` of a 0-d input never reaches
  flopscope's return line at all, because `numpy.roll` raises `ValueError`
  first. Both are asserted against numpy's own behaviour rather than
  remembered, so they track numpy rather than this changelog.
  `tests/test_fft_free.py` keeps its billed numbers (8, 5, 8, 8) and moves its
  `numpy.allclose` check outside the budgeted region: the comparison is a
  metered op now that the result is a flopscope type, so the old form measured
  the op plus the check and read 63 where it means 8. Restructured, not
  relaxed, with the premise pinned by its own test.
- **symmetry**: `expand_dims` no longer dies on high-rank input. The
  symmetry-transport helper allocated an `np.empty` of `(ndim + 1)!` float64
  elements — 152 TiB at rank 15 — solely to read one `.shape`, and did so even
  when the array carried no symmetry group, so calls failed somewhere around
  rank 14 while numpy itself handles rank 41. The remap is now pure shape
  arithmetic that allocates nothing, with axis normalization delegated to
  numpy's own `normalize_axis_tuple` so negative, tuple, repeated, and
  out-of-range axis forms raise exactly what `numpy.expand_dims` raises across
  the supported numpy range. Symmetry groups for calls that already worked are
  unchanged, and `expand_dims` still bills 0 FLOPs.
- **budget**: a non-finite `flop_budget` is now refused at construction.
  Validation only tested `flop_budget <= 0`, and NaN compares False against
  every bound, so a NaN budget was accepted, propagated through
  `flops_remaining`, and made the exhaustion comparison always False —
  `BudgetExhaustedError` became unreachable and the budget silently stopped
  enforcing. No billed amount changes.
- **budget**: `wall_time_s` and `residual_wall_time_s` now report live values
  while the context is open instead of `None`, matching what
  `budget_summary_dict()` already reported for the same two quantities. Both
  route through the helper the summary path uses. Values read after the
  context exits — what Whestbench bills from — are unchanged. Note the client
  is being changed the other way for the same reads (it cannot compute the
  backend/overhead split locally and will raise instead of returning `None`),
  so an in-process live read returns a float where a client live read raises.
- **cost-model**: `cov` and `corrcoef` given a 0-d `y` now raise `ValueError`
  with a message naming the operand, instead of an `IndexError` from
  flopscope's own shape arithmetic. The guard runs before `budget.deduct`, so
  nothing is billed before or after — the refusal set is unchanged, only the
  exception class and message. numpy refuses the same calls whenever `x`
  carries more than one observation; it accepts a 0-d `y` in the single
  observation case (`np.cov([1.0], np.float64(2.0))`), which flopscope refused
  before this change and still refuses. Un-refusing it would change
  accept/refuse on a billable call, so it is left for the repricing wave.
- **cost-model**: `tensordot` now parses `axes` the way `np.tensordot` parses
  it, and refuses what numpy refuses before any budget is charged. Three
  measured gaps, all of them a charge for work that never ran or a rejection
  of work numpy runs: a repeated contracted axis (`axes=([0, 0], [0, 0])`) was
  priced as a genuine double contraction and charged before numpy's
  `ValueError`; numpy integer scalars (`np.int64`, `np.int32`, ...) failed an
  `isinstance(..., int)` test and were rejected with `TypeError` in both the
  whole-`axes` and per-operand spellings, though numpy accepts them; and a
  one-shot `axes` spec was drained for flopscope's own geometry and the
  exhausted object then forwarded on, so the contraction was priced and
  charged and only afterwards refused. numpy now receives the normalised
  pairing rather than the caller's object, so the spec is read exactly once.
  Every `axes` spelling numpy accepts is still accepted and bills exactly what
  it billed before. What is guaranteed for an *invalid* spec is that it is
  refused before `budget.deduct`, so `flops_used` is untouched; the exception
  **type** is not guaranteed to match `np.tensordot`'s, because numpy reorders
  these checks between its own releases. numpy 2.4 added an explicit
  duplicate-axis check ahead of the shape indexing that 2.0-2.3 reach first,
  so `axes=([0, 0], [0, 1])` against a rank-1 second operand is an `IndexError`
  on numpy 2.2 and a `ValueError` on 2.4; no single fixed order can match the
  supported range. flopscope validates in one order on every numpy -- axis
  counts, then axis validity and range, then duplicates, then extents -- and a
  differential sweep against live `np.tensordot` runs as a test, asserting the
  accept/reject decision, `flops_used == 0` on every refusal, and identical
  results, but not the exception type.
- **cost-model**: a `tensordot` contracted axis given as a 0-d `ndarray` no
  longer charges for a call that then fails on numpy 2.4. numpy accepts that
  spelling up to 2.3 and refuses it from 2.4, where its new duplicate check
  hashes the axis objects and a 0-d `ndarray` is unhashable — so the raw
  object being forwarded meant `budget.deduct` had already charged the
  contraction when numpy raised. Each axis is now canonicalised to a plain
  `int` before pricing and before being handed back to numpy, so the call runs,
  and bills the same amount, on every supported numpy. This is deliberately
  more permissive than numpy 2.4 on that one spelling; refusing it instead
  would break working code on the numpy currently pinned in production.
- **billing**: floor `ufunc.reduceat` at one base-ufunc application per produced cell.
- **billing**: route `fnp.ix_` through the NumPy timing boundary so Boolean-mask
  scans count as backend time instead of Flopscope overhead.
- **billing**: `UnsupportedDtypeError` raised from inside a wrapper built by
  an internal factory (e.g. `mean`/`std`/`var`/`nanmean`/`nanstd`/`nanvar`)
  now names the actual operation in its message instead of the factory's
  generic internal closure name.
- **docs**: `cost-model.md` now states the shipped weights for module-level
  `random.choice` (4.0, the reorder tier) and `random.sample` (1.0, the
  plain-draw tier). The page had carried the pre-review values since the
  four-factor rewrite, contradicting `default_weights.json` in both
  directions. Documentation only — no billed amount changes. Also narrows
  the "fixed output dtype ⇒ not neutral" rule to the *distribution* samplers,
  which the Random section had already carved `choice`/`shuffle`/
  `permutation`/`permuted` out of. Reported in #176.
- **cost-model**: contraction wrappers now share one einsum subscript
  allocator and one out-of-letters price. Previously `tensordot`, `dot`/
  `inner`, and `tensordot`'s full-inner fast path each allocated subscript
  labels independently and disagreed about what to do when the 52-letter
  alphabet ran out: `tensordot` fell back to a multiply-only formula that
  charged `alpha` instead of the honest `2 * alpha - M`, `dot` and `inner`
  let a bare `StopIteration` escape, and the full-inner path allocated only
  26 letters and leaked a NumPy `ValueError` above 26 dimensions. All four
  call sites now route through `_contraction_subscripts`; when it runs out
  of letters they price the contraction from the same FMA=2 model without
  an einsum subscript string, and the resulting charge is never lower than
  what the einsum path would compute for the same operands, though it can
  be higher. A `CostFallbackWarning` — deduplicated per operation name, and
  suppressible with `flops.configure(symmetry_warnings=False)` — flags the
  calls where symmetry or repeated-operand savings could be forfeited.
  Where the fallback's charge is the accumulation total itself, complex
  operands now bill at the exact `(8K - 2) / (2K - 1)` ratio instead of
  raising; where a symmetry adjustment moves the charge off that total,
  they still fail closed with the existing `RuntimeError`.
- **cost-model**: `dot` and `inner` now keep a `SymmetricTensor` operand's
  surviving symmetry when the 52-letter budget is exceeded, instead of
  dropping it. `tensordot`'s partial-contraction, non-oversized fallback arm
  already composed the post-contraction group onto the output axes and
  priced the unique-element fraction there — its full-inner fallback and its
  oversized-symmetry arm both discard the group instead (`out_sym = None`);
  the shared `dot`/`inner` fallback discarded it on every arm, so the same
  contraction cost more above the budget than below it purely because of
  rank, and returned a plain array where the einsum path returns a
  `SymmetricTensor` (measured: 64 versus 40 FLOPs on a 2-axis-symmetric
  rank-27 operand). The direction was over-billing, never under-billing.
  The composition bails to the dense price — unchanged from before — when a
  group's order exceeds
  `dimino_budget` or the enumeration exceeds it mid-composition, so neither
  operation gains a failure mode above the budget. Repeated-operand
  (`dot(x, x)`) savings are still forfeited there, and still warned about.
- **cost-model**: `tensordot`'s label-budget fallback never normalised
  negative axis indices, mispricing partial contractions above the
  52-letter budget in both directions. A negative axis skipped by the
  contracted-size divisor left it at 1, over-billing (a reproduced 256x
  ratio at rank sum 54, `axes=([-26], [0])`: 33,488,896 instead of
  130,816); a negative axis that leaked into the output shape inflated `M`
  until `2*alpha - M` collapsed to the old multiply-only price,
  under-billing (`axes=([1], [-27])`: 65,536 instead of 130,816). Both now
  cost 130,816, matching the positive-axis spelling. Those figures are FLOP
  costs, before the op weight and dtype rate that scale them into a bill.
  An axis still out-of-range after normalisation now raises `IndexError`
  before `budget.deduct` runs, rather than being silently skipped.
- **cost-model**: `dot`, `inner` and `tensordot` now refuse a contraction
  whose paired axes do not line up — unequal numbers of paired axes, or a
  pair whose extents differ — with `ValueError` before any cost is computed
  and before `budget.deduct` runs. `deduct` charges on entry and does not
  roll back when the wrapped NumPy call raises, so such a call previously
  consumed budget for arithmetic that never happened, and could report
  `BudgetExhaustedError` in place of NumPy's shape error. Neither pricing
  path caught it on its own: above the 52-letter subscript budget the
  arithmetic fallback prices from shapes alone and never inspects the
  pairing, and below it the einsum route only appears to — `einsum`
  broadcasts an extent of 1, so `ij,jk->ik` priced `j=1` against `j=7` in
  full (measured: 390 FLOPs on a rank-2 pair) before NumPy rejected the
  call. All three pricing arms of `tensordot` (full-inner, oversized-
  symmetry, and the ordinary partial contraction) are covered, on both sides
  of the letter budget. The check is exactly NumPy's own predicate: none of
  the three operations broadcasts a size-1 contracted axis, every unequal
  pair including `0` against `n` is a `ValueError`, and equal extents are
  always accepted — `0` against `0` included, which stays a legal empty
  contraction costing 0. Valid contractions are unaffected: a 900-case
  differential sweep against plain NumPy found no accepted call whose price
  or outcome changed. An axis spec that is both mis-paired and out of range
  reports the pairing `ValueError`, matching NumPy's own ordering, rather
  than the `IndexError` for the out-of-range index.
- **cost-model**: a legal empty contraction (a contracted axis of extent `0`)
  now costs 0 even when a non-trivial symmetry survives on the output. The
  symmetry scaler floored its adjusted charge at 1 — correct for real work
  whose scaled price rounds down to nothing, wrong for a contraction that
  performed no multiplies and no accumulations — so above the 52-letter
  subscript budget, where the label-free fallback routes through that scaler,
  `dot`, `inner` and `tensordot` charged 1 FLOP for arithmetic that never
  happened, breaking the zero-domain rule the arithmetic fallback and the
  einsum path both hold. The floored charge also no longer matched the
  accumulation total it was derived from, so the call site withheld its exact
  complex factor and the fail-closed guard raised `RuntimeError` on complex
  operands: the same call billed 0 below the letter budget and failed above
  it. Both are fixed by preserving a zero input cost through the scaler. The
  floor itself is unchanged for non-zero costs, symmetry discounts on
  non-empty contractions are unchanged (measured: a rank-27 pair with a
  surviving `S_2` still bills 315 of a dense 525), and `ufunc.outer`, the
  scaler's other caller, clamps its dense cost to at least 1 before calling
  and so is unaffected. Those figures are FLOP costs, before the op weight
  and dtype rate that scale them into a bill.

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
