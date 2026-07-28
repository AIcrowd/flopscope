# fix(billing): price einsum's destination write and put its wall in backend

## What is wrong

`einsum` is the only contraction that does not forward `out=` to numpy. Every other one
hands the destination down and numpy writes the caller's buffer directly, inside the
kernel: one write per output element, already paid for by the contraction's own cost.
That is why `matmul(a, b, out=d)` bills exactly what `matmul(a, b)` bills, and why the
cost model calls such a destination price-neutral -- it is the buffer numpy would have
allocated anyway.

On the einsum path numpy writes a buffer of its own, and the wrapper then copies that
buffer into the caller's destination by hand
(`src/flopscope/_einsum.py`, the `if out is not None:` branch). That copy is a second,
full, materialising pass over the data -- `dest.size` elements written sequentially, on
top of the pass already paid for. It was charged nothing, and its wall time was
attributed to flopscope rather than to the call.

The rule it breaks is not a special case for contractions. From
`docs/reference/cost-model.md`, "The unifying philosophy -- every byte written is
metered":

> Any op that writes a new buffer is charged at least 1 per element written, whether the
> values it writes are computed (`sin(x)`), copied (`concatenate`), replicated (`tile`),
> or a repeated constant (`ones`).

and, four paragraphs later, why that rule exists at all:

> a participant who can move arbitrary amounts of data for free can launder real compute
> through a materializing copy chain.

`einsum(subs, *operands, out=dest)` was that chain, reachable in one call.

## The measured attribution

All figures measured on this branch, macOS arm64, numpy 2.4.6, flopscope 0.9.1.

### The charge

Forty calls of `einsum("ij,jk->ik", a, b)` on 2048x2048 float32, with and without a
destination, against the same computation written out in full:

| spelling | FLOPs billed |
| --- | ---: |
| `r = einsum(a, b)` | 687,026,995,200 |
| `einsum(a, b, out=dest)` | 687,026,995,200 |
| `r = einsum(a, b); copyto(dest, r)` | 687,194,767,360 |
| `matmul(a, b, out=dest)` | 687,026,995,200 |

The second and third rows compute the same values and leave the same bytes in the same
buffer. They differed by 167,772,160 FLOPs -- 40 x 4,194,304 elements at rate 1.0, one
unit per element written -- entirely on the strength of which spelling was used.

The degenerate forms are sharper, because there the copy is the whole call:

| call | elements moved | FLOPs billed before | `fnp.copyto` for the same bytes |
| --- | ---: | ---: | ---: |
| `einsum("ij->ij", src, out=dest)`, 2048x2048 f32 | 4,194,304 | 0 | 4,194,304 |
| `einsum("ij->ij", src, out=dest)`, 2048x2048 f64 | 4,194,304 | 0 | 8,388,608 |
| `einsum("ij->ji", src, out=dest)`, 2048x2048 f32 | 4,194,304 | 0 | 4,194,304 |

The destination is correctly and completely written in every one of those -- asserted,
not assumed. The transposing form took 6.19 ms of real memory traffic for zero FLOPs.

Without a destination the same subscripts are correctly free: numpy returns a view,
nothing is materialised, and weight 0 is right. It is the destination that turns the
view into a write.

### The wall time

`_call_numpy` exists so that every numpy call inside a counted-op wrapper reports its
duration as backend time; its docstring says all such calls MUST go through it. This
copy did not. Being outside the contraction's `deduct` block did not send its wall to
residual either, which was the first thing checked here: the enclosing
`@_counted_wrapper` bills its entire non-backend remainder to
`flopscope_overhead_time_s` (`src/flopscope/_budget.py`, `wrapper_own_overhead`). Both
`flopscope_backend_time_s` and `flopscope_overhead_time_s` are subtracted when
`residual_wall_time_s` is formed, so the write appeared in no meter a caller reads.

Isolated on a 4096x4096 float64 destination, where the contraction is an identity and
the copy is essentially the entire call (mean of 5 runs, order-controlled):

| | wall | backend | overhead | residual | FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| before | 4.88 ms | 0.01 ms | 4.85 ms | 0.01 ms | 0 |
| after | 4.49 ms | 4.25 ms | 0.23 ms | 0.01 ms | 33,554,432 |

The wall does not move: no work is added or removed, only the bucket changes.

### Line by line, and what is not the cause

The einsum wrapper was instrumented with a per-statement timer and driven over
1,536 calls on 256x256 float32, then swept across sizes. Per call, in microseconds:

| line | 64 wide | 256 wide | 1024 wide | 2048 wide | scales with |
| --- | ---: | ---: | ---: | ---: | --- |
| contraction through `_call_numpy` (backend) | 7.20 | 28.77 | 991.21 | 6600.11 | data |
| destination copy | 1.09 | 4.30 | 108.79 | 496.78 | data |
| `_resolve_cost_and_output_symmetry` | 7.35 | 6.26 | 22.86 | 32.31 | per call |
| `deduct` enter/exit bookkeeping | 6.70 | 5.61 | 21.56 | 30.51 | per call |
| `_execute_pairwise`, outside `_call_numpy` | 2.10 | 1.81 | 7.07 | 11.19 | per step |
| `note_write` | 1.46 | 1.31 | 6.97 | 11.85 | per call |

The destination copy is the only line in the wrapper whose cost tracks the caller's
data: it grows 456-fold across a 1024-fold growth in elements, while every other
non-backend line stays within a factor of five. That is what makes it the participant's
work rather than flopscope's.

`_execute_pairwise` is not the leak, and this was the more plausible of the two
candidates going in. The contraction itself already runs through `_call_numpy` and lands
in backend; what remains outside it in that function is list pops, a `zip` and
zero-copy view casts -- 2.0 us per call at 64 wide, 11.2 us at 2048 wide, fixed per step
rather than per element (a two-step path adds about 1.6 us over a one-step path at the
same size). That is precisely the flopscope-internal work `_call_numpy`'s own docstring
assigns to overhead, so it is correctly bucketed and this change does not touch it.

`#160` is not the defect. Routing the pairwise steps through numpy's optimized path is a
correct performance fix, and it is why the contraction line above sits in backend at
all. What it changed is that choosing this spelling stopped being slow, which made an
accounting defect that predates it worth reaching for. Reverting it would hide the
defect behind a performance penalty, not fix it.

## The invariant

The price of a computation must not depend on which of two equivalent spellings the
caller used. Concretely, for the same operands and the same destination:

    einsum(subs, *operands, out=dest)  ==  einsum(subs, *operands) + copyto(dest, result)

and the destination write, being a numpy call inside a counted-op wrapper, reports its
duration as backend time like every other one.

## The fix

`src/flopscope/_einsum.py`, 33 lines net. The copy is charged as `copyto`, with
`copyto`'s own cost (`dest.size`) and its own dtypes tuple (`source.dtype, dest.dtype`)
rather than the contraction's, and it runs through `_call_numpy` inside that `deduct`.

Using `copyto`'s own resolution rather than folding `numel` into the contraction's
`flop_cost` is what makes the two spellings agree on complex operands as well: a
contraction's complex factor is computed exactly per call, a copy's is 2, and folding
would have priced the write at the contraction's factor.

`_call_numpy` records the write itself -- `np.copyto` is in `_MUTATES_FIRST_ARG` -- so
the hand-rolled `note_write` this replaces is no longer needed. The write epoch still
bumps exactly once per call, measured, and a validated symmetry tag observing the
destination's buffer, or an untagged alias of it, is still voided by the write.

Forwarding `out=` to numpy instead, so that no second copy exists at all, was
considered and not taken. It would be the better end state -- it removes the work rather
than pricing it -- but it is not a small change on the pairwise path, where only the
final step could write the destination, and it moves the symmetry check to after the
destination has already been overwritten, so a `SymmetryError` would leave the caller's
buffer clobbered where today it is untouched. That is a behaviour change worth making
deliberately and separately, not as a side effect of a billing fix.

## Blast radius

The added charge is exactly `numel(dest)` at the copy rate, so the relative increase is
set by the contraction's arithmetic intensity -- FLOPs produced per output element
written. Measured, 256 wide, float32:

| contraction | before | after | increase |
| --- | ---: | ---: | ---: |
| `ij,jk->ik` | 33,488,896 | 33,554,432 | +0.196% |
| `ij,j->i` | 130,816 | 131,072 | +0.196% |
| `ij,jk,kl->il` | 66,977,792 | 67,043,328 | +0.098% |
| `bij,bjk->bik`, 32x64x64 | 16,646,144 | 16,777,216 | +0.787% |
| `ijk,jl->ilk`, 64 | 33,292,288 | 33,554,432 | +0.787% |
| `i,j->ij` | 65,536 | 131,072 | +100% |
| `ij->ji` | 0 | 65,536 | from zero |
| `ij->ij` | 0 | 65,536 | from zero |

An outer product pays 100% because it writes exactly as many elements as it computes;
that is the rule working, not an overcharge. The identity and transpose forms go from
zero to `numel`, which is the whole point.

On a realistic workload -- 6,144 `einsum("ij,jk->ik", out=)` calls over 256x256 float32,
propagating twelve 32-layer networks -- the total moves from 205,755,777,024 to
206,158,430,208 FLOPs, +0.196%. The wall-time shift on that workload (the copy is about
4.3 us of a 65 us call) sits inside this machine's run-to-run spread and is not
separately resolvable there; the isolated measurement above is where it is
unambiguous.

Unaffected: `einsum` without a destination, and `matmul`, `dot`, `inner`, `outer`,
`vecdot`, `matvec` and `vecmat`, all of which hand `out=` to numpy and write the
caller's buffer once, inside the kernel already paid for. `flops_used` never decreases
for any call, `out=(d,)` still bills exactly what `out=d` bills, and a refused
destination still costs zero.

## Tests

`tests/test_einsum_out_destination_billing.py`, eight cases. Six fail on the unfixed
source. The pins are exact-equality FLOP counts rather than timing comparisons, because
the invariant is about price, not speed, and because the timing-inequality form flakes
in this suite under parallel load. The two cases that pass on both sides are the
directions this change must not disturb: the destination-free identity stays a free
view, and a wider destination still bills the wider rate.

The bucket half is pinned as a mechanism rather than as one clock beating another -- an
op record must exist for the write, priced at what `fnp.copyto` charges for that write,
and carrying non-zero backend duration, which happens only if the copy went through
`_call_numpy` while that op's timer was live. Restoring a bare `_np.copyto`, inside the
block or outside it, leaves the record at zero backend; dropping the charge leaves no
record at all.

`tests/test_nonnumeric_out_billing.py` asserted that `einsum(out=)` bills exactly what
`einsum()` bills, which is the behaviour corrected here. What that file exists to pin is
that a non-numeric destination cannot discount the arithmetic's rate, so it now measures
the destination write and subtracts it, keeping the rate assertion exact instead of
letting it drift with the write's price.

`docs/reference/cost-model.md` gains a paragraph under the zero-work contraction rule,
which said `einsum('ij->ji', z)` charges 0 without distinguishing the viewing form from
the materialising one.

## Suite

Three consecutive full runs each, same environment.

    before  abd9570   6869 passed, 63 skipped, 1 xfailed, 0 failed
    after   5b90d64   6877 passed, 63 skipped, 1 xfailed, 0 failed

The eight added tests account for the difference. `ruff check`, `ruff format --check`
and `pyright` are clean on the changed files.

## Validation

Run with the project's own exclusions from `pyproject.toml`, minus `-n auto` (no `pytest-xdist`
here), and minus four files needing `scipy` or `hypothesis`, which are not installed on this
machine:

| | failed | passed |
| --- | ---: | ---: |
| `f1a38be` (main) | 2 | 7012 |
| this branch | 2 | 7020 |

The same two fail on both -- `test_ops_json_sync.py::test_ops_json_in_sync_with_registry` and
`test_random_symmetric_docs.py::test_random_symmetric_record_uses_real_docstring` -- so they
predate this change. The eight extra passes are the new tests.

The new file and the extended one pass on their own (29), as do the three suites nearest the
change together with `test_einsum_out_casting_parity.py` from #168 (100).
