# Unary ufunc computation and destination billing (B6)

## Intent

Close B6 in the shared single-output and multi-output unary ufunc wrappers. Calls selecting the same NumPy loop must pay the same price whether selected by `dtype=`, `signature=`, or `sig=`; a destination remains an independent billing participant. Enforce the existing published dtype policy rather than introduce new weights or operation formulas.

## Root cause and measured baseline

On the retained release branch, four float32 values passed to `exp(dtype='d')` bill 128 under production weights. `signature='d->d'` bills 64, `sig='D->D'` bills 64 despite computing complex128, and `dtype='f', out=float64_buffer` bills 64. A forced float64 loop with float32 destination also bills 64. Multi-output `modf(signature='d->dd')` and `frexp(signature='d->di')` exhibit the same omission (8 unit FLOPs rather than the width-adjusted price).

The unary wrappers skip resolved-loop billing for explicit signatures and omit destination participants when `dtype=` is explicit. The retained branch already has constrained-loop resolution and one-time keyword freezing for binary ufuncs; reuse these helpers, avoiding a second implementation of NumPy signature semantics.

## Required behavior

- Freeze dtype-like values, signature tuples and aliases once before operand snapshots; forward exactly the frozen constraints to NumPy. Preserve NumPy keyword-conflict errors, including presence of `dtype=None`.
- Resolve every constrained computation input/output slot with NumPy `ufunc.resolve_dtypes`, passing explicit casting. A forced narrow loop may bill narrowly even for wider source operands. Unconstrained calls retain their existing operand floor.
- Compute billing must not be inferred from the returned array: narrow destinations cannot hide a wider loop.
- Inspect each destination independently of constraints using base ndarray descriptors. Single-output destinations retain the existing unary complex-destination behavior; changing their arithmetic-versus-store-only complex classification is outside B6. Forced complex loops must use the operation's complex factor even when the destination is real under unsafe casting.
- Multi-output destinations are compared with the resolved loop's natural per-slot output dtypes. Naming frexp's natural int32 exponent remains price-neutral; widening a slot can raise the rate. Preserve the existing nout multiplier.
- Unsupported loops and conflicting constraints reject without charge. Preserve existing Flopscope nonnumeric/complex-illegal error precedence where applicable. Numerical execution remains inside the existing timed boundary.
- Support NumPy 2.0–2.4 and Python >=3.10 without dependency changes. Preserve values, result shape/dtype, destination identity, warnings and accepted signature spellings.
- Probe actual client/server reachability; add wire regressions for supported forms and document any protocol restriction rather than assume local exposure.

## Non-goals

No weight tuning; no output-shape repricing (#288), binary behavior changes, generalized mutation audit, external PR dependency, or additional operations outside B6. Retained HEAD is the baseline. Separate open PRs are references only.

## Validation

Exact production prices: four-element exp float64 128, complex128 396, and explicit float32 with float64 destination 128. Unit tests also assert loop records, NumPy parity, aliases/tuple signatures, forced narrowing, predicates, multi-output natural/wider/partial destinations, conflict/invalid-call zero charge, and single-resolution stateful dtype-like arguments. Focused tests and existing dtype/complex/ufunc suites, static checks, sync/generated-doc verification, and actual wire tests precede delivery.

## Implementation decisions

Reuse `_freeze_binary_ufunc_type_kwargs` without renaming its existing callers, and `_ufunc_loop_signature`. Extract only the small shared unary billing resolution if it reduces single/multi-output drift. Document the unary application of the existing constraint/destination policy. No policy question remains beyond implementation review.
