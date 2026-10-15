# Unary Ufunc Dtype Billing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Bill unary forced loops and destinations consistently with existing dtype policy.

**Architecture:** Reuse the retained branch's NumPy constraint freezer and resolver in both unary factories. Keep NumPy execution and wrapping intact; resolve compute and destination independently.

**Tech Stack:** Python >=3.10; NumPy 2.0–2.4; pytest; Ruff; Pyright.

**Spec:** `docs/superpowers/specs/2026-10-15-unary-ufunc-dtype-billing-design.md`

## Global Constraints

- Support NumPy 2.0–2.4 and Python >=3.10 without dependency changes.
- Retained HEAD is the baseline. Separate open PRs are references only.
- No weight tuning, output-shape repricing, or unary complex-store policy change.
- Preserve execution timing, NumPy parity and natural multi-output destination neutrality.

## Review Focus

- A forced complex loop narrowed into real out must retain complex arithmetic pricing.
- A tuple signature containing a stateful dtype object must resolve exactly once.
- Boolean-output predicates must bill their numeric input loop rather than bool output.
- Frexp's natural exponent destination must not introduce float64 promotion.
- NumPy-invalid conflicting aliases, signature slots and casting must cost zero.

### Task 1: Unary forced-loop regression and implementation

**Files:** `tests/test_unary_ufunc_constraints.py`; `src/flopscope/_pointwise.py`.

**Interfaces:** Consume `_freeze_binary_ufunc_type_kwargs(kwargs)`, `_ufunc_loop_signature(ufunc, *dtypes, signature=..., casting=..., strict=...)`, existing dtype/store helpers. Preserve public unary signatures.

- [x] Add production exp assertions (128/396/128) and parameterized NumPy parity for sig/signature/tuple forms, forced narrowing, predicates, casting, stateful constraints and rejected forms.
- [x] Run new tests with root venv and worktree PYTHONPATH; confirm billing failures on retained baseline.
- [x] Resolve constrained single-output loops and independently include store dtype; preserve unforced operand floors and explicit complex loop factor.
- [x] Add and observe failures for modf/frexp forced-loop and destination cases, then apply the same resolution to multi-output wrappers using resolved output slots.
- [x] Run focused and existing dtype/complex/ufunc tests; inspect any changed behavior against spec.

### Task 2: Wire, documentation and delivery

**Files:** applicable real-server test module; `docs/reference/cost-model.md`; `CHANGELOG.md`; docstrings and generated API index if changed.

**Interfaces:** Existing client/server fixtures; no protocol change unless necessary for already accepted forms.

- [x] Probe and add actual wire assertions for accepted constraint/destination spellings and exact charged delta.
- [x] Explain unary constrained-loop and independent destination pricing in docs/changelog; update generated index if docstrings change.
- [x] Run Ruff/Pyright, timing guards, sync/docs verification and appropriate parity/integration tests. Parent runs combined release checks.
- [x] Inspect diff and record precise checks/limitations. Commit a conventional billing fix, push isolated branch, open individual issue and PR against retained release baseline, and report links.

## Execution record

- User explicitly approved the B6 spec and plan: “Approve and implement B6.”
- Initial single/multi-output regression run: 24 failed, 17 passed on baseline; fixed resolver then 41 passed. Destination-cast rejection tests separately failed before preflight implementation, then passed.
- Wire regressions run against original baseline: all 7 failed; fixed implementation: all 7 passed. These establish actual production protocol reachability.
- Ruling: validate destination casts with the exact resolved loop via NumPy metadata, rather than execute first and charge afterwards. This preserves advance budget enforcement and prevents invalid casting calls consuming budget.
- Ruling: keep existing unary complex-destination classification; extending the binary store-only complex factor is a separate policy change, outside B6.
- Shared root environment is used with explicit worktree PYTHONPATH, never uv sync. Isolated NumPy boundary environments do not change it.

- Final clean-tree default suite: 10,553 passed, 125 skipped, 2 xfailed. Full real client/server integration: 48 passed. Ruff check/format clean; Pyright 0 errors and 4 existing warnings. Generated API verification and client synchronization pass.
- Parent performed fresh review of the complete implementation and found no blocker.
- Issue: https://github.com/AIcrowd/flopscope/issues/304 ; PR: https://github.com/AIcrowd/flopscope/pull/310 . No merge or release performed.
