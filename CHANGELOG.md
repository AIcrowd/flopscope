# Changelog

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

## Unreleased

### Fix

- **flopscope timing split**: `BudgetContext` now reports a precise client/server
  decomposition — `backend` = pure server numpy kernel, `overhead` = all flopscope
  machinery (client dispatch + wire + server marshaling), `residual` = the
  participant's own Python only. The server reports `compute_time` as kernel-only;
  the client times the full op dispatch (including result reconstruction). Fixes
  the prior all-zero (and the later round-trip-approximation) behavior on the
  server-backed path.
- **flopscope cost queries**: `flops.einsum_cost` / `flops.svd_cost` now count their
  server round-trip as `overhead` (via `@timed_dispatch`) instead of leaking it into
  the billed `residual` bucket when a participant calls them inside a budget context.
  The implicit global-default budget-open round-trip is likewise wrapped defensively.
- **flopscope flops_used cache**: `BudgetContext.__exit__` now refreshes `flops_used`
  from the server's `budget_breakdown` in the `budget_close` response (it is nested at
  `result.budget_breakdown`, one level deeper than `budget_status`). A plain `with`
  block now reports the authoritative server count — and `render_budget_summary()`
  shows it — without the prior `summary()`-call workaround.

### Added

- **budget summary timing**: `render_budget_summary()` now shows a session
  wall-time breakdown (backend / overhead / residual) beneath FLOP usage, in both
  the Rich and plain-text renderers, surfacing the per-context timing split.
