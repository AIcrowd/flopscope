# Changelog

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
