# Makefile — mirrors .github/workflows/ci.yml so local runs match remote CI.
#
#   make ci          run the full pipeline (lint → test → docs)
#   make lint        ruff check + format check
#   make typecheck   pyright in standard mode
#   make test        pytest with coverage
#   make docs-build  generate API docs, verify, then build the site
#   make docs-serve  live-preview the docs locally
#
# Prerequisites:  uv sync --all-extras   (installs dev + docs extras)

SHELL := /bin/bash
UV    := uv run

# ---------------------------------------------------------------------------
# Composite targets
# ---------------------------------------------------------------------------
.PHONY: ci
ci: lint check-lock lint-commits typecheck test test-numpy-compat check-sync check-sync-versions docs-build  ## Run the full CI pipeline locally

# ---------------------------------------------------------------------------
# Lint  (mirrors: CI → lint job)
# ---------------------------------------------------------------------------
.PHONY: lint
lint:  ## Ruff lint + format check
	$(UV) ruff check .
	$(UV) ruff format --check .

.PHONY: lint-commits
lint-commits:  ## Conventional-commit check on PR commits (origin/main..HEAD)
	@if ! git rev-parse --verify origin/main >/dev/null 2>&1; then \
		echo "lint-commits: origin/main not found; run 'git fetch origin main' first"; exit 1; \
	fi
	$(UV) gitlint --ignore-stdin --commits origin/main..HEAD

.PHONY: fmt
fmt:  ## Auto-fix lint and format issues
	$(UV) ruff check --fix .
	$(UV) ruff format .

.PHONY: typecheck
typecheck:  ## Pyright (standard mode) over src/ and tests/
	$(UV) pyright src/flopscope tests

# ---------------------------------------------------------------------------
# Test  (mirrors: CI → test job)
# ---------------------------------------------------------------------------
.PHONY: test
test:  ## Run pytest with coverage (fails if < 90%); generates API docs first if missing
	@test -f website/.generated/op-doc-imports.ts || $(UV) python scripts/generate_api_docs.py
	$(UV) pytest --cov=flopscope --cov-fail-under=90

.PHONY: test-numpy-compat
test-numpy-compat:  ## Run NumPy's own tests against flopscope
	$(UV) pytest tests/numpy_compat/ -n auto -q \
		--pyargs numpy._core.tests.test_umath \
		          numpy._core.tests.test_ufunc \
		          numpy._core.tests.test_numeric \
		          numpy.linalg.tests.test_linalg \
		          numpy.fft.tests.test_pocketfft \
		          numpy.fft.tests.test_helper \
		          numpy.polynomial.tests.test_polynomial \
		          numpy.random.tests.test_random

.PHONY: test-client-parity
test-client-parity:  ## GATE: targeted client-parity tests (must be green)
	$(UV) pytest tests/client_compat/ \
		--ignore=tests/client_compat/methods \
		--ignore=tests/client_compat/test_numpy_function_classes.py \
		-n auto -q
	$(UV) pytest tests/client_compat/methods/ \
		--ignore=tests/client_compat/methods/test_numpy_classes.py \
		-n auto -q

.PHONY: test-client-parity-measure
test-client-parity-measure:  ## MEASUREMENT (non-blocking): numpy's own suite vs the client
	-$(UV) pytest tests/client_compat/test_numpy_function_classes.py -n auto -q
	-$(UV) pytest tests/client_compat/methods/test_numpy_classes.py -n auto -q

.PHONY: client-parity-inventory
client-parity-inventory:  ## Run the client-parity harness and emit the categorized failure inventory
	$(UV) python scripts/client_parity_inventory.py

# ---------------------------------------------------------------------------
# Docs  (mirrors: CI → docs job)
# ---------------------------------------------------------------------------
.PHONY: docs-build
docs-build:  ## Generate API data and build website
	$(UV) python scripts/generate_api_docs.py
	$(UV) python scripts/generate_api_docs.py --verify
	cd website && (npm run build || (npx --yes next build && node scripts/generate-llmstxt.mjs))
	cd website && npm run check:gh-pages

.PHONY: docs-serve
docs-serve:  ## Generate API data, then serve docs locally with live reload
	$(UV) python scripts/generate_api_docs.py
	cd website && npm run dev

.PHONY: docs-deploy
docs-deploy:  ## Docs deploy is handled by CI on push to main
	@echo "Docs deploy is handled by CI on push to main"

# ---------------------------------------------------------------------------
# Client-Server Sync
# ---------------------------------------------------------------------------
.PHONY: check-sync
check-sync:  ## Verify client is in sync with core library
	$(UV) python scripts/sync_client.py --check
	$(UV) pytest tests/test_client_server_parity.py tests/test_serialization_parity.py -v

.PHONY: check-sync-versions
check-sync-versions:  ## Verify all package versions are in lockstep
	$(UV) python scripts/check_version_sync.py

# Note: bare `uv lock --check` (NOT `$(UV)` = `uv run`, and NOT `uv sync`).
# `uv sync` would silently refresh a stale lock instead of failing — which is
# how the lockfiles drifted a release behind unnoticed. `--check` only reports.
.PHONY: check-lock
check-lock:  ## Verify all three uv.lock files are in sync with their pyproject.toml
	uv lock --check
	uv lock --check --directory flopscope-server
	uv lock --check --directory flopscope-client

.PHONY: relock
relock:  ## Refresh all three uv.lock files (run after a version bump; fixes check-lock)
	uv lock
	uv lock --directory flopscope-server
	uv lock --directory flopscope-client

.PHONY: sync-client
sync-client:  ## Regenerate client files from core library
	$(UV) python scripts/sync_client.py

.PHONY: test-integration
test-integration:  ## Run client-server integration tests
	cd flopscope-client && $(UV) pytest tests/test_full_integration.py -v --tb=short

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
.PHONY: install
install:  ## Install all deps (dev + docs) and set up git hooks
	uv sync --all-extras
	git config core.hooksPath .githooks

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
.PHONY: bench-accumulation
bench-accumulation:  ## Cold + warm latency benchmark for einsum_accumulation_cost
	$(UV) python benchmarks/accumulation/bench_cost_compute.py

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
