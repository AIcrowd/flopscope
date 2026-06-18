"""Validate that default_weights.json and the migrated docs cover all counted operations.

This test ensures that every non-free, non-blacklisted operation in the
registry either:
  1. Has a direct weight in default_weights.json, OR
  2. Is a known alias of a weighted operation, OR
  3. Is listed in a benchmark module's ops list (weights pending generation), OR
  4. Falls into a documented exclusion category (bitwise, complex, etc.)

It also validates that the generated API reference data still exposes every
weighted operation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from flopscope._registry import REGISTRY

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "src" / "flopscope" / "data" / "weights.json"
DEFAULT_WEIGHTS_PATH = ROOT / "src" / "flopscope" / "data" / "default_weights.json"
OPS_INDEX_PATH = ROOT / "website" / "public" / "ops.json"

# Ensure benchmarks package is importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks._bitwise import BITWISE_OPS  # noqa: E402
from benchmarks._complex import COMPLEX_OPS  # noqa: E402
from benchmarks._contractions import CONTRACTION_OPS  # noqa: E402
from benchmarks._fft import FFT_OPS  # noqa: E402
from benchmarks._linalg import LINALG_OPS  # noqa: E402
from benchmarks._linalg_delegates import LINALG_DELEGATE_OPS  # noqa: E402
from benchmarks._misc import MISC_OPS  # noqa: E402
from benchmarks._pointwise import BINARY_OPS, SPECIAL_OPS, UNARY_OPS  # noqa: E402
from benchmarks._polynomial import POLYNOMIAL_OPS  # noqa: E402
from benchmarks._random import RANDOM_OPS  # noqa: E402
from benchmarks._reductions import REDUCTION_OPS  # noqa: E402
from benchmarks._sorting import SORTING_OPS  # noqa: E402
from benchmarks._window import WINDOW_OPS  # noqa: E402

# ---------------------------------------------------------------------------
# Ops covered by benchmark modules (weights pending bare-metal generation).
# ---------------------------------------------------------------------------
BENCHMARKED_OPS: frozenset[str] = frozenset(
    set(SORTING_OPS)
    | set(CONTRACTION_OPS)
    | set(MISC_OPS)
    | set(WINDOW_OPS)
    | set(UNARY_OPS)
    | set(BINARY_OPS)
    | set(SPECIAL_OPS)
    | set(REDUCTION_OPS)
    | set(RANDOM_OPS)
    | set(POLYNOMIAL_OPS)
    | set(FFT_OPS)
    | set(LINALG_OPS)
    | set(BITWISE_OPS)
    | set(COMPLEX_OPS)
    | set(LINALG_DELEGATE_OPS)
)

# ---------------------------------------------------------------------------
# Alias map: operation name -> canonical name whose weight it should inherit.
# These are NumPy aliases, deprecated names, or nan-aware variants that share
# the same underlying FP instruction profile as their canonical form.
# ---------------------------------------------------------------------------
ALIAS_MAP: dict[str, str] = {
    # NumPy 2.x short aliases
    "acos": "arccos",
    "acosh": "arccosh",
    "asin": "arcsin",
    "asinh": "arcsinh",
    "atan": "arctan",
    "atan2": "arctan2",
    "atanh": "arctanh",
    "pow": "power",
    # Legacy aliases
    "absolute": "abs",
    "amax": "max",
    "amin": "min",
    "around": "rint",
    "fix": "trunc",
    "round": "rint",
    # nan-aware variants (same FP profile as non-nan version)
    "nanargmax": "argmax",
    "nanargmin": "argmin",
    "nancumprod": "cumprod",
    "nancumsum": "cumsum",
    # NumPy 2.x array API names
    "cumulative_prod": "cumprod",
    "cumulative_sum": "cumsum",
    # Reduction aliases
    "ptp": "max",
    # Binary alias
    "divmod": "floor_divide",
    # Deprecated name
    "trapz": "trapezoid",
    # Random aliases
    "random.ranf": "random.random_sample",
    "random.sample": "random.random_sample",
}

# ---------------------------------------------------------------------------
# Exclusion categories: operations that don't produce meaningful float64
# perf weights because they operate on integers, complex types, or have
# shape-dependent cost that can't be captured by a single scalar weight.
# ---------------------------------------------------------------------------

# No excluded ops remaining — all counted ops are now in default_weights.json.
ALL_EXCLUDED: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# API-name canonicalization (derived, not hand-maintained)
# ---------------------------------------------------------------------------
# A weighted op may be listed in ops.json under a *canonical* name (e.g. weights
# has `abs`/`row_stack`, but the generated reference lists `absolute`/`vstack`).
# Rather than hand-maintain that mapping — which silently drifted once, when the
# generator started treating `row_stack` as a deprecated alias of `vstack` and the
# hardcoded list lacked the entry — derive it from the generator's OWN resolver,
# `scripts/generate_api_docs.py::load_alias_map`, so the two can never diverge.


def _load_alias_map() -> dict[str, str]:
    """Return the generator's alias map (op name -> canonical name).

    Imported from ``scripts/generate_api_docs.py`` the same way
    ``test_cost_model_coverage`` does — importlib plus ``sys.modules`` registration
    so the module's ``@dataclass`` field annotations resolve on Python 3.14.
    """
    spec = importlib.util.spec_from_file_location(
        "_gen_api_docs", ROOT / "scripts" / "generate_api_docs.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.load_alias_map(REGISTRY)


def _canonical_api_names(alias_map: dict[str, str]) -> dict[str, str]:
    """Flatten each alias chain to its TERMINAL canonical — the name under which the
    generator emits the op in ops.json (e.g. ``around -> round -> rint`` collapses
    to ``around -> rint``). The ``seen`` set bounds any cycle.
    """
    resolved: dict[str, str] = {}
    for name in alias_map:
        seen: set[str] = set()
        cur = name
        while cur in alias_map and cur not in seen:
            seen.add(cur)
            cur = alias_map[cur]
        resolved[name] = cur
    return resolved


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def weights() -> dict[str, float]:
    """Load the BILLED weights — default_weights.json is the source of truth."""
    assert DEFAULT_WEIGHTS_PATH.exists(), (
        f"default_weights.json not found at {DEFAULT_WEIGHTS_PATH}"
    )
    data = json.loads(DEFAULT_WEIGHTS_PATH.read_text())
    assert "weights" in data, "default_weights.json missing 'weights' key"
    return data["weights"]


@pytest.fixture(scope="module")
def api_operations() -> dict[str, dict]:
    """Load the generated API reference operation index."""
    assert OPS_INDEX_PATH.exists(), f"ops.json not found at {OPS_INDEX_PATH}"
    data = json.loads(OPS_INDEX_PATH.read_text())
    assert "operations" in data, "ops.json missing 'operations' key"
    return {entry["name"]: entry for entry in data["operations"]}


@pytest.fixture(scope="module")
def api_name_aliases() -> dict[str, str]:
    """Weighted-op name -> the canonical name it appears under in ops.json.

    Derived from the generator's own ``load_alias_map`` (not a hand-maintained
    list) so it cannot silently drift from ops.json — which is exactly how a
    missing ``row_stack -> vstack`` entry once broke this test.
    """
    return _canonical_api_names(_load_alias_map())


@pytest.fixture(scope="module")
def counted_ops() -> set[str]:
    """Module-level counted operations that need direct weight entries.

    Excludes method-level entries on Generator/RandomState — those inherit
    cost from their cost_formula at dispatch time and don't need their own
    rows in default_weights.json.
    """
    return {
        name
        for name, entry in REGISTRY.items()
        if entry["category"]
        not in (
            "free",
            "blacklisted",
            "free_random_method",
            "counted_random_method",
        )
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWeightsJsonCoverage:
    """Verify default_weights.json covers all benchmarkable operations."""

    def test_all_weighted_ops_are_in_registry(self, weights: dict[str, float]):
        """Every op in default_weights.json should exist in the registry."""
        extra = set(weights) - set(REGISTRY)
        assert not extra, (
            f"weights.json contains {len(extra)} ops not in registry: "
            f"{sorted(extra)[:10]}"
        )

    def test_all_weights_are_positive(self, weights: dict[str, float]):
        """Weights must be non-negative (0 is allowed for near-free ops)."""
        negative = {k: v for k, v in weights.items() if v < 0}
        assert not negative, f"Negative weights: {negative}"

    def test_baseline_add_is_present(self, weights: dict[str, float]):
        """The baseline op (add) must be weighted."""
        assert "add" in weights, "Baseline 'add' missing from weights"

    def test_counted_ops_are_covered_or_excluded(
        self,
        weights: dict[str, float],
        counted_ops: set[str],
    ):
        """Every counted op must be weighted, in a benchmark module, aliased, or excluded."""
        covered = set(weights)
        # An alias is accounted for if its target is either already weighted
        # or listed in a benchmark module's ops list.
        alias_targets = set(weights) | BENCHMARKED_OPS
        aliased = {name for name in ALIAS_MAP if ALIAS_MAP[name] in alias_targets}
        accounted_for = covered | aliased | BENCHMARKED_OPS | ALL_EXCLUDED

        uncovered = sorted(counted_ops - accounted_for)
        assert not uncovered, (
            f"{len(uncovered)} counted ops are not weighted, benchmarked, aliased, "
            f"or excluded:\n"
            + "\n".join(
                f"  {name:30s} ({REGISTRY[name]['category']})"
                for name in uncovered[:20]
            )
            + ("\n  ..." if len(uncovered) > 20 else "")
        )

    def test_alias_targets_exist(self, weights: dict[str, float]):
        """Every alias target in ALIAS_MAP should be weighted or in a benchmark module."""
        alias_targets = set(weights) | BENCHMARKED_OPS
        missing_targets = {
            f"{alias} -> {target}"
            for alias, target in ALIAS_MAP.items()
            if target not in alias_targets
        }
        assert not missing_targets, (
            f"Alias targets missing from weights and benchmark modules: "
            f"{sorted(missing_targets)}"
        )

    def test_excluded_ops_are_actually_in_registry(self, counted_ops: set[str]):
        """Excluded ops should actually be in the registry (catch stale exclusions)."""
        stale = sorted(ALL_EXCLUDED - counted_ops)
        assert not stale, f"Excluded ops not in registry (stale?): {stale}"

    def test_no_double_accounting(self, weights: dict[str, float]):
        """Ops should not be both excluded AND weighted/benchmarked."""
        double = sorted(ALL_EXCLUDED & (set(weights) | BENCHMARKED_OPS))
        # Allow ops that are in BENCHMARKED_OPS to also be excluded --
        # this can happen during transition.  Only flag ops that are in
        # weights.json AND excluded.
        truly_double = sorted(ALL_EXCLUDED & set(weights))
        assert not truly_double, (
            f"Ops in both weights.json and exclusion sets: {truly_double}"
        )

    def test_benchmarked_ops_exist_in_registry(self):
        """Every op listed in a benchmark module should be in the registry."""
        missing = sorted(BENCHMARKED_OPS - set(REGISTRY))
        assert not missing, f"Benchmark ops not in registry: {missing}"

    def test_meta_has_required_fields(self):
        """weights.json metadata should have hardware/software/config."""
        data = json.loads(WEIGHTS_PATH.read_text())
        meta = data["meta"]
        assert "hardware" in meta
        assert "software" in meta
        assert "benchmark_config" in meta
        assert meta["benchmark_config"]["measurement_mode"] in ("perf", "timing")


class TestDocsWeightCoverage:
    """Verify the API reference exposes weighted operations."""

    def test_all_weighted_ops_appear_in_api_reference(
        self,
        weights: dict[str, float],
        api_operations: dict[str, dict],
        api_name_aliases: dict[str, str],
    ):
        """Every weighted op should appear in the generated API reference data."""
        canonical_weights = {
            api_name_aliases.get(name, name): weight for name, weight in weights.items()
        }
        missing = [
            name for name in sorted(canonical_weights) if name not in api_operations
        ]
        assert not missing, (
            f"{len(missing)} weighted ops missing from API reference data:\n"
            + "\n".join(f"  {name}" for name in missing[:20])
        )

        mismatched = {
            name: (expected, api_operations[name]["weight"])
            for name, expected in canonical_weights.items()
            if api_operations[name]["weight"] != expected
        }
        assert not mismatched, (
            "ops.json weights diverge from default_weights.json for:\n"
            + "\n".join(
                f"  {name}: default_weights.json={expected}, ops.json={actual}"
                for name, (expected, actual) in sorted(mismatched.items())[:20]
            )
        )
