"""Tests for the method-level → module-level weight aliasing (issue #18 follow-up).

Note: tests/conftest.py has an autouse `reset_weights()` fixture that runs
before every test, so we explicitly call `load_weights()` to load empirical
defaults when needed.
"""

from flopscope._registry import REGISTRY
from flopscope._weights import get_weight, load_weights


class TestRuntimeAliasing:
    def test_direct_same_name_match(self):
        # Generator.normal → random.normal (16.0 in default_weights.json)
        load_weights()
        assert get_weight("random.Generator.normal") == get_weight("random.normal")

    def test_rename_integers_to_randint(self):
        load_weights()
        assert get_weight("random.Generator.integers") == get_weight("random.randint")

    def test_rename_random_to_random_sample(self):
        load_weights()
        assert get_weight("random.Generator.random") == get_weight(
            "random.random_sample"
        )

    def test_randomstate_methods_alias_directly(self):
        load_weights()
        assert get_weight("random.RandomState.randn") == get_weight("random.randn")
        assert get_weight("random.RandomState.normal") == get_weight("random.normal")
        assert get_weight("random.RandomState.exponential") == get_weight(
            "random.exponential"
        )

    def test_explicit_override_wins_over_alias(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text(
            '{"weights": {"random.normal": 16.0, "random.Generator.normal": 99.0}}'
        )
        load_weights(str(path), use_packaged_default=False)
        # Explicit entry wins: 99.0, not aliased 16.0
        assert get_weight("random.Generator.normal") == 99.0
        # Module-level still 16.0
        assert get_weight("random.normal") == 16.0


class TestNonRandomOpsUnchanged:
    """Make sure aliasing only triggers for random method-level ops."""

    def test_top_level_op_unchanged(self):
        load_weights()
        # `add` should still resolve normally
        assert get_weight("add") == 1.0  # default empirical

    def test_unknown_op_returns_default(self):
        load_weights()
        # Unknown op falls through to 1.0
        assert get_weight("totally.unknown.op") == 1.0


class TestCoverage:
    """Property test: every counted_random_method entry has a non-trivial weight
    via either explicit entry, alias fallback, or rename map."""

    def test_every_counted_random_method_has_resolvable_weight(self):
        load_weights()
        # multivariate_normal is a composite op: tier factors folded into
        # flop_cost, so weight is intentionally 1.0 (not a mis-alias).
        _composite_ops = {"multivariate_normal"}
        unresolved = []
        for name, entry in REGISTRY.items():
            if entry.get("category") != "counted_random_method":
                continue
            short = name.rsplit(".", 1)[-1]
            if short in _composite_ops:
                continue
            w = get_weight(name)
            # Distribution-class samplers (transcendental) should resolve to ~16
            if (
                any(
                    tok in name
                    for tok in (
                        "normal",
                        "exponential",
                        "gamma",
                        "beta",
                        "lognormal",
                        "weibull",
                    )
                )
                and "_normal_" not in name
            ):
                if w == 1.0:
                    unresolved.append((name, w))
        assert not unresolved, (
            f"Distribution method-level ops with surprising 1.0 weight "
            f"(should alias to ~16 via fallback): {unresolved}"
        )


class TestOutlierExplicitEntries:
    """The 3 method-level ops without direct module-level analogs are pinned in JSON."""

    def test_multivariate_hypergeometric_pinned(self):
        load_weights()
        # Pinned to closest-analog (random.hypergeometric, 16.0)
        assert get_weight("random.Generator.multivariate_hypergeometric") == 16.0

    def test_permuted_pinned(self):
        load_weights()
        # Explicit access-tier entry (no module-level analog to alias from);
        # matches random.permutation's own weight, 4.0.
        assert get_weight("random.Generator.permuted") == 4.0

    def test_tomaxint_pinned(self):
        load_weights()
        # Pinned to closest-analog (random.randint, 1.0)
        assert get_weight("random.RandomState.tomaxint") == 1.0

    def test_sample_aligns_with_random_sample(self):
        load_weights()
        # random.sample is a plain uniform draw (documented alias of
        # random_sample), not a selection, so it bills at the same 1.0 tier as
        # its twins -- not the 4.0 access tier it was mistakenly given.
        assert get_weight("random.sample") == 1.0
        assert get_weight("random.sample") == get_weight("random.random_sample")
