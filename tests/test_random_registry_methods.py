"""Tests asserting the registry contains method-level entries for
random.Generator with the right shape and category coverage.
"""

import numpy as np

from flopscope._registry import REGISTRY

# Methods on numpy.random.Generator that are NOT counted (free passthrough).
GENERATOR_FREE = {"bit_generator", "spawn"}

# All other public attributes on Generator are counted samplers.
GENERATOR_COUNTED = {
    "beta",
    "binomial",
    "bytes",
    "chisquare",
    "choice",
    "dirichlet",
    "exponential",
    "f",
    "gamma",
    "geometric",
    "gumbel",
    "hypergeometric",
    "integers",
    "laplace",
    "logistic",
    "lognormal",
    "logseries",
    "multinomial",
    "multivariate_hypergeometric",
    "multivariate_normal",
    "negative_binomial",
    "noncentral_chisquare",
    "noncentral_f",
    "normal",
    "pareto",
    "permutation",
    "permuted",
    "poisson",
    "power",
    "random",
    "rayleigh",
    "shuffle",
    "standard_cauchy",
    "standard_exponential",
    "standard_gamma",
    "standard_normal",
    "standard_t",
    "triangular",
    "uniform",
    "vonmises",
    "wald",
    "weibull",
    "zipf",
}


class TestGeneratorRegistryCoverage:
    def test_every_public_method_has_a_registry_entry(self):
        public = {n for n in dir(np.random.Generator) if not n.startswith("_")}
        registered = {
            n[len("random.Generator.") :]
            for n in REGISTRY
            if n.startswith("random.Generator.")
        }
        missing = public - registered
        assert not missing, (
            f"Generator methods missing from registry: {sorted(missing)}"
        )

    def test_counted_entries_have_required_fields(self):
        for short in GENERATOR_COUNTED:
            op = f"random.Generator.{short}"
            assert op in REGISTRY, f"missing entry: {op}"
            entry = REGISTRY[op]
            assert entry["category"] == "counted_random_method", op
            assert entry["module"] == "numpy.random", op
            assert "cost_formula" in entry, op
            assert "notes" in entry, op

    def test_free_entries_have_required_fields(self):
        for short in GENERATOR_FREE:
            op = f"random.Generator.{short}"
            assert op in REGISTRY, f"missing entry: {op}"
            entry = REGISTRY[op]
            assert entry["category"] == "free_random_method", op
            assert entry["module"] == "numpy.random", op

    def test_cost_formula_is_known(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        for op, entry in REGISTRY.items():
            if entry.get("category") != "counted_random_method":
                continue
            assert entry["cost_formula"] in COST_FORMULAS, (
                f"{op}: unknown cost_formula {entry['cost_formula']!r}"
            )


class TestSpecificCostFormulaAssignments:
    def test_shuffle_uses_shape_axis(self):
        # First-principles: Fisher-Yates does shape[axis] RNG draws,
        # regardless of slice width. See issue #18 follow-up.
        # Pins the curated exact-formula strings (cost-model sheet); both
        # resolve to the _shape_axis callable in COST_FORMULAS.
        assert REGISTRY["random.Generator.shuffle"]["cost_formula"] == (
            "shape[axis] (axis defaults to 0); i.e. one RNG draw per slice along the "
            "shuffled axis, regardless of slice width"
        )
        assert REGISTRY["random.RandomState.shuffle"]["cost_formula"] == "shape[0]"

    def test_permutation_uses_shape_axis(self):
        # Same Fisher-Yates internals as shuffle.
        assert REGISTRY["random.Generator.permutation"]["cost_formula"] == (
            "shape[axis] of x (default axis=0) for ndarray input; int input n -> n; "
            "non-ndarray sequences bill len(x) regardless of axis  (Fisher-Yates draws; "
            "axis=None is rejected by numpy)"
        )
        assert REGISTRY["random.RandomState.permutation"]["cost_formula"] == (
            "x.shape[0] (Fisher-Yates draws; int input n costs n; axis fixed at 0 -- "
            "RandomState has no axis kwarg)"
        )

    def test_permuted_uses_numel_input(self):
        # Genuinely numel: independent shuffle per slice along the axis.
        assert REGISTRY["random.Generator.permuted"]["cost_formula"] == (
            "numel(input) for ndarray input; non-ndarray sequences bill len(x) (outer "
            "length); axis does not change the cost"
        )

    def test_bytes_uses_length(self):
        assert REGISTRY["random.Generator.bytes"]["cost_formula"] == (
            "length (the requested byte count; max(length, 1))"
        )

    def test_choice_uses_choice_cost(self):
        assert REGISTRY["random.Generator.choice"]["cost_formula"] == (
            "replace=True: numel(output), plus 3*n + numel(output)*ceil(log2(n)) if p "
            "given; replace=False: n if p is None else sort_cost(n); n = pool's "
            "leading-axis length (int a, len(a), or a.shape[0]) regardless of the axis "
            "kwarg. Canonical (replace=True, p=None): size"
        )

    def test_standard_normal_uses_numel_output(self):
        assert (
            REGISTRY["random.Generator.standard_normal"]["cost_formula"]
            == "numel(output)"
        )


# Methods on numpy.random.RandomState that are NOT counted (free).
RANDOMSTATE_FREE = {"get_state", "seed", "set_state"}

# All other public attributes on RandomState are counted samplers.
RANDOMSTATE_COUNTED = {
    "beta",
    "binomial",
    "bytes",
    "chisquare",
    "choice",
    "dirichlet",
    "exponential",
    "f",
    "gamma",
    "geometric",
    "gumbel",
    "hypergeometric",
    "laplace",
    "logistic",
    "lognormal",
    "logseries",
    "multinomial",
    "multivariate_normal",
    "negative_binomial",
    "noncentral_chisquare",
    "noncentral_f",
    "normal",
    "pareto",
    "permutation",
    "poisson",
    "power",
    "rand",
    "randint",
    "randn",
    "random",
    "random_integers",
    "random_sample",
    "rayleigh",
    "shuffle",
    "standard_cauchy",
    "standard_exponential",
    "standard_gamma",
    "standard_normal",
    "standard_t",
    "tomaxint",
    "triangular",
    "uniform",
    "vonmises",
    "wald",
    "weibull",
    "zipf",
}


class TestRandomStateRegistryCoverage:
    def test_every_public_method_has_a_registry_entry(self):
        public = {n for n in dir(np.random.RandomState) if not n.startswith("_")}
        registered = {
            n[len("random.RandomState.") :]
            for n in REGISTRY
            if n.startswith("random.RandomState.")
        }
        missing = public - registered
        assert not missing, (
            f"RandomState methods missing from registry: {sorted(missing)}"
        )

    def test_counted_entries_have_required_fields(self):
        for short in RANDOMSTATE_COUNTED:
            op = f"random.RandomState.{short}"
            assert op in REGISTRY, f"missing entry: {op}"
            entry = REGISTRY[op]
            assert entry["category"] == "counted_random_method", op
            assert entry["module"] == "numpy.random", op
            assert "cost_formula" in entry, op

    def test_free_entries_have_required_fields(self):
        for short in RANDOMSTATE_FREE:
            op = f"random.RandomState.{short}"
            assert op in REGISTRY, f"missing entry: {op}"
            entry = REGISTRY[op]
            assert entry["category"] == "free_random_method", op
            assert entry["module"] == "numpy.random", op
