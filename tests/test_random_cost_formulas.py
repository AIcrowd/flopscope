"""Unit tests for fnp.random cost formula vocabulary."""

import numpy as np

from flopscope._flops import _ceil_log2


class TestNumelOutput:
    def test_array_result(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["numel(output)"]
        assert formula((), {}, np.zeros((10, 20))) == 200

    def test_scalar_result(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["numel(output)"]
        assert formula((), {}, np.float64(1.0)) == 1

    def test_zero_size_returns_one(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["numel(output)"]
        assert formula((), {}, np.zeros(0)) == 1


class TestNumelInput:
    def test_first_positional_array(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["numel(input)"]
        a = np.arange(50)
        assert formula((a,), {}, None) == 50

    def test_zero_size_returns_one(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["numel(input)"]
        assert formula((np.zeros(0),), {}, None) == 1


class TestLength:
    def test_positional(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["length"]
        assert formula((42,), {}, b"x" * 42) == 42

    def test_keyword(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["length"]
        assert formula((), {"length": 7}, b"y" * 7) == 7

    def test_zero_returns_one(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["length"]
        assert formula((0,), {}, b"") == 1


class TestSortCost:
    def test_integer_population(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["sort_cost(n)"]
        # `a` passed as the integer 16 (population size, range [0, 16))
        assert formula((16,), {}, np.zeros(5)) == 16 * _ceil_log2(16)

    def test_array_population(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["sort_cost(n)"]
        a = np.arange(20)
        assert formula((a,), {}, np.zeros(5)) == 20 * _ceil_log2(20)

    def test_zero_d_array_population(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["sort_cost(n)"]
        a = np.array(16)  # 0-D scalar array; numpy.choice accepts as int(a)
        assert formula((a,), {}, np.zeros(5)) == 16 * _ceil_log2(16)


class TestChoiceCost:
    def test_with_replacement_uses_numel_output(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["choice_cost"]
        result = np.zeros(20)
        cost = formula((100,), {"size": 20, "replace": True}, result)
        assert cost == 20

    def test_without_replacement_uses_n(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["choice_cost"]
        result = np.zeros(5)
        # Fisher-Yates O(n): p=None path charges n, not sort_cost(n).
        cost = formula((16,), {"size": 5, "replace": False}, result)
        assert cost == 16

    def test_without_replacement_with_p_uses_sort_cost(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["choice_cost"]
        result = np.zeros(5)
        # Data-dependent rejection loop: sort_cost(n) conservative floor.
        p = np.ones(16) / 16
        cost = formula((16,), {"size": 5, "replace": False, "p": p}, result)
        assert cost == 16 * _ceil_log2(16)

    def test_replace_passed_positionally(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["choice_cost"]
        # Generator.choice signature: choice(a, size, replace, p, axis, shuffle)
        result = np.zeros(5)
        # replace=False positional, p=None: Fisher-Yates charges n.
        cost = formula((16, 5, False), {}, result)
        assert cost == 16

    def test_2d_pool_axis_kwarg_replace_true_with_p(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["choice_cost"]
        pool = np.arange(16).reshape(2, 8)
        p = np.ones(8) / 8
        result = np.zeros(3)
        # Generator.choice samples along `axis`: n = shape[1] = 8, NOT shape[0].
        # base 3 + CDF build 3*8 + 3 draws * ceil(log2(8)) = 3 + 24 + 9 = 36.
        cost = formula((pool,), {"size": 3, "replace": True, "p": p, "axis": 1}, result)
        assert cost == 3 + 3 * 8 + 3 * _ceil_log2(8)
        assert cost == 36

    def test_2d_pool_axis_positional_replace_true_with_p(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["choice_cost"]
        pool = np.arange(16).reshape(2, 8)
        p = np.ones(8) / 8
        result = np.zeros(3)
        # axis as the 5th positional per Generator.choice's signature:
        # choice(a, size, replace, p, axis, shuffle).
        cost = formula((pool, 3, True, p, 1), {}, result)
        assert cost == 3 + 3 * 8 + 3 * _ceil_log2(8)

    def test_2d_pool_axis_kwarg_replace_false(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["choice_cost"]
        pool = np.arange(16).reshape(2, 8)
        result = np.zeros(3)
        # replace=False, p=None: Fisher-Yates/Floyd over the sampled axis
        # charges n = shape[1] = 8, not shape[0] = 2.
        cost = formula((pool,), {"size": 3, "replace": False, "axis": 1}, result)
        assert cost == 8

    def test_2d_pool_default_axis_charges_first_dim(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["choice_cost"]
        pool = np.arange(16).reshape(2, 8)
        result = np.zeros(3)
        # No axis argument -> axis=0 default. Pins the legacy signatures
        # (RandomState.choice / module-level random.choice have no axis and
        # require 1-D pools): n = shape[0] exactly as before.
        cost = formula((pool,), {"size": 3, "replace": False}, result)
        assert cost == 2


class TestRegistry:
    def test_all_named_formulas_present(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        # Spec-mandated formula names
        assert set(COST_FORMULAS.keys()) >= {
            "numel(output)",
            "numel(input)",
            "length",
            "sort_cost(n)",
            "choice_cost",
        }


class TestShapeAxis:
    def test_1d_array(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["shape[axis]"]
        a = np.arange(50)
        assert formula((a,), {}, None) == 50

    def test_2d_default_axis_charges_first_dim(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["shape[axis]"]
        a = np.zeros((5, 10))
        # default axis=0 → cost is shape[0]=5, NOT numel=50
        assert formula((a,), {}, None) == 5

    def test_2d_explicit_axis(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["shape[axis]"]
        a = np.zeros((5, 10))
        assert formula((a,), {"axis": 1}, None) == 10
        assert formula((a,), {"axis": 0}, None) == 5

    def test_axis_none_means_flatten(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["shape[axis]"]
        a = np.zeros((5, 10))
        # numpy treats axis=None as flatten-then-operate; cost = numel
        assert formula((a,), {"axis": None}, None) == 50

    def test_int_input(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        # permutation(16) — Fisher-Yates of [0,16); cost = 16
        formula = COST_FORMULAS["shape[axis]"]
        assert formula((16,), {}, None) == 16

    def test_list_input(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["shape[axis]"]
        assert formula(([1, 2, 3, 4],), {}, None) == 4

    def test_zero_d_array(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["shape[axis]"]
        a = np.array(7)  # 0-D scalar array
        # 0-D has no shape[0]; numpy choice/permutation treats as int(a)
        # Defensive: don't crash, return at least 1
        assert formula((a,), {}, None) >= 1

    def test_2d_positional_axis(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["shape[axis]"]
        a = np.zeros((5, 10))
        # Positional axis (numpy: shuffle(x, axis=0))
        assert formula((a, 1), {}, None) == 10  # shape[1]
        assert formula((a, 0), {}, None) == 5  # shape[0]

    def test_2d_positional_none_axis(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["shape[axis]"]
        a = np.zeros((5, 10))
        # Positional axis=None means flatten → numel
        assert formula((a, None), {}, None) == 50


class TestShapeAxisRegistration:
    def test_shape_axis_in_formulas(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        assert "shape[axis]" in COST_FORMULAS
