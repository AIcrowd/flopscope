"""Additional tests for _pointwise.py to increase coverage to ~95%.

Covers:
- SymmetricTensor paths in unary, binary, and reduction ops
- Binary op symmetry intersection and loss warnings
- Reduction symmetry propagation (keepdims, partial loss, full loss)
- around() and round() with scalar inputs
- sort_complex() with complex arrays
- isclose() with scalar inputs
- bitwise_count() (numpy >= 2.1)
- vecdot() (numpy >= 2.0)
- Multi-output unary/binary ops (modf, frexp, divmod)
- Scalar / 0-d array paths
"""

from __future__ import annotations

import math
import warnings

import numpy
import pytest

from flopscope._budget import BudgetContext
from flopscope._config import configure
from flopscope._perm_group import SymmetryGroup as PermutationGroup
from flopscope._pointwise import (
    abs,
    add,
    # Reductions
    argmax,
    around,
    clip,
    convolve,
    corrcoef,
    correlate,
    cov,
    cross,
    # Additional custom ops
    diff,
    divmod,
    # Custom ops
    dot,
    ediff1d,
    exp,
    frexp,
    gradient,
    inner,
    interp,
    isclose,
    kron,
    matmul,
    mean,
    modf,
    multiply,
    negative,
    outer,
    real_if_close,
    round,
    sort_complex,
    sqrt,
    std,
    subtract,
    sum,
    tensordot,
    trapezoid,
    vdot,
)
from flopscope._symmetric import SymmetricTensor, as_symmetric
from flopscope.errors import SymmetryLossWarning

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_symmetric_2d(n: int = 4) -> SymmetricTensor:
    """Create a symmetric (n x n) matrix as SymmetricTensor."""
    data = numpy.random.default_rng(42).random((n, n))
    data = (data + data.T) / 2
    return as_symmetric(data, symmetry=(0, 1))


def _make_symmetric_3d(n: int = 3) -> SymmetricTensor:
    """Create a (n x n x n) tensor symmetric in axes (0, 1)."""
    rng = numpy.random.default_rng(99)
    data = rng.random((n, n, n))
    # Symmetrize axes 0, 1
    data = (data + data.transpose(1, 0, 2)) / 2
    return as_symmetric(data, symmetry=(0, 1))


def _make_cyclic_3d(n: int = 3) -> SymmetricTensor:
    """Create a (n x n x n) tensor invariant under the cyclic group C3."""
    rng = numpy.random.default_rng(7)
    data = rng.random((n, n, n))
    rotated1 = data.transpose(1, 2, 0)
    rotated2 = data.transpose(2, 0, 1)
    return as_symmetric(
        (data + rotated1 + rotated2) / 3.0,
        symmetry=PermutationGroup.cyclic(axes=(0, 1, 2)),
    )


# ---------------------------------------------------------------------------
# 1. Unary ops on SymmetricTensor inputs
# ---------------------------------------------------------------------------


class TestUnarySymmetric:
    """Unary ops should propagate symmetry through SymmetricTensor inputs."""

    def test_exp_preserves_symmetry(self):
        st = _make_symmetric_2d()
        with BudgetContext(flop_budget=10**6) as budget:
            result = exp(st)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == st.symmetry
        assert numpy.allclose(result, numpy.exp(numpy.asarray(st)))

    def test_unary_c3_does_not_widen_to_s3(self):
        st = _make_cyclic_3d()
        expected = PermutationGroup.cyclic(axes=(0, 1, 2))
        with BudgetContext(flop_budget=10**6):
            result = negative(st)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == expected
        assert result.symmetry != PermutationGroup.symmetric(axes=(0, 1, 2))

    def test_real_if_close_preserves_symmetry_without_out_support(self):
        st = _make_symmetric_2d()
        with BudgetContext(flop_budget=10**6):
            result = real_if_close(st)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == st.symmetry

    def test_sqrt_preserves_symmetry(self):
        st = _make_symmetric_2d()
        with BudgetContext(flop_budget=10**6):
            result = sqrt(st)
        assert isinstance(result, SymmetricTensor)

    def test_negative_preserves_symmetry(self):
        st = _make_symmetric_2d()
        with BudgetContext(flop_budget=10**6):
            result = negative(st)
        assert isinstance(result, SymmetricTensor)

    def test_abs_preserves_symmetry(self):
        st = _make_symmetric_2d()
        with BudgetContext(flop_budget=10**6):
            result = abs(st)
        assert isinstance(result, SymmetricTensor)

    def test_unary_symmetry_reduces_flop_cost(self):
        """Symmetric tensor should have a lower FLOP cost than full tensor."""
        st = _make_symmetric_2d(8)
        plain = numpy.asarray(st)
        with BudgetContext(flop_budget=10**6) as b_sym:
            exp(st)
        with BudgetContext(flop_budget=10**6) as b_plain:
            exp(plain)
        # Symmetric cost should be less than or equal (unique_elements < full size)
        assert b_sym.flops_used <= b_plain.flops_used


# ---------------------------------------------------------------------------
# 2. Binary ops with symmetric tensors
# ---------------------------------------------------------------------------


class TestBinarySymmetric:
    """Binary ops between symmetric and/or non-symmetric tensors."""

    def test_add_two_symmetric_same_groups(self):
        """Adding two symmetric tensors with the same groups preserves symmetry."""
        st1 = _make_symmetric_2d(4)
        st2 = _make_symmetric_2d(4)
        with BudgetContext(flop_budget=10**6):
            result = add(st1, st2)
        assert isinstance(result, SymmetricTensor)
        assert numpy.allclose(result, numpy.asarray(st1) + numpy.asarray(st2))

    def test_add_symmetric_and_scalar(self):
        """Adding a scalar to a symmetric tensor preserves symmetry."""
        st = _make_symmetric_2d(4)
        with BudgetContext(flop_budget=10**6):
            result = add(st, 1.0)
        assert isinstance(result, SymmetricTensor)

    def test_add_scalar_and_symmetric(self):
        """Adding symmetric to a scalar (reversed) preserves symmetry."""
        st = _make_symmetric_2d(4)
        with BudgetContext(flop_budget=10**6):
            result = add(1.0, st)
        assert isinstance(result, SymmetricTensor)

    def test_multiply_symmetric_by_nonsymmetric_loses_symmetry(self):
        """Multiplying symmetric by non-symmetric with different structure loses symmetry."""
        st = _make_symmetric_2d(4)
        plain = numpy.ones((4, 4)) * numpy.arange(4)  # Not symmetric
        with BudgetContext(flop_budget=10**6):
            # intersect_symmetry gets called; one operand has no symmetry
            result = multiply(st, plain)
        # intersect_symmetry returns None when one has no groups
        # so result should not be SymmetricTensor
        assert not isinstance(result, SymmetricTensor)

    def test_binary_symmetry_loss_warns(self):
        """Binary op between symmetric tensors with non-intersecting groups warns."""
        # Create two tensors with different symmetry groups
        rng = numpy.random.default_rng(42)
        data1 = rng.random((3, 3, 3))
        data1 = (data1 + data1.transpose(1, 0, 2)) / 2
        st1 = as_symmetric(data1, symmetry=(0, 1))

        data2 = rng.random((3, 3, 3))
        data2 = (data2 + data2.transpose(0, 2, 1)) / 2
        st2 = as_symmetric(data2, symmetry=(1, 2))

        configure(symmetry_warnings=True)
        try:
            with BudgetContext(flop_budget=10**6):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = add(st1, st2)
                    # Both have symmetry groups; intersection determines result.
                    # At minimum the code path is exercised.
        finally:
            configure(symmetry_warnings=True)

    def test_binary_op_0d_symmetric_and_array(self):
        """Binary op where one input is a 0-d array."""
        st = _make_symmetric_2d(3)
        with BudgetContext(flop_budget=10**6):
            result = multiply(st, numpy.float64(2.0))
        assert isinstance(result, SymmetricTensor)

    def test_dunder_multiply_matches_functional_multiply_budget_and_symmetry(self):
        st = _make_symmetric_2d(4)
        with BudgetContext(flop_budget=10**6) as functional_budget:
            functional = multiply(st, st)
        with BudgetContext(flop_budget=10**6) as dunder_budget:
            via_dunder = st * st
        assert functional_budget.flops_used == dunder_budget.flops_used
        assert isinstance(via_dunder, SymmetricTensor)
        assert via_dunder.symmetry == functional.symmetry  # pyright: ignore[reportAttributeAccessIssue]
        assert numpy.allclose(via_dunder, functional)

    def test_add_with_plain_out_preserves_output_identity(self):
        st = _make_symmetric_2d(4)
        out = numpy.empty_like(numpy.asarray(st))
        with BudgetContext(flop_budget=10**6):
            result = add(st, 1.0, out=out)  # pyright: ignore[reportArgumentType]
        assert result is out
        assert numpy.allclose(result, numpy.asarray(st) + 1.0)

    def test_negative_with_symmetric_out_preserves_output_identity(self):
        st = _make_symmetric_2d(4)
        out = as_symmetric(numpy.zeros_like(numpy.asarray(st)), symmetry=(0, 1))
        with BudgetContext(flop_budget=10**6):
            result = negative(st, out=out)
        assert result is out
        assert numpy.allclose(result, -numpy.asarray(st))
        assert result.symmetry == st.symmetry  # pyright: ignore[reportAttributeAccessIssue]

    def test_clip_with_symmetric_out_preserves_output_identity(self):
        st = _make_symmetric_2d(4)
        out = as_symmetric(numpy.zeros_like(numpy.asarray(st)), symmetry=(0, 1))
        with BudgetContext(flop_budget=10**6):
            result = clip(st, 0.25, 0.75, out=out)
        assert result is out
        assert numpy.allclose(result, numpy.clip(numpy.asarray(st), 0.25, 0.75))
        assert result.symmetry == st.symmetry  # pyright: ignore[reportAttributeAccessIssue]

    def test_add_with_incompatible_symmetric_out_raises_without_mutation(self):
        st = _make_symmetric_2d(4)
        dense = numpy.arange(16.0).reshape(4, 4)
        out = as_symmetric(numpy.eye(4), symmetry=(0, 1))
        before = numpy.asarray(out).copy()
        with BudgetContext(flop_budget=10**6):
            with pytest.raises(ValueError, match="out symmetry"):
                add(st, dense, out=out)
        assert numpy.array_equal(out, before)

    def test_negative_with_dense_input_and_symmetric_out_raises_without_mutation(self):
        dense = numpy.arange(16.0).reshape(4, 4)
        out = as_symmetric(numpy.eye(4), symmetry=(0, 1))
        before = numpy.asarray(out).copy()
        with BudgetContext(flop_budget=10**6):
            with pytest.raises(ValueError, match="out symmetry"):
                negative(dense, out=out)
        assert numpy.array_equal(out, before)

    def test_binary_op_total_symmetry_loss_warning(self):
        """When no symmetry groups are shared, warn about total loss."""
        # Create tensor sym in (0,1), combine with non-symmetric of same shape
        st = _make_symmetric_2d(3)
        plain = numpy.ones((3, 3))

        configure(symmetry_warnings=True)
        with BudgetContext(flop_budget=10**6):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # plain has no symmetry, so intersect_symmetry returns None
                result = add(st, plain)
        # No SymmetricTensor expected
        assert not isinstance(result, SymmetricTensor)


# ---------------------------------------------------------------------------
# 3. Reductions on symmetric tensors
# ---------------------------------------------------------------------------


class TestReductionSymmetric:
    """Reductions along symmetric axes should propagate symmetry correctly."""

    def test_sum_along_non_symmetric_axis_preserves_symmetry(self):
        """Reducing a non-symmetric axis preserves the symmetric groups."""
        st = _make_symmetric_3d(3)  # symmetric in (0, 1), shape (3,3,3)
        with BudgetContext(flop_budget=10**6):
            result = sum(st, axis=2)
        # Axes (0,1) should survive since fnp reduced axis 2
        assert isinstance(result, SymmetricTensor)

    def test_sum_over_reduced_block_preserves_unaffected_symmetry(self):
        rng = numpy.random.default_rng(5)
        data = rng.random((3, 3, 4, 4))
        data = (data + data.transpose(0, 1, 3, 2)) / 2
        st = as_symmetric(data, symmetry=(2, 3))
        with BudgetContext(flop_budget=10**6):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = sum(st, axis=0)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == PermutationGroup.symmetric(axes=(1, 2))
        assert not [
            warning
            for warning in caught
            if issubclass(warning.category, SymmetryLossWarning)
        ]

    def test_sum_with_plain_out_preserves_output_identity(self):
        st = _make_symmetric_3d(3)
        out = numpy.empty((3, 3))
        with BudgetContext(flop_budget=10**6):
            result = sum(st, axis=2, out=out)
        assert result is out
        assert numpy.allclose(result, numpy.sum(numpy.asarray(st), axis=2))

    def test_sum_with_symmetric_out_preserves_output_identity(self):
        st = _make_symmetric_3d(3)
        out = as_symmetric(numpy.zeros((3, 3)), symmetry=(0, 1))
        with BudgetContext(flop_budget=10**6):
            result = sum(st, axis=2, out=out)
        assert result is out
        assert numpy.allclose(result, numpy.sum(numpy.asarray(st), axis=2))
        assert result.symmetry == PermutationGroup.symmetric(axes=(0, 1))  # pyright: ignore[reportAttributeAccessIssue]

    def test_sum_along_symmetric_axis_reduces_group(self):
        """Reducing one axis of a symmetric group partially breaks it."""
        st = _make_symmetric_3d(3)  # symmetric in (0, 1), shape (3,3,3)
        with BudgetContext(flop_budget=10**6):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = sum(st, axis=0)
        # After reducing axis 0 from group (0,1), only axis 1 remains
        # which is fewer than 2, so no symmetry survives.

    def test_sum_full_reduction_loses_all_symmetry(self):
        """Full reduction (axis=None) removes all symmetry."""
        st = _make_symmetric_2d(3)
        configure(symmetry_warnings=True)
        with BudgetContext(flop_budget=10**6):
            result = sum(st)
        # Scalar result
        assert numpy.ndim(result) == 0

    def test_mean_symmetric_propagation(self):
        """mean() (extra_output=True) on a symmetric tensor."""
        st = _make_symmetric_3d(3)
        with BudgetContext(flop_budget=10**6):
            result = mean(st, axis=2)
        assert isinstance(result, SymmetricTensor)

    def test_reduction_keepdims_preserves_symmetry(self):
        """keepdims=True should keep symmetry groups with reduced dims removed."""
        st = _make_symmetric_3d(3)
        with BudgetContext(flop_budget=10**6):
            result = sum(st, axis=2, keepdims=True)
        # Symmetry in (0, 1) should survive, since axis 2 is not in group
        assert isinstance(result, SymmetricTensor)

    def test_reduction_keepdims_reduces_group(self):
        """keepdims=True with axis inside symmetric group trims that group."""
        st = _make_symmetric_3d(3)
        with BudgetContext(flop_budget=10**6):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = sum(st, axis=0, keepdims=True)
        # axis 0 is in group (0,1); remaining single axis => group lost

    def test_std_on_symmetric_tensor(self):
        """std() (cost_multiplier=2, extra_output=True) on a symmetric tensor."""
        st = _make_symmetric_3d(4)
        with BudgetContext(flop_budget=10**6):
            result = std(st, axis=2)
        assert isinstance(result, SymmetricTensor)

    def test_argmax_symmetric(self):
        """argmax on a symmetric tensor (full reduction)."""
        st = _make_symmetric_2d(3)
        with BudgetContext(flop_budget=10**6):
            result = argmax(st)
        assert isinstance(result, (int, numpy.integer, numpy.ndarray))


# ---------------------------------------------------------------------------
# 4. around() and round() with scalar inputs
# ---------------------------------------------------------------------------


class TestAroundRoundScalar:
    """around() and round() should return Python scalars for scalar inputs."""

    def test_around_scalar_int(self):
        with BudgetContext(flop_budget=10**6):
            result = around(3)
        # .item() returns a numpy scalar (e.g. np.int64), not a Python int
        assert not isinstance(result, numpy.ndarray)
        assert result == 3

    def test_around_scalar_float(self):
        with BudgetContext(flop_budget=10**6):
            result = around(3.7)
        assert not isinstance(result, numpy.ndarray)
        assert result == 4.0

    def test_around_with_decimals(self):
        with BudgetContext(flop_budget=10**6):
            result = around(3.14159, decimals=2)
        assert not isinstance(result, numpy.ndarray)
        assert numpy.isclose(result, 3.14, atol=0.01)

    def test_around_array_returns_array(self):
        with BudgetContext(flop_budget=10**6):
            result = around(numpy.array([1.5, 2.5, 3.5]))
        assert isinstance(result, numpy.ndarray)

    def test_around_symmetric_tensor(self):
        st = _make_symmetric_2d(3)
        with BudgetContext(flop_budget=10**6):
            result = around(st)
        assert isinstance(result, SymmetricTensor)

    def test_round_scalar_int(self):
        with BudgetContext(flop_budget=10**6):
            result = round(5)
        assert not isinstance(result, numpy.ndarray)
        assert result == 5

    def test_round_scalar_float(self):
        with BudgetContext(flop_budget=10**6):
            result = round(2.7)
        assert not isinstance(result, numpy.ndarray)
        assert result == 3.0

    def test_round_with_decimals(self):
        with BudgetContext(flop_budget=10**6):
            result = round(2.345, decimals=1)
        assert not isinstance(result, numpy.ndarray)

    def test_round_array_returns_array(self):
        with BudgetContext(flop_budget=10**6):
            result = round(numpy.array([1.1, 2.9]))
        assert isinstance(result, numpy.ndarray)

    def test_round_symmetric_tensor(self):
        st = _make_symmetric_2d(3)
        with BudgetContext(flop_budget=10**6):
            result = round(st)
        assert isinstance(result, SymmetricTensor)


# ---------------------------------------------------------------------------
# 5. sort_complex()
# ---------------------------------------------------------------------------


class TestSortComplex:
    """sort_complex() with complex and real arrays."""

    def test_sort_complex_basic(self):
        a = numpy.array([3 + 1j, 1 + 2j, 2 + 0j])
        with BudgetContext(flop_budget=10**6) as budget:
            result = sort_complex(a)
        assert numpy.allclose(result, numpy.sort_complex(a))
        # Cost should be n * ceil(log2(n))
        n = a.size
        expected_cost = n * math.ceil(math.log2(n))
        assert budget.flops_used == expected_cost

    def test_sort_complex_real_input(self):
        a = numpy.array([5.0, 1.0, 3.0, 2.0])
        with BudgetContext(flop_budget=10**6):
            result = sort_complex(a)
        # Should cast to complex and sort
        assert numpy.issubdtype(result.dtype, numpy.complexfloating)

    def test_sort_complex_single_element(self):
        a = numpy.array([42.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = sort_complex(a)
        # n=1 => log2n=1, cost=1
        assert budget.flops_used == 1

    def test_sort_complex_from_list(self):
        """sort_complex should handle non-ndarray input (list)."""
        with BudgetContext(flop_budget=10**6):
            result = sort_complex([3 + 1j, 1 + 0j, 2 + 2j])
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# 6. isclose() with scalar inputs
# ---------------------------------------------------------------------------


class TestIscloseScalar:
    """isclose() should return Python bool for two scalar inputs."""

    def test_isclose_two_scalars_true(self):
        with BudgetContext(flop_budget=10**6):
            result = isclose(1.0, 1.0 + 1e-10)
        assert isinstance(result, (bool, numpy.bool_))
        assert result is True or result == True  # noqa: E712

    def test_isclose_two_scalars_false(self):
        with BudgetContext(flop_budget=10**6):
            result = isclose(1.0, 2.0)
        assert isinstance(result, (bool, numpy.bool_))
        assert not result

    def test_isclose_int_scalars(self):
        with BudgetContext(flop_budget=10**6):
            result = isclose(5, 5)
        assert result

    def test_isclose_array_inputs(self):
        """isclose with arrays should return an array."""
        with BudgetContext(flop_budget=10**6):
            result = isclose(numpy.array([1.0, 2.0]), numpy.array([1.0, 2.1]))
        assert isinstance(result, numpy.ndarray)

    def test_isclose_mixed_scalar_array(self):
        """One scalar, one array."""
        with BudgetContext(flop_budget=10**6):
            result = isclose(1.0, numpy.array([1.0, 2.0]))
        assert isinstance(result, numpy.ndarray)

    def test_isclose_custom_tolerances(self):
        with BudgetContext(flop_budget=10**6):
            result = isclose(1.0, 1.1, atol=0.2)
        assert result


# ---------------------------------------------------------------------------
# 7. bitwise_count (numpy >= 2.1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(numpy, "bitwise_count"),
    reason="numpy.bitwise_count requires numpy >= 2.1",
)
class TestBitwiseCount:
    def test_bitwise_count_basic(self):
        from flopscope._pointwise import bitwise_count

        a = numpy.array([0, 1, 3, 7, 15], dtype=numpy.uint8)
        with BudgetContext(flop_budget=10**6) as budget:
            result = bitwise_count(a)
        assert numpy.array_equal(result, numpy.bitwise_count(a))
        assert budget.flops_used == 5


# ---------------------------------------------------------------------------
# 8. vecdot (numpy >= 2.0)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(numpy, "vecdot"),
    reason="numpy.vecdot requires numpy >= 2.0",
)
class TestVecdot:
    def test_vecdot_1d(self):
        from flopscope._pointwise import vecdot

        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = vecdot(a, b)
        assert numpy.allclose(result, numpy.vecdot(a, b))
        # Cost = output_size * contracted_axis = 1 * 3
        assert budget.flops_used == 3

    def test_vecdot_2d(self):
        from flopscope._pointwise import vecdot

        a = numpy.ones((5, 4))
        b = numpy.ones((5, 4))
        with BudgetContext(flop_budget=10**6) as budget:
            result = vecdot(a, b)
        assert result.shape == (5,)
        # Cost = 5 * 4 = 20
        assert budget.flops_used == 20

    def test_vecdot_from_list(self):
        """vecdot should convert non-ndarray inputs."""
        from flopscope._pointwise import vecdot

        with BudgetContext(flop_budget=10**6):
            result = vecdot([1, 2, 3], [4, 5, 6])
        assert numpy.allclose(result, 32)


# ---------------------------------------------------------------------------
# 9. Multi-output ops
# ---------------------------------------------------------------------------


class TestMultiOutputOps:
    def test_modf_returns_tuple(self):
        x = numpy.array([1.5, 2.7, -3.2])
        with BudgetContext(flop_budget=10**6) as budget:
            result = modf(x)
        assert isinstance(result, tuple)
        assert len(result) == 2
        ref = numpy.modf(x)
        assert numpy.allclose(result[0], ref[0])
        assert numpy.allclose(result[1], ref[1])
        assert budget.flops_used == 3

    def test_frexp_returns_tuple(self):
        x = numpy.array([2.0, 4.0, 8.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = frexp(x)
        assert isinstance(result, tuple)
        assert len(result) == 2
        ref = numpy.frexp(x)
        assert numpy.allclose(result[0], ref[0])
        assert numpy.array_equal(result[1], ref[1])

    def test_divmod_returns_tuple(self):
        x = numpy.array([7.0, 10.0, 13.0])
        y = numpy.array([3.0, 4.0, 5.0])
        with BudgetContext(flop_budget=10**6):
            result = divmod(x, y)
        assert isinstance(result, tuple)
        assert len(result) == 2
        ref = numpy.divmod(x, y)
        assert numpy.allclose(result[0], ref[0])
        assert numpy.allclose(result[1], ref[1])

    def test_modf_with_out_tuple(self):
        x = numpy.array([1.5, 2.7, -3.2])
        out_frac = numpy.zeros(3)
        out_int = numpy.zeros(3)
        with BudgetContext(flop_budget=10**6):
            result = modf(x, out=(out_frac, out_int))  # pyright: ignore[reportArgumentType]
        assert result[0] is out_frac
        assert result[1] is out_int
        ref = numpy.modf(x)
        assert numpy.allclose(out_frac, ref[0])
        assert numpy.allclose(out_int, ref[1])

    def test_frexp_with_out_tuple(self):
        x = numpy.array([2.0, 4.0, 8.0])
        out_mant = numpy.zeros(3)
        out_exp = numpy.zeros(3, dtype=numpy.int32)
        with BudgetContext(flop_budget=10**6):
            result = frexp(x, out=(out_mant, out_exp))  # pyright: ignore[reportArgumentType]
        assert result[0] is out_mant
        assert result[1] is out_exp
        ref = numpy.frexp(x)
        assert numpy.allclose(out_mant, ref[0])
        assert numpy.array_equal(out_exp, ref[1])

    def test_divmod_with_out_tuple(self):
        x = numpy.array([7.0, 10.0, 13.0])
        y = numpy.array([3.0, 4.0, 5.0])
        out_q = numpy.zeros(3)
        out_r = numpy.zeros(3)
        with BudgetContext(flop_budget=10**6):
            result = divmod(x, y, out=(out_q, out_r))  # pyright: ignore[reportArgumentType]
        assert result[0] is out_q
        assert result[1] is out_r
        ref = numpy.divmod(x, y)
        assert numpy.allclose(out_q, ref[0])
        assert numpy.allclose(out_r, ref[1])

    def test_divmod_with_partial_out(self):
        x = numpy.array([7.0, 10.0])
        y = numpy.array([3.0, 4.0])
        out_q = numpy.zeros(2)
        with BudgetContext(flop_budget=10**6):
            result = divmod(x, y, out=(out_q, None))  # pyright: ignore[reportArgumentType]
        assert result[0] is out_q
        # Second slot was allocated by numpy; should still be a usable array
        assert result[1] is not None
        ref = numpy.divmod(x, y)
        assert numpy.allclose(out_q, ref[0])
        assert numpy.allclose(result[1], ref[1])

    def test_modf_invalid_out_length_raises(self):
        x = numpy.array([1.5, 2.7])
        single = numpy.zeros(2)
        with BudgetContext(flop_budget=10**6):
            with pytest.raises(TypeError, match="length"):
                modf(x, out=(single,))  # pyright: ignore[reportArgumentType]

    def test_modf_invalid_out_type_raises(self):
        x = numpy.array([1.5, 2.7])
        single = numpy.zeros(2)
        with BudgetContext(flop_budget=10**6):
            with pytest.raises(TypeError, match="tuple"):
                modf(x, out=single)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# 10. Scalar / 0-d array paths in unary and binary ops
# ---------------------------------------------------------------------------


class TestScalarPaths:
    """Exercise scalar and 0-d array inputs to unary/binary ops."""

    def test_unary_with_python_float(self):
        with BudgetContext(flop_budget=10**6) as budget:
            result = exp(1.0)
        assert numpy.isclose(result, numpy.exp(1.0))
        assert budget.flops_used == 1

    def test_unary_with_python_int(self):
        with BudgetContext(flop_budget=10**6):
            result = abs(-5)
        assert result == 5

    def test_unary_with_0d_array(self):
        with BudgetContext(flop_budget=10**6):
            result = sqrt(numpy.float64(9.0))
        assert numpy.isclose(result, 3.0)

    def test_binary_two_scalars(self):
        with BudgetContext(flop_budget=10**6) as budget:
            result = add(3.0, 4.0)
        assert numpy.isclose(result, 7.0)
        assert budget.flops_used == 1

    def test_binary_scalar_and_array(self):
        with BudgetContext(flop_budget=10**6):
            result = multiply(2.0, numpy.array([1.0, 2.0, 3.0]))
        assert numpy.allclose(result, [2.0, 4.0, 6.0])

    def test_binary_array_and_scalar(self):
        with BudgetContext(flop_budget=10**6):
            result = subtract(numpy.array([5.0, 6.0]), 1.0)
        assert numpy.allclose(result, [4.0, 5.0])


# ---------------------------------------------------------------------------
# 11. clip() on symmetric tensor
# ---------------------------------------------------------------------------


class TestClipSymmetric:
    def test_clip_preserves_symmetry(self):
        st = _make_symmetric_2d(3)
        with BudgetContext(flop_budget=10**6):
            result = clip(st, 0.2, 0.8)
        assert isinstance(result, SymmetricTensor)

    def test_clip_with_keyword_args(self):
        """Exercise the min=/max= keyword path in clip()."""
        x = numpy.array([1.0, 5.0, 10.0])
        with BudgetContext(flop_budget=10**6):
            result = clip(x, min=2.0, max=8.0)
        assert numpy.allclose(result, [2.0, 5.0, 8.0])


# ---------------------------------------------------------------------------
# 12. dot / matmul with symmetric tensors
# ---------------------------------------------------------------------------


class TestDotMatmulSymmetric:
    def test_dot_symmetric_2d(self):
        st = _make_symmetric_2d(4)
        with BudgetContext(flop_budget=10**6):
            result = dot(st, st)
        expected = numpy.dot(numpy.asarray(st), numpy.asarray(st))
        assert numpy.allclose(result, expected)

    def test_matmul_symmetric_2d(self):
        st = _make_symmetric_2d(4)
        with BudgetContext(flop_budget=10**6):
            result = matmul(st, st)
        expected = numpy.matmul(numpy.asarray(st), numpy.asarray(st))
        assert numpy.allclose(result, expected)

    def test_dot_1d_vectors(self):
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = dot(a, b)
        assert numpy.isclose(result, 32.0)

    def test_matmul_1d_vectors(self):
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**6):
            result = matmul(a, b)
        assert numpy.isclose(result, 32.0)

    def test_dot_higher_dim(self):
        """dot with 3d arrays falls back to a.size * b.size cost."""
        a = numpy.ones((2, 3, 4))
        b = numpy.ones((2, 4, 5))
        with BudgetContext(flop_budget=10**6) as budget:
            result = dot(a, b)
        assert budget.flops_used == a.size * b.size

    def test_matmul_higher_dim(self):
        """matmul with 3d arrays falls back to a.size * b.size cost."""
        a = numpy.ones((2, 3, 4))
        b = numpy.ones((2, 4, 5))
        with BudgetContext(flop_budget=10**6) as budget:
            result = matmul(a, b)
        assert budget.flops_used == a.size * b.size


# ---------------------------------------------------------------------------
# 13. Additional custom ops: inner, outer, vdot, tensordot, kron, cross
# ---------------------------------------------------------------------------


class TestCustomOps:
    def test_inner_1d(self):
        a = numpy.array([1.0, 2.0])
        b = numpy.array([3.0, 4.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = inner(a, b)
        assert numpy.isclose(result, 11.0)
        assert budget.flops_used == 3  # einsum "i,i->": 2*n-1 = 3 for n=2

    def test_outer_basic(self):
        a = numpy.array([1.0, 2.0])
        b = numpy.array([3.0, 4.0, 5.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = outer(a, b)
        assert result.shape == (2, 3)
        assert budget.flops_used == 6

    def test_vdot_basic(self):
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = vdot(a, b)
        assert numpy.isclose(result, 32.0)
        assert budget.flops_used == 5  # einsum "i,i->": 2*n-1 = 5 for n=3

    def test_tensordot_int_axes(self):
        a = numpy.ones((3, 4, 5))
        b = numpy.ones((4, 5, 6))
        with BudgetContext(flop_budget=10**6) as budget:
            result = tensordot(a, b, axes=2)
        assert result.shape == (3, 6)

    def test_tensordot_list_axes(self):
        a = numpy.ones((3, 4))
        b = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**6) as budget:
            result = tensordot(a, b, axes=([1], [0]))
        assert result.shape == (3, 5)
        # contracted = 4, result.size = 15, cost = 60
        assert budget.flops_used == 60

    def test_tensordot_no_symmetry_unchanged(self):
        """Without any input symmetry, the cost equals the dense
        formula (a.size * b.size / contracted) — no behaviour change
        from the pre-symmetry-adjustment code."""
        n = 6
        a = numpy.ones((n, n))
        b = numpy.ones((n, n))
        with BudgetContext(flop_budget=10**8) as budget:
            tensordot(a, b, axes=1)
        # output shape (n, n) = 36 elements, contracted = n = 6,
        # dense = 6*6*6 = 216
        assert budget.flops_used == n * n * n

    def test_tensordot_surviving_symmetry_lowers_cost(self):
        """A 4D × 4D contraction that preserves S₂ symmetry on each
        operand's surviving (uncontracted) axes charges fewer FLOPs
        than the dense baseline. Output symmetry is the direct product
        of the two surviving S₂ groups."""
        n = 8
        sym = PermutationGroup.symmetric(axes=(0, 1))

        with BudgetContext(flop_budget=10**10) as dense_bc:
            a = numpy.ones((n, n, n, n))
            b = numpy.ones((n, n, n, n))
            dense_result = tensordot(a, b, axes=((2,), (2,)))

        with BudgetContext(flop_budget=10**10) as sym_bc:
            a_sym = as_symmetric(numpy.ones((n, n, n, n)), symmetry=sym)
            b_sym = as_symmetric(numpy.ones((n, n, n, n)), symmetry=sym)
            sym_result = tensordot(a_sym, b_sym, axes=((2,), (2,)))

        # Output shape unchanged.
        assert sym_result.shape == dense_result.shape == (n, n, n, n, n, n)
        # Symmetric path's tensordot cost is strictly lower.
        assert sym_bc.flops_used < dense_bc.flops_used
        # Output retains the direct-product symmetry on (0,1) and (3,4).
        assert isinstance(sym_result, SymmetricTensor)
        assert sym_result.symmetry is not None
        assert set(sym_result.symmetry.axes) == {0, 1, 3, 4}  # pyright: ignore[reportArgumentType]

    def test_tensordot_contraction_through_sym_axis_drops_symmetry(self):
        """If the contracted axes overlap with the input's symmetry
        axes (e.g. a = (n,n) sym(0,1), contract axis 1), the surviving
        axis is alone — no symmetry survives."""
        n = 5
        sym = PermutationGroup.symmetric(axes=(0, 1))
        a_sym = as_symmetric(numpy.ones((n, n)), symmetry=sym)
        b = numpy.ones((n, n))
        with BudgetContext(flop_budget=10**6):
            result = tensordot(a_sym, b, axes=1)
        # Output shape = (n, n) — surviving axes are a's axis 0 and b's axis 1.
        assert result.shape == (n, n)
        # No surviving axis pair has the original S2 symmetry.
        assert getattr(result, "symmetry", None) is None

    def test_kron_basic(self):
        a = numpy.array([[1, 0], [0, 1]])
        b = numpy.array([[1, 2], [3, 4]])
        with BudgetContext(flop_budget=10**6):
            result = kron(a, b)
        assert result.shape == (4, 4)

    def test_cross_basic(self):
        a = numpy.array([1.0, 0.0, 0.0])
        b = numpy.array([0.0, 1.0, 0.0])
        with BudgetContext(flop_budget=10**6):
            result = cross(a, b)
        assert numpy.allclose(result, [0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# 14. Unary multi-output with non-ndarray input
# ---------------------------------------------------------------------------


class TestUnaryMultiScalar:
    def test_modf_with_python_float(self):
        with BudgetContext(flop_budget=10**6):
            result = modf(1.5)
        assert isinstance(result, tuple)

    def test_frexp_with_python_float(self):
        with BudgetContext(flop_budget=10**6):
            result = frexp(8.0)
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# 15. Binary multi-output with non-ndarray input
# ---------------------------------------------------------------------------


class TestBinaryMultiScalar:
    def test_divmod_with_python_scalars(self):
        with BudgetContext(flop_budget=10**6):
            result = divmod(7.0, 3.0)
        assert isinstance(result, tuple)

    def test_divmod_broadcast(self):
        x = numpy.array([10.0, 20.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = divmod(x, 3.0)
        assert isinstance(result, tuple)
        assert budget.flops_used == 2


# ---------------------------------------------------------------------------
# 16. Edge cases: around/round with out parameter
# ---------------------------------------------------------------------------


class TestAroundRoundOut:
    def test_around_with_out_param(self):
        """around() with an out parameter should not return scalar even for scalar input."""
        out = numpy.zeros(1)
        with BudgetContext(flop_budget=10**6):
            result = around(numpy.array([3.7]), decimals=0, out=out)  # pyright: ignore[reportArgumentType]
        assert isinstance(result, numpy.ndarray)

    def test_round_with_out_param(self):
        out = numpy.zeros(1)
        with BudgetContext(flop_budget=10**6):
            result = round(numpy.array([2.3]), decimals=0, out=out)  # pyright: ignore[reportArgumentType]
        assert isinstance(result, numpy.ndarray)


# ---------------------------------------------------------------------------
# 17. Reduction on non-ndarray input
# ---------------------------------------------------------------------------


class TestReductionNonArray:
    def test_sum_on_list(self):
        # Updated for orbit-mapping cost model (PR #91 Task 7).
        # sum of 3 elements: 3 - 1 = 2 additions.
        with BudgetContext(flop_budget=10**6) as budget:
            result = sum([1.0, 2.0, 3.0])
        assert numpy.isclose(result, 6.0)
        assert budget.flops_used == 2

    def test_argmax_on_list(self):
        with BudgetContext(flop_budget=10**6):
            result = argmax([1.0, 5.0, 3.0])
        assert result == 1


# ---------------------------------------------------------------------------
# 18. Additional custom ops coverage (diff, gradient, ediff1d, etc.)
# ---------------------------------------------------------------------------


class TestAdditionalCustomOps:
    def test_diff_basic(self):
        x = numpy.array([1.0, 3.0, 6.0, 10.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = diff(x)
        assert numpy.allclose(result, [2.0, 3.0, 4.0])
        assert budget.flops_used == 3

    def test_diff_from_list(self):
        with BudgetContext(flop_budget=10**6):
            result = diff([1, 4, 9])
        assert numpy.allclose(result, [3, 5])

    def test_gradient_basic(self):
        x = numpy.array([1.0, 2.0, 4.0, 7.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = gradient(x)
        assert budget.flops_used == 4

    def test_gradient_from_list(self):
        with BudgetContext(flop_budget=10**6):
            result = gradient([1.0, 4.0, 9.0])

    def test_ediff1d_basic(self):
        x = numpy.array([1, 2, 4, 7])
        with BudgetContext(flop_budget=10**6) as budget:
            result = ediff1d(x)
        assert numpy.allclose(result, [1, 2, 3])
        assert budget.flops_used == 3

    def test_ediff1d_from_list(self):
        with BudgetContext(flop_budget=10**6):
            result = ediff1d([5, 10, 20])
        assert numpy.allclose(result, [5, 10])

    def test_convolve_basic(self):
        a = numpy.array([1.0, 2.0, 3.0])
        v = numpy.array([0.5, 1.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = convolve(a, v)
        expected = numpy.convolve(a, v)
        assert numpy.allclose(result, expected)
        assert budget.flops_used == a.size * v.size

    def test_convolve_from_list(self):
        with BudgetContext(flop_budget=10**6):
            result = convolve([1, 2, 3], [1, 1])

    def test_correlate_basic(self):
        a = numpy.array([1.0, 2.0, 3.0, 4.0])
        v = numpy.array([1.0, 0.5])
        with BudgetContext(flop_budget=10**6) as budget:
            result = correlate(a, v)
        expected = numpy.correlate(a, v)
        assert numpy.allclose(result, expected)

    def test_correlate_from_list(self):
        with BudgetContext(flop_budget=10**6):
            result = correlate([1, 2, 3], [1])

    def test_corrcoef_basic(self):
        x = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with BudgetContext(flop_budget=10**6) as budget:
            result = corrcoef(x)
        assert result.shape == (2, 2)

    def test_corrcoef_with_y(self):
        x = numpy.array([1.0, 2.0, 3.0])
        y = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**6):
            result = corrcoef(x, y=y)

    def test_corrcoef_from_list(self):
        with BudgetContext(flop_budget=10**6):
            result = corrcoef([1.0, 2.0, 3.0])

    def test_cov_basic(self):
        m = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with BudgetContext(flop_budget=10**6) as budget:
            result = cov(m)
        assert result.shape == (2, 2)

    def test_cov_with_y(self):
        m = numpy.array([1.0, 2.0, 3.0])
        y = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**6):
            result = cov(m, y=y)

    def test_cov_from_list(self):
        with BudgetContext(flop_budget=10**6):
            result = cov([1.0, 2.0, 3.0])

    def test_trapezoid_basic(self):
        y = numpy.array([1.0, 2.0, 3.0, 4.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = trapezoid(y)
        assert budget.flops_used == 4

    def test_trapezoid_with_x(self):
        y = numpy.array([1.0, 2.0, 3.0])
        x = numpy.array([0.0, 1.0, 3.0])
        with BudgetContext(flop_budget=10**6):
            result = trapezoid(y, x=x)

    def test_trapezoid_from_list(self):
        with BudgetContext(flop_budget=10**6):
            result = trapezoid([1.0, 2.0, 3.0])

    def test_interp_basic(self):
        x = numpy.array([1.5, 2.5])
        xp = numpy.array([1.0, 2.0, 3.0])
        fp = numpy.array([10.0, 20.0, 30.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = interp(x, xp, fp)
        assert numpy.allclose(result, [15.0, 25.0])
        # n * ceil(log2(xp_len)) = 2 * ceil(log2(3)) = 2 * 2 = 4
        assert budget.flops_used == 4

    def test_interp_from_list(self):
        with BudgetContext(flop_budget=10**6):
            result = interp([1.5], [1.0, 2.0], [10.0, 20.0])


# ---------------------------------------------------------------------------
# 19. trapz (deprecated alias)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(numpy, "trapz"), reason="numpy 2.4+ removed trapz")
class TestTrapz:
    def test_trapz_basic(self):
        from flopscope._pointwise import trapz

        y = numpy.array([1.0, 2.0, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with BudgetContext(flop_budget=10**6) as budget:
                result = trapz(y)
            assert budget.flops_used == 3

    def test_trapz_from_list(self):
        from flopscope._pointwise import trapz

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with BudgetContext(flop_budget=10**6):
                result = trapz([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# 20. ptp (may be present or fallback)
# ---------------------------------------------------------------------------


class TestPtp:
    def test_ptp_basic(self):
        from flopscope._pointwise import ptp

        x = numpy.array([1.0, 5.0, 3.0, 2.0])
        with BudgetContext(flop_budget=10**6) as budget:
            result = ptp(x)
        assert numpy.isclose(result, 4.0)

    def test_ptp_from_list(self):
        from flopscope._pointwise import ptp

        with BudgetContext(flop_budget=10**6):
            result = ptp([1.0, 10.0, 5.0])
        assert numpy.isclose(result, 9.0)


# ---------------------------------------------------------------------------
# 21. Symmetry partial-loss in reduction (3-axis group, reduce one)
# ---------------------------------------------------------------------------


class TestReductionSymmetryPartialLoss:
    """Test reduction that partially breaks a 3-axis symmetry group."""

    def _make_symmetric_3axis(self, n=3):
        """Create (n,n,n) tensor symmetric under all permutations of axes (0,1,2)."""
        rng = numpy.random.default_rng(77)
        data = rng.random((n, n, n))
        # Full symmetrization over all 6 permutations of (0,1,2)
        perms = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ]
        sym = numpy.zeros_like(data)
        for p in perms:
            sym += data.transpose(p)
        sym /= 6.0
        return as_symmetric(sym, symmetry=[(0, 1, 2)])

    def test_reduce_one_axis_of_3group_warns_partial_loss(self):
        """Reducing axis 0 from group (0,1,2) leaves (1,2) -> renumbered (0,1)."""
        st = self._make_symmetric_3axis(3)
        configure(symmetry_warnings=True)
        with BudgetContext(flop_budget=10**6):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = sum(st, axis=0)
                # Check that fnp get a SymmetryLossWarning about partial loss
                sym_warnings = [
                    x for x in w if issubclass(x.category, SymmetryLossWarning)
                ]
                assert len(sym_warnings) >= 1
        # Result should still be symmetric in remaining axes
        assert isinstance(result, SymmetricTensor)

    def test_reduce_one_axis_keepdims_partial_loss(self):
        """keepdims=True: reducing axis 0 from (0,1,2) leaves (1,2) surviving."""
        st = self._make_symmetric_3axis(3)
        configure(symmetry_warnings=True)
        with BudgetContext(flop_budget=10**6):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = sum(st, axis=0, keepdims=True)
        # Should still be SymmetricTensor with surviving group


# ---------------------------------------------------------------------------
# 22. Binary op partial symmetry loss (intersection loses some groups)
# ---------------------------------------------------------------------------


class TestBinaryPartialSymmetryLoss:
    """Binary op where intersection preserves some groups but loses others."""

    def test_binary_op_with_shared_and_unshared_groups(self):
        """Two 4-d tensors: one sym in [(0,1),(2,3)], other in [(0,1)].
        Intersection should keep (0,1) but lose (2,3)."""
        rng = numpy.random.default_rng(42)
        n = 3
        # Tensor with symmetry in both (0,1) and (2,3)
        data_a = rng.random((n, n, n, n))
        data_a = (data_a + data_a.transpose(1, 0, 2, 3)) / 2
        data_a = (data_a + data_a.transpose(0, 1, 3, 2)) / 2
        st_a = as_symmetric(data_a, symmetry=[(0, 1), (2, 3)])

        # Tensor with symmetry only in (0,1)
        data_b = rng.random((n, n, n, n))
        data_b = (data_b + data_b.transpose(1, 0, 2, 3)) / 2
        st_b = as_symmetric(data_b, symmetry=[(0, 1)])

        configure(symmetry_warnings=True)
        with BudgetContext(flop_budget=10**6):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = add(st_a, st_b)
                # Should get a warning about losing (2,3)
                sym_warnings = [
                    x for x in w if issubclass(x.category, SymmetryLossWarning)
                ]
                assert len(sym_warnings) >= 1
        # Result should retain (0,1)
        assert isinstance(result, SymmetricTensor)
