"""Tests for automatic symmetric annotation on tensor creation operations."""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._symmetric import SymmetricTensor


def _s2():
    return flops.SymmetryGroup.symmetric(axes=(0, 1))


def _young_2x2():
    return flops.SymmetryGroup.young(blocks=((0, 1), (2, 3)))


@pytest.mark.parametrize(
    ("factory", "args", "expected_symmetry"),
    [
        (fnp.eye, (3,), _s2()),
        (fnp.identity, (3,), _s2()),
        (fnp.diag, ([1, 2, 3],), _s2()),
        (fnp.diagflat, ([1, 2, 3],), _s2()),
    ],
)
def test_structural_constructors_keep_current_exact_symmetry(
    factory, args, expected_symmetry
):
    result = factory(*args)
    assert isinstance(result, SymmetricTensor)
    assert result.symmetry == expected_symmetry


# ---------------------------------------------------------------------------
# Tier 1: Always symmetric (identity / diagonal)
# ---------------------------------------------------------------------------


class TestEye:
    def test_eye_still_returns_symmetric_tensor_with_exact_group(self):
        result = fnp.eye(3)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()

    def test_square_is_symmetric(self):
        result = fnp.eye(3)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        np.testing.assert_array_equal(result, np.eye(3))

    def test_square_explicit_M_equal(self):
        result = fnp.eye(4, M=4)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()

    def test_non_square_not_symmetric(self):
        result = fnp.eye(3, M=4)
        assert not isinstance(result, SymmetricTensor)

    def test_off_diagonal_not_symmetric(self):
        result = fnp.eye(3, k=1)
        assert not isinstance(result, SymmetricTensor)

    def test_off_diagonal_non_square_not_symmetric(self):
        result = fnp.eye(3, M=4, k=2)
        assert not isinstance(result, SymmetricTensor)

    def test_dtype_preserved(self):
        result = fnp.eye(3, dtype=np.float32)
        assert isinstance(result, SymmetricTensor)
        assert result.dtype == np.float32


class TestIdentity:
    def test_returns_symmetric(self):
        result = fnp.identity(4)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        np.testing.assert_array_equal(result, np.identity(4))

    def test_size_1(self):
        result = fnp.identity(1)
        assert isinstance(result, SymmetricTensor)


class TestDiag:
    def test_1d_input_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        result = fnp.diag(v)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        np.testing.assert_array_equal(result, np.diag(v))

    def test_1d_off_diagonal_not_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        result = fnp.diag(v, k=1)
        assert not isinstance(result, SymmetricTensor)

    def test_2d_input_extracts_diagonal(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = fnp.diag(m)
        assert not isinstance(result, SymmetricTensor)
        assert result.ndim == 1

    def test_list_input(self):
        result = fnp.diag([1, 2, 3])
        assert isinstance(result, SymmetricTensor)


class TestDiagflat:
    def test_returns_symmetric(self):
        result = fnp.diagflat([1, 2, 3])
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        np.testing.assert_array_equal(result, np.diagflat([1, 2, 3]))

    def test_off_diagonal_not_symmetric(self):
        result = fnp.diagflat([1, 2, 3], k=1)
        assert not isinstance(result, SymmetricTensor)

    def test_2d_input_flattened(self):
        result = fnp.diagflat([[1, 2], [3, 4]])
        assert isinstance(result, SymmetricTensor)
        assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# Tier 2: Constant-fill with square shape
# ---------------------------------------------------------------------------


class TestZeros:
    def test_square_symmetric(self):
        result = fnp.zeros((3, 3))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()

    def test_non_square_not_symmetric(self):
        result = fnp.zeros((3, 4))
        assert not isinstance(result, SymmetricTensor)

    def test_1d_not_symmetric(self):
        result = fnp.zeros((5,))
        assert not isinstance(result, SymmetricTensor)

    def test_3d_equal_extents_infers_full_symmetry(self):
        result = fnp.zeros((3, 3, 3))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == flops.SymmetryGroup.symmetric(axes=(0, 1, 2))

    def test_distinct_extents_stay_plain(self):
        result = fnp.zeros((3, 4, 5))
        assert not isinstance(result, SymmetricTensor)

    def test_scalar_not_symmetric(self):
        result = fnp.zeros(())
        assert not isinstance(result, SymmetricTensor)


class TestOnes:
    def test_square_symmetric(self):
        result = fnp.ones((4, 4))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()

    def test_non_square_not_symmetric(self):
        result = fnp.ones((2, 3))
        assert not isinstance(result, SymmetricTensor)

    def test_1d_not_symmetric(self):
        result = fnp.ones((10,))
        assert not isinstance(result, SymmetricTensor)

    def test_single_equal_extent_block_infers_partial_symmetry(self):
        result = fnp.ones((3, 3, 2))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == flops.SymmetryGroup.symmetric(axes=(0, 1))


class TestFull:
    def test_square_symmetric(self):
        result = fnp.full((3, 3), 5.0)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        assert np.all(result == 5.0)

    def test_non_square_not_symmetric(self):
        result = fnp.full((3, 4), 5.0)
        assert not isinstance(result, SymmetricTensor)

    def test_two_equal_extent_blocks_infer_young_symmetry(self):
        result = fnp.full((3, 3, 2, 2), 7.0)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == flops.SymmetryGroup.young(blocks=((0, 1), (2, 3)))
        assert np.all(result == 7.0)

    def test_high_rank_scalar_comparison_keeps_symmetry(self):
        result = fnp.full((2, 2, 2, 2, 2, 2, 2, 2), 1.0)
        compared = result == 1.0
        assert isinstance(compared, SymmetricTensor)
        assert compared.symmetry == result.symmetry  # pyright: ignore[reportAttributeAccessIssue]
        assert np.all(compared)


# ---------------------------------------------------------------------------
# Tier 3: Propagation from input / shape detection
# ---------------------------------------------------------------------------


class TestZerosLike:
    def test_propagates_from_symmetric_tensor(self):
        S = flops.as_symmetric(np.eye(3), symmetry=(0, 1))
        result = fnp.zeros_like(S)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        np.testing.assert_array_equal(result, np.zeros((3, 3)))

    def test_plain_input_non_square_not_symmetric(self):
        a = np.ones((3, 4))
        result = fnp.zeros_like(a)
        assert not isinstance(result, SymmetricTensor)

    def test_plain_input_shape_infers_constant_fill_symmetry(self):
        result = fnp.zeros_like(np.empty((4, 4, 2)))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == flops.SymmetryGroup.symmetric(axes=(0, 1))

    def test_propagates_multi_group(self):
        data = np.zeros((3, 3, 3, 3))
        S = flops.as_symmetric(data, symmetry=((0, 1), (2, 3)))
        result = fnp.zeros_like(S)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _young_2x2()


class TestOnesLike:
    def test_propagates_from_symmetric_tensor(self):
        S = flops.as_symmetric(np.eye(4), symmetry=(0, 1))
        result = fnp.ones_like(S)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()

    def test_plain_input_non_square_not_symmetric(self):
        result = fnp.ones_like(np.zeros((5, 4)))
        assert not isinstance(result, SymmetricTensor)

    def test_plain_input_shape_infers_young_symmetry(self):
        result = fnp.ones_like(np.empty((2, 2, 5, 5)))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == flops.SymmetryGroup.young(blocks=((0, 1), (2, 3)))


class TestFullLike:
    def test_propagates_from_symmetric_tensor(self):
        S = flops.as_symmetric(np.eye(3), symmetry=(0, 1))
        result = fnp.full_like(S, 7.0)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        assert np.all(result == 7.0)

    def test_non_square_not_symmetric(self):
        result = fnp.full_like(np.ones((2, 3)), 1.0)
        assert not isinstance(result, SymmetricTensor)

    def test_incompatible_carried_symmetry_falls_back_to_shape_inference(self):
        source = flops.as_symmetric(np.eye(3), symmetry=(0, 1))
        result = fnp.full_like(source, 1.0, shape=(2, 3, 5, 5, 7, 7))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == flops.SymmetryGroup.young(blocks=((2, 3), (4, 5)))

    def test_shape_override_preserves_compatible_carried_symmetry(self):
        source = flops.as_symmetric(np.zeros((3, 3, 2)), symmetry=(0, 1))
        result = fnp.zeros_like(source, shape=(5, 5, 5))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()

    @pytest.mark.parametrize(
        ("factory", "args"),
        [
            (fnp.zeros_like, ()),
            (fnp.ones_like, ()),
            (fnp.full_like, (5.0,)),
        ],
    )
    def test_shape_override_drops_incompatible_symmetry(self, factory, args):
        source = flops.as_symmetric(np.eye(3), symmetry=(0, 1))
        result = factory(source, *args, shape=(2, 3))
        assert result.shape == (2, 3)
        assert not isinstance(result, SymmetricTensor)


# ---------------------------------------------------------------------------
# Integration: symmetry flows through downstream operations
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_eye_through_unary_pointwise(self):
        with flops.BudgetContext(flop_budget=10**6) as budget:
            eye5 = fnp.eye(5)
            result = fnp.exp(eye5)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == _s2()
            # 5*6/2 = 15 unique elements
            assert budget.flops_used == 15

    def test_zeros_add_symmetric_preserves(self):
        with flops.BudgetContext(flop_budget=10**6):
            Z = fnp.zeros((3, 3))
            S = flops.as_symmetric(
                np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float),
                symmetry=(0, 1),
            )
            result = fnp.add(Z, S)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == _s2()

    def test_identity_inv_round_trip(self):
        with flops.BudgetContext(flop_budget=10**6):
            ident = fnp.identity(4)
            inv_ident = fnp.linalg.inv(ident)
            assert isinstance(inv_ident, SymmetricTensor)
            np.testing.assert_allclose(inv_ident, np.eye(4))

    def test_issue_42_broadcast_and_pointwise_keep_block_symmetry(self):
        n = 3
        A = flops.as_symmetric(
            np.ones((n, n, n, n)),
            symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1, 2, 3)),
        )
        B = flops.as_symmetric(np.ones((n, n)), symmetry=(0, 1))

        B_bc = fnp.broadcast_to(B, A.shape)
        assert isinstance(B_bc, SymmetricTensor)
        assert B_bc.symmetry == _young_2x2()

        result = fnp.multiply(A, B)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _young_2x2()

    def test_batched_inv_preserves_exact_group(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        B = np.array([[5.0, 0.5], [0.5, 2.5]])
        C = np.array([[6.0, 1.5], [1.5, 4.0]])
        batched = np.empty((2, 2, 2, 2), dtype=float)
        batched[0, 0] = A
        batched[0, 1] = B
        batched[1, 0] = B
        batched[1, 1] = C

        exact = flops.as_symmetric(batched, symmetry=((0, 1), (2, 3)))

        with flops.BudgetContext(flop_budget=10**6):
            result = fnp.linalg.inv(exact)

        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == exact.symmetry

    def test_creation_ops_cost(self):
        with flops.BudgetContext(flop_budget=10**6) as budget:
            fnp.eye(10)  # free
            fnp.identity(10)  # free
            fnp.zeros((10, 10))  # free
            fnp.ones((10, 10))  # free
            fnp.full((10, 10), 3.14)  # numel(output)=100
            fnp.diag(np.arange(10, dtype=float))  # numel(output)=100
            fnp.diagflat(np.arange(5, dtype=float))  # numel(output)=25
            assert budget.flops_used == 100 + 100 + 25


class TestSymmetryInferredSlot:
    """The `_symmetry_inferred` slot is present and defaults to False."""

    def test_slot_present_and_defaults_false(self):
        import numpy as np

        import flopscope as flops
        from flopscope._symmetric import SymmetricTensor

        arr = SymmetricTensor(
            np.zeros((3, 3)),
            symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
        )
        assert arr._symmetry_inferred is False


class TestSymmetryInferredSlotPickleRoundtrip:
    """Pickle round-trip must not leave `_symmetry_inferred` uninitialized.

    The spec says the marker's pickle-survival is unspecified — but the slot
    must always be accessible after unpickling (defaulting to False if not
    preserved is fine; AttributeError is not).
    """

    def test_pickle_roundtrip_does_not_raise_on_attribute_access(self):
        import pickle

        import numpy as np

        import flopscope as flops
        from flopscope._symmetric import SymmetricTensor

        arr = SymmetricTensor(
            np.zeros((3, 3)),
            symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
        )
        restored = pickle.loads(pickle.dumps(arr))
        # The slot must exist and be readable. Either False or True is OK
        # per spec; AttributeError is not.
        value = restored._symmetry_inferred
        assert value in (True, False)


class TestSymmetryInferredSlotCopyDoesNotPropagate:
    """copy() must not propagate the inferred marker (spec: marker is per-array)."""

    def test_copy_of_inferred_array_clears_marker(self):
        import numpy as np

        import flopscope as flops
        from flopscope._symmetric import SymmetricTensor

        arr = SymmetricTensor(
            np.zeros((3, 3)),
            symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
        )
        # Manually flip the marker to simulate an inferred-symmetry array
        # (the helper that does this is in Task 3 — here we set it directly
        # so we can exercise the copy path in isolation).
        arr._symmetry_inferred = True
        copy = arr.copy()
        assert copy._symmetry_inferred is False


class TestWrapWithInferredSymmetry:
    def test_sets_marker(self):
        import numpy as np

        import flopscope as flops
        from flopscope._symmetric import SymmetricTensor
        from flopscope._symmetry_utils import wrap_with_inferred_symmetry

        result = wrap_with_inferred_symmetry(
            np.zeros((3, 3)),
            flops.SymmetryGroup.symmetric(axes=(0, 1)),
        )
        assert isinstance(result, SymmetricTensor)
        assert result._symmetry_inferred is True
        assert result.symmetry == flops.SymmetryGroup.symmetric(axes=(0, 1))

    def test_none_symmetry_returns_plain_ndarray(self):
        import numpy as np

        from flopscope._symmetric import SymmetricTensor
        from flopscope._symmetry_utils import wrap_with_inferred_symmetry

        result = wrap_with_inferred_symmetry(np.zeros((3, 4)), None)
        assert not isinstance(result, SymmetricTensor)
