"""A symmetry tag must assert no more than the buffer supports.

Validation accepts data that is symmetric *within a tolerance*, but the cost
model reads the tag it grants as exact: every position in an orbit after the
first is priced as a redundant degree of freedom, and the buffer is never read
again. Those two readings disagree for any buffer whose orbit entries merely
agree closely, and the gap is wide enough to carry independent values through
it -- scale a tensor down until its differences fall under ``atol``, collect
the tag, scale back up.

Canonicalizing at the boundary is what makes the two readings agree: one entry
per orbit survives, so whatever was hidden in the others is gone before the tag
exists. These tests pin that property (exact invariance, zero tolerance), the
representative rule that decides which entry survives, and the two things the
fix must not cost -- the Reynolds path's behaviour, and a copy on data that is
already exact.
"""

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._canonical_symmetry import (
    _canonical_map_cached,
    canonical_copy,
    is_exactly_invariant,
)
from flopscope._perm_group import SymmetryGroup, _Permutation

BUDGET = 10**14

# A power of two, so scaling down and back up is exact in binary floating
# point and cannot itself be blamed for any difference we observe.
SCALE = 2.0**-40


def _budget():
    return flops.BudgetContext(flop_budget=BUDGET, quiet=True)


def _generator_perms(group, ndim):
    """Full-rank transpose permutation for each non-identity generator."""
    axes = group.axes if group.axes is not None else tuple(range(group.degree))
    perms = []
    for gen in group.generators:
        if gen.is_identity:
            continue
        perm = list(range(ndim))
        for i in range(group.degree):
            perm[axes[i]] = axes[gen.array_form[i]]
        perms.append(tuple(perm))
    return perms


def _assert_exactly_invariant(array, group):
    """Invariant under every generator with NO tolerance whatsoever."""
    raw = np.asarray(array)
    for perm in _generator_perms(group, raw.ndim):
        assert np.array_equal(raw, np.transpose(raw, perm)), (
            f"not exactly invariant under generator permutation {perm}"
        )


def _custom_group(axes=(0, 1, 2)):
    """A group defined by raw generators rather than a named constructor."""
    return SymmetryGroup(
        _Permutation([1, 2, 0]), _Permutation([1, 0, 2]), axes=tuple(axes)
    )


GROUPS = [
    pytest.param(SymmetryGroup.symmetric(axes=(0, 1)), (5, 5), id="symmetric-S2"),
    pytest.param(SymmetryGroup.symmetric(axes=(0, 1, 2)), (4, 4, 4), id="symmetric-S3"),
    pytest.param(SymmetryGroup.cyclic(axes=(0, 1, 2)), (4, 4, 4), id="cyclic-C3"),
    pytest.param(
        SymmetryGroup.dihedral(axes=(0, 1, 2, 3)), (3, 3, 3, 3), id="dihedral-D4"
    ),
    pytest.param(_custom_group(), (4, 4, 4), id="custom-generators"),
    pytest.param(
        SymmetryGroup.symmetric(axes=(1, 2)), (2, 4, 4), id="symmetric-inner-axes"
    ),
]


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------


class TestToleranceGapIsClosed:
    def test_scaled_down_asymmetric_data_is_accepted_but_not_kept(self):
        """The tolerant check still admits it; the tag no longer over-claims.

        The pre-fix behaviour was that both halves of this test's premise
        held at once: validation passed AND the orbit still held two
        different values. Only the first may survive.
        """
        group = SymmetryGroup.symmetric(axes=(0, 1))
        raw = np.array([[1.0, 2.0], [9.0, 3.0]])

        # The claim is false at full scale, and is refused.
        with _budget(), pytest.raises(flops.SymmetryError):
            flops.as_symmetric(raw, symmetry=group)

        # Scaled down, the same claim passes the tolerance policy unchanged.
        scaled = raw * SCALE
        assert np.allclose(scaled, scaled.T, atol=1e-6, rtol=1e-5)
        with _budget():
            tagged = flops.as_symmetric(scaled, symmetry=group)

        values = np.asarray(tagged)
        assert values[0, 1] == values[1, 0], (
            "orbit still holds two different values under a tag read as exact"
        )
        _assert_exactly_invariant(values, group)

    def test_scaling_back_up_cannot_recover_the_hidden_value(self):
        """The round trip the defect depended on now yields nothing."""
        group = SymmetryGroup.symmetric(axes=(0, 1))
        raw = np.array([[1.0, 2.0], [9.0, 3.0]])

        with _budget():
            tagged = flops.as_symmetric(raw * SCALE, symmetry=group)
            restored = tagged * (1.0 / SCALE)

        assert not np.array_equal(np.asarray(restored), raw)
        _assert_exactly_invariant(restored, group)

    @pytest.mark.parametrize("group,shape", GROUPS)
    def test_round_trip_stays_exact_for_every_group(self, group, shape):
        rng = np.random.default_rng(11)
        asymmetric = rng.standard_normal(shape) * SCALE

        with _budget():
            tagged = flops.as_symmetric(asymmetric, symmetry=group)
            restored = tagged * (1.0 / SCALE)

        _assert_exactly_invariant(tagged, group)
        _assert_exactly_invariant(restored, group)

    def test_bare_constructor_is_the_same_boundary(self):
        """``SymmetricTensor(...)`` is ingress too, and canonicalizes as well."""
        from flopscope._symmetric import SymmetricTensor

        group = SymmetryGroup.symmetric(axes=(0, 1))
        with _budget():
            tagged = SymmetricTensor(
                np.array([[1.0, 2.0], [9.0, 3.0]]) * SCALE, symmetry=group
            )
        _assert_exactly_invariant(tagged, group)

    @pytest.mark.parametrize(
        "wrapper_name",
        [
            "wrap_with_symmetry",
            "wrap_with_derived_symmetry",
            "wrap_with_inferred_symmetry",
        ],
    )
    def test_importable_wrappers_cannot_mint_a_free_tag(self, wrapper_name):
        """Only one wrapper is trusted, and these are not it.

        Each of these is exempt from validation when it runs inside a counted
        op, which is where the package uses them. Called directly, from
        outside any flopscope op, they must be checked and charged -- so none
        of them is a way around the constructor for anyone who imports it.
        """
        import flopscope._symmetry_utils as symmetry_utils

        wrapper = getattr(symmetry_utils, wrapper_name)
        group = SymmetryGroup.symmetric(axes=(0, 1))
        asymmetric = np.random.default_rng(3).random((6, 6))
        with _budget(), pytest.raises(flops.SymmetryError):
            wrapper(asymmetric, group)

    def test_the_one_trusted_wrapper_is_still_only_reachable_by_import(self):
        """``wrap_with_trusted_symmetry`` stays trusted, and is the only one.

        Two internal sites cannot inherit a counted frame and so genuinely
        need it. Pinned here so that list stays short and deliberate: this is
        the single remaining route to an unchecked tag, and it should not grow
        a fourth member by accident.
        """
        from flopscope._symmetric import _TRUSTED_SYMMETRY_WRAPPER_CODES
        from flopscope._symmetry_utils import wrap_with_trusted_symmetry

        assert _TRUSTED_SYMMETRY_WRAPPER_CODES == frozenset(
            {wrap_with_trusted_symmetry.__code__}
        )

    def test_the_trusted_wrapper_does_attach_without_checking(self):
        """Stated outright rather than left for someone to discover.

        This is the one route that still tags a buffer nobody looked at, and
        the reason it survives is that ``_build_symmetric_proxy`` and
        ``matrix_transpose`` genuinely need it. It is a private import, absent
        from the op registry and from both public namespaces, so it is not
        reachable from a submission -- but it is reachable in-process, and a
        test that quietly omitted it would read as though nothing were open.
        Closing it needs a capability the caller cannot forge, which is a
        larger change than this one.
        """
        from flopscope._symmetry_utils import wrap_with_trusted_symmetry

        group = SymmetryGroup.symmetric(axes=(0, 1))
        asymmetric = np.random.default_rng(5).random((6, 6))
        with _budget() as budget:
            tagged = wrap_with_trusted_symmetry(asymmetric, group)
        assert tagged.symmetry is not None
        assert budget.flops_used == 0
        assert not is_exactly_invariant(np.asarray(tagged), group)

        # It is not reachable by name from either public namespace.
        assert not hasattr(flops, "wrap_with_trusted_symmetry")
        assert not hasattr(fnp, "wrap_with_trusted_symmetry")
        from flopscope._registry import REGISTRY

        assert "wrap_with_trusted_symmetry" not in REGISTRY

    def test_matrix_transpose_stays_free(self):
        """The registered-free transform must not start paying to keep its tag."""
        group = SymmetryGroup.symmetric(axes=(0, 1))
        base = np.random.default_rng(7).standard_normal((16, 16))
        with _budget() as budget:
            tagged = flops.as_symmetric((base + base.T) / 2, symmetry=group)
            before = budget.flops_used
            transposed = fnp.matrix_transpose(tagged)
            assert budget.flops_used == before
        assert transposed.symmetry is not None


# ---------------------------------------------------------------------------
# Which value survives
# ---------------------------------------------------------------------------


class TestCanonicalRepresentative:
    def test_keeps_the_first_entry_rather_than_averaging(self):
        """The documented rule, and the one case that tells the modes apart."""
        group = SymmetryGroup.symmetric(axes=(0, 1))
        raw = np.array([[1.0, 2.0], [9.0, 3.0]])

        with _budget():
            copied = flops.symmetrize(raw, symmetry=group, mode="canonical-copy")

        np.testing.assert_array_equal(
            np.asarray(copied), np.array([[1.0, 2.0], [2.0, 3.0]])
        )
        # Emphatically not the Reynolds answer, whose off-diagonal is 5.5.
        assert np.asarray(copied)[0, 1] != 5.5

    def test_representative_is_the_lexicographically_smallest_index(self):
        """Every orbit resolves to its smallest C-order flat index."""
        group = SymmetryGroup.symmetric(axes=(0, 1, 2))
        shape = (3, 3, 3)
        flat = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)

        with _budget():
            copied = np.asarray(
                flops.symmetrize(flat, symmetry=group, mode="canonical-copy")
            )

        for index in np.ndindex(shape):
            orbit_min = min(
                int(flat[tuple(index[p] for p in perm)])
                for perm in [
                    (0, 1, 2),
                    (0, 2, 1),
                    (1, 0, 2),
                    (1, 2, 0),
                    (2, 0, 1),
                    (2, 1, 0),
                ]
            )
            assert copied[index] == orbit_min

    def test_discarded_entries_cannot_influence_the_result(self):
        """Two inputs differing only off-representative agree afterwards."""
        group = SymmetryGroup.symmetric(axes=(0, 1))
        a = np.array([[1.0, 2.0], [9.0, 3.0]])
        b = np.array([[1.0, 2.0], [-500.0, 3.0]])

        with _budget():
            ca = flops.symmetrize(a, symmetry=group, mode="canonical-copy")
            cb = flops.symmetrize(b, symmetry=group, mode="canonical-copy")

        np.testing.assert_array_equal(np.asarray(ca), np.asarray(cb))


# ---------------------------------------------------------------------------
# What must not change
# ---------------------------------------------------------------------------


class TestReynoldsUnchanged:
    def test_default_mode_still_averages(self):
        group = SymmetryGroup.symmetric(axes=(0, 1))
        raw = np.array([[1.0, 2.0], [9.0, 3.0]])
        with _budget():
            projected = flops.symmetrize(raw, symmetry=group)
        np.testing.assert_allclose(
            np.asarray(projected), np.array([[1.0, 5.5], [5.5, 3.0]])
        )

    def test_default_matches_explicit_reynolds_in_value_and_price(self):
        group = SymmetryGroup.symmetric(axes=(0, 1))
        raw = np.random.default_rng(5).standard_normal((6, 6))

        with _budget() as implicit:
            a = flops.symmetrize(raw, symmetry=group)
        with _budget() as explicit:
            b = flops.symmetrize(raw, symmetry=group, mode="reynolds-projection")

        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert implicit.flops_used == explicit.flops_used

    def test_reynolds_bills_the_group_order_and_canonical_copy_bills_numel(self):
        group = SymmetryGroup.symmetric(axes=(0, 1))
        raw = np.ones((8, 8))

        with _budget() as reynolds:
            flops.symmetrize(raw, symmetry=group)
        with _budget() as copied:
            flops.symmetrize(raw, symmetry=group, mode="canonical-copy")
        with _budget() as plain_copy:
            fnp.copy(fnp.asarray(raw))

        # Both scale by the same dtype rate, so their ratio is the model's.
        assert reynolds.flops_used == copied.flops_used * (group.order() + 1)
        # Copying one entry per orbit is a copy, and is priced as one.
        assert copied.flops_used == plain_copy.flops_used

    def test_unknown_mode_is_refused(self):
        with _budget(), pytest.raises(ValueError, match="unknown symmetrize mode"):
            flops.symmetrize(
                np.ones((4, 4)),
                symmetry=SymmetryGroup.symmetric(axes=(0, 1)),
                mode="nonsense",
            )


class TestAsSymmetricBilling:
    @pytest.mark.parametrize("group,shape", GROUPS)
    def test_price_does_not_depend_on_whether_canonicalization_ran(self, group, shape):
        """Enforcement is the library's own business, not a charge to the caller."""
        rng = np.random.default_rng(17)
        base = rng.standard_normal(shape)
        # Exactly invariant: canonicalization short-circuits.
        with _budget():
            exact = np.asarray(
                flops.symmetrize(base, symmetry=group, mode="canonical-copy")
            )
        # Only tolerantly invariant: canonicalization copies.
        approximate = rng.standard_normal(shape) * SCALE

        with _budget() as no_copy:
            flops.as_symmetric(exact, symmetry=group)
        with _budget() as with_copy:
            flops.as_symmetric(approximate, symmetry=group)

        assert no_copy.flops_used == with_copy.flops_used

    def test_caller_buffer_is_never_mutated(self):
        group = SymmetryGroup.symmetric(axes=(0, 1))
        raw = np.array([[1.0, 2.0], [9.0, 3.0]]) * SCALE
        before = raw.copy()
        with _budget():
            flops.as_symmetric(raw, symmetry=group)
        np.testing.assert_array_equal(raw, before)

    def test_inexact_input_detaches_from_the_caller_buffer(self):
        """The consequence of copy-on-inexact, stated so it is not a surprise.

        A tag built from data that only passed the tolerant check is a copy,
        so writing through it -- via ``out=`` -- no longer reaches the array
        the caller handed in. This is the price of the tag being true, and it
        applies only to inexact input; the exact case keeps its alias, which
        the neighbouring test pins.
        """
        group = SymmetryGroup.symmetric(axes=(0, 1))
        rng = np.random.default_rng(53)
        destination = rng.standard_normal((8, 8)) * SCALE
        assert np.allclose(destination, destination.T, atol=1e-6, rtol=1e-5)
        assert not is_exactly_invariant(destination, group)
        original = destination.copy()

        with _budget():
            tagged = flops.as_symmetric(destination, symmetry=group)
            source = flops.as_symmetric(np.ones((8, 8)), symmetry=group)
            fnp.exp(source, out=tagged)

        assert not np.shares_memory(np.asarray(tagged), destination)
        np.testing.assert_array_equal(destination, original)

    def test_exactly_symmetric_input_is_not_copied(self):
        """The honest case keeps its zero-copy view, and its memory layout."""
        group = SymmetryGroup.symmetric(axes=(0, 1))
        rng = np.random.default_rng(23)
        base = rng.standard_normal((6, 6))
        exact = (base + base.T) / 2
        assert is_exactly_invariant(exact, group)

        with _budget():
            tagged = flops.as_symmetric(exact, symmetry=group)

        assert np.shares_memory(np.asarray(tagged), exact)


# ---------------------------------------------------------------------------
# Dtypes
# ---------------------------------------------------------------------------


DTYPES = [
    np.float32,
    np.float64,
    np.int32,
    np.int64,
    np.uint8,
    np.bool_,
    np.complex64,
    np.complex128,
]


class TestDtypes:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_canonical_copy_preserves_dtype_exactly(self, dtype):
        """Unlike averaging, copying needs no promotion."""
        group = SymmetryGroup.symmetric(axes=(0, 1))
        raw = np.array([[1, 2], [9, 3]], dtype=dtype)

        with _budget():
            copied = flops.symmetrize(raw, symmetry=group, mode="canonical-copy")

        assert np.asarray(copied).dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_canonicalized_result_is_exactly_invariant(self, dtype):
        group = SymmetryGroup.symmetric(axes=(0, 1))
        raw = np.array([[1, 2], [9, 3]], dtype=dtype)
        with _budget():
            copied = flops.symmetrize(raw, symmetry=group, mode="canonical-copy")
        _assert_exactly_invariant(copied, group)

    def test_reynolds_still_promotes_where_it_must(self):
        """The averaging branch keeps its float64 accumulation."""
        group = SymmetryGroup.symmetric(axes=(0, 1))
        raw = np.array([[1.0, 2.0], [9.0, 3.0]], dtype=np.float32)
        with _budget():
            projected = flops.symmetrize(raw, symmetry=group)
        assert np.asarray(projected).dtype == np.float64


# ---------------------------------------------------------------------------
# The orbit map and its cache
# ---------------------------------------------------------------------------


class TestOrbitMapCache:
    def test_same_shape_different_action_do_not_share_a_map(self):
        """The key must separate groups that act differently on one shape."""
        shape = (4, 4, 4)
        rng = np.random.default_rng(29)
        raw = rng.standard_normal(shape)

        symmetric = canonical_copy(raw, SymmetryGroup.symmetric(axes=(0, 1, 2)))
        cyclic = canonical_copy(raw, SymmetryGroup.cyclic(axes=(0, 1, 2)))

        assert not np.array_equal(symmetric, cyclic)
        _assert_exactly_invariant(symmetric, SymmetryGroup.symmetric(axes=(0, 1, 2)))
        _assert_exactly_invariant(cyclic, SymmetryGroup.cyclic(axes=(0, 1, 2)))

    def test_same_group_different_axes_do_not_share_a_map(self):
        shape = (4, 4, 4)
        rng = np.random.default_rng(31)
        raw = rng.standard_normal(shape)

        on_01 = canonical_copy(raw, SymmetryGroup.symmetric(axes=(0, 1)))
        on_12 = canonical_copy(raw, SymmetryGroup.symmetric(axes=(1, 2)))

        assert not np.array_equal(on_01, on_12)
        _assert_exactly_invariant(on_01, SymmetryGroup.symmetric(axes=(0, 1)))
        _assert_exactly_invariant(on_12, SymmetryGroup.symmetric(axes=(1, 2)))

    def test_same_action_different_shape_do_not_share_a_map(self):
        group = SymmetryGroup.symmetric(axes=(0, 1))
        small = canonical_copy(np.arange(9.0).reshape(3, 3), group)
        large = canonical_copy(np.arange(16.0).reshape(4, 4), group)
        assert small.shape == (3, 3)
        assert large.shape == (4, 4)
        _assert_exactly_invariant(small, group)
        _assert_exactly_invariant(large, group)

    def test_repeated_calls_reuse_the_cached_map(self):
        group = SymmetryGroup.symmetric(axes=(0, 1))
        raw = np.ones((7, 7))
        canonical_copy(raw, group)
        before = _canonical_map_cached.cache_info()
        for _ in range(20):
            canonical_copy(raw, group)
        after = _canonical_map_cached.cache_info()
        assert after.misses == before.misses
        assert after.hits > before.hits

    def test_map_is_not_writeable(self):
        """A shared cached map must not be mutable through a caller's handle."""
        from flopscope._canonical_symmetry import canonical_map

        mapping = canonical_map((4, 4), SymmetryGroup.symmetric(axes=(0, 1)))
        with pytest.raises(ValueError):
            mapping[0] = 3

    def test_writeable_flag_cannot_be_re_enabled(self):
        """Read-only is not enough on its own.

        NumPy lets a caller flip ``writeable`` back on for an array that owns
        its data, so handing out the cached array itself would leave the map
        editable in place -- and a doctored map mis-canonicalizes every later
        call for that shape and group, which is the whole fix undone quietly.
        Callers get a view, whose base refuses the flag.
        """
        from flopscope._canonical_symmetry import canonical_map

        mapping = canonical_map((8, 8), SymmetryGroup.symmetric(axes=(0, 1)))
        assert not mapping.flags.owndata
        with pytest.raises(ValueError):
            mapping.flags.writeable = True

    def test_clear_cache_drains_the_orbit_maps(self):
        """The public aggregate must reach this cache; entries are array-sized."""
        from flopscope._canonical_symmetry import canonical_map

        canonical_map((9, 9), SymmetryGroup.symmetric(axes=(0, 1)))
        assert _canonical_map_cached.cache_info().currsize > 0
        flops.clear_cache()
        assert _canonical_map_cached.cache_info().currsize == 0

    def test_cache_is_bounded_by_bytes_not_entry_count(self):
        """Entries scale with the tensors they describe, so a count is no bound.

        ``symmetrize(mode="canonical-copy")`` is a registered op, so a caller
        can mint a map for any shape it likes. Bounding entries rather than
        bytes would cap the number of maps while leaving the footprint free to
        grow with the shapes requested.
        """
        from flopscope._canonical_symmetry import (
            _CANONICAL_MAP_CACHE_BYTES,
            canonical_map,
        )

        group = SymmetryGroup.symmetric(axes=(0, 1))
        flops.clear_cache()
        for side in range(600, 640):
            canonical_map((side, side), group)
        assert _canonical_map_cached.nbytes <= _CANONICAL_MAP_CACHE_BYTES
        flops.clear_cache()

    def test_a_map_larger_than_the_whole_budget_is_served_but_not_kept(self):
        """One outsized request must not evict everything and still not fit."""
        from flopscope._canonical_symmetry import (
            _CANONICAL_MAP_CACHE_BYTES,
            _generator_fingerprint,
            _OrbitMapCache,
        )

        tiny = _OrbitMapCache(max_bytes=8)  # smaller than any real map
        group = SymmetryGroup.symmetric(axes=(0, 1))
        mapping = tiny.get((4, 4), _generator_fingerprint(group))
        assert mapping.nbytes > 8
        assert tiny.cache_info().currsize == 0
        assert tiny.nbytes == 0
        # And the served map is still correct.
        assert np.array_equal(
            mapping, canonical_copy(np.arange(16.0).reshape(4, 4), group).ravel()
        )
        assert _CANONICAL_MAP_CACHE_BYTES > 0

    def test_eviction_does_not_change_results(self):
        """A map rebuilt after eviction must equal the one that was dropped."""
        from flopscope._canonical_symmetry import _generator_fingerprint, _OrbitMapCache

        group = SymmetryGroup.symmetric(axes=(0, 1))
        fingerprint = _generator_fingerprint(group)
        reference = np.array(
            _OrbitMapCache(max_bytes=10**9).get((6, 6), fingerprint), copy=True
        )

        cache = _OrbitMapCache(max_bytes=400)  # room for roughly one map
        first = np.array(cache.get((6, 6), fingerprint), copy=True)
        for side in (7, 8, 9):
            cache.get((side, side), fingerprint)  # force eviction
        rebuilt = cache.get((6, 6), fingerprint)

        assert np.array_equal(first, reference)
        assert np.array_equal(rebuilt, reference)


class TestSignedZero:
    """``-0.0 == 0.0``, but ``copysign`` tells them apart.

    A sign bit in a position the cost model prices as redundant is
    information, so equality alone is too generous a test for "already
    exact": the buffer has to go down the copying path.
    """

    def test_signed_zero_is_not_treated_as_already_invariant(self):
        group = SymmetryGroup.symmetric(axes=(0, 1))
        mixed = np.array([[1.0, 0.0], [-0.0, 2.0]])
        assert np.array_equal(mixed, mixed.T)  # `==` cannot see it
        assert not is_exactly_invariant(mixed, group)

    def test_tagging_removes_the_sign_bit_from_the_orbit(self):
        group = SymmetryGroup.symmetric(axes=(0, 1))
        with _budget():
            tagged = flops.as_symmetric(
                np.array([[1.0, 0.0], [-0.0, 2.0]]), symmetry=group
            )
        signs = np.signbit(np.asarray(tagged))
        assert np.array_equal(signs, signs.T), (
            "sign bit survived in a position priced as redundant"
        )

    def test_copysign_cannot_read_an_asymmetry_back_out(self):
        """The end-to-end route: the recovered signs must be symmetric."""
        group = SymmetryGroup.symmetric(axes=(0, 1))
        rng = np.random.default_rng(101)
        base = np.zeros((8, 8))
        # Scatter negative zeros asymmetrically through the buffer.
        mask = rng.random((8, 8)) < 0.5
        base[mask] = -0.0
        with _budget():
            tagged = flops.as_symmetric(base, symmetry=group)
            recovered = fnp.copysign(np.ones((8, 8)), tagged)
        values = np.asarray(recovered)
        assert np.array_equal(values, values.T)

    def test_complex_signed_zero_is_caught_on_both_components(self):
        group = SymmetryGroup.symmetric(axes=(0, 1))
        mixed = np.array([[1 + 0j, complex(0.0, 0.0)], [complex(-0.0, -0.0), 2 + 0j]])
        assert np.array_equal(mixed, mixed.T)
        assert not is_exactly_invariant(mixed, group)

    def test_ordinary_symmetric_data_still_short_circuits(self):
        """The check must not have become so strict that nothing passes."""
        group = SymmetryGroup.symmetric(axes=(0, 1))
        rng = np.random.default_rng(103)
        base = rng.standard_normal((16, 16))
        exact = (base + base.T) / 2
        assert is_exactly_invariant(exact, group)
        with _budget():
            tagged = flops.as_symmetric(exact, symmetry=group)
        assert np.shares_memory(np.asarray(tagged), exact)


# ---------------------------------------------------------------------------
# Cost of the boundary
# ---------------------------------------------------------------------------


class TestTrustedPropagationStaysCheap:
    def test_downstream_ops_do_not_rebuild_or_reapply_the_map(self):
        """Canonicalize once at ingress; propagate algebraically thereafter."""
        group = SymmetryGroup.symmetric(axes=(0, 1))
        rng = np.random.default_rng(37)
        base = rng.standard_normal((8, 8))
        exact = (base + base.T) / 2

        with _budget():
            tagged = flops.as_symmetric(exact, symmetry=group)
            before = _canonical_map_cached.cache_info()
            for _ in range(25):
                fnp.exp(tagged)
                fnp.multiply(tagged, 2.0)
                tagged.T  # noqa: B018 - exercising the transpose propagation
                tagged[0:8]
            after = _canonical_map_cached.cache_info()

        assert (after.hits, after.misses) == (before.hits, before.misses)

    def test_propagated_results_still_carry_their_tag(self):
        group = SymmetryGroup.symmetric(axes=(0, 1))
        base = np.random.default_rng(41).standard_normal((6, 6))
        with _budget():
            tagged = flops.as_symmetric((base + base.T) / 2, symmetry=group)
            assert fnp.exp(tagged).symmetry is not None


# ---------------------------------------------------------------------------
# Groups too large to enumerate
# ---------------------------------------------------------------------------


class TestOversizedGroups:
    def _oversized(self):
        return SymmetryGroup.symmetric(axes=tuple(range(9))), (2,) * 9

    def test_reynolds_refuses_and_names_the_alternative(self):
        group, shape = self._oversized()
        with _budget(), pytest.raises(ValueError, match="canonical-copy"):
            flops.symmetrize(np.zeros(shape), symmetry=group)

    def test_refusal_costs_nothing(self):
        """An operation that cannot finish must not be billed for trying."""
        group, shape = self._oversized()
        with _budget() as budget:
            with pytest.raises(ValueError):
                flops.symmetrize(np.zeros(shape), symmetry=group)
            assert budget.flops_used == 0

    def test_canonical_copy_still_works_there(self):
        group, shape = self._oversized()
        raw = np.random.default_rng(43).standard_normal(shape)
        with _budget() as budget:
            copied = flops.symmetrize(raw, symmetry=group, mode="canonical-copy")
        with _budget() as plain_copy:
            fnp.copy(fnp.asarray(raw))
        _assert_exactly_invariant(copied, group)
        assert budget.flops_used == plain_copy.flops_used

    def test_random_symmetric_refuses_and_offers_the_same_way_through(self):
        group, shape = self._oversized()
        with _budget(), pytest.raises(ValueError, match="canonical-copy"):
            fnp.random.symmetric(shape, group)

        with _budget():
            sample = fnp.random.symmetric(shape, group, mode="canonical-copy")
        _assert_exactly_invariant(sample, group)

    def test_random_symmetric_default_is_unchanged(self):
        group = SymmetryGroup.symmetric(axes=(0, 1))
        with _budget() as implicit:
            fnp.random.symmetric((6, 6), group)
        with _budget() as explicit:
            fnp.random.symmetric((6, 6), group, mode="reynolds-projection")
        assert implicit.flops_used == explicit.flops_used

    def test_random_symmetric_rejects_an_unknown_mode(self):
        with _budget(), pytest.raises(ValueError, match="unknown symmetric mode"):
            fnp.random.symmetric((4, 4), SymmetryGroup.symmetric(axes=(0, 1)), mode="x")


# ---------------------------------------------------------------------------
# Non-constant fills
# ---------------------------------------------------------------------------


class TestNonScalarFillIsNotSymmetric:
    """A fill that varies across the array leaves no orbit constant.

    ``full``/``full_like`` infer symmetry from shape alone, which describes a
    constant fill. Broadcasting an array through them writes different values
    into positions the inferred tag would call redundant.
    """

    def test_full_with_array_fill_carries_no_tag(self):
        with _budget():
            result = fnp.full((3, 3), np.array([1.0, 2.0, 3.0]))
        assert getattr(result, "symmetry", None) is None

    def test_full_like_with_array_fill_carries_no_tag(self):
        with _budget():
            template = fnp.zeros((3, 3))
            assert template.symmetry is not None  # constant fill: genuinely symmetric
            result = fnp.full_like(template, np.array([1.0, 2.0, 3.0]))
        assert getattr(result, "symmetry", None) is None

    def test_array_fill_is_priced_like_untagged_data(self):
        fill = np.array([1.0, 2.0, 3.0])
        with _budget():
            tagged = fnp.full_like(fnp.zeros((3, 3)), fill)

        def _billed(fn):
            with _budget() as budget:
                fn()
                return budget.flops_used

        honest = _billed(lambda: fnp.sin(fnp.asarray(np.broadcast_to(fill, (3, 3)))))
        assert _billed(lambda: fnp.sin(tagged)) == honest

    @pytest.mark.parametrize("value", [0.0, 3.5, -1])
    def test_scalar_fill_keeps_its_legitimate_tag(self, value):
        with _budget():
            full = fnp.full((4, 4), value)
            full_like = fnp.full_like(fnp.zeros((4, 4)), value)
        assert full.symmetry is not None
        assert full_like.symmetry is not None
        _assert_exactly_invariant(full, full.symmetry)
