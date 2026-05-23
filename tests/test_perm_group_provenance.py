"""Pin helper provenance preservation for SymmetryGroup (issue #73)."""

from __future__ import annotations

import pytest

from flopscope._perm_group import _GROUP_INTERN, SymmetryGroup
from flopscope._symmetry_utils import (
    embed_group,
    intersect_groups,
    reduce_group,
    remap_group_axes,
    restrict_group_to_axes,
)


@pytest.fixture(autouse=True)
def _isolate_intern_registry():
    _GROUP_INTERN.clear()
    yield
    _GROUP_INTERN.clear()


class TestEmbedGroupProvenance:
    def test_embed_preserves_kind_when_axes_already_full(self):
        g = SymmetryGroup.symmetric(axes=(0, 1, 2))
        embedded = embed_group(g, ndim=3)
        # No rank change → same instance via interning
        assert embedded is g
        assert embedded is not None
        assert embedded._known_kind == ("symmetric", (0, 1, 2))


class TestRemapGroupAxesProvenance:
    def test_remap_leaf_kind_symmetric(self):
        g = SymmetryGroup.symmetric(axes=(0, 1, 2))
        remapped = remap_group_axes(g, {0: 3, 1: 4, 2: 5})
        assert remapped is not None
        assert remapped._known_kind == ("symmetric", (3, 4, 5))

    def test_remap_leaf_kind_cyclic(self):
        g = SymmetryGroup.cyclic(axes=(0, 1, 2))
        remapped = remap_group_axes(g, {0: 3, 1: 4, 2: 5})
        assert remapped is not None
        assert remapped._known_kind == ("cyclic", (3, 4, 5))

    def test_remap_direct_product_recurses(self):
        g = SymmetryGroup.direct_product(
            SymmetryGroup.symmetric(axes=(0, 1)),
            SymmetryGroup.cyclic(axes=(2, 3, 4)),
        )
        remapped = remap_group_axes(g, {0: 10, 1: 11, 2: 12, 3: 13, 4: 14})
        assert remapped is not None
        # After sort: cyclic(12,13,14) < symmetric(10,11) lexicographically
        assert remapped._known_kind == (
            "direct_product",
            (("cyclic", (12, 13, 14)), ("symmetric", (10, 11))),
        )


class TestRestrictGroupToAxesProvenance:
    def test_restrict_to_full_axes_preserves_symmetric_kind(self):
        g = SymmetryGroup.symmetric(axes=(0, 1, 2, 3))
        restricted = restrict_group_to_axes(g, axes=(0, 1, 2, 3))
        # Identity restriction → same instance via interning
        assert restricted is g
        assert restricted is not None
        assert restricted._known_kind == ("symmetric", (0, 1, 2, 3))

    def test_restrict_to_full_axes_preserves_cyclic_kind(self):
        g = SymmetryGroup.cyclic(axes=(0, 1, 2))
        restricted = restrict_group_to_axes(g, axes=(0, 1, 2))
        assert restricted is g

    def test_restrict_strict_subset_of_symmetric_returns_subsymmetric_without_kind(
        self,
    ):
        # `restrict_group_to_axes` composes setwise_stabilizer + restrict, so
        # a strict subset of a free-permuting group projects to S_|T| on T.
        # Our provenance preservation is scoped to the no-op case (T == A);
        # strict-subset results carry `_known_kind=None` — the "sub-symmetric
        # is still symmetric" rule lives in `reduce_group`, not here.
        g = SymmetryGroup.symmetric(axes=(0, 1, 2, 3))
        restricted = restrict_group_to_axes(g, axes=(0, 1, 2))
        assert restricted is not None
        assert restricted.order() == 6  # S_3
        assert restricted._known_kind is None


class TestIntersectGroupsProvenance:
    def test_same_kind_intersect_returns_same_instance(self):
        # symmetric(A) interned, so two calls give the same object.
        a = SymmetryGroup.symmetric(axes=(0, 1, 2))
        b = SymmetryGroup.symmetric(axes=(0, 1, 2))
        assert a is b  # sanity: interning works
        result = intersect_groups(a, b, ndim=3)
        assert result is a
        assert result is not None
        assert result._known_kind == ("symmetric", (0, 1, 2))

    def test_identity_kind_intersect_returns_none(self):
        # Two identity-kind groups intersect to themselves, which is trivial
        # → existing convention is to return None for trivial intersections.
        a = SymmetryGroup.symmetric(axes=(0,))  # tagged identity
        b = SymmetryGroup.symmetric(axes=(0,))
        result = intersect_groups(a, b, ndim=1)
        assert result is None  # trivial group → None per existing convention

    def test_unknown_intersection_stays_none(self):
        # symmetric ∩ cyclic on the same axes: intersection is cyclic
        # (cyclic ⊂ symmetric), but we don't claim that in the conservative
        # rule. _known_kind stays None.
        s = SymmetryGroup.symmetric(axes=(0, 1, 2))
        c = SymmetryGroup.cyclic(axes=(0, 1, 2))
        result = intersect_groups(s, c, ndim=3)
        assert result is not None
        assert result._known_kind is None  # conservative — no inference


class TestReduceGroupProvenance:
    def test_reduce_symmetric_produces_smaller_symmetric(self):
        g = SymmetryGroup.symmetric(axes=(0, 1, 2, 3, 4))
        result = reduce_group(g, ndim=5, axis=(3, 4))
        # |G \ R| = 3, axes (0,1,2) after reduction remap to (0,1,2)
        assert result is not None
        assert result._known_kind == ("symmetric", (0, 1, 2))

    def test_reduce_symmetric_keepdims(self):
        g = SymmetryGroup.symmetric(axes=(0, 1, 2, 3))
        # With keepdims, reduced axes stay at their positions.
        result = reduce_group(g, ndim=4, axis=(2, 3), keepdims=True)
        assert result is not None
        # The 2 surviving axes keep their tensor positions (0, 1)
        assert result._known_kind == ("symmetric", (0, 1))

    def test_reduce_all_axes_returns_none(self):
        g = SymmetryGroup.symmetric(axes=(0, 1, 2))
        result = reduce_group(g, ndim=3, axis=(0, 1, 2))
        assert result is None

    def test_reduce_direct_product_recurses(self):
        g = SymmetryGroup.direct_product(
            SymmetryGroup.symmetric(axes=(0, 1)),
            SymmetryGroup.cyclic(axes=(2, 3, 4)),
        )
        # Reduce one axis from the symmetric factor (drops it to trivial).
        result = reduce_group(g, ndim=5, axis=(0,))
        # Surviving: cyclic(2,3,4) only; symmetric(1) becomes trivial.
        # After remap (axis 0 removed), cyclic axes shift to (1,2,3).
        assert result is not None
        assert result._known_kind == ("cyclic", (1, 2, 3))


class TestBroadcastGroupProvenance:
    def test_broadcast_no_op_preserves_kind(self):
        from flopscope._symmetry_utils import broadcast_group

        g = SymmetryGroup.symmetric(axes=(0, 1, 2))
        broadcasted = broadcast_group(g, input_shape=(4, 4, 4), output_shape=(4, 4, 4))
        assert broadcasted is g

    def test_broadcast_adds_symmetric_factor_for_new_axes(self):
        from flopscope._symmetry_utils import broadcast_group

        # Input shape (4, 4) broadcast to (3, 3, 4, 4):
        # two newly-broadcast leading axes of size 3 → symmetric factor on (0, 1)
        g = SymmetryGroup.symmetric(axes=(0, 1))
        result = broadcast_group(g, input_shape=(4, 4), output_shape=(3, 3, 4, 4))
        # Inner symmetric on input axes (0,1) remaps to output axes (2,3).
        # Plus a new symmetric on the two created (3,3) axes (0,1).
        assert result is not None
        assert result._known_kind == (
            "direct_product",
            (("symmetric", (0, 1)), ("symmetric", (2, 3))),
        )
