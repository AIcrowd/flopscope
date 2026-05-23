"""Pin helper provenance preservation for SymmetryGroup (issue #73)."""

from __future__ import annotations

import pytest

from flopscope._perm_group import _GROUP_INTERN, SymmetryGroup
from flopscope._symmetry_utils import embed_group, remap_group_axes


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
        assert embedded._known_kind == ("symmetric", (0, 1, 2))


class TestRemapGroupAxesProvenance:
    def test_remap_leaf_kind_symmetric(self):
        g = SymmetryGroup.symmetric(axes=(0, 1, 2))
        remapped = remap_group_axes(g, {0: 3, 1: 4, 2: 5})
        assert remapped._known_kind == ("symmetric", (3, 4, 5))

    def test_remap_leaf_kind_cyclic(self):
        g = SymmetryGroup.cyclic(axes=(0, 1, 2))
        remapped = remap_group_axes(g, {0: 3, 1: 4, 2: 5})
        assert remapped._known_kind == ("cyclic", (3, 4, 5))

    def test_remap_direct_product_recurses(self):
        g = SymmetryGroup.direct_product(
            SymmetryGroup.symmetric(axes=(0, 1)),
            SymmetryGroup.cyclic(axes=(2, 3, 4)),
        )
        remapped = remap_group_axes(g, {0: 10, 1: 11, 2: 12, 3: 13, 4: 14})
        # After sort: cyclic(12,13,14) < symmetric(10,11) lexicographically
        assert remapped._known_kind == (
            "direct_product",
            (("cyclic", (12, 13, 14)), ("symmetric", (10, 11))),
        )
