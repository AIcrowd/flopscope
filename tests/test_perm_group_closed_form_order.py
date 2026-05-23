"""Closed-form ``SymmetryGroup.order()`` formulas pinned against ``_dimino``."""

from __future__ import annotations

import math

import pytest

from flopscope._perm_group import SymmetryGroup, _closed_form_order, _dimino


class TestClosedFormOrder:
    def test_identity_kind(self):
        assert _closed_form_order(("identity", (0,))) == 1
        assert _closed_form_order(("identity", ("a", "b", "c"))) == 1

    def test_symmetric_kind(self):
        for n in range(2, 7):
            kind = ("symmetric", tuple(range(n)))
            assert _closed_form_order(kind) == math.factorial(n)

    def test_cyclic_kind(self):
        for n in range(2, 11):
            assert _closed_form_order(("cyclic", tuple(range(n)))) == n

    def test_dihedral_kind(self):
        for n in range(3, 11):
            assert _closed_form_order(("dihedral", tuple(range(n)))) == 2 * n

    def test_direct_product_kind(self):
        children = (("symmetric", (0, 1)), ("symmetric", (2, 3, 4)))
        # |S_2| * |S_3| = 2 * 6 = 12
        assert _closed_form_order(("direct_product", children)) == 12

    def test_direct_product_recursive(self):
        nested = (
            ("direct_product", (("symmetric", (0, 1)), ("cyclic", (2, 3, 4)))),
            ("symmetric", (5, 6)),
        )
        # (|S_2| * |C_3|) * |S_2| = (2 * 3) * 2 = 12
        assert _closed_form_order(("direct_product", nested)) == 12

    def test_unknown_kind_raises(self):
        with pytest.raises(AssertionError):
            _closed_form_order(("nonsense", (0, 1, 2)))


class TestOrderDispatch:
    def test_order_dispatches_to_closed_form_when_kind_set(self):
        # If _known_kind is set, order() must not enumerate elements.
        g = SymmetryGroup.symmetric(axes=(0, 1, 2, 3))
        g._order = None
        g._elements = None
        g._known_kind = ("symmetric", (0, 1, 2, 3))
        # Sentinel: replace _generators with something that would fail
        # if _dimino tried to enumerate. The closed-form path doesn't
        # touch _generators, so this must succeed.
        g._generators = ()
        assert g.order() == 24

    def test_order_falls_back_to_dimino_when_kind_none(self):
        g = SymmetryGroup.symmetric(axes=(0, 1, 2))
        g._order = None
        g._known_kind = None
        assert g.order() == 6  # via _dimino


class TestFactoryTagging:
    def test_symmetric_tag(self):
        assert SymmetryGroup.symmetric(axes=(0, 1, 2))._known_kind == (
            "symmetric",
            (0, 1, 2),
        )

    def test_symmetric_degree_one_is_identity(self):
        assert SymmetryGroup.symmetric(axes=(0,))._known_kind == ("identity", (0,))

    def test_cyclic_tag(self):
        assert SymmetryGroup.cyclic(axes=(0, 1, 2))._known_kind == (
            "cyclic",
            (0, 1, 2),
        )

    def test_cyclic_degree_one_is_identity(self):
        assert SymmetryGroup.cyclic(axes=(0,))._known_kind == ("identity", (0,))

    def test_dihedral_tag(self):
        assert SymmetryGroup.dihedral(axes=(0, 1, 2, 3))._known_kind == (
            "dihedral",
            (0, 1, 2, 3),
        )

    def test_dihedral_degree_two_falls_through_to_symmetric(self):
        # dihedral(k=2) falls through to symmetric (existing behavior); k>=2 → symmetric tag
        assert SymmetryGroup.dihedral(axes=(0, 1))._known_kind == (
            "symmetric",
            (0, 1),
        )

    def test_direct_product_sorts_children(self):
        a = SymmetryGroup.symmetric(axes=(3, 4))
        b = SymmetryGroup.symmetric(axes=(0, 1, 2))
        g_ab = SymmetryGroup.direct_product(a, b)
        g_ba = SymmetryGroup.direct_product(b, a)
        expected = (
            "direct_product",
            (("symmetric", (0, 1, 2)), ("symmetric", (3, 4))),
        )
        assert g_ab._known_kind == expected
        assert g_ba._known_kind == expected

    def test_direct_product_unknown_child_poisons(self):
        known = SymmetryGroup.symmetric(axes=(0, 1))
        unknown = SymmetryGroup.from_generators([[1, 0]], axes=(2, 3))
        # from_generators does not tag, so the direct product can't either
        g = SymmetryGroup.direct_product(known, unknown)
        assert g._known_kind is None

    def test_young_tag_via_direct_product(self):
        # young always routes through direct_product
        y = SymmetryGroup.young(blocks=[(0, 1), (2, 3, 4)])
        assert y._known_kind == (
            "direct_product",
            (("symmetric", (0, 1)), ("symmetric", (2, 3, 4))),
        )

    def test_direct_product_sorts_mixed_axis_types(self):
        # Regression: tuple comparison fails on mixed-type axes.
        # The sort uses key=repr so it stays total-ordered.
        a = SymmetryGroup.symmetric(axes=("x", "y"))
        b = SymmetryGroup.symmetric(axes=(0, 1))
        # Should not raise.
        g = SymmetryGroup.direct_product(a, b)
        assert g._known_kind is not None
        assert g._known_kind[0] == "direct_product"
        assert len(g._known_kind[1]) == 2

    def test_from_generators_no_tag(self):
        g = SymmetryGroup.from_generators([[1, 0]], axes=(0, 1))
        assert g._known_kind is None

    def test_order_uses_closed_form_for_tagged_factories(self):
        # Now that factories tag, order() should be O(1) — verify by checking
        # that _elements wasn't enumerated.
        g = SymmetryGroup.symmetric(axes=(0, 1, 2, 3, 4, 5, 6))
        assert g.order() == 5040
        assert g._elements is None  # not enumerated; _dimino not called


class TestClosedFormVsDimino:
    """Pin the closed-form formula against _dimino ground truth.

    For small n, _dimino enumeration is the source of truth. The closed
    form should match it exactly.
    """

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_symmetric_matches_dimino(self, n):
        g = SymmetryGroup.symmetric(axes=tuple(range(n)))
        assert g.order() == len(_dimino(g._generators)) == math.factorial(n)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_cyclic_matches_dimino(self, n):
        g = SymmetryGroup.cyclic(axes=tuple(range(n)))
        assert g.order() == len(_dimino(g._generators)) == n

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
    def test_dihedral_matches_dimino(self, n):
        g = SymmetryGroup.dihedral(axes=tuple(range(n)))
        assert g.order() == len(_dimino(g._generators)) == 2 * n

    def test_direct_product_matches_dimino(self):
        a = SymmetryGroup.symmetric(axes=(0, 1, 2))
        b = SymmetryGroup.cyclic(axes=(3, 4, 5, 6))
        g = SymmetryGroup.direct_product(a, b)
        # |S_3 × C_4| = 6 * 4 = 24
        assert g.order() == len(_dimino(g._generators)) == 24

    def test_young_matches_dimino(self):
        g = SymmetryGroup.young(blocks=[(0, 1), (2, 3, 4)])
        # |S_2 × S_3| = 2 * 6 = 12
        assert g.order() == len(_dimino(g._generators)) == 12
