"""Closed-form ``SymmetryGroup.order()`` formulas pinned against ``_dimino``."""

from __future__ import annotations

import math

import pytest

from flopscope._perm_group import _closed_form_order


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
