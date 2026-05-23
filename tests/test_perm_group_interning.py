"""Pin instance interning for SymmetryGroup (issue #73)."""

from __future__ import annotations

import pytest

from flopscope._perm_group import _GROUP_INTERN, SymmetryGroup


@pytest.fixture(autouse=True)
def _isolate_intern_registry():
    # Clear before and after each test so tests don't see each other's
    # interned instances. Process-global state is hostile to test isolation.
    _GROUP_INTERN.clear()
    yield
    _GROUP_INTERN.clear()


class TestSymmetricInterning:
    def test_same_axes_returns_same_instance(self):
        a = SymmetryGroup.symmetric(axes=(0, 1, 2))
        b = SymmetryGroup.symmetric(axes=(0, 1, 2))
        assert a is b

    def test_different_axes_returns_different_instance(self):
        a = SymmetryGroup.symmetric(axes=(0, 1, 2))
        b = SymmetryGroup.symmetric(axes=(0, 1, 3))
        assert a is not b


class TestCyclicDihedralIdentityInterning:
    def test_cyclic_interns(self):
        a = SymmetryGroup.cyclic(axes=(0, 1, 2, 3))
        b = SymmetryGroup.cyclic(axes=(0, 1, 2, 3))
        assert a is b

    def test_dihedral_interns(self):
        a = SymmetryGroup.dihedral(axes=(0, 1, 2, 3))
        b = SymmetryGroup.dihedral(axes=(0, 1, 2, 3))
        assert a is b

    def test_identity_interns(self):
        # symmetric(axes=(x,)) is tagged as identity
        a = SymmetryGroup.symmetric(axes=(0,))
        b = SymmetryGroup.symmetric(axes=(0,))
        assert a is b


class TestDirectProductInterning:
    def test_sorted_children_intern_together(self):
        a = SymmetryGroup.symmetric(axes=(3, 4))
        b = SymmetryGroup.symmetric(axes=(0, 1, 2))
        g_ab = SymmetryGroup.direct_product(a, b)
        g_ba = SymmetryGroup.direct_product(b, a)
        assert g_ab is g_ba


class TestUnknownKindDoesNotIntern:
    def test_from_generators_does_not_intern(self):
        a = SymmetryGroup.from_generators([[1, 0]], axes=(0, 1))
        b = SymmetryGroup.from_generators([[1, 0]], axes=(0, 1))
        assert a is not b

    def test_from_payload_does_not_intern_with_factory(self):
        # symmetric → payload → from_payload produces an unknown-kind copy
        sym = SymmetryGroup.symmetric(axes=(0, 1, 2))
        roundtripped = SymmetryGroup.from_payload(sym.to_payload())
        assert roundtripped == sym  # value-equal
        assert roundtripped is not sym  # but not interned together


class TestEdgeCaseWarts:
    def test_young_single_block_not_interned_with_symmetric(self):
        sym = SymmetryGroup.symmetric(axes=(0, 1, 2))
        young = SymmetryGroup.young(blocks=[(0, 1, 2)])
        assert young == sym  # value-equal (both S_3)
        assert young is not sym  # but different kind tags

    def test_d3_not_interned_with_s3(self):
        s3 = SymmetryGroup.symmetric(axes=(0, 1, 2))
        d3 = SymmetryGroup.dihedral(axes=(0, 1, 2))
        assert s3 == d3  # value-equal (D_3 ≅ S_3)
        assert s3 is not d3  # but different kind tags
