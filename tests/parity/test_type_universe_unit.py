from __future__ import annotations

from tests.parity.corpus import type_universe


def test_covers_every_position():
    names = {name for name, _ in type_universe.POSITIONS}
    assert {
        "positional",
        "keyword",
        "list-element",
        "index-key",
        "slice-bound",
    } <= names


def test_includes_the_seed_bug_value():
    assert "complex" in {name for name, _, _ in type_universe.VALUES}


def test_numpy_values_are_tagged_so_they_can_be_skipped():
    for name, _, tags in type_universe.VALUES:
        if name.startswith("np-"):
            assert "requires:numpy" in tags


def test_builds_the_full_cross_product():
    assert len(type_universe.build()) == len(type_universe.VALUES) * len(
        type_universe.POSITIONS
    )


def test_case_ids_encode_value_and_position():
    ids = {case.id for case in type_universe.build()}
    assert "types/complex::slice-bound" in ids
