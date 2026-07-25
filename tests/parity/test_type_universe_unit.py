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
        "second-positional",
        "dict-literal",
    } <= names


def test_includes_the_seed_bug_value():
    assert "complex" in {name for name, _, _ in type_universe.VALUES}


def test_includes_numpy_complex_scalars():
    names = {name for name, _, _ in type_universe.VALUES}
    assert {"np-complex64", "np-complex128"} <= names


def test_includes_negative_integer_boundaries():
    names = {name for name, _, _ in type_universe.VALUES}
    assert {"int64-min", "int64-min-minus-one", "huge-negative-int"} <= names


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


def test_every_case_is_tagged_with_its_position():
    for case in type_universe.build():
        assert any(tag.startswith("position:") for tag in case.tags), case.id
