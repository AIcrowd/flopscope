"""Unit tests for the Case descriptor. Pure — no backend, no server."""

from __future__ import annotations

import pytest

from tests.parity.case import Case


def test_round_trips_through_json():
    case = Case(
        id="idiom/slice-bound-remote",
        source="V[: fnp.argmax(V)]",
        setup="",
        tags=frozenset({"family:4", "tier:fast"}),
    )
    assert Case.from_json(case.to_json()) == case


def test_tags_are_order_independent_in_json():
    a = Case(id="x/y", source="1", tags=frozenset({"b", "a"}))
    b = Case(id="x/y", source="1", tags=frozenset({"a", "b"}))
    assert a.to_json() == b.to_json()


def test_family_extracts_the_family_tag():
    assert Case(id="x/y", source="1", tags=frozenset({"family:6"})).family() == "6"
    assert Case(id="x/y", source="1").family() is None


def test_id_must_be_structured():
    with pytest.raises(ValueError, match="must be '<source>/<name>'"):
        Case(id="noslash", source="1")


def test_case_is_hashable_and_frozen():
    case = Case(id="x/y", source="1")
    assert {case}
    with pytest.raises(AttributeError):
        case.source = "2"  # type: ignore[misc]


def test_id_segments_must_be_non_empty():
    for bad in ("/", "foo/", "/bar"):
        with pytest.raises(ValueError, match="must be '<source>/<name>'"):
            Case(id=bad, source="1")


def test_id_name_segment_may_contain_slashes():
    # Names legitimately contain further separators, e.g. grid op ids.
    assert Case(id="grid/fft.rfft::axis-int", source="1").id.startswith("grid/")
