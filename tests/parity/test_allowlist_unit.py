"""Strictness runs in BOTH directions: unexplained divergences fail, and so do
stale entries. Fixing a bug therefore forces deletion of its entry in the same
pull request, so the list cannot accumulate dead weight."""

from __future__ import annotations

from tests.parity.allowlist import Category, Entry, apply, validate_entries
from tests.parity.compare import Divergence

_DIV = Divergence("idiom/complex-mul", "outcome", "returned", "raised")

_ENTRY = Entry(
    case_id="idiom/complex-mul",
    dimension="outcome",
    category=Category.KNOWN_BUG,
    reason="Family 4: Python complex falls through _encode_arg.",
    issue="https://github.com/AIcrowd/flopscope/issues/1",
)


def test_unexplained_divergence_is_reported():
    result = apply([_DIV], entries=())
    assert result.unexplained == [_DIV]
    assert result.stale == []


def test_explained_divergence_is_allowed():
    result = apply([_DIV], entries=(_ENTRY,))
    assert result.unexplained == []
    assert result.allowed == [_DIV]


def test_stale_entry_is_reported():
    result = apply([], entries=(_ENTRY,))
    assert result.stale == [_ENTRY]


def test_allowlist_is_keyed_per_dimension_not_per_case():
    value_div = Divergence("idiom/complex-mul", "value", "f:aa", "f:bb")
    result = apply([_DIV, value_div], entries=(_ENTRY,))
    assert result.allowed == [_DIV]
    assert result.unexplained == [value_div]


def test_counts_are_reported_per_category():
    result = apply([_DIV], entries=(_ENTRY,))
    assert result.counts["known-bug"] == 1
    assert result.counts["proxy-inherent"] == 0


def test_known_bug_without_issue_fails_validation():
    bad = Entry(
        case_id="a/b",
        dimension="value",
        category=Category.KNOWN_BUG,
        reason="tracked",
        issue=None,
    )
    problems = validate_entries((bad,))
    assert any("requires an issue link" in p for p in problems)


def test_empty_reason_fails_validation():
    bad = Entry(
        case_id="a/b",
        dimension="value",
        category=Category.PROXY_INHERENT,
        reason="   ",
    )
    problems = validate_entries((bad,))
    assert any("non-empty reason" in p for p in problems)


def test_unknown_dimension_fails_validation():
    bad = Entry(
        case_id="a/b",
        dimension="colour",
        category=Category.PROXY_INHERENT,
        reason="n/a",
    )
    problems = validate_entries((bad,))
    assert any("not a known dimension" in p for p in problems)


def test_duplicate_entries_fail_validation():
    problems = validate_entries((_ENTRY, _ENTRY))
    assert any("duplicate" in p for p in problems)


def test_shipped_entries_are_valid():
    from tests.parity.allowlist import ENTRIES

    assert validate_entries(ENTRIES) == []


def test_glob_case_id_matches_every_case_it_covers():
    glob_entry = Entry(
        case_id="grid/fft.*::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason="Family 9: fft submodule is missing from the client surface.",
        issue="INTERNAL-P4-family-9",
    )
    div_a = Divergence("grid/fft.rfft::array", "outcome", "returned", "raised")
    div_b = Divergence("grid/fft.fftn::vector", "outcome", "returned", "raised")
    result = apply([div_a, div_b], entries=(glob_entry,))
    assert result.unexplained == []
    assert result.stale == []
    assert set(result.allowed) == {div_a, div_b}


def test_glob_case_id_does_not_match_outside_its_pattern():
    glob_entry = Entry(
        case_id="grid/fft.*::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason="Family 9.",
        issue="INTERNAL-P4-family-9",
    )
    unrelated = Divergence("grid/sum::array", "outcome", "returned", "raised")
    result = apply([unrelated], entries=(glob_entry,))
    assert result.unexplained == [unrelated]
    # The entry matched nothing, so it is stale, exactly like a literal entry.
    assert result.stale == [glob_entry]


def test_glob_entry_still_requires_dimension_to_match_exactly():
    glob_entry = Entry(
        case_id="grid/fft.*::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason="Family 9.",
        issue="INTERNAL-P4-family-9",
    )
    value_div = Divergence("grid/fft.rfft::array", "value", "1", "2")
    result = apply([value_div], entries=(glob_entry,))
    assert result.unexplained == [value_div]
    assert result.stale == [glob_entry]


def test_match_counts_are_reported_per_entry():
    glob_entry = Entry(
        case_id="grid/fft.*::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason="Family 9.",
        issue="INTERNAL-P4-family-9",
    )
    div_a = Divergence("grid/fft.rfft::array", "outcome", "returned", "raised")
    div_b = Divergence("grid/fft.fftn::vector", "outcome", "returned", "raised")
    result = apply([_DIV, div_a, div_b], entries=(_ENTRY, glob_entry))
    assert result.match_counts[_ENTRY] == 1
    assert result.match_counts[glob_entry] == 2


def test_unmatched_entry_has_a_zero_match_count():
    result = apply([], entries=(_ENTRY,))
    assert result.match_counts[_ENTRY] == 0
    assert result.stale == [_ENTRY]
