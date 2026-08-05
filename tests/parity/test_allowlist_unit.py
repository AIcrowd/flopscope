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


def test_numeric_dtype_exception_gap_entries_are_exact_cases():
    from tests.parity.allowlist import ENTRIES

    issue = "INTERNAL-PARITY-NUMERIC-DTYPE-EXCEPTIONS"
    expected = {
        ("types/range::dict-literal", "exc_bases"),
        ("types/memoryview::positional", "exc_bases"),
        ("types/memoryview::keyword", "exc_bases"),
        ("types/memoryview::second-positional", "exc_bases"),
        ("types/memoryview::dict-literal", "exc_bases"),
        ("types/bytearray::positional", "exc_bases"),
        ("types/bytearray::keyword", "exc_bases"),
        ("types/bytearray::second-positional", "exc_bases"),
        ("types/bytearray::dict-literal", "exc_bases"),
        ("types/slice-object::positional", "exc_bases"),
        ("types/slice-object::keyword", "exc_bases"),
        ("types/slice-object::list-element", "exc_bases"),
        ("types/slice-object::second-positional", "exc_bases"),
        ("types/slice-object::dict-literal", "exc_bases"),
        ("types/slice-object::constructor", "exc_bases"),
        ("types/ellipsis::positional", "exc_bases"),
        ("types/ellipsis::keyword", "exc_bases"),
        ("types/ellipsis::list-element", "exc_bases"),
        ("types/ellipsis::second-positional", "exc_bases"),
        ("types/ellipsis::dict-literal", "exc_bases"),
        ("types/ellipsis::constructor", "exc_bases"),
        ("types/int-enum::dict-literal", "exc_type"),
        ("types/int-enum::dict-literal", "exc_bases"),
        ("types/nested-list::dict-literal", "exc_type"),
        ("types/nested-list::dict-literal", "exc_bases"),
        ("types/remote-scalar::dict-literal", "exc_bases"),
        ("types/np-int64::dict-literal", "exc_bases"),
        ("types/np-bool::dict-literal", "exc_bases"),
    }
    related_existing = {("types/remote-scalar::dict-literal", "exc_type")}
    all_entries_by_key = {entry.key(): entry for entry in ENTRIES}
    selected = expected | related_existing
    entries = tuple(all_entries_by_key[key] for key in selected)

    assert len(expected) == 28
    assert {entry.key() for entry in entries} == selected
    assert all(entry.issue == issue for entry in entries)
    assert all(entry.category is Category.KNOWN_BUG for entry in entries)
    assert all(not any(char in entry.case_id for char in "*?[") for entry in entries)

    server_dict_refusal = {
        ("types/memoryview::dict-literal", "exc_bases"),
        ("types/bytearray::dict-literal", "exc_bases"),
        ("types/remote-scalar::dict-literal", "exc_type"),
        ("types/remote-scalar::dict-literal", "exc_bases"),
        ("types/int-enum::dict-literal", "exc_type"),
        ("types/int-enum::dict-literal", "exc_bases"),
        ("types/nested-list::dict-literal", "exc_type"),
        ("types/nested-list::dict-literal", "exc_bases"),
    }
    entries_by_key = {entry.key(): entry for entry in entries}
    for key in server_dict_refusal:
        reason = entries_by_key[key].reason
        assert "server pre-dispatch unresolved-dict refusal" in reason
        assert "plain TypeError" in reason
        assert "UnsupportedDtypeError" in reason


def test_buffer_allowlist_entries_are_literal_current_observations():
    from tests.parity.allowlist import ENTRIES

    families = ("memoryview", "bytearray")
    expected: set[tuple[str, str]] = set()
    for family in families:
        prefix = f"types/{family}"
        expected.update(
            (f"{prefix}::{position}", "exc_type")
            for position in (
                "positional",
                "keyword",
                "second-positional",
                "slice-bound",
                "dict-literal",
            )
        )
        expected.update(
            (f"{prefix}::{position}", "flops")
            for position in ("list-element", "index-key")
        )
        expected.update(
            (f"{prefix}::{position}", "outcome")
            for position in ("list-element", "index-key", "constructor")
        )

    entries = tuple(
        entry
        for entry in ENTRIES
        if entry.case_id.startswith(("types/memoryview::", "types/bytearray::"))
        and entry.dimension in {"exc_type", "flops", "outcome"}
    )
    entries_by_key = {entry.key(): entry for entry in entries}

    assert set(entries_by_key) == expected
    assert all(not any(char in entry.case_id for char in "*?[") for entry in entries)

    for family in families:
        prefix = f"types/{family}"
        for position in ("positional", "keyword", "second-positional"):
            reason = entries_by_key[(f"{prefix}::{position}", "exc_type")].reason
            assert "ValueError" in reason
            assert "UnsupportedDtypeError" in reason
        slice_reason = entries_by_key[(f"{prefix}::slice-bound", "exc_type")].reason
        assert "TypeError" in slice_reason
        assert "ValueError" in slice_reason
        dict_reason = entries_by_key[(f"{prefix}::dict-literal", "exc_type")].reason
        assert "server pre-dispatch unresolved-dict refusal" in dict_reason
        assert "UnsupportedDtypeError" in dict_reason
        assert "plain TypeError" in dict_reason

        list_flops = entries_by_key[(f"{prefix}::list-element", "flops")].reason
        assert "8 FLOPs" in list_flops
        assert "UnsupportedDtypeError" in list_flops
        index_flops = entries_by_key[(f"{prefix}::index-key", "flops")].reason
        assert "8 FLOPs" in index_flops
        assert "IndexError" in index_flops

        list_outcome = entries_by_key[(f"{prefix}::list-element", "outcome")].reason
        assert "returned" in list_outcome
        assert "UnsupportedDtypeError" in list_outcome
        index_outcome = entries_by_key[(f"{prefix}::index-key", "outcome")].reason
        assert "returned" in index_outcome
        assert "IndexError" in index_outcome
        constructor = entries_by_key[(f"{prefix}::constructor", "outcome")].reason
        assert "uint8" in constructor
        assert "UnsupportedDtypeError" in constructor


def test_broad_exception_entries_describe_their_current_paths():
    from tests.parity.allowlist import ENTRIES

    entries = {entry.key(): entry for entry in ENTRIES}
    expected_fragments = {
        ("types/range::*", "exc_type"): (
            "positional, keyword, and second-positional",
            "ValueError",
            "dict-literal",
            "UnsupportedDtypeError",
            "RemoteSerializationError",
        ),
        ("types/slice-object::*", "exc_type"): (
            "six matched positions",
            "UnsupportedDtypeError",
            "RemoteSerializationError",
        ),
        ("types/ellipsis::*", "exc_type"): (
            "six matched positions",
            "UnsupportedDtypeError",
            "RemoteSerializationError",
        ),
        ("types/np-int64::*", "exc_type"): (
            "list-element",
            "ValueError",
            "dict-literal",
            "UnsupportedDtypeError",
            "RemoteSerializationError",
        ),
        ("types/np-bool::*", "exc_type"): (
            "list-element",
            "ValueError",
            "dict-literal",
            "UnsupportedDtypeError",
            "RemoteSerializationError",
        ),
    }
    for key, fragments in expected_fragments.items():
        reason = entries[key].reason
        assert all(fragment in reason for fragment in fragments)
