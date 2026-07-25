"""The corpus must be well-formed before it is worth running."""

from __future__ import annotations

from typing import cast

from tests.parity.corpus import all_cases, fast_cases


def test_every_case_id_is_unique():
    ids = [case.id for case in all_cases()]
    assert len(ids) == len(set(ids))


def test_every_idiom_case_names_a_family():
    for case in all_cases():
        if case.id.startswith("idiom/"):
            assert case.family() is not None, f"{case.id} has no family tag"


def test_fast_cases_are_a_subset_tagged_for_the_fast_tier():
    fast = fast_cases()
    assert fast, "the fast tier must not be empty"
    for case in fast:
        assert "tier:fast" in case.tags
    assert set(fast).issubset(set(all_cases()))


def test_every_family_has_at_least_one_fast_case():
    families: set[str] = cast(
        set[str], {c.family() for c in all_cases() if c.family() is not None}
    )
    fast_families: set[str] = cast(
        set[str], {c.family() for c in fast_cases() if c.family() is not None}
    )
    assert families == fast_families, (
        f"families with no fast-tier case: {sorted(families - fast_families)}"
    )


def test_no_audit_prose_leaked_into_the_corpus():
    # The repo is public; the audit is not.
    import pathlib

    text = pathlib.Path("tests/parity/corpus/idioms.py").read_text()
    for banned in ("exploit", "budget bypass", "participant-venv"):
        assert banned not in text.lower()
