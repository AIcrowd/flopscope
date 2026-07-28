from __future__ import annotations

from tests.parity.corpus import registry_grid


def test_enumerates_the_core_registry():
    names = registry_grid.op_names()
    assert len(names) > 500, f"expected the full registry, got {len(names)}"
    assert "sum" in names
    assert "fft.rfft" in names


def test_builds_one_case_per_op_and_pattern_minus_exclusions():
    cases = registry_grid.build()
    full_grid = len(registry_grid.op_names()) * len(registry_grid.PATTERNS)
    excluded = (
        registry_grid.SEGFAULT_EXCLUDED_CASE_IDS
        | registry_grid.UNINITIALIZED_VALUE_EXCLUDED_CASE_IDS
    )
    assert len(cases) == full_grid - len(excluded)


def test_case_ids_encode_op_and_pattern():
    ids = {case.id for case in registry_grid.build()}
    assert "grid/sum::axis-tuple" in ids


def test_tuple_axis_is_one_of_the_patterns():
    assert "axis-tuple" in {name for name, _ in registry_grid.PATTERNS}


def test_undriven_ops_are_reported_not_dropped():
    # Whatever cannot be driven must be COUNTED, never silently skipped.
    assert isinstance(registry_grid.undriven(), dict)


def test_segfaulting_cases_are_excluded_from_the_built_corpus():
    # These 17 case ids crash the in-process worker (SIGSEGV, exit 139); a
    # crash cannot be caught, so they must never reach `build()`'s output.
    ids = {case.id for case in registry_grid.build()}
    assert registry_grid.SEGFAULT_EXCLUDED_CASE_IDS, "exclusion set must not be empty"
    assert ids.isdisjoint(registry_grid.SEGFAULT_EXCLUDED_CASE_IDS)


def test_undriven_accounts_for_every_segfault_exclusion():
    # Silent truncation is the failure mode this harness exists to prevent:
    # every excluded case must be COUNTED somewhere, with a reason attached.
    undriven = registry_grid.undriven()
    for case_id in registry_grid.SEGFAULT_EXCLUDED_CASE_IDS:
        assert case_id in undriven
        assert undriven[case_id].strip()


def test_exactly_seventeen_cases_are_segfault_excluded():
    # Pinned to the measured count so a silent change in the exclusion set
    # (accidentally widening or narrowing it) fails loudly.
    assert len(registry_grid.SEGFAULT_EXCLUDED_CASE_IDS) == 17


def test_uninitialized_value_cases_are_excluded_from_the_built_corpus():
    # `empty`/`empty_like` return uninitialized memory for these patterns;
    # their value can never be meaningfully compared, so they must never
    # reach `build()`'s output.
    ids = {case.id for case in registry_grid.build()}
    assert registry_grid.UNINITIALIZED_VALUE_EXCLUDED_CASE_IDS, (
        "exclusion set must not be empty"
    )
    assert ids.isdisjoint(registry_grid.UNINITIALIZED_VALUE_EXCLUDED_CASE_IDS)


def test_undriven_accounts_for_every_uninitialized_value_exclusion():
    # Silent truncation is the failure mode this harness exists to prevent:
    # every excluded case must be COUNTED somewhere, with a reason attached.
    undriven = registry_grid.undriven()
    for case_id in registry_grid.UNINITIALIZED_VALUE_EXCLUDED_CASE_IDS:
        assert case_id in undriven
        assert undriven[case_id].strip()


def test_exactly_nine_cases_are_uninitialized_value_excluded():
    # Pinned to the measured count so a silent change in the exclusion set
    # (accidentally widening or narrowing it) fails loudly.
    assert len(registry_grid.UNINITIALIZED_VALUE_EXCLUDED_CASE_IDS) == 9
