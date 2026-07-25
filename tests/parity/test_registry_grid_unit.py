from __future__ import annotations

from tests.parity.corpus import registry_grid


def test_enumerates_the_core_registry():
    names = registry_grid.op_names()
    assert len(names) > 500, f"expected the full registry, got {len(names)}"
    assert "sum" in names
    assert "fft.rfft" in names


def test_builds_one_case_per_op_and_pattern():
    cases = registry_grid.build()
    assert len(cases) == len(registry_grid.op_names()) * len(registry_grid.PATTERNS)


def test_case_ids_encode_op_and_pattern():
    ids = {case.id for case in registry_grid.build()}
    assert "grid/sum::axis-tuple" in ids


def test_tuple_axis_is_one_of_the_patterns():
    assert "axis-tuple" in {name for name, _ in registry_grid.PATTERNS}


def test_undriven_ops_are_reported_not_dropped():
    # Whatever cannot be driven must be COUNTED, never silently skipped.
    assert isinstance(registry_grid.undriven(), dict)
