"""Assemble every corpus source into one list of cases."""

from __future__ import annotations

from tests.parity.case import Case
from tests.parity.corpus import idioms


def all_cases() -> list[Case]:
    """Every case from every source."""
    return list(idioms.CASES)


def fast_cases() -> list[Case]:
    """The blocking tier: one case per family, plus the surface check."""
    return [case for case in all_cases() if "tier:fast" in case.tags]
