"""Known client/in-process divergences, with a written reason for each.

Strictness runs in BOTH directions:

* a divergence with no entry FAILS (a new regression);
* an entry with no matching divergence FAILS as stale and must be deleted;
* a KNOWN_BUG entry without an issue link FAILS at load time.

Fixing a defect therefore forces the deletion of its entry in the same pull
request. ``KNOWN_BUG`` reaching zero is the definition of done for the whole
wire-protocol programme. ``PROXY_INHERENT`` should stay small; a creeping count
is a smell that belongs in review.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from tests.parity.compare import DIMENSIONS, Divergence


class Category(enum.Enum):
    #: Cannot work through a remote proxy, ever (e.g. a local buffer pointer).
    PROXY_INHERENT = "proxy-inherent"
    #: A deliberate, documented product choice.
    ACCEPTED_DIVERGENCE = "accepted"
    #: A real defect, tracked, not yet fixed.
    KNOWN_BUG = "known-bug"


@dataclass(frozen=True)
class Entry:
    case_id: str
    dimension: str
    category: Category
    reason: str
    issue: str | None = None

    def key(self) -> tuple[str, str]:
        return (self.case_id, self.dimension)


@dataclass(frozen=True)
class AllowlistResult:
    unexplained: list[Divergence]
    stale: list[Entry]
    allowed: list[Divergence]
    counts: dict[str, int]


#: Populated by Task 11 from a first full run. Starts empty so that the eight
#: meta-tests and the canary in Task 6 exercise the unexplained path.
ENTRIES: tuple[Entry, ...] = ()


def validate_entries(entries: tuple[Entry, ...]) -> list[str]:
    """Return a list of schema problems; empty means valid."""
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        where = f"{entry.case_id}[{entry.dimension}]"
        if entry.dimension not in DIMENSIONS:
            problems.append(f"{where}: {entry.dimension!r} is not a known dimension")
        if not entry.reason.strip():
            problems.append(f"{where}: every entry needs a non-empty reason")
        if entry.category is Category.KNOWN_BUG and not (entry.issue or "").strip():
            problems.append(f"{where}: category known-bug requires an issue link")
        if entry.key() in seen:
            problems.append(f"{where}: duplicate entry")
        seen.add(entry.key())
    return problems


def apply(
    divergences: list[Divergence], entries: tuple[Entry, ...] = ENTRIES
) -> AllowlistResult:
    """Split *divergences* into allowed and unexplained, and find stale entries."""
    by_key = {entry.key(): entry for entry in entries}
    matched: set[tuple[str, str]] = set()
    allowed: list[Divergence] = []
    unexplained: list[Divergence] = []

    for divergence in divergences:
        if divergence.key() in by_key:
            allowed.append(divergence)
            matched.add(divergence.key())
        else:
            unexplained.append(divergence)

    stale = [entry for entry in entries if entry.key() not in matched]
    counts = {category.value: 0 for category in Category}
    for entry in entries:
        counts[entry.category.value] += 1
    return AllowlistResult(unexplained, stale, allowed, counts)
