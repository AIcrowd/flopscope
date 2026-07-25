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
import fnmatch
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
    """One allowlisted divergence, or one root-cause-shaped family of them.

    ``case_id`` is either an exact case id (``idiom/complex-mul``) or an
    ``fnmatch`` glob (``grid/fft.*::*``) — the same idiom
    ``tests/client_compat/conftest.py`` uses for its xfail patterns. A glob
    lets one entry explain every divergence that shares one root cause
    instead of forcing an individual entry per case, but it can also hide a
    lot: see ``AllowlistResult.match_counts``, which reports exactly how many
    divergences each entry matched, so a glob silently covering hundreds of
    cases is visible to a reviewer rather than invisible.
    """

    case_id: str
    dimension: str
    category: Category
    reason: str
    issue: str | None = None

    def key(self) -> tuple[str, str]:
        return (self.case_id, self.dimension)

    def matches(self, divergence: Divergence) -> bool:
        """True if this entry explains *divergence*.

        The dimension must match exactly. The case id must either be
        identical to the divergence's case id, or match it as an ``fnmatch``
        glob.
        """
        if self.dimension != divergence.dimension:
            return False
        return divergence.case_id == self.case_id or fnmatch.fnmatch(
            divergence.case_id, self.case_id
        )


@dataclass(frozen=True)
class AllowlistResult:
    unexplained: list[Divergence]
    stale: list[Entry]
    allowed: list[Divergence]
    counts: dict[str, int]
    #: How many divergences each entry matched. A glob entry that silently
    #: covers hundreds of divergences must be obvious to a reviewer, not
    #: invisible; this is what makes that visible. An entry present in
    #: ``entries`` but absent (or mapped to 0) here is stale.
    match_counts: dict[Entry, int]


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
    """Split *divergences* into allowed and unexplained, and find stale entries.

    A divergence matched by no entry FAILS as unexplained; an entry matching
    no divergence FAILS as stale. A divergence may be matched by more than
    one entry (overlapping globs); it is allowed as long as at least one
    entry matches, and every matching entry's count in ``match_counts`` is
    incremented.
    """
    match_counts: dict[Entry, int] = dict.fromkeys(entries, 0)
    allowed: list[Divergence] = []
    unexplained: list[Divergence] = []

    for divergence in divergences:
        matching = [entry for entry in entries if entry.matches(divergence)]
        if not matching:
            unexplained.append(divergence)
            continue
        allowed.append(divergence)
        for entry in matching:
            match_counts[entry] += 1

    stale = [entry for entry in entries if match_counts[entry] == 0]
    counts = {category.value: 0 for category in Category}
    for entry in entries:
        counts[entry.category.value] += 1
    return AllowlistResult(unexplained, stale, allowed, counts, match_counts)
