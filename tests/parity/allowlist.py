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


#: Populated from a full differential corpus run: both backends were driven
#: over the whole case corpus, every divergence found was triaged, and each
#: entry below records one root cause. Each entry is grouped by root cause: a
#: whole missing client namespace, a whole server-side error-wrapping
#: mechanism, a single op, or a single case. `KNOWN_BUG` reaching zero is the
#: definition of done for the whole wire-protocol programme.
ENTRIES: tuple[Entry, ...] = (
    Entry(
        case_id="grid/fft.*::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `fft.fft` attribute (the dotted"
            " namespace it lives under is missing entirely on the client), so the"
            " client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape. (18 ops in this family.)"
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/fft.*::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `fft.fft` attribute, so the client's"
            " exception is AttributeError; in-process the call reaches a real"
            " implementation and raises a more specific exception, so the base-class"
            " chain differs too. (18 ops in this family.)"
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/fft.*::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `fft.fft` attribute. For grid patterns"
            " whose arguments happen to be valid, the in-process call succeeds, while"
            " the client always raises AttributeError before dispatch. (18 ops in this"
            " family.)"
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/fft.*::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `fft.fft` attribute, so the call never"
            " dispatches and 0 FLOPs are billed on the client, while in-process the"
            " call runs and bills its real cost. (18 ops in this family.)"
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/random.Generator.*::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `random.Generator.bit_generator`"
            " attribute (the dotted namespace it lives under is missing entirely on the"
            " client), so the client raises AttributeError before the call is ever"
            " dispatched. In-process the same dotted path resolves to numpy's real"
            " callable, so the exact exception raised in-process instead depends on the"
            " grid pattern's argument shape. (30 ops in this family.)"
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/random.RandomState.*::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `random.RandomState.beta` attribute (the"
            " dotted namespace it lives under is missing entirely on the client), so"
            " the client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape. (49 ops in this family.)"
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/linalg.*::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `linalg.cholesky` and a real server-"
            " side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape). (22 ops"
            " in this family.)"
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/linalg.*::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `linalg.cholesky`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError). (22 ops in this family.)"
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/linalg.*::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `linalg.cholesky`'s server-side failure in a generic"
            " `FlopscopeServerError` for a pattern where in-process the call succeeds"
            " outright. (22 ops in this family.)"
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/stats.*::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`stats.cauchy.cdf` is listed in the core op registry, but the dotted"
            " attribute path is not actually reachable on the in-process"
            " `flopscope.numpy` module, so in-process this call raises AttributeError."
            " The client's dispatch does not require the same attribute path and"
            " reaches a real implementation instead, so it raises a different, pattern-"
            " dependent exception (or succeeds). This is the mirror image of the"
            " client-surface-gap family (the gap is on the in-process side here); no"
            " family in the mapping covers that direction, so it is filed unclassified."
            " (24 ops in this family.)"
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/stats.*::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`stats.cauchy.cdf`'s dotted attribute path is not reachable on the in-"
            " process `flopscope.numpy` module, so in-process this call always raises"
            " AttributeError, while the client's dispatch reaches a real implementation"
            " and succeeds for patterns whose arguments are valid. Unclassified: no"
            " given family covers an in-process (rather than client) attribute gap. (24"
            " ops in this family.)"
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/stats.*::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "`stats.cauchy.cdf`'s dotted attribute path is not reachable in-process,"
            " so the in-process call raises before billing anything (0 FLOPs), while"
            " the client's dispatch reaches a real implementation and bills its real"
            " cost. Unclassified: no given family covers an in-process attribute gap."
            " (24 ops in this family.)"
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/base_repr::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `base_repr` attribute (the dotted"
            " namespace it lives under is missing entirely on the client), so the"
            " client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/base_repr::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `base_repr` attribute. For grid patterns"
            " whose arguments happen to be valid, the in-process call succeeds, while"
            " the client always raises AttributeError before dispatch."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/base_repr::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `base_repr` attribute, so the call never"
            " dispatches and 0 FLOPs are billed on the client, while in-process the"
            " call runs and bills its real cost."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/binary_repr::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `binary_repr` attribute (the dotted"
            " namespace it lives under is missing entirely on the client), so the"
            " client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/broadcast::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `broadcast` attribute (the dotted"
            " namespace it lives under is missing entirely on the client), so the"
            " client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/errstate::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `errstate` attribute (the dotted"
            " namespace it lives under is missing entirely on the client), so the"
            " client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/fromfile::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `fromfile` attribute (the dotted"
            " namespace it lives under is missing entirely on the client), so the"
            " client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/fromregex::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `fromregex` attribute (the dotted"
            " namespace it lives under is missing entirely on the client), so the"
            " client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/fromstring::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `fromstring` attribute (the dotted"
            " namespace it lives under is missing entirely on the client), so the"
            " client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/fromstring::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `fromstring` attribute. For grid"
            " patterns whose arguments happen to be valid, the in-process call"
            " succeeds, while the client always raises AttributeError before dispatch."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/fromstring::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `fromstring` attribute, so the call"
            " never dispatches and 0 FLOPs are billed on the client, while in-process"
            " the call runs and bills its real cost."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/get_printoptions::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `get_printoptions` attribute (the dotted"
            " namespace it lives under is missing entirely on the client), so the"
            " client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/get_printoptions::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `get_printoptions` attribute. For grid"
            " patterns whose arguments happen to be valid, the in-process call"
            " succeeds, while the client always raises AttributeError before dispatch."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/geterr::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `geterr` attribute (the dotted namespace"
            " it lives under is missing entirely on the client), so the client raises"
            " AttributeError before the call is ever dispatched. In-process the same"
            " dotted path resolves to numpy's real callable, so the exact exception"
            " raised in-process instead depends on the grid pattern's argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/geterr::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `geterr` attribute. For grid patterns"
            " whose arguments happen to be valid, the in-process call succeeds, while"
            " the client always raises AttributeError before dispatch."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/isnat::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `isnat` attribute (the dotted namespace"
            " it lives under is missing entirely on the client), so the client raises"
            " AttributeError before the call is ever dispatched. In-process the same"
            " dotted path resolves to numpy's real callable, so the exact exception"
            " raised in-process instead depends on the grid pattern's argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/isnat::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `isnat` attribute, so the call never"
            " dispatches and 0 FLOPs are billed on the client, while in-process the"
            " call runs and bills its real cost."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/ndenumerate::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `ndenumerate` attribute (the dotted"
            " namespace it lives under is missing entirely on the client), so the"
            " client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/ndindex::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `ndindex` attribute (the dotted"
            " namespace it lives under is missing entirely on the client), so the"
            " client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/nditer::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `nditer` attribute (the dotted namespace"
            " it lives under is missing entirely on the client), so the client raises"
            " AttributeError before the call is ever dispatched. In-process the same"
            " dotted path resolves to numpy's real callable, so the exact exception"
            " raised in-process instead depends on the grid pattern's argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/set_printoptions::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `set_printoptions` attribute (the dotted"
            " namespace it lives under is missing entirely on the client), so the"
            " client raises AttributeError before the call is ever dispatched. In-"
            " process the same dotted path resolves to numpy's real callable, so the"
            " exact exception raised in-process instead depends on the grid pattern's"
            " argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/set_printoptions::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `set_printoptions` attribute. For grid"
            " patterns whose arguments happen to be valid, the in-process call"
            " succeeds, while the client always raises AttributeError before dispatch."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/seterr::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `seterr` attribute (the dotted namespace"
            " it lives under is missing entirely on the client), so the client raises"
            " AttributeError before the call is ever dispatched. In-process the same"
            " dotted path resolves to numpy's real callable, so the exact exception"
            " raised in-process instead depends on the grid pattern's argument shape."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/seterr::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `seterr` attribute, so the client's"
            " exception is AttributeError; in-process the call reaches a real"
            " implementation and raises a more specific exception, so the base-class"
            " chain differs too."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/seterr::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `seterr` attribute. For grid patterns"
            " whose arguments happen to be valid, the in-process call succeeds, while"
            " the client always raises AttributeError before dispatch."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/seterr::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `fnp` module has no `seterr` attribute, so the call never"
            " dispatches and 0 FLOPs are billed on the client, while in-process the"
            " call runs and bills its real cost."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="grid/array_split::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `array_split` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/clip::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `clip` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/clip::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `clip`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/compress::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `compress` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/compress::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `compress`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/corrcoef::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `corrcoef` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/corrcoef::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `corrcoef`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/cov::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `cov` and a real server-side failure is"
            " surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/cov::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `cov`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/delete::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `delete` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/delete::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `delete`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/diff::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `diff` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/diff::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `diff`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/dot::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `dot` and a real server-side failure is"
            " surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/dot::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `dot`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/dsplit::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `dsplit` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/extract::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `extract` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/extract::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `extract`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/fliplr::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `fliplr` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/fliplr::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `fliplr`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/flipud::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `flipud` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/flipud::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `flipud`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/full_like::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `full_like` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/gcd::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `gcd` and a real server-side failure is"
            " surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/gcd::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `gcd`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/gradient::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `gradient` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/gradient::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `gradient`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/histogramdd::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `histogramdd` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/histogramdd::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `histogramdd`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/hsplit::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `hsplit` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/inner::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `inner` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/inner::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `inner`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/lcm::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `lcm` and a real server-side failure is"
            " surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/lcm::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `lcm`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/mintypecode::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `mintypecode` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/mintypecode::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `mintypecode`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/nanpercentile::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `nanpercentile` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/nanquantile::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `nanquantile` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/percentile::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `percentile` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/positive::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `positive` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/positive::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `positive`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/quantile::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `quantile` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/rot90::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `rot90` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/rot90::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `rot90`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/sign::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `sign` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/sign::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `sign`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/sort::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `sort` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/sort::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `sort`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/sort_complex::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `sort_complex` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/sort_complex::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `sort_complex`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/split::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `split` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/tensordot::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `tensordot` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/tensordot::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `tensordot`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/tile::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `tile` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/trapezoid::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `trapezoid` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/trapezoid::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `trapezoid`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/trapz::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `trapz` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/trapz::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `trapz`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/vsplit::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `vsplit` and a real server-side failure"
            " is surfaced to the client as a generic `FlopscopeServerError`, losing the"
            " concrete exception type numpy raises in-process (which varies with the"
            " grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.beta::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.beta` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.chisquare::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.chisquare` and a real server-"
            " side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.choice::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.choice` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.dirichlet::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.dirichlet` and a real server-"
            " side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.exponential::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.exponential` and a real server-"
            " side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.f::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.f` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.gamma::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.gamma` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.geometric::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.geometric` and a real server-"
            " side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.gumbel::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.gumbel` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.laplace::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.laplace` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.logistic::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.logistic` and a real server-"
            " side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.lognormal::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.lognormal` and a real server-"
            " side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.logseries::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.logseries` and a real server-"
            " side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.negative_binomial::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.negative_binomial` and a real"
            " server-side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.noncentral_chisquare::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.noncentral_chisquare` and a"
            " real server-side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.normal::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.normal` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.pareto::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.pareto` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.permutation::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.permutation` and a real server-"
            " side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.permutation::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `random.permutation`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.poisson::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.poisson` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.power::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.power` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.random::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.random` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.random_sample::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.random_sample` and a real"
            " server-side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.ranf::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.ranf` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.rayleigh::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.rayleigh` and a real server-"
            " side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.sample::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.sample` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.shuffle::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.shuffle` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.shuffle::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client wraps `random.shuffle`'s server-side failure in a generic"
            " `FlopscopeServerError`, so the exception's base-class chain collapses to"
            " `[Exception, BaseException]` instead of the concrete in-process hierarchy"
            " (e.g. LookupError, ValueError, ArithmeticError)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.standard_gamma::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.standard_gamma` and a real"
            " server-side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.standard_t::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.standard_t` and a real server-"
            " side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.uniform::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.uniform` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.vonmises::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.vonmises` and a real server-"
            " side failure is surfaced to the client as a generic"
            " `FlopscopeServerError`, losing the concrete exception type numpy raises"
            " in-process (which varies with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.wald::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.wald` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.weibull::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.weibull` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/random.zipf::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client reaches the server for `random.zipf` and a real server-side"
            " failure is surfaced to the client as a generic `FlopscopeServerError`,"
            " losing the concrete exception type numpy raises in-process (which varies"
            " with the grid pattern's argument shape)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/as_symmetric::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`as_symmetric` is listed in the core op registry, but the dotted"
            " attribute path is not actually reachable on the in-process"
            " `flopscope.numpy` module, so in-process this call raises AttributeError."
            " The client's dispatch does not require the same attribute path and"
            " reaches a real implementation instead, so it raises a different, pattern-"
            " dependent exception (or succeeds). This is the mirror image of the"
            " client-surface-gap family (the gap is on the in-process side here); no"
            " family in the mapping covers that direction, so it is filed unclassified."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/common_type::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`common_type` is listed in the core op registry, but the dotted attribute"
            " path is not actually reachable on the in-process `flopscope.numpy`"
            " module, so in-process this call raises AttributeError. The client's"
            " dispatch does not require the same attribute path and reaches a real"
            " implementation instead, so it raises a different, pattern-dependent"
            " exception (or succeeds). This is the mirror image of the client-surface-"
            " gap family (the gap is on the in-process side here); no family in the"
            " mapping covers that direction, so it is filed unclassified."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/common_type::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`common_type`'s dotted attribute path is not reachable on the in-process"
            " `flopscope.numpy` module, so in-process this call always raises"
            " AttributeError, while the client's dispatch reaches a real implementation"
            " and succeeds for patterns whose arguments are valid. Unclassified: no"
            " given family covers an in-process (rather than client) attribute gap."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/getitem::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`getitem` is listed in the core op registry, but the dotted attribute"
            " path is not actually reachable on the in-process `flopscope.numpy`"
            " module, so in-process this call raises AttributeError. The client's"
            " dispatch does not require the same attribute path and reaches a real"
            " implementation instead, so it raises a different, pattern-dependent"
            " exception (or succeeds). This is the mirror image of the client-surface-"
            " gap family (the gap is on the in-process side here); no family in the"
            " mapping covers that direction, so it is filed unclassified."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/is_symmetric::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`is_symmetric` is listed in the core op registry, but the dotted"
            " attribute path is not actually reachable on the in-process"
            " `flopscope.numpy` module, so in-process this call raises AttributeError."
            " The client's dispatch does not require the same attribute path and"
            " reaches a real implementation instead, so it raises a different, pattern-"
            " dependent exception (or succeeds). This is the mirror image of the"
            " client-surface-gap family (the gap is on the in-process side here); no"
            " family in the mapping covers that direction, so it is filed unclassified."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/random.symmetric::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`random.symmetric` is listed in the core op registry, but the dotted"
            " attribute path is not actually reachable on the in-process"
            " `flopscope.numpy` module, so in-process this call raises AttributeError."
            " The client's dispatch does not require the same attribute path and"
            " reaches a real implementation instead, so it raises a different, pattern-"
            " dependent exception (or succeeds). This is the mirror image of the"
            " client-surface-gap family (the gap is on the in-process side here); no"
            " family in the mapping covers that direction, so it is filed unclassified."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/symmetrize::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`symmetrize` is listed in the core op registry, but the dotted attribute"
            " path is not actually reachable on the in-process `flopscope.numpy`"
            " module, so in-process this call raises AttributeError. The client's"
            " dispatch does not require the same attribute path and reaches a real"
            " implementation instead, so it raises a different, pattern-dependent"
            " exception (or succeeds). This is the mirror image of the client-surface-"
            " gap family (the gap is on the in-process side here); no family in the"
            " mapping covers that direction, so it is filed unclassified."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/trim_zeros::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`trim_zeros` is listed in the core op registry, but the dotted attribute"
            " path is not actually reachable on the in-process `flopscope.numpy`"
            " module, so in-process this call raises AttributeError. The client's"
            " dispatch does not require the same attribute path and reaches a real"
            " implementation instead, so it raises a different, pattern-dependent"
            " exception (or succeeds). This is the mirror image of the client-surface-"
            " gap family (the gap is on the in-process side here); no family in the"
            " mapping covers that direction, so it is filed unclassified."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/cumulative_prod::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client independently raises `TypeError` for `cumulative_prod` instead"
            " of proxying the concrete exception numpy raises in-process for this"
            " pattern (a milder variant of the client's generic error wrapping: here"
            " the client raises its own validation error rather than a transport"
            " wrapper)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/cumulative_sum::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client independently raises `TypeError` for `cumulative_sum` instead"
            " of proxying the concrete exception numpy raises in-process for this"
            " pattern (a milder variant of the client's generic error wrapping: here"
            " the client raises its own validation error rather than a transport"
            " wrapper)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/einsum::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client independently raises `TypeError` for `einsum` instead of"
            " proxying the concrete exception numpy raises in-process for this pattern"
            " (a milder variant of the client's generic error wrapping: here the client"
            " raises its own validation error rather than a transport wrapper)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/einsum::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client independently raises `TypeError` for `einsum` instead of"
            " proxying the concrete exception numpy raises in-process for this pattern"
            " (a milder variant of the client's generic error wrapping: here the client"
            " raises its own validation error rather than a transport wrapper)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/finfo::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client independently raises `TypeError` for `finfo` instead of"
            " proxying the concrete exception numpy raises in-process for this pattern"
            " (a milder variant of the client's generic error wrapping: here the client"
            " raises its own validation error rather than a transport wrapper)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/iinfo::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client independently raises `TypeError` for `iinfo` instead of"
            " proxying the concrete exception numpy raises in-process for this pattern"
            " (a milder variant of the client's generic error wrapping: here the client"
            " raises its own validation error rather than a transport wrapper)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/squeeze::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The client independently raises `TypeError` for `squeeze` instead of"
            " proxying the concrete exception numpy raises in-process for this pattern"
            " (a milder variant of the client's generic error wrapping: here the client"
            " raises its own validation error rather than a transport wrapper)."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/load::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`load` is a file-I/O op; given the grid pattern's array argument (not a"
            " real path), in-process `numpy.load` raises TypeError from argument"
            " validation, while the client instead attempts to treat the argument as"
            " file content/path and raises OSError."
        ),
        issue="INTERNAL-P5-family-3",
    ),
    Entry(
        case_id="grid/savez*::*",
        dimension="flops",
        category=Category.ACCEPTED_DIVERGENCE,
        reason=(
            "In-process `numpy.savez` validates its arguments only after the op has"
            " been charged, so the grid pattern's bad call bills 176 and then raises"
            " TypeError. The server refuses the same call before dispatch and charges"
            " nothing. Charging for a call that produces no result is the worse of the"
            " two behaviours, so the client staying at zero here is deliberate -- the"
            " same refuse-before-charging rule the server applies everywhere else."
        ),
        issue="INTERNAL-P3-refuse-before-charging",
    ),
    Entry(
        case_id="grid/savez*::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "Both backends reject the grid pattern's bad `savez` call, but not with"
            " the same class: in-process numpy raises `TypeError` (bases Exception,"
            " BaseException) from its own argument validation, while the server's"
            " refusal surfaces as a `LookupError` subclass (bases LookupError,"
            " Exception, BaseException). Refusing is right; refusing under a"
            " different base-class chain is not, and code catching `TypeError` sees"
            " only one of them."
        ),
        issue="INTERNAL-P5-family-3",
    ),
    Entry(
        case_id="grid/savez::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`savez` is a file-I/O op; given the grid pattern's array argument (not a"
            " real path), in-process `numpy.savez` raises TypeError from argument"
            " validation, while the client instead attempts to treat the argument as"
            " file content/path and raises OSError."
        ),
        issue="INTERNAL-P5-family-3",
    ),
    Entry(
        case_id="grid/savez_compressed::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`savez_compressed` is a file-I/O op; given the grid pattern's array"
            " argument (not a real path), in-process `numpy.savez_compressed` raises"
            " TypeError from argument validation, while the client instead attempts to"
            " treat the argument as file content/path and raises OSError."
        ),
        issue="INTERNAL-P5-family-3",
    ),
    Entry(
        case_id="grid/astype::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`astype` raises `TypeError` in-process uniformly for these patterns; the"
            " client instead either wraps a server failure into `FlopscopeServerError`"
            " or raises `ValueError` directly, depending on the pattern - the concrete"
            " exception identity does not survive the wire."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/astype::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "For some grid patterns `astype` raises in-process while the client's"
            " dispatch succeeds and returns a value instead."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/astype::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "`astype`'s exception happens at a different point in each backend's"
            " processing, so the FLOPs billed before failing differ too."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="grid/allclose::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`allclose` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " container disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/allclose::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`allclose` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " pytype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/allclose::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`allclose` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " dtype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/allclose::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`allclose` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " shape disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/array_equal::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`array_equal` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so container disagrees (the same mechanism as the `ndim`-returns-"
            " int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/array_equal::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`array_equal` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so pytype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/array_equal::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`array_equal` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so dtype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/array_equal::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`array_equal` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so shape disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/array_equiv::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`array_equiv` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so container disagrees (the same mechanism as the `ndim`-returns-"
            " int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/array_equiv::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`array_equiv` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so pytype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/array_equiv::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`array_equiv` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so dtype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/array_equiv::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`array_equiv` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so shape disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/iscomplexobj::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`iscomplexobj` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so container disagrees (the same mechanism as the `ndim`-returns-"
            " int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/iscomplexobj::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`iscomplexobj` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so pytype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/iscomplexobj::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`iscomplexobj` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so dtype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/iscomplexobj::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`iscomplexobj` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so shape disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isfortran::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`isfortran` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " container disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isfortran::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`isfortran` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " pytype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isfortran::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`isfortran` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " dtype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isfortran::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`isfortran` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " shape disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isrealobj::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`isrealobj` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " container disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isrealobj::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`isrealobj` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " pytype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isrealobj::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`isrealobj` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " dtype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isrealobj::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`isrealobj` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " shape disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isscalar::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`isscalar` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " container disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isscalar::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`isscalar` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " pytype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isscalar::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`isscalar` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " dtype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isscalar::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`isscalar` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " shape disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/iterable::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`iterable` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " container disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/iterable::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`iterable` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " pytype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/iterable::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`iterable` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " dtype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/iterable::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`iterable` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " shape disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/may_share_memory::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`may_share_memory` returns a plain Python/NumPy predicate or count value"
            " (a bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so container disagrees (the same mechanism as the `ndim`-returns-"
            " int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/may_share_memory::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`may_share_memory` returns a plain Python/NumPy predicate or count value"
            " (a bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so pytype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/may_share_memory::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`may_share_memory` returns a plain Python/NumPy predicate or count value"
            " (a bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so dtype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/may_share_memory::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`may_share_memory` returns a plain Python/NumPy predicate or count value"
            " (a bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so shape disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/shares_memory::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`shares_memory` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so container disagrees (the same mechanism as the `ndim`-returns-"
            " int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/shares_memory::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`shares_memory` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so pytype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/shares_memory::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`shares_memory` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so dtype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/shares_memory::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`shares_memory` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so shape disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/ndim::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`ndim` returns a plain Python/NumPy predicate or count value (a bool or"
            " an int) in-process for this pattern; the client wraps the same result in"
            " a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " container disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/ndim::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`ndim` returns a plain Python/NumPy predicate or count value (a bool or"
            " an int) in-process for this pattern; the client wraps the same result in"
            " a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " pytype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/ndim::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`ndim` returns a plain Python/NumPy predicate or count value (a bool or"
            " an int) in-process for this pattern; the client wraps the same result in"
            " a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " dtype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/ndim::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`ndim` returns a plain Python/NumPy predicate or count value (a bool or"
            " an int) in-process for this pattern; the client wraps the same result in"
            " a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " shape disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/size::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`size` returns a plain Python/NumPy predicate or count value (a bool or"
            " an int) in-process for this pattern; the client wraps the same result in"
            " a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " container disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/size::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`size` returns a plain Python/NumPy predicate or count value (a bool or"
            " an int) in-process for this pattern; the client wraps the same result in"
            " a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " pytype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/size::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`size` returns a plain Python/NumPy predicate or count value (a bool or"
            " an int) in-process for this pattern; the client wraps the same result in"
            " a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " dtype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/size::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`size` returns a plain Python/NumPy predicate or count value (a bool or"
            " an int) in-process for this pattern; the client wraps the same result in"
            " a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " shape disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/count_nonzero::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`count_nonzero` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so container disagrees (the same mechanism as the `ndim`-returns-"
            " int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/count_nonzero::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`count_nonzero` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so pytype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/count_nonzero::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`count_nonzero` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so dtype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/count_nonzero::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`count_nonzero` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so shape disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/linalg.matrix_rank::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.matrix_rank` returns a plain Python/NumPy predicate or count"
            " value (a bool or an int) in-process for this pattern; the client wraps"
            " the same result in a `RemoteScalar` proxy instead of unwrapping it to the"
            " bare value, so container disagrees (the same mechanism as the"
            " `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/linalg.matrix_rank::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.matrix_rank` returns a plain Python/NumPy predicate or count"
            " value (a bool or an int) in-process for this pattern; the client wraps"
            " the same result in a `RemoteScalar` proxy instead of unwrapping it to the"
            " bare value, so pytype disagrees (the same mechanism as the"
            " `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/linalg.matrix_rank::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.matrix_rank` returns a plain Python/NumPy predicate or count"
            " value (a bool or an int) in-process for this pattern; the client wraps"
            " the same result in a `RemoteScalar` proxy instead of unwrapping it to the"
            " bare value, so dtype disagrees (the same mechanism as the `ndim`-returns-"
            " int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/linalg.matrix_rank::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.matrix_rank` returns a plain Python/NumPy predicate or count"
            " value (a bool or an int) in-process for this pattern; the client wraps"
            " the same result in a `RemoteScalar` proxy instead of unwrapping it to the"
            " bare value, so shape disagrees (the same mechanism as the `ndim`-returns-"
            " int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isfinite::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`isfinite` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " pytype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isinf::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`isinf` returns a plain Python/NumPy predicate or count value (a bool or"
            " an int) in-process for this pattern; the client wraps the same result in"
            " a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " pytype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/isnan::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`isnan` returns a plain Python/NumPy predicate or count value (a bool or"
            " an int) in-process for this pattern; the client wraps the same result in"
            " a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " pytype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/lexsort::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`lexsort` returns a plain Python/NumPy predicate or count value (a bool"
            " or an int) in-process for this pattern; the client wraps the same result"
            " in a `RemoteScalar` proxy instead of unwrapping it to the bare value, so"
            " pytype disagrees (the same mechanism as the `ndim`-returns-int case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/searchsorted::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`searchsorted` returns a plain Python/NumPy predicate or count value (a"
            " bool or an int) in-process for this pattern; the client wraps the same"
            " result in a `RemoteScalar` proxy instead of unwrapping it to the bare"
            " value, so pytype disagrees (the same mechanism as the `ndim`-returns-int"
            " case)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/trace::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`trace` returns a genuine computed scalar (or `None`) in-process for this"
            " pattern; the client wraps it in a `RemoteScalar` proxy (reporting a dtype"
            " where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/linalg.norm::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.norm` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/linalg.trace::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.trace` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/linalg.vector_norm::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.vector_norm` returns a genuine computed scalar (or `None`) in-"
            " process for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/trapezoid::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`trapezoid` returns a genuine computed scalar (or `None`) in-process for"
            " this pattern; the client wraps it in a `RemoteScalar` proxy (reporting a"
            " dtype where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/trapz::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`trapz` returns a genuine computed scalar (or `None`) in-process for this"
            " pattern; the client wraps it in a `RemoteScalar` proxy (reporting a dtype"
            " where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/corrcoef::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`corrcoef` returns a genuine computed scalar (or `None`) in-process for"
            " this pattern; the client wraps it in a `RemoteScalar` proxy (reporting a"
            " dtype where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/take::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`take` returns a genuine computed scalar (or `None`) in-process for this"
            " pattern; the client wraps it in a `RemoteScalar` proxy (reporting a dtype"
            " where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/trim_zeros::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`trim_zeros` returns a genuine computed scalar (or `None`) in-process for"
            " this pattern; the client wraps it in a `RemoteScalar` proxy (reporting a"
            " dtype where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/vdot::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`vdot` returns a genuine computed scalar (or `None`) in-process for this"
            " pattern; the client wraps it in a `RemoteScalar` proxy (reporting a dtype"
            " where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/linalg.cond::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.cond` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/linalg.multi_dot::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.multi_dot` returns a genuine computed scalar (or `None`) in-"
            " process for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/poly::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`poly` returns a genuine computed scalar (or `None`) in-process for this"
            " pattern; the client wraps it in a `RemoteScalar` proxy (reporting a dtype"
            " where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so container disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/poly::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`poly` returns a genuine computed scalar (or `None`) in-process for this"
            " pattern; the client wraps it in a `RemoteScalar` proxy (reporting a dtype"
            " where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/poly::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`poly` returns a genuine computed scalar (or `None`) in-process for this"
            " pattern; the client wraps it in a `RemoteScalar` proxy (reporting a dtype"
            " where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so dtype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/poly::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`poly` returns a genuine computed scalar (or `None`) in-process for this"
            " pattern; the client wraps it in a `RemoteScalar` proxy (reporting a dtype"
            " where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so shape disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/polyval::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`polyval` returns a genuine computed scalar (or `None`) in-process for"
            " this pattern; the client wraps it in a `RemoteScalar` proxy (reporting a"
            " dtype where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/mintypecode::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`mintypecode` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so container disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/mintypecode::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`mintypecode` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/mintypecode::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`mintypecode` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so dtype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/mintypecode::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`mintypecode` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so shape disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/min_scalar_type::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`min_scalar_type` returns a genuine computed scalar (or `None`) in-"
            " process for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so container disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/min_scalar_type::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`min_scalar_type` returns a genuine computed scalar (or `None`) in-"
            " process for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/min_scalar_type::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`min_scalar_type` returns a genuine computed scalar (or `None`) in-"
            " process for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so dtype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/result_type::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`result_type` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so container disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/result_type::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`result_type` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/result_type::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`result_type` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so dtype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/linalg.matrix_norm::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.matrix_norm` returns a genuine computed scalar (or `None`) in-"
            " process for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/copyto::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`copyto` returns a genuine computed scalar (or `None`) in-process for"
            " this pattern; the client wraps it in a `RemoteScalar` proxy (reporting a"
            " dtype where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so container disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/copyto::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`copyto` returns a genuine computed scalar (or `None`) in-process for"
            " this pattern; the client wraps it in a `RemoteScalar` proxy (reporting a"
            " dtype where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/copyto::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`copyto` returns a genuine computed scalar (or `None`) in-process for"
            " this pattern; the client wraps it in a `RemoteScalar` proxy (reporting a"
            " dtype where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so dtype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/copyto::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`copyto` returns a genuine computed scalar (or `None`) in-process for"
            " this pattern; the client wraps it in a `RemoteScalar` proxy (reporting a"
            " dtype where in-process there is none) instead of unwrapping it to the"
            " equivalent bare value, so shape disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/fill_diagonal::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`fill_diagonal` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so container disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/fill_diagonal::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`fill_diagonal` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/fill_diagonal::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`fill_diagonal` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so dtype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/fill_diagonal::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`fill_diagonal` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so shape disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/random.seed::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`random.seed` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so container disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/random.seed::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`random.seed` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/random.seed::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`random.seed` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so dtype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/random.seed::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`random.seed` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so shape disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/random.shuffle::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`random.shuffle` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so container disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/random.shuffle::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`random.shuffle` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/random.shuffle::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`random.shuffle` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so dtype disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/random.shuffle::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`random.shuffle` returns a genuine computed scalar (or `None`) in-process"
            " for this pattern; the client wraps it in a `RemoteScalar` proxy"
            " (reporting a dtype where in-process there is none) instead of unwrapping"
            " it to the equivalent bare value, so shape disagrees."
        ),
        issue="INTERNAL-P2-family-5",
    ),
    Entry(
        case_id="grid/convolve::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`convolve` returns a real array/tuple in-process for this pattern, but"
            " its specific type identity (an ndarray subclass, a namedtuple, or a list)"
            " is not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/cov::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`cov` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/diff::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`diff` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/ediff1d::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`ediff1d` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/gradient::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`gradient` returns a real array/tuple in-process for this pattern, but"
            " its specific type identity (an ndarray subclass, a namedtuple, or a list)"
            " is not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/kron::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`kron` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/linalg.outer::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.outer` returns a real array/tuple in-process for this pattern,"
            " but its specific type identity (an ndarray subclass, a namedtuple, or a"
            " list) is not preserved once it crosses to the client, so pytype"
            " disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/outer::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`outer` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/sort_complex::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`sort_complex` returns a real array/tuple in-process for this pattern,"
            " but its specific type identity (an ndarray subclass, a namedtuple, or a"
            " list) is not preserved once it crosses to the client, so pytype"
            " disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/diag::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`diag` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/diagflat::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`diagflat` returns a real array/tuple in-process for this pattern, but"
            " its specific type identity (an ndarray subclass, a namedtuple, or a list)"
            " is not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/expand_dims::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`expand_dims` returns a real array/tuple in-process for this pattern, but"
            " its specific type identity (an ndarray subclass, a namedtuple, or a list)"
            " is not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/ones::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`ones` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/zeros::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`zeros` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/bmat::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`bmat` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/unique_all::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`unique_all` returns a real array/tuple in-process for this pattern, but"
            " its specific type identity (an ndarray subclass, a namedtuple, or a list)"
            " is not preserved once it crosses to the client, so container disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/unique_all::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`unique_all` returns a real array/tuple in-process for this pattern, but"
            " its specific type identity (an ndarray subclass, a namedtuple, or a list)"
            " is not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/unique_counts::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`unique_counts` returns a real array/tuple in-process for this pattern,"
            " but its specific type identity (an ndarray subclass, a namedtuple, or a"
            " list) is not preserved once it crosses to the client, so container"
            " disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/unique_counts::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`unique_counts` returns a real array/tuple in-process for this pattern,"
            " but its specific type identity (an ndarray subclass, a namedtuple, or a"
            " list) is not preserved once it crosses to the client, so pytype"
            " disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/unique_inverse::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`unique_inverse` returns a real array/tuple in-process for this pattern,"
            " but its specific type identity (an ndarray subclass, a namedtuple, or a"
            " list) is not preserved once it crosses to the client, so container"
            " disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/unique_inverse::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`unique_inverse` returns a real array/tuple in-process for this pattern,"
            " but its specific type identity (an ndarray subclass, a namedtuple, or a"
            " list) is not preserved once it crosses to the client, so pytype"
            " disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/linalg.qr::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.qr` returns a real array/tuple in-process for this pattern, but"
            " its specific type identity (an ndarray subclass, a namedtuple, or a list)"
            " is not preserved once it crosses to the client, so container disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/linalg.qr::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.qr` returns a real array/tuple in-process for this pattern, but"
            " its specific type identity (an ndarray subclass, a namedtuple, or a list)"
            " is not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/linalg.svd::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.svd` returns a real array/tuple in-process for this pattern, but"
            " its specific type identity (an ndarray subclass, a namedtuple, or a list)"
            " is not preserved once it crosses to the client, so container disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/linalg.svd::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`linalg.svd` returns a real array/tuple in-process for this pattern, but"
            " its specific type identity (an ndarray subclass, a namedtuple, or a list)"
            " is not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/array_split::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`array_split` returns a real array/tuple in-process for this pattern, but"
            " its specific type identity (an ndarray subclass, a namedtuple, or a list)"
            " is not preserved once it crosses to the client, so container disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/array_split::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`array_split` returns a real array/tuple in-process for this pattern, but"
            " its specific type identity (an ndarray subclass, a namedtuple, or a list)"
            " is not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/hsplit::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`hsplit` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so container disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/hsplit::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`hsplit` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/split::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`split` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so container disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/split::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`split` returns a real array/tuple in-process for this pattern, but its"
            " specific type identity (an ndarray subclass, a namedtuple, or a list) is"
            " not preserved once it crosses to the client, so pytype disagrees."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="types/array-array-f::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `array-array-f` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/array-array-f::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `array-array-f` value has no clean wire encoding on the client, so"
            " the exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/array-array-f::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `array-array-f`"
            " value for this argument position; the client cannot encode it onto the"
            " wire and raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/array-array-f::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `array-array-f` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/bytes::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `bytes` value has no clean wire encoding on the client; depending on"
            " where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/bytes::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `bytes` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/complex::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `complex` value has no clean wire encoding on the client; depending"
            " on where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/complex::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `complex` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/complex::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `complex` value for"
            " this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/complex::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `complex` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/datetime::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `datetime` value has no clean wire encoding on the client; depending"
            " on where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/datetime::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `datetime` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/datetime::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `datetime` value"
            " for this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/datetime::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `datetime` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/decimal::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `decimal` value has no clean wire encoding on the client; depending"
            " on where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/decimal::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `decimal` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/decimal::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `decimal` value for"
            " this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/decimal::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `decimal` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/dict::*",
        dimension="flops",
        category=Category.ACCEPTED_DIVERGENCE,
        reason=(
            "In-process numpy coerces a `dict` operand to an `object` array and runs"
            " the kernel, charging for it, before anything notices the result is"
            " unusable. The server cannot deliver an `object` array to the client at"
            " all, so it refuses the operand before dispatch and charges nothing."
            " Charging for a result that provably cannot be returned is the worse"
            " behaviour of the two, so the client stays at zero here deliberately."
        ),
        issue="INTERNAL-P3-refuse-before-charging",
    ),
    Entry(
        case_id="types/*::dict-literal",
        dimension="flops",
        category=Category.ACCEPTED_DIVERGENCE,
        reason=(
            "Same deliberate choice as the `types/dict::*` flops entry, reached from"
            " the other direction: the dict-literal position wraps every value family"
            " in a `dict`, so whatever the value is, the operand is one the server"
            " refuses before dispatch while in-process numpy runs and charges for it."
        ),
        issue="INTERNAL-P3-refuse-before-charging",
    ),
    Entry(
        case_id="types/dict::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `dict` value has no clean wire encoding on the client; depending on"
            " where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/dict::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `dict` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/dict-with-handle::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `dict-with-handle` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/dict-with-handle::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `dict-with-handle` value has no clean wire encoding on the client, so"
            " the exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/dict-with-handle::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `dict-with-handle`"
            " value for this argument position; the client cannot encode it onto the"
            " wire and raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/dict-with-handle::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `dict-with-handle` value fails to encode onto the wire before the"
            " call dispatches, so the client bills 0 FLOPs where in-process the call"
            " ran and billed a nonzero amount (or vice versa when the value encodes but"
            " the in-process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/dtype-named-object::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `dtype-named-object` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/dtype-named-object::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `dtype-named-object` value has no clean wire encoding on the client,"
            " so the exception it raises there has a different base-class chain than"
            " the concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/dtype-named-object::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `dtype-named-object` value fails to encode onto the wire before the"
            " call dispatches, so the client bills 0 FLOPs where in-process the call"
            " ran and billed a nonzero amount (or vice versa when the value encodes but"
            " the in-process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/ellipsis::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `ellipsis` value has no clean wire encoding on the client; depending"
            " on where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/ellipsis::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `ellipsis` value"
            " for this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/ellipsis::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `ellipsis` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/fraction::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `fraction` value has no clean wire encoding on the client; depending"
            " on where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/fraction::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `fraction` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/fraction::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `fraction` value"
            " for this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/fraction::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `fraction` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/frozenset::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `frozenset` value has no clean wire encoding on the client; depending"
            " on where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/frozenset::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `frozenset` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/frozenset::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `frozenset` value"
            " for this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/frozenset::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `frozenset` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/generator::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `generator` value has no clean wire encoding on the client; depending"
            " on where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/generator::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `generator` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/generator::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `generator` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/handle-lookalike::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `handle-lookalike` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/handle-lookalike::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `handle-lookalike` value has no clean wire encoding on the client, so"
            " the exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/handle-lookalike::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `handle-lookalike`"
            " value for this argument position; the client cannot encode it onto the"
            " wire and raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/handle-lookalike::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `handle-lookalike` value fails to encode onto the wire before the"
            " call dispatches, so the client bills 0 FLOPs where in-process the call"
            " ran and billed a nonzero amount (or vice versa when the value encodes but"
            " the in-process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/huge-int::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `huge-int` value has no clean wire encoding on the client; depending"
            " on where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/huge-int::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `huge-int` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/huge-int::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `huge-int` value"
            " for this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/huge-int::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `huge-int` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/huge-negative-int::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `huge-negative-int` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/huge-negative-int::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `huge-negative-int` value has no clean wire encoding on the client,"
            " so the exception it raises there has a different base-class chain than"
            " the concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/huge-negative-int::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `huge-negative-int`"
            " value for this argument position; the client cannot encode it onto the"
            " wire and raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/huge-negative-int::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `huge-negative-int` value fails to encode onto the wire before the"
            " call dispatches, so the client bills 0 FLOPs where in-process the call"
            " ran and billed a nonzero amount (or vice versa when the value encodes but"
            " the in-process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/int64-max::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `int64-max` value has no clean wire encoding on the client; depending"
            " on where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/int64-max::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `int64-max` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/int64-min::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `int64-min` value has no clean wire encoding on the client; depending"
            " on where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/int64-min::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `int64-min` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/int64-min-minus-one::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `int64-min-minus-one` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/int64-min-minus-one::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `int64-min-minus-one` value has no clean wire encoding on the client,"
            " so the exception it raises there has a different base-class chain than"
            " the concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/int64-min-minus-one::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `int64-min-minus-"
            " one` value for this argument position; the client cannot encode it onto"
            " the wire and raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/int64-min-minus-one::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `int64-min-minus-one` value fails to encode onto the wire before the"
            " call dispatches, so the client bills 0 FLOPs where in-process the call"
            " ran and billed a nonzero amount (or vice versa when the value encodes but"
            " the in-process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-bool::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-bool` value has no clean wire encoding on the client; depending"
            " on where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-bool::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `np-bool` value for"
            " this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-bool::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-bool` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-complex128::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-complex128` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-complex128::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-complex128` value has no clean wire encoding on the client, so"
            " the exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-complex128::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `np-complex128`"
            " value for this argument position; the client cannot encode it onto the"
            " wire and raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-complex128::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-complex128` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-complex64::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-complex64` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-complex64::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-complex64` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-complex64::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `np-complex64`"
            " value for this argument position; the client cannot encode it onto the"
            " wire and raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-complex64::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-complex64` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-float16::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-float16` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-float16::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-float16` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-float16::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `np-float16` value"
            " for this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-float16::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-float16` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-float32::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-float32` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-float32::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-float32` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-float32::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `np-float32` value"
            " for this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-float32::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-float32` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-int64::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-int64` value has no clean wire encoding on the client; depending"
            " on where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-int64::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `np-int64` value"
            " for this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-int64::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-int64` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-ndarray-0d::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-ndarray-0d` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-ndarray-0d::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-ndarray-0d` value has no clean wire encoding on the client, so"
            " the exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-ndarray-0d::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `np-ndarray-0d`"
            " value for this argument position; the client cannot encode it onto the"
            " wire and raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-ndarray-0d::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-ndarray-0d` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-ndarray-1d::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-ndarray-1d` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-ndarray-1d::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-ndarray-1d` value has no clean wire encoding on the client, so"
            " the exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-ndarray-1d::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `np-ndarray-1d`"
            " value for this argument position; the client cannot encode it onto the"
            " wire and raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/np-ndarray-1d::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `np-ndarray-1d` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/range::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `range` value has no clean wire encoding on the client; depending on"
            " where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/range::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `range` value for"
            " this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/range::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `range` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/remote-array::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `remote-array` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/remote-array::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `remote-array` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/remote-array::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `remote-array` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/set::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `set` value has no clean wire encoding on the client; depending on"
            " where it appears, the client either raises `RemoteSerializationError`"
            " directly while encoding the argument, or reaches a different failure than"
            " the equivalent in-process rejection (numpy either accepts the value"
            " directly or raises its own `TypeError`/`ValueError`/`IndexError` while"
            " in-process handles it structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/set::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `set` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/set::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `set` value for"
            " this argument position; the client cannot encode it onto the wire and"
            " raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/set::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `set` value fails to encode onto the wire before the call dispatches,"
            " so the client bills 0 FLOPs where in-process the call ran and billed a"
            " nonzero amount (or vice versa when the value encodes but the in-process"
            " side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/slice-object::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `slice-object` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/slice-object::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "In-process, numpy accepts or structurally handles the `slice-object`"
            " value for this argument position; the client cannot encode it onto the"
            " wire and raises instead."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/slice-object::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `slice-object` value fails to encode onto the wire before the call"
            " dispatches, so the client bills 0 FLOPs where in-process the call ran and"
            " billed a nonzero amount (or vice versa when the value encodes but the in-"
            " process side rejects it first)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/uint64-max::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `uint64-max` value has no clean wire encoding on the client;"
            " depending on where it appears, the client either raises"
            " `RemoteSerializationError` directly while encoding the argument, or"
            " reaches a different failure than the equivalent in-process rejection"
            " (numpy either accepts the value directly or raises its own"
            " `TypeError`/`ValueError`/`IndexError` while in-process handles it"
            " structurally)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/uint64-max::*",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The `uint64-max` value has no clean wire encoding on the client, so the"
            " exception it raises there has a different base-class chain than the"
            " concrete in-process exception."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/bytearray::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `bytearray` value's wire encoding is inconsistent by argument"
            " position: some positions raise `ValueError` in-process against"
            " `FlopscopeServerError` on the client, others raise `TypeError` in-process"
            " against `ValueError` on the client - the client never reproduces the"
            " concrete in-process exception for this value."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/bytearray::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `bytearray` value's wire encoding is inconsistent by argument"
            " position: where it fails to encode at all the client bills 0 against a"
            " nonzero in-process cost; where it does encode (list-element) the client"
            " bills one FLOP less than in-process for the same call."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/memoryview::*",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "The `memoryview` value's wire encoding is inconsistent by argument"
            " position: some positions raise `ValueError` in-process against"
            " `FlopscopeServerError` on the client, others raise `TypeError` in-process"
            " against `ValueError` on the client - the client never reproduces the"
            " concrete in-process exception for this value."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/memoryview::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The `memoryview` value's wire encoding is inconsistent by argument"
            " position: where it fails to encode at all the client bills 0 against a"
            " nonzero in-process cost; where it does encode (list-element) the client"
            " bills one FLOP less than in-process for the same call."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/int-enum::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "An `IntEnum` member used as a scalar operand promotes the result to"
            " `float64` in-process but only to `float32` on the client (or vice versa"
            " depending on the position), so the two backends disagree on the promoted"
            " dtype for the same arithmetic."
        ),
        issue="INTERNAL-P2-family-1",
    ),
    Entry(
        case_id="types/int-enum::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "Indexing with an `IntEnum` member returns a bare `float32` scalar in-"
            " process but a `RemoteScalar` proxy on the client."
        ),
        issue="INTERNAL-P2-family-1",
    ),
    Entry(
        case_id="types/int-enum::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "An `IntEnum` member used as a scalar operand promotes to a different"
            " dtype on each backend (see the accompanying `dtype` entry), which changes"
            " the billed FLOP count for the same arithmetic."
        ),
        issue="INTERNAL-P2-family-1",
    ),
    Entry(
        case_id="types/remote-scalar::list-element",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.concatenate([V, V[0]])` bills 7 FLOPs in-process but 14 on the"
            " client for the identical call - a pure cost-model mismatch with no"
            " accompanying outcome, dtype, or value divergence."
        ),
        issue="INTERNAL-P3-family-13",
    ),
    Entry(
        case_id="types/remote-scalar::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "A remote scalar handle (`V[0]`) used as an `asarray`/second-positional"
            " argument round-trips as `float32` in-process but `float64` on the client."
        ),
        issue="INTERNAL-P2-family-1",
    ),
    Entry(
        case_id="types/remote-scalar::constructor",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.asarray(V[0])` promotes to a different dtype on each backend (see"
            " the accompanying `dtype` entry), which changes the billed FLOP count for"
            " the same call."
        ),
        issue="INTERNAL-P2-family-1",
    ),
    Entry(
        case_id="types/nested-list::index-key",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`V[[[1.0]]]` raises in-process (a triply-nested list is not a valid"
            " index) but the client's indexing accepts it and returns a value instead."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="types/nested-list::index-key",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "`V[[[1.0]]]` raises before billing anything in-process (0 FLOPs), while"
            " the client's indexing accepts the value and bills 4 FLOPs."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="idiom/scalar-float32-add",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`V[0] + V[1]` returns a bare `float32` in-process but a `RemoteScalar`"
            " proxy on the client."
        ),
        issue="INTERNAL-P2-family-1",
    ),
    Entry(
        case_id="idiom/scalar-int-array-dtype",
        dimension="value",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.array(10) * I` produces different element values in-process (int64"
            " arithmetic) than on the client, which promotes to float64."
        ),
        issue="INTERNAL-P2-family-1",
    ),
    Entry(
        case_id="idiom/scalar-int-array-dtype",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.array(10) * I` returns `int64` in-process but `float64` on the"
            " client - the two backends promote the scalar-times-int-array"
            " multiplication differently."
        ),
        issue="INTERNAL-P2-family-1",
    ),
    Entry(
        case_id="idiom/scalar-int-array-dtype",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The dtype promotion difference in `fnp.array(10) * I` (see the"
            " accompanying `dtype`/`value` entries) also changes the billed FLOP count."
        ),
        issue="INTERNAL-P2-family-1",
    ),
    Entry(
        case_id="idiom/float-index",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`V[2.5]` raises `IndexError` in-process (a float is not a valid index)"
            " but the client coerces it and returns a value."
        ),
        issue="INTERNAL-P5-family-2",
    ),
    Entry(
        case_id="idiom/float-slice-bound",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`V[: 7 / 2]` raises in-process (a float slice bound is invalid) but the"
            " client coerces it and returns a value."
        ),
        issue="INTERNAL-P5-family-2",
    ),
    Entry(
        case_id="idiom/handle-lookalike-string",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.sum('a0')` raises `TypeError` in-process (numpy rejects the string"
            " outright) but the client's content-sniffing treats `'a0'` as a lookalike"
            " for a remote handle and raises `KeyError` instead."
        ),
        issue="INTERNAL-P5-family-3",
    ),
    Entry(
        case_id="idiom/handle-lookalike-string",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `KeyError` (a `LookupError`) for `fnp.sum('a0')` has a"
            " different base-class chain than the in-process `TypeError`."
        ),
        issue="INTERNAL-P5-family-3",
    ),
    Entry(
        case_id="idiom/handle-lookalike-string",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.sum('a0')` bills 1 FLOP in-process before rejecting the string; the"
            " client's handle-lookalike sniffing rejects it before billing anything (0"
            " FLOPs)."
        ),
        issue="INTERNAL-P5-family-3",
    ),
    Entry(
        case_id="idiom/complex-scalar-mul",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "A Python `complex` scalar has no wire form: `fnp.astype(V, 'complex64') *"
            " 1j` succeeds in-process but raises on the client (the harness canary for"
            " this family)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="idiom/complex-scalar-mul",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The client bills a partial cost (6 FLOPs) before failing to encode the"
            " Python `complex` operand, against 42 FLOPs for the completed in-process"
            " call."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="idiom/slice-bound-remote",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`V[: fnp.argmax(V)]` (a remote scalar used as a slice bound) succeeds in-"
            " process but raises on the client."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="idiom/asarray-complex-list",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.asarray([1 + 2j, 3 - 1j])` (a list of Python `complex` scalars)"
            " succeeds in-process but raises on the client, which cannot encode"
            " `complex` onto the wire."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="idiom/asarray-complex-list",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The client bills 0 FLOPs failing to encode the complex list, against 8"
            " FLOPs for the completed in-process call."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="idiom/complex-element-read",
        dimension="pytype",
        category=Category.ACCEPTED_DIVERGENCE,
        reason=(
            "`fnp.full((2,), 1.0, dtype='complex128')[0]` now returns on both"
            " backends: in-process as a bare `complex128` scalar, and on the client"
            " as a 0-d array-wrapper handle. A complex scalar has no encodable"
            " bare-value wire form, so the client necessarily wraps a"
            " handle-delivered result rather than unwrapping it - the same shape"
            " as the accepted `idiom/ndim-returns-int` `pytype` divergence."
        ),
        issue="INTERNAL-P5-scalar-wrapper",
    ),
    Entry(
        case_id="idiom/tuple-axis-sum",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.sum(A, axis=(0, 1))` (a tuple axis) succeeds in-process but raises"
            " on the client - the tuple does not survive encoding as an `axis` argument"
            " (matches the `grid/*::axis-tuple` pattern across every reduction op)."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="idiom/split-returns-list",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.split(V, 2) + [V]` succeeds in-process, where `split` returns a"
            " `list` that concatenates with `[V]`; on the client `split` returns a"
            " `tuple` instead, so the same `+` raises `TypeError`."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="idiom/out-of-bounds-index",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`V[99]` raises `IndexError` in-process; the client wraps the same failure"
            " in a generic `FlopscopeServerError` instead."
        ),
        issue="INTERNAL-P5-family-7",
    ),
    Entry(
        case_id="idiom/out-of-bounds-index",
        dimension="exc_bases",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's `FlopscopeServerError` for `V[99]` has a flat `[Exception,"
            " BaseException]` base-class chain instead of `IndexError`'s `LookupError`"
            " chain."
        ),
        issue="INTERNAL-P5-family-7",
    ),
    Entry(
        case_id="idiom/string-array-read",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.asarray(['foo', 'bar']).tolist()` succeeds in-process; the client"
            " cannot decode the string array's dtype and raises instead. Family 8"
            " (undecodable dtypes) has no phase assigned in the mapping provided for"
            " this task, so this is filed as unclassified."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="idiom/fft-rfft",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.fft.rfft(V)` succeeds in-process; the client's `fnp` module has no"
            " `fft` submodule at all, so it raises `AttributeError` (matches the whole"
            " `grid/fft.*` client-surface gap)."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="idiom/fft-rfft",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The client's missing `fft` submodule means the call never dispatches and"
            " bills 0 FLOPs, against 45 for the completed in-process call."
        ),
        issue="INTERNAL-P4-family-9",
    ),
    Entry(
        case_id="idiom/huge-int-operand",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`V * 2**70` (an int too large for any wire integer format) succeeds in-"
            " process via Python's arbitrary-precision arithmetic; the client cannot"
            " encode it and raises instead."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="idiom/huge-int-operand",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "The client bills 0 FLOPs failing to encode the huge int operand, against"
            " 6 for the completed in-process call."
        ),
        issue="INTERNAL-P5-family-10",
    ),
    Entry(
        case_id="idiom/ndim-returns-int",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.ndim(A)` returns a bare `int` in-process (no dtype); the client"
            " wraps it in a `RemoteScalar` proxy that reports `int64`."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="idiom/ndim-returns-int",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.ndim(A)` returns a bare `int` in-process (no shape); the client's"
            " `RemoteScalar` proxy reports a `[]` shape instead."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="idiom/ndim-returns-int",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.ndim(A)` returns a bare `int` (container `scalar`) in-process; the"
            " client wraps it in an array-shaped proxy instead."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="idiom/ndim-returns-int",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.ndim(A)` returns a bare `int` in-process but a `RemoteScalar` proxy"
            " on the client (the harness's own precedent case for this family)."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/random.[!GR]*::*",
        dimension="pytype",
        category=Category.KNOWN_BUG,
        reason=(
            "Several legacy `random.*` sampling/selection functions (e.g."
            " `random.choice` picking a single element, `random.poisson` called"
            " with no arguments) return a bare Python/NumPy scalar in-process for"
            " these patterns; the client wraps the same result in a `RemoteScalar`"
            " proxy instead of unwrapping it (the same mechanism as the"
            " `ndim`-returns-int case). Uses `[!GR]` to exclude"
            " `random.Generator.*`/`random.RandomState.*`, which are covered by"
            " their own client-surface entries."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/random.[!GR]*::*",
        dimension="container",
        category=Category.KNOWN_BUG,
        reason=(
            "The same legacy `random.*` scalar-wrapping mechanism as the `pytype`"
            " entry above: in-process these calls return a bare scalar (container"
            " `scalar`), while the client wraps the result in an array-shaped"
            " proxy."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/random.[!GR]*::*",
        dimension="dtype",
        category=Category.KNOWN_BUG,
        reason=(
            "The same legacy `random.*` scalar-wrapping mechanism as the `pytype`"
            " entry above: in-process these calls return a bare scalar with no"
            " `dtype`, while the client's wrapping proxy reports one."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/random.[!GR]*::*",
        dimension="shape",
        category=Category.KNOWN_BUG,
        reason=(
            "The same legacy `random.*` scalar-wrapping mechanism as the `pytype`"
            " entry above: in-process these calls return a bare scalar with no"
            " `shape`, while the client's wrapping proxy reports `[]`."
        ),
        issue="INTERNAL-P5-family-12",
    ),
    Entry(
        case_id="grid/result_type::*",
        dimension="value",
        category=Category.KNOWN_BUG,
        reason=(
            "`result_type` returns a NumPy dtype-class object in-process,"
            " fingerprinted via this harness's generic `repr`-based fallback (e.g."
            " `dtype('float32')`); the client instead returns a plain descriptive"
            " string (`'float32'`) for the same result. This is a dtype-object"
            " undecodability issue (family 8), which has no phase assigned in the"
            " mapping provided for this task, so it is filed as unclassified."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/min_scalar_type::*",
        dimension="value",
        category=Category.KNOWN_BUG,
        reason=(
            "`min_scalar_type` returns a NumPy dtype-class object in-process,"
            " fingerprinted via this harness's generic `repr`-based fallback (e.g."
            " `dtype('float32')`); the client instead returns a plain descriptive"
            " string (`'float32'`) for the same result. This is a dtype-object"
            " undecodability issue (family 8), which has no phase assigned in the"
            " mapping provided for this task, so it is filed as unclassified."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="grid/*::axis-tuple",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "Passing a tuple `axis` (e.g. `axis=(0, 1)`) succeeds in-process for every"
            " reduction op the grid drives this way, but raises on the client - the"
            " tuple does not survive encoding as an `axis` argument (the same defect as"
            " the `idiom/tuple-axis-sum` canary)."
        ),
        issue="INTERNAL-P2-family-6",
    ),
    Entry(
        case_id="grid/array::*",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.array(X)` bills a nonzero amount in-process (a real copy) but the"
            " client always bills 0 for the same call - a pure cost-model mismatch"
            " between the two backends' `array` implementations."
        ),
        issue="INTERNAL-P3-family-13",
    ),
    Entry(
        case_id="grid/array::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.array(X, axis=...)` raises in-process (`array` does not accept an"
            " `axis` keyword) but the client silently accepts it and returns a value -"
            " the client is more permissive than in-process argument validation."
            " Unclassified: this is a validation gap, not a case any given family"
            " cleanly covers."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="types/memoryview::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "The `memoryview` value's wire encoding is inconsistent by argument"
            " position: in-process it is always accepted, while the client rejects it"
            " outright in some positions (`index-key`, `list-element`) and accepts it"
            " in others."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/bytearray::*",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "The `bytearray` value's wire encoding is inconsistent by argument"
            " position: in-process it is always accepted, while the client rejects it"
            " outright in some positions (`index-key`, `list-element`) and accepts it"
            " in others."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/remote-scalar::index-key",
        dimension="outcome",
        category=Category.KNOWN_BUG,
        reason=(
            "`V[V[0]]` (indexing with a remote scalar handle) raises in-process (a"
            " non-Python-int index) but the client's indexing accepts it and returns a"
            " value instead."
        ),
        issue="INTERNAL-P5-unclassified",
    ),
    Entry(
        case_id="types/remote-scalar::dict-literal",
        dimension="exc_type",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.multiply(V, {'k': V[0]})` (a remote scalar nested in a dict literal)"
            " raises `TypeError` in-process; the client's argument encoder does not"
            " recurse into `dict` values the way it does for `list`/`tuple`, so it"
            " raises `RemoteSerializationError` while trying to encode the dict itself."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/remote-scalar::dict-literal",
        dimension="flops",
        category=Category.KNOWN_BUG,
        reason=(
            "`fnp.multiply(V, {'k': V[0]})` bills 6 FLOPs in-process before rejecting"
            " the dict argument; the client's encoder rejects the dict before billing"
            " anything (0 FLOPs)."
        ),
        issue="INTERNAL-P2-family-4",
    ),
    Entry(
        case_id="types/bytes::constructor",
        dimension="outcome",
        category=Category.ACCEPTED_DIVERGENCE,
        reason=(
            "`fnp.asarray(b'\\x01\\x02')` produces an in-process array with a"
            " byte-string dtype (`|S2`). The client has no decodable"
            " representation for string/byte-string dtypes, so the server refuses"
            " to mint a handle for it rather than returning one the client could"
            " not read back - a deliberate choice, consistent with the client's"
            " own `array()` already rejecting `bytes`/`str` inputs outright."
        ),
        issue="INTERNAL-P5-dtype-representation",
    ),
    Entry(
        case_id="types/dict::constructor",
        dimension="outcome",
        category=Category.ACCEPTED_DIVERGENCE,
        reason=(
            "`fnp.asarray({'k': 1})` produces an in-process array with `object`"
            " dtype. The client has no decodable representation for `object`"
            " dtype, so the server refuses to mint a handle for it rather than"
            " returning one the client could not read back - a deliberate choice."
        ),
        issue="INTERNAL-P5-dtype-representation",
    ),
    Entry(
        case_id="types/dict::second-positional",
        dimension="outcome",
        category=Category.ACCEPTED_DIVERGENCE,
        reason=(
            "`fnp.where(M, {'k': 1}, 0.0)` produces an in-process array with"
            " `object` dtype (from mixing a `dict` operand into the result). The"
            " client has no decodable representation for `object` dtype, so the"
            " server refuses to mint a handle for it rather than returning one the"
            " client could not read back - a deliberate choice."
        ),
        issue="INTERNAL-P5-dtype-representation",
    ),
    Entry(
        case_id="grid/where::scalar-operand",
        dimension="outcome",
        category=Category.ACCEPTED_DIVERGENCE,
        reason=(
            "`fnp.where(V, 2.0)` (the two-argument, condition-only form) produces"
            " an in-process array with `object` dtype. The client has no decodable"
            " representation for `object` dtype, so the server refuses to mint a"
            " handle for it rather than returning one the client could not read"
            " back - a deliberate choice."
        ),
        issue="INTERNAL-P5-dtype-representation",
    ),
)


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
