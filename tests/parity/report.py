"""Render a parity run into something a human can act on.

Coverage counts are printed on every run: silent truncation is what makes a
green suite untrustworthy.
"""

from __future__ import annotations

from tests.parity.allowlist import AllowlistResult
from tests.parity.runner import RunResult


def render(result: RunResult, allow: AllowlistResult, coverage: dict) -> str:
    lines: list[str] = ["=== flopscope client/in-process parity ==="]

    if result.infrastructure_failure:
        lines.append(f"INFRASTRUCTURE FAILURE: {result.infrastructure_failure}")
        return "\n".join(lines)

    if coverage:
        lines.append("")
        lines.append("Coverage:")
        for key, value in sorted(coverage.items()):
            lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append(f"Allowlist: {dict(sorted(allow.counts.items()))}")

    if allow.match_counts:
        lines.append("")
        lines.append("Allowlist entries (match count - a glob can hide a lot):")
        by_count = sorted(
            allow.match_counts.items(),
            key=lambda pair: (-pair[1], pair[0].case_id, pair[0].dimension),
        )
        for entry, count in by_count:
            lines.append(
                f"  [{count:>5}] {entry.case_id} [{entry.dimension}] "
                f"({entry.category.value}) {entry.reason}"
            )

    if allow.unexplained:
        lines.append("")
        lines.append(f"UNEXPLAINED DIVERGENCES ({len(allow.unexplained)}):")
        for divergence in allow.unexplained:
            lines.append(
                f"  {divergence.case_id} [{divergence.dimension}] "
                f"inproc={divergence.inproc!r} client={divergence.client!r}"
            )

    if allow.stale:
        lines.append("")
        lines.append(f"STALE ALLOWLIST ENTRIES ({len(allow.stale)}) - delete these:")
        for entry in allow.stale:
            lines.append(f"  {entry.case_id} [{entry.dimension}] {entry.reason}")

    if result.flaky:
        lines.append("")
        lines.append(f"FLAKY (quarantined, not parity failures) ({len(result.flaky)}):")
        for case_id in result.flaky:
            lines.append(f"  {case_id}")

    if not allow.unexplained and not allow.stale:
        lines.append("")
        lines.append("No unexplained divergences and no stale entries.")
    return "\n".join(lines)
