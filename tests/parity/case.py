"""Serialisable description of one differential parity case.

Cases are DATA, not closures: they cross a process boundary to the workers and
are referenced by id from allowlist entries, so they must be serialisable and
greppable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    """One expression to evaluate identically on both backends.

    Parameters
    ----------
    id:
        Stable, structured ``<source>/<name>`` identifier. Allowlist entries
        reference it; renaming a case orphans its entry (which fails the build
        under strict allowlisting, by design).
    source:
        Python expression evaluated with ``fnp`` and the standard fixtures bound.
    setup:
        Optional statements executed before ``source``, in the same namespace.
    tags:
        Free-form markers, e.g. ``family:6``, ``tier:fast``, ``requires:numpy``.
    """

    id: str
    source: str
    setup: str = ""
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        parts = self.id.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"case id {self.id!r} must be '<source>/<name>'")

    def family(self) -> str | None:
        """Return the defect-family label this case covers, if tagged."""
        for tag in self.tags:
            if tag.startswith("family:"):
                return tag.split(":", 1)[1]
        return None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "setup": self.setup,
            "tags": sorted(self.tags),
        }

    @classmethod
    def from_json(cls, data: dict) -> Case:
        return cls(
            id=data["id"],
            source=data["source"],
            setup=data.get("setup", ""),
            tags=frozenset(data.get("tags", ())),
        )
