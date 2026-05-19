"""FlopscopePathInfo: wraps opt_einsum's PathInfo and adds an accumulation field.

opt_einsum's PathInfo may be a frozen dataclass; we hold a reference to it
and forward field accesses through __getattr__ rather than subclassing,
which is the safest cross-version approach.
"""

from __future__ import annotations

from typing import Any

from ._cost import AccumulationCost


class FlopscopePathInfo:
    """Mutable wrapper around opt_einsum's PathInfo with an accumulation field.

    Field forwarding: any attribute not on the wrapper itself is looked up on
    the wrapped inner PathInfo. The `optimized_cost` property prefers the
    accumulation total when attached.
    """

    __slots__ = ("_inner", "accumulation")

    def __init__(
        self,
        *,
        inner: Any,
        accumulation: AccumulationCost | None,
    ) -> None:
        self._inner = inner
        self.accumulation = accumulation

    @classmethod
    def from_inner(
        cls,
        *,
        inner: Any,
        accumulation: AccumulationCost | None,
    ) -> FlopscopePathInfo:
        return cls(inner=inner, accumulation=accumulation)

    @property
    def optimized_cost(self) -> int:
        """The charged FLOP cost. Returns accumulation.total when attached;
        otherwise falls back to the inner PathInfo's optimized_cost."""
        if self.accumulation is not None:
            return self.accumulation.total
        return getattr(self._inner, "optimized_cost", 0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __str__(self) -> str:
        """Render the underlying PathInfo's table.

        Pre-§6.4 we monkey-patched the inner's naive_cost/optimized_cost
        before rendering, because the wrapper's optimized_cost
        (= accumulation.total) differed from inner.optimized_cost (= sum of
        per-step flop_cost). After the reconciliation refactor (commit
        69d88ec8f) those numbers are equal by construction, so the inner's
        own renderer is correct without mutation.
        """
        fmt = getattr(self._inner, "format_table", None)
        if fmt is None:
            return self.__repr__()
        return fmt()

    def __repr__(self) -> str:
        return (
            f"FlopscopePathInfo(optimized_cost={self.optimized_cost}, "
            f"path={getattr(self._inner, 'path', [])}, "
            f"accumulation={'<attached>' if self.accumulation else None})"
        )
