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
        # Sync inner.steps[i].flop_cost to accumulation.per_step[i].total so
        # the check_consistency invariant (optimized_cost == sum(step.flop_cost)
        # == accumulation.total) holds even when the oracle threads symmetry
        # through per-step costs (Task 17b).
        if (
            accumulation is not None
            and accumulation.per_step
            and hasattr(inner, "steps")
        ):
            steps = inner.steps
            for step, acc_step in zip(steps, accumulation.per_step):
                try:
                    step.flop_cost = acc_step.total
                except (AttributeError, TypeError):
                    pass  # StepInfo may be frozen in some configurations; skip
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
        # Attach regime per step from accumulation per_step (if path-aware).
        if self.accumulation is not None:
            per_step = self.accumulation.per_step or (self.accumulation,)
            for step, acc_step in zip(self._inner.steps, per_step):
                if acc_step.per_component:
                    object.__setattr__(step, "_regime", acc_step.per_component[0].regime_id)
                else:
                    object.__setattr__(step, "_regime", "-")
        return fmt()

    def check_consistency(self) -> bool:
        """Verify the three cost surfaces agree:
          info.optimized_cost == sum(s.flop_cost for s in info.steps)
                              == info.accumulation.total

        Returns True on success; raises AssertionError with a diagnostic
        message otherwise. Use this in tests or after manually mutating
        the wrapper to confirm invariants hold.
        """
        sum_steps = sum(
            getattr(s, "flop_cost", 0)
            for s in getattr(self._inner, "steps", [])
        )
        acc_total = (
            self.accumulation.total if self.accumulation is not None else None
        )
        opt_cost = self.optimized_cost
        if acc_total is not None and opt_cost != acc_total:
            raise AssertionError(
                f"check_consistency: optimized_cost ({opt_cost}) != "
                f"accumulation.total ({acc_total})"
            )
        if (
            opt_cost != sum_steps
            and self.accumulation is not None
            and self.accumulation.per_step
        ):
            raise AssertionError(
                f"check_consistency: optimized_cost ({opt_cost}) != "
                f"sum(steps.flop_cost) ({sum_steps})"
            )
        return True

    def __repr__(self) -> str:
        return (
            f"FlopscopePathInfo(optimized_cost={self.optimized_cost}, "
            f"path={getattr(self._inner, 'path', [])}, "
            f"accumulation={'<attached>' if self.accumulation else None})"
        )
