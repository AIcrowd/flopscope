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
        if accumulation is not None and hasattr(inner, "steps"):
            steps = inner.steps
            if accumulation.per_step:
                # Multi-step path: per_step is populated by _walk_path_and_aggregate.
                per_step_pre = (
                    accumulation.pre_reductions_per_step
                    if accumulation.pre_reductions_per_step
                    else ((),) * len(accumulation.per_step)
                )
                for step, acc_step, pre_red in zip(
                    steps, accumulation.per_step, per_step_pre, strict=False
                ):
                    try:
                        step.flop_cost = acc_step.total
                    except (AttributeError, TypeError):
                        pass  # StepInfo may be frozen in some configurations; skip
                    try:
                        step.pre_reductions = pre_red
                    except (AttributeError, TypeError):
                        pass
                    # Sprint 4: thread cost_source from AccumulationCost to
                    # StepInfo so tests / renderers can inspect the winning
                    # category (per-input / joint-burnside / output-burnside).
                    try:
                        step.cost_source = acc_step.cost_source
                    except (AttributeError, TypeError):
                        pass
            elif len(steps) == 1:
                # Bug B fix: single-step (2-op) einsum — aggregate_einsum returns a
                # flat AccumulationCost with no per_step entries.  The single step's
                # flop_cost must still be updated so that the renderer header and
                # per-step "flops" column agree with info.optimized_cost.
                try:
                    steps[0].flop_cost = accumulation.total
                except (AttributeError, TypeError):
                    pass
                # Sprint 3: wire pre_reductions for single-step einsums.
                if accumulation.pre_reductions_per_step:
                    try:
                        steps[0].pre_reductions = accumulation.pre_reductions_per_step[
                            0
                        ]
                    except (AttributeError, TypeError, IndexError):
                        pass
                # Sprint 4: thread cost_source for single-step einsums.
                # Single-step (2-op) path goes through aggregate_einsum
                # directly, which always uses per-input (Cat A). Default to
                # "per-input" when the aggregate cost_source is None and the
                # cost is not a fallback.
                try:
                    src = accumulation.cost_source
                    if src is None and not accumulation.fallback_used:
                        src = "per-input"
                    steps[0].cost_source = src
                except (AttributeError, TypeError):
                    pass

            # After syncing step flop_cost values, recompute inner.optimized_cost,
            # inner.naive_cost (dense baseline), and inner.speedup so the renderer
            # header agrees with the per-step column.  inner.naive_cost is already
            # the sum of dense_flop_cost (Bug A fix in build_path_info); we must
            # not change it here — only optimized_cost and speedup need updating.
            new_opt_cost = sum(getattr(s, "flop_cost", 0) for s in steps)
            try:
                inner.optimized_cost = new_opt_cost
            except (AttributeError, TypeError):
                pass
            try:
                naive = getattr(inner, "naive_cost", 0)
                inner.speedup = (naive / new_opt_cost) if new_opt_cost > 0 else 1.0
            except (AttributeError, TypeError):
                pass

            # Recompute per-step symmetry_savings now that flop_cost is correct.
            for step in steps:
                dense = getattr(step, "dense_flop_cost", 0)
                sym_cost = getattr(step, "flop_cost", dense)
                try:
                    step.symmetry_savings = (
                        max(0.0, 1.0 - sym_cost / dense) if dense > 0 else 0.0
                    )
                except (AttributeError, TypeError):
                    pass

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
        # Attach regime, acc_step, and pre_reductions per step from accumulation
        # per_step (if path-aware).
        if self.accumulation is not None:
            per_step = self.accumulation.per_step or (self.accumulation,)
            per_step_pre = (
                self.accumulation.pre_reductions_per_step
                if self.accumulation.pre_reductions_per_step
                else ((),) * len(per_step)
            )
            for step, acc_step, pre_red in zip(
                self._inner.steps, per_step, per_step_pre, strict=False
            ):
                if acc_step.per_component:
                    object.__setattr__(
                        step, "_regime", acc_step.per_component[0].regime_id
                    )
                else:
                    object.__setattr__(step, "_regime", "-")
                object.__setattr__(step, "_acc_step", acc_step)
                object.__setattr__(step, "pre_reductions", pre_red)
                # Sprint 4: thread cost_source from AccumulationCost to StepInfo
                # so tests / renderers can inspect the winning category.
                object.__setattr__(step, "cost_source", acc_step.cost_source)
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
            getattr(s, "flop_cost", 0) for s in getattr(self._inner, "steps", [])
        )
        acc_total = self.accumulation.total if self.accumulation is not None else None
        opt_cost = self.optimized_cost
        if acc_total is not None and opt_cost != acc_total:
            raise AssertionError(
                f"check_consistency: optimized_cost ({opt_cost}) != "
                f"accumulation.total ({acc_total})"
            )
        if opt_cost != sum_steps and self.accumulation is not None:
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
