"""Session — ties together ConnectionStore, BudgetContext, and CommsTracker for a single participant session."""

from __future__ import annotations

from typing import Any

import numpy as np

import flopscope as flops
from flopscope_server._comms_tracker import CommsTracker
from flopscope_server._connection_store import ConnectionStore


class Session:
    """A single participant session combining ConnectionStore, BudgetContext, and CommsTracker.

    Parameters
    ----------
    flop_budget : int
        Maximum number of FLOPs allowed for this session.
    conn_store : ConnectionStore | None
        Connection-lifetime store (arrays + generators). When provided the
        server shares ONE store across all per-MLP sessions so that
        module-level / ``setup()``-time handles survive across MLPs
        (issue #107). When ``None`` (standalone / unit-test use) a private
        ``ConnectionStore`` is created so a bare ``Session`` still works in
        isolation.
    """

    def __init__(
        self, flop_budget: int, conn_store: ConnectionStore | None = None
    ) -> None:
        # The array + generator stores live for the CONNECTION, not the budget
        # session. The server injects one shared ConnectionStore across all MLPs
        # so module-level / setup()-time handles survive (issue #107). When none
        # is injected (standalone / unit-test use) a private one is created so a
        # bare Session still works in isolation.
        self._conn = conn_store if conn_store is not None else ConnectionStore()
        self._comms_tracker = CommsTracker()
        self._budget_ctx = flops.BudgetContext(
            flop_budget=flop_budget,
            quiet=True,
        )
        self._budget_ctx.__enter__()
        self._is_open = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """True if this session is still active (not yet closed)."""
        return self._is_open

    @property
    def budget_remaining(self) -> int:
        """FLOPs remaining in the current budget."""
        return self._budget_ctx.flops_remaining

    @property
    def budget_context(self) -> flops.BudgetContext:
        """The active BudgetContext for this session.

        Raises
        ------
        RuntimeError
            If the session is already closed.
        """
        if not self._is_open:
            raise RuntimeError(
                "Session is closed; BudgetContext is no longer available."
            )
        return self._budget_ctx

    @property
    def comms_tracker(self) -> CommsTracker:
        """The CommsTracker for this session."""
        return self._comms_tracker

    # ------------------------------------------------------------------
    # Array operations (delegate to the connection store)
    # ------------------------------------------------------------------

    def store_array(self, arr: Any) -> str:
        """Store *arr* and return its handle ID. Delegates to the connection store."""
        return self._conn.arrays.put(arr)

    def get_array(self, handle: str) -> Any:
        """Return the array for *handle*. Delegates to the connection store."""
        return self._conn.arrays.get(handle)

    def array_metadata(self, handle: str) -> dict:
        """Return metadata dict for *handle*. Delegates to the connection store."""
        return self._conn.arrays.metadata(handle)

    def free_arrays(self, handles: list) -> None:
        """Remove arrays by handle; silently ignore unknown handles."""
        self._conn.arrays.free(handles)

    # ------------------------------------------------------------------
    # Generator operations (server-side RNG handles)
    # ------------------------------------------------------------------

    def store_generator(self, gen: Any) -> str:
        """Store an RNG ``Generator`` and return its handle ID (``g0``, ``g1`` …)."""
        return self._conn.generators.put(gen)

    def get_generator(self, handle: str) -> Any:
        """Return the ``Generator`` for *handle*.

        Raises
        ------
        KeyError
            If *handle* is not a known generator handle.
        """
        return self._conn.generators.get(handle)

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def budget_status(self) -> dict:
        """Return the current FLOP budget status.

        Returns
        -------
        dict with keys:
            flop_budget: total budget
            flops_used: FLOPs consumed so far
            flops_remaining: budget minus used
        """
        return {
            "flop_budget": self._budget_ctx.flop_budget,
            "flops_used": self._budget_ctx.flops_used,
            "flops_remaining": self._budget_ctx.flops_remaining,
        }

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def close(self) -> dict:
        """Close the session, exiting the BudgetContext.

        Returns
        -------
        dict with keys:
            budget_summary: str — human-readable FLOP budget summary,
                including a namespace section when labeled ops were recorded
            budget_breakdown: dict — machine-readable summary data with
                ``by_namespace`` buckets for direct ingestion
            comms_summary: dict — CommsTracker summary

        Raises
        ------
        RuntimeError
            If the session is already closed.
        """
        if not self._is_open:
            raise RuntimeError("Session is already closed.")

        self._budget_ctx.__exit__(None, None, None)
        budget_breakdown = self._budget_ctx.summary_dict(by_namespace=True)
        show_namespaces = any(
            namespace is not None
            for namespace in budget_breakdown.get("by_namespace", {})
        )
        budget_summary = self._budget_ctx.summary(by_namespace=show_namespaces)
        comms_summary = self._comms_tracker.summary()
        # The array + generator stores are owned by the connection, not the
        # session, so they are intentionally NOT cleared here — module-level
        # handles must survive across MLPs (issue #107). Memory is bounded by
        # client GC-frees + ArrayStore.MAX_ARRAY_COUNT.
        self._is_open = False

        return {
            "budget_summary": budget_summary,
            "budget_breakdown": budget_breakdown,
            "comms_summary": comms_summary,
        }
