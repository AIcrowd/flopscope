"""ArrayStore — in-process dict-based mapping from handle IDs to numpy arrays."""

from __future__ import annotations

import os
from typing import Any

#: Maximum number of arrays allowed in a single store (configurable via env var).
MAX_ARRAY_COUNT = int(os.environ.get("FLOPSCOPE_MAX_ARRAY_COUNT", "100000"))

# Handle ids are allocated from a PROCESS-GLOBAL monotonic counter (not reset
# per ArrayStore/session). This guarantees ids are never reused across sessions,
# so a stale free for a handle from a closed session can never alias a live
# handle in a new session (ArrayStore.free silently ignores unknown ids). The
# client frees handles on GC, and GC of a prior-session proxy can fire during a
# later session — global ids make that safe. The counter is unbounded, but a
# server process serves one submission, so it stays far below any int limit.
_next_handle_id = 0


def _alloc_handle() -> str:
    """Allocate the next unique handle id from the process-global counter."""
    global _next_handle_id
    # Single-threaded server (one ZMQ REP loop), so no lock is needed.
    handle = f"a{_next_handle_id}"
    _next_handle_id += 1
    return handle


def _reset_handle_counter() -> None:
    """Reset the global handle counter. TEST-ONLY (production never resets)."""
    global _next_handle_id
    _next_handle_id = 0


class ArrayStore:
    """Simple store that maps string handle IDs to numpy arrays.

    Handle IDs are allocated from a process-global monotonic counter (see
    :func:`_alloc_handle`), so IDs remain unique across all ArrayStore instances
    and sessions for the lifetime of the server process.
    """

    def __init__(self) -> None:
        self._arrays: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def put(self, arr: Any) -> str:
        """Store *arr* and return its handle ID.

        Raises
        ------
        MemoryError
            If the store already contains :data:`MAX_ARRAY_COUNT` arrays.
        """
        if len(self._arrays) >= MAX_ARRAY_COUNT:
            raise MemoryError(f"array store limit reached: {MAX_ARRAY_COUNT} arrays")
        handle = _alloc_handle()
        self._arrays[handle] = arr
        return handle

    def get(self, handle: str) -> Any:
        """Return the array for *handle*.

        Raises
        ------
        KeyError
            If *handle* is not in the store.
        """
        if handle not in self._arrays:
            raise KeyError(f"Array handle {handle!r} not found in store")
        return self._arrays[handle]

    def metadata(self, handle: str) -> dict:
        """Return metadata dict for *handle*.

        Returns
        -------
        dict
            ``{"id": handle, "shape": list[int], "dtype": str}``

        Raises
        ------
        KeyError
            If *handle* is not in the store.
        """
        arr = self.get(handle)  # propagates KeyError with helpful message
        meta = {
            "id": handle,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        symmetry = getattr(arr, "symmetry", None)
        if symmetry is not None:
            meta["symmetry"] = symmetry.to_payload()
        return meta

    def free(self, handles: list[str]) -> None:
        """Remove arrays by handle; silently ignore unknown handles."""
        for handle in handles:
            self._arrays.pop(handle, None)

    def clear(self) -> None:
        """Remove all arrays from the store."""
        self._arrays.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of arrays currently in the store."""
        return len(self._arrays)
