"""GeneratorStore — in-process mapping from handle IDs to RNG Generator objects.

Mirrors :mod:`flopscope_server._array_store`: handle ids come from a
PROCESS-GLOBAL monotonic counter so they are never reused across budget
sessions. A generator created at module-import / ``setup()`` time (held by the
warm participant subprocess across MLPs) therefore keeps a stable, unambiguous
handle for the connection lifetime — it can never silently alias a generator
minted in a later session. No size cap (generators are few; matches the prior
per-Session behaviour which had none).
"""

from __future__ import annotations

from typing import Any

# Process-global monotonic counter (mirrors _array_store._next_handle_id).
# Never reset in production; reset only in tests for deterministic handle names.
_next_gen_id = 0


def _alloc_gen() -> str:
    """Allocate the next unique generator handle id from the process-global counter."""
    global _next_gen_id
    # Single-threaded server (one ZMQ REP loop), so no lock is needed.
    handle = f"g{_next_gen_id}"
    _next_gen_id += 1
    return handle


def _reset_gen_counter() -> None:
    """Reset the global generator counter. TEST-ONLY (production never resets)."""
    global _next_gen_id
    _next_gen_id = 0


class GeneratorStore:
    """Maps string handle IDs (``g0``, ``g1`` …) to RNG ``Generator`` objects.

    Handle IDs come from a process-global monotonic counter (see
    :func:`_alloc_gen`), so IDs are unique across all GeneratorStore instances
    and budget sessions for the lifetime of the server process.
    """

    def __init__(self) -> None:
        self._gens: dict[str, Any] = {}

    def put(self, gen: Any) -> str:
        """Store *gen* and return its handle ID."""
        handle = _alloc_gen()
        self._gens[handle] = gen
        return handle

    def get(self, handle: str) -> Any:
        """Return the ``Generator`` for *handle*.

        Raises
        ------
        KeyError
            If *handle* is not a known generator handle.
        """
        if handle not in self._gens:
            raise KeyError(f"Generator handle {handle!r} not found in store")
        return self._gens[handle]

    def free(self, handles: list[str]) -> None:
        """Remove generators by handle; silently ignore unknown handles."""
        for handle in handles:
            self._gens.pop(handle, None)

    def clear(self) -> None:
        """Remove all generators from the store."""
        self._gens.clear()

    @property
    def count(self) -> int:
        """Number of generators currently in the store."""
        return len(self._gens)
