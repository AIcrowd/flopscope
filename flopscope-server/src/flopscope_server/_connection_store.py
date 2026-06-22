"""ConnectionStore — connection-lifetime handle storage (arrays + RNG generators).

Owned by :class:`flopscope_server._server.FlopscopeServer` for the lifetime of
the server process (= one worker = the warm participant subprocess's single ZMQ
connection). Injected into each per-MLP :class:`flopscope_server._session.Session`
so that handles minted at module-import / ``setup()`` time survive across MLPs.

Memory is bounded by the flopscope-client's GC-driven frees (weakref.finalize ->
batched ``free``) plus the :data:`flopscope_server._array_store.MAX_ARRAY_COUNT`
backstop — NOT by per-session clearing. ``budget_open``/``budget_close`` reset
only the FLOP counter (the ``Session``'s ``BudgetContext``), never this store.
"""

from __future__ import annotations

from flopscope_server._array_store import ArrayStore
from flopscope_server._generator_store import GeneratorStore


class ConnectionStore:
    """Bundles the array + generator stores that live for the server connection."""

    def __init__(self) -> None:
        self.arrays = ArrayStore()
        self.generators = GeneratorStore()
