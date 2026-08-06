"""Public names that exist in flopscope but are server-side / in-process only.

These are NOT available in the flopscope client (the remote/grader env): cost
introspection, accounting/cache helpers, and result/AST type classes whose
client equivalent is ``RemoteArray``. Synced to the client by
``scripts/sync_client.py``; consulted by the client ``__getattr__`` paths and
the API-parity guard.
"""

from __future__ import annotations

SERVER_ONLY: frozenset[str] = frozenset(
    {
        # result / AST types (client uses RemoteArray + .symmetry metadata)
        "FlopscopeArray",
        "SymmetricTensor",
        "PathInfo",
        "StepInfo",
        # accounting / cache tooling
        "namespace",
        "clear_einsum_cache",
        "einsum_cache_info",
        # flops.* cost-introspection helpers (Task C4)
        # All weighted cost helpers from flopscope.accounting re-exported under
        # the flops.* public prefix.  einsum_cost/svd_cost/pointwise_cost/
        # reduction_cost are already implemented as client-side proxies in
        # flopscope-client/src/flopscope/flops.py and are intentionally absent.
        "flops.bartlett_cost",
        "flops.blackman_cost",
        "flops.cholesky_cost",
        "flops.cond_cost",
        "flops.det_cost",
        "flops.eig_cost",
        "flops.eigh_cost",
        "flops.eigvals_cost",
        "flops.eigvalsh_cost",
        "flops.fft_cost",
        "flops.fftn_cost",
        "flops.hamming_cost",
        "flops.hanning_cost",
        "flops.hfft_cost",
        "flops.inv_cost",
        "flops.kaiser_cost",
        "flops.lstsq_cost",
        "flops.matrix_norm_cost",
        "flops.matrix_power_cost",
        "flops.matrix_rank_cost",
        "flops.multi_dot_cost",
        "flops.norm_cost",
        "flops.pinv_cost",
        "flops.poly_cost",
        "flops.polyadd_cost",
        "flops.polyder_cost",
        "flops.polydiv_cost",
        "flops.polyfit_cost",
        "flops.polyint_cost",
        "flops.polymul_cost",
        "flops.polysub_cost",
        "flops.polyval_cost",
        "flops.qr_cost",
        "flops.rfft_cost",
        "flops.rfftn_cost",
        "flops.roots_cost",
        "flops.slogdet_cost",
        "flops.solve_cost",
        "flops.svdvals_cost",
        "flops.tensorinv_cost",
        "flops.tensorsolve_cost",
        "flops.trace_cost",
        "flops.unwrap_cost",
        "flops.vector_norm_cost",
    }
)
