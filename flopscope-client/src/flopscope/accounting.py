"""flopscope.accounting — FLOP cost-estimation helpers (client surface).

Mirrors the native ``flopscope.accounting`` public namespace; cost helpers are
re-exported from the client's ``flops`` module.
"""

from flopscope.flops import (  # noqa: F401
    einsum_cost,
    pointwise_cost,
    reduction_cost,
    svd_cost,
)

__all__ = ["pointwise_cost", "reduction_cost", "einsum_cost", "svd_cost"]
