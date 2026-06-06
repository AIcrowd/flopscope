"""Public FLOP cost estimation utilities for the lightweight client.

Local helpers mirror the core ``flopscope.flops`` public API shape. They apply
client-side FLOP weights when configured via ``flopscope._weights``; more complex
helpers continue to proxy to the server.
"""

from __future__ import annotations

from collections.abc import Sequence

from flopscope._dispatch import timed_dispatch
from flopscope._math_compat import prod as _prod
from flopscope._weights import get_weight

# ---------------------------------------------------------------------------
# Local cost functions (no server needed)
# ---------------------------------------------------------------------------


def _weight_cost(op_name: str, analytical_cost: int) -> int:
    """Convert an analytical FLOP count into a weighted public estimate."""
    return int(analytical_cost * get_weight(op_name))


def pointwise_cost(op_name: str, *, shape: tuple[int, ...]) -> int:
    """Return the weighted client-side cost of a pointwise operation.

    Parameters
    ----------
    op_name:
        Operation name used for weight lookup.
    shape:
        Shape of the array the operation is applied to.

    Returns
    -------
    int
        Weighted public cost estimate for a single pointwise operation.
    """
    if not isinstance(op_name, str):
        raise TypeError("pointwise_cost() requires op_name as the first argument")
    return _weight_cost(op_name, max(_prod(shape), 1))


def reduction_cost(
    op_name: str,
    *,
    input_shape: tuple[int, ...],
    axis: int | None = None,
) -> int:
    """Return the weighted client-side cost of a reduction operation.

    Parameters
    ----------
    op_name:
        Operation name used for weight lookup.
    input_shape:
        Shape of the input array.
    axis:
        Axis along which the reduction is performed.  ``None`` means
        reduce over all elements.

    Returns
    -------
    int
        Weighted public cost estimate for the reduction.
    """
    if not isinstance(op_name, str):
        raise TypeError("reduction_cost() requires op_name as the first argument")
    total = max(_prod(input_shape), 1)
    if axis is None:
        return _weight_cost(op_name, total)
    # Reduction along a single axis: cost is the total element count
    # (each element participates once).
    return _weight_cost(op_name, total)


# ---------------------------------------------------------------------------
# Server-proxied cost functions
# ---------------------------------------------------------------------------


@timed_dispatch
def einsum_cost(subscripts: str, shapes: Sequence[tuple[int, ...]]) -> int:
    """Query the server for the FLOP cost of an einsum operation.

    Parameters
    ----------
    subscripts:
        Einstein summation subscript string.
    shapes:
        Shapes of the input arrays.

    Returns
    -------
    int
        Estimated FLOP cost.
    """
    from flopscope._connection import get_connection
    from flopscope._protocol import encode_request

    conn = get_connection()
    resp = conn.send_recv(
        encode_request(
            "flops.einsum_cost",
            kwargs={"subscripts": subscripts, "shapes": [list(s) for s in shapes]},
        )
    )
    result = resp.get("result", {})
    return int(result.get("value", 0))


@timed_dispatch
def svd_cost(m: int, n: int, k: int = 0) -> int:
    """Query the server for the FLOP cost of an SVD operation.

    Parameters
    ----------
    m:
        Number of rows.
    n:
        Number of columns.
    k:
        Number of singular values to compute (0 means all).

    Returns
    -------
    int
        Estimated FLOP cost.
    """
    from flopscope._connection import get_connection
    from flopscope._protocol import encode_request

    conn = get_connection()
    resp = conn.send_recv(
        encode_request(
            "flops.svd_cost",
            kwargs={"m": m, "n": n, "k": k},
        )
    )
    result = resp.get("result", {})
    return int(result.get("value", 0))
