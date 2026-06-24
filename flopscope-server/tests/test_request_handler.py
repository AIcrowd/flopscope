"""Tests for RequestHandler — written first (TDD)."""

from __future__ import annotations

import numpy as np
import pytest
from flopscope_server._request_handler import RequestHandler
from flopscope_server._session import Session

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def session():
    s = Session(flop_budget=1_000_000)
    yield s
    if s.is_open:
        s.close()


@pytest.fixture()
def handler(session):
    return RequestHandler(session)


# ---------------------------------------------------------------------------
# Free ops: zeros / ones
# ---------------------------------------------------------------------------


def test_handle_zeros(handler, session):
    resp = handler.handle({"op": "zeros", "args": [(3, 4)], "kwargs": {}})
    assert resp["status"] == "ok"
    handle = resp["result"]["id"]
    arr = session.get_array(handle)
    np.testing.assert_array_equal(arr, np.zeros((3, 4)))
    assert resp["result"]["shape"] == [3, 4]
    assert resp["result"]["dtype"] == "float64"


def test_handle_ones(handler, session):
    resp = handler.handle({"op": "ones", "args": [(2, 3)], "kwargs": {}})
    assert resp["status"] == "ok"
    handle = resp["result"]["id"]
    arr = session.get_array(handle)
    np.testing.assert_array_equal(arr, np.ones((2, 3)))


# ---------------------------------------------------------------------------
# Counted unary: exp
# ---------------------------------------------------------------------------


def test_handle_unary_exp(handler, session):
    # Create an input array first
    inp = np.array([0.0, 1.0, 2.0])
    h_in = session.store_array(inp)

    resp = handler.handle({"op": "exp", "args": [h_in], "kwargs": {}})
    assert resp["status"] == "ok"
    h_out = resp["result"]["id"]
    result = session.get_array(h_out)
    np.testing.assert_allclose(result, np.exp(inp))


# ---------------------------------------------------------------------------
# Counted binary: add (two handles)
# ---------------------------------------------------------------------------


def test_handle_binary_add(handler, session):
    a = session.store_array(np.array([1.0, 2.0, 3.0]))
    b = session.store_array(np.array([10.0, 20.0, 30.0]))

    resp = handler.handle({"op": "add", "args": [a, b], "kwargs": {}})
    assert resp["status"] == "ok"
    result = session.get_array(resp["result"]["id"])
    np.testing.assert_array_equal(result, [11.0, 22.0, 33.0])


# ---------------------------------------------------------------------------
# Binary with scalar: handle + float
# ---------------------------------------------------------------------------


def test_handle_binary_with_scalar(handler, session):
    a = session.store_array(np.array([1.0, 2.0, 3.0]))

    resp = handler.handle({"op": "add", "args": [a, 10.0], "kwargs": {}})
    assert resp["status"] == "ok"
    result = session.get_array(resp["result"]["id"])
    np.testing.assert_array_equal(result, [11.0, 12.0, 13.0])


# ---------------------------------------------------------------------------
# Reduction: sum
# ---------------------------------------------------------------------------


def test_handle_reduction_sum(handler, session):
    a = session.store_array(np.array([1.0, 2.0, 3.0]))

    resp = handler.handle({"op": "sum", "args": [a], "kwargs": {}})
    assert resp["status"] == "ok"
    # sum returns a scalar (0-d array)
    result = resp["result"]
    # Could be stored as 0-d array or returned as scalar value
    if "id" in result:
        arr = session.get_array(result["id"])
        assert float(arr) == 6.0
    else:
        assert result["value"] == 6.0


# ---------------------------------------------------------------------------
# Einsum: string subscript + handle args
# ---------------------------------------------------------------------------


def test_handle_einsum(handler, session):
    W = session.store_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    x = session.store_array(np.array([1.0, 1.0]))

    resp = handler.handle({"op": "einsum", "args": ["ij,j->i", W, x], "kwargs": {}})
    assert resp["status"] == "ok"
    result = session.get_array(resp["result"]["id"])
    np.testing.assert_allclose(result, [3.0, 7.0])


# ---------------------------------------------------------------------------
# Analytical cost estimators (flopscope.accounting) — the client proxies these
# as flops.* ops. Regression guard for "unknown op: flops.einsum_cost".
# ---------------------------------------------------------------------------


def test_flops_cost_ops_are_whitelisted():
    from flopscope_server._protocol import WHITELIST

    assert "flops.einsum_cost" in WHITELIST
    assert "flops.svd_cost" in WHITELIST


def test_handle_einsum_cost(handler, session):
    import flopscope as flops

    resp = handler.handle(
        {
            "op": "flops.einsum_cost",
            "kwargs": {"subscripts": "ij,jk->ik", "shapes": [[4, 5], [5, 6]]},
        }
    )
    assert resp["status"] == "ok"
    # session BudgetContext is active -> native accounting uses the same weights.
    assert resp["result"]["value"] == flops.accounting.einsum_cost(
        "ij,jk->ik", [(4, 5), (5, 6)]
    )
    assert resp["result"]["value"] > 0


def test_handle_svd_cost(handler, session):
    import flopscope as flops

    full = handler.handle(
        {"op": "flops.svd_cost", "kwargs": {"m": 128, "n": 64, "k": 0}}
    )
    topk = handler.handle(
        {"op": "flops.svd_cost", "kwargs": {"m": 128, "n": 64, "k": 8}}
    )
    assert full["status"] == "ok" and topk["status"] == "ok"
    # client surface uses k=0 to mean FULL svd (native k=None), NOT "top-0".
    assert full["result"]["value"] == flops.accounting.svd_cost(128, 64)
    assert topk["result"]["value"] == flops.accounting.svd_cost(128, 64, k=8)
    assert full["result"]["value"] > topk["result"]["value"] > 0


# ---------------------------------------------------------------------------
# create_from_data
# ---------------------------------------------------------------------------


def test_handle_create_from_data(handler, session):
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    resp = handler.handle(
        {
            "op": "create_from_data",
            "data": arr.tobytes(),
            "shape": [3],
            "dtype": "float64",
        }
    )
    assert resp["status"] == "ok"
    stored = session.get_array(resp["result"]["id"])
    np.testing.assert_array_equal(stored, arr)


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def test_handle_fetch(handler, session):
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    handle = session.store_array(arr)

    resp = handler.handle({"op": "fetch", "id": handle})
    assert resp["status"] == "ok"
    assert resp["data"] == arr.tobytes()
    assert resp["shape"] == [3]
    assert resp["dtype"] == "float64"


# ---------------------------------------------------------------------------
# fetch_slice
# ---------------------------------------------------------------------------


def test_handle_fetch_slice(handler, session):
    arr = np.arange(10, dtype=np.float64)
    handle = session.store_array(arr)

    resp = handler.handle({"op": "fetch_slice", "id": handle, "slices": [[2, 5]]})
    assert resp["status"] == "ok"
    expected = arr[2:5]
    assert resp["data"] == expected.tobytes()
    assert resp["shape"] == [3]
    assert resp["dtype"] == "float64"


def test_handle_fetch_slice_does_not_allocate_new_handle(handler, session):
    arr = np.arange(10, dtype=np.float64)
    handle = session.store_array(arr)
    initial_count = session._conn.arrays.count

    first = handler.handle({"op": "fetch_slice", "id": handle, "slices": [[2, 5]]})
    second = handler.handle({"op": "fetch_slice", "id": handle, "slices": [[4, 7]]})

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert session._conn.arrays.count == initial_count


# ---------------------------------------------------------------------------
# free
# ---------------------------------------------------------------------------


def test_handle_free(handler, session):
    h1 = session.store_array(np.array([1.0]))
    h2 = session.store_array(np.array([2.0]))

    resp = handler.handle({"op": "free", "ids": [h1, h2]})
    assert resp["status"] == "ok"

    with pytest.raises(KeyError):
        session.get_array(h1)
    with pytest.raises(KeyError):
        session.get_array(h2)


# ---------------------------------------------------------------------------
# budget_status
# ---------------------------------------------------------------------------


def test_handle_budget_status(handler):
    resp = handler.handle({"op": "budget_status"})
    assert resp["status"] == "ok"
    result = resp["result"]
    assert "flop_budget" in result
    assert "flops_used" in result
    assert "flops_remaining" in result
    assert result["flop_budget"] == 1_000_000


# ---------------------------------------------------------------------------
# Error: unknown handle
# ---------------------------------------------------------------------------


def test_handle_unknown_handle_returns_error(handler):
    resp = handler.handle({"op": "exp", "args": ["a999"], "kwargs": {}})
    assert resp["status"] == "error"
    assert resp["error_type"] == "KeyError"


# ---------------------------------------------------------------------------
# Error: budget exhausted
# ---------------------------------------------------------------------------


def test_handle_budget_exhausted_returns_error():
    # Tiny budget that will be exceeded
    s = Session(flop_budget=1)
    h = RequestHandler(s)

    # Store a large array
    big = np.ones((100, 100))
    handle = s.store_array(big)

    resp = h.handle({"op": "exp", "args": [handle], "kwargs": {}})
    assert resp["status"] == "error"
    assert resp["error_type"] == "BudgetExhaustedError"
    assert "message" in resp

    s.close()


# ---------------------------------------------------------------------------
# Budget info included in operation responses
# ---------------------------------------------------------------------------


def test_budget_info_included_in_operation_responses(handler, session):
    resp = handler.handle({"op": "zeros", "args": [(3,)], "kwargs": {}})
    assert resp["status"] == "ok"
    assert "budget" in resp
    assert "flops_remaining" in resp["budget"]

    # Now do a counted op
    h = resp["result"]["id"]
    resp2 = handler.handle({"op": "exp", "args": [h], "kwargs": {}})
    assert resp2["status"] == "ok"
    assert "budget" in resp2
    # After exp, some flops should have been used
    assert resp2["budget"]["flops_remaining"] < resp["budget"]["flops_remaining"]


# ---------------------------------------------------------------------------
# FIX 3: _resolve_arg recurses into lists
# ---------------------------------------------------------------------------


def test_resolve_arg_recurse_list(handler, session):
    """Handles inside lists (e.g. concatenate([a, b])) are resolved."""
    a = session.store_array(np.array([1.0, 2.0]))
    b = session.store_array(np.array([3.0, 4.0]))

    resp = handler.handle(
        {
            "op": "concatenate",
            "args": [[{"__handle__": a}, {"__handle__": b}]],
            "kwargs": {},
        }
    )
    assert resp["status"] == "ok"
    result = session.get_array(resp["result"]["id"])
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# FIX 6: _pack_result handles mixed array/scalar tuples
# ---------------------------------------------------------------------------


def test_pack_result_mixed_tuple(handler, session):
    """FIX 6: _pack_result handles tuples containing both arrays and scalars."""
    # Directly test _pack_result with a mixed tuple
    arr = np.array([1.0, 2.0])
    scalar = np.float64(3.14)
    mixed = (arr, scalar)

    result = handler._pack_result(mixed)
    assert result["status"] == "ok"
    assert "multi" in result["result"]
    items = result["result"]["multi"]
    assert len(items) == 2
    # First item should be an array (has "id")
    assert "id" in items[0]
    assert items[0]["shape"] == [2]
    # Second item should be a scalar (has "value")
    assert "value" in items[1]
    assert abs(items[1]["value"] - 3.14) < 1e-10


def test_pack_result_uses_symmetry_payload(handler):
    import flopscope as we

    arr = np.array([[1.0, 2.0], [2.0, 3.0]])
    st = we.as_symmetric(arr, symmetry=we.SymmetryGroup.symmetric(axes=(0, 1)))

    result = handler._pack_result(st)

    assert result["status"] == "ok"
    assert result["result"]["symmetry"] == {"axes": [0, 1], "generators": [[1, 0]]}
    assert "symmetry_info" not in result["result"]


# ---------------------------------------------------------------------------
# FIX 9: Max array size limit
# ---------------------------------------------------------------------------


def test_create_from_data_size_limit(handler, session, monkeypatch):
    """create_from_data rejects arrays exceeding the size limit."""
    import flopscope_server._request_handler as rh

    monkeypatch.setattr(rh, "MAX_ARRAY_BYTES", 100)  # 100 bytes limit

    # Create data that exceeds limit
    arr = np.ones((200,), dtype=np.float64)  # 200 * 8 = 1600 bytes
    resp = handler.handle(
        {
            "op": "create_from_data",
            "data": arr.tobytes(),
            "shape": [200],
            "dtype": "float64",
        }
    )
    assert resp["status"] == "error"
    assert resp["error_type"] == "ValueError"
    assert "too large" in resp["message"]


def test_result_array_size_limit(handler, session, monkeypatch):
    """Operations producing arrays exceeding the limit return an error."""
    import flopscope_server._request_handler as rh

    monkeypatch.setattr(rh, "MAX_ARRAY_BYTES", 100)  # 100 bytes limit

    # ones((200,)) produces 200 * 8 = 1600 bytes
    resp = handler.handle({"op": "ones", "args": [(200,)], "kwargs": {}})
    assert resp["status"] == "error"
    assert resp["error_type"] == "ValueError"
    assert "too large" in resp["message"]


# ---------------------------------------------------------------------------
# Ellipsis (...) indexing: client encodes {"__ellipsis__": True}
# (prod regression sub 310351: "can not serialize 'ellipsis' object")
# ---------------------------------------------------------------------------


def test_decode_index_key_ellipsis(handler):
    """The {"__ellipsis__": True} wire form decodes to Ellipsis (str + bytes keys)."""
    assert handler._decode_index_key({"__ellipsis__": True}) is Ellipsis
    assert handler._decode_index_key({b"__ellipsis__": True}) is Ellipsis
