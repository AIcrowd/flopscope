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


def test_decode_index_key_marker_semantics(handler):
    from flopscope_server._request_handler import _decode_index_key

    # bare wire-list is always a tuple (basic multi-axis index)
    assert handler._decode_index_key([0, 2]) == (0, 2)
    assert _decode_index_key([0, 2]) == (0, 2)
    # marked list is a fancy-index Python list
    assert handler._decode_index_key({"__list__": [0, 2]}) == [0, 2]
    assert _decode_index_key({"__list__": [0, 2]}) == [0, 2]
    # msgpack bytes-key form of the fancy-list marker (wire uses bytes keys)
    assert handler._decode_index_key({b"__list__": [0, 2]}) == [0, 2]
    assert _decode_index_key({b"__list__": [0, 2]}) == [0, 2]
    # nested: tuple(int, fancy-list)
    assert handler._decode_index_key([1, {"__list__": [0, 2]}]) == (1, [0, 2])
    # module-level decoder: same nested case too
    assert _decode_index_key([1, {"__list__": [0, 2]}]) == (1, [0, 2])


# ---------------------------------------------------------------------------
# __getitem__ billing (positive lock)
# ---------------------------------------------------------------------------


def test_getitem_advanced_indexing_bills_on_computed_array(handler, session):
    """The real ``{"op": "__getitem__"}`` wire path DOES deduct FLOPs for
    advanced (fancy / boolean) indexing when the indexed handle is backed by a
    ``FlopscopeArray`` — i.e. the state after any flopscope op has touched the
    array. ``FlopscopeArray.__getitem__`` runs server-side and bills.

    (See ``test_getitem_advanced_indexing_bills_on_first_touch_data`` below
    for the ``create_from_data`` first-touch case, where the stored object
    used to be a plain ``numpy.ndarray`` and the override never fired — this
    test locks the already-working path so it can't silently regress.)
    """
    a = np.arange(1000, dtype=np.float32)
    src = handler.handle(
        {
            "op": "create_from_data",
            "data": a.tobytes(),
            "shape": [1000],
            "dtype": "float32",
        }
    )
    # abs() returns a FlopscopeArray, so the result handle is FlopscopeArray-
    # backed — exactly the state advanced indexing must bill against.
    fa_h = handler.handle(
        {"op": "abs", "args": [{"__handle__": src["result"]["id"]}], "kwargs": {}}
    )["result"]["id"]
    from flopscope._ndarray import FlopscopeArray

    assert isinstance(session.get_array(fa_h), FlopscopeArray)

    # Fancy (integer-array) index of 100 elements -> 4 * 100 = 400.
    idx = np.arange(0, 1000, 10, dtype=np.int64)
    idx_h = handler.handle(
        {
            "op": "create_from_data",
            "data": idx.tobytes(),
            "shape": [100],
            "dtype": "int64",
        }
    )["result"]["id"]
    before = session.budget_status()["flops_used"]
    resp = handler.handle(
        {
            "op": "__getitem__",
            "args": [{"__handle__": fa_h}, {"__handle__": idx_h}],
            "kwargs": {},
        }
    )
    assert resp["status"] == "ok"
    assert session.budget_status()["flops_used"] - before == 400

    # Boolean mask with 250 True -> 250 gathered + 1000 scanned = 1000 + 4*250.
    mask = np.zeros(1000, dtype=bool)
    mask[:250] = True
    mask_h = handler.handle(
        {
            "op": "create_from_data",
            "data": mask.tobytes(),
            "shape": [1000],
            "dtype": "bool",
        }
    )["result"]["id"]
    before = session.budget_status()["flops_used"]
    resp = handler.handle(
        {
            "op": "__getitem__",
            "args": [{"__handle__": fa_h}, {"__handle__": mask_h}],
            "kwargs": {},
        }
    )
    assert resp["status"] == "ok"
    assert session.budget_status()["flops_used"] - before == 1000 + 4 * 250


# ---------------------------------------------------------------------------
# Python-list index key = fancy indexing (wire marker {"__list__": [...]})
# ---------------------------------------------------------------------------


def test_handle_getitem_python_list_is_fancy_index(handler, session):
    import numpy as np

    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    handle = session.store_array(arr)
    resp = handler.handle(
        {
            "op": "__getitem__",
            "args": [{"__handle__": handle}, {"__list__": [0, 2]}],
            "kwargs": {},
        }
    )
    assert resp["status"] == "ok"
    out = session.get_array(resp["result"]["id"])
    np.testing.assert_array_equal(out, arr[[0, 2]])


# ---------------------------------------------------------------------------
# __getitem__ / astype billing on FIRST-TOUCH data (budget-bypass regression)
#
# Session.store_array used to store the raw np.frombuffer(...).copy() ndarray
# from create_from_data verbatim. Any op that runs directly against that
# handle via a bare Python dunder/method call (__getitem__, astype) -- rather
# than through a flopscope counted wrapper -- hit numpy's own unbilled
# ndarray implementation instead of FlopscopeArray's billing override, so
# advanced indexing performed as the FIRST op on fresh client-ingested data
# silently billed 0 FLOPs. Session.store_array now view-casts every plain
# ndarray to FlopscopeArray on the way in, so first-touch data bills exactly
# like the already-touched case above.
# ---------------------------------------------------------------------------


def test_getitem_advanced_indexing_bills_on_first_touch_data(handler, session):
    """Fancy/boolean indexing performed DIRECTLY on a handle returned by
    ``create_from_data`` (no intervening flopscope op) must bill exactly like
    the already-touched case in
    ``test_getitem_advanced_indexing_bills_on_computed_array`` above. Before
    the ``Session.store_array`` fix, both assertions below failed with 0
    billed FLOPs because the stored object was a plain ``numpy.ndarray`` and
    ``arr[key]`` resolved to numpy's own unbilled ``__getitem__``.
    """
    a = np.arange(1000, dtype=np.float32)
    src_h = handler.handle(
        {
            "op": "create_from_data",
            "data": a.tobytes(),
            "shape": [1000],
            "dtype": "float32",
        }
    )["result"]["id"]

    from flopscope._ndarray import FlopscopeArray

    assert isinstance(session.get_array(src_h), FlopscopeArray)

    # Fancy (integer-array) index of 100 elements -> 4 * 100 = 400.
    idx = np.arange(0, 1000, 10, dtype=np.int64)
    idx_h = handler.handle(
        {
            "op": "create_from_data",
            "data": idx.tobytes(),
            "shape": [100],
            "dtype": "int64",
        }
    )["result"]["id"]
    before = session.budget_status()["flops_used"]
    resp = handler.handle(
        {
            "op": "__getitem__",
            "args": [{"__handle__": src_h}, {"__handle__": idx_h}],
            "kwargs": {},
        }
    )
    assert resp["status"] == "ok"
    assert session.budget_status()["flops_used"] - before == 400

    # Boolean mask with 250 True -> 250 gathered + 1000 scanned = 1000 + 4*250.
    mask = np.zeros(1000, dtype=bool)
    mask[:250] = True
    mask_h = handler.handle(
        {
            "op": "create_from_data",
            "data": mask.tobytes(),
            "shape": [1000],
            "dtype": "bool",
        }
    )["result"]["id"]
    before = session.budget_status()["flops_used"]
    resp = handler.handle(
        {
            "op": "__getitem__",
            "args": [{"__handle__": src_h}, {"__handle__": mask_h}],
            "kwargs": {},
        }
    )
    assert resp["status"] == "ok"
    assert session.budget_status()["flops_used"] - before == 1000 + 4 * 250


def test_astype_on_first_touch_data_preserves_flopscopearray_typing(handler, session):
    """``astype`` bills like ``copy`` (Option B billing fix: registry weight
    1.0 -- a real cast/copy is charged ``numel`` at the heavier of
    source/destination dtype rate, matching ``FlopscopeArray.astype``'s own
    formula). Here a 1000-element float64->float32 narrowing cast bills
    `1000 * rate(float64)=2.0 = 2000` (float64 is the heavier operand).

    What this test actually regresses on: before the ``Session.store_array``
    fix, calling ``astype`` directly on a ``create_from_data`` handle (a
    plain ``numpy.ndarray``) resolved to base ``numpy.ndarray.astype`` and
    stored its plain-ndarray result verbatim -- silently extending the
    billing bypass to every downstream op on THAT result too, not just the
    astype call itself. After the fix, the astype result is stored as a
    ``FlopscopeArray`` and a subsequent fancy-index read on it bills
    correctly instead of continuing the bypass chain.
    """
    a = np.arange(1000, dtype=np.float64)
    src_h = handler.handle(
        {
            "op": "create_from_data",
            "data": a.tobytes(),
            "shape": [1000],
            "dtype": "float64",
        }
    )["result"]["id"]

    before = session.budget_status()["flops_used"]
    resp = handler.handle(
        {"op": "astype", "args": [{"__handle__": src_h}, "float32"], "kwargs": {}}
    )
    assert resp["status"] == "ok"
    # Real cast (default copy=True): billed like copy, at the heavier of
    # source (float64, rate 2.0) / destination (float32, rate 1.0) --
    # 1000 * 2.0 = 2000. This assertion is not the regression under test.
    assert session.budget_status()["flops_used"] - before == 2000

    from flopscope._ndarray import FlopscopeArray

    astype_h = resp["result"]["id"]
    assert isinstance(session.get_array(astype_h), FlopscopeArray)

    # The chain must stay billable: a fancy-index read on the astype OUTPUT
    # bills like any other FlopscopeArray-backed handle (4 * 100 = 400).
    idx = np.arange(0, 1000, 10, dtype=np.int64)
    idx_h = handler.handle(
        {
            "op": "create_from_data",
            "data": idx.tobytes(),
            "shape": [100],
            "dtype": "int64",
        }
    )["result"]["id"]
    before2 = session.budget_status()["flops_used"]
    resp2 = handler.handle(
        {
            "op": "__getitem__",
            "args": [{"__handle__": astype_h}, {"__handle__": idx_h}],
            "kwargs": {},
        }
    )
    assert resp2["status"] == "ok"
    assert session.budget_status()["flops_used"] - before2 == 400


# ---------------------------------------------------------------------------
# save / savez / savez_compressed billing (bill-only, no data transfer,
# nothing written server-side) — closes the client's free-save budget bypass:
# flopscope-client used to write the file entirely locally and never
# round-trip to the server, so the 4*numel egress cost was never billed.
# ---------------------------------------------------------------------------


def test_handle_save_bills_four_times_numel(handler, session):
    """A ``save`` request for a single handle deducts 4*(numel + ndim*8),
    matches the in-process ``flops.save`` formula, writes nothing, and
    returns no result payload (the client already wrote the file locally).

    The ``+ndim*8`` term is the array's ``.npy`` shape header: a
    participant-controlled channel (e.g. ``zeros((0, K))`` has 0 elements
    but an arbitrary ``K``), billed alongside the element data."""
    a = np.arange(1000, dtype=np.float32)
    h = handler.handle(
        {
            "op": "create_from_data",
            "data": a.tobytes(),
            "shape": [1000],
            "dtype": "float32",
        }
    )["result"]["id"]

    before = session.budget_status()["flops_used"]
    resp = handler.handle({"op": "save", "args": [{"__handle__": h}], "kwargs": {}})

    assert resp["status"] == "ok"
    assert resp["result"] is None
    # numel(1000) + shape-header (ndim=1 -> 1*8 bytes).
    assert session.budget_status()["flops_used"] - before == 4 * (1000 + 8)
    # budget info in the response matches a fresh query (server-owned counting).
    assert resp["budget"] == session.budget_status()


def test_handle_savez_bills_sum_of_all_handles(handler, session):
    """``savez``/``savez_compressed`` deduct 4*(sum of numel across every
    handle passed + sum of ndim*8 shape-header bytes), matching the
    in-process ``flops.savez`` formula. No names blob is passed here (this
    test exercises the handler's raw handle-summation mechanics, not a full
    client round-trip with member names -- see
    test_handle_savez_bills_same_as_in_process below for that)."""
    a = np.arange(250, dtype=np.float32)
    b = np.arange(150, dtype=np.float32)
    h1 = handler.handle(
        {
            "op": "create_from_data",
            "data": a.tobytes(),
            "shape": [250],
            "dtype": "float32",
        }
    )["result"]["id"]
    h2 = handler.handle(
        {
            "op": "create_from_data",
            "data": b.tobytes(),
            "shape": [150],
            "dtype": "float32",
        }
    )["result"]["id"]

    # numel(250 + 150) + shape-header (2 handles, each ndim=1 -> 2*8 bytes).
    before = session.budget_status()["flops_used"]
    resp = handler.handle(
        {
            "op": "savez",
            "args": [{"__handle__": h1}, {"__handle__": h2}],
            "kwargs": {},
        }
    )
    assert resp["status"] == "ok"
    assert session.budget_status()["flops_used"] - before == 4 * (250 + 150 + 2 * 8)

    before2 = session.budget_status()["flops_used"]
    resp2 = handler.handle(
        {
            "op": "savez_compressed",
            "args": [{"__handle__": h1}, {"__handle__": h2}],
            "kwargs": {},
        }
    )
    assert resp2["status"] == "ok"
    assert session.budget_status()["flops_used"] - before2 == 4 * (250 + 150 + 2 * 8)


def test_handle_savez_bills_same_as_in_process(handler, session, tmp_path):
    """The server's savez billing (data handle + a synthetic 1-D uint8 names
    blob handle -- what a real client round-trip sends, see ``_handle_save``'s
    docstring) must match what the in-process ``flops.savez`` wrapper bills
    for the same call, including the shape-header channel on both the data
    array and the synthetic names blob (Step 3/4 of the shape-header fix).

    Both sides are measured as deltas on the SAME ``session`` fixture's
    already-active ``BudgetContext`` (``Session.__init__`` enters one for the
    session's lifetime -- a second top-level ``flops.BudgetContext`` can't be
    opened here; ``BudgetContext``s don't nest). Calling ``fnp.savez``
    directly bills into that active context exactly like a participant's own
    in-process call would during a live session, so this is a faithful
    apples-to-apples in-process-vs-server comparison, not a workaround."""
    import numpy as np

    import flopscope.numpy as fnp

    x = np.ones((3, 4), dtype=np.float64)

    # In-process reference billing.
    before_in_process = session.budget_status()["flops_used"]
    fnp.savez(str(tmp_path / "in_process.npz"), x=fnp.asarray(x))
    in_process = session.budget_status()["flops_used"] - before_in_process

    # Server path: data array + concatenated names blob ("x") as a 1-D uint8 array.
    data = session.store_array(x)
    names = session.store_array(np.frombuffer(b"x", dtype=np.uint8))
    before_server = session.budget_status()["flops_used"]
    resp = handler.handle(
        {
            "op": "savez",
            "args": [{"__handle__": data}, {"__handle__": names}],
            "kwargs": {},
        }
    )
    assert resp["status"] == "ok"
    server = session.budget_status()["flops_used"] - before_server

    assert server == in_process


def test_handle_save_bills_nothing_stored(handler, session):
    """save/savez must not allocate a new array handle -- pure bill-only op."""
    a = np.arange(64, dtype=np.float32)
    h = handler.handle(
        {
            "op": "create_from_data",
            "data": a.tobytes(),
            "shape": [64],
            "dtype": "float32",
        }
    )["result"]["id"]
    before_count = session._conn.arrays.count

    resp = handler.handle({"op": "save", "args": [{"__handle__": h}], "kwargs": {}})

    assert resp["status"] == "ok"
    assert session._conn.arrays.count == before_count


def test_handle_save_unknown_handle_returns_error(handler):
    resp = handler.handle(
        {"op": "save", "args": [{"__handle__": "a999"}], "kwargs": {}}
    )
    assert resp["status"] == "error"
    assert resp["error_type"] == "KeyError"


def test_handle_save_insufficient_budget_returns_error():
    """A save that would overshoot the remaining budget is rejected -- the
    server, not the client, is the sole authority on whether the egress is
    affordable."""
    s = Session(flop_budget=100)
    h = RequestHandler(s)
    a = np.arange(1000, dtype=np.float32)  # would cost 4*1000 = 4000 > 100
    handle = h.handle(
        {
            "op": "create_from_data",
            "data": a.tobytes(),
            "shape": [1000],
            "dtype": "float32",
        }
    )["result"]["id"]

    resp = h.handle({"op": "save", "args": [{"__handle__": handle}], "kwargs": {}})
    assert resp["status"] == "error"
    assert resp["error_type"] == "BudgetExhaustedError"

    s.close()
