"""The wire's multi-result form must carry the container type.

``{"multi": [...]}`` was designed for ops returning a homogeneous list of
arrays (``nonzero`` returns one per dimension) and has no slot for the
container type. Namedtuple-returning ops (``linalg.svd``, ``linalg.qr``, ...)
satisfy ``isinstance(result, tuple)``, so they flow down the same path and
their type is dropped here, at pack time — which is why ``svd(a).U`` works
in-process and raises ``AttributeError`` on a remote client.

The container is described generically from ``type(result).__name__`` and
``type(result)._fields``: every numpy structured result is an ordinary
namedtuple, so a numpy release that adds one needs no change here.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest
from flopscope_server._request_handler import RequestHandler
from flopscope_server._session import Session

Pair = namedtuple("Pair", ["first", "second"])


@pytest.fixture()
def handler():
    session = Session(flop_budget=10**15)
    yield RequestHandler(session)
    session.close()


def test_namedtuple_result_carries_its_container_type(handler):
    packed = handler._pack_result(Pair(np.arange(3.0), np.arange(2.0)))
    assert packed["status"] == "ok"
    assert packed["result"]["multi_type"] == {
        "name": "Pair",
        "fields": ["first", "second"],
    }


def test_namedtuple_elements_are_packed_exactly_as_before(handler):
    plain = handler._pack_result((np.arange(3.0), np.float64(2.5)))
    named = handler._pack_result(Pair(np.arange(3.0), np.float64(2.5)))
    # Handle ids differ (two separate stores); everything else must match.
    assert len(named["result"]["multi"]) == len(plain["result"]["multi"])
    assert named["result"]["multi"][0]["shape"] == plain["result"]["multi"][0]["shape"]
    assert named["result"]["multi"][0]["dtype"] == plain["result"]["multi"][0]["dtype"]
    assert named["result"]["multi"][1] == plain["result"]["multi"][1]


def test_plain_tuple_result_carries_no_container_type(handler):
    packed = handler._pack_result((np.arange(3.0), np.arange(2.0)))
    assert packed["status"] == "ok"
    assert "multi_type" not in packed["result"]


def test_list_result_carries_no_container_type(handler):
    packed = handler._pack_result([np.arange(3.0), np.arange(2.0)])
    assert packed["status"] == "ok"
    assert "multi_type" not in packed["result"]


def test_numpy_structured_results_are_described_without_a_lookup_table(handler):
    # The server must not need to know about SVDResult/QRResult/... by name:
    # whatever numpy calls the container and its fields is what goes on the
    # wire. Asserting against numpy's own values (rather than literals) keeps
    # this honest across numpy releases.
    a = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    for result in (np.linalg.svd(a), np.linalg.qr(a), np.linalg.eigh(a)):
        packed = handler._pack_result(result)
        assert packed["status"] == "ok"
        assert packed["result"]["multi_type"] == {
            "name": type(result).__name__,
            "fields": list(type(result)._fields),
        }


def test_a_tuple_subclass_without_fields_carries_no_container_type(handler):
    class Weird(tuple):
        pass

    packed = handler._pack_result(Weird((np.arange(3.0), np.arange(2.0))))
    assert packed["status"] == "ok"
    assert "multi_type" not in packed["result"]


def test_a_bogus_fields_attribute_is_not_put_on_the_wire(handler):
    # Only a genuine namedtuple shape is describable. Anything else must fall
    # back to the plain multi form rather than emitting a container the client
    # cannot rebuild (or, worse, one whose arity disagrees with the payload).
    class Bogus(tuple):
        _fields = ("only_one",)

    packed = handler._pack_result(Bogus((np.arange(3.0), np.arange(2.0))))
    assert packed["status"] == "ok"
    assert "multi_type" not in packed["result"]


def test_non_identifier_field_names_are_not_put_on_the_wire(handler):
    class Bogus(tuple):
        _fields = ("ok", "not an identifier")

    packed = handler._pack_result(Bogus((np.arange(3.0), np.arange(2.0))))
    assert packed["status"] == "ok"
    assert "multi_type" not in packed["result"]


def test_duplicate_field_names_are_not_put_on_the_wire(handler):
    class Bogus(tuple):
        _fields = ("same", "same")

    packed = handler._pack_result(Bogus((np.arange(3.0), np.arange(2.0))))
    assert packed["status"] == "ok"
    assert "multi_type" not in packed["result"]


def test_underscore_field_names_are_not_put_on_the_wire(handler):
    # collections.namedtuple refuses leading-underscore field names, so a
    # container described with one could never be rebuilt on the client.
    class Bogus(tuple):
        _fields = ("ok", "_private")

    packed = handler._pack_result(Bogus((np.arange(3.0), np.arange(2.0))))
    assert packed["status"] == "ok"
    assert "multi_type" not in packed["result"]
