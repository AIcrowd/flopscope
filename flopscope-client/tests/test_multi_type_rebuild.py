"""Rebuild a namedtuple result from the wire's ``multi_type`` description.

The server describes a structured result's container generically (its name and
field names); this side turns that back into a real namedtuple with
``collections.namedtuple`` — pure stdlib, because the client's only runtime
dependencies are pyzmq and msgpack and it must never import numpy.

Both compatibility directions are pinned here: a response with no
``multi_type`` (an older server) must still yield exactly the plain tuple it
always did, and a description this client cannot honour must degrade to a plain
tuple rather than raising.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from flopscope._remote_array import (
    RemoteArray,
    RemoteScalar,
    _namedtuple_class,
    _result_from_response,
)


def _multi_response(multi_type=None):
    result = {
        "multi": [
            {"id": "h1", "shape": [2, 2], "dtype": "float64"},
            {"id": "h2", "shape": [2], "dtype": "float64"},
            {"id": "h3", "shape": [2, 2], "dtype": "float64"},
        ],
    }
    if multi_type is not None:
        result["multi_type"] = multi_type
    return {"status": "ok", "result": result}


_SVD = {"name": "SVDResult", "fields": ["U", "S", "Vh"]}


class TestRebuild:
    def test_fields_are_reachable_by_name(self):
        out = _result_from_response(_multi_response(_SVD))
        assert isinstance(out.U, RemoteArray)
        assert out.U.handle_id == "h1"
        assert out.S.handle_id == "h2"
        assert out.Vh.handle_id == "h3"

    def test_the_container_reports_the_servers_name_and_fields(self):
        out = _result_from_response(_multi_response(_SVD))
        assert type(out).__name__ == "SVDResult"
        assert out._fields == ("U", "S", "Vh")

    def test_it_is_still_a_tuple_so_indexing_and_unpacking_work(self):
        out = _result_from_response(_multi_response(_SVD))
        assert isinstance(out, tuple)
        assert len(out) == 3
        u, s, vh = out
        assert (u.handle_id, s.handle_id, vh.handle_id) == ("h1", "h2", "h3")
        assert out[0] is u

    def test_two_element_unpacking_of_a_two_field_result_works(self):
        # examples/04_svd_usage.py unpacks qr as `q, r = fnp.linalg.qr(x)`.
        resp = {
            "status": "ok",
            "result": {
                "multi": [
                    {"id": "q1", "shape": [2, 2], "dtype": "float64"},
                    {"id": "r1", "shape": [2, 2], "dtype": "float64"},
                ],
                "multi_type": {"name": "QRResult", "fields": ["Q", "R"]},
            },
        }
        q, r = _result_from_response(resp)
        assert q.handle_id == "q1"
        assert r.handle_id == "r1"

    def test_scalar_elements_are_rebuilt_by_name_too(self):
        resp = {
            "status": "ok",
            "result": {
                "multi": [
                    {"value": 1.0, "dtype": "float64"},
                    {"value": 2.5, "dtype": "float64"},
                ],
                "multi_type": {
                    "name": "SlogdetResult",
                    "fields": ["sign", "logabsdet"],
                },
            },
        }
        out = _result_from_response(resp)
        assert isinstance(out.sign, RemoteScalar)
        assert float(out.logabsdet) == 2.5


class TestClassCache:
    def test_the_same_description_reuses_one_class(self):
        # A hot loop calling svd must not mint a fresh class per call.
        first = _result_from_response(_multi_response(_SVD))
        second = _result_from_response(_multi_response(_SVD))
        assert type(first) is type(second)

    def test_different_descriptions_get_different_classes(self):
        svd = _namedtuple_class("SVDResult", ("U", "S", "Vh"))
        qr = _namedtuple_class("QRResult", ("Q", "R"))
        assert svd is not None and qr is not None
        assert svd is not qr

    def test_the_same_name_with_different_fields_is_not_confused(self):
        a = _namedtuple_class("Result", ("x", "y"))
        b = _namedtuple_class("Result", ("p", "q"))
        assert a is not b
        assert a is not None and b is not None
        assert a._fields == ("x", "y")
        assert b._fields == ("p", "q")


class TestOlderServer:
    """New client + old server: no ``multi_type`` key, plain tuple as today."""

    def test_a_response_without_multi_type_is_a_plain_tuple(self):
        out = _result_from_response(_multi_response())
        assert type(out) is tuple
        assert len(out) == 3
        assert out[0].handle_id == "h1"

    def test_a_response_without_multi_type_has_no_fields(self):
        out = _result_from_response(_multi_response())
        assert not hasattr(out, "_fields")


class TestMalformedDescription:
    """A description this client cannot honour degrades, never raises."""

    def test_arity_mismatch_falls_back_to_a_plain_tuple(self):
        out = _result_from_response(
            _multi_response({"name": "Short", "fields": ["only", "two"]})
        )
        assert type(out) is tuple
        assert len(out) == 3

    def test_a_non_identifier_field_falls_back_to_a_plain_tuple(self):
        out = _result_from_response(
            _multi_response({"name": "Bad", "fields": ["ok", "not an id", "x"]})
        )
        assert type(out) is tuple

    def test_a_non_identifier_name_falls_back_to_a_plain_tuple(self):
        out = _result_from_response(
            _multi_response({"name": "not a name", "fields": ["a", "b", "c"]})
        )
        assert type(out) is tuple

    def test_a_missing_fields_key_falls_back_to_a_plain_tuple(self):
        out = _result_from_response(_multi_response({"name": "SVDResult"}))
        assert type(out) is tuple

    def test_a_non_mapping_description_falls_back_to_a_plain_tuple(self):
        out = _result_from_response(_multi_response(["SVDResult", ["U", "S", "Vh"]]))
        assert type(out) is tuple

    def test_duplicate_fields_fall_back_to_a_plain_tuple(self):
        out = _result_from_response(
            _multi_response({"name": "Dup", "fields": ["a", "a", "b"]})
        )
        assert type(out) is tuple

    def test_a_keyword_field_name_falls_back_to_a_plain_tuple(self):
        out = _result_from_response(
            _multi_response({"name": "Kw", "fields": ["class", "b", "c"]})
        )
        assert type(out) is tuple

    def test_an_unhonourable_description_is_not_retried_into_a_cache_miss(self):
        # Repeated bad descriptions must stay cheap and stay non-raising.
        for _ in range(3):
            assert _namedtuple_class("Bad", ("a", "a")) is None


class TestNoNumpyImport:
    def test_rebuilding_works_with_numpy_unimportable(self):
        # The client's only runtime dependencies are pyzmq and msgpack, so the
        # rebuild must be pure stdlib. Proven by running it in a fresh
        # interpreter where importing numpy raises.
        script = textwrap.dedent(
            """
            import sys

            class _NoNumpy:
                def find_module(self, name, path=None):
                    return None

                def find_spec(self, name, path=None, target=None):
                    if name == "numpy" or name.startswith("numpy."):
                        raise ImportError("numpy is not installed")
                    return None

            sys.meta_path.insert(0, _NoNumpy())

            from flopscope._remote_array import _result_from_response

            out = _result_from_response(
                {
                    "status": "ok",
                    "result": {
                        "multi": [
                            {"id": "h1", "shape": [2, 2], "dtype": "float64"},
                            {"id": "h2", "shape": [2], "dtype": "float64"},
                            {"id": "h3", "shape": [2, 2], "dtype": "float64"},
                        ],
                        "multi_type": {
                            "name": "SVDResult",
                            "fields": ["U", "S", "Vh"],
                        },
                    },
                }
            )
            assert "numpy" not in sys.modules, "the rebuild imported numpy"
            print(type(out).__name__, out.U.handle_id, out._fields)
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "SVDResult h1 ('U', 'S', 'Vh')", proc.stdout
