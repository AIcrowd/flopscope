"""The client's local-callback ops raise RemoteCallbackError (not an opaque
msgpack TypeError) when handed a Python callable on the client/server backend.

These tests need no live server: the error fires at encode time, before any
connection is made.
"""

import time

import msgpack
import pytest
from flopscope._protocol import encode_request
from flopscope._registry_data import LOCAL_CALLBACK_OPS

import flopscope
from flopscope.errors import RemoteCallbackError, RemoteSerializationError


def test_local_callback_ops_set():
    # LOCAL_CALLBACK_OPS is generated from the server registry's
    # `local_callback` flag (scripts/sync_client.py) and content-checked
    # against it independently: CI's client-server-sync job runs
    # `sync_client.py --check`, and the root suite's
    # test_callback_op_registration.py::test_client_registry_data_is_in_sync
    # re-derives the expected set from the registry. A hand-copied literal
    # here would be a third copy of that list to keep in sync by hand --
    # exactly how `mask_indices` went missing before. Assert the structural
    # invariant this module actually relies on instead: every flagged op is a
    # callable reachable on the client's public surface.
    assert isinstance(LOCAL_CALLBACK_OPS, frozenset) and LOCAL_CALLBACK_OPS
    assert all(
        isinstance(op, str) and callable(getattr(flopscope, op, None))
        for op in LOCAL_CALLBACK_OPS
    )


def test_encode_request_rejects_callables():
    # Precondition: msgpack genuinely cannot serialize a Python function.
    with pytest.raises((TypeError, ValueError)):
        encode_request("apply_along_axis", args=[lambda x: x], kwargs={})


class _PropertyDtype:
    def __init__(self):
        self.calls = 0

    @property
    def _flopscope_dtype_name(self):
        self.calls += 1
        time.sleep(0.1)
        raise AssertionError("participant property executed")


class _NameDescriptor:
    def __get__(self, instance, owner):
        instance.calls += 1
        time.sleep(0.1)
        raise AssertionError("participant descriptor executed")


class _DescriptorDtype:
    name = _NameDescriptor()

    def __init__(self):
        self.calls = 0


@pytest.mark.parametrize("spec", [_PropertyDtype(), _DescriptorDtype()])
def test_dtype_descriptor_raises_remote_callback_without_execution(spec, monkeypatch):
    network_calls = 0

    def network_forbidden():
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network accessed")

    monkeypatch.setattr(flopscope, "get_connection", network_forbidden)
    with pytest.raises(RemoteCallbackError, match="dtype.*descriptor"):
        flopscope.array([1.0], dtype=spec)
    assert spec.calls == 0
    assert network_calls == 0


def test_inert_unsupported_dtype_like_value_remains_serialization_error():
    class InertDtype:
        pass

    with pytest.raises(RemoteSerializationError):
        flopscope.concatenate([InertDtype()])


def test_unsupported_dtype_does_not_execute_repr_before_network(monkeypatch):
    class ReprDtype:
        def __init__(self):
            self.repr_calls = 0

        def __repr__(self):
            self.repr_calls += 1
            raise AssertionError("participant repr executed")

    spec = ReprDtype()
    network_calls = 0

    def network_forbidden():
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network accessed")

    monkeypatch.setattr(flopscope, "get_connection", network_forbidden)
    with pytest.raises(TypeError, match="Cannot interpret dtype"):
        flopscope.array([1.0], dtype=spec)
    assert spec.repr_calls == 0
    assert network_calls == 0


def test_proxy_dtype_does_not_execute_class_property_before_network(monkeypatch):
    class HostileDtype:
        def __init__(self):
            self.class_calls = 0
            self.dtype_calls = 0

        @property
        def __class__(self):
            self.class_calls += 1
            raise AssertionError("participant __class__ property executed")

        @property
        def _flopscope_dtype_name(self):
            self.dtype_calls += 1
            raise AssertionError("participant dtype property executed")

    spec = HostileDtype()
    network_calls = 0

    def network_forbidden():
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network accessed")

    monkeypatch.setattr(flopscope, "get_connection", network_forbidden)
    with pytest.raises(RemoteCallbackError, match="dtype.*descriptor"):
        flopscope.zeros((1,), dtype=spec)
    assert spec.class_calls == 0
    assert spec.dtype_calls == 0
    assert network_calls == 0


def test_serialization_diagnostic_does_not_execute_class_property(monkeypatch):
    class HostileDtype:
        def __init__(self):
            self.class_calls = 0

        @property
        def __class__(self):
            self.class_calls += 1
            raise AssertionError("participant __class__ property executed")

    spec = HostileDtype()
    network_calls = 0

    def network_forbidden():
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network accessed")

    monkeypatch.setattr(flopscope, "get_connection", network_forbidden)
    with pytest.raises(RemoteSerializationError, match="zeros"):
        flopscope.zeros((1,), dtype=spec)
    assert spec.class_calls == 0
    assert network_calls == 0


def test_serialization_diagnostic_does_not_execute_metaclass_name(monkeypatch):
    name_calls = 0

    class HostileMeta(type):
        def __getattribute__(cls, name):
            nonlocal name_calls
            if name == "__name__":
                name_calls += 1
                raise AssertionError("participant metaclass name lookup executed")
            return super().__getattribute__(name)

    class HostileDtype(metaclass=HostileMeta):
        pass

    network_calls = 0

    def network_forbidden():
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network accessed")

    monkeypatch.setattr(flopscope, "get_connection", network_forbidden)
    with pytest.raises(RemoteSerializationError, match="zeros"):
        flopscope.zeros((1,), dtype=HostileDtype())
    assert name_calls == 0
    assert network_calls == 0


@pytest.mark.parametrize("spec", [bytearray(b"dtype"), memoryview(b"dtype")])
def test_proxy_preserves_msgpack_binary_buffer_forms(spec, monkeypatch):
    class Encoded(Exception):
        pass

    encoded = []

    def capture_encode_request(op_name, args, kwargs):
        encoded.append((op_name, args, kwargs))
        raise Encoded

    monkeypatch.setattr(flopscope, "encode_request", capture_encode_request)
    with pytest.raises(Encoded):
        flopscope.zeros((1,), dtype=spec)
    assert encoded == [("zeros", [[1]], {"dtype": spec})]


def test_proxy_preserves_exact_msgpack_extension_values(monkeypatch):
    class Encoded(Exception):
        pass

    extension = msgpack.ExtType(7, b"dtype")
    encoded = []

    def capture_encode_request(op_name, args, kwargs):
        encoded.append((kwargs["dtype"], encode_request(op_name, args, kwargs)))
        raise Encoded

    monkeypatch.setattr(flopscope, "encode_request", capture_encode_request)
    with pytest.raises(Encoded):
        flopscope.zeros((1,), dtype=extension)

    encoded_extension, wire = encoded[0]
    assert encoded_extension is extension
    decoded = msgpack.unpackb(wire, raw=False)
    assert decoded["kwargs"]["dtype"] == extension


def test_proxy_preserves_msgpack_container_and_string_subclasses(monkeypatch):
    class L(list):
        pass

    class T(tuple):
        pass

    class D(dict):
        pass

    class S(str):
        pass

    class Encoded(Exception):
        pass

    encoded = []

    def capture_encode_request(op_name, args, kwargs):
        encoded.append((op_name, args, kwargs))
        raise Encoded

    monkeypatch.setattr(flopscope, "encode_request", capture_encode_request)
    for spec, expected in (
        (L([1]), [1]),
        (T((1,)), [1]),
        (D(a=1), {"a": 1}),
        (S("x"), "x"),
    ):
        with pytest.raises(Encoded):
            flopscope.zeros((1,), dtype=spec)
        assert encoded[-1][0:2] == ("zeros", [[1]])
        assert encoded[-1][2]["dtype"] == expected


def test_proxy_normalizes_container_subclasses_without_hooks(monkeypatch):
    class L(list):
        def __init__(self, *args):
            super().__init__(*args)
            self.calls = 0

        def __iter__(self):
            self.calls += 1
            raise AssertionError("participant list iterator executed")

    class T(tuple):
        def __new__(cls, *args):
            value = super().__new__(cls, *args)
            value.calls = 0
            return value

        def __iter__(self):
            self.calls += 1
            raise AssertionError("participant tuple iterator executed")

    class D(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        def items(self):
            self.calls += 1
            raise AssertionError("participant dict items executed")

    class Encoded(Exception):
        pass

    encoded = []

    def capture_encode_request(op_name, args, kwargs):
        encoded.append(kwargs["dtype"])
        raise Encoded

    monkeypatch.setattr(flopscope, "encode_request", capture_encode_request)
    for spec, expected_type, expected_value in (
        (L([1]), list, [1]),
        (T((1,)), list, [1]),
        (D(a=1), dict, {"a": 1}),
    ):
        with pytest.raises(Encoded):
            flopscope.zeros((1,), dtype=spec)
        assert type(encoded[-1]) is expected_type
        assert encoded[-1] == expected_value
        assert spec.calls == 0


def test_container_subclass_with_unsupported_value_is_serialization_error(monkeypatch):
    class L(list):
        def __init__(self, *args):
            super().__init__(*args)
            self.calls = 0

        def __iter__(self):
            self.calls += 1
            raise AssertionError("participant list iterator executed")

    spec = L([object()])
    network_calls = 0

    def network_forbidden():
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network accessed")

    monkeypatch.setattr(flopscope, "get_connection", network_forbidden)
    with pytest.raises(RemoteSerializationError, match="zeros"):
        flopscope.zeros((1,), dtype=spec)
    assert spec.calls == 0
    assert network_calls == 0


def test_dict_normalization_rejects_armed_custom_key_without_rehash(monkeypatch):
    class ArmedKey:
        def __init__(self):
            self.armed = False
            self.hash_calls = 0
            self.eq_calls = 0

        def __hash__(self):
            if self.armed:
                self.hash_calls += 1
                raise AssertionError("participant key hash executed")
            return 1

        def __eq__(self, other):
            if self.armed:
                self.eq_calls += 1
                raise AssertionError("participant key equality executed")
            return self is other

    key = ArmedKey()
    spec = {key: 1}
    key.armed = True
    network_calls = 0

    def network_forbidden():
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network accessed")

    monkeypatch.setattr(flopscope, "get_connection", network_forbidden)
    with pytest.raises(RemoteSerializationError, match="dictionary key"):
        flopscope.zeros((1,), dtype=spec)
    assert key.hash_calls == 0
    assert key.eq_calls == 0
    assert network_calls == 0


def test_dict_normalization_preserves_safe_keys_and_normalizes_string_subclass(
    monkeypatch,
):
    class StringKey(str):
        def __new__(cls, value):
            result = super().__new__(cls, value)
            result.armed = False
            result.hash_calls = 0
            return result

        def __hash__(self):
            if self.armed:
                self.hash_calls += 1
                raise AssertionError("participant string-key hash executed")
            return super().__hash__()

    key = StringKey("subclass")
    spec = {"text": 1, 2: "int", 2.5: "float", b"bytes": "binary", key: 3}
    key.armed = True

    class Encoded(Exception):
        pass

    encoded = []

    def capture_encode_request(op_name, args, kwargs):
        encoded.append(kwargs["dtype"])
        raise Encoded

    monkeypatch.setattr(flopscope, "encode_request", capture_encode_request)
    with pytest.raises(Encoded):
        flopscope.zeros((1,), dtype=spec)
    assert encoded == [
        {"text": 1, 2: "int", 2.5: "float", b"bytes": "binary", "subclass": 3}
    ]
    assert key.hash_calls == 0


def test_proxy_normalizes_msgpack_scalar_subclasses_without_hooks(monkeypatch):
    class B(bytes):
        def __init__(self, *args):
            self.calls = 0

        def __bytes__(self):
            self.calls += 1
            raise AssertionError("participant bytes conversion executed")

    class BA(bytearray):
        def __init__(self, *args):
            super().__init__(*args)
            self.calls = 0

        def __iter__(self):
            self.calls += 1
            raise AssertionError("participant bytearray iterator executed")

    class IntSubclass(int):
        def __new__(cls, value):
            result = super().__new__(cls, value)
            result.calls = 0
            return result

        def __int__(self):
            self.calls += 1
            raise AssertionError("participant int conversion executed")

    class FloatSubclass(float):
        def __new__(cls, value):
            result = super().__new__(cls, value)
            result.calls = 0
            return result

        def __float__(self):
            self.calls += 1
            raise AssertionError("participant float conversion executed")

    class Encoded(Exception):
        pass

    original_encode_request = flopscope.encode_request
    encoded = []
    packed = []

    def capture_encode_request(op_name, args, kwargs):
        encoded.append(kwargs["dtype"])
        packed.append(original_encode_request(op_name, args, kwargs))
        raise Encoded

    monkeypatch.setattr(flopscope, "encode_request", capture_encode_request)
    for spec, expected_type, expected_value in (
        (B(b"x"), bytes, b"x"),
        (BA(b"x"), bytearray, bytearray(b"x")),
        (IntSubclass(2), int, 2),
        (FloatSubclass(2.5), float, 2.5),
    ):
        with pytest.raises(Encoded):
            flopscope.zeros((1,), dtype=spec)
        decoded = msgpack.unpackb(packed[-1], raw=False)
        value = decoded["kwargs"]["dtype"]
        assert type(encoded[-1]) is expected_type
        assert value == expected_value
        assert spec.calls == 0


def test_unset_dtype_slot_is_not_a_raw_attribute_error(monkeypatch):
    class UnsetDtype:
        __slots__ = ("name",)

    network_calls = 0

    def network_forbidden():
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network accessed")

    monkeypatch.setattr(flopscope, "get_connection", network_forbidden)
    with pytest.raises(TypeError, match="Cannot interpret dtype"):
        flopscope.array([1.0], dtype=UnsetDtype())
    assert network_calls == 0


@pytest.mark.parametrize("proxy_type", [flopscope.RemoteScalar, flopscope.RemoteArray])
def test_uninitialized_proxy_subclass_raises_serialization_error(
    proxy_type, monkeypatch
):
    class ProxySubclass(proxy_type):
        pass

    proxy = ProxySubclass.__new__(ProxySubclass)
    network_calls = 0

    def network_forbidden():
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network accessed")

    monkeypatch.setattr(flopscope, "get_connection", network_forbidden)
    with pytest.raises(RemoteSerializationError, match="uninitialized remote proxy"):
        flopscope.zeros((1,), dtype=proxy)
    assert network_calls == 0


@pytest.mark.parametrize("op", sorted(LOCAL_CALLBACK_OPS))
def test_callback_op_raises_remote_callback_error(op):
    fn = getattr(flopscope, op)
    with pytest.raises(RemoteCallbackError) as ei:
        fn(lambda *a, **k: 0.0)  # a Python callable can't cross the wire
    assert op in str(ei.value)
