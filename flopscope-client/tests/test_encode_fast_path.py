"""The `_encode_arg` exact-type fast path: equivalence, hardening, and cost.

`_encode_arg` recurses to every leaf of every argument, so its per-leaf cost is
multiplied by the participant-chosen element count. The hardened type checks are
correct but expensive, and the commonest leaves (plain floats) used to fall
through all of them into `_resolve_dtype_wire_name`. These tests pin the fast
path that short-circuits them, and — more importantly — pin that it cannot be
reached by anything that is not *exactly* one of the passthrough types.
"""

import math

import pytest
from flopscope._remote_array import _encode_arg

from flopscope import _remote_array
from flopscope.errors import RemoteSerializationError

# Leaf values whose encoding is the identity. Exactly the types the fast path
# claims, with the awkward values that a naive shortcut would get wrong.
IDENTITY_LEAVES = [
    0.0,
    -0.0,
    1.5,
    -2.5,
    1e308,
    5e-324,
    math.inf,
    -math.inf,
    0,
    1,
    -1,
    2**63,
    -(2**63),
    10**40,
    True,
    False,
    "float64",
    "bool_",
    "int32",
    "longdouble",
    "",
    "naïve",
    "name",
    None,
]


@pytest.mark.parametrize("value", IDENTITY_LEAVES, ids=repr)
def test_fast_path_leaf_encodes_to_itself(value):
    """The fast path must be a pure no-op: same object, not merely equal."""
    assert _encode_arg(value) is value


def test_nan_leaf_encodes_to_itself():
    nan = float("nan")
    assert _encode_arg(nan) is nan


def test_fast_path_leaves_skip_the_hardened_checks(monkeypatch):
    """Cost regression guard, asserted structurally rather than by wall clock.

    A timing assertion here would be flaky in CI. Counting the hardened calls
    the fast path exists to avoid is deterministic and pins the actual
    mechanism: a plain float leaf must not touch `_has_proxy_base` or
    `_resolve_dtype_wire_name` at all.
    """
    calls = {"proxy_base": 0, "dtype_wire": 0}

    real_has_proxy_base = _remote_array._has_proxy_base
    real_resolve = _remote_array._resolve_dtype_wire_name

    def counting_has_proxy_base(value, proxy_type):
        calls["proxy_base"] += 1
        return real_has_proxy_base(value, proxy_type)

    def counting_resolve(spec):
        calls["dtype_wire"] += 1
        return real_resolve(spec)

    monkeypatch.setattr(_remote_array, "_has_proxy_base", counting_has_proxy_base)
    monkeypatch.setattr(_remote_array, "_resolve_dtype_wire_name", counting_resolve)

    _encode_arg(1.5)
    _encode_arg(7)
    _encode_arg(True)
    _encode_arg("float64")
    _encode_arg(None)

    assert calls == {"proxy_base": 0, "dtype_wire": 0}


def test_bulk_numeric_payload_is_encoded_without_per_leaf_dtype_resolution(monkeypatch):
    """The whole point: a large literal payload must not pay per-element."""
    calls = {"dtype_wire": 0}
    real_resolve = _remote_array._resolve_dtype_wire_name

    def counting_resolve(spec):
        calls["dtype_wire"] += 1
        return real_resolve(spec)

    monkeypatch.setattr(_remote_array, "_resolve_dtype_wire_name", counting_resolve)

    payload = [[float(c) for c in range(100)] for _ in range(100)]
    encoded = _encode_arg(payload)

    assert encoded == payload
    assert calls["dtype_wire"] == 0


# ---------------------------------------------------------------------------
# The fast path must not become a hole in the encoding hardening.
# ---------------------------------------------------------------------------


class _LyingClassFloat(float):
    """A float subclass that lies to `isinstance` via `__class__`."""

    @property
    def __class__(self):
        return float


def test_subclass_lying_about_its_class_does_not_take_the_fast_path():
    value = _LyingClassFloat(1.5)
    assert isinstance(value, float)  # the lie works on isinstance
    assert type(value) is not float  # identity does not believe it

    encoded = _encode_arg(value)
    # The hardened branch normalises a float subclass to a plain float; the
    # fast path would have returned the subclass instance itself.
    assert encoded is not value
    assert type(encoded) is float
    assert encoded == 1.5


def test_hostile_metaclass_cannot_reach_the_fast_path():
    """Why the fast path is a chain of `is`, and must never become a set.

    `type(arg) in {float, int, ...}` and `_TABLE.get(type(arg))` both consult
    `__eq__`/`__hash__` on the *metaclass*, which participant code controls. A
    hostile metaclass makes both forms return a match for an arbitrary object.
    Only `type(arg) is float` is an unspoofable pointer comparison.
    """

    class _EvilMeta(type):
        def __hash__(cls):
            return hash(float)

        def __eq__(cls, other):
            return other is float

    class _Hostile(metaclass=_EvilMeta):
        @property
        def __class__(self):
            return float

    hostile = _Hostile()

    # Demonstrate the vectors this test exists to keep closed.
    assert type(hostile) in frozenset({float, int, str, bool, type(None)})
    assert type(hostile) in {float: 1, int: 2}
    assert isinstance(hostile, float)
    # ...none of which the fast path uses:
    assert type(hostile) is not float

    # Encoding must fall through to the hardened path, which does not recognise
    # this object as any proxy or dtype and passes it through untouched — but it
    # must get there by identity checks, not by believing the metaclass.
    assert _encode_arg(hostile) is hostile


def test_str_and_int_subclasses_are_still_normalised():
    class _Str(str):
        pass

    class _Int(int):
        pass

    encoded_str = _encode_arg(_Str("float64"))
    assert type(encoded_str) is str and encoded_str == "float64"

    encoded_int = _encode_arg(_Int(5))
    assert type(encoded_int) is int and encoded_int == 5


def test_dtype_like_args_still_resolve_to_wire_names():
    """The fast path must not shadow real dtype resolution."""
    assert _encode_arg(float) == "float64"
    assert _encode_arg(int) == "int64"
    assert _encode_arg(bool) == "bool"
    assert _encode_arg(complex) == "complex128"


def test_containers_of_fast_path_leaves_round_trip():
    assert _encode_arg([1.0, 2, "x", True, None]) == [1.0, 2, "x", True, None]
    assert _encode_arg((1.0, 2)) == [1.0, 2]
    assert _encode_arg({"a": 1.0, 2: "b"}) == {"a": 1.0, 2: "b"}


def test_rebinding_a_builtin_cannot_redirect_the_fast_path():
    """Why the fast path compares against import-time constants.

    The bare builtin names resolve through `builtins` at call time, so a
    participant doing `builtins.float = dict` would make `type(arg) is float`
    true for a dict — returning it verbatim and skipping the
    `_is_safe_wire_key` guard in the mapping branch, which is what stops a
    composite key reaching the wire.
    """
    import builtins

    payload = {(1, 2): "v"}  # a key `_is_safe_wire_key` must reject

    with pytest.raises(RemoteSerializationError):
        _encode_arg(payload)

    original = builtins.float
    builtins.float = dict
    try:
        with pytest.raises(RemoteSerializationError):
            _encode_arg(payload)
    finally:
        builtins.float = original
