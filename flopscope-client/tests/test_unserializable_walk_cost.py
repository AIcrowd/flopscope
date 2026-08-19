"""`_describe_unserializable`: the pre-pack walk must stay eager, and stay cheap.

The walk visits every leaf of every dispatched payload, so its per-leaf cost is
multiplied by a participant-chosen element count — the same shape as the
`_encode_arg` regression next door.

The tempting fix is to defer it into the `except` handler and let msgpack decide,
since msgpack rejects everything the walk rejects. That is WRONG, and these tests
say why: msgpack's packer consults `__class__`, which participant code can define
as a property, so deferring lets a hostile object execute its own code inside our
dispatch. The walk decides everything from `type()` and is the gate that stops
that. So the walk stays where it is and is made cheap instead.
"""

import builtins

import msgpack
import pytest

import flopscope
from flopscope import _describe_unserializable, _is_exact_msgpack_scalar

# Every type msgpack can carry, plus values that must be refused.
ACCEPTED = [
    1.5,
    -0.0,
    float("inf"),
    7,
    -(2**62),
    "x",
    "",
    True,
    False,
    None,
    b"ab",
    bytearray(b"ab"),
    memoryview(b"ab"),
    msgpack.ExtType(5, b"ab"),
    msgpack.Timestamp(1, 0),
]

REFUSED = [
    {1, 2},
    frozenset([1]),
    1 + 2j,
    object(),
    len,
    Ellipsis,
    NotImplemented,
]


@pytest.mark.parametrize("value", ACCEPTED, ids=repr)
def test_every_msgpack_native_scalar_is_accepted(value):
    assert _is_exact_msgpack_scalar(value) is True
    assert _describe_unserializable([value], {}) == ""


@pytest.mark.parametrize("value", REFUSED, ids=repr)
def test_non_native_values_are_refused_and_named(value):
    assert _is_exact_msgpack_scalar(value) is False
    described = _describe_unserializable([value], {})
    assert described, "a refused value must be described for the error message"
    assert type(value).__name__ in described


def test_accepted_set_matches_the_declared_allowlist_exactly():
    """The loop and the allowlist must not drift apart.

    `_is_exact_msgpack_scalar` scans `_MSGPACK_OK`; reordering it for speed is
    safe precisely because every test is exact identity. This pins that the
    membership decided by the function is exactly the declared tuple.
    """
    assert {type(v) for v in ACCEPTED} == set(flopscope._MSGPACK_OK)


def test_subclasses_are_not_exact_scalars_but_are_still_serializable():
    """Exactness is the point: a subclass must not match by identity."""

    class F(float):
        pass

    class S(str):
        pass

    assert _is_exact_msgpack_scalar(F(1.5)) is False
    assert _is_exact_msgpack_scalar(S("x")) is False
    # str subclasses are covered by the subclass-base allowlist, so the walk
    # still passes them; float subclasses are normalised earlier by _encode_arg.
    assert _describe_unserializable([S("x")], {}) == ""


# ---------------------------------------------------------------------------
# The reason the walk cannot be deferred to msgpack.
# ---------------------------------------------------------------------------


def test_walk_refuses_a_hostile_object_without_running_its_hooks():
    """`type()` only — never `__class__`, which the participant controls."""
    executed = []

    class Hostile:
        @property
        def __class__(self):
            executed.append("__class__")
            raise AssertionError("participant __class__ property executed")

    hostile = Hostile()
    described = _describe_unserializable([hostile], {})

    assert described, "hostile object must be refused"
    assert executed == [], "the walk must not touch participant hooks"


def test_msgpack_would_execute_the_hostile_hook_if_the_walk_were_deferred():
    """Pins the reason the eager position is load-bearing, not just tidier.

    If this ever stops raising through the participant's property, the walk
    could be moved into the error path and the payload cost would drop to zero.
    Until then, deferring it hands control to participant code.
    """
    executed = []

    class Hostile:
        @property
        def __class__(self):
            executed.append("__class__")
            raise AssertionError("participant __class__ property executed")

    with pytest.raises(AssertionError, match="participant __class__ property"):
        msgpack.packb(Hostile())
    assert executed == ["__class__"]


# ---------------------------------------------------------------------------
# Cost: structural, so it cannot go flaky on a loaded CI box.
# ---------------------------------------------------------------------------


def test_scalar_check_allocates_no_iterator_per_leaf(monkeypatch):
    """The walk's per-leaf cost was dominated by a generator it built each time.

    Asserting wall-clock here would flake; asserting that the hot check does not
    call `any()` pins the actual mechanism that made it expensive.
    """
    calls = []
    real_any = builtins.any

    def counting_any(iterable):
        calls.append(1)
        return real_any(iterable)

    monkeypatch.setattr(builtins, "any", counting_any)

    payload = [float(i) for i in range(500)]
    assert _describe_unserializable([payload], {}) == ""
    assert calls == []


def test_bulk_payload_walk_visits_each_leaf_once(monkeypatch):
    """Guards against the walk gaining a second traversal."""
    seen = []
    real = flopscope._is_exact_msgpack_scalar

    def counting(value):
        seen.append(value)
        return real(value)

    monkeypatch.setattr(flopscope, "_is_exact_msgpack_scalar", counting)

    payload = [float(i) for i in range(200)]
    _describe_unserializable([payload], {})

    leaves = [v for v in seen if type(v) is float]
    assert len(leaves) == 200, "each leaf must be visited exactly once"
