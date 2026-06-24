"""RemoteScalar combined with a RemoteArray must broadcast to a RemoteArray.

REAL prod regression (2026-06-24 re-grade): participants wrote
``acc = acc + coef * tensor`` where ``coef`` is a 0-D ``RemoteScalar`` (from
indexing/reduction) and ``tensor`` is a ``RemoteArray``. ``RemoteScalar``'s
arithmetic dunders wrapped the result of ``self._value * other`` back in a
``RemoteScalar`` unconditionally; when ``other`` was a ``RemoteArray`` the
product is itself a ``RemoteArray``, producing a malformed "scalar" whose
``_value`` is an array. The next op that took that object as an argument tried
to wire-encode it and crashed with::

    TypeError: can not serialize 'RemoteArray' object

(and, when it sat inside a ``stack`` list, the proxy's
``RemoteSerializationError: stack() received an argument of type 'RemoteArray'``).

numpy's contract is ``scalar (op) array -> array`` (``np.float64(2) *
np.array([1, 2]) == array([2, 4])``). These tests exercise the real
client->server path so the malformed-scalar regression can't return.

All tests rely on the ambient ``BudgetContext`` from the autouse
``_fresh_connection_and_budget`` fixture; they must NOT open their own.
"""

from __future__ import annotations

import pytest

import flopscope as fnp  # the CLIENT (conftest puts flopscope-client/src first)

# Forward op, reverse op, and the expected element value for scalar=2.0,
# array element=4.0 (so we cover commutative + non-commutative ops both ways).
_OPS = [
    ("mul", lambda s, a: s * a, lambda a, s: a * s, 8.0, 8.0),
    ("add", lambda s, a: s + a, lambda a, s: a + s, 6.0, 6.0),
    ("sub", lambda s, a: s - a, lambda a, s: a - s, -2.0, 2.0),
    ("truediv", lambda s, a: s / a, lambda a, s: a / s, 0.5, 2.0),
]


def _scalar_two():
    """A genuine RemoteScalar (0-D) with value 2.0, via indexing."""
    s = fnp.array([2.0, 99.0])[0]
    assert type(s).__name__ == "RemoteScalar", f"expected RemoteScalar, got {type(s)}"
    return s


@pytest.mark.parametrize("name,fwd,rev,fwd_val,rev_val", _OPS)
def test_scalar_op_array_returns_array(name, fwd, rev, fwd_val, rev_val):
    s = _scalar_two()
    arr = fnp.array([4.0, 4.0, 4.0])

    fwd_res = fwd(s, arr)  # scalar OP array
    rev_res = rev(arr, s)  # array OP scalar

    # Must broadcast to an ARRAY, not a (malformed) RemoteScalar.
    assert type(fwd_res).__name__ == "RemoteArray", (
        f"scalar {name} array returned {type(fwd_res).__name__}, expected RemoteArray"
    )
    assert type(rev_res).__name__ == "RemoteArray", (
        f"array {name} scalar returned {type(rev_res).__name__}, expected RemoteArray"
    )
    assert fwd_res.tolist() == [fwd_val] * 3
    assert rev_res.tolist() == [rev_val] * 3


def test_scalar_array_product_is_wire_serializable():
    """The exact crash: the malformed scalar only blew up when used as an op ARG."""
    s = _scalar_two()
    arr = fnp.array([4.0, 4.0, 4.0])
    product = s * arr  # was a RemoteScalar wrapping a RemoteArray -> raw on the wire
    # Using `product` as an argument forces it through encode_request/msgpack.
    out = arr + product
    assert out.tolist() == [12.0, 12.0, 12.0]


def test_accumulation_pattern_from_prod():
    """``acc = acc + coef * tensor`` with coef a RemoteScalar (sub 309722 shape)."""
    acc = fnp.array([0.0, 0.0, 0.0])
    tensor = fnp.array([1.0, 2.0, 3.0])
    for coef_src in (fnp.array([0.5, 0.0])[0], fnp.array([2.0, 0.0])[0]):
        acc = acc + coef_src * tensor  # crashed here pre-fix
    assert acc.tolist() == [2.5, 5.0, 7.5]


def test_scalar_array_product_in_stack_list():
    """A malformed scalar inside a stack() list (sub 310080 shape)."""
    s = _scalar_two()
    a = fnp.array([1.0, 1.0])
    b = fnp.array([3.0, 3.0])
    stacked = fnp.stack([s * a, s + b])  # each element was a malformed scalar pre-fix
    assert stacked.tolist() == [[2.0, 2.0], [5.0, 5.0]]


def test_scalar_scalar_arithmetic_still_scalar():
    """Regression guard: scalar (op) scalar/number must stay a RemoteScalar."""
    s = _scalar_two()
    assert type(s * 3).__name__ == "RemoteScalar"
    assert type(s + s).__name__ == "RemoteScalar"
    assert float(s * 3) == 6.0
    assert float(s + s) == 4.0
