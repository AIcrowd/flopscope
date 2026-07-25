"""Source 2: exotic argument values crossed with argument POSITION.

Position matters as much as type: the same value behaves differently as a
positional argument, a keyword, a list element, an index key and a slice bound.
"""

from __future__ import annotations

from tests.parity.case import Case

_NP = frozenset({"requires:numpy"})
_NONE: frozenset[str] = frozenset()

#: (name, expression producing the value, extra tags)
VALUES: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("complex", "1j", _NONE),
    ("range", "range(2)", _NONE),
    ("set", "{1, 2}", _NONE),
    ("frozenset", "frozenset({1, 2})", _NONE),
    ("generator", "(x for x in [1, 2])", _NONE),
    ("memoryview", "memoryview(b'\\x01\\x02')", _NONE),
    ("bytearray", "bytearray(b'\\x01\\x02')", _NONE),
    ("bytes", "b'\\x01\\x02'", _NONE),
    ("array-array-f", "__import__('array').array('f', [1.0, 2.0])", _NONE),
    ("decimal", "__import__('decimal').Decimal('1.5')", _NONE),
    ("fraction", "__import__('fractions').Fraction(1, 2)", _NONE),
    ("datetime", "__import__('datetime').date(2026, 7, 25)", _NONE),
    ("slice-object", "slice(0, 2)", _NONE),
    ("ellipsis", "...", _NONE),
    ("int-enum", "__import__('enum').IntEnum('E', {'A': 1}).A", _NONE),
    ("huge-int", "2**70", _NONE),
    ("int64-max", "2**63 - 1", _NONE),
    ("uint64-max", "2**64 - 1", _NONE),
    ("int64-min", "-(2**63)", _NONE),
    ("int64-min-minus-one", "-(2**63) - 1", _NONE),
    ("huge-negative-int", "-(2**70)", _NONE),
    ("nested-list", "[[[1.0]]]", _NONE),
    ("dict", "{'k': 1}", _NONE),
    ("dict-with-handle", "{'k': V}", _NONE),
    ("handle-lookalike", "'a0'", _NONE),
    ("dtype-named-object", "type('S', (), {'name': 'float32'})()", _NONE),
    ("remote-array", "V", _NONE),
    ("remote-scalar", "V[0]", _NONE),
    ("np-float32", "__import__('numpy').float32(1.5)", _NP),
    ("np-int64", "__import__('numpy').int64(3)", _NP),
    ("np-bool", "__import__('numpy').bool_(True)", _NP),
    ("np-float16", "__import__('numpy').float16(1.5)", _NP),
    ("np-complex64", "__import__('numpy').complex64(1 + 2j)", _NP),
    ("np-complex128", "__import__('numpy').complex128(1 + 2j)", _NP),
    ("np-ndarray-1d", "__import__('numpy').array([1.0, 2.0])", _NP),
    ("np-ndarray-0d", "__import__('numpy').array(1.0)", _NP),
)

#: (name, template containing {value})
POSITIONS: tuple[tuple[str, str], ...] = (
    ("positional", "fnp.multiply(V, {value})"),
    ("keyword", "fnp.clip(V, a_min=0.0, a_max={value})"),
    ("list-element", "fnp.concatenate([V, {value}])"),
    ("index-key", "V[{value}]"),
    ("slice-bound", "V[:{value}]"),
    ("second-positional", "fnp.where(M, {value}, 0.0)"),
    ("dict-literal", "fnp.multiply(V, {{'k': {value}}})"),
    ("constructor", "fnp.asarray({value})"),
)


def build() -> tuple[Case, ...]:
    return tuple(
        Case(
            id=f"types/{value_name}::{position_name}",
            source=template.format(value=value_source),
            tags=frozenset(
                {"src:types", f"value:{value_name}", f"position:{position_name}"}
            )
            | extra,
        )
        for value_name, value_source, extra in VALUES
        for position_name, template in POSITIONS
    )
