"""Every charged op carries a complex billing classification."""

import numpy as np
import pytest

from flopscope._registry import REGISTRY

CHARGED_CATEGORIES = {
    "counted_unary",
    "counted_binary",
    "counted_reduction",
    "counted_custom",
    "counted_random_method",
}
VALID_SPECIALS = {"exact", "illegal"}


def test_every_charged_op_is_classified():
    missing, invalid = [], []
    for name, entry in REGISTRY.items():
        if entry["category"] not in CHARGED_CATEGORIES:
            continue
        factor = entry.get("complex_factor")
        if factor is None:
            missing.append(name)
        elif not (
            factor in VALID_SPECIALS
            or (isinstance(factor, (int, float)) and float(factor) >= 1.0)
        ):
            invalid.append((name, factor))
    assert not missing, f"unclassified charged ops: {sorted(missing)}"
    assert not invalid, f"invalid complex_factor values: {sorted(invalid)}"


@pytest.mark.parametrize(
    "op,expected",
    [
        ("add", 2.0),
        ("multiply", 6.0),
        ("divide", 11.0),
        ("reciprocal", 6.0),
        ("absolute", 4.0),
        ("sqrt", 10.0),
        ("var", 2.5),
        ("conj", 1.0),
        ("angle", 1.0),
        ("sort_complex", 2.0),
        ("einsum", "exact"),
        ("matmul", "exact"),
        ("left_shift", "illegal"),
        ("floor", "illegal"),
    ],
)
def test_seed_classifications(op, expected):
    assert REGISTRY[op]["complex_factor"] == expected


@pytest.mark.parametrize("op", ["floor", "left_shift", "fmod"])
def test_illegal_ops_actually_raise_in_numpy(op):
    z = np.ones(2, dtype=np.complex128)
    with pytest.raises(TypeError):
        getattr(np, op)(z, z) if getattr(np, op).nin == 2 else getattr(np, op)(z)
