"""Tests for the flopscope numpy function registry."""

# pyright: reportMissingImports=false
from flopscope._registry import REGISTRY, REGISTRY_META


def test_registry_meta_has_numpy_version():
    assert "numpy_version" in REGISTRY_META
    parts = REGISTRY_META["numpy_version"].split(".")
    assert len(parts) >= 2


def test_registry_meta_has_last_updated():
    assert "last_updated" in REGISTRY_META


def test_all_entries_have_required_fields():
    for name, entry in REGISTRY.items():
        assert "category" in entry, f"{name} missing 'category'"
        assert "module" in entry, f"{name} missing 'module'"
        assert "notes" in entry, f"{name} missing 'notes'"
        assert entry["category"] in (
            "counted_unary",
            "counted_binary",
            "counted_reduction",
            "counted_custom",
            "counted_random_method",
            "free",
            "free_random_method",
            "blacklisted",
            "unclassified",
        ), f"{name} has invalid category: {entry['category']}"


def test_existing_counted_unary_ops():
    expected = [
        "exp",
        "log",
        "log2",
        "log10",
        "abs",
        "negative",
        "sqrt",
        "square",
        "sin",
        "cos",
        "tanh",
        "sign",
        "ceil",
        "floor",
    ]
    for name in expected:
        assert name in REGISTRY, f"{name} not in registry"
        assert REGISTRY[name]["category"] == "counted_unary"


def test_existing_counted_binary_ops():
    expected = [
        "add",
        "subtract",
        "multiply",
        "divide",
        "maximum",
        "minimum",
        "power",
        "mod",
    ]
    for name in expected:
        assert name in REGISTRY, f"{name} not in registry"
        assert REGISTRY[name]["category"] == "counted_binary"


def test_existing_counted_reduction_ops():
    expected = [
        "sum",
        "max",
        "min",
        "prod",
        "mean",
        "std",
        "var",
        "argmax",
        "argmin",
        "cumsum",
        "cumprod",
    ]
    for name in expected:
        assert name in REGISTRY, f"{name} not in registry"
        assert REGISTRY[name]["category"] == "counted_reduction"


def test_existing_array_ops():
    # Some former free ops are now counted_custom (budget deduction added)
    still_free = [
        "zeros",
        "transpose",
    ]
    now_counted = [
        "concatenate",
        "stack",
        "diag",
        "arange",
        "linspace",
        "where",
        "pad",
        "isnan",
        "isinf",
        "isfinite",
        "triu",
        "tril",
        "hstack",
        "column_stack",
        "row_stack",
        # Task 4: value-writing creation & layout copies
        "ones",
        "ones_like",
        "eye",
        "identity",
        "copy",
        "reshape",
        "ravel",
        "require",
        "fft.fftshift",
        "fft.ifftshift",
    ]
    for name in still_free:
        assert name in REGISTRY, f"{name} not in registry"
        assert REGISTRY[name]["category"] == "free"
    for name in now_counted:
        assert name in REGISTRY, f"{name} not in registry"
        assert REGISTRY[name]["category"] == "counted_custom"


def test_existing_custom_ops():
    expected = ["dot", "matmul", "einsum", "clip"]
    for name in expected:
        assert name in REGISTRY, f"{name} not in registry"
        assert REGISTRY[name]["category"] == "counted_custom"


def test_linalg_svd():
    assert "linalg.svd" in REGISTRY
    assert REGISTRY["linalg.svd"]["category"] == "counted_custom"


def test_no_unclassified():
    unclassified = [n for n, e in REGISTRY.items() if e["category"] == "unclassified"]
    assert len(unclassified) == 0, f"Unclassified: {unclassified}"


def test_all_numpy_functions_in_registry():
    """Every function discovered by the audit should be in the registry."""
    import sys

    sys.path.insert(0, "scripts")
    from numpy_audit import introspect_numpy

    discovered = introspect_numpy()
    missing = [name for name in discovered if name not in REGISTRY]
    assert len(missing) == 0, (
        f"Missing from registry ({len(missing)}): {missing[:20]}..."
    )
