#!/usr/bin/env python
"""numpy_audit.py — Introspect NumPy and compare against the flopscope registry.

Usage
-----
    python scripts/numpy_audit.py                  # rich table (default)
    python scripts/numpy_audit.py --json           # JSON output
    python scripts/numpy_audit.py --ci             # plain text, non-zero exit if uncovered
    python scripts/numpy_audit.py --filter unclassified
    python scripts/numpy_audit.py --module linalg
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import types
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Names that are intentionally excluded from the audit.
# These are either manually handled in flopscope, are low-level C types,
# deprecated helpers, or class constructors that don't represent mathematical
# operations we need to FLOP-count.
# ---------------------------------------------------------------------------
SKIP_NAMES: frozenset[str] = frozenset(
    [
        # --- array construction (manually handled in flopscope) ---
        "array",
        "asarray",
        "asanyarray",
        # --- decided gaps: NOT handled, and skipped so the audit does not
        # --- re-report them. flopscope does not implement these at all (they
        # --- raise AttributeError) and they are deliberately absent from the
        # --- weight-0 free tier: billing a layout change at 0 would be a free
        # --- data-movement path. Implementing them is a repricing decision, so
        # --- the suppression stays until that decision is made.
        "ascontiguousarray",
        "asfortranarray",
        # --- dtype / type-system ---
        "dtype",
        "finfo",
        "iinfo",
        "errstate",
        # --- scalar dtype classes (concrete) ---
        "bool_",
        "byte",
        "ubyte",
        "short",
        "ushort",
        "intc",
        "uintc",
        "int_",
        "uint",
        "longlong",
        "ulonglong",
        "intp",
        "uintp",
        "half",
        "single",
        "double",
        "float_",
        "longdouble",
        "longfloat",
        "csingle",
        "singlecomplex",
        "cdouble",
        "cfloat",
        "complex_",
        "clongdouble",
        "clongfloat",
        "longcomplex",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "float128",  # may not exist on all platforms
        "complex64",
        "complex128",
        "complex256",  # may not exist on all platforms
        "bytes_",
        "str_",
        "string_",
        "unicode_",
        "void",
        "object_",
        "datetime64",
        "timedelta64",
        "character",
        "flexible",
        "generic",
        "number",
        "inexact",
        "integer",
        "signedinteger",
        "unsignedinteger",
        "floating",
        "complexfloating",
        "True_",
        "False_",
        # --- abstract base types / meta ---
        "ufunc",
        "flatiter",
        # --- array containers / iterators ---
        "ndarray",
        "nditer",
        "ndenumerate",
        "ndindex",
        "matrix",
        "memmap",
        "chararray",
        "recarray",
        "record",
        "broadcast",
        "busdaycalendar",
        # --- polynomial / other class wrappers ---
        "vectorize",
        "poly1d",
        # --- printing / config ---
        "set_printoptions",
        "get_printoptions",
        "printoptions",
        "format_parser",
        # --- exception / warning classes ---
        "AxisError",
        "ComplexWarning",
        "ModuleDeprecationWarning",
        "RankWarning",
        "TooHardError",
        "VisibleDeprecationWarning",
        "LinAlgError",
        # --- utility/test classes ---
        "DataSource",
        "Tester",
        # --- internal / deprecated callables ---
        "add_docstring",
        "add_newdoc",
        "add_newdoc_ufunc",
        "lookfor",
        "who",
        "disp",
        "info",
        "source",
        "pkgload",
        "test",
        # --- constants / non-callables that might slip through ---
        "little_endian",
        "ALLOW_THREADS",
        "BUFSIZE",
        "CLIP",
        "ERR_CALL",
        "ERR_DEFAULT",
        "ERR_IGNORE",
        "ERR_LOG",
        "ERR_PRINT",
        "ERR_RAISE",
        "ERR_WARN",
        "FLOATING_POINT_SUPPORT",
        "FPE_DIVIDEBYZERO",
        "FPE_INVALID",
        "FPE_OVERFLOW",
        "FPE_UNDERFLOW",
        "MAXDIMS",
        "NaN",
        "Inf",
        "Infinity",
        "NAN",
        "NINF",
        "NZERO",
        "PINF",
        "PZERO",
        "pi",
        "e",
        "inf",
        "nan",
        "newaxis",
        "RAISE",
        "WRAP",
        "ScalarType",
        "typecodes",
        "sctypes",
        "sctypeDict",
        "nbytes",
        "cast",
        "w",
        "UFUNC_PYVALS_NAME",
        "error_message",
        "numarray",
        "oldnumeric",
        "c_",
        "r_",
        "s_",
        "index_exp",
        "mgrid",
        "ogrid",
        # --- random module classes (not functions to wrap) ---
        "BitGenerator",
        "Generator",
        "MT19937",
        "PCG64",
        "PCG64DXSM",
        "Philox",
        "RandomState",
        "SFC64",
        "SeedSequence",
        "mtrand",
        "bit_generator",
        "get_bit_generator",
        "set_bit_generator",
        # --- fft helper module ---
        "helper",
        # --- linalg self-reference ---
        "linalg",
    ]
)

# Submodules to walk (in addition to top-level numpy)
SUBMODULES = {
    "linalg": np.linalg,
    "fft": np.fft,
    "random": np.random,
}

# Submodules to never include (walk targets only)
EXCLUDED_SUBMODULE_PREFIXES = (
    "testing",
    "lib",
    "compat",
    "ctypeslib",
    "ma",
    "char",
    "emath",
    "core",
    "rec",
)


def _classify_kind(obj) -> str:
    """Return a human-readable kind string for a callable object."""
    if isinstance(obj, np.ufunc):
        return "ufunc"
    t = type(obj)
    tname = t.__name__
    if tname == "builtin_function_or_method":
        return "builtin"
    if tname == "function":
        return "function"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "callable"
    return "unknown"


def _should_skip(name: str, obj) -> bool:
    """Return True if this name/object should be excluded from the audit."""
    # Skip private names
    if name.startswith("_"):
        return True
    # Skip explicitly listed names
    if name in SKIP_NAMES:
        return True
    # Skip modules
    if isinstance(obj, types.ModuleType):
        return True
    # Skip non-callables
    if not callable(obj):
        return True
    # Skip type/class objects that are dtype or exception/warning types
    if inspect.isclass(obj):
        # Keep only classes that are proper callable wrappers (none in our list)
        return True
    return False


def introspect_numpy() -> dict[str, dict]:
    """Walk numpy and its submodules to discover all public callables.

    Returns
    -------
    dict
        Maps qualified name (e.g. ``"exp"``, ``"linalg.svd"``) to a metadata
        dict with keys ``"module"`` (str) and ``"kind"`` (str).
    """
    discovered: dict[str, dict] = {}

    # Suppress numpy FutureWarning / DeprecationWarning for deprecated aliases
    # (e.g. np.bytes, np.int, np.float) that trigger on attribute access.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (FutureWarning, DeprecationWarning))

        # --- top-level numpy ---
        for name in dir(np):
            if name.startswith("_"):
                continue
            obj = getattr(np, name, None)
            if obj is None:
                continue
            if _should_skip(name, obj):
                continue
            kind = _classify_kind(obj)
            discovered[name] = {"module": "numpy", "kind": kind}

        # --- submodules ---
        for prefix, mod in SUBMODULES.items():
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                if name in SKIP_NAMES:
                    continue
                obj = getattr(mod, name, None)
                if obj is None:
                    continue
                if _should_skip(name, obj):
                    continue
                # Skip if this is the same object as a top-level numpy function
                # (i.e., it's already covered at the top level)
                top_obj = getattr(np, name, None)
                if top_obj is obj:
                    continue
                qualified = f"{prefix}.{name}"
                kind = _classify_kind(obj)
                discovered[qualified] = {"module": f"numpy.{prefix}", "kind": kind}

    return discovered


def load_registry() -> tuple[dict, dict]:
    """Load the flopscope registry.

    Returns
    -------
    (REGISTRY_META, REGISTRY)
        The registry metadata and the full registry dict.
        Returns ``({}, {})`` if the registry does not exist yet.
    """
    try:
        from flopscope._registry import REGISTRY, REGISTRY_META

        return REGISTRY_META, REGISTRY
    except ImportError:
        return {}, {}


# Registry keys implemented as syntax rather than a module attribute — e.g.
# ``getitem`` bills ``arr[key]`` via ``FlopscopeArray.__getitem__``, so
# ``getattr(flopscope.numpy, "getitem")`` raises even though it is implemented.
# Mirrors ``scripts/generate_api_docs.py``'s ``METHOD_SURFACE_OPS``.
_METHOD_SURFACE_OPS = {"getitem"}


def compare(
    discovered: dict[str, dict],
    registry: dict,
) -> dict[str, list]:
    """Compare discovered numpy callables against the flopscope registry.

    Parameters
    ----------
    discovered:
        Output of :func:`introspect_numpy`.
    registry:
        ``REGISTRY`` dict from :func:`load_registry` (second element).

    Returns
    -------
    dict with keys:
        - ``covered`` — in registry and importable from flopscope
        - ``registered_not_implemented`` — in registry but not importable
        - ``unclassified`` — discovered but not in registry at all
        - ``blacklisted`` — in registry with category 'blacklisted'
        - ``stale`` — registered, absent from numpy, and not implemented
        - ``flopscope_native`` — registered + implemented but outside numpy's
          own surface (so never "discovered"): scipy-stats mirrors (``stats.*``),
          flopscope extensions like the symmetric API, and method-surface keys
          like ``getitem``. NOT stale — these are implemented on purpose.
    """
    # Determine which registered functions are actually importable. flopscope's
    # numpy-compatible surface lives under ``flopscope.numpy`` (e.g.
    # ``flopscope.numpy.sort``), the stats surface under ``flopscope.stats``,
    # and a few native ops (the symmetric API, registry ``module ==
    # "flopscope"``) at the top-level ``flopscope`` namespace.
    import flopscope as we
    import flopscope.numpy as fnp
    import flopscope.stats as fstats

    implemented_names = set()
    for name in registry:
        if name in _METHOD_SURFACE_OPS:
            implemented_names.add(name)
            continue
        if name.startswith("stats."):
            root, parts = fstats, name[len("stats.") :].split(".")
        elif registry[name].get("module") == "flopscope":
            root, parts = we, name.split(".")
        else:
            root, parts = fnp, name.split(".")
        try:
            obj = root
            for part in parts:
                obj = getattr(obj, part)
            implemented_names.add(name)
        except (AttributeError, TypeError):
            pass

    result: dict[str, list] = {
        "covered": [],
        "registered_not_implemented": [],
        "unclassified": [],
        "blacklisted": [],
        "stale": [],
        "flopscope_native": [],
    }

    for name in sorted(discovered):
        if name in registry:
            entry = registry[name]
            if entry["category"] == "blacklisted":
                result["blacklisted"].append(name)
            elif name in implemented_names:
                result["covered"].append(name)
            else:
                result["registered_not_implemented"].append(name)
        else:
            result["unclassified"].append(name)

    # Registry ops with no numpy counterpart: distinguish flopscope-native
    # extensions / method-surface keys (implemented on purpose, not in numpy)
    # from genuinely stale entries (registered but neither in numpy nor
    # implemented). Blacklisted ops belong in their own bucket regardless.
    for name in sorted(registry):
        if name in discovered:
            continue
        entry = registry[name]
        if entry.get("category") == "blacklisted":
            if name not in result["blacklisted"]:
                result["blacklisted"].append(name)
        elif name in implemented_names:
            result["flopscope_native"].append(name)
        else:
            result["stale"].append(name)

    return result


def audit_random_class(cls_name: str, cls: type, op_prefix: str) -> list[str]:
    """Diff numpy class method set against the registry slice for that class.

    Returns an empty list when the registry is in sync with numpy. Otherwise
    returns lines of the form ``+ Generator.foo`` (numpy added a method)
    or ``- Generator.bar`` (registry has an entry numpy no longer exposes).
    """
    from flopscope._registry import REGISTRY

    registry_methods = {
        n[len(op_prefix) :]
        for n, e in REGISTRY.items()
        if n.startswith(op_prefix)
        and e.get("category") in {"counted_random_method", "free_random_method"}
    }
    numpy_methods = {n for n in dir(cls) if not n.startswith("_")}

    added = numpy_methods - registry_methods
    removed = registry_methods - numpy_methods
    return [f"+ {cls_name}.{m}" for m in sorted(added)] + [
        f"- {cls_name}.{m}" for m in sorted(removed)
    ]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _category_color(category: str) -> str:
    return {
        "covered": "green",
        "unclassified": "yellow",
        "blacklisted": "dim",
        "stale": "red",
        "registered_not_implemented": "cyan",
        "flopscope_native": "blue",
    }.get(category, "white")


def print_rich_report(
    discovered: dict[str, dict],
    comparison: dict[str, list],
    registry: dict,
    filter_category: str | None = None,
    filter_module: str | None = None,
) -> None:
    """Print a colour-coded rich table to stdout."""
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print_plain_report(
            discovered, comparison, registry, filter_category, filter_module
        )
        return

    console = Console()

    # Build a flat list of (name, category)
    rows: list[tuple[str, str]] = []
    for cat, names in comparison.items():
        for name in names:
            rows.append((name, cat))
    rows.sort(key=lambda x: x[0])

    if filter_category:
        rows = [(n, c) for n, c in rows if c == filter_category]
    if filter_module:
        rows = [
            (n, c)
            for n, c in rows
            if (
                n.startswith(filter_module + ".")
                or (filter_module == "numpy" and "." not in n)
            )
        ]

    table = Table(
        title="flopscope / NumPy coverage audit",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        expand=True,
    )
    table.add_column("Function", style="bold", no_wrap=True)
    table.add_column("Module")
    table.add_column("Kind")
    table.add_column("Status", no_wrap=True)
    table.add_column("Notes", ratio=2)

    for name, cat in rows:
        meta = discovered.get(name, {})
        reg_entry = registry.get(name, {})
        color = _category_color(cat)
        notes = reg_entry.get("notes", "")
        if len(notes) > 50:
            notes = notes[:47] + "..."
        table.add_row(
            f"[{color}]{name}[/{color}]",
            meta.get("module", "—"),
            meta.get("kind", "—"),
            f"[{color}]{cat}[/{color}]",
            notes,
        )

    console.print(table)

    # Summary
    console.print()
    console.print("[bold]Summary[/bold]")
    for cat, names in comparison.items():
        color = _category_color(cat)
        console.print(f"  [{color}]{cat}[/{color}]: {len(names)}")
    console.print(f"\n  [bold]Total discovered[/bold]: {len(discovered)}")


def print_plain_report(
    discovered: dict[str, dict],
    comparison: dict[str, list],
    registry: dict,
    filter_category: str | None = None,
    filter_module: str | None = None,
) -> None:
    """Print a plain-text report suitable for CI logs."""
    print("=" * 60)
    print("flopscope / NumPy coverage audit")
    print("=" * 60)

    for cat, names in comparison.items():
        if filter_category and cat != filter_category:
            continue
        filtered = names
        if filter_module:
            filtered = [
                n
                for n in names
                if (
                    n.startswith(filter_module + ".")
                    or (filter_module == "numpy" and "." not in n)
                )
            ]
        print(f"\n--- {cat.upper()} ({len(filtered)}) ---")
        for name in filtered:
            meta = discovered.get(name, {})
            reg_entry = registry.get(name, {})
            notes = reg_entry.get("notes", "")
            if len(notes) > 50:
                notes = notes[:47] + "..."
            print(f"  {name:40s}  {meta.get('module', ''):16s}  {notes}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for cat, names in comparison.items():
        print(f"  {cat}: {len(names)}")
    print(f"  Total discovered: {len(discovered)}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit NumPy callables against the flopscope registry."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full audit result as JSON.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Plain-text output; exit with non-zero status if there are unclassified functions.",
    )
    parser.add_argument(
        "--filter",
        metavar="CATEGORY",
        dest="filter_category",
        help="Show only entries in this category (covered, unclassified, blacklisted, stale, registered_not_implemented).",
    )
    parser.add_argument(
        "--module",
        metavar="NAME",
        dest="filter_module",
        help="Show only entries from this module (numpy, linalg, fft, random).",
    )
    args = parser.parse_args()

    discovered = introspect_numpy()
    registry_meta, registry = load_registry()
    comparison = compare(discovered, registry)

    if args.json:
        output = {
            "discovered": discovered,
            "comparison": comparison,
        }
        print(json.dumps(output, indent=2))
        return

    # Issue #18: ensure Generator / RandomState method coverage stays in sync.
    random_diffs = audit_random_class(
        "Generator", np.random.Generator, "random.Generator."
    ) + audit_random_class("RandomState", np.random.RandomState, "random.RandomState.")
    if random_diffs:
        print("\n=== random class method drift ===", file=sys.stderr)
        for line in random_diffs:
            print(line, file=sys.stderr)
        if args.ci:
            sys.exit(1)

    if args.ci:
        print_plain_report(
            discovered,
            comparison,
            registry,
            filter_category=args.filter_category,
            filter_module=args.filter_module,
        )
        if comparison["unclassified"]:
            sys.exit(1)
        return

    print_rich_report(
        discovered,
        comparison,
        registry,
        filter_category=args.filter_category,
        filter_module=args.filter_module,
    )


if __name__ == "__main__":
    main()
