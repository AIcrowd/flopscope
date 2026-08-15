"""Guard: every documented public-API name is either client-reachable, blacklisted,
or SERVER_ONLY.

This prevents the #129 bug class — a documented public-API op silently missing
from the client package — from recurring undetected.

Approach
--------
Client reachability is computed statically (importlib file-load of individual
client modules + AST analysis) rather than by importing the full client package.
This avoids the ``pyzmq`` / ``msgpack`` dependency that the client's
``__init__.py`` pulls in transitively, keeping the test runnable in the core
venv (no ZMQ socket needed).  The static model is conservative on the side of
marking things *reachable* (it counts every op in the auto-proxy registry plus
all explicit names from __init__.py), so a false-negative here is "we think it's
reachable but it isn't".  The integration test suite in
``flopscope-client/tests/`` verifies the runtime contract; this guard catches
*structural* omissions before they reach CI.

Classification tiers (A / B / C)
----------------------------------
A  Client-reachable  — getattr-walk from ``flopscope`` top-level succeeds.
B  ``blacklisted``   — registry ``category == "blacklisted"`` (intentionally
                        excluded from the client).
C  ``SERVER_ONLY``   — listed in ``src/flopscope/_server_only.py`` (in-process /
                        grader-analysis API, not exposed to participants via the
                        client).

Any name that is documented (in ``collect_public_api_surface_names()``) but falls
into none of A–C is a gap identical to the #129 ``symmetrize`` bug and must be
classified before merging.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TEST_DIR = Path(__file__).resolve().parent
ROOT = _TEST_DIR.parent
CLIENT_SRC = ROOT / "flopscope-client" / "src"
SCRIPTS_DIR = ROOT / "scripts"
CORE_SRC = ROOT / "src"

# ---------------------------------------------------------------------------
# Deferred follow-up: these names are documented but the client submodule
# that would expose them (flopscope.testing) is not yet implemented.
# TODO: remove when flopscope-client ships a `testing` submodule with
#       assert_allclose / assert_array_equal proxy stubs.
# ---------------------------------------------------------------------------
_KNOWN_FOLLOWUP: frozenset[str] = frozenset(
    {
        "testing.assert_allclose",
        "testing.assert_array_equal",
    }
)

# ---------------------------------------------------------------------------
# Helpers: load individual client files without zmq
# ---------------------------------------------------------------------------


def _load_client_file(rel: str, mod_name: str) -> types.ModuleType:
    """Load a single file from the client package by path, registering it in
    sys.modules under *mod_name* so repeated loads return the cached copy."""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = CLIENT_SRC / rel
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load client module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _ast_top_funcnames(src_text: str) -> set[str]:
    """Return names of public top-level functions defined in *src_text*."""
    tree = ast.parse(src_text)
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


def _ast_explicit_names(src_text: str) -> set[str]:
    """Return public names introduced by top-level imports or simple assignments."""
    tree = ast.parse(src_text)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                n = alias.asname or alias.name
                if not n.startswith("_"):
                    names.add(n)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.add(target.id)
    return names


def _ast_all_list(src_text: str) -> set[str]:
    """Return the names listed in a module-level ``__all__ = [...]`` literal."""
    tree = ast.parse(src_text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        return {
                            str(elt.value)
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        }
    return set()


# ---------------------------------------------------------------------------
# Static reachability model for the flopscope-client package
# ---------------------------------------------------------------------------
#
# The model is built once and cached.  It does NOT import the client package
# (no zmq needed); instead it:
#   • loads _registry_data.py (pure-Python, no external deps) to obtain
#     FUNCTION_CATEGORIES — the same registry the auto-proxy loop uses.
#   • parses each submodule's __init__.py with the ast module to find what
#     names they define / export.
#
# Conservative bias: we over-approximate reachability (a name is "reachable"
# if the client's code WOULD expose it, even before runtime confirms it).  A
# runtime false-negative ("we think it's there but it isn't") is caught by the
# integration tests in flopscope-client/tests/.


def _build_client_reachability() -> dict[str, bool]:
    """Return a mapping from public surface name -> reachable-on-client (bool).

    Uses static analysis only — no ZMQ socket, no network call.
    """
    # ---- 1. Load client registry data (FUNCTION_CATEGORIES) ----
    rd = _load_client_file("flopscope/_registry_data.py", "_parity_cr_registry_data")
    fc: dict[str, str] = rd.FUNCTION_CATEGORIES  # name -> category

    # ---- 2. Top-level client surface ----
    # Special-cased ops defined explicitly in __init__.py (not auto-generated).
    _SPECIAL_CASED = frozenset(
        {"array", "asarray", "einsum", "load", "save", "savez", "savez_compressed"}
    )
    # Auto-proxy loop: for every non-blacklisted, non-dotted op in the registry
    # that is not special-cased, the client __init__.py creates a proxy function
    # with globals()[op_name] = _make_proxy(op_name).
    auto_proxied_top = {
        k
        for k, v in fc.items()
        if v != "blacklisted" and "." not in k and k not in _SPECIAL_CASED
    }
    # Explicit imports and assignments in __init__.py
    init_explicit = _ast_explicit_names(
        (CLIENT_SRC / "flopscope/__init__.py").read_text()
    )
    client_top = auto_proxied_top | init_explicit | _SPECIAL_CASED

    # ---- 3. Submodule surfaces ----
    # random: auto-proxied random.X ops (non-blacklisted, no nested dots) plus
    # the three RNG class proxies.
    client_random: set[str] = {
        k[len("random.") :]
        for k, v in fc.items()
        if k.startswith("random.")
        and v != "blacklisted"
        and "." not in k[len("random.") :]
    } | {"Generator", "RandomState", "SeedSequence"}

    # fft: auto-proxied fft.X ops
    client_fft: set[str] = {
        k[len("fft.") :]
        for k, v in fc.items()
        if k.startswith("fft.") and v != "blacklisted"
    }

    # linalg: auto-proxied linalg.X ops
    client_linalg: set[str] = {
        k[len("linalg.") :]
        for k, v in fc.items()
        if k.startswith("linalg.") and v != "blacklisted"
    }

    # stats: the __all__ list in stats/__init__.py is the authoritative export.
    client_stats = _ast_all_list(
        (CLIENT_SRC / "flopscope/stats/__init__.py").read_text()
    )

    # flops: top-level function definitions in flops.py (the server-proxied /
    # locally-computed cost helpers).  __getattr__ raises AttributeError for
    # everything else, so only explicitly-defined functions count.
    client_flops = _ast_top_funcnames((CLIENT_SRC / "flopscope/flops.py").read_text())

    # testing: no testing submodule in the client yet (see _KNOWN_FOLLOWUP).
    client_testing: set[str] = set()

    # ---- 4. Build lookup ----
    # Load generate_api_docs via importlib so pyright can resolve the import
    # (sys.path mutation is not visible to the type-checker).
    _gad_spec = importlib.util.spec_from_file_location(
        "generate_api_docs", SCRIPTS_DIR / "generate_api_docs.py"
    )
    assert _gad_spec is not None and _gad_spec.loader is not None
    _gad_mod = importlib.util.module_from_spec(_gad_spec)
    sys.modules[_gad_spec.name] = _gad_mod
    _gad_spec.loader.exec_module(_gad_mod)  # type: ignore[union-attr]
    collect_public_api_surface_names = _gad_mod.collect_public_api_surface_names

    surface = collect_public_api_surface_names()

    sub_lookup: dict[str, set[str]] = {
        "random": client_random,
        "fft": client_fft,
        "linalg": client_linalg,
        "stats": client_stats,
        "flops": client_flops,
        "testing": client_testing,
    }

    result: dict[str, bool] = {}
    for name in surface:
        parts = name.split(".", 1)
        top = parts[0]
        if len(parts) == 1:
            result[name] = top in client_top
        else:
            sub_name = parts[1]
            result[name] = sub_name in sub_lookup.get(top, set())

    return result


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _get_blacklisted() -> frozenset[str]:
    """Names with category=='blacklisted' in the core REGISTRY."""
    if str(CORE_SRC) not in sys.path:
        sys.path.insert(0, str(CORE_SRC))
    from flopscope._registry import REGISTRY

    return frozenset(
        n for n, e in REGISTRY.items() if e.get("category") == "blacklisted"
    )


def _get_server_only() -> frozenset[str]:
    """Names in SERVER_ONLY from the core package."""
    if str(CORE_SRC) not in sys.path:
        sys.path.insert(0, str(CORE_SRC))
    from flopscope._server_only import SERVER_ONLY

    return SERVER_ONLY


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _unclassified(
    surface: dict[str, bool],
    blacklisted: frozenset[str],
    server_only: frozenset[str],
) -> list[str]:
    """Return names that are not client-reachable, not blacklisted, not
    SERVER_ONLY, and not in _KNOWN_FOLLOWUP — i.e. unclassified gaps.

    Both the main guard test and the guard self-check call this function so
    that the classification logic lives in exactly one place.
    """
    return [
        name
        for name, reachable in sorted(surface.items())
        if not reachable
        and name not in blacklisted
        and name not in server_only
        and name not in _KNOWN_FOLLOWUP
    ]


# ---------------------------------------------------------------------------
# The guard test
# ---------------------------------------------------------------------------


def test_every_documented_name_is_classified():
    """Every name in the documented public API must fall into tier A, B, or C.

    Tier A — client-reachable (getattr-walk from ``flopscope`` top-level).
    Tier B — ``blacklisted`` in the registry (intentionally excluded).
    Tier C — in ``SERVER_ONLY`` (server-side / analysis API).

    If this test fails, each listed name must be intentionally classified
    before the PR merges — do NOT silently dump names into ``_KNOWN_FOLLOWUP``
    to force green.  See the module docstring for the classification decision
    tree.
    """
    reachability = _build_client_reachability()
    blacklisted = _get_blacklisted()
    server_only = _get_server_only()

    unclassified = _unclassified(reachability, blacklisted, server_only)

    assert unclassified == [], (
        "The following documented public-API names are neither client-reachable "
        "(tier A), blacklisted (tier B), SERVER_ONLY (tier C), nor in "
        "_KNOWN_FOLLOWUP:\n"
        + "\n".join(f"  {n}" for n in unclassified)
        + "\n\nFor each name, decide:\n"
        "  • participant-facing op that should be client-reachable → that's a "
        "bug (e.g. #129); report it and add the client proxy.\n"
        "  • server-side tooling → add to src/flopscope/_server_only.py and "
        "re-run scripts/sync_client.py.\n"
        "  • numpy-meaningless op → blacklist in the registry and re-sync.\n"
        "  • genuinely deferred follow-up → add to _KNOWN_FOLLOWUP with a "
        "TODO comment (keep this set MINIMAL)."
    )


def test_known_followup_names_are_actually_absent():
    """Ensure _KNOWN_FOLLOWUP names are NOT accidentally becoming reachable.

    If a name in _KNOWN_FOLLOWUP *is* now reachable on the client, it should
    be removed from _KNOWN_FOLLOWUP so the guard stays tight.
    """
    reachability = _build_client_reachability()
    blacklisted = _get_blacklisted()
    server_only = _get_server_only()

    # Names in _KNOWN_FOLLOWUP should remain unclassified (absent from client).
    # If they've been classified, that's good news — remove them from
    # _KNOWN_FOLLOWUP and let the main test verify them normally.
    became_reachable = [
        name
        for name in _KNOWN_FOLLOWUP
        if reachability.get(name, False) or name in blacklisted or name in server_only
    ]
    assert became_reachable == [], (
        "_KNOWN_FOLLOWUP entries that are now classified (remove them from "
        "_KNOWN_FOLLOWUP):\n" + "\n".join(f"  {n}" for n in became_reachable)
    )


def test_guard_flags_an_unclassified_name():
    """Negative self-check: the guard's _unclassified() helper must flag a
    synthetic name that is none of tier A / B / C / _KNOWN_FOLLOWUP.

    This proves the detection logic is live — a future refactor that
    accidentally neuters the guard (e.g. by always returning an empty list)
    will cause THIS test to fail rather than silently passing everything.

    The fake name is chosen to be provably absent from every classification
    tier so the test is deterministic and does not rely on any real op name.
    """
    FAKE = "definitely_not_a_real_flopscope_symbol_xyz"

    # Build a minimal surface that contains only the fake name, marked as
    # not-reachable on the client.
    fake_surface: dict[str, bool] = {FAKE: False}

    # The fake name must not accidentally land in any real classification set.
    blacklisted = _get_blacklisted()
    server_only = _get_server_only()
    assert FAKE not in blacklisted, f"{FAKE!r} unexpectedly in blacklisted"
    assert FAKE not in server_only, f"{FAKE!r} unexpectedly in SERVER_ONLY"
    assert FAKE not in _KNOWN_FOLLOWUP, f"{FAKE!r} unexpectedly in _KNOWN_FOLLOWUP"

    # The shared helper must flag the fake name as unclassified.
    gaps = _unclassified(fake_surface, blacklisted, server_only)
    assert FAKE in gaps, (
        f"_unclassified() did not flag {FAKE!r} — the guard's detection logic "
        "appears to be broken (it should have returned this name as an unclassified gap)."
    )
