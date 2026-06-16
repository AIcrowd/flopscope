# tests/test_docs_generation_hygiene.py
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "gen_api_docs", ROOT / "scripts" / "generate_api_docs.py"
)
assert (
    _spec is not None and _spec.loader is not None
)  # keep pyright happy (CI checks tests/)
gen = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gen
_spec.loader.exec_module(gen)


def test_generated_output_paths_lists_live_write_set():
    paths = {str(p) for p in gen.generated_output_paths()}
    # The two directories the live generator actually writes.
    assert any(p.endswith("website/.generated") for p in paths)
    assert any(p.endswith("website/public/api-data") for p in paths)
    # ops.json is the cost-model snapshot and stays TRACKED — never an "ignored output".
    assert not any(p.endswith("public/ops.json") for p in paths)
    # Hand-written pages must NOT be in the generated set (else we'd untrack them).
    assert not any(p.endswith("content/docs/api/index.mdx") for p in paths)
    assert not any(p.endswith("content/docs/api/numpy.mdx") for p in paths)


import subprocess


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True
    ).stdout


def test_guard_b_generated_paths_untracked_and_ignored():
    """No generator output may be tracked; each declared path must be gitignored."""
    for p in gen.generated_output_paths():
        rel = p.relative_to(ROOT)
        tracked = _git("ls-files", "--", str(rel)).strip()
        assert not tracked, f"generated path still tracked: {rel}\n{tracked[:300]}"
        # `git check-ignore` on a bare directory only matches a `dir/` rule when the
        # directory exists; probe a path UNDER directory outputs so the check is
        # robust in a fresh checkout (CI) where the dir has not been generated yet.
        probe = str(rel / "_probe") if p.suffix == "" else str(rel)
        rc = subprocess.run(["git", "check-ignore", "-q", probe], cwd=ROOT).returncode
        assert rc == 0, f"generated path not gitignored: {rel}"


def test_guard_a_generation_leaves_git_clean():
    """Regenerating must not dirty any TRACKED file EXCEPT website/public/ops.json.

    ops.json is the only tracked file the generator writes; its `summary` fields
    come from the installed numpy's docstrings and so vary across the numpy
    version matrix (guarded separately by `generate_api_docs.py --check`, which
    excludes summary). Every other generator output is gitignored, so regeneration
    must leave the rest of the tracked tree byte-clean.
    """
    subprocess.run(
        ["uv", "run", "python", "scripts/generate_api_docs.py"], cwd=ROOT, check=True
    )
    dirty = [
        ln
        for ln in _git("status", "--porcelain").splitlines()
        if ln.strip() and not ln.rstrip().endswith("website/public/ops.json")
    ]
    assert not dirty, "generation dirtied tracked files:\n" + "\n".join(dirty)
