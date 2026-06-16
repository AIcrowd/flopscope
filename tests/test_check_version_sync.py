"""Tests for scripts/check_version_sync.py.

The drift-detection tests must be version-agnostic — they discover the
repo's current version at test time and inject a *different* version to
simulate drift. Hardcoding the current version (e.g. "0.3.0") would
silently break the tests every time `cz bump` ran.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_version_sync.py"

_PYPROJECT_VERSION_RE = re.compile(
    r'^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', re.MULTILINE
)


def _run(cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the version-sync script with cwd, return CompletedProcess."""
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def _current_version(repo: Path) -> str:
    """Read the X.Y.Z currently in the repo's root pyproject.toml."""
    text = (repo / "pyproject.toml").read_text()
    m = _PYPROJECT_VERSION_RE.search(text)
    assert m is not None, f"no version line found in {repo / 'pyproject.toml'}"
    return m.group(1)


def _other_version(version: str) -> str:
    """Return a different X.Y.Z guaranteed to differ from `version`.

    Bumps the patch digit by 99 — large enough to avoid colliding with
    any plausible nearby release, small enough to stay parseable.
    """
    major, minor, patch = version.split(".")
    return f"{major}.{minor}.{int(patch) + 99}"


def test_real_repo_is_in_sync():
    """The actual repo at HEAD must be in sync. If this fails, fix the drift."""
    result = _run(REPO_ROOT)
    assert result.returncode == 0, (
        f"check_version_sync failed on the real repo.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """Copy the version-bearing files into a temp dir mimicking repo structure."""
    dest = tmp_path / "repo"
    dest.mkdir()
    files = [
        "pyproject.toml",
        "src/flopscope/__init__.py",
        "flopscope-server/pyproject.toml",
        "flopscope-server/src/flopscope_server/__init__.py",
        "flopscope-client/pyproject.toml",
        "flopscope-client/src/flopscope/__init__.py",
    ]
    for rel in files:
        src = REPO_ROOT / rel
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    script_dst = dest / "scripts" / "check_version_sync.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SCRIPT, script_dst)
    return dest


def test_root_pyproject_drift_detected(repo_copy: Path):
    """Bumping root pyproject without bumping subpackages fails the check."""
    current = _current_version(repo_copy)
    other = _other_version(current)
    pp = repo_copy / "pyproject.toml"
    pp.write_text(
        pp.read_text().replace(f'version = "{current}"', f'version = "{other}"', 1)
    )
    result = _run(repo_copy)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert other in combined or current in combined


def test_init_file_drift_detected(repo_copy: Path):
    """Bumping only an __init__ file fails the check."""
    current = _current_version(repo_copy)
    other = _other_version(current)
    init = repo_copy / "flopscope-server" / "src" / "flopscope_server" / "__init__.py"
    init.write_text(init.read_text().replace(f'"{current}"', f'"{other}"'))
    result = _run(repo_copy)
    assert result.returncode != 0


def test_cross_pin_drift_detected(repo_copy: Path):
    """Server's flopscope==X.Y.Z pin must match root version."""
    current = _current_version(repo_copy)
    other = _other_version(current)
    pp = repo_copy / "flopscope-server" / "pyproject.toml"
    pp.write_text(
        pp.read_text().replace(f'"flopscope=={current}"', f'"flopscope=={other}"')
    )
    result = _run(repo_copy)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "flopscope==" in combined or "cross-pin" in combined.replace(" ", "")


def test_client_pyproject_drift_detected(repo_copy: Path):
    """Client's pyproject version drift is caught."""
    current = _current_version(repo_copy)
    other = _other_version(current)
    pp = repo_copy / "flopscope-client" / "pyproject.toml"
    pp.write_text(
        pp.read_text().replace(f'version = "{current}"', f'version = "{other}"', 1)
    )
    result = _run(repo_copy)
    assert result.returncode != 0


def test_server_extra_pin_drift_detected(repo_copy: Path):
    """Root [server] extra's flopscope-server== pin must match server version.

    This is the trap that silently broke `pip install "flopscope[server]==0.4.0"`
    between v0.3.0 and v0.4.0: the pin in the extra wasn't tracked by commitizen
    so it stayed at the old version after `cz bump`. v0.4.1 fixes the tracking
    via a `pyproject.toml:flopscope-server==` entry in `version_files`; this
    test guards against regression.
    """
    current = _current_version(repo_copy)
    other = _other_version(current)
    pp = repo_copy / "pyproject.toml"
    pp.write_text(
        pp.read_text().replace(
            f'"flopscope-server=={current}"', f'"flopscope-server=={other}"'
        )
    )
    result = _run(repo_copy)
    assert result.returncode != 0


_ALL_VERSION_FILES = [
    "pyproject.toml",
    "src/flopscope/__init__.py",
    "flopscope-server/pyproject.toml",
    "flopscope-server/src/flopscope_server/__init__.py",
    "flopscope-client/pyproject.toml",
    "flopscope-client/src/flopscope/__init__.py",
]


def _rewrite_all_versions(repo: Path, old: str, new: str) -> None:
    """Replace every occurrence of `old` with `new` across all version files."""
    for rel in _ALL_VERSION_FILES:
        p = repo / rel
        p.write_text(p.read_text().replace(old, new))


def _prerelease(version: str) -> str:
    """A realistic next-minor release candidate, e.g. 0.7.0 -> 0.8.0rc0."""
    major, minor = version.split(".")[:2]
    return f"{major}.{int(minor) + 1}.0rc0"


def test_prerelease_in_sync_passes(repo_copy: Path):
    """All eight locations at the same PEP 440 prerelease (e.g. 0.8.0rc0) is in sync.

    The pre-generalization regexes required `X.Y.Z` immediately followed by `"`,
    so a prerelease suffix made the pyproject/cross-pin reads raise. This guards
    that `cz bump --prerelease rc` versions validate cleanly.
    """
    current = _current_version(repo_copy)
    pre = _prerelease(current)
    _rewrite_all_versions(repo_copy, current, pre)
    result = _run(repo_copy)
    assert result.returncode == 0, (
        f"prerelease version {pre} should validate in sync.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert pre in result.stdout


def test_prerelease_drift_detected(repo_copy: Path):
    """rc0 vs rc1 must be caught — the check compares the full prerelease, not the X.Y.Z core."""
    current = _current_version(repo_copy)
    pre = _prerelease(current)  # e.g. 0.8.0rc0
    _rewrite_all_versions(repo_copy, current, pre)
    init = repo_copy / "flopscope-client" / "src" / "flopscope" / "__init__.py"
    init.write_text(init.read_text().replace(pre, pre[:-1] + "1"))  # rc0 -> rc1
    result = _run(repo_copy)
    assert result.returncode != 0, (
        f"rc0 vs rc1 drift should be detected.\nstdout:\n{result.stdout}"
    )
