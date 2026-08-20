from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from email import message_from_bytes
from email.message import Message
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/verify_budget_summary_release.py"
)


def _verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_budget_summary_release", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(wheel: Path) -> Message:
    import zipfile

    with zipfile.ZipFile(wheel) as archive:
        names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(names) == 1, f"expected one METADATA member in {wheel}: {names}"
        return message_from_bytes(archive.read(names[0]))


def _one_wheel(root: Path, distribution: str) -> Path:
    normalized = distribution.replace("-", "_")
    matches = sorted(root.rglob(f"{normalized}-*.whl"))
    assert len(matches) == 1, (
        f"expected one {distribution} wheel under {root}: {matches}"
    )
    return matches[0]


def _run(
    *args: str, env: dict[str, str] | None = None, cwd: Path | None = None
) -> None:
    subprocess.run(args, check=True, env=env, cwd=cwd)


def test_exact_dependency_rejects_postrelease_false_positive() -> None:
    verifier = _verifier()

    with pytest.raises(AssertionError, match="exactly flopscope-server==0.10.0"):
        verifier.require_exact_dependency(
            ["flopscope-server==0.10.0.post1 ; extra == 'server'"],
            distribution="flopscope-server",
            version="0.10.0",
            required_extra="server",
        )


def test_control_token_read_has_a_deadline() -> None:
    verifier = _verifier()
    read_fd, write_fd = os.pipe()

    class LiveProcess:
        stderr = None

        @staticmethod
        def poll() -> None:
            return None

    try:
        with pytest.raises(AssertionError, match="timed out waiting for control token"):
            verifier.read_control_token(LiveProcess(), read_fd, timeout_s=0.01)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_server_cleans_resources_when_popen_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verifier = _verifier()
    read_fd, write_fd = os.pipe()

    class TempDirectory:
        name = str(tmp_path)
        cleaned = False

        def cleanup(self) -> None:
            self.cleaned = True

    temp = TempDirectory()
    monkeypatch.setattr(verifier.tempfile, "TemporaryDirectory", lambda **_: temp)
    monkeypatch.setattr(verifier.os, "pipe", lambda: (read_fd, write_fd))

    def fail_popen(*args: object, **kwargs: object) -> None:
        raise OSError("popen failed")

    monkeypatch.setattr(verifier.subprocess, "Popen", fail_popen)
    try:
        with pytest.raises(OSError, match="popen failed"):
            with verifier.server("python", "0.10.0", tokenized=True):
                pass
        assert temp.cleaned
        for fd in (read_fd, write_fd):
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        for fd in (read_fd, write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def test_server_cleans_process_and_resources_when_token_read_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verifier = _verifier()
    read_fd, write_fd = os.pipe()

    class TempDirectory:
        name = str(tmp_path)
        cleaned = False

        def cleanup(self) -> None:
            self.cleaned = True

    class LiveProcess:
        stderr = None
        returncode: int | None = None
        terminated = False
        waited = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout: float) -> int:
            self.waited = True
            assert timeout == 5
            assert self.returncode is not None
            return self.returncode

    temp = TempDirectory()
    process = LiveProcess()
    monkeypatch.setattr(verifier.tempfile, "TemporaryDirectory", lambda **_: temp)
    monkeypatch.setattr(verifier.os, "pipe", lambda: (read_fd, write_fd))
    monkeypatch.setattr(verifier.subprocess, "Popen", lambda *_, **__: process)

    def fail_token_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("token read failed")

    monkeypatch.setattr(verifier, "read_control_token", fail_token_read)
    try:
        with pytest.raises(AssertionError, match="token read failed"):
            with verifier.server("python", "0.10.0", tokenized=True):
                pass
        assert process.terminated
        assert process.waited
        assert temp.cleaned
        for fd in (read_fd, write_fd):
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        for fd in (read_fd, write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def test_client_subprocess_has_a_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = _verifier()

    def timeout(args: list[str], **kwargs: object) -> None:
        timeout_s = kwargs["timeout"]
        assert timeout_s == verifier.CLIENT_TIMEOUT_S
        assert isinstance(timeout_s, (int, float))
        raise subprocess.TimeoutExpired(args, timeout_s)

    monkeypatch.setattr(verifier.subprocess, "run", timeout)
    with pytest.raises(AssertionError, match="client timed out"):
        verifier.client("python", "ipc:///tmp/unused.sock", "summary")


def test_budget_summary_release_wheels(tmp_path: Path) -> None:
    raw_dist = os.environ.get("FLOPSCOPE_RELEASE_DIST_DIR")
    if not raw_dist:
        pytest.skip(
            "FLOPSCOPE_RELEASE_DIST_DIR is required for wheel contract verification"
        )

    dist = Path(raw_dist).resolve()
    core = _one_wheel(dist, "flopscope")
    server = _one_wheel(dist, "flopscope-server")
    client = _one_wheel(dist, "flopscope-client")

    core_meta = _metadata(core)
    server_meta = _metadata(server)
    client_meta = _metadata(client)
    versions = {core_meta["Version"], server_meta["Version"], client_meta["Version"]}
    assert len(versions) == 1, f"wheel versions are not lockstep: {versions}"
    version = versions.pop()

    core_requires = core_meta.get_all("Requires-Dist", [])
    server_requires = server_meta.get_all("Requires-Dist", [])
    verifier = _verifier()
    verifier.require_exact_dependency(
        core_requires,
        distribution="flopscope-server",
        version=version,
        required_extra="server",
    )
    verifier.require_exact_dependency(
        server_requires,
        distribution="flopscope",
        version=version,
    )

    server_venv = tmp_path / "server-venv"
    client_venv = tmp_path / "client-venv"
    _run("uv", "venv", str(server_venv), "--python", "3.10")
    _run("uv", "venv", str(client_venv), "--python", "3.10")
    _run(
        "uv",
        "pip",
        "install",
        "--python",
        str(server_venv / "bin/python"),
        str(core),
        str(server),
    )
    _run(
        "uv", "pip", "install", "--python", str(client_venv / "bin/python"), str(client)
    )

    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    _run(
        str(SCRIPT),
        "--server-python",
        str(server_venv / "bin/python"),
        "--client-python",
        str(client_venv / "bin/python"),
        "--expected-version",
        version,
        env=clean_env,
        cwd=tmp_path,
    )

    optimized_wrong_version = subprocess.run(
        [
            sys.executable,
            "-O",
            str(SCRIPT),
            "--server-python",
            str(server_venv / "bin/python"),
            "--client-python",
            str(client_venv / "bin/python"),
            "--expected-version",
            f"{version}.post1",
        ],
        check=False,
        capture_output=True,
        env=clean_env,
        cwd=tmp_path,
        timeout=30,
    )
    assert optimized_wrong_version.returncode != 0
    assert b"VersionMismatch" in optimized_wrong_version.stderr
