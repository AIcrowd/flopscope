"""Python helper: subprocess to the Node JS oracle."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

ORACLE_DIR = Path(__file__).parent / "_js_oracle"
RUN_MJS = ORACLE_DIR / "run.mjs"


def is_available() -> bool:
    """True iff Node is on PATH and the oracle script exists."""
    return shutil.which("node") is not None and RUN_MJS.exists()


def run_js_oracle(
    *,
    subscripts: str,
    output: str,
    operand_names: tuple[str, ...],
    per_op_symmetry: tuple[Any, ...] | None,
    sizes_by_label: dict[str, int],
    timeout: float = 30.0,
) -> dict:
    """Run the JS engine in a Node subprocess and return its analysis as a Python dict."""
    if not is_available():
        raise RuntimeError(
            "JS oracle unavailable: install Node.js (nvm use 23) and ensure "
            f"{RUN_MJS} exists."
        )

    payload = json.dumps(
        {
            "subscripts": subscripts,
            "output": output,
            "operand_names": list(operand_names),
            "per_op_symmetry": list(per_op_symmetry) if per_op_symmetry else None,
            "sizes_by_label": sizes_by_label,
        }
    )

    env = os.environ.copy()
    proc = subprocess.run(
        ["node", str(RUN_MJS)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=ORACLE_DIR,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"JS oracle failed (exit {proc.returncode}):\nstdout={proc.stdout!r}\n"
            f"stderr={proc.stderr!r}"
        )
    return json.loads(proc.stdout)
