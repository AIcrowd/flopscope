from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY_REL = "src/flopscope/_registry.py"
_GH = "https://github.com/AIcrowd/flopscope/blob"


def current_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def registry_entry_line(op: str) -> int | None:
    pat = re.compile(rf'^\s*"{re.escape(op)}"\s*:\s*\{{')
    text = (REPO / REGISTRY_REL).read_text().splitlines()
    for i, line in enumerate(text, 1):
        if pat.match(line):
            return i
    return None


def permalink(rel_path: str, line: int, sha: str) -> str:
    return f"{_GH}/{sha}/{rel_path}#L{line}"
