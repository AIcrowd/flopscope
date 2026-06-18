"""Snapshot the FULL flopscope.numpy public surface for the client parity guard.

flopscope-client cannot import full flopscope (it is numpy-free, and both
packages are named ``flopscope``), so the client's surface-parity guard compares
its live surface against this committed snapshot. Run from the repo root in the
full (numpy) venv.

    uv run python scripts/generate_client_surface_snapshot.py          # write
    uv run python scripts/generate_client_surface_snapshot.py --check  # CI gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import flopscope.numpy as fnp

SNAPSHOT = (
    Path(__file__).resolve().parent.parent
    / "flopscope-client"
    / "tests"
    / "fixtures"
    / "full_numpy_surface.json"
)


def build_surface() -> dict[str, dict[str, bool]]:
    names = set(getattr(fnp, "__all__", []))
    names.update(n for n in dir(fnp) if not n.startswith("_"))
    out: dict[str, dict[str, bool]] = {}
    for name in sorted(names):
        try:
            obj = getattr(fnp, name)
        except Exception:
            continue
        out[name] = {"callable": callable(obj), "is_type": isinstance(obj, type)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check", action="store_true", help="fail if the committed snapshot is stale"
    )
    args = ap.parse_args()

    payload = json.dumps(build_surface(), indent=2, sort_keys=True) + "\n"

    if args.check:
        current = SNAPSHOT.read_text() if SNAPSHOT.exists() else ""
        if current != payload:
            print(
                "full_numpy_surface.json is stale. Regenerate with:\n"
                "  uv run python scripts/generate_client_surface_snapshot.py",
                file=sys.stderr,
            )
            sys.exit(1)
        print("full_numpy_surface.json is up to date")
    else:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(payload)
        print(f"wrote {SNAPSHOT}")


if __name__ == "__main__":
    main()
