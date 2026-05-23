"""Calibration script for the dimino_budget setting.

Measures cold ``_dimino`` enumeration time plus ``burnside_unique_count``
time across representative permutation groups, then prints a
recommendation for ``dimino_budget``.

Run with::

    python -m benchmarks._perm_group_calibration
    python -m benchmarks._perm_group_calibration --budget-ms 50 \\
        --output benchmarks/_perm_group_calibration.json

Output is informational. Users on faster/slower machines run this and
``flops.configure(dimino_budget=<recommended>)``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass

from flopscope._perm_group import SymmetryGroup, _dimino


@dataclass(frozen=True)
class Measurement:
    label: str
    group_order: int
    degree: int
    dimino_ms: float
    burnside_ms: float

    @property
    def total_ms(self) -> float:
        return self.dimino_ms + self.burnside_ms


def _measure_one(group: SymmetryGroup, label: str) -> Measurement:
    """Time cold ``_dimino`` + Burnside enumeration on one group."""
    # Cold dimino: bypass the cache by calling _dimino directly.
    t = time.perf_counter()
    _dimino(group._generators)
    dimino_ms = (time.perf_counter() - t) * 1000.0

    # Burnside: realistic size_dict (uniform 4-dim across all axes).
    size_dict = dict.fromkeys(range(group.degree), 4)
    t = time.perf_counter()
    group.burnside_unique_count(size_dict)
    burnside_ms = (time.perf_counter() - t) * 1000.0

    return Measurement(
        label=label,
        group_order=group.order(),
        degree=group.degree,
        dimino_ms=dimino_ms,
        burnside_ms=burnside_ms,
    )


def _sample_groups() -> list[tuple[str, SymmetryGroup]]:
    """Return the representative group sample for calibration."""
    samples: list[tuple[str, SymmetryGroup]] = []
    for n in (3, 4, 5, 6, 7, 8, 9):
        samples.append((f"S_{n}", SymmetryGroup.symmetric(axes=tuple(range(n)))))
    for n in (4, 8, 16, 32, 64):
        samples.append((f"C_{n}", SymmetryGroup.cyclic(axes=tuple(range(n)))))
    for n in (4, 8, 16, 32, 64):
        samples.append((f"D_{n}", SymmetryGroup.dihedral(axes=tuple(range(n)))))
    samples.append(
        (
            "S_3 x S_3",
            SymmetryGroup.direct_product(
                SymmetryGroup.symmetric(axes=(0, 1, 2)),
                SymmetryGroup.symmetric(axes=(3, 4, 5)),
            ),
        )
    )
    samples.append(
        (
            "S_4 x S_4",
            SymmetryGroup.direct_product(
                SymmetryGroup.symmetric(axes=(0, 1, 2, 3)),
                SymmetryGroup.symmetric(axes=(4, 5, 6, 7)),
            ),
        )
    )
    samples.append(
        (
            "C_5 x C_5 x C_5",
            SymmetryGroup.direct_product(
                SymmetryGroup.cyclic(axes=(0, 1, 2, 3, 4)),
                SymmetryGroup.cyclic(axes=(5, 6, 7, 8, 9)),
                SymmetryGroup.cyclic(axes=(10, 11, 12, 13, 14)),
            ),
        )
    )
    return samples


def _print_table(measurements: list[Measurement]) -> None:
    header = f"{'group':<22}{'|G|':>12}{'dimino':>12}{'burnside':>12}{'total':>12}"
    print(header)
    print("-" * len(header))
    for m in measurements:
        print(
            f"{m.label:<22}"
            f"{m.group_order:>12}"
            f"{m.dimino_ms:>10.2f}ms"
            f"{m.burnside_ms:>10.2f}ms"
            f"{m.total_ms:>10.2f}ms"
        )


def _recommend_budget(measurements: list[Measurement], budget_ms: float) -> int:
    """Largest |G| whose total cold cost stays under ``budget_ms``."""
    under_budget = [m for m in measurements if m.total_ms <= budget_ms]
    if not under_budget:
        return 1  # nothing fits; degenerate machine
    return max(m.group_order for m in under_budget)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=100.0,
        help="Wall-clock budget per group construction (default: 100ms)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path",
    )
    args = parser.parse_args()

    samples = _sample_groups()
    measurements = [_measure_one(g, label) for label, g in samples]

    _print_table(measurements)
    recommended = _recommend_budget(measurements, args.budget_ms)
    print()
    print(
        f"Recommended dimino_budget: {recommended}   "
        f"(group_order_budget_ms={args.budget_ms}, machine: {platform.platform()})"
    )

    if args.output:
        payload = {
            "machine": platform.platform(),
            "budget_ms": args.budget_ms,
            "recommended_dimino_budget": recommended,
            "measurements": [
                {
                    "label": m.label,
                    "group_order": m.group_order,
                    "degree": m.degree,
                    "dimino_ms": m.dimino_ms,
                    "burnside_ms": m.burnside_ms,
                    "total_ms": m.total_ms,
                }
                for m in measurements
            ],
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote calibration data to {args.output}")


if __name__ == "__main__":
    main()
