from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class CostRow:
    op: str
    module: str
    status: str  # charged | free | blacklisted
    category: str
    flop_cost_formula: str
    weight: float
    complex_factor: str  # number as str, "exact", or "illegal"
    dtype_rate_rule: str
    example_input: str
    raw_flop_cost: int
    raw_flop_cost_2x: str  # int or "" (non-scalable)
    billed_int16: str  # int or "n/a"/"raises"
    billed_fp32: str
    billed_fp64: str
    billed_complex128: str
    complex_penalty: str  # ratio like "8.0" or "—"
    notes: str
    numpy_range: str
    registry_ref: str  # github permalink
    cost_impl_ref: str  # github permalink


COLUMNS = [f.name for f in fields(CostRow)]

LEGEND = {
    "op": "flopscope operation name",
    "module": "numpy / linalg / fft / random / stats",
    "status": "charged | free | blacklisted",
    "category": "billing mechanism (counted_custom, counted_reduction, ...)",
    "flop_cost_formula": "exact parameterized raw-FLOP formula",
    "weight": "per-op weight multiplier",
    "complex_factor": "complex-structure factor (number / exact / illegal)",
    "dtype_rate_rule": "which dtype is billed (operands/output/heavier/accumulator/neutral/float64-forced)",
    "example_input": "the canonical worked-example input",
    "raw_flop_cost": "measured flop_cost on the canonical input (unit weights, real dtype)",
    "raw_flop_cost_2x": "measured flop_cost at 2x input (shape-sensitive families)",
    "billed_int16": "measured billed cost, int16 input, production weights",
    "billed_fp32": "measured billed cost, float32 input",
    "billed_fp64": "measured billed cost, float64 input",
    "billed_complex128": "measured billed cost, complex128 input",
    "complex_penalty": "billed_complex128 / billed_fp32 (dimensionless)",
    "notes": "registry notes",
    "numpy_range": "min/max numpy version supported",
    "registry_ref": "GitHub permalink to the registry entry line",
    "cost_impl_ref": "GitHub permalink to the cost-computation source line",
}
