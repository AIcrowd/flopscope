# FLOP weight data files

## `default_weights.json` — THE SOURCE OF TRUTH (edit this)
The applied per-op FLOP weights that flopscope actually bills. `_weights.py`
loads it at import; it is the only weights file shipped in the wheel; and the
published API reference (`ops.json`) is generated from it. **Weight/policy
changes go here** (e.g. "make data-movement ops free"). Keep values in the
calibrated tier set `{0, 1, 4, 8, 16}` (enforced by `test_weight_tier_policy`).

## `weights.json` — frozen calibration evidence (do NOT bill from this)
Raw benchmark results + per-op measurement details (`meta.per_op_details`) from
the one-time hardware calibration that kick-started the tiers. **Not loaded at
runtime, not the billing source** — retained as historical evidence (read by
`benchmarks/` and the weights metadata checks).

## `weights.csv` — human calibration spreadsheet (evidence)
The reviewable export of the calibration (empirical vs reviewer vs applied),
synced to Google Sheets via `scripts/upload_to_sheets.py`. Historical evidence.

## Re-calibrating (rare)
Weight-setting is now mostly policy. If you ever re-measure: run `benchmarks/`,
inspect the results, then **manually** update `default_weights.json` if you
choose to act on them. There is intentionally no auto-derivation script.
