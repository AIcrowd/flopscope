"""Known CLIENT-vs-native divergences that are proxy-inherent / by-design.

Phase 1 seeds ONLY the structurally-unavoidable divergences — the ones that can
never work through a remote proxy regardless of any fix. Everything else that
fails is a CANDIDATE gap to triage at the Phase-1 decision gate; do NOT pre-xfail
real gaps here (that would hide them from the inventory).

Patterns are matched against the pytest nodeid with fnmatch (glob) OR as a plain
substring (mirrors tests/numpy_compat/conftest.py). xfail is non-strict, so an
entry that unexpectedly passes is reported as xpass, not a failure.
"""

# Seeded EMPTY after the first measurement. The structurally-"by-design"
# patterns first guessed here (setitem/contiguous/strides) do NOT actually
# manifest in this harness: NumPy's tests build their operands with the native
# np.array (kept native — it is in _patch_client._SKIP), so item assignment and
# memory-layout introspection run against real ndarrays and PASS. Seeding them
# produced 147 spurious xpass. Operator/method gaps on a real RemoteArray (the
# audit's bitwise &, argsort, etc.) are covered directly by
# test_native_feature_smoke.py and the audit-gap tests, not by this numpy-suite
# harness. Populate this ledger at the Phase-1 decision gate from the measured
# inventory (each entry: documented by-design / proxy-internal reason).
XFAIL_PATTERNS: dict[str, str] = {}
