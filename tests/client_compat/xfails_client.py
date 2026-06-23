"""Known CLIENT-vs-native divergences that are proxy-inherent / by-design.

Phase 1 seeds ONLY the structurally-unavoidable divergences — the ones that can
never work through a remote proxy regardless of any fix. Everything else that
fails is a CANDIDATE gap to triage at the Phase-1 decision gate; do NOT pre-xfail
real gaps here (that would hide them from the inventory).

Patterns are matched against the pytest nodeid with fnmatch (glob) OR as a plain
substring (mirrors tests/numpy_compat/conftest.py). xfail is non-strict, so an
entry that unexpectedly passes is reported as xpass, not a failure.
"""
XFAIL_PATTERNS: dict[str, str] = {
    # RemoteArray is immutable by design — item assignment cannot be supported.
    "*test_*setitem*": "client RemoteArray is immutable by design",
    # A remote proxy has no local memory buffer, so C/Fortran-contiguity,
    # strides, and byte-layout assertions are meaningless against it.
    "*test_*contiguous*": "remote proxy has no local memory layout",
    "*test_*strides*": "remote proxy has no local memory layout",
}
