# NumPy Compatibility Test Harness

Runs NumPy's own test suite with flopscope monkeypatched in.

## How it works

`conftest.py` freezes the original numpy module, rebinds flopscope's internal
`_np` references to the frozen copy, then patches most non-ufunc flopscope
functions onto numpy. This lets NumPy's tests call flopscope's versions while
flopscope internally calls unpatched numpy (avoiding infinite recursion).

Ufuncs (101) and blacklisted ops (32) are skipped. Everything else -- free ops,
counted custom ops, submodule functions (linalg, fft, random) -- is patched.

See `docs/concepts/numpy-compatibility-testing.md` for full details.

## Running

```bash
# Run everything (recommended)
make test-numpy-compat

# Run a single suite
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -n auto -q

# Filter to specific functions
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -k "sqrt" -n auto -v
```

## Current results (2026-05-23)

| Suite | Module | Passed | xfailed |
|-------|--------|--------|---------|
| Core math | `test_umath` | 4,667 | 16 |
| Ufunc infra | `test_ufunc` | 799 | 26 |
| Numeric ops | `test_numeric` | 1,560 | 20 |
| Linear algebra | `test_linalg` | 438 | 2 |
| FFT | `test_pocketfft` | 150 | 0 |
| Polynomials | `test_polynomial` | 41 | 0 |
| Random | `test_random` | 143 | 2 |
| **Total** | | **8,098** | **66** |

**Note:** The harness was restored on 2026-05-23 after a regression in the JAX-style rebrand where `_current_flopscope_numpy()` incorrectly pointed to an unpatched module, leaving `_PATCHED` empty. With the fix applied, all non-blacklisted operations are now properly instrumented during NumPy's test runs.

## Triaging failures

1. Run a test module and capture failures
2. For each failure, determine the category (see xfails.py)
3. Add to XFAIL_PATTERNS with the category and explanation
4. Failures we WANT to fix go into GitHub issues instead
