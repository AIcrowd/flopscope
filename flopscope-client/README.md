# flopscope-client

[![PyPI version](https://img.shields.io/pypi/v/flopscope-client.svg)](https://pypi.org/project/flopscope-client/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Lightweight drop-in replacement for [flopscope](https://pypi.org/project/flopscope/) that proxies all operations to a remote [`flopscope-server`](https://pypi.org/project/flopscope-server/) over ZMQ + msgpack.

`flopscope-client` provides the same `import flopscope` Python module as the main `flopscope` distribution, but with **no NumPy dependency** — it forwards every counted operation to a `flopscope-server` process. Use this in constrained environments where you want flopscope's FLOP-counting API but cannot ship numpy + the full library (e.g. sandboxed participant containers in the [ARC Whitebox Estimation Challenge](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026)).

## Install instead of, not alongside

`flopscope-client` occupies the same `flopscope` Python import namespace as the main package. The two are **mutually exclusive** — installing both leads to file-overlap in `flopscope/`. Choose one:

```bash
# Lightweight: client-only, no numpy
pip install flopscope-client

# Heavy: full library on the local machine
pip install flopscope

# Server-side: both flopscope + flopscope-server, pinned together
pip install "flopscope[server]"
```

## Quick start

The client connects to a `flopscope-server` instance specified by the `FLOPSCOPE_SERVER_URL` environment variable:

```bash
export FLOPSCOPE_SERVER_URL=tcp://flopscope-server.example.com:15555
# or for a local UNIX socket:
export FLOPSCOPE_SERVER_URL=ipc:///tmp/flopscope.sock
```

Then use flopscope normally — the import path and API are identical to the main distribution:

```python
import flopscope as flops
import flopscope.numpy as fnp

with flops.BudgetContext(flop_budget=1_000_000):
    a = fnp.array([1.0, 2.0, 3.0])
    b = fnp.array([4.0, 5.0, 6.0])
    result = fnp.add(a, b)   # round-trips to the server, runs there
    flops.budget_summary()
```

On the first request, the client performs a version handshake with the server. A version mismatch raises `ConnectionError` with both versions in the error message — keep `flopscope-server` and `flopscope-client` on the same release.

## Differences from NumPy

The API mirrors NumPy, with one semantic difference worth knowing up front: **flopscope arrays are immutable** (like [JAX](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates)). Item assignment and indexed in-place updates raise `TypeError`:

```python
arr[i] = value      # TypeError: flopscope arrays are immutable
arr[i] += value     # same error — desugars to arr[i] = arr[i] + value
```

Build results functionally instead — collect the pieces in a list and `fnp.stack(...)` them, or use a whole-array update (`arr = arr + x`, which rebinds rather than mutates). See the [Immutable arrays](https://aicrowd.github.io/flopscope/docs/getting-started/competition/#immutable-arrays) guide for the accumulation recipe.

## When to choose this over the main `flopscope` install

| Scenario | Install |
|----------|---------|
| You want flopscope's API in a process that has full NumPy + scientific stack | `pip install flopscope` |
| You're shipping a sandboxed container that must not contain numpy / scipy | `pip install flopscope-client` |
| You're running the server side that hosts computation for many clients | `pip install flopscope-server` (brings `flopscope` along) |
| You're deploying both sides on one machine, version-pinned | `pip install "flopscope[server]"` |

## Architecture overview

```
┌─────────────────────────────┐         ZMQ + msgpack         ┌──────────────────────────────┐
│  flopscope-client process   │ ───────────────────────────▶  │  flopscope-server process    │
│  (sandbox; no numpy)        │                                │  (full flopscope + numpy)    │
│  `import flopscope as flops`│ ◀───────────────────────────  │  Executes ops, tracks budget │
└─────────────────────────────┘                                └──────────────────────────────┘
```

The client serializes each operation (op name, args, kwargs) as msgpack, sends it over the ZMQ REQ/REP socket, and decodes the response. Budget tracking, symmetry-aware FLOP counting, and operation-cost analytics all happen on the server side; the client just relays calls.

## Related

- [`flopscope`](https://pypi.org/project/flopscope/) — full NumPy-backed library (alternative install)
- [`flopscope-server`](https://pypi.org/project/flopscope-server/) — the server-side runtime this client connects to
- [Documentation](https://aicrowd.github.io/flopscope/) — full guides
- [GitHub](https://github.com/AIcrowd/flopscope) — source, CHANGELOG, contributor guide

## License

[MIT](https://opensource.org/licenses/MIT)
