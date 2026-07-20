# flopscope-server

[![PyPI version](https://img.shields.io/pypi/v/flopscope-server.svg)](https://pypi.org/project/flopscope-server/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Server-side runtime for the [flopscope](https://pypi.org/project/flopscope/) client-server architecture.

`flopscope-server` runs the heavy NumPy-backed `flopscope` library on behalf of remote `flopscope-client` processes that connect over ZMQ + msgpack. Use it when you want to isolate untrusted or sandboxed code (typically participant submissions in the [ARC Whitebox Estimation Challenge](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026)) from the actual computation environment.

## When to install this

You want this on the **server** side of a client-server flopscope deployment. It pulls in `flopscope`, `numpy`, `pyzmq`, and `msgpack`:

```bash
pip install flopscope-server
```

Equivalent extra on the main flopscope distribution (installs both at the same exact version):

```bash
pip install "flopscope[server]"
```

## Quick start

### Launch the server

```bash
# Over a UNIX domain socket (recommended for local / single-host deployments)
python -m flopscope_server --url ipc:///tmp/flopscope.sock

# Or over TCP
python -m flopscope_server --url tcp://127.0.0.1:15555
```

### Connect a client

In a separate process (typically a sandboxed container) that has [`flopscope-client`](https://pypi.org/project/flopscope-client/) installed:

```bash
export FLOPSCOPE_SERVER_URL=ipc:///tmp/flopscope.sock
```

```python
import flopscope as flops
import flopscope.numpy as fnp

with flops.BudgetContext(flop_budget=1_000_000):
    a = fnp.array([1.0, 2.0, 3.0])
    b = fnp.array([4.0, 5.0, 6.0])
    fnp.add(a, b)
    flops.budget_summary()
```

The client's first request performs a version handshake with the server; mismatched flopscope versions raise `ConnectionError` with both versions in the message. Keep `flopscope-server` and `flopscope-client` on the same release.

## Architecture overview

```
┌─────────────────────────────┐         ZMQ + msgpack         ┌──────────────────────────────┐
│  flopscope-client process   │ ───────────────────────────▶  │  flopscope-server process    │
│  (sandbox; no numpy)        │                                │  (full flopscope + numpy)    │
│  `import flopscope as flops`│ ◀───────────────────────────  │  Executes ops, tracks budget │
└─────────────────────────────┘                                └──────────────────────────────┘
```

Every counted operation (matmul, einsum, FFT, reductions, …) is dispatched to the server, executed against the real library, and the result + remaining budget streamed back. The client never sees numpy.

## Related

- [`flopscope`](https://pypi.org/project/flopscope/) — the NumPy-backed library hosted by this server
- [`flopscope-client`](https://pypi.org/project/flopscope-client/) — the lightweight proxy clients use to talk to a flopscope-server
- [Documentation](https://aicrowd.github.io/flopscope/) — full guides including client/server architecture and Docker recipes
- [GitHub](https://github.com/AIcrowd/flopscope) — source, CHANGELOG, contributor guide

## License

[MIT](https://opensource.org/licenses/MIT)
