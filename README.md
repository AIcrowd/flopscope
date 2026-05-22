<div align="center">
<img src="website/public/logo.png" alt="flopscope" height="80">
<h1>flopscope</h1>
<p><strong>NumPy-compatible math primitives with analytical FLOP counting</strong></p>
</div>

<div align="center">

[![CI](https://github.com/AIcrowd/flopscope/actions/workflows/ci.yml/badge.svg)](https://github.com/AIcrowd/flopscope/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://aicrowd.github.io/flopscope/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/AIcrowd/flopscope/actions/workflows/ci.yml)

</div>

*Built for the [ARC Whitebox Estimation Challenge](https://aicrowd.com) by [AIcrowd](https://aicrowd.com)*

---

**flopscope** is a drop-in replacement for a subset of NumPy that counts floating-point operations as you compute. Algorithms submitted to the ARC Whitebox Estimation Challenge are scored by their analytical FLOP cost, not wall-clock time, so researchers can focus on **algorithmic innovation** rather than hardware tuning. Every arithmetic call deducts from a fixed budget; exceed it and execution stops immediately.

## Why flopscope?

<table>
<tr><th>NumPy</th><th>flopscope</th></tr>
<tr>
<td>

```python
import numpy as np


depth, width = 5, 256

# Weight init
scale = np.sqrt(2 / width)
weights = [
    np.random.randn(width, width) * scale
    for _ in range(depth)
]

# Forward pass
x = np.random.randn(width)
h = x
for i, W in enumerate(weights):
    h = np.einsum('ij,j->i', W, h)
    if i < depth - 1:
        h = np.maximum(h, 0)
# Total FLOPs? No idea.
```

</td>
<td>

```python
import flopscope as flops
import flopscope.numpy as fnp

depth, width = 5, 256

# Weight init
scale = fnp.sqrt(2 / width)
weights = [
    fnp.random.randn(width, width) * scale
    for _ in range(depth)
]

# Forward pass
x = fnp.random.randn(width)
h = x
for i, W in enumerate(weights):
    h = fnp.einsum('ij,j->i', W, h)
    if i < depth - 1:
        h = fnp.maximum(h, 0)
flops.budget_summary()  # 6,231,041 FLOPs
```

</td>
</tr>
</table>

## Key Features

- **NumPy-compatible API** -- `import flopscope.numpy as fnp` for counted NumPy ops, with `import flopscope as flops` for budgets and symmetry helpers
- **Analytical FLOP counting** -- deterministic, hardware-independent cost tracking
- **Budget enforcement** -- operations are checked before execution; exceeding the budget raises a clear error
- **Symmetry-aware einsum** -- automatic FLOP savings for repeated operands and declared symmetry groups
- **Transparent diagnostics** -- inspect per-operation costs, cumulative budget usage, and detailed summaries at any time
- **Inspect costs analytically** -- `flops.einsum_accumulation_cost(...)` and `flops.reduction_accumulation_cost(...)` return the exact FLOP count for an einsum or reduction without running the op
- **Truncated SVD** -- top-k singular value decomposition with `O(m * n * k)` cost

## What's Supported

| Module | Operations | Cost Model | Status |
|--------|-----------|------------|--------|
| NumPy surface (`fnp.*`) | 333 | Varies by category (unary, binary, reduction, free) | Supported |
| `fnp.linalg` | 31 | Per-operation formulas | Supported |
| `fnp.fft` | 18 | `5n * ceil(log2(n))` for transforms | Supported |
| `fnp.random` | 51 | `numel(output)` per sample; shuffle: `n*ceil(log2(n))` | Supported |
| `flops.stats` | 24 | Per-distribution CDF/PDF/PPF formulas | Supported |
| `fnp.polynomial` | 10 | Per-operation formulas | Supported |
| **Total** | **473 supported** | | **35 blocked** |

Blocked operations (I/O, config, and system calls) raise a helpful `AttributeError` with a link to the docs.

## Quick Start

### Installation

```bash
pip install git+https://github.com/AIcrowd/flopscope.git
# or
uv add git+https://github.com/AIcrowd/flopscope.git
```

### Basic Usage

```python
import flopscope as flops
import flopscope.numpy as fnp

depth, width = 5, 256

with flops.BudgetContext(flop_budget=10**8, wall_time_limit_s=5.0) as budget:
    # Weight init
    scale = fnp.sqrt(2 / width)
    weights = [fnp.random.randn(width, width) * scale
               for _ in range(depth)]

    # Forward pass
    x = fnp.random.randn(width)
    h = x
    for i, W in enumerate(weights):
        h = fnp.einsum('ij,j->i', W, h)
        if i < depth - 1:
            h = fnp.maximum(h, 0)

    print(budget.summary())   # current context summary

flops.budget_summary()           # accumulated session/global summary
```

```
flopscope FLOP Budget Summary
=============================
  Total budget:             100,000,000
  Used:                       6,231,041  (6.2%)
  Remaining:                 93,768,959  (93.8%)

  By operation:
    random.randn            5,246,976  ( 84.2%)  [6 calls]
    einsum                    655,360  ( 10.5%)  [5 calls]
    multiply                  327,680  (  5.3%)  [5 calls]
    maximum                     1,024  (  0.0%)  [4 calls]
    sqrt                            1  (  0.0%)  [1 call]

  Total Wall Time:     ...s
  Flopscope Backend:   ...s  (...%)
  Flopscope Overhead:  ...s  (...%)
  Residual Wall Time:  ...s  (...%)
```

### Plan Your Budget Before Executing

```python
# Query FLOP costs without running anything (no BudgetContext needed)
cost = flops.accounting.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
print(f"Matmul cost: {cost:,}")  # 33,554,432

cost = flops.accounting.svd_cost(m=256, n=256, k=10)
print(f"SVD cost: {cost:,}")     # 2,621,440
```

For symmetry-aware inspection that takes actual array inputs (and reflects
declared symmetry), use the accumulation cost APIs:

```python
import numpy as np
A = np.zeros((256, 256))
B = np.zeros((256, 256))
# Returns an AccumulationCost decomposition without running the op
cost = flops.einsum_accumulation_cost('ij,jk->ik', A, B)
print(cost.total)                # 33,554,432

cost = flops.reduction_accumulation_cost(A, op_factor=1)
print(cost.total)                # 65,535
```

### Symmetry Savings

When you pass the same array object multiple times, flopscope automatically
detects the symmetry and reduces the FLOP count:

```python
with flops.BudgetContext(flop_budget=10**8) as budget:
    X = fnp.ones((100, 100))

    # Gram matrix: both operands are the same X.
    # flopscope auto-detects this and induces S2{j,k} on the output,
    # giving ~1/2 the dense cost (since R[j,k] = R[k,j]).
    R = fnp.einsum("ij,ik->jk", X, X)
    print(f"Cost with equal-operand detection: {budget.flops_used:,}")
```

This works for any einsum where the same Python object appears at multiple
operand positions. See the
[exploit-symmetry guide](website/content/docs/guides/symmetry.mdx) for more examples
including triple products and block symmetries.

## How It Works

1. **FLOPs are tracked automatically.** A global default budget activates on first use, or you can wrap code in an explicit `BudgetContext` for a custom limit. Free ops (tensor creation, reshaping) cost 0 FLOPs.
2. **FLOP costs are analytical.** Costs are computed from tensor shapes, not measured from execution. A matmul of `(m, k) @ (k, n)` always costs `m * k * n` FLOPs regardless of hardware.
3. **Budget is checked before execution.** If an operation would exceed the remaining budget, `BudgetExhaustedError` is raised and the operation does not run.
4. **All tensors are plain `numpy.ndarray`.** Standard flopscope arrays are regular NumPy arrays with no hidden state. `SymmetricTensor` is a lightweight `ndarray` subclass that carries symmetry metadata for einsum savings — it works everywhere a normal array does.

## Sharp Edges

**Budget is always active.** A global default budget (1e15 FLOPs, configurable via `FLOPSCOPE_DEFAULT_BUDGET` env var) activates automatically. Use an explicit `BudgetContext` to set a custom limit.

**35 operations are blocked.** I/O, config, and system-level functions (`save`, `load`, `set_printoptions`, etc.) raise `AttributeError` by design. These have no meaningful FLOP cost and are not part of the competition API.

**sort, argsort, trace, and random sampling all have analytical FLOP costs** based on their algorithmic complexity.

**Nested explicit BudgetContexts are not allowed.** Opening an explicit `BudgetContext` while another explicit `BudgetContext` is already active raises `RuntimeError`. However, opening an explicit context while only the global default is active is fine — the explicit context temporarily replaces the default.

**Cost is analytical, not wall-clock.** Two operations with the same shapes always report the same FLOP cost, regardless of data values, cache effects, or hardware. This is intentional -- it makes scores reproducible across machines.

**SymmetricTensor propagation rules.** Symmetry metadata (used by einsum for FLOP savings) propagates through reshaping, slicing, and unary pointwise operations (e.g., `exp`, `log`). Binary pointwise operations (e.g., `add`, `multiply`) intersect the symmetry groups of both operands. Reductions may drop symmetry on the reduced axis. If the result doesn't carry the symmetry you expect, declare symmetry groups explicitly with `flops.as_symmetric()`.

## Documentation

**Getting Started**

- [Installation & Setup](website/content/docs/getting-started/installation.mdx)
- [Your First Budget](website/content/docs/getting-started/quickstart.mdx)

**How-To Guides**

- [Use Einsum](website/content/docs/guides/einsum.mdx)
- [Exploit Symmetry](website/content/docs/guides/symmetry.mdx)
- [Use Linear Algebra](website/content/docs/guides/linalg.mdx)
- [Plan Your Budget](website/content/docs/guides/budget-planning.mdx)
- [Debug Budget Overruns](website/content/docs/guides/budget-planning.mdx)
- [Migrate from NumPy](website/content/docs/guides/migrate-from-numpy.mdx)

**Concepts**

- [FLOP Counting Model](website/content/docs/understanding/flop-counting-model.mdx)
- [Operation Categories](website/content/docs/understanding/operation-categories.mdx)

**Development**

- [Contributor Guide](website/content/docs/development/contributing.mdx)

**API Reference**

- [Full API Reference](website/content/docs/api/index.mdx)

**For AI Agents:**

- [Cheat Sheet](website/content/docs/api/for-agents.mdx) -- compact reference for fast lookup

## Development

```bash
git clone https://github.com/AIcrowd/flopscope.git
cd flopscope
make install
make test                      # core test suite
make docs-serve                # local docs at http://127.0.0.1:8000
```

For the monorepo layout, client/server workflows, and generated-doc rules, see
[Contributor Guide](website/content/docs/development/contributing.mdx).

## Citation

```bibtex
@misc{flopscope2026,
  title  = {flopscope: NumPy-compatible math primitives with FLOP counting},
  author = {AIcrowd},
  year   = {2026},
  url    = {https://github.com/AIcrowd/flopscope},
  note   = {Built for the ARC Whitebox Estimation Challenge}
}
```

## License

[MIT](https://opensource.org/licenses/MIT)
