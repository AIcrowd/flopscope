#!/usr/bin/env python
"""generate_api_docs.py — Generate API reference data and verify coverage.

Usage
-----
    python scripts/generate_api_docs.py              # generate API data
    python scripts/generate_api_docs.py --verify     # verify API data only
"""

from __future__ import annotations

import argparse
import csv
import doctest
import html
import importlib
import inspect
import io
import json
import multiprocessing as mp
import os
import re
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = (ROOT / "src").resolve()
WEBSITE = ROOT / "website"
PUBLIC_DIR = WEBSITE / "public"
API_DATA_DIR = PUBLIC_DIR / "api-data" / "ops"
PUBLIC_API_DATA_DIR = PUBLIC_DIR / "api-data" / "public-api"
GENERATED_DIR = WEBSITE / ".generated"
API_INDEX_PATH = WEBSITE / "content" / "docs" / "api" / "index.mdx"
OP_DOCS_DIR = WEBSITE / "content" / "docs" / "api" / "ops"
SYMBOL_DOCS_DIR = WEBSITE / "content" / "docs" / "api" / "symbols"
API_EXAMPLES_DIR = WEBSITE / "content" / "api-examples"
# Legacy MkDocs paths kept for helper functions that are no longer invoked by
# the current docs pipeline. The active build path only emits website/public/ops.json.
DOCS = ROOT / "docs"
API_DIR = DOCS / "api"
REF_DIR = DOCS / "reference"
WEIGHTS_PATH = ROOT / "src" / "flopscope" / "data" / "weights.json"
WEIGHTS_CSV_PATH = ROOT / "src" / "flopscope" / "data" / "weights.csv"

# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def _module_is_from_local_checkout(module: object | None) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False

    try:
        return Path(module_file).resolve().is_relative_to(SRC_ROOT)
    except OSError:
        return False


def _import_local_flopscope_module(module_name: str):
    src_root = str(SRC_ROOT)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    root_module = sys.modules.get("flopscope")
    if root_module is not None and not _module_is_from_local_checkout(root_module):
        for name in list(sys.modules):
            if name == "flopscope" or name.startswith("flopscope."):
                sys.modules.pop(name, None)

    module = importlib.import_module(module_name)
    if not _module_is_from_local_checkout(module):
        raise RuntimeError(
            f"Failed to import {module_name} from local checkout under {SRC_ROOT}"
        )
    return module


def load_registry() -> dict[str, dict]:
    registry_module = _import_local_flopscope_module("flopscope._registry")
    return registry_module.REGISTRY


# ---------------------------------------------------------------------------
# Module mapping: which flopscope source modules cover which registry ops
# ---------------------------------------------------------------------------

# Maps (registry module, category prefix) → list of mkdocstrings directives
# and the doc page they belong to.
#
# Existing pages (not generated):
#   counted-ops.md  → ::: flopscope._pointwise, ::: flopscope._einsum.einsum
#   free-ops.md     → ::: flopscope._free_ops
#
# Generated pages:
GENERATED_PAGES: dict[str, dict] = {
    "api/linalg.md": {
        "title": "Linear Algebra",
        "description": (
            "Operations from `flopscope.numpy.linalg`. "
            "Cost formulas vary per operation — see each function's docstring."
        ),
        "directives": [
            "flopscope.numpy.linalg._svd",
            "flopscope.numpy.linalg._decompositions",
            "flopscope.numpy.linalg._solvers",
            "flopscope.numpy.linalg._properties",
            "flopscope.numpy.linalg._compound",
            "flopscope.numpy.linalg._aliases",
        ],
        "registry_modules": {"numpy.linalg"},
    },
    "api/fft.md": {
        "title": "FFT",
        "description": textwrap.dedent("""\
            Fast Fourier Transform operations from `flopscope.numpy.fft`. All FFT
            transforms are counted. Real-valued transforms (`rfft`) cost roughly
            half of complex transforms.

            ## Cost Summary

            | Operation | Cost Formula |
            |-----------|-------------|
            | `fft`, `ifft` | $5n \\cdot \\lceil\\log_2 n\\rceil$ |
            | `fft2`, `ifft2` | $5N \\cdot \\lceil\\log_2 N\\rceil$ where $N = n_1 \\cdot n_2$ |
            | `fftn`, `ifftn` | $5N \\cdot \\lceil\\log_2 N\\rceil$ where $N = \\prod_i n_i$ |
            | `rfft`, `irfft` | $5(n/2) \\cdot \\lceil\\log_2 n\\rceil$ |
            | `rfft2`, `irfft2` | $5(N/2) \\cdot \\lceil\\log_2 N\\rceil$ where $N = n_1 \\cdot n_2$ |
            | `rfftn`, `irfftn` | $5(N/2) \\cdot \\lceil\\log_2 N\\rceil$ where $N = \\prod_i n_i$ |
            | `hfft`, `ihfft` | $5n_{\\text{out}} \\cdot \\lceil\\log_2 n_{\\text{out}}\\rceil$ |
            | `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` | $0$ (free) |

            ## Examples

            ```python
            import flopscope as flops

            with flops.BudgetContext(flop_budget=1_000_000) as budget:
                signal = flops.random.randn(1024)    # free
                spectrum = flops.fft.fft(signal)     # 5 * 1024 * 10 = 51,200 FLOPs
                freqs = flops.fft.fftfreq(1024)      # free
                print(f"FFT cost: {budget.flops_used:,}")  # 51,200
            ```

            ## API Reference
        """),
        "directives": [
            "flopscope.numpy.fft._transforms",
            "flopscope.numpy.fft._free",
        ],
        "registry_modules": {"numpy.fft"},
    },
    "api/random.md": {
        "title": "Random",
        "description": (
            "Random number generation from `flopscope.numpy.random`. "
            "Sampling operations are **counted** — each sample costs "
            "`numel(output)` FLOPs, and shuffle/permutation costs "
            "`n * ceil(log2(n))` FLOPs. Only configuration helpers "
            "(`seed`, `get_state`, `set_state`, `default_rng`) are free (0 FLOPs)."
        ),
        "directives": [
            "flopscope.numpy.random",
        ],
        "registry_modules": {"numpy.random"},
    },
    "api/polynomial.md": {
        "title": "Polynomial",
        "description": textwrap.dedent("""\
            Polynomial operations from `flopscope`. These wrap NumPy's polynomial
            functions with FLOP counting.

            ## Cost Summary

            | Operation | Cost Formula |
            |-----------|-------------|
            | `polyval` | $2 \\cdot m \\cdot \\text{deg}$ (Horner's method, FMA=2) |
            | `polyadd`, `polysub` | $\\max(n_1, n_2)$ |
            | `polymul`, `polydiv` | $n_1 \\cdot n_2$ |
            | `polyfit` | $2m \\cdot (\\text{deg}+1)^2$ |
            | `poly` | $n^2$ |
            | `roots` | $10n^3$ (companion matrix eigendecomposition) |
            | `polyder`, `polyint` | $n$ |

            ## Examples

            ```python
            import flopscope as flops

            with flops.BudgetContext(flop_budget=1_000_000) as budget:
                coeffs = flops.array([1.0, -3.0, 2.0])  # x^2 - 3x + 2 (free)
                x = flops.linspace(0, 5, 100)            # free
                y = flops.polyval(coeffs, x)             # 2 * 100 * 2 = 400 FLOPs
                print(f"polyval cost: {budget.flops_used}")  # 400
            ```

            ## API Reference
        """),
        "directives": [
            "flopscope._polynomial",
        ],
        "registry_modules": {"flopscope._polynomial"},
    },
    "api/stats.md": {
        "title": "Statistical Distributions",
        "description": textwrap.dedent("""\
            Statistical distributions from `flopscope.stats`. This submodule
            provides a **subset of scipy.stats** — each distribution is a
            singleton with `.pdf()`, `.cdf()`, and `.ppf()` methods that
            match the scipy API exactly (same signatures, same numerical
            results to within 1e-12).

            Unlike NumPy operations which are direct wrappers, these functions
            reproduce the `scipy.stats` interface so that participants can use
            standard statistical distributions without importing scipy.

            ## Cost Summary

            | Distribution | pdf | cdf | ppf |
            |-------------|-----|-----|-----|
            | `norm` | $10n$ | $20n$ | $40n$ |
            | `uniform` | $3n$ | $3n$ | $3n$ |
            | `expon` | $5n$ | $5n$ | $5n$ |
            | `cauchy` | $6n$ | $5n$ | $5n$ |
            | `logistic` | $8n$ | $5n$ | $5n$ |
            | `laplace` | $5n$ | $5n$ | $5n$ |
            | `lognorm` | $15n$ | $25n$ | $45n$ |
            | `truncnorm` | $30n$ | $30n$ | $50n$ |

            where $n$ = `numel(x)` (or `numel(q)` for ppf).

            ## Examples

            ```python
            import flopscope as flops

            with flops.BudgetContext(flop_budget=1_000_000) as budget:
                x = flops.linspace(-3, 3, 1000)         # free
                pdf_vals = flops.stats.norm.pdf(x)       # 10 * 1000 = 10,000 FLOPs
                cdf_vals = flops.stats.norm.cdf(x)       # 20 * 1000 = 20,000 FLOPs
                q = flops.array([0.025, 0.975])           # free
                bounds = flops.stats.norm.ppf(q)          # 40 * 2 = 80 FLOPs
                print(f"Total: {budget.flops_used:,}")  # 30,080
            ```

            ## Compatibility

            All outputs are verified against `scipy.stats` in the test suite.
            See `tests/test_stats_*.py`.

            ## API Reference
        """),
        "directives": [
            "flopscope.stats._norm",
            "flopscope.stats._uniform",
            "flopscope.stats._expon",
            "flopscope.stats._cauchy",
            "flopscope.stats._logistic",
            "flopscope.stats._laplace",
            "flopscope.stats._lognorm",
            "flopscope.stats._truncnorm",
        ],
        "registry_modules": set(),  # stats ops are not in the numpy registry
    },
    "api/window.md": {
        "title": "Window Functions",
        "description": textwrap.dedent("""\
            Window function wrappers from `flopscope`. These generate window
            arrays used in signal processing (e.g., for windowed FFTs).

            ## Cost Summary

            | Operation | Cost Formula | Notes |
            |-----------|-------------|-------|
            | `bartlett` | $n$ | Linear taper |
            | `hamming` | $n$ | One cosine term |
            | `hanning` | $n$ | One cosine term |
            | `blackman` | $3n$ | Three cosine terms |
            | `kaiser` | $3n$ | Bessel function evaluation |

            ## Examples

            ```python
            import flopscope as flops

            with flops.BudgetContext(flop_budget=1_000_000) as budget:
                win = flops.hamming(256)   # 256 FLOPs
                win2 = flops.kaiser(256)   # 768 FLOPs (3 * 256)
                print(f"Window cost: {budget.flops_used}")  # 1024
            ```

            ## API Reference
        """),
        "directives": [
            "flopscope._window",
        ],
        "registry_modules": {"flopscope._window"},
    },
}

# Existing (non-generated) pages and what registry modules they cover
EXISTING_PAGES: dict[str, set[str]] = {
    "api/counted-ops.md": set(),  # covers numpy counted ops + einsum + unwrap
    "api/free-ops.md": set(),  # covers numpy free ops
    "api/symmetric.md": set(),  # covers SymmetricTensor, SymmetryInfo, as_symmetric
}


# ---------------------------------------------------------------------------
# Generate API reference pages
# ---------------------------------------------------------------------------

HEADER = (
    "<!-- Auto-generated by scripts/generate_api_docs.py. Do not edit manually. -->"
)


def generate_api_page(page_path: str, page_info: dict) -> None:
    """Generate a single API reference markdown page."""
    out = DOCS / page_path
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        HEADER,
        f"# {page_info['title']}",
        "",
        page_info["description"],
        "",
    ]

    for directive in page_info["directives"]:
        lines.append(f"::: {directive}")
        lines.append("")

    out.write_text("\n".join(lines))
    print(f"  Generated {page_path}")


# ---------------------------------------------------------------------------
# Generate operation audit page
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    "free",
    "counted_unary",
    "counted_binary",
    "counted_reduction",
    "counted_custom",
    "blacklisted",
]

CATEGORY_LABELS = {
    "free": ("Free Operations (0 FLOPs)", "0 FLOPs"),
    "counted_unary": ("Counted Unary Operations", r"$\text{numel}(\text{output})$"),
    "counted_binary": ("Counted Binary Operations", r"$\text{numel}(\text{output})$"),
    "counted_reduction": (
        "Counted Reduction Operations",
        r"$\text{numel}(\text{input})$",
    ),
    "counted_custom": ("Counted Custom Operations", "Per-operation formula"),
    "counted_random_method": ("Counted Random Methods", "Per-method formula"),
    "free_random_method": ("Free Random Methods", "0 FLOPs"),
    "blacklisted": ("Blacklisted Operations", "Not available"),
}

CATEGORY_EMOJI = {
    "free": "\U0001f7e2",  # green circle
    "counted_unary": "\U0001f7e1",  # yellow circle
    "counted_binary": "\U0001f7e1",
    "counted_reduction": "\U0001f7e1",
    "counted_custom": "\U0001f7e0",  # orange circle
    "counted_random_method": "\U0001f7e1",  # yellow circle, like other counted_*
    "free_random_method": "\U0001f7e2",  # green circle, like free
    "blacklisted": "\U0001f534",  # red circle
}

# ---------------------------------------------------------------------------
# Per-operation cost formulas (plain text + LaTeX)
# ---------------------------------------------------------------------------

CUSTOM_COSTS: dict[str, tuple[str, str]] = {
    "einsum": (
        "op_factor * product of all index dims",
        r"$\text{op\_factor} \cdot \prod_i d_i$",
    ),
    "einsum_path": ("0 (planning only)", "$0$"),
    "dot": (
        "2 * m * k * n - m * n (FMA=2)",
        r"$2 \cdot m \cdot k \cdot n - m \cdot n$",
    ),
    "matmul": (
        "2 * m * k * n - m * n (FMA=2)",
        r"$2 \cdot m \cdot k \cdot n - m \cdot n$",
    ),
    "inner": ("n", "$n$"),
    "outer": ("m * n", r"$m \cdot n$"),
    "tensordot": ("product of contracted dims * output size", r"$\prod_i d_i$"),
    "vdot": ("n", "$n$"),
    "vecdot": ("n", "$n$"),
    "kron": ("m1*m2 * n1*n2", r"$m_1 m_2 \cdot n_1 n_2$"),
    "clip": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "cross": ("numel(output)", r"$\text{numel}(\text{output})$"),
    "diff": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "ediff1d": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "gradient": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "convolve": ("n * m", r"$n \cdot m$"),
    "correlate": ("n * m", r"$n \cdot m$"),
    "corrcoef": ("n^2 * m", r"$n^2 \cdot m$"),
    "cov": ("n^2 * m", r"$n^2 \cdot m$"),
    "interp": ("n * log(m)", r"$n \cdot \log m$"),
    "trapezoid": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "trapz": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "linalg.svd": ("m * n * k", r"$m \cdot n \cdot k$"),
    "linalg.svdvals": ("m * n * min(m,n)", r"$m \cdot n \cdot \min(m,n)$"),
    "linalg.cholesky": ("n^3", r"$n^3$"),
    "linalg.qr": ("m * n * min(m,n)", r"$m \cdot n \cdot \min(m,n)$"),
    "linalg.eig": ("n^3", r"$n^3$"),
    "linalg.eigh": ("n^3", r"$n^3$"),
    "linalg.eigvals": ("n^3", r"$n^3$"),
    "linalg.eigvalsh": ("n^3", r"$n^3$"),
    "linalg.cross": ("delegates to flops.cross", r"delegates to `cross`"),
    "linalg.matmul": ("delegates to flops.matmul", r"delegates to `matmul`"),
    "linalg.outer": ("delegates to flops.outer", r"delegates to `outer`"),
    "linalg.tensordot": ("delegates to flops.tensordot", r"delegates to `tensordot`"),
    "linalg.vecdot": ("delegates to flops.vecdot", r"delegates to `vecdot`"),
    "linalg.solve": ("n^3", r"$n^3$"),
    "linalg.inv": ("n^3", r"$n^3$"),
    "linalg.lstsq": ("m * n * min(m,n)", r"$m \cdot n \cdot \min(m,n)$"),
    "linalg.pinv": ("m * n * min(m,n)", r"$m \cdot n \cdot \min(m,n)$"),
    "linalg.tensorsolve": ("n^3", r"$n^3$"),
    "linalg.tensorinv": ("n^3", r"$n^3$"),
    "linalg.trace": ("n", "$n$"),
    "linalg.det": ("n^3", r"$n^3$"),
    "linalg.slogdet": ("n^3", r"$n^3$"),
    "linalg.norm": ("depends on ord", r"varies"),
    "linalg.vector_norm": ("numel", r"$n$"),
    "linalg.matrix_norm": ("depends on ord", r"varies"),
    "linalg.cond": ("m * n * min(m,n)", r"$m \cdot n \cdot \min(m,n)$"),
    "linalg.matrix_rank": ("m * n * min(m,n)", r"$m \cdot n \cdot \min(m,n)$"),
    "linalg.multi_dot": ("optimal chain cost", r"optimal chain"),
    "linalg.matrix_power": ("log2(k) * n^3", r"$\lfloor\log_2 k\rfloor \cdot n^3$"),
    "fft.fft": ("5n * ceil(log2(n))", r"$5n \cdot \lceil\log_2 n\rceil$"),
    "fft.ifft": ("5n * ceil(log2(n))", r"$5n \cdot \lceil\log_2 n\rceil$"),
    "fft.fft2": ("5N * ceil(log2(N))", r"$5N \cdot \lceil\log_2 N\rceil$"),
    "fft.ifft2": ("5N * ceil(log2(N))", r"$5N \cdot \lceil\log_2 N\rceil$"),
    "fft.fftn": ("5N * ceil(log2(N))", r"$5N \cdot \lceil\log_2 N\rceil$"),
    "fft.ifftn": ("5N * ceil(log2(N))", r"$5N \cdot \lceil\log_2 N\rceil$"),
    "fft.rfft": ("5(n/2) * ceil(log2(n))", r"$5(n/2) \cdot \lceil\log_2 n\rceil$"),
    "fft.irfft": ("5(n/2) * ceil(log2(n))", r"$5(n/2) \cdot \lceil\log_2 n\rceil$"),
    "fft.rfft2": ("5(N/2) * ceil(log2(N))", r"$5(N/2) \cdot \lceil\log_2 N\rceil$"),
    "fft.irfft2": ("5(N/2) * ceil(log2(N))", r"$5(N/2) \cdot \lceil\log_2 N\rceil$"),
    "fft.rfftn": ("5(N/2) * ceil(log2(N))", r"$5(N/2) \cdot \lceil\log_2 N\rceil$"),
    "fft.irfftn": ("5(N/2) * ceil(log2(N))", r"$5(N/2) \cdot \lceil\log_2 N\rceil$"),
    "fft.hfft": ("5n * ceil(log2(n))", r"$5n \cdot \lceil\log_2 n\rceil$"),
    "fft.ihfft": ("5n * ceil(log2(n))", r"$5n \cdot \lceil\log_2 n\rceil$"),
    "polyval": ("2 * m * deg (FMA=2)", r"$2 \cdot m \cdot \text{deg}$"),
    "polyadd": ("max(n1, n2)", r"$\max(n_1, n_2)$"),
    "polysub": ("max(n1, n2)", r"$\max(n_1, n_2)$"),
    "polyder": ("n", "$n$"),
    "polyint": ("n", "$n$"),
    "polymul": ("n1 * n2", r"$n_1 \cdot n_2$"),
    "polydiv": ("n1 * n2", r"$n_1 \cdot n_2$"),
    "polyfit": ("2m * (deg+1)^2", r"$2m \cdot (\text{deg}+1)^2$"),
    "poly": ("n^2", r"$n^2$"),
    "roots": ("10n^3", r"$10n^3$"),
    "bartlett": ("n", "$n$"),
    "blackman": ("3n", "$3n$"),
    "hamming": ("n", "$n$"),
    "hanning": ("n", "$n$"),
    "kaiser": ("3n", "$3n$"),
    "unwrap": ("numel(input)", r"$\text{numel}(\text{input})$"),
    # stats distributions (not in numpy registry — these are scipy-compatible)
    "stats.norm.pdf": ("10n", r"$10n$"),
    "stats.norm.cdf": ("20n", r"$20n$"),
    "stats.norm.ppf": ("40n", r"$40n$"),
    "stats.uniform.pdf": ("3n", r"$3n$"),
    "stats.uniform.cdf": ("3n", r"$3n$"),
    "stats.uniform.ppf": ("3n", r"$3n$"),
    "stats.expon.pdf": ("5n", r"$5n$"),
    "stats.expon.cdf": ("5n", r"$5n$"),
    "stats.expon.ppf": ("5n", r"$5n$"),
    "stats.cauchy.pdf": ("6n", r"$6n$"),
    "stats.cauchy.cdf": ("5n", r"$5n$"),
    "stats.cauchy.ppf": ("5n", r"$5n$"),
    "stats.logistic.pdf": ("8n", r"$8n$"),
    "stats.logistic.cdf": ("5n", r"$5n$"),
    "stats.logistic.ppf": ("5n", r"$5n$"),
    "stats.laplace.pdf": ("5n", r"$5n$"),
    "stats.laplace.cdf": ("5n", r"$5n$"),
    "stats.laplace.ppf": ("5n", r"$5n$"),
    "stats.lognorm.pdf": ("15n", r"$15n$"),
    "stats.lognorm.cdf": ("25n", r"$25n$"),
    "stats.lognorm.ppf": ("45n", r"$45n$"),
    "stats.truncnorm.pdf": ("30n", r"$30n$"),
    "stats.truncnorm.cdf": ("30n", r"$30n$"),
    "stats.truncnorm.ppf": ("50n", r"$50n$"),
}

CATEGORY_COST_LATEX: dict[str, tuple[str, str]] = {
    "free": ("0", "$0$"),
    "counted_unary": ("numel(output)", r"$\text{numel}(\text{output})$"),
    "counted_binary": ("numel(output)", r"$\text{numel}(\text{output})$"),
    "counted_reduction": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "counted_custom": ("per-operation", "varies"),
    "counted_random_method": ("per-method formula", "varies — see cost_formula"),
    "free_random_method": ("0", "$0$"),
    "blacklisted": ("N/A", "N/A"),
}


# ---------------------------------------------------------------------------
# Operation doc model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OperationDocRecord:
    name: str
    canonical_name: str
    slug: str
    href: str
    canonical_path: str
    legacy_href: str
    module: str
    area: str
    flopscope_ref: str
    numpy_ref: str
    category: str
    display_type: str
    cost_formula: str
    cost_formula_latex: str
    weight: float
    notes: str
    aliases: list[str]
    signature: str
    summary: str = ""
    provenance_label: str = ""
    provenance_url: str = ""
    provenance_ref: str = ""
    upstream_ref_label: str = ""
    upstream_source_label: str = ""
    flopscope_source_url: str = ""
    upstream_source_url: str = ""
    parameters: list[DocField] | None = None
    returns: list[DocField] | None = None
    see_also: list[DocLink] | None = None
    notes_sections: list[str] | None = None
    example: DocExample | None = None
    body_sections: list[dict] | None = None
    doc_coverage: dict[str, object] = field(default_factory=dict)
    previous: OperationNavLink | None = None
    next: OperationNavLink | None = None
    api_docs_html: str = ""
    flopscope_examples_html: str = ""


_WORKER_ALIAS_MAP: dict[str, str] | None = None
_WORKER_SUPPORTED_OPS: set[str] | None = None
_WORKER_INTERNAL_REFS: dict[str, dict[str, str]] | None = None


@dataclass(slots=True)
class DocField:
    name: str
    type: str
    body: list[str]


@dataclass(slots=True)
class DocLink:
    label: str
    target: str
    description: str = ""
    description_inline: list[dict] | None = None
    role: str = ""
    original_target: str = ""
    unresolved: bool = False
    href: str | None = None
    external_url: str | None = None


@dataclass(slots=True)
class DocExample:
    code: str
    output: str = ""
    source: str = "derived"


@dataclass(slots=True)
class OperationNavLink:
    href: str
    label: str


@dataclass(slots=True)
class ParsedDoc:
    summary: str
    parameters: list[DocField]
    returns: list[DocField]
    see_also: list[DocLink]
    notes: list[str]
    note_lines: list[str]
    examples: list[DocExample]
    extended_summary: list[str] = field(default_factory=list)
    raises: list[DocField] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    warns: list[DocField] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    upstream_signature: str = ""
    sections: list[dict] | None = None
    coverage: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RelatedGuideLink:
    title: str
    href: str


@dataclass(slots=True)
class PublicApiSymbolRecord:
    name: str
    canonical_name: str
    slug: str
    href: str
    canonical_path: str
    kind: str
    module: str
    import_path: str
    display_name: str
    summary: str
    signature: str
    aliases: list[str]
    source_url: str = ""
    upstream_source_url: str = ""
    related_guides: list[RelatedGuideLink] = field(default_factory=list)
    members: list[OperationNavLink] = field(default_factory=list)
    status_note: str = ""
    body_sections: list[dict] | None = None


# ---------------------------------------------------------------------------
# Helper functions for op references and cost lookup
# ---------------------------------------------------------------------------


def normalize_area(module: str) -> str:
    """Map a registry module name to its doc area."""
    if module in {"numpy", "flopscope"}:
        return "core"
    if module.startswith("numpy."):
        return module.removeprefix("numpy.")
    if module == "flopscope.stats":
        return "stats"
    return "core"


def slug_for_operation(name: str) -> str:
    """Return a URL-safe slug for an operation name."""
    return name.replace(".", "-")


def detail_href_for_slug(slug: str) -> str:
    """Return the legacy docs route for a generated operation page."""
    return f"/docs/api/ops/{slug}/"


def detail_json_href_for_slug(slug: str) -> str:
    """Return the public JSON payload route for a generated operation page."""
    return f"/api-data/ops/{slug}.json"


CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def slug_for_api_segment(name: str) -> str:
    """Return a URL-safe slug for one public API path segment."""
    normalized = name.replace("_", "-")
    normalized = CAMEL_BOUNDARY_RE.sub("-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.lower()


def canonical_api_path_for_name(name: str) -> str:
    """Return the canonical docs path key for a public API symbol."""
    return "/".join(slug_for_api_segment(part) for part in name.split("."))


def canonical_api_href_for_name(name: str) -> str:
    """Return the canonical docs route for a public API symbol."""
    return f"/docs/api/{canonical_api_path_for_name(name)}/"


def slug_for_symbol(name: str) -> str:
    """Return a URL-safe slug for a public API symbol."""
    normalized = name.replace(".", "-").replace("_", "-")
    normalized = CAMEL_BOUNDARY_RE.sub("-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.lower()


PUBLIC_API_TOP_LEVEL = (
    # flopscope-specific primitives (live at ``flopscope.X``)
    "BudgetContext",
    "FlopscopeArray",
    "OpRecord",
    "PathInfo",
    "StepInfo",
    "SymmetricTensor",
    "SymmetryGroup",
    "as_symmetric",
    "budget",
    "budget_live",
    "budget_reset",
    "budget_summary",
    "budget_summary_dict",
    "configure",
    "is_symmetric",
    "namespace",
    "symmetrize",
    # NumPy-shaped helpers exposed under ``flopscope.numpy.X`` that we still
    # surface as canonical top-level documentation entries.
    "base_repr",
    "binary_repr",
    "broadcast",
    "clear_einsum_cache",
    "einsum_cache_info",
    "errstate",
    "finfo",
    "fromfile",
    "fromregex",
    "fromstring",
    "get_printoptions",
    "geterr",
    "iinfo",
    "isnat",
    "ndenumerate",
    "ndindex",
    "nditer",
    "printoptions",
    "set_printoptions",
    "seterr",
)

PUBLIC_API_MODULES = ("random", "stats", "flops", "testing")

# Maps the user-facing module label (e.g. "random") to the actual python
# module path (e.g. "flopscope.numpy.random"). The labels match the canonical
# names used in URLs and aliases (``flops.random.X`` → ``flopscope.numpy.random.X``).
PUBLIC_API_MODULE_PATHS: dict[str, str] = {
    "random": "flopscope.numpy.random",
    "testing": "flopscope.numpy.testing",
    "stats": "flopscope.stats",
    "flops": "flopscope.accounting",
}


def _resolve_public_api_module(module_label: str):
    """Resolve a public API module label to its concrete Python module."""
    module_path = PUBLIC_API_MODULE_PATHS.get(module_label, f"flopscope.{module_label}")
    return _import_local_flopscope_module(module_path)


def _public_api_segment_slug(segment: str) -> str:
    """Normalize one import-path segment for canonical public API routes."""
    normalized = segment.replace("_", "-")
    normalized = CAMEL_BOUNDARY_RE.sub("-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.lower()


def canonical_api_href(import_path: str) -> str:
    """Return the canonical public API docs href for an import path."""
    # Accept user-facing aliases like ``flops.stats.norm.pdf`` in addition to
    # the canonical ``flopscope.stats.norm.pdf`` form. ``flops.`` is the public
    # alias for ``flopscope.`` and ``flops.X.Y`` rewrites to ``flopscope.X.Y``
    # for url derivation.
    if import_path.startswith("flops."):
        import_path = "flopscope." + import_path.removeprefix("flops.")

    prefix = "flopscope."
    if not import_path.startswith(prefix):
        raise ValueError(f"Unsupported import path: {import_path}")

    remainder = import_path.removeprefix(prefix)
    if not remainder:
        raise ValueError(f"Unsupported import path: {import_path}")

    namespace: list[str]
    if remainder == "numpy" or remainder.startswith("numpy."):
        tail = remainder.removeprefix("numpy")
        namespace = ["numpy"]
    elif remainder == "stats" or remainder.startswith("stats."):
        tail = remainder.removeprefix("stats")
        namespace = ["stats"]
    elif remainder == "accounting" or remainder.startswith("accounting."):
        tail = remainder.removeprefix("accounting")
        namespace = ["accounting"]
    else:
        tail = remainder
        namespace = ["flopscope"]

    parts = namespace + [part for part in tail.lstrip(".").split(".") if part]
    return f"/docs/api/{'/'.join(_public_api_segment_slug(part) for part in parts)}/"


def display_type_for_category(category: str) -> str:
    """Return the UI display type for a registry category."""
    if category == "blacklisted":
        return "blocked"
    if category == "free" or category == "free_random_method":
        return "free"
    if category.startswith("counted_") and category != "counted_custom":
        return "counted"
    return "custom"


def flopscope_ref(name: str, module: str) -> str:
    """Derive the flopscope call reference from an op name and registry module."""
    if module == "numpy.linalg":
        return f"fnp.linalg.{name.removeprefix('linalg.')}"
    if module == "numpy.fft":
        return f"fnp.fft.{name.removeprefix('fft.')}"
    if module == "numpy.random":
        return f"fnp.random.{name.removeprefix('random.')}"
    if module == "flopscope.stats":
        return f"flops.{name}"
    return f"fnp.{name}"


def numpy_ref(name: str, module: str) -> str:
    """Derive the NumPy call reference from an op name and registry module."""
    if module == "numpy.linalg":
        return f"np.linalg.{name.removeprefix('linalg.')}"
    if module == "numpy.fft":
        return f"np.fft.{name.removeprefix('fft.')}"
    if module == "numpy.random":
        return f"np.random.{name.removeprefix('random.')}"
    if module == "flopscope.stats":
        return f"scipy.{name}"
    return f"np.{name}"


def provenance_label_for_operation(module: str) -> str:
    """Return the upstream docs attribution label for an operation page."""
    if module == "flopscope.stats":
        return "Adapted from SciPy docs"
    return "Adapted from NumPy docs"


def provenance_ref_for_operation(name: str, module: str) -> str:
    """Return the upstream reference label that matches the docs URL."""
    if module == "flopscope.stats":
        return f"scipy.stats.{name.split('.')[1]}"
    return numpy_ref(name, module)


def upstream_ref_label_for_operation(module: str) -> str:
    """Return the metadata label for the upstream API reference."""
    if module == "flopscope.stats":
        return "SciPy Ref"
    return "NumPy Ref"


def upstream_source_label_for_operation(module: str) -> str:
    """Return the metadata label for the upstream source link."""
    if module == "flopscope.stats":
        return "scipy source"
    return "numpy source"


PUBLIC_SYMBOL_GUIDES: dict[str, list[tuple[str, str]]] = {
    "BudgetContext": [
        ("Budget Planning & Debugging", "/docs/guides/budget-planning"),
        ("How Flopscope works", "/docs/understanding/how-flopscope-works"),
    ],
    "OpRecord": [("Budget Planning & Debugging", "/docs/guides/budget-planning")],
    "BudgetExhaustedError": [
        ("Budget Planning & Debugging", "/docs/guides/budget-planning"),
        ("Competition Quickstart", "/docs/getting-started/competition"),
    ],
    "TimeExhaustedError": [
        ("Competition Quickstart", "/docs/getting-started/competition")
    ],
    "NoBudgetContextError": [
        ("For Agents", "/docs/api/for-agents"),
        ("Quickstart", "/docs/getting-started/quickstart"),
    ],
    "SymmetryError": [("Symmetry Savings", "/docs/guides/symmetry")],
    "SymmetryLossWarning": [("Symmetry Savings", "/docs/guides/symmetry")],
    "SymmetricTensor": [
        ("Symmetry Savings", "/docs/guides/symmetry"),
        ("Symmetry Detection Deep Dive", "/docs/understanding/symmetry-detection"),
    ],
    "SymmetryInfo": [
        ("Symmetry Savings", "/docs/guides/symmetry"),
        ("Symmetry Detection Deep Dive", "/docs/understanding/symmetry-detection"),
    ],
    "as_symmetric": [("Symmetry Savings", "/docs/guides/symmetry")],
    "is_symmetric": [("Symmetry Savings", "/docs/guides/symmetry")],
    "PathInfo": [("Einsum Guide", "/docs/guides/einsum")],
    "StepInfo": [("Einsum Guide", "/docs/guides/einsum")],
}


COST_FORMULA_LABELS: dict[str, tuple[str, str]] = {
    "numel(output)": ("numel(output)", r"$\text{numel}(\text{output})$"),
    "numel(input)": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "shape[axis]": ("shape[axis]", r"$\text{shape}[\text{axis}]$"),
    "length": ("length", r"$\text{length}$"),
    "sort_cost(n)": ("sort_cost(n)", r"$n \cdot \lceil \log_2 n \rceil$"),
    "choice_cost": (
        "numel(output) if replace, else sort_cost(n)",
        r"$\text{numel}(\text{output})$ if replace, else $n \cdot \lceil \log_2 n \rceil$",
    ),
}


def cost_for_op(name: str, category: str) -> tuple[str, str]:
    """Return the cost-string + LaTeX-string pair for an op.

    For method-level random ops, defer to the per-op ``cost_formula`` field
    in the registry; for everything else, use the per-category fallback.
    """
    if category == "free_random_method":
        return ("0", "$0$")
    if category == "counted_random_method":
        from flopscope._registry import REGISTRY

        formula = REGISTRY.get(name, {}).get("cost_formula", "")
        return COST_FORMULA_LABELS.get(formula, ("per-method", "varies"))
    if name in CUSTOM_COSTS:
        return CUSTOM_COSTS[name]
    return CATEGORY_COST_LATEX.get(category, ("unknown", "unknown"))


NUMPY_REF_PATTERN = re.compile(r"\b(?:np|numpy)\.(?=[A-Za-z_])")


def rewrite_api_refs(text: str) -> str:
    """Rewrite NumPy API references to their flopscope equivalents."""
    return NUMPY_REF_PATTERN.sub("flops.", text)


def public_doc_contract_rules() -> dict[str, object]:
    return {
        "require_examples_for": [
            "flopscope.BudgetContext",
            "flopscope.budget",
            "flopscope.budget_live",
            "flopscope.budget_summary",
            "flopscope.budget_summary_dict",
            "flopscope.budget_reset",
            "flopscope.configure",
            "flopscope.namespace",
            "flopscope.numpy.clear_einsum_cache",
            "flopscope.numpy.einsum_cache_info",
            "flopscope.as_symmetric",
            "flopscope.is_symmetric",
            "flopscope.symmetrize",
            "flopscope.numpy.random.symmetric",
            "flopscope.accounting.einsum_cost",
            "flopscope.accounting.pointwise_cost",
            "flopscope.accounting.reduction_cost",
            "flopscope.accounting.svd_cost",
        ],
        "require_parameters_and_returns_for_kind": {"function", "method"},
        "stale_alias_patterns": [
            r"\bimport flopscope as we\b",
            r"\bimport whest\b",
            r"\bwe\.",
        ],
    }


def _requires_structured_public_callable_docs(import_path: str) -> bool:
    if import_path.startswith("flopscope.stats."):
        return True
    if import_path.startswith("flopscope.accounting."):
        return True
    return import_path in {
        "flopscope.BudgetContext",
        "flopscope.budget",
        "flopscope.budget_live",
        "flopscope.budget_reset",
        "flopscope.budget_summary",
        "flopscope.budget_summary_dict",
        "flopscope.configure",
        "flopscope.namespace",
        "flopscope.as_symmetric",
        "flopscope.is_symmetric",
        "flopscope.symmetrize",
        "flopscope.numpy.clear_einsum_cache",
        "flopscope.numpy.einsum_cache_info",
        "flopscope.numpy.random.symmetric",
    }


def find_public_doc_contract_violations(
    *,
    import_path: str,
    kind: str,
    summary: str,
    sections: dict[str, list[str]],
    require_parameters: bool = True,
) -> list[str]:
    rules = public_doc_contract_rules()
    violations: list[str] = []

    if not summary.strip():
        violations.append(f"{import_path}: missing summary")

    if kind in rules[
        "require_parameters_and_returns_for_kind"
    ] and _requires_structured_public_callable_docs(import_path):
        if require_parameters and not sections.get("Parameters"):
            violations.append(f"{import_path}: missing Parameters section")
        if not sections.get("Returns"):
            violations.append(f"{import_path}: missing Returns section")

    if import_path in set(rules["require_examples_for"]) and not sections.get(
        "Examples"
    ):
        violations.append(f"{import_path}: missing Examples section")

    for line in sections.get("Examples", []):
        for pattern in rules["stale_alias_patterns"]:
            if re.search(pattern, line):
                violations.append(f"{import_path}: stale alias in examples -> {line}")

    return violations


def _split_paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.rstrip())
            continue
        if current:
            paragraphs.append(" ".join(part.strip() for part in current).strip())
            current = []
    if current:
        paragraphs.append(" ".join(part.strip() for part in current).strip())
    return paragraphs


RST_ROLE_SCAN_PATTERN = re.compile(r":(?P<role>(?:py:)?[A-Za-z0-9_]+):`")
ROLE_TARGET_PATTERN = re.compile(r"^(?P<title>.+?)\s*<(?P<target>.+)>$")
_REGISTERED_RST_ROLES: set[str] = set()


def _new_doc_coverage() -> dict[str, list[dict]]:
    return {
        "unresolved_references": [],
        "unsupported_directives": [],
        "raw_blocks": [],
    }


def _record_coverage_event(
    coverage: dict[str, object] | None, category: str, payload: dict[str, object]
) -> None:
    if coverage is None:
        return
    coverage.setdefault(category, []).append(payload)


def _ensure_rst_role_registered(role_name: str) -> None:
    if role_name in _REGISTERED_RST_ROLES or role_name == "math":
        return

    from docutils import nodes
    from docutils.parsers.rst import roles

    def _generic_role(  # type: ignore[override]
        role: str,
        rawtext: str,
        text: str,
        lineno: int,
        inliner,
        options=None,
        content=None,
    ):
        node = nodes.inline(rawtext, text)
        node["role_name"] = role
        return [node], []

    roles.register_local_role(role_name, _generic_role)
    _REGISTERED_RST_ROLES.add(role_name)


def _register_roles_from_text(text: str) -> None:
    for match in RST_ROLE_SCAN_PATTERN.finditer(text):
        _ensure_rst_role_registered(match.group("role"))


def _normalize_reference_target(target: str) -> str:
    normalized = " ".join(target.strip().split())
    if normalized.startswith("flopscope."):
        return f"flops.{normalized.removeprefix('flopscope.')}"
    return normalized


def _operation_candidate_from_target(target: str) -> str:
    normalized = _normalize_reference_target(target)
    if normalized.startswith("flops."):
        return normalized.removeprefix("flops.")
    if normalized.startswith("numpy."):
        return normalized.removeprefix("numpy.")
    if normalized.startswith("np."):
        return normalized.removeprefix("np.")
    if normalized.startswith("scipy.stats."):
        return f"stats.{normalized.removeprefix('scipy.stats.')}"
    return normalized


def _docs_url_for_reference_target(target: str, *, role: str = "") -> str:
    normalized = _normalize_reference_target(rewrite_api_refs(target))

    if role == "pep":
        digits = re.sub(r"\D", "", normalized)
        if digits:
            return f"https://peps.python.org/pep-{int(digits):04d}/"

    if normalized.startswith(("scipy.", "stats.")):
        scipy_target = (
            normalized
            if normalized.startswith("scipy.")
            else f"scipy.stats.{normalized.removeprefix('stats.')}"
        )
        return (
            f"https://docs.scipy.org/doc/scipy/reference/generated/{scipy_target}.html"
        )

    candidate = _operation_candidate_from_target(normalized)
    if not candidate:
        return ""

    if candidate.startswith(("linalg.", "fft.", "random.", "ufunc.")):
        return (
            f"https://numpy.org/doc/stable/reference/generated/numpy.{candidate}.html"
        )

    if candidate.startswith("stats."):
        return (
            "https://docs.scipy.org/doc/scipy/reference/generated/"
            f"scipy.stats.{candidate.removeprefix('stats.')}.html"
        )

    if "." not in candidate:
        return (
            f"https://numpy.org/doc/stable/reference/generated/numpy.{candidate}.html"
        )

    return ""


def _resolve_reference_destination(
    target: str,
    *,
    original_target: str = "",
    role: str = "",
    alias_map: dict[str, str] | None = None,
    supported_ops: set[str] | None = None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> tuple[str, str, str]:
    alias_map = alias_map or {}
    supported_ops = supported_ops or set()
    internal_refs = internal_refs or {}
    normalized_target = _normalize_reference_target(target)

    lookup_candidates = [normalized_target]
    candidate = _operation_candidate_from_target(target)
    if candidate and candidate not in lookup_candidates:
        lookup_candidates.append(candidate)
    if (
        normalized_target.startswith("flops.")
        and normalized_target.removeprefix("flops.") not in lookup_candidates
    ):
        lookup_candidates.append(normalized_target.removeprefix("flops."))
    if (
        normalized_target.startswith("flopscope.")
        and normalized_target.removeprefix("flopscope.") not in lookup_candidates
    ):
        lookup_candidates.append(normalized_target.removeprefix("flopscope."))

    for lookup in lookup_candidates:
        ref = internal_refs.get(lookup)
        if ref:
            return ref["href"], "", ref["canonical_name"]

    canonical = resolve_canonical_name(candidate, alias_map)
    canonical_ref = internal_refs.get(canonical)
    if canonical_ref:
        # Found via the canonical name even though the original lookup_candidates
        # didn't contain it (e.g. target was a short alias like "abs" that
        # didn't appear in internal_refs because alias_map injection ran after).
        return canonical_ref["href"], "", canonical_ref["canonical_name"]
    if canonical in supported_ops:
        # Defensive fallback. If we reach here, the op is in supported_ops but
        # not in internal_refs — meaning some construction path didn't include
        # it. The canonical_api_href format is the right post-rebrand route;
        # the previous /docs/api/ops/<slug> form was the pre-rebrand path
        # and now produces broken links on deploy.
        import_path = (
            f"flopscope.numpy.{slug_for_operation(canonical).replace('-', '_')}"
        )
        return (
            canonical_api_href(import_path),
            "",
            canonical,
        )

    external_url = _docs_url_for_reference_target(original_target or target, role=role)
    return "", external_url, canonical


def _parse_role_payload(payload: str) -> tuple[str, str | None, bool, bool]:
    normalized = " ".join(payload.strip().split())
    suppress_link = normalized.startswith("!")
    if suppress_link:
        normalized = normalized[1:].lstrip()

    explicit_title: str | None = None
    target = normalized
    explicit_match = ROLE_TARGET_PATTERN.match(normalized)
    if explicit_match:
        explicit_title = explicit_match.group("title").strip()
        target = explicit_match.group("target").strip()

    shorten_display = target.startswith("~")
    if shorten_display:
        target = target[1:]

    return target, explicit_title, suppress_link, shorten_display


def _linkish_inline_node(
    token: str,
    *,
    alias_map: dict[str, str] | None = None,
    supported_ops: set[str] | None = None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> dict | None:
    target, explicit_title, suppress_link, shorten_display = _parse_role_payload(token)
    if suppress_link:
        return None

    rewritten_target = _normalize_reference_target(rewrite_api_refs(target))
    candidate = _operation_candidate_from_target(rewritten_target)
    canonical_candidate = resolve_canonical_name(candidate, alias_map or {})
    api_like = "." in rewritten_target or target.lstrip("~").startswith(
        ("np.", "numpy.", "flops.", "flopscope.", "scipy.")
    )
    if canonical_candidate not in (supported_ops or set()) and not api_like:
        return None

    href, external_url, canonical = _resolve_reference_destination(
        rewritten_target,
        original_target=target,
        alias_map=alias_map,
        supported_ops=supported_ops,
        internal_refs=internal_refs,
    )
    if not href and not external_url:
        return None

    display_text = (
        rewrite_api_refs(explicit_title)
        if explicit_title
        else rewritten_target.split(".")[-1]
        if shorten_display
        else rewrite_api_refs(target)
    )
    return {
        "kind": "link",
        "text": display_text,
        "target": canonical,
        "href": href,
        "external_url": external_url,
    }


def _inline_text_from_children(children: list[object]) -> str:
    parts: list[str] = []
    for child in children:
        try:
            parts.append(child.astext())
        except AttributeError:
            parts.append(str(child))
    return "".join(parts)


def _convert_docutils_inline_node(
    node: object,
    *,
    coverage: dict[str, object] | None = None,
    supported_ops: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    from docutils import nodes as docutils_nodes

    supported_ops = supported_ops or set()
    alias_map = alias_map or {}

    if isinstance(node, docutils_nodes.Text):
        return [{"kind": "text", "text": str(node)}]

    if isinstance(node, docutils_nodes.math):
        return [{"kind": "math", "latex": node.astext()}]

    if isinstance(node, docutils_nodes.reference):
        return [
            {
                "kind": "link",
                "text": node.astext(),
                "href": node.get("refuri", ""),
                "external_url": node.get("refuri", ""),
            }
        ]

    if isinstance(node, docutils_nodes.inline) and node.get("role_name"):
        role_name = node["role_name"]
        target, explicit_title, suppress_link, shorten_display = _parse_role_payload(
            node.astext()
        )
        rewritten_target = _normalize_reference_target(rewrite_api_refs(target))
        href, external_url, canonical = _resolve_reference_destination(
            rewritten_target,
            original_target=target,
            role=role_name,
            alias_map=alias_map,
            supported_ops=supported_ops,
            internal_refs=internal_refs,
        )
        display_text = (
            rewrite_api_refs(explicit_title)
            if explicit_title
            else rewritten_target.split(".")[-1]
            if shorten_display
            else rewrite_api_refs(target)
        )

        unresolved = not suppress_link and not href and not external_url
        if unresolved:
            _record_coverage_event(
                coverage,
                "unresolved_references",
                {
                    "role": role_name,
                    "target": rewritten_target,
                    "original_target": target,
                    "display_text": display_text,
                },
            )

        return [
            {
                "kind": "role_reference",
                "role": role_name,
                "target": canonical,
                "original_target": target,
                "display_text": display_text,
                "suppress_link": suppress_link,
                "explicit_title": explicit_title is not None,
                "href": "" if suppress_link else href,
                "external_url": "" if suppress_link else external_url,
                "unresolved": unresolved,
            }
        ]

    if isinstance(node, docutils_nodes.literal):
        token = node.astext()
        link_node = _linkish_inline_node(
            token,
            alias_map=alias_map,
            supported_ops=supported_ops,
            internal_refs=internal_refs,
        )
        if link_node is not None:
            return [link_node]
        return [{"kind": "code", "text": token}]

    if isinstance(node, docutils_nodes.title_reference):
        token = node.astext()
        link_node = _linkish_inline_node(
            token,
            alias_map=alias_map,
            supported_ops=supported_ops,
            internal_refs=internal_refs,
        )
        if link_node is not None:
            return [link_node]
        return [{"kind": "code", "text": token}]

    if isinstance(node, docutils_nodes.emphasis):
        return [
            {
                "kind": "emphasis",
                "text": _inline_text_from_children(list(node.children)),
            }
        ]

    if isinstance(node, docutils_nodes.strong):
        return [
            {"kind": "strong", "text": _inline_text_from_children(list(node.children))}
        ]

    _record_coverage_event(
        coverage,
        "raw_blocks",
        {
            "kind": "inline_node",
            "node_type": node.__class__.__name__,
            "raw_text": getattr(node, "astext", lambda: str(node))(),
        },
    )
    return [{"kind": "text", "text": getattr(node, "astext", lambda: str(node))()}]


def parse_inline_nodes(
    text: str,
    *,
    supported_ops: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    coverage: dict[str, object] | None = None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """Parse inline reST/Sphinx markup into a structured AST."""

    text = text.strip()
    if not text:
        return []

    from docutils.core import publish_doctree

    _register_roles_from_text(text)
    doctree = publish_doctree(
        text,
        settings_overrides={
            "warning_stream": io.StringIO(),
            "report_level": 5,
            "halt_level": 6,
        },
    )

    if not doctree.children:
        return [{"kind": "text", "text": text}]

    paragraph = doctree.children[0]
    children = getattr(paragraph, "children", doctree.children)

    nodes: list[dict] = []
    for child in children:
        nodes.extend(
            _convert_docutils_inline_node(
                child,
                coverage=coverage,
                supported_ops=supported_ops,
                alias_map=alias_map,
                internal_refs=internal_refs,
            )
        )
    return nodes


def _coalesce_lines(lines: list[str]) -> str:
    return " ".join(part.strip() for part in lines).strip()


def _is_prompt_line(line: str) -> tuple[bool, str | None]:
    stripped = line.lstrip()
    if stripped.startswith(">>>"):
        return True, ">>>"
    if stripped.startswith("..."):
        return True, "..."
    return False, None


def rewrite_example_text(text: str) -> str:
    """Rewrite upstream NumPy example text to the current flopscope naming."""
    rewritten = rewrite_api_refs(text)
    rewritten = rewritten.replace("import numpy as np", "import flopscope.numpy as fnp")
    return rewritten


def _paragraphs_from_lines(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    buffer: list[str] = []
    for raw in lines:
        line = raw.rstrip()
        if line.strip():
            buffer.append(line)
            continue

        if buffer:
            paragraphs.append(_coalesce_lines(buffer))
            buffer = []

    if buffer:
        paragraphs.append(_coalesce_lines(buffer))
    return paragraphs


DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+([A-Za-z0-9_-]+)::\s*(.*)$")
DIRECTIVE_OPTION_RE = re.compile(r"^\s*:([^:]+):\s*(.*)$")
STRUCTURED_DIRECTIVES = {
    "deprecated",
    "math",
    "note",
    "plot",
    "versionadded",
    "versionchanged",
    "warning",
}


def _language_from_literal_block(node: object) -> str | None:
    classes = list(getattr(node, "get", lambda *_args, **_kwargs: [])("classes", []))
    for class_name in classes:
        if class_name not in {"code", "literal", "literal-block"}:
            return class_name
    return None


def _parse_doctest_lines(text: str) -> list[dict]:
    doctest_lines: list[dict] = []
    for raw in text.splitlines():
        is_prompt, prompt = _is_prompt_line(raw)
        if is_prompt:
            doctest_lines.append(
                {
                    "kind": "input",
                    "prompt": prompt,
                    "text": raw.lstrip()[3:].strip(),
                }
            )
            continue
        doctest_lines.append({"kind": "output", "text": raw.rstrip()})
    return doctest_lines


def _convert_docutils_blocks(
    nodes_to_convert: list[object],
    *,
    supported_ops: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    coverage: dict[str, object] | None = None,
    example_mode: bool = False,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    from docutils import nodes as docutils_nodes

    blocks: list[dict] = []

    for node in nodes_to_convert:
        if isinstance(node, docutils_nodes.paragraph):
            blocks.append(
                {
                    "type": "paragraph",
                    "inline": [
                        inline_node
                        for child in node.children
                        for inline_node in _convert_docutils_inline_node(
                            child,
                            coverage=coverage,
                            supported_ops=supported_ops,
                            alias_map=alias_map,
                            internal_refs=internal_refs,
                        )
                    ],
                }
            )
            continue

        if isinstance(node, docutils_nodes.definition_list):
            items = []
            for item in node.children:
                if not isinstance(item, docutils_nodes.definition_list_item):
                    continue
                term_parts = []
                body_blocks: list[dict] = []
                for child in item.children:
                    if isinstance(
                        child, (docutils_nodes.term, docutils_nodes.classifier)
                    ):
                        for inline_child in child.children:
                            term_parts.extend(
                                _convert_docutils_inline_node(
                                    inline_child,
                                    coverage=coverage,
                                    supported_ops=supported_ops,
                                    alias_map=alias_map,
                                    internal_refs=internal_refs,
                                )
                            )
                    elif isinstance(child, docutils_nodes.definition):
                        body_blocks.extend(
                            _convert_docutils_blocks(
                                list(child.children),
                                supported_ops=supported_ops,
                                alias_map=alias_map,
                                coverage=coverage,
                                example_mode=example_mode,
                                internal_refs=internal_refs,
                            )
                        )
                items.append({"term_inline": term_parts, "blocks": body_blocks})
            blocks.append({"type": "definition_list", "items": items})
            continue

        if isinstance(node, docutils_nodes.bullet_list):
            blocks.append(
                {
                    "type": "list",
                    "ordered": False,
                    "items": [
                        {
                            "blocks": _convert_docutils_blocks(
                                list(item.children),
                                supported_ops=supported_ops,
                                alias_map=alias_map,
                                coverage=coverage,
                                example_mode=example_mode,
                                internal_refs=internal_refs,
                            )
                        }
                        for item in node.children
                        if isinstance(item, docutils_nodes.list_item)
                    ],
                }
            )
            continue

        if isinstance(node, docutils_nodes.enumerated_list):
            blocks.append(
                {
                    "type": "list",
                    "ordered": True,
                    "items": [
                        {
                            "blocks": _convert_docutils_blocks(
                                list(item.children),
                                supported_ops=supported_ops,
                                alias_map=alias_map,
                                coverage=coverage,
                                example_mode=example_mode,
                                internal_refs=internal_refs,
                            )
                        }
                        for item in node.children
                        if isinstance(item, docutils_nodes.list_item)
                    ],
                }
            )
            continue

        if isinstance(node, docutils_nodes.doctest_block):
            text = node.astext()
            if example_mode:
                text = rewrite_example_text(text)
            blocks.append(
                {"type": "doctest_block", "lines": _parse_doctest_lines(text)}
            )
            continue

        if isinstance(node, docutils_nodes.literal_block):
            text = node.astext()
            if example_mode:
                text = rewrite_example_text(text)
            blocks.append(
                {
                    "type": "literal_block",
                    "text": text,
                    "language": _language_from_literal_block(node),
                }
            )
            continue

        if isinstance(node, docutils_nodes.math_block):
            blocks.append({"type": "math_block", "formulas": [node.astext()]})
            continue

        if isinstance(node, docutils_nodes.block_quote):
            blocks.extend(
                _convert_docutils_blocks(
                    list(node.children),
                    supported_ops=supported_ops,
                    alias_map=alias_map,
                    coverage=coverage,
                    example_mode=example_mode,
                    internal_refs=internal_refs,
                )
            )
            continue

        if isinstance(node, (docutils_nodes.system_message, docutils_nodes.table)):
            raw_text = node.astext()
            _record_coverage_event(
                coverage,
                "raw_blocks",
                {
                    "kind": node.__class__.__name__,
                    "raw_text": raw_text,
                },
            )
            blocks.append(
                {
                    "type": "raw_block",
                    "raw_kind": node.__class__.__name__,
                    "raw_text": raw_text,
                }
            )
            continue

        raw_text = getattr(node, "astext", lambda node=node: str(node))()
        _record_coverage_event(
            coverage,
            "raw_blocks",
            {
                "kind": node.__class__.__name__,
                "raw_text": raw_text,
            },
        )
        blocks.append(
            {
                "type": "raw_block",
                "raw_kind": node.__class__.__name__,
                "raw_text": raw_text,
            }
        )

    return blocks


def _parse_standard_rst_blocks(
    lines: list[str],
    *,
    supported_ops: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    coverage: dict[str, object] | None = None,
    example_mode: bool = False,
    text_transform=None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    if not any(line.strip() for line in lines):
        return []

    from docutils.core import publish_doctree

    source = "\n".join(lines).strip("\n")
    if text_transform is not None:
        source = text_transform(source)
    _register_roles_from_text(source)
    doctree = publish_doctree(
        source,
        settings_overrides={
            "warning_stream": io.StringIO(),
            "report_level": 5,
            "halt_level": 6,
        },
    )
    return _convert_docutils_blocks(
        list(doctree.children),
        supported_ops=supported_ops,
        alias_map=alias_map,
        coverage=coverage,
        example_mode=example_mode,
        internal_refs=internal_refs,
    )


def _consume_directive_block(
    lines: list[str], index: int
) -> tuple[str, str, list[str], int]:
    raw = lines[index].rstrip("\n")
    match = DIRECTIVE_RE.match(raw)
    if not match:
        raise ValueError(f"Expected directive line, got: {raw!r}")

    directive_name = match.group(1)
    remainder = match.group(2).strip()
    index += 1
    payload: list[str] = []
    total = len(lines)
    while index < total:
        current = lines[index].rstrip("\n")
        if current.startswith((" ", "\t")) or not current.strip():
            payload.append(current)
            index += 1
            continue
        break
    return directive_name, remainder, payload, index


def _split_directive_payload(
    payload: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    if not payload:
        return [], []

    dedented = textwrap.dedent("\n".join(payload)).splitlines()
    options: list[dict[str, str]] = []
    index = 0
    while index < len(dedented):
        line = dedented[index]
        if not line.strip():
            index += 1
            if options:
                break
            continue
        match = DIRECTIVE_OPTION_RE.match(line)
        if not match:
            break
        options.append({"name": match.group(1), "value": match.group(2)})
        index += 1

    content_lines = dedented[index:]
    while content_lines and not content_lines[0].strip():
        content_lines = content_lines[1:]
    while content_lines and not content_lines[-1].strip():
        content_lines = content_lines[:-1]
    return options, content_lines


def _math_formulas_from_directive(
    remainder: str, content_lines: list[str]
) -> list[str]:
    paragraphs: list[list[str]] = []
    current: list[str] = [remainder] if remainder else []
    for line in content_lines:
        if line.strip():
            current.append(line.strip())
            continue
        if current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    return [
        _coalesce_lines(paragraph)
        for paragraph in paragraphs
        if _coalesce_lines(paragraph)
    ]


def _build_directive_block(
    directive_name: str,
    remainder: str,
    payload: list[str],
    *,
    supported_ops: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    coverage: dict[str, object] | None = None,
    example_mode: bool = False,
    text_transform=None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> dict:
    supported_ops = supported_ops or set()
    alias_map = alias_map or {}
    options, content_lines = _split_directive_payload(payload)
    argument_text = (
        text_transform(remainder) if text_transform is not None else remainder
    )

    if directive_name == "math":
        formulas = _math_formulas_from_directive(argument_text, content_lines)
        return {"type": "math_block", "formulas": formulas}

    if directive_name in {"versionadded", "versionchanged", "deprecated"}:
        version = ""
        detail_text = argument_text
        if argument_text:
            parts = argument_text.split(maxsplit=1)
            version = parts[0]
            detail_text = parts[1] if len(parts) > 1 else ""
        return {
            "type": "directive_block",
            "directive": directive_name,
            "version": version,
            "argument_inline": parse_inline_nodes(
                detail_text,
                supported_ops=supported_ops,
                alias_map=alias_map,
                coverage=coverage,
                internal_refs=internal_refs,
            ),
            "options": options,
            "content_blocks": [],
            "supported": True,
            "raw_source": "\n".join(
                [
                    f".. {directive_name}::{(' ' + remainder) if remainder else ''}".rstrip()
                ]
                + payload
            ).strip("\n"),
        }

    if directive_name in {"note", "warning"}:
        return {
            "type": "directive_block",
            "directive": directive_name,
            "argument_inline": parse_inline_nodes(
                argument_text,
                supported_ops=supported_ops,
                alias_map=alias_map,
                coverage=coverage,
                internal_refs=internal_refs,
            )
            if argument_text
            else [],
            "options": options,
            "content_blocks": _parse_standard_rst_blocks(
                content_lines,
                supported_ops=supported_ops,
                alias_map=alias_map,
                coverage=coverage,
                example_mode=example_mode,
                text_transform=text_transform,
                internal_refs=internal_refs,
            ),
            "supported": True,
            "raw_source": "\n".join(
                [
                    f".. {directive_name}::{(' ' + remainder) if remainder else ''}".rstrip()
                ]
                + payload
            ).strip("\n"),
        }

    if directive_name == "plot":
        plot_source = "\n".join(content_lines).strip("\n")
        if text_transform is not None and plot_source:
            plot_source = text_transform(plot_source)
        return {
            "type": "directive_block",
            "directive": directive_name,
            "argument_inline": parse_inline_nodes(
                argument_text,
                supported_ops=supported_ops,
                alias_map=alias_map,
                coverage=coverage,
                internal_refs=internal_refs,
            )
            if argument_text
            else [],
            "options": options,
            "content_blocks": (
                [{"type": "literal_block", "text": plot_source, "language": "python"}]
                if plot_source
                else []
            ),
            "supported": True,
            "raw_source": "\n".join(
                [
                    f".. {directive_name}::{(' ' + remainder) if remainder else ''}".rstrip()
                ]
                + payload
            ).strip("\n"),
        }

    raw_source = "\n".join(
        [f".. {directive_name}::{(' ' + remainder) if remainder else ''}".rstrip()]
        + payload
    ).strip("\n")
    _record_coverage_event(
        coverage,
        "unsupported_directives",
        {
            "directive": directive_name,
            "raw_source": raw_source,
        },
    )
    return {
        "type": "directive_block",
        "directive": directive_name,
        "argument_inline": parse_inline_nodes(
            argument_text,
            supported_ops=supported_ops,
            alias_map=alias_map,
            coverage=coverage,
            internal_refs=internal_refs,
        )
        if argument_text
        else [],
        "options": options,
        "content_blocks": [],
        "supported": False,
        "raw_source": raw_source,
    }


def _parse_examples_to_blocks(
    lines: list[str],
    *,
    supported_ops: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    coverage: dict[str, object] | None = None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """Convert Examples lines into a structured AST."""
    return _parse_rich_doc_blocks(
        lines,
        supported_ops=supported_ops,
        alias_map=alias_map,
        coverage=coverage,
        example_mode=True,
        text_transform=rewrite_example_text,
        internal_refs=internal_refs,
    )


def _parse_notes_to_blocks(
    lines: list[str],
    *,
    supported_ops: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    coverage: dict[str, object] | None = None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """Convert Notes lines into a structured AST."""
    return _parse_rich_doc_blocks(
        lines,
        supported_ops=supported_ops,
        alias_map=alias_map,
        coverage=coverage,
        internal_refs=internal_refs,
    )


def _parse_rich_doc_blocks(
    lines: list[str],
    *,
    supported_ops: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    coverage: dict[str, object] | None = None,
    example_mode: bool = False,
    text_transform=None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    blocks: list[dict] = []
    index = 0
    total = len(lines)
    buffer: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        blocks.extend(
            _parse_standard_rst_blocks(
                buffer,
                supported_ops=supported_ops,
                alias_map=alias_map,
                coverage=coverage,
                example_mode=example_mode,
                text_transform=text_transform,
                internal_refs=internal_refs,
            )
        )
        buffer = []

    while index < total:
        raw = lines[index].rstrip("\n")
        if DIRECTIVE_RE.match(raw):
            flush_buffer()
            directive_name, remainder, payload, index = _consume_directive_block(
                lines, index
            )
            blocks.append(
                _build_directive_block(
                    directive_name,
                    remainder,
                    payload,
                    supported_ops=supported_ops,
                    alias_map=alias_map,
                    coverage=coverage,
                    example_mode=example_mode,
                    text_transform=text_transform,
                    internal_refs=internal_refs,
                )
            )
            continue

        buffer.append(raw)
        index += 1

    flush_buffer()

    return blocks


def _parse_field_body_to_blocks(
    lines: list[str],
    *,
    supported_ops: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    coverage: dict[str, object] | None = None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """Convert parameter/return body lines into a structured AST."""
    return _parse_rich_doc_blocks(
        lines,
        supported_ops=supported_ops,
        alias_map=alias_map,
        coverage=coverage,
        internal_refs=internal_refs,
    )


def _build_field_list_block(
    title: str,
    fields: list[DocField],
    *,
    supported_ops: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    coverage: dict[str, object] | None = None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> dict | None:
    if not fields:
        return None

    def preview_inline(body_blocks: list[dict]) -> list[dict]:
        for block in body_blocks:
            if block.get("type") == "paragraph":
                return list(block.get("inline", []))
        return []

    items = [
        {
            "type": "field_list",
            "name": field.name,
            "data_type": field.type,
            "inline": preview_inline(
                body_blocks := _parse_field_body_to_blocks(
                    field.body,
                    supported_ops=supported_ops,
                    alias_map=alias_map,
                    coverage=coverage,
                    internal_refs=internal_refs,
                )
            ),
            "body_blocks": body_blocks,
        }
        for field in fields
    ]

    return {
        "title": title,
        "blocks": [{"type": "field_list", "title": title, "items": items}],
    }


def _build_sections_for_doc(
    parsed: ParsedDoc,
    *,
    supported_ops: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    sections: list[dict] = []
    coverage = parsed.coverage

    def build_text_section(
        title: str,
        lines: list[str],
        *,
        example_mode: bool = False,
        text_transform=None,
    ) -> None:
        if not lines:
            return
        section_blocks = _parse_rich_doc_blocks(
            lines,
            supported_ops=supported_ops,
            alias_map=alias_map,
            coverage=coverage,
            example_mode=example_mode,
            text_transform=text_transform,
            internal_refs=internal_refs,
        )
        if section_blocks:
            sections.append({"title": title, "blocks": section_blocks})

    if parsed.summary:
        summary_inline = parse_inline_nodes(
            parsed.summary,
            supported_ops=supported_ops,
            alias_map=alias_map,
            coverage=coverage,
            internal_refs=internal_refs,
        )
        if summary_inline:
            sections.append(
                {
                    "title": "Summary",
                    "blocks": [{"type": "paragraph", "inline": summary_inline}],
                }
            )

    build_text_section("Extended Summary", parsed.extended_summary)

    parameters_block = _build_field_list_block(
        "Parameters",
        parsed.parameters,
        supported_ops=supported_ops,
        alias_map=alias_map,
        coverage=coverage,
        internal_refs=internal_refs,
    )
    if parameters_block:
        sections.append(parameters_block)

    returns_block = _build_field_list_block(
        "Returns",
        parsed.returns,
        supported_ops=supported_ops,
        alias_map=alias_map,
        coverage=coverage,
        internal_refs=internal_refs,
    )
    if returns_block:
        sections.append(returns_block)

    raises_block = _build_field_list_block(
        "Raises",
        parsed.raises,
        supported_ops=supported_ops,
        alias_map=alias_map,
        coverage=coverage,
        internal_refs=internal_refs,
    )
    if raises_block:
        sections.append(raises_block)

    warns_block = _build_field_list_block(
        "Warns",
        parsed.warns,
        supported_ops=supported_ops,
        alias_map=alias_map,
        coverage=coverage,
        internal_refs=internal_refs,
    )
    if warns_block:
        sections.append(warns_block)

    if parsed.see_also:
        sections.append(
            {
                "title": "See also",
                "blocks": [
                    {
                        "type": "link_list",
                        "links": [asdict(link) for link in parsed.see_also],
                    }
                ],
            }
        )

    note_lines = list(parsed.note_lines)
    if note_lines:
        note_blocks = _parse_notes_to_blocks(
            note_lines,
            supported_ops=supported_ops,
            alias_map=alias_map,
            coverage=coverage,
            internal_refs=internal_refs,
        )
        if note_blocks:
            sections.append({"title": "Notes", "blocks": note_blocks})

    build_text_section("Warnings", parsed.warnings)
    build_text_section("References", parsed.references)

    if parsed.examples:
        example_lines = [
            line.rstrip("\n") for line in parsed.examples[0].code.split("\n")
        ]
        example_blocks = _parse_examples_to_blocks(
            example_lines,
            supported_ops=supported_ops,
            alias_map=alias_map,
            coverage=coverage,
            internal_refs=internal_refs,
        )
        if example_blocks:
            sections.append({"title": "Examples", "blocks": example_blocks})

    return sections


def derive_example_from_upstream(example_block: str) -> DocExample:
    """Derive a single flopscope example from a doctest-style upstream snippet."""
    parser = doctest.DocTestParser()
    parsed = parser.get_doctest(
        textwrap.dedent(example_block).strip("\n"), {}, "flopscope-example", "", 0
    )

    code_parts: list[str] = []
    output_parts: list[str] = []
    for example in parsed.examples:
        source = rewrite_example_text(example.source).strip()
        if source:
            code_parts.append(source.rstrip())

        want = rewrite_example_text(example.want).strip()
        if want:
            output_parts.append(want.rstrip())

    code = "\n\n".join(code_parts).strip()
    output = "\n\n".join(output_parts).strip()
    return DocExample(code=code, output=output)


def parse_numpy_docstring(raw_doc: str) -> ParsedDoc:
    """Parse a NumPy-style docstring into a structured internal model."""
    try:
        from numpydoc.docscrape import NumpyDocString
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("numpydoc is required to parse upstream docstrings") from exc

    doc = NumpyDocString(textwrap.dedent(raw_doc).strip("\n"))

    summary = " ".join(part.strip() for part in doc["Summary"]).strip()
    upstream_signature = str(doc["Signature"]).strip()

    def parse_field_section(
        section_name: str, *, single_element_is_type: bool = False
    ) -> list[DocField]:
        return [
            DocField(name=name, type=type_, body=[line.rstrip() for line in desc])
            for name, type_, desc in doc[section_name]
        ]

    parameters = parse_field_section("Parameters")
    returns = parse_field_section("Returns")
    raises = parse_field_section("Raises", single_element_is_type=True)
    warns = parse_field_section("Warns", single_element_is_type=True)
    see_also: list[DocLink] = []
    for entry in doc["See Also"]:
        if not entry:
            continue
        targets, desc = entry
        description = " ".join(desc).strip() if desc else ""
        if isinstance(targets, list):
            for target, role in targets:
                see_also.append(
                    DocLink(
                        label=target,
                        target=target,
                        original_target=target,
                        role=role or "",
                        description=description,
                    )
                )
            continue
        see_also.append(
            DocLink(
                label=targets,
                target=targets,
                original_target=targets,
                description=description,
            )
        )
    extended_summary = [line.rstrip("\n") for line in doc["Extended Summary"]]
    raw_notes = [line.rstrip("\n") for line in doc["Notes"]]
    notes = _split_paragraphs(raw_notes)
    warnings = [line.rstrip("\n") for line in doc["Warnings"]]
    references = [line.rstrip("\n") for line in doc["References"]]

    example_text = "\n".join(line.rstrip() for line in doc["Examples"]).strip()
    examples = [DocExample(code=example_text)] if example_text else []

    return ParsedDoc(
        summary=summary,
        parameters=parameters,
        returns=returns,
        see_also=see_also,
        notes=notes,
        note_lines=raw_notes,
        examples=examples,
        extended_summary=extended_summary,
        raises=raises,
        warnings=warnings,
        warns=warns,
        references=references,
        upstream_signature=upstream_signature,
        sections=None,
        coverage=_new_doc_coverage(),
    )


def docs_url_for_operation(name: str, module: str) -> str:
    """Return the upstream reference docs URL for a supported operation."""
    if module == "numpy.linalg":
        return (
            "https://numpy.org/doc/stable/reference/generated/"
            f"numpy.linalg.{name.removeprefix('linalg.')}.html"
        )
    if module == "numpy.fft":
        return (
            "https://numpy.org/doc/stable/reference/generated/"
            f"numpy.fft.{name.removeprefix('fft.')}.html"
        )
    if module == "numpy.random":
        return (
            "https://numpy.org/doc/stable/reference/generated/"
            f"numpy.random.{name.removeprefix('random.')}.html"
        )
    if module == "flopscope.stats":
        dist_name = name.split(".")[1]
        return (
            "https://docs.scipy.org/doc/scipy/reference/generated/"
            f"scipy.stats.{dist_name}.html"
        )
    return f"https://numpy.org/doc/stable/reference/generated/numpy.{name}.html"


def _repo_source_url(
    obj: object, *, repo_blob_root: str, package_root: Path | None = None
) -> str:
    """Build a GitHub source URL for an imported object when inspect can locate it."""
    target = inspect.unwrap(obj)
    try:
        source_path = Path(
            inspect.getsourcefile(target) or inspect.getfile(target)
        ).resolve()
        _, start_line = inspect.getsourcelines(target)
    except (OSError, TypeError):  # pragma: no cover - inspect support varies by object
        return ""

    if package_root is None:
        try:
            relative = source_path.relative_to(ROOT)
        except ValueError:
            return ""
    else:
        try:
            relative = source_path.relative_to(package_root.parent)
        except ValueError:
            return ""

    return f"{repo_blob_root}/{relative.as_posix()}#L{start_line}"


def _resolve_attr(root: object, dotted_name: str) -> object:
    current = root
    for part in dotted_name.split("."):
        current = getattr(current, part)
    return current


def _worker_init_operation_doc_context(
    alias_items: tuple[tuple[str, str], ...],
    supported_ops: tuple[str, ...],
    internal_refs: dict[str, dict[str, str]],
) -> None:
    """Initialize per-process context for parallel operation doc workers."""
    global _WORKER_ALIAS_MAP
    global _WORKER_SUPPORTED_OPS
    global _WORKER_INTERNAL_REFS
    _WORKER_ALIAS_MAP = dict(alias_items)
    _WORKER_SUPPORTED_OPS = set(supported_ops)
    _WORKER_INTERNAL_REFS = internal_refs


def _worker_build_operation_record(
    task: tuple[
        str,
        str,
        str,
        list[str],
        float,
        str,
        str,
    ],
) -> OperationDocRecord:
    """Build a single operation doc record inside a worker process."""
    name, module, category, aliases, weight, notes, owned_example_html = task

    return _build_operation_record(
        name,
        module,
        category,
        aliases,
        weight,
        notes,
        owned_example_html,
        alias_map=_WORKER_ALIAS_MAP,
        supported_ops=_WORKER_SUPPORTED_OPS,
        internal_refs=_WORKER_INTERNAL_REFS,
    )


def _build_operation_record(
    name: str,
    module: str,
    category: str,
    aliases: list[str],
    weight: float,
    notes: str,
    owned_example_html: str,
    *,
    alias_map: dict[str, str] | None,
    supported_ops: set[str] | None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> OperationDocRecord:
    """Build one in-memory operation record."""
    if alias_map is None or supported_ops is None:
        raise RuntimeError(
            "Parallel worker context not initialized; call _worker_init_operation_doc_context()"
        )

    (
        signature,
        parsed_doc,
        derived_example,
        flopscope_source_url,
        upstream_source_url,
    ) = build_structured_doc(
        name,
        module,
        owned_example_html,
        alias_map=alias_map,
        supported_ops=supported_ops,
        internal_refs=internal_refs,
    )
    cost_plain, cost_latex = cost_for_op(name, category)
    canonical_href = canonical_api_href(
        flopscope_ref(name, module).replace("fnp.", "flopscope.numpy.", 1)
    )
    return OperationDocRecord(
        name=name,
        canonical_name=name,
        slug=slug_for_operation(name),
        href=canonical_href,
        canonical_path=canonical_href.removeprefix("/docs/api/").rstrip("/"),
        legacy_href=detail_href_for_slug(slug_for_operation(name)),
        module=module,
        area=normalize_area(module),
        flopscope_ref=flopscope_ref(name, module),
        numpy_ref=numpy_ref(name, module),
        category=category,
        display_type=display_type_for_category(category),
        cost_formula=cost_plain,
        cost_formula_latex=cost_latex,
        weight=weight,
        notes=notes,
        aliases=aliases,
        signature=signature,
        summary=parsed_doc.summary,
        provenance_label=provenance_label_for_operation(module),
        provenance_url=docs_url_for_operation(name, module),
        provenance_ref=provenance_ref_for_operation(name, module),
        upstream_ref_label=upstream_ref_label_for_operation(module),
        upstream_source_label=upstream_source_label_for_operation(module),
        flopscope_source_url=flopscope_source_url,
        upstream_source_url=upstream_source_url,
        parameters=parsed_doc.parameters,
        returns=parsed_doc.returns,
        see_also=parsed_doc.see_also,
        notes_sections=parsed_doc.notes,
        example=derived_example,
        body_sections=parsed_doc.sections,
        doc_coverage=parsed_doc.coverage,
        api_docs_html="",
        flopscope_examples_html=owned_example_html,
    )


def resolve_live_objects(name: str, module: str) -> tuple[object, object | None]:
    """Resolve the live flopscope object and its upstream NumPy/SciPy counterpart."""
    import numpy as np

    flops = _import_local_flopscope_module("flopscope")
    fnp = _import_local_flopscope_module("flopscope.numpy")

    if module == "numpy.linalg":
        short_name = name.removeprefix("linalg.")
        return getattr(fnp.linalg, short_name), getattr(np.linalg, short_name, None)
    if module == "numpy.fft":
        short_name = name.removeprefix("fft.")
        return getattr(fnp.fft, short_name), getattr(np.fft, short_name, None)
    if module == "numpy.random":
        short_name = name.removeprefix("random.")
        if "." in short_name:
            # Method-level entry: "Generator.beta" or "RandomState.randn".
            # Class-level getattr bypasses the instance-only __getattribute__
            # gate, so this returns the wrapped method directly.
            cls_short, method_name = short_name.split(".", 1)
            cls_fnp = getattr(fnp.random, cls_short)
            cls_np = getattr(np.random, cls_short, None)
            return (
                getattr(cls_fnp, method_name),
                getattr(cls_np, method_name, None) if cls_np is not None else None,
            )
        np_obj = getattr(np.random, short_name, None)
        fnp_obj = getattr(fnp.random, short_name, None)
        # Fall back to upstream numpy object when flopscope wraps it as deprecated/unimplemented.
        return fnp_obj if fnp_obj is not None else np_obj, np_obj
    if module == "flopscope.stats":
        try:
            from scipy import stats as scipy_stats
        except (
            Exception
        ):  # pragma: no cover - scipy availability is environment specific
            scipy_stats = None
        return _resolve_attr(flops.stats, name.removeprefix("stats.")), (
            _resolve_attr(scipy_stats, name.removeprefix("stats."))
            if scipy_stats is not None
            else None
        )
    obj = getattr(flops, name, None)
    if obj is None:
        obj = getattr(fnp, name)
    return obj, getattr(np, name, None)


def _rewrite_doc_field(field: DocField) -> DocField:
    return DocField(
        name=field.name,
        type=rewrite_api_refs(field.type),
        body=[rewrite_api_refs(line) for line in field.body],
    )


def _rewrite_doc_link(link: DocLink) -> DocLink:
    rewritten_target = _normalize_reference_target(rewrite_api_refs(link.target))
    rewritten_original = link.original_target or link.target
    return DocLink(
        label=rewrite_api_refs(link.label),
        target=rewritten_target,
        description=rewrite_api_refs(link.description),
        description_inline=None,
        role=link.role,
        original_target=rewritten_original,
        unresolved=False,
        href="",
        external_url="",
    )


def _rewrite_parsed_doc(
    parsed: ParsedDoc,
    *,
    alias_map: dict[str, str],
    supported_ops: set[str],
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> ParsedDoc:
    parsed.summary = rewrite_api_refs(parsed.summary)
    parsed.extended_summary = [
        rewrite_api_refs(line) for line in parsed.extended_summary
    ]
    parsed.parameters = [_rewrite_doc_field(field) for field in parsed.parameters]
    parsed.returns = [_rewrite_doc_field(field) for field in parsed.returns]
    parsed.raises = [_rewrite_doc_field(field) for field in parsed.raises]
    parsed.warns = [_rewrite_doc_field(field) for field in parsed.warns]
    parsed.see_also = [
        resolve_doc_link(
            link,
            alias_map=alias_map,
            supported_ops=supported_ops,
            coverage=parsed.coverage,
            internal_refs=internal_refs,
        )
        for link in parsed.see_also
    ]
    parsed.notes = [rewrite_api_refs(note) for note in parsed.notes]
    parsed.note_lines = [rewrite_api_refs(line) for line in parsed.note_lines]
    parsed.warnings = [rewrite_api_refs(line) for line in parsed.warnings]
    parsed.references = [rewrite_api_refs(line) for line in parsed.references]
    parsed.sections = _build_sections_for_doc(
        parsed,
        supported_ops=supported_ops,
        alias_map=alias_map,
        internal_refs=internal_refs,
    )
    return parsed


def resolve_doc_link(
    link: DocLink,
    *,
    alias_map: dict[str, str],
    supported_ops: set[str],
    coverage: dict[str, object] | None = None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> DocLink:
    """Resolve a parsed see-also entry into internal and external link targets."""
    rewritten = _rewrite_doc_link(link)
    href, external_url, canonical_target = _resolve_reference_destination(
        rewritten.target,
        original_target=rewritten.original_target or rewritten.target,
        role=rewritten.role,
        alias_map=alias_map,
        supported_ops=supported_ops,
        internal_refs=internal_refs,
    )
    unresolved = not href and not external_url
    if unresolved:
        _record_coverage_event(
            coverage,
            "unresolved_references",
            {
                "role": rewritten.role or "see_also",
                "target": rewritten.target,
                "original_target": rewritten.original_target or rewritten.target,
                "display_text": rewritten.label,
            },
        )

    label = rewritten.label
    if href and canonical_target in supported_ops and not label.startswith("flops."):
        label = f"flops.{canonical_target}"

    return DocLink(
        label=label,
        target=canonical_target,
        description=rewritten.description,
        description_inline=parse_inline_nodes(
            rewritten.description,
            supported_ops=supported_ops,
            alias_map=alias_map,
            coverage=coverage,
            internal_refs=internal_refs,
        )
        if rewritten.description
        else [],
        role=rewritten.role,
        original_target=rewritten.original_target or rewritten.target,
        unresolved=unresolved,
        href=href,
        external_url=external_url,
    )


def build_structured_doc(
    name: str,
    module: str,
    owned_example_html: str = "",
    *,
    alias_map: dict[str, str] | None = None,
    supported_ops: set[str] | None = None,
    internal_refs: dict[str, dict[str, str]] | None = None,
) -> tuple[str, ParsedDoc, DocExample | None, str, str]:
    """Resolve live objects and build the structured doc model for one op."""
    import numpy as np

    flopscope_obj, upstream_obj = resolve_live_objects(name, module)
    alias_map = alias_map or {}
    supported_ops = supported_ops or set()
    if module == "flopscope.stats":
        raw_doc = inspect.getdoc(flopscope_obj) or inspect.getdoc(upstream_obj) or ""
    else:
        raw_doc = inspect.getdoc(upstream_obj) or inspect.getdoc(flopscope_obj) or ""
    parsed = _rewrite_parsed_doc(
        parse_numpy_docstring(raw_doc),
        alias_map=alias_map,
        supported_ops=supported_ops,
        internal_refs=internal_refs,
    )

    example: DocExample | None = None
    if parsed.examples:
        example = derive_example_from_upstream(parsed.examples[0].code)

    try:
        signature = f"{flopscope_ref(name, module)}{inspect.signature(flopscope_obj)}"
    except (TypeError, ValueError):
        signature = f"{flopscope_ref(name, module)}(...)"

    flopscope_source_url = _repo_source_url(
        flopscope_obj,
        repo_blob_root="https://github.com/AIcrowd/flopscope/blob/main",
    )

    upstream_source_url = ""
    if module != "flopscope.stats" and upstream_obj is not None:
        numpy_root = Path(np.__file__).resolve().parent
        upstream_source_url = _repo_source_url(
            upstream_obj,
            repo_blob_root=f"https://github.com/numpy/numpy/blob/v{np.__version__}",
            package_root=numpy_root,
        )
        if not upstream_source_url and isinstance(upstream_obj, np.ufunc):
            upstream_source_url = (
                "https://github.com/numpy/numpy/blob/"
                f"v{np.__version__}/numpy/_core/code_generators/ufunc_docstrings.py"
            )

    return signature, parsed, example, flopscope_source_url, upstream_source_url


def _public_symbol_kind(obj: object, canonical_name: str) -> str:
    if inspect.isclass(obj):
        if issubclass(obj, Warning):
            return "warning"
        if issubclass(obj, Exception):
            return "error"
        return "class"
    if not callable(obj):
        return "object"
    if canonical_name.startswith("flops.") and canonical_name.endswith("_cost"):
        return "cost_helper"
    return "function"


def _top_level_symbol_aliases(name: str) -> list[str]:
    return [
        name,
        f"flops.{name}",
        f"flopscope.{name}",
        f"fnp.{name}",
        f"flopscope.numpy.{name}",
    ]


def _flops_symbol_aliases(name: str) -> list[str]:
    return [
        f"flops.{name}",
        f"flops.accounting.{name}",
        f"flopscope.accounting.{name}",
        name,
    ]


def _module_symbol_aliases(module_label: str, name: str) -> list[str]:
    aliases = [
        f"{module_label}.{name}",
        f"flops.{module_label}.{name}",
    ]
    actual_path = PUBLIC_API_MODULE_PATHS.get(module_label)
    if actual_path:
        aliases.append(f"{actual_path}.{name}")
    else:
        aliases.append(f"flopscope.{module_label}.{name}")
    return aliases


def _dedupe_aliases(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _public_symbol_summary(raw_doc: str, parsed: ParsedDoc) -> str:
    summary = parsed.summary.strip()
    if summary:
        return summary
    for line in raw_doc.splitlines():
        stripped = line.strip()
        if stripped:
            return rewrite_api_refs(stripped)
    return ""


def _symbol_signature(import_path: str, obj: object) -> str:
    try:
        return f"{import_path}{inspect.signature(obj)}"
    except (TypeError, ValueError):
        return import_path


def _related_guides_for_symbol(canonical_name: str) -> list[RelatedGuideLink]:
    guides = list(PUBLIC_SYMBOL_GUIDES.get(canonical_name, []))

    if canonical_name.startswith("flops."):
        guides.append(("Budget Planning & Debugging", "/docs/guides/budget-planning"))
        if canonical_name.startswith(
            (
                "flops.solve_cost",
                "flops.svd_cost",
                "flops.inv_cost",
                "flops.lstsq_cost",
                "flops.pinv_cost",
                "flops.cholesky_cost",
                "flops.eig_cost",
                "flops.eigh_cost",
                "flops.qr_cost",
            )
        ):
            guides.append(("Linear Algebra Guide", "/docs/guides/linalg"))
        if canonical_name.startswith(
            (
                "flops.fft_cost",
                "flops.fftn_cost",
                "flops.hfft_cost",
                "flops.rfft_cost",
                "flops.rfftn_cost",
            )
        ):
            guides.append(("FFT Guide", "/docs/guides/fft"))
        if canonical_name == "flops.einsum_cost":
            guides.append(("Einsum Guide", "/docs/guides/einsum"))

    if canonical_name in {"Permutation", "PermutationGroup", "Cycle"}:
        guides.append(("Symmetry Savings", "/docs/guides/symmetry"))

    deduped: list[RelatedGuideLink] = []
    seen: set[tuple[str, str]] = set()
    for title, href in guides:
        key = (title, href)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(RelatedGuideLink(title=title, href=href))
    return deduped


def collect_public_api_surface_names() -> set[str]:
    """Return the explicit public API surface that must have docs coverage."""
    _import_local_flopscope_module("flopscope")

    names = set(PUBLIC_API_TOP_LEVEL)
    for module_label in PUBLIC_API_MODULES:
        module = _resolve_public_api_module(module_label)
        exports = getattr(module, "__all__", None)
        if exports is None:
            raise RuntimeError(
                f"{module.__name__} must define __all__ for API docs generation"
            )
        names.update(f"{module_label}.{name}" for name in exports)
    return names


def _public_symbol_member_links(
    canonical_name: str,
    registry: dict[str, dict],
    alias_map: dict[str, str],
) -> list[OperationNavLink]:
    prefix = f"{canonical_name}."
    if not canonical_name.startswith("stats."):
        return []

    children = sorted(
        {
            resolve_canonical_name(name, alias_map)
            for name, info in registry.items()
            if info["category"] != "blacklisted" and name.startswith(prefix)
        }
    )
    return [
        OperationNavLink(
            href=canonical_api_href_for_name(name),
            label=f"flops.{name}",
        )
        for name in children
    ]


def _public_symbol_status_note(canonical_name: str, registry: dict[str, dict]) -> str:
    info = registry.get(canonical_name)
    if not info or info["category"] != "blacklisted":
        return ""

    return (
        "This symbol is exported from flopscope, but it is excluded from the "
        "operation registry and therefore does not appear in the operation "
        "cost index."
    )


def _build_operation_internal_refs(
    registry: dict[str, dict], alias_map: dict[str, str]
) -> dict[str, dict[str, str]]:
    refs: dict[str, dict[str, str]] = {}
    for name, info in registry.items():
        if info["category"] == "blacklisted":
            continue
        canonical = resolve_canonical_name(name, alias_map)
        if canonical != name:
            continue
        import_path = flopscope_ref(canonical, info["module"]).replace(
            "fnp.", "flopscope.numpy.", 1
        )
        entry = {
            "canonical_name": canonical,
            "href": canonical_api_href(import_path),
        }
        op_aliases = {
            name,
            flopscope_ref(name, info["module"]),
            flopscope_ref(name, info["module"]).replace("fnp.", "flopscope.numpy.", 1),
        }
        for alias in op_aliases:
            refs[alias] = entry

    # Include short aliases from `alias_map` (e.g. "abs" -> "absolute",
    # "acos" -> "arccos"). Without this, callers of
    # `_resolve_reference_destination` that look up an alias-form target
    # miss `internal_refs` entirely and fall through to the legacy
    # `/docs/api/ops/<slug>` URL — producing broken deploy links on the
    # generated numpy-op pages. The alias-injection mirrors what
    # `build_public_api_symbol_records` does at its own internal_refs
    # construction site.
    for alias, target in alias_map.items():
        if target in refs:
            refs[str(alias)] = refs[target]

    return refs


def build_public_api_symbol_records(
    registry: dict[str, dict], alias_map: dict[str, str]
) -> list[PublicApiSymbolRecord]:
    flops = _import_local_flopscope_module("flopscope")

    supported_ops = {
        name
        for name, info in registry.items()
        if info["category"] != "blacklisted"
        and resolve_canonical_name(name, alias_map) == name
    }
    symbol_specs: dict[str, dict[str, object]] = {}
    canonical_by_id: dict[int, str] = {}
    fnp = _import_local_flopscope_module("flopscope.numpy")

    for name in PUBLIC_API_TOP_LEVEL:
        if hasattr(flops, name):
            obj = getattr(flops, name)
            display_module = getattr(obj, "__module__", "flopscope")
            import_path = f"flops.{name}"
            display_name = f"flops.{name}"
        else:
            obj = getattr(fnp, name)
            display_module = getattr(obj, "__module__", "flopscope.numpy")
            import_path = f"fnp.{name}"
            display_name = f"fnp.{name}"
        canonical_by_id[id(obj)] = name
        symbol_specs[name] = {
            "canonical_name": name,
            "obj": obj,
            "kind": _public_symbol_kind(obj, name),
            "module": display_module,
            "import_path": import_path,
            "display_name": display_name,
            "aliases": _top_level_symbol_aliases(name),
        }

    for module_label in PUBLIC_API_MODULES:
        module = _resolve_public_api_module(module_label)
        exports = getattr(module, "__all__", [])
        for name in exports:
            canonical_name = f"{module_label}.{name}"
            if canonical_name in supported_ops:
                continue

            obj = getattr(module, name)
            canonical_from_id = canonical_by_id.get(id(obj), canonical_name)
            aliases = (
                _flops_symbol_aliases(name)
                if module_label == "flops"
                else _module_symbol_aliases(module_label, name)
            )
            if canonical_from_id in symbol_specs:
                symbol_specs[canonical_from_id]["aliases"] = _dedupe_aliases(
                    list(symbol_specs[canonical_from_id]["aliases"]) + aliases
                )
                continue

            canonical_by_id[id(obj)] = canonical_from_id
            symbol_specs[canonical_from_id] = {
                "canonical_name": canonical_from_id,
                "obj": obj,
                "kind": _public_symbol_kind(obj, canonical_from_id),
                "module": getattr(obj, "__module__", f"flopscope.{module_label}"),
                "import_path": f"flops.{canonical_name}",
                "display_name": f"flops.{canonical_name}",
                "aliases": _dedupe_aliases(aliases),
            }

    internal_refs = _build_operation_internal_refs(registry, alias_map)
    for spec in symbol_specs.values():
        entry = {
            "canonical_name": str(spec["canonical_name"]),
            "href": canonical_api_href_for_name(str(spec["canonical_name"])),
        }
        for alias in spec["aliases"]:
            internal_refs[str(alias)] = entry

    records: list[PublicApiSymbolRecord] = []
    for canonical_name in sorted(symbol_specs):
        spec = symbol_specs[canonical_name]
        obj = spec["obj"]
        raw_doc = inspect.getdoc(obj) or ""
        parsed = _rewrite_parsed_doc(
            parse_numpy_docstring(raw_doc),
            alias_map=alias_map,
            supported_ops=supported_ops,
            internal_refs=internal_refs,
        )
        import_path = str(spec["import_path"])
        record = PublicApiSymbolRecord(
            name=canonical_name,
            canonical_name=canonical_name,
            slug=slug_for_symbol(canonical_name),
            href=canonical_api_href_for_name(canonical_name),
            canonical_path=canonical_api_path_for_name(canonical_name),
            kind=str(spec["kind"]),
            module=str(spec["module"]),
            import_path=import_path,
            display_name=str(spec["display_name"]),
            summary=_public_symbol_summary(raw_doc, parsed),
            signature=_symbol_signature(import_path, obj),
            aliases=_dedupe_aliases([canonical_name] + list(spec["aliases"])),
            source_url=_repo_source_url(
                obj,
                repo_blob_root="https://github.com/AIcrowd/flopscope/blob/main",
            ),
            related_guides=_related_guides_for_symbol(canonical_name),
            members=_public_symbol_member_links(canonical_name, registry, alias_map),
            status_note=_public_symbol_status_note(canonical_name, registry),
            body_sections=parsed.sections or [],
        )
        records.append(record)

    return records


ALIAS_NOTE_PATTERN = re.compile(r"alias for ([\w./]+)", re.IGNORECASE)
ALIAS_REASON_PATTERN = re.compile(r"Alias of ([\w./]+)", re.IGNORECASE)
EXAMPLE_FENCE_PATTERN = re.compile(
    r"```(?P<lang>[A-Za-z0-9_-]+)?\n(?P<body>[\s\S]*?)```", re.MULTILINE
)


def choose_alias_target(raw_target: str, registry: dict[str, dict]) -> str | None:
    """Return the first canonical registry target encoded in alias metadata."""
    cleaned = raw_target.strip().rstrip(").,;:")
    candidates = [part.strip().rstrip(").,;:") for part in cleaned.split("/")]
    for candidate in candidates:
        if candidate in registry:
            return candidate
    return None


def load_alias_map(registry: dict[str, dict]) -> dict[str, str]:
    """Derive alias -> canonical mappings from registry notes and weights metadata."""
    alias_map: dict[str, str] = {}

    for name, info in registry.items():
        match = ALIAS_NOTE_PATTERN.search(info.get("notes", ""))
        if not match:
            continue
        target = choose_alias_target(match.group(1), registry)
        if (
            target
            and target != name
            and registry.get(target, {}).get("category") == info.get("category")
        ):
            alias_map[name] = target

    if WEIGHTS_CSV_PATH.exists():
        with WEIGHTS_CSV_PATH.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("Status") != "alias":
                    continue
                name = row.get("Operation", "").strip()
                if not name or name in alias_map or name not in registry:
                    continue
                reason = row.get("Exclusion Reason", "")
                match = ALIAS_REASON_PATTERN.search(reason)
                if not match:
                    continue
                target = choose_alias_target(match.group(1), registry)
                if not target or target == name:
                    continue
                if registry.get(target, {}).get("category") != registry.get(
                    name, {}
                ).get("category"):
                    continue
                # Prefer registry-authored canonical targets when metadata disagrees.
                if alias_map.get(target) == name:
                    continue
                alias_map[name] = target

    return alias_map


def resolve_canonical_name(name: str, alias_map: dict[str, str]) -> str:
    """Resolve an alias chain to its canonical target."""
    current = name
    seen = {current}
    while current in alias_map and alias_map[current] not in seen:
        current = alias_map[current]
        seen.add(current)
    return current


def load_operation_weights() -> dict[str, float]:
    """Load per-operation weights for docs manifests."""
    if not WEIGHTS_PATH.exists():
        return {}
    raw = json.loads(WEIGHTS_PATH.read_text())
    return raw.get("weights", {})


def example_file_for(name: str, example_root: Path) -> Path:
    """Return the owned-example path for a canonical operation name."""
    return example_root / f"{name}.mdx"


def build_example_coverage(
    operations: list[OperationDocRecord | str],
    example_root: Path | None = None,
    *,
    overrides: dict[str, DocExample] | None = None,
    derived_examples: dict[str, DocExample] | None = None,
) -> dict[str, dict]:
    """Compute example coverage for canonical operation pages."""
    overrides = overrides or {}
    derived_examples = derived_examples or {}
    coverage: dict[str, dict] = {}

    for item in operations:
        if isinstance(item, OperationDocRecord):
            name = item.name
            record_example = item.example
            provenance_url = item.provenance_url
        else:
            name = item
            record_example = None
            provenance_url = ""

        if name in overrides:
            example = overrides[name]
            coverage[name] = {
                "has_flopscope_examples": True,
                "has_inherited_examples": False,
                "example_count": 1,
                "example_sources": ["override"],
                "coverage_status": "override",
                "example_source": example.source,
            }
            continue

        if example_root is not None:
            path = example_file_for(name, example_root)
            if path.exists():
                source = path.read_text()
                example_count = len(EXAMPLE_FENCE_PATTERN.findall(source)) or 1
                coverage[name] = {
                    "has_flopscope_examples": True,
                    "has_inherited_examples": False,
                    "example_count": example_count,
                    "example_sources": [str(path)],
                    "coverage_status": "owned",
                    "example_source": "owned",
                }
                continue

        if name in derived_examples or record_example is not None:
            example = derived_examples.get(name, record_example)
            if example is None:
                example = DocExample(code="", source="derived")
            coverage[name] = {
                "has_flopscope_examples": False,
                "has_inherited_examples": True,
                "example_count": 1,
                "example_sources": [provenance_url or "derived"],
                "coverage_status": "derived",
                "example_source": example.source,
            }
            continue

        coverage[name] = {
            "has_flopscope_examples": False,
            "has_inherited_examples": False,
            "example_count": 0,
            "example_sources": [],
            "coverage_status": "missing",
            "example_source": "missing",
        }

    return coverage


def render_example_markdown_html(source: str) -> str:
    """Render simple MDX example snippets into HTML for the op-doc manifest."""
    rendered: list[str] = []
    cursor = 0
    for match in EXAMPLE_FENCE_PATTERN.finditer(source):
        prose = source[cursor : match.start()].strip()
        if prose:
            rendered.append(f"<p>{html.escape(prose)}</p>")

        language = (match.group("lang") or "").strip()
        class_attr = f' class="language-{html.escape(language)}"' if language else ""
        body = html.escape(match.group("body").rstrip("\n"))
        rendered.append(f"<pre><code{class_attr}>{body}</code></pre>")
        cursor = match.end()

    trailing = source[cursor:].strip()
    if trailing:
        rendered.append(f"<p>{html.escape(trailing)}</p>")

    return "\n".join(rendered)


def load_flopscope_example_html(name: str, example_root: Path) -> str:
    """Load and render owned flopscope examples for a canonical operation."""
    path = example_file_for(name, example_root)
    if not path.exists():
        return ""

    return render_example_markdown_html(path.read_text())


def build_alias_groups(
    registry: dict[str, dict], alias_map: dict[str, str]
) -> dict[str, list[str]]:
    """Group aliases by resolved canonical name."""
    alias_groups: dict[str, list[str]] = {}
    for alias in sorted(alias_map):
        canonical = resolve_canonical_name(alias, alias_map)
        if canonical == alias or canonical not in registry:
            continue
        alias_groups.setdefault(canonical, []).append(alias)
    return alias_groups


def resolve_operation_weight(
    name: str,
    registry: dict[str, dict],
    weights: dict[str, float],
    alias_map: dict[str, str],
    alias_groups: dict[str, list[str]] | None = None,
) -> float:
    """Resolve a stable weight for canonical and alias rows."""
    canonical = resolve_canonical_name(name, alias_map)
    if canonical in weights:
        return weights[canonical]

    aliases = alias_groups.get(canonical, []) if alias_groups is not None else []
    for alias in aliases:
        if alias in weights:
            return weights[alias]

    if canonical not in registry and name in weights:
        return weights[name]

    info = registry.get(canonical) or registry.get(name)
    if info is not None and info.get("category") == "free":
        return 0.0

    return 1.0


def build_operation_doc_records(
    registry: dict[str, dict], *, workers: int = 1
) -> list[OperationDocRecord]:
    """Build canonical operation doc records for supported operations."""
    alias_map = load_alias_map(registry)
    alias_groups = build_alias_groups(registry, alias_map)
    weights = load_operation_weights()
    supported_ops = {
        name
        for name, info in registry.items()
        if info["category"] != "blacklisted"
        and resolve_canonical_name(name, alias_map) == name
    }
    internal_refs = _build_operation_internal_refs(registry, alias_map)
    tasks: list[tuple[str, str, str, list[str], float, str, str]] = []
    for name, info in sorted(registry.items()):
        if info["category"] == "blacklisted":
            continue
        if resolve_canonical_name(name, alias_map) != name:
            continue

        aliases = sorted(alias_groups.get(name, []))
        weight = resolve_operation_weight(
            name=name,
            registry=registry,
            weights=weights,
            alias_map=alias_map,
            alias_groups=alias_groups,
        )

        module = info["module"]
        owned_example_html = load_flopscope_example_html(name, API_EXAMPLES_DIR)
        tasks.append(
            (
                name,
                module,
                info["category"],
                aliases,
                weight,
                info.get("notes", ""),
                owned_example_html,
            )
        )

    if workers > 1:
        max_workers = min(
            workers,
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1,
        )
        mp_context = (
            mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        )
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=_worker_init_operation_doc_context,
            initargs=(
                tuple(alias_map.items()),
                tuple(sorted(supported_ops)),
                internal_refs,
            ),
        ) as executor:
            records = list(executor.map(_worker_build_operation_record, tasks))
    else:
        records = [
            _build_operation_record(
                *task,
                alias_map=alias_map,
                supported_ops=supported_ops,
                internal_refs=internal_refs,
            )
            for task in tasks
        ]

    for index, record in enumerate(records):
        if index > 0:
            previous = records[index - 1]
            record.previous = OperationNavLink(
                href=previous.href,
                label=previous.flopscope_ref,
            )
        if index + 1 < len(records):
            nxt = records[index + 1]
            record.next = OperationNavLink(
                href=nxt.href,
                label=nxt.flopscope_ref,
            )

    return records


def render_operation_stub(op: OperationDocRecord) -> str:
    """Render a generated standalone MDX page stub for one canonical operation."""
    return f'---\ntitle: "{op.flopscope_ref}"\n---\n\n<OperationDocPage name="{op.name}" />\n'


def render_public_symbol_stub(symbol: PublicApiSymbolRecord) -> str:
    """Render a generated standalone MDX page stub for one canonical public symbol."""
    lines = [f'title: "{symbol.display_name}"']
    if symbol.summary:
        lines.append(f"description: {json.dumps(symbol.summary)}")
    return (
        "---\n"
        + "\n".join(lines)
        + f'\n---\n\n<PublicApiSymbolPage name="{symbol.name}" />\n'
    )


def write_json(path: Path, payload: object) -> None:
    """Write a deterministic JSON artifact with trailing newline."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _serialize_public_op_payload(record: OperationDocRecord) -> dict[str, object]:
    """Build the public per-op JSON contract from the rich internal record."""
    return {
        "schema_version": 1,
        "slug": record.slug,
        "detail_href": record.href,
        "detail_json_href": detail_json_href_for_slug(record.slug),
        "source": {
            "flopscope": record.flopscope_source_url or None,
            "numpy": record.upstream_source_url or record.provenance_url or None,
        },
        "op": {
            "name": record.name,
            "module": record.module,
            "flopscope_ref": record.flopscope_ref,
            "numpy_ref": record.numpy_ref,
            "category": record.category,
            "status": "blocked" if record.category == "blacklisted" else "supported",
            "weight": record.weight,
            "cost_formula": record.cost_formula,
            "cost_formula_latex": record.cost_formula_latex,
            "notes": record.notes,
            "summary": record.summary,
            "signature": record.signature,
        },
        "docs": {
            "sections": record.body_sections or [],
        },
    }


def write_operation_doc_artifacts(
    records: list[OperationDocRecord], website_root: Path
) -> None:
    """Emit generated operation payloads, refs, and static-import metadata."""
    generated_dir = website_root / ".generated"
    op_docs_payload_dir = generated_dir / "ops"
    public_dir = website_root / "public"
    public_payload_dir = public_dir / "api-data" / "ops"
    generated_dir.mkdir(parents=True, exist_ok=True)
    op_docs_payload_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    public_payload_dir.mkdir(parents=True, exist_ok=True)
    for existing_payload in op_docs_payload_dir.glob("*.json"):
        if existing_payload.stem not in {record.slug for record in records}:
            existing_payload.unlink()
    for existing_payload in public_payload_dir.glob("*.json"):
        if existing_payload.stem not in {record.slug for record in records}:
            existing_payload.unlink()

    docs_manifest: dict[str, dict[str, object]] = {}
    refs_manifest: dict[str, dict[str, str]] = {}
    ops_index: list[dict[str, object]] = []
    # Emit `{ default: unknown }` — consumers cast to `OperationDocRecord`
    # themselves (app/docs/api/ops/[slug]/page.tsx). TypeScript's inferred
    # type of the imported JSON is too wide (e.g. `type: string` vs a
    # literal union like `"field_list"`) to line up with the record's
    # narrow discriminated unions, which breaks `next build`.
    import_map_lines = [
        "export const opDocImports: Record<string, () => Promise<{ default: unknown }>> = {",
    ]

    for record in sorted(records, key=lambda op: op.name):
        internal_payload = asdict(record)
        public_payload = _serialize_public_op_payload(record)
        write_json(op_docs_payload_dir / f"{record.slug}.json", internal_payload)
        write_json(public_payload_dir / f"{record.slug}.json", public_payload)
        docs_manifest[record.name] = internal_payload
        refs_manifest[record.name] = {
            "label": record.flopscope_ref,
            "href": record.href,
            "canonical_name": record.canonical_name,
        }
        for alias in record.aliases:
            refs_manifest[alias] = {
                "label": record.flopscope_ref,
                "href": record.href,
                "canonical_name": record.canonical_name,
            }
        ops_index.append(
            {
                "name": record.name,
                "slug": record.slug,
                "canonical_path": record.canonical_path,
                "detail_href": record.href,
                "detail_json_href": detail_json_href_for_slug(record.slug),
                "module": record.module,
                "flopscope_ref": record.flopscope_ref,
                "numpy_ref": record.numpy_ref,
                "category": record.category,
                "cost_formula": record.cost_formula,
                "cost_formula_latex": record.cost_formula_latex,
                "free": record.category == "free",
                "blocked": record.category == "blacklisted",
                "status": "blocked"
                if record.category == "blacklisted"
                else "supported",
                "notes": record.notes,
                "summary": record.summary,
                "weight": record.weight,
                "area": record.area,
                "display_type": record.display_type,
            }
        )
        import_map_lines.append(
            f'  "{record.slug}": () => import("./ops/{record.slug}.json"),'
        )

    import_map_lines.extend(
        [
            "};",
            "",
            "export const opDocSlugs = [",
            *[
                f'  "{record.slug}",'
                for record in sorted(records, key=lambda op: op.slug)
            ],
            "] as const;",
            "",
        ]
    )
    write_json(generated_dir / "op-docs.json", docs_manifest)
    write_json(generated_dir / "op-refs.json", refs_manifest)
    write_json(
        public_dir / "ops.json", {"operations": ops_index, "total": len(ops_index)}
    )
    (generated_dir / "op-doc-imports.ts").write_text("\n".join(import_map_lines))
    print(f"  Generated .generated/ops/*.json ({len(records)} operations)")
    print(f"  Generated public/api-data/ops/*.json ({len(records)} operations)")
    print(f"  Generated public/ops.json ({len(ops_index)} operations)")
    print(f"  Generated {generated_dir / 'op-doc-imports.ts'}")
    print(f"  Generated {generated_dir / 'op-docs.json'}")
    print(f"  Generated {generated_dir / 'op-refs.json'}")


def write_public_symbol_artifacts(
    records: list[PublicApiSymbolRecord], website_root: Path
) -> None:
    """Emit generated public-symbol payloads and static import metadata."""
    generated_dir = website_root / ".generated"
    symbol_docs_payload_dir = generated_dir / "symbols"
    generated_dir.mkdir(parents=True, exist_ok=True)
    symbol_docs_payload_dir.mkdir(parents=True, exist_ok=True)

    expected_pages = {record.slug for record in records}
    for existing_payload in symbol_docs_payload_dir.glob("*.json"):
        if existing_payload.stem not in expected_pages:
            existing_payload.unlink()

    docs_manifest: dict[str, dict[str, object]] = {}
    import_map_lines = [
        "export const symbolDocImports: Record<string, () => Promise<{ default: unknown }>> = {",
    ]
    for record in sorted(records, key=lambda symbol: symbol.name):
        write_json(symbol_docs_payload_dir / f"{record.slug}.json", asdict(record))
        docs_manifest[record.name] = asdict(record)
        import_map_lines.append(
            f'  "{record.slug}": () => import("./symbols/{record.slug}.json"),'
        )

    import_map_lines.extend(
        [
            "};",
            "",
            "export const symbolDocSlugs = [",
            *[
                f'  "{record.slug}",'
                for record in sorted(records, key=lambda symbol: symbol.slug)
            ],
            "] as const;",
            "",
        ]
    )
    write_json(generated_dir / "public-api-symbols.json", docs_manifest)
    (generated_dir / "symbol-doc-imports.ts").write_text("\n".join(import_map_lines))
    print(f"  Generated {generated_dir / 'public-api-symbols.json'}")
    print(f"  Generated {generated_dir / 'symbol-doc-imports.ts'}")


def write_public_api_route_artifacts(
    operation_records: list[OperationDocRecord],
    symbol_records: list[PublicApiSymbolRecord],
    website_root: Path,
) -> None:
    """Emit canonical route manifests for generated public API pages."""
    generated_dir = website_root / ".generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    routes: dict[str, dict[str, str]] = {}
    legacy_redirects: dict[str, str] = {}

    for record in sorted(operation_records, key=lambda op: op.name):
        routes[record.canonical_path] = {
            "kind": "op",
            "slug": record.slug,
            "href": record.href,
            "canonical_name": record.canonical_name,
        }
        legacy_redirects[record.slug] = record.href

    for record in sorted(symbol_records, key=lambda symbol: symbol.name):
        routes[record.canonical_path] = {
            "kind": "symbol",
            "slug": record.slug,
            "href": record.href,
            "canonical_name": record.canonical_name,
        }

    write_json(generated_dir / "public-api-routes.json", routes)
    write_json(generated_dir / "legacy-op-redirects.json", legacy_redirects)
    write_json(
        generated_dir / "public-api-refs.json",
        build_public_api_refs_manifest(
            operation_records=operation_records,
            symbol_records=symbol_records,
        ),
    )
    print(f"  Generated {generated_dir / 'public-api-routes.json'}")
    print(f"  Generated {generated_dir / 'legacy-op-redirects.json'}")
    print(f"  Generated {generated_dir / 'public-api-refs.json'}")


def build_public_api_refs_manifest(
    *,
    operation_records: list[OperationDocRecord],
    symbol_records: list[PublicApiSymbolRecord],
) -> dict[str, dict[str, str]]:
    """Build the unified public API ref manifest for docs linking and linting."""
    manifest: dict[str, dict[str, str]] = {}

    for record in operation_records:
        entry = {
            "canonical_name": record.canonical_name,
            "href": record.href,
            "label": record.flopscope_ref,
            "kind": "op",
            "module": record.flopscope_ref.rsplit(".", 1)[0],
            "source_url": record.flopscope_source_url,
            "import_path": record.flopscope_ref,
        }
        namespace = (
            record.canonical_name.rsplit(".", 1)[0]
            if "." in record.canonical_name
            else ""
        )
        for alias in _dedupe_aliases([record.name] + list(record.aliases)):
            manifest[alias] = entry
            alias_import_path = (
                f"flops.{alias}"
                if not namespace or "." in alias
                else f"flops.{namespace}.{alias}"
            )
            manifest[alias_import_path] = entry
            manifest[alias_import_path.replace("flops.", "flopscope.", 1)] = entry
        manifest[record.flopscope_ref] = entry
        manifest[record.flopscope_ref.replace("flops.", "flopscope.", 1)] = entry

    for record in symbol_records:
        entry = {
            "canonical_name": record.canonical_name,
            "href": record.href,
            "label": record.display_name,
            "kind": record.kind,
            "module": record.module,
            "source_url": record.source_url,
            "import_path": record.import_path,
        }
        for alias in record.aliases:
            manifest[alias] = entry

    return manifest


def _public_api_entry_kind(obj: object) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
        return "function"
    if callable(obj):
        return "callable"
    return "object"


def _make_public_api_manifest_entry(
    import_path: str,
    *,
    obj: object | None = None,
    kind: str | None = None,
    module: str | None = None,
    is_callable: bool | None = None,
) -> dict[str, object]:
    resolved_kind = kind or (
        _public_api_entry_kind(obj) if obj is not None else "object"
    )
    resolved_callable = (
        is_callable
        if is_callable is not None
        else callable(obj)
        if obj is not None
        else False
    )
    return {
        "canonical_name": import_path,
        "href": canonical_api_href(import_path),
        "import_path": import_path,
        "kind": resolved_kind,
        "module": module or getattr(obj, "__module__", ""),
        "callable": resolved_callable,
    }


TOP_LEVEL_PUBLIC_EXPORTS = (
    "BudgetContext",
    "FlopscopeArray",
    "PathInfo",
    "StepInfo",
    "Permutation",
    "PermutationGroup",
    "Cycle",
    "SymmetricTensor",
    "SymmetryInfo",
    "budget",
    "budget_live",
    "budget_reset",
    "budget_summary",
    "budget_summary_dict",
    "configure",
    "namespace",
    "as_symmetric",
    "is_symmetric",
    "symmetrize",
)


def _public_api_namespace(import_path: str) -> str:
    if import_path.startswith("flopscope.numpy."):
        return "numpy"
    if import_path.startswith("flopscope.stats."):
        return "stats"
    if import_path.startswith("flopscope.accounting."):
        return "accounting"
    return "flopscope"


def _resolve_public_api_object(import_path: str) -> object:
    root = _import_local_flopscope_module("flopscope")
    obj: object = root
    for part in import_path.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


def _public_api_aliases(import_path: str) -> list[str]:
    aliases = [import_path]
    if import_path.startswith("flopscope.numpy."):
        aliases.append(import_path.replace("flopscope.numpy.", "fnp.", 1))
    elif import_path.startswith("flopscope.accounting."):
        aliases.append(
            import_path.replace("flopscope.accounting.", "flops.accounting.", 1)
        )
    elif import_path.startswith("flopscope.stats."):
        aliases.append(import_path.replace("flopscope.", "flops.", 1))
    elif import_path.startswith("flopscope."):
        aliases.append(import_path.replace("flopscope.", "flops.", 1))
    return _dedupe_aliases(aliases)


def _iter_public_callable_exports(
    module: object, import_prefix: str, *, skip: set[str] | None = None
) -> list[tuple[str, object]]:
    exports: list[tuple[str, object]] = []
    skip = skip or set()
    for name in getattr(module, "__all__", []):
        if name in skip or not hasattr(module, name):
            continue
        obj = getattr(module, name)
        if inspect.ismodule(obj):
            continue
        if not (inspect.isclass(obj) or callable(obj)):
            continue
        exports.append((f"{import_prefix}.{name}", obj))
    return exports


def build_public_api_manifest(
    registry: dict[str, dict],
) -> dict[str, dict[str, object]]:
    """Build a canonical import-path keyed manifest for the public API."""
    manifest: dict[str, dict[str, object]] = {}
    flops = _import_local_flopscope_module("flopscope")
    fnp = _import_local_flopscope_module("flopscope.numpy")
    fnp_random = _import_local_flopscope_module("flopscope.numpy.random")
    stats_module = _import_local_flopscope_module("flopscope.stats")
    accounting_module = _import_local_flopscope_module("flopscope.accounting")
    alias_map = load_alias_map(registry)

    for name in TOP_LEVEL_PUBLIC_EXPORTS:
        if not hasattr(flops, name):
            continue
        obj = getattr(flops, name)
        import_path = f"flopscope.{name}"
        manifest[import_path] = _make_public_api_manifest_entry(import_path, obj=obj)

    for name, info in sorted(registry.items()):
        if info["category"] == "blacklisted":
            continue
        if resolve_canonical_name(name, alias_map) != name:
            continue
        import_path = flopscope_ref(name, info["module"]).replace(
            "fnp.", "flopscope.numpy.", 1
        )
        manifest[import_path] = _make_public_api_manifest_entry(
            import_path,
            kind="function",
            module=info["module"],
            is_callable=True,
        )

    registry_import_paths = set(manifest)
    for import_path, obj in _iter_public_callable_exports(
        fnp, "flopscope.numpy", skip={"linalg", "fft", "random"}
    ):
        if import_path in registry_import_paths:
            continue
        manifest[import_path] = _make_public_api_manifest_entry(import_path, obj=obj)

    for import_path, obj in _iter_public_callable_exports(
        fnp_random, "flopscope.numpy.random"
    ):
        if import_path in registry_import_paths:
            continue
        manifest[import_path] = _make_public_api_manifest_entry(import_path, obj=obj)

    for name in getattr(stats_module, "__all__", []):
        if not hasattr(stats_module, name):
            continue
        obj = getattr(stats_module, name)
        if inspect.ismodule(obj):
            continue
        import_path = f"flopscope.stats.{name}"
        manifest[import_path] = _make_public_api_manifest_entry(import_path, obj=obj)
        for member_name in sorted(dir(obj)):
            if member_name.startswith("_"):
                continue
            member = getattr(obj, member_name, None)
            if not callable(member):
                continue
            member_import_path = f"{import_path}.{member_name}"
            manifest[member_import_path] = _make_public_api_manifest_entry(
                member_import_path,
                obj=member,
            )

    for import_path, obj in _iter_public_callable_exports(
        accounting_module, "flopscope.accounting"
    ):
        manifest[import_path] = _make_public_api_manifest_entry(import_path, obj=obj)

    return manifest


def build_operation_cost_index_entries(
    manifest: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Return manifest entries that belong in the operation cost index."""
    included_prefixes = ("flopscope.numpy.", "flopscope.stats.")
    return sorted(
        [
            entry
            for entry in manifest.values()
            if entry["import_path"].startswith(included_prefixes)
            and bool(entry["callable"])
            and entry["kind"] != "class"
        ],
        key=lambda entry: str(entry["import_path"]),
    )


def _public_api_leaf_slug(import_path: str) -> str:
    """Return the generated payload slug for one canonical public API leaf."""
    return slug_for_symbol(import_path)


def _public_api_payload_from_operation(record: OperationDocRecord) -> dict[str, object]:
    import_path = record.flopscope_ref.replace("fnp.", "flopscope.numpy.", 1)
    payload = asdict(record)
    payload["canonical_name"] = import_path
    payload["href"] = canonical_api_href(import_path)
    payload["canonical_path"] = payload["href"]
    payload["legacy_href"] = detail_href_for_slug(record.slug)
    payload["import_path"] = import_path
    payload["display_name"] = import_path
    payload["kind"] = "function"
    payload["operation"] = {
        "name": record.name,
        "category": record.category,
        "free": record.category == "free",
        "blocked": record.category == "blacklisted",
        "cost_formula": record.cost_formula,
        "cost_formula_latex": record.cost_formula_latex,
    }
    return payload


def _public_api_payload_from_symbol(
    entry: dict[str, object],
    *,
    alias_map: dict[str, str],
    supported_ops: set[str],
    internal_refs: dict[str, dict[str, str]],
) -> dict[str, object]:
    import_path = str(entry["import_path"])
    obj = _resolve_public_api_object(import_path)
    raw_doc = inspect.getdoc(obj) or ""
    parsed = _rewrite_parsed_doc(
        parse_numpy_docstring(raw_doc),
        alias_map=alias_map,
        supported_ops=supported_ops,
        internal_refs=internal_refs,
    )
    examples = parsed.examples
    return {
        "name": import_path.rsplit(".", 1)[-1],
        "canonical_name": import_path,
        "slug": _public_api_leaf_slug(import_path),
        "href": str(entry["href"]),
        "canonical_path": str(entry["href"]),
        "legacy_href": str(entry["href"]),
        "module": str(entry["module"]),
        "import_path": import_path,
        "display_name": import_path,
        "flopscope_ref": import_path,
        "numpy_ref": "",
        "signature": _symbol_signature(import_path, obj),
        "summary": _public_symbol_summary(raw_doc, parsed),
        "area": _public_api_namespace(import_path),
        "display_type": "",
        "weight": 1.0,
        "aliases": [
            alias for alias in _public_api_aliases(import_path) if alias != import_path
        ],
        "notes": "",
        "cost_formula": "",
        "cost_formula_latex": "",
        "flopscope_source_url": _repo_source_url(
            obj,
            repo_blob_root="https://github.com/AIcrowd/flopscope/blob/main",
        ),
        "upstream_source_url": "",
        "parameters": [asdict(field) for field in parsed.parameters],
        "returns": [asdict(field) for field in parsed.returns],
        "see_also": [asdict(link) for link in parsed.see_also],
        "notes_sections": parsed.notes,
        "example": asdict(examples[0]) if examples else None,
        "body_sections": parsed.sections or [],
        "doc_coverage": parsed.coverage,
        "previous": None,
        "next": None,
        "kind": str(entry["kind"]),
        "operation": None,
    }


def write_public_api_leaf_artifacts(
    operation_records: list[OperationDocRecord],
    registry: dict[str, dict],
    website_root: Path,
) -> None:
    """Emit canonical public-API leaf payloads, route refs, and a public index."""
    manifest = build_public_api_manifest(registry)
    operation_cost_index = build_operation_cost_index_entries(manifest)
    operation_cost_import_paths = {
        str(entry["import_path"]) for entry in operation_cost_index
    }
    alias_map = load_alias_map(registry)
    supported_ops = {
        name
        for name, info in registry.items()
        if info["category"] != "blacklisted"
        and resolve_canonical_name(name, alias_map) == name
    }

    generated_dir = website_root / ".generated"
    leaf_payload_dir = generated_dir / "public-api"
    public_api_data_dir = website_root / "public" / "api-data" / "public-api"
    generated_dir.mkdir(parents=True, exist_ok=True)
    leaf_payload_dir.mkdir(parents=True, exist_ok=True)
    public_api_data_dir.mkdir(parents=True, exist_ok=True)

    expected_payload_slugs = {
        _public_api_leaf_slug(import_path) for import_path in manifest
    }
    for existing_payload in leaf_payload_dir.glob("*.json"):
        if existing_payload.stem not in expected_payload_slugs:
            existing_payload.unlink()

    route_map: dict[str, dict[str, object]] = {}
    public_index_entries: list[dict[str, object]] = []
    public_operation_cost_entries: list[dict[str, object]] = []
    import_map_lines = [
        "export const publicApiLeafImports: Record<string, () => Promise<{ default: unknown }>> = {",
    ]
    operation_records_by_import_path = {
        record.flopscope_ref.replace("fnp.", "flopscope.numpy.", 1): record
        for record in operation_records
    }
    internal_refs = _build_operation_internal_refs(registry, alias_map)
    for import_path in manifest:
        href = str(manifest[import_path]["href"])
        entry = {"canonical_name": import_path, "href": href}
        for alias in _public_api_aliases(import_path):
            internal_refs[alias] = entry

    for import_path in sorted(manifest):
        entry = manifest[import_path]
        slug = _public_api_leaf_slug(import_path)
        is_operation_cost_leaf = import_path in operation_cost_import_paths
        operation_record = operation_records_by_import_path.get(import_path)
        payload = (
            _public_api_payload_from_operation(operation_record)
            if operation_record is not None
            else _public_api_payload_from_symbol(
                entry,
                alias_map=alias_map,
                supported_ops=supported_ops,
                internal_refs=internal_refs,
            )
        )
        payload["schema_version"] = 1
        payload["is_operation_cost_leaf"] = is_operation_cost_leaf
        write_json(leaf_payload_dir / f"{slug}.json", payload)
        route_map[str(entry["href"])] = {
            "canonical_name": str(entry["canonical_name"]),
            "href": str(entry["href"]),
            "import_path": import_path,
            "slug": slug,
            "kind": str(payload.get("kind") or entry["kind"]),
            "module": str(entry["module"]),
            "callable": bool(entry["callable"]),
        }
        public_summary = {
            "href": str(entry["href"]),
            "import_path": import_path,
            "display_name": str(payload.get("display_name") or import_path),
            "summary": str(payload.get("summary") or ""),
            "kind": str(payload.get("kind") or entry["kind"]),
            "namespace": _public_api_namespace(import_path),
            "module": str(entry["module"]),
            "callable": bool(entry["callable"]),
            "slug": slug,
            "is_operation_cost_leaf": is_operation_cost_leaf,
        }
        public_index_entries.append(public_summary)
        if is_operation_cost_leaf:
            public_operation_cost_entries.append(public_summary)
        import_map_lines.append(
            f'  "{import_path}": () => import("./public-api/{slug}.json"),'
        )

    import_map_lines.extend(
        [
            "};",
            "",
            "export const publicApiLeafImportPaths = [",
            *[f'  "{import_path}",' for import_path in sorted(manifest)],
            "] as const;",
            "",
        ]
    )

    write_json(
        generated_dir / "public-api-route-map.json",
        {"schema_version": 1, "routes": route_map},
    )
    write_json(
        public_api_data_dir / "index.json",
        {
            "schema_version": 1,
            "entries": public_index_entries,
            "operation_cost_index": public_operation_cost_entries,
            "total": len(public_index_entries),
        },
    )
    (generated_dir / "public-api-leaf-imports.ts").write_text(
        "\n".join(import_map_lines)
    )
    print(
        f"  Generated .generated/public-api/*.json ({len(public_index_entries)} leaves)"
    )
    print(f"  Generated {generated_dir / 'public-api-route-map.json'}")
    print(f"  Generated {generated_dir / 'public-api-leaf-imports.ts'}")
    print(f"  Generated {public_api_data_dir / 'index.json'}")


def write_op_doc_coverage_artifact(
    records: list[OperationDocRecord], website_root: Path
) -> None:
    """Write doc-parser coverage for unresolved references and raw fallbacks."""
    generated_dir = website_root / ".generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        record.name: {
            "unresolved_references": record.doc_coverage.get(
                "unresolved_references", []
            ),
            "unsupported_directives": record.doc_coverage.get(
                "unsupported_directives", []
            ),
            "raw_blocks": record.doc_coverage.get("raw_blocks", []),
            "has_issues": any(
                record.doc_coverage.get(key)
                for key in (
                    "unresolved_references",
                    "unsupported_directives",
                    "raw_blocks",
                )
            ),
        }
        for record in records
    }
    write_json(generated_dir / "op-doc-coverage.json", payload)
    print(f"  Generated {generated_dir / 'op-doc-coverage.json'}")


def write_example_coverage_artifact(
    coverage: dict[str, dict], website_root: Path
) -> None:
    """Write the owned-example coverage manifest."""
    generated_dir = website_root / ".generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    write_json(generated_dir / "api-example-coverage.json", coverage)
    print(f"  Generated {generated_dir / 'api-example-coverage.json'}")


def assert_supported_docs_env() -> None:
    """Fail fast unless the docs generator is running on supported NumPy."""
    try:
        import numpy as np
        from numpy.lib import NumpyVersion
    except (
        Exception
    ) as exc:  # pragma: no cover - import failure is environment-specific
        raise RuntimeError("generate_api_docs.py requires NumPy >= 2.0,<2.5") from exc

    version = NumpyVersion(np.__version__)
    if not (NumpyVersion("2.0.0") <= version < NumpyVersion("2.5.0")):
        raise RuntimeError(
            f"generate_api_docs.py requires NumPy >= 2.0,<2.5; found {np.__version__}"
        )


def generate_audit_page(registry: dict[str, dict]) -> None:
    """Generate docs/reference/operation-audit.md — 7-column searchable table."""
    REF_DIR.mkdir(parents=True, exist_ok=True)
    by_cat: dict[str, list[str]] = {cat: [] for cat in CATEGORY_ORDER}
    for name, info in sorted(registry.items()):
        cat = info["category"]
        if cat in by_cat:
            by_cat[cat].append(name)

    lines = [
        HEADER,
        "# Operation Audit",
        "",
        "Complete inventory of every NumPy operation and its flopscope status.",
        "Generated from the operation registry (`_registry.py`).",
        "",
        "## Summary",
        "",
        "| Category | Count | Cost |",
        "|----------|-------|------|",
    ]
    total = 0
    for cat in CATEGORY_ORDER:
        label, cost = CATEGORY_LABELS[cat]
        count = len(by_cat[cat])
        total += count
        lines.append(f"| {cat} | {count} | {cost} |")
    lines.append(f"| **Total** | **{total}** | |")
    lines.append("")

    lines.append("## All Operations")
    lines.append("")
    lines.append("| Operation | flopscope | NumPy | Category | Cost | Status | Notes |")
    lines.append("|-----------|-----------|-------|----------|------|--------|-------|")
    for name, info in sorted(registry.items()):
        cat = info["category"]
        mod = info["module"]
        me_ref = flopscope_ref(name, mod)
        np_ref = numpy_ref(name, mod)
        _, latex = cost_for_op(name, cat)
        emoji = CATEGORY_EMOJI.get(cat, "")
        status = "blocked" if cat == "blacklisted" else "supported"
        status_display = f"{emoji} {status}"
        notes = info.get("notes", "")
        if cat == "blacklisted":
            me_ref = "\u2014"
        lines.append(
            f"| `{name}` | `{me_ref}` | `{np_ref}` | {cat} | {latex} | {status_display} | {notes} |"
        )
    lines.append("")
    out = REF_DIR / "operation-audit.md"
    out.write_text("\n".join(lines))
    print(f"  Generated reference/operation-audit.md ({total} operations)")


def generate_ops_json(registry: dict[str, dict]) -> None:
    """Generate website/public/ops.json — machine-readable operation manifest."""
    weights = load_operation_weights()
    alias_map = load_alias_map(registry)
    alias_groups = build_alias_groups(registry, alias_map)

    ops = []
    for name, info in sorted(registry.items()):
        cat = info["category"]
        mod = info["module"]
        plain, latex = cost_for_op(name, cat)
        ops.append(
            {
                "name": name,
                "module": mod,
                "flopscope_ref": flopscope_ref(name, mod),
                "numpy_ref": numpy_ref(name, mod),
                "category": cat,
                "cost_formula": plain,
                "cost_formula_latex": latex,
                "free": cat == "free",
                "blocked": cat == "blacklisted",
                "status": "blocked" if cat == "blacklisted" else "supported",
                "notes": info.get("notes", ""),
                "weight": resolve_operation_weight(
                    name=name,
                    registry=registry,
                    weights=weights,
                    alias_map=alias_map,
                    alias_groups=alias_groups,
                ),
            }
        )
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    out = PUBLIC_DIR / "ops.json"
    out.write_text(json.dumps({"operations": ops, "total": len(ops)}, indent=2))
    print(f"  Generated ops.json ({len(ops)} operations)")


def generate_cheat_sheet(registry: dict[str, dict]) -> None:
    """Generate docs/reference/cheat-sheet.md — agent-optimized cost reference."""
    REF_DIR.mkdir(parents=True, exist_ok=True)
    by_cat: dict[str, list[str]] = {cat: [] for cat in CATEGORY_ORDER}
    for name, info in sorted(registry.items()):
        cat = info["category"]
        if cat in by_cat:
            by_cat[cat].append(name)

    lines = [
        HEADER,
        "# FLOP Cost Cheat Sheet",
        "",
        "> **flopscope is NOT NumPy.** All computation requires a `BudgetContext`.",
        "> Some operations cost FLOPs, some are free, and 32 are blocked entirely.",
        "",
        "## Key Constraints",
        "",
        "- All counted operations require an active `BudgetContext`",
        "- Budget is checked *before* execution — exceeding it raises `BudgetExhaustedError`",
        "- 32 operations are blocked (I/O, config, state functions)",
        "- `sort`, `argsort`, `trace`, and random sampling are **counted** with analytical FLOP costs",
        "",
        "## Cost by Category",
        "",
        "| Category | Count | Cost Formula |",
        "|----------|-------|-------------|",
    ]
    for cat in CATEGORY_ORDER:
        _, latex = CATEGORY_COST_LATEX[cat]
        label = cat.removeprefix("counted_").replace("_", " ").title()
        if cat == "free":
            label = "Free"
        elif cat == "blacklisted":
            label = "Blocked"
        lines.append(f"| {label} | {len(by_cat[cat])} | {latex} |")
    lines.append("")

    custom_ops = by_cat.get("counted_custom", [])
    if custom_ops:
        lines.append("## Custom Operation Costs")
        lines.append("")
        lines.append("| Operation | Cost Formula | Notes |")
        lines.append("|-----------|-------------|-------|")
        for name in custom_ops:
            _, latex = cost_for_op(name, "counted_custom")
            notes = registry[name].get("notes", "")
            lines.append(f"| `{name}` | {latex} | {notes} |")
        lines.append("")

    lines.append("## Free Operations (complete list)")
    lines.append("")
    free_ops = [f"`{op}`" for op in by_cat["free"]]
    for i in range(0, len(free_ops), 8):
        lines.append(", ".join(free_ops[i : i + 8]))
    lines.append("")

    lines.append("## Blocked Operations (complete list)")
    lines.append("")
    blocked_ops = [f"`{op}`" for op in by_cat["blacklisted"]]
    for i in range(0, len(blocked_ops), 8):
        lines.append(", ".join(blocked_ops[i : i + 8]))
    lines.append("")

    out = REF_DIR / "cheat-sheet.md"
    out.write_text("\n".join(lines))
    print("  Generated reference/cheat-sheet.md")


# ---------------------------------------------------------------------------
# Update counted-ops.md to include polynomial, window, unwrap
# ---------------------------------------------------------------------------


def update_counted_ops_page(registry: dict[str, dict]) -> None:
    """Ensure counted-ops.md includes directives for polynomial, window, unwrap."""
    page = API_DIR / "counted-ops.md"
    content = page.read_text()

    additions = []
    for module_directive, _label in [
        ("flopscope._polynomial", "polynomial"),
        ("flopscope._window", "window"),
        ("flopscope._unwrap", "unwrap"),
    ]:
        if f"::: {module_directive}" not in content:
            additions.append(f"\n::: {module_directive}")

    if additions:
        new_content = content.rstrip() + "\n" + "\n".join(additions) + "\n"
        page.write_text(new_content)
        print(
            f"  Updated api/counted-ops.md with: {', '.join(a.strip().removeprefix('::: ') for a in additions)}"
        )
    else:
        print("  api/counted-ops.md already up to date")


# ---------------------------------------------------------------------------
# Verify coverage
# ---------------------------------------------------------------------------


def get_module_public_names(module_path: str) -> set[str]:
    """Import a module and return its public callable names."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return set()

    if hasattr(mod, "__all__"):
        return {n for n in mod.__all__ if callable(getattr(mod, n, None))}

    return {
        n
        for n, obj in inspect.getmembers(mod)
        if not n.startswith("_") and callable(obj)
    }


def extract_directives_from_file(path: Path) -> list[str]:
    """Extract ::: directives from a markdown file."""
    directives = []
    if not path.exists():
        return directives
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("::: "):
            directives.append(stripped[4:].strip())
    return directives


def _public_doc_sections_from_payload(
    payload: dict[str, object],
) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}

    if payload.get("parameters"):
        sections["Parameters"] = [
            str(item.get("name", ""))
            for item in payload.get("parameters", [])
            if isinstance(item, dict)
        ]
    if payload.get("returns"):
        sections["Returns"] = [
            str(item.get("name", ""))
            for item in payload.get("returns", [])
            if isinstance(item, dict)
        ]

    for section in payload.get("body_sections", []) or []:
        if not isinstance(section, dict):
            continue
        title = str(section.get("title", "")).strip()
        if not title:
            continue
        sections.setdefault(title, []).append(json.dumps(section, sort_keys=True))

    example = payload.get("example")
    if isinstance(example, dict):
        sections.setdefault("Examples", [])
        for key in ("code", "output"):
            value = str(example.get(key, "")).strip()
            if value:
                sections["Examples"].append(value)

    return sections


def collect_public_doc_contract_violations() -> list[str]:
    public_api_index_path = PUBLIC_API_DATA_DIR / "index.json"
    if not public_api_index_path.exists():
        return [f"public API index missing at {public_api_index_path}"]

    public_api_index = json.loads(public_api_index_path.read_text())
    violations: list[str] = []
    for entry in public_api_index.get("entries", []):
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug", "")).strip()
        import_path = str(entry.get("import_path", "")).strip()
        if not slug or not import_path:
            continue
        payload_path = GENERATED_DIR / "public-api" / f"{slug}.json"
        if not payload_path.exists():
            violations.append(
                f"{import_path}: missing generated payload at {payload_path}"
            )
            continue
        payload = json.loads(payload_path.read_text())
        require_parameters = True
        try:
            obj = _resolve_public_api_object(import_path)
            signature = inspect.signature(obj)
            require_parameters = len(signature.parameters) > 0
        except (TypeError, ValueError, AttributeError):
            require_parameters = True
        violations.extend(
            find_public_doc_contract_violations(
                import_path=import_path,
                kind=str(payload.get("kind", "function")),
                summary=str(payload.get("summary", "")),
                sections=_public_doc_sections_from_payload(payload),
                require_parameters=require_parameters,
            )
        )

    return violations


def verify_coverage(registry: dict[str, dict]) -> bool:
    """Verify the generated API reference data matches the registry."""
    if not API_INDEX_PATH.exists():
        print(f"\nAPI index NOT FOUND at {API_INDEX_PATH}")
        return False

    api_index = API_INDEX_PATH.read_text()
    if "<ApiNamespaceHub />" not in api_index:
        print(f"\nAPI index at {API_INDEX_PATH} no longer renders <ApiNamespaceHub />")
        return False

    operation_cost_index_path = (
        WEBSITE / "content" / "docs" / "api" / "operation-cost-index.mdx"
    )
    if not operation_cost_index_path.exists():
        print(f"\nOperation Cost Index page NOT FOUND at {operation_cost_index_path}")
        return False

    if "<OperationCostIndex />" not in operation_cost_index_path.read_text():
        print(
            f"\nOperation Cost Index page at {operation_cost_index_path} "
            "no longer renders <OperationCostIndex />"
        )
        return False

    print("API index renders the namespace hub and a dedicated operation-cost page.")

    ops_json_path = PUBLIC_DIR / "ops.json"
    if not ops_json_path.exists():
        print(f"\nops.json NOT FOUND at {ops_json_path}")
        return False

    ops_data = json.loads(ops_json_path.read_text())
    if "operations" not in ops_data:
        print(f"\nops.json missing 'operations' key at {ops_json_path}")
        return False

    operations = ops_data["operations"]
    if not operations:
        print("\nops.json has no operations")
        return False

    missing_detail_fields = [
        op["name"]
        for op in operations
        if not {"slug", "canonical_path", "detail_href", "detail_json_href", "summary"}
        <= set(op)
    ]
    if missing_detail_fields:
        print("\nops.json entries missing detail fields:")
        for name in missing_detail_fields[:20]:
            print(f"  {name}")
        return False

    print(f"ops.json covers {len(operations)} supported canonical operations.")

    op_docs_path = GENERATED_DIR / "op-docs.json"
    if not op_docs_path.exists():
        print(f"\nop-docs.json NOT FOUND at {op_docs_path}")
        return False

    op_refs_path = GENERATED_DIR / "op-refs.json"
    if not op_refs_path.exists():
        print(f"\nop-refs.json NOT FOUND at {op_refs_path}")
        return False

    example_coverage_path = GENERATED_DIR / "api-example-coverage.json"
    if not example_coverage_path.exists():
        print(f"\napi-example-coverage.json NOT FOUND at {example_coverage_path}")
        return False

    doc_coverage_path = GENERATED_DIR / "op-doc-coverage.json"
    if not doc_coverage_path.exists():
        print(f"\nop-doc-coverage.json NOT FOUND at {doc_coverage_path}")
        return False

    op_docs = json.loads(op_docs_path.read_text())
    if "absolute" not in op_docs:
        print(
            f"\nop-docs.json missing canonical entry for 'absolute' at {op_docs_path}"
        )
        return False

    op_refs = json.loads(op_refs_path.read_text())
    abs_ref = op_refs.get("abs")
    if (
        not isinstance(abs_ref, dict)
        or abs_ref.get("label") != "fnp.absolute"
        or abs_ref.get("href") != "/docs/api/numpy/absolute/"
        or abs_ref.get("canonical_name") != "absolute"
    ):
        print(
            "\nop-refs.json missing structured alias entry for "
            "'abs' -> '/docs/api/numpy/absolute/'"
        )
        return False

    public_api_routes_path = GENERATED_DIR / "public-api-routes.json"
    if not public_api_routes_path.exists():
        print(f"\npublic-api-routes.json NOT FOUND at {public_api_routes_path}")
        return False

    legacy_redirects_path = GENERATED_DIR / "legacy-op-redirects.json"
    if not legacy_redirects_path.exists():
        print(f"\nlegacy-op-redirects.json NOT FOUND at {legacy_redirects_path}")
        return False

    public_api_refs_path = GENERATED_DIR / "public-api-refs.json"
    if not public_api_refs_path.exists():
        print(f"\npublic-api-refs.json NOT FOUND at {public_api_refs_path}")
        return False

    public_api_symbols_path = GENERATED_DIR / "public-api-symbols.json"
    if not public_api_symbols_path.exists():
        print(f"\npublic-api-symbols.json NOT FOUND at {public_api_symbols_path}")
        return False

    symbol_import_map_path = GENERATED_DIR / "symbol-doc-imports.ts"
    if not symbol_import_map_path.exists():
        print(f"\nGenerated symbol import map NOT FOUND at {symbol_import_map_path}")
        return False

    public_api_routes = json.loads(public_api_routes_path.read_text())
    if public_api_routes.get("stats/norm", {}).get("href") != "/docs/api/stats/norm/":
        print("\npublic-api-routes.json missing canonical entry for stats/norm")
        return False
    if (
        public_api_routes.get("stats/norm/pdf", {}).get("href")
        != "/docs/api/stats/norm/pdf/"
    ):
        print("\npublic-api-routes.json missing canonical entry for stats/norm/pdf")
        return False

    legacy_redirects = json.loads(legacy_redirects_path.read_text())
    if legacy_redirects.get("einsum") != "/docs/api/numpy/einsum/":
        print("\nlegacy-op-redirects.json missing redirect for einsum")
        return False

    public_api_refs = json.loads(public_api_refs_path.read_text())
    missing_surface = sorted(
        name
        for name in collect_public_api_surface_names()
        if name not in public_api_refs
    )
    if missing_surface:
        print("\npublic-api-refs.json missing public API entries:")
        for name in missing_surface[:20]:
            print(f"  {name}")
        return False

    public_payload = API_DATA_DIR / "absolute.json"
    if not public_payload.exists():
        print(f"\nPer-op public payload missing for 'absolute' at {public_payload}")
        return False

    generated_payload = GENERATED_DIR / "ops" / "absolute.json"
    if not generated_payload.exists():
        print(
            f"\nPer-op generated payload missing for 'absolute' at {generated_payload}"
        )
        return False

    import_map_path = GENERATED_DIR / "op-doc-imports.ts"
    if not import_map_path.exists():
        print(f"\nGenerated import map NOT FOUND at {import_map_path}")
        return False

    public_api_route_map_path = GENERATED_DIR / "public-api-route-map.json"
    if not public_api_route_map_path.exists():
        print(f"\nPublic API route map NOT FOUND at {public_api_route_map_path}")
        return False

    public_api_import_map_path = GENERATED_DIR / "public-api-leaf-imports.ts"
    if not public_api_import_map_path.exists():
        print(f"\nPublic API import map NOT FOUND at {public_api_import_map_path}")
        return False

    public_api_index_path = PUBLIC_API_DATA_DIR / "index.json"
    if not public_api_index_path.exists():
        print(f"\nPublic API index NOT FOUND at {public_api_index_path}")
        return False

    public_api_route_map = json.loads(public_api_route_map_path.read_text())
    absolute_route = public_api_route_map.get("routes", {}).get(
        "/docs/api/numpy/absolute/"
    )
    if (
        not isinstance(absolute_route, dict)
        or absolute_route.get("import_path") != "flopscope.numpy.absolute"
    ):
        print(
            "\npublic-api-route-map.json missing canonical route entry for "
            "'/docs/api/numpy/absolute/'"
        )
        return False

    public_api_index = json.loads(public_api_index_path.read_text())
    if (
        "entries" not in public_api_index
        or "operation_cost_index" not in public_api_index
    ):
        print(
            "\npublic API index missing 'entries' or 'operation_cost_index' "
            f"at {public_api_index_path}"
        )
        return False

    if not any(
        entry.get("import_path") == "flopscope.numpy.absolute"
        for entry in public_api_index["entries"]
        if isinstance(entry, dict)
    ):
        print(
            "\npublic API index missing canonical entry for 'flopscope.numpy.absolute'"
        )
        return False

    public_api_import_map = public_api_import_map_path.read_text()
    if (
        '"flopscope.numpy.absolute": () => import("./public-api/flopscope-numpy-absolute.json")'
        not in public_api_import_map
    ):
        print(
            "\npublic-api-leaf-imports.ts missing static import for "
            "'flopscope.numpy.absolute'"
        )
        return False

    public_doc_violations = collect_public_doc_contract_violations()
    if public_doc_violations:
        print("\npublic API doc contract violations:")
        for violation in public_doc_violations[:50]:
            print(f"  {violation}")
        if len(public_doc_violations) > 50:
            print(f"  ... and {len(public_doc_violations) - 50} more")
        return False

    print(
        "Generated operation doc manifests, canonical public API artifacts, parser coverage, and example coverage are present."
    )

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate API reference data from the flopscope registry"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify coverage only (no generation)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Use multiple processes to build operation docs (default: 1)",
    )
    args = parser.parse_args()

    assert_supported_docs_env()
    registry = load_registry()

    if args.verify:
        ok = verify_coverage(registry)
        sys.exit(0 if ok else 1)

    print("Generating API reference data...")
    worker_count = max(1, args.workers)
    records = build_operation_doc_records(registry, workers=worker_count)
    alias_map = load_alias_map(registry)
    symbol_records = build_public_api_symbol_records(registry, alias_map)
    write_operation_doc_artifacts(records, WEBSITE)
    write_public_symbol_artifacts(symbol_records, WEBSITE)
    write_public_api_route_artifacts(records, symbol_records, WEBSITE)
    write_public_api_leaf_artifacts(records, registry, WEBSITE)
    write_op_doc_coverage_artifact(records, WEBSITE)
    example_coverage = build_example_coverage(records, API_EXAMPLES_DIR)
    write_example_coverage_artifact(example_coverage, WEBSITE)

    print("\nDone. Run with --verify to check coverage.")


if __name__ == "__main__":
    main()
