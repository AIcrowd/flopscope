# FLOP Weight Calibration Results

## Introduction

Per-operation FLOP weights are multiplicative correction factors that bridge
the gap between flopscope's analytical cost formulas and the actual
floating-point instruction cost observed on hardware. When weights are
loaded, the effective cost of an operation becomes:

$$
\text{cost}(\text{op}) = C(\text{op}, \text{shapes}) \times w(\text{op})
$$

where $C$ is the analytical FLOP formula and $w$ is the weight.
A weight of 25.9 for `sin` means that each analytical FLOP of sine costs
approximately 26 times more in actual floating-point instructions than a
FLOP of addition.

## Methodology

### The correction formula (v3 — overhead-subtracted)

The raw correction factor is measured as:

$$
\alpha_{\text{raw}}(\text{op}) = \mathrm{median}_{D} \left[ \frac{F(\text{op})}{C(\text{op}, \text{params}) \times R} \right]
$$

The weight is computed by subtracting numpy's ufunc dispatch overhead:

$$
w(\text{op}) = \max\bigl(\alpha_{\text{raw}}(\text{op}) - \text{overhead}_{\text{category}}, \ 0\bigr)
$$

Where:

- $\alpha_{\text{raw}}(\text{op})$ is the **raw correction factor** -- the ratio of hardware-observed FP instructions to the analytical FLOP count (FMA = 2 ops, textbook).
- $F(\text{op})$ is the total SIMD-width-weighted count of retired floating-point instructions, measured via the Intel PMU counters `fp_arith_inst_retired.*` (scalar x1, 128-bit x2, 256-bit x4, 512-bit x8).
- $C(\text{op}, \text{params})$ is the analytical FLOP count from flopscope's cost formula (e.g., `numel(output)` for pointwise ops).
- $R$ is the number of repeats per distribution.
- The **median** across 3 input distributions is reported.
- $\text{overhead}_{\text{category}}$ is the ufunc dispatch overhead measured from `np.abs` (bitwise sign-clear, generates zero FP arithmetic — all measured FP instructions are pure overhead). Subtracted per category to remove numpy implementation noise.

### Ufunc overhead subtraction

Numpy's ufunc dispatch layer generates spurious FP instructions (type
resolution, iterator setup, error-state management) even for operations
that perform no FP arithmetic. This overhead is measured and subtracted:

| Category | Overhead source | Typical value |
|----------|----------------|---------------|
| `ufunc_unary` | $\alpha(\texttt{abs})$ | ~0.3 |
| `ufunc_binary` | $\alpha(\texttt{add}) - 1.0$ | ~0.6 |
| `ufunc_reduction` | same as unary | ~0.3 |
| `blas` / `linalg` | 0 (bypasses ufunc) | 0 |
| `custom` (fft, sort, etc.) | 0 | 0 |
| `instructions` (bitwise) | 0 (different counter) | 0 |

After subtraction, weights are clamped to a minimum of 0. Values below 1.0
are expected for ops with less FP work than the overhead measurement
(e.g., bitwise ops that generate 0 FP instructions).

**Note on BLAS/linalg FMA ops:** Both flopscope's analytical FLOP count and
`fp_arith_inst_retired` count each FMA as 2 ops (one multiply + one add).
Pure-FMA ops like matmul therefore show weights near 1.0 (no convention mismatch).

### Measurement modes by category

| Mode | Counter | Used for |
|------|---------|----------|
| **perf** | `fp_arith_inst_retired.*` (SIMD-weighted) | FP operations (default) |
| **instructions** | `instructions` (total retired) | Integer/bitwise ops |
| **timing** | `time.perf_counter_ns()` | Validation; fallback when perf unavailable |

**Complex-number operations** (angle, conj, real, imag, etc.) are measured
with perf counters on complex128 input arrays. Two type-check operations
(`iscomplexobj`, `isrealobj`) use the `instructions` counter.

## Measurement environment

| Parameter | Value |
|-----------|-------|
| CPU | Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz |
| Cores | 128 physical / 128 threads |
| RAM | 0.0 GB |
| Arch | x86_64 (AVX-512 capable) |
| Cache | L1d 48 KB, L1i 32 KB, L2 1280 KB, L3 54 MB |
| Instance | AWS EC2 c6i.metal (bare metal, full PMU access) |
| OS | Linux 6.1.166-197.305.amzn2023.x86_64 |
| Python | 3.14.4 |
| NumPy | 2.1.3 |
| BLAS | auto unknown |
| Measurement mode | perf (hardware counters: `fp_arith_inst_retired.*`) |
| dtype | float64 |
| Repeats | 10 per distribution |
| Distributions | 3 per operation |
| Methodology version | 3.0 |
| Baseline alpha(add) | 1.600096 |
    - **Date:** 2026-04-13
    - **Total calibration time:** 7828.9 seconds

## Baseline details

Ufunc overhead is measured from `np.abs` (bitwise, zero FP arithmetic)
and `np.add` (1 FP add per element). After subtracting per-category
overhead, weights represent the true hardware cost per analytical FLOP:

- **Benchmark command:** `np.add(x, y, out=_out)`
- **Array size:** x: (10000000,), dtype=float64
- **Measured perf instructions:** 160009598.0
- **Measured timing:** 195797715.0 ns
- **$\alpha(\text{add})$:** 1.600096

**[Download full review spreadsheet (CSV)](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/data/weights.csv)**

## Weight tables

### Pointwise Unary (47 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `exp` | 16.0000 | 22.0000 | high | numel(output) | [\_pointwise.py:245](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L245) | Element-wise e^x. |
| `exp2` | 16.0000 | 15.0000 | high | numel(output) | [\_pointwise.py:313](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L313) | Element-wise 2^x. |
| `expm1` | 16.0000 | 41.0000 | high | numel(output) | [\_pointwise.py:314](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L314) | Element-wise e^x - 1 (accurate near zero). |
| `log` | 16.0000 | 31.3410 | high | numel(output) | [\_pointwise.py:246](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L246) | Element-wise natural logarithm. |
| `log2` | 16.0000 | 34.8410 | high | numel(output) | [\_pointwise.py:247](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L247) | Element-wise base-2 logarithm. |
| `log10` | 16.0000 | 35.3410 | high | numel(output) | [\_pointwise.py:248](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L248) | Element-wise base-10 logarithm. |
| `log1p` | 16.0000 | 41.1581 | high | numel(output) | [\_pointwise.py:327](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L327) | Element-wise log(1+x) (accurate near zero). |
| `cbrt` | 16.0000 | 38.0000 | high | numel(output) | [\_pointwise.py:307](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L307) | Element-wise cube root. |
| `sin` | 16.0000 | 39.8606 | high | numel(output) | [\_pointwise.py:253](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L253) | Element-wise sine. |
| `cos` | 16.0000 | 39.9073 | high | numel(output) | [\_pointwise.py:254](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L254) | Element-wise cosine. |
| `tan` | 16.0000 | 60.0000 | high | numel(output) | [\_pointwise.py:379](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L379) | Element-wise tangent. |
| `arcsin` | 16.0000 | 55.9901 | high | numel(output) | [\_pointwise.py:270](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L270) | Element-wise inverse sine. |
| `arccos` | 16.0000 | 52.9901 | high | numel(output) | [\_pointwise.py:268](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L268) | Element-wise inverse cosine. |
| `arctan` | 16.0000 | 47.0000 | high | numel(output) | [\_pointwise.py:272](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L272) | Element-wise inverse tangent. |
| `sinh` | 16.0000 | 33.0000 | high | numel(output) | [\_pointwise.py:365](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L365) | Element-wise hyperbolic sine. |
| `cosh` | 16.0000 | 28.0000 | high | numel(output) | [\_pointwise.py:310](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L310) | Element-wise hyperbolic cosine. |
| `tanh` | 16.0000 | 33.0000 | high | numel(output) | [\_pointwise.py:255](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L255) | Element-wise hyperbolic tangent. |
| `arcsinh` | 16.0000 | 79.0000 | high | numel(output) | [\_pointwise.py:271](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L271) | Element-wise inverse hyperbolic sine. |
| `arccosh` | 16.0000 | 82.5008 | high | numel(output) | [\_pointwise.py:269](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L269) | Element-wise inverse hyperbolic cosine. |
| `arctanh` | 16.0000 | 71.9901 | high | numel(output) | [\_pointwise.py:273](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L273) | Element-wise inverse hyperbolic tangent. |
| `sinc` | 16.0000 | 41.1250 | high | numel(output) | [\_pointwise.py:364](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L364) | Re-benchmarked on c6i.metal with correct input. α_raw=41.4251, overhead=0.3001, weight=41.1250 |
| `i0` | 16.0000 | 111.3745 | high | numel(output) | [\_pointwise.py:317](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L317) | Re-benchmarked on c6i.metal with correct input. α_raw=111.6746, overhead=0.3001, weight=111.3745 |
| `abs` | 1.0000 | 0.8690 | high | numel(output) | [\_pointwise.py:249](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L249) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.8690 |
| `negative` | 1.0000 | 0.8447 | high | numel(output) | [\_pointwise.py:250](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L250) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.8447 |
| `positive` | 1.0000 | 0.9146 | high | numel(output) | [\_pointwise.py:330](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L330) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.9146 |
| `sqrt` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:251](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L251) | Element-wise square root. |
| `square` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:252](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L252) | Element-wise x^2. |
| `reciprocal` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:335](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L335) | Element-wise 1/x. |
| `ceil` | 1.0000 | 0.8783 | high | numel(output) | [\_pointwise.py:257](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L257) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.8783 |
| `floor` | 1.0000 | 0.8783 | high | numel(output) | [\_pointwise.py:258](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L258) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.8783 |
| `trunc` | 1.0000 | 0.8777 | high | numel(output) | [\_pointwise.py:380](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L380) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.8777 |
| `rint` | 1.0000 | 0.8722 | high | numel(output) | [\_pointwise.py:336](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L336) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.8722 |
| `sign` | 1.0000 | 1.0057 | high | numel(output) | [\_pointwise.py:256](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L256) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=1.0057 |
| `signbit` | 1.0000 | 0.3407 | high | numel(output) | [\_pointwise.py:363](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L363) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.3407 |
| `fabs` | 1.0000 | 1.1184 | high | numel(output) | [\_pointwise.py:315](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L315) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=1.1184 |
| `deg2rad` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:311](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L311) | Alias for radians. |
| `rad2deg` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:331](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L331) | Alias for degrees. |
| `degrees` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:312](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L312) | Convert radians to degrees element-wise. |
| `radians` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:332](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L332) | Convert degrees to radians element-wise. |
| `logical_not` | 1.0000 | 0.4378 | high | numel(output) | [\_pointwise.py:328](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L328) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.4378 |
| `frexp` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:384](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L384) | Decompose x into mantissa and exponent element-wise. |
| `modf` | 1.0000 | 0.9901 | low | numel(output) | [\_pointwise.py:383](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L383) | Return fractional and integral parts element-wise. |
| `spacing` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:378](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L378) | Return ULP spacing for each element. |
| `nan_to_num` | 1.0000 | 3.0865 | high | numel(output) | [\_pointwise.py:329](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L329) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=3.0865 |
| `isneginf` | 1.0000 | 0.8331 | high | numel(output) | [\_pointwise.py:323](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L323) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.8331 |
| `isposinf` | 1.0000 | 0.9379 | high | numel(output) | [\_pointwise.py:324](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L324) | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.9379 |
| `isclose` | 1.0000 | 3.0000 | medium | numel(output) | [\_pointwise.py:399](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L399) | Element-wise approximate equality test. |

### Pointwise Binary (34 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `floor_divide` | 16.0000 | 3.1888 | medium | numel(output) | [\_pointwise.py:441](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L441) | Element-wise floor division. |
| `power` | 16.0000 | 72.1819 | medium | numel(output) | [\_pointwise.py:424](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L424) | Element-wise exponentiation x**y. |
| `float_power` | 16.0000 | 31.1853 | low | numel(output) | [\_pointwise.py:440](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L440) | Element-wise exponentiation in float64. |
| `mod` | 16.0000 | 0.1821 | low | numel(output) | [\_pointwise.py:425](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L425) | Element-wise modulo. |
| `remainder` | 16.0000 | 0.1821 | low | numel(output) | [\_pointwise.py:463](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L463) | Element-wise remainder (same as mod). |
| `fmod` | 16.0000 | 5.5996 | high | numel(output) | [\_pointwise.py:444](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L444) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=5.5996 |
| `arctan2` | 16.0000 | 53.0000 | high | numel(output) | [\_pointwise.py:431](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L431) | Element-wise arctan(y/x) considering quadrant. |
| `hypot` | 16.0000 | 10.5006 | high | numel(output) | [\_pointwise.py:449](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L449) | Element-wise Euclidean norm sqrt(x1^2 + x2^2). |
| `logaddexp` | 16.0000 | 32.5991 | low | numel(output) | [\_pointwise.py:455](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L455) | log(exp(x1) + exp(x2)) element-wise. |
| `logaddexp2` | 16.0000 | 34.0363 | low | numel(output) | [\_pointwise.py:456](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L456) | log2(2**x1 + 2**x2) element-wise. |
| `matvec` | 1.0000 | 0.5551 |  | output_size * contracted_axis |  | EC2 timing = 0.56. BLAS matrix-vector product — efficient. |
| `vecmat` | 1.0000 | 0.6085 |  | output_size * contracted_axis |  | EC2 timing = 0.61. BLAS vector-matrix product — efficient. |
| `add` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:418](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L418) | Element-wise addition. |
| `subtract` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:419](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L419) | Element-wise subtraction. |
| `multiply` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:420](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L420) | Element-wise multiplication. |
| `divide` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:421](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L421) | Element-wise true division. |
| `true_divide` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:465](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L465) | Element-wise true division (explicit). |
| `maximum` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:422](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L422) | Element-wise maximum (propagates NaN). |
| `minimum` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:423](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L423) | Element-wise minimum (propagates NaN). |
| `fmax` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:442](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L442) | Element-wise maximum ignoring NaN. |
| `fmin` | 1.0000 | 1.0000 | medium | numel(output) | [\_pointwise.py:443](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L443) | Element-wise minimum ignoring NaN. |
| `greater` | 1.0000 | 0.5759 | high | numel(output) | [\_pointwise.py:446](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L446) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=0.5759 |
| `greater_equal` | 1.0000 | 0.5734 | high | numel(output) | [\_pointwise.py:447](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L447) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=0.5734 |
| `less` | 1.0000 | 0.5761 | high | numel(output) | [\_pointwise.py:453](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L453) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=0.5761 |
| `less_equal` | 1.0000 | 0.5745 | high | numel(output) | [\_pointwise.py:454](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L454) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=0.5745 |
| `equal` | 1.0000 | 0.5761 | high | numel(output) | [\_pointwise.py:439](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L439) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=0.5761 |
| `not_equal` | 1.0000 | 0.5736 | high | numel(output) | [\_pointwise.py:461](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L461) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=0.5736 |
| `logical_and` | 1.0000 | 0.8026 | high | numel(output) | [\_pointwise.py:457](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L457) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=0.8026 |
| `logical_or` | 1.0000 | 0.7986 | high | numel(output) | [\_pointwise.py:458](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L458) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=0.7986 |
| `logical_xor` | 1.0000 | 0.8007 | high | numel(output) | [\_pointwise.py:459](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L459) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=0.8007 |
| `copysign` | 1.0000 | 1.1021 | high | numel(output) | [\_pointwise.py:438](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L438) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=1.1021 |
| `nextafter` | 1.0000 | 5.7999 | low | numel(output) | [\_pointwise.py:460](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L460) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=5.7999 |
| `ldexp` | 1.0000 | 3.3667 | high | numel(output) | [\_pointwise.py:451](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L451) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=3.3667 |
| `heaviside` | 1.0000 | 1.3916 | high | numel(output) | [\_pointwise.py:448](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L448) | Timing-based (fp_arith_inst_retired blind to comparisons). Ratio vs add=1.3916 |

### Reductions (35 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `std` | 2.0000 | 4.0000 | high | numel(input) | [\_pointwise.py:533](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L533) | Standard deviation; cost_multiplier=2 (two passes). |
| `var` | 2.0000 | 4.0000 | high | numel(input) | [\_pointwise.py:534](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L534) | Variance; cost_multiplier=2 (two passes). |
| `nanstd` | 2.0000 | 4.0000 | high | numel(input) | [\_pointwise.py:564](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L564) | Standard deviation ignoring NaNs. |
| `nanvar` | 2.0000 | 4.0000 | high | numel(input) | [\_pointwise.py:566](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L566) | Variance ignoring NaNs. |
| `sum` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:528](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L528) | Sum of array elements. |
| `prod` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:531](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L531) | Product of array elements. |
| `mean` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:532](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L532) | Arithmetic mean of array elements. |
| `max` | 1.0000 | 1.0010 | medium | numel(input) | [\_pointwise.py:529](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L529) | Maximum value of array. |
| `min` | 1.0000 | 1.0010 | medium | numel(input) | [\_pointwise.py:530](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L530) | Minimum value of array. |
| `argmax` | 1.0000 | 0.2320 | low | numel(input) | [\_pointwise.py:535](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L535) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=0.2320 |
| `argmin` | 1.0000 | 0.2313 | low | numel(input) | [\_pointwise.py:536](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L536) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=0.2313 |
| `any` | 1.0000 | 0.2513 | low | numel(input) | [\_pointwise.py:547](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L547) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=0.2513 |
| `all` | 1.0000 | 0.2525 | low | numel(input) | [\_pointwise.py:544](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L544) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=0.2525 |
| `cumsum` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:537](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L537) | Cumulative sum of array elements. |
| `cumprod` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:538](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L538) | Cumulative product of array elements. |
| `nansum` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:565](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L565) | Sum ignoring NaNs. |
| `nanmean` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:558](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L558) | Mean ignoring NaNs. |
| `nanmax` | 1.0000 | 1.0010 | medium | numel(input) | [\_pointwise.py:557](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L557) | Maximum ignoring NaNs. |
| `nanmin` | 1.0000 | 1.0010 | medium | numel(input) | [\_pointwise.py:560](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L560) | Minimum ignoring NaNs. |
| `nanprod` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:562](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L562) | Product ignoring NaNs. |
| `median` | 1.0000 | 5.3855 | low | numel(input) | [\_pointwise.py:552](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L552) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=5.3855 |
| `nanmedian` | 1.0000 | 5.7796 | low | numel(input) | [\_pointwise.py:559](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L559) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=5.7796 |
| `percentile` | 1.0000 | 6.5693 | low | numel(input) | [\_pointwise.py:567](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L567) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=6.5693 |
| `nanpercentile` | 1.0000 | 6.9821 | low | numel(input) | [\_pointwise.py:561](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L561) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=6.9821 |
| `quantile` | 1.0000 | 6.5837 | low | numel(input) | [\_pointwise.py:568](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L568) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=6.5837 |
| `nanquantile` | 1.0000 | 6.9870 | low | numel(input) | [\_pointwise.py:563](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L563) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=6.9870 |
| `count_nonzero` | 1.0000 | 0.7773 | low | numel(input) | [\_pointwise.py:549](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L549) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=0.7773 |
| `average` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:548](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L548) | Weighted average of array elements. |
| `nanargmax` | 1.0000 | 1.4897 | low | numel(input) | [\_pointwise.py:553](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L553) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=1.4897 |
| `nanargmin` | 1.0000 | 1.4777 | low | numel(input) | [\_pointwise.py:554](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L554) | Timing-based weight (fp_arith_inst_retired blind to this op). Ratio vs add=1.4777 |
| `ptp` | 1.0000 | 2.0020 | high | numel(input) | [\_pointwise.py:581](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L581) | Peak-to-peak (max - min) range of array. |
| `nancumprod` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:555](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L555) | Cumulative product ignoring NaNs. |
| `nancumsum` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:556](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L556) | Cumulative sum ignoring NaNs. |
| `cumulative_sum` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:551](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L551) | Cumulative sum (NumPy 2.x array API). |
| `cumulative_prod` | 1.0000 | 1.0000 | medium | numel(input) | [\_pointwise.py:550](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L550) | Cumulative product (NumPy 2.x array API). |

### Sorting (17 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `sort` | 1.0000 | 4.0363 | medium | n * ceil(log2(n)) | [\_sorting\_ops.py:43](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L43) | Comparison sort; cost = n*ceil(log2(n)) per slice. |
| `argsort` | 1.0000 | 4.8736 | high | n * ceil(log2(n)) | [\_sorting\_ops.py:62](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L62) | Indirect sort; cost = n*ceil(log2(n)) per slice. |
| `lexsort` | 1.0000 | 0.3723 | low | k * n * ceil(log2(n)) | [\_sorting\_ops.py:88](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L88) | Multi-key sort; cost = k*n*ceil(log2(n)). |
| `partition` | 1.0000 | 4.4268 | medium | n | [\_sorting\_ops.py:113](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L113) | Quickselect; cost = n per slice. |
| `argpartition` | 1.0000 | 4.6015 | medium | n * len(kth) | [\_sorting\_ops.py:140](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L140) | Indirect partition; cost = n per slice. |
| `searchsorted` | 1.0000 | 0.7445 | low | m * ceil(log2(n)) | [\_sorting\_ops.py:170](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L170) | Binary search; cost = m*ceil(log2(n)). |
| `unique` | 1.0000 | 4.0363 | medium | n * ceil(log2(n)) | [\_sorting\_ops.py:231](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L231) | Sort-based unique; cost = n*ceil(log2(n)). |
| `in1d` | 1.0000 | 4.2770 | high | (n+m) * ceil(log2(n+m)) | [\_sorting\_ops.py:317](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L317) | Set membership; cost = (n+m)*ceil(log2(n+m)). |
| `isin` | 1.0000 | 4.2770 | high | (n+m) * ceil(log2(n+m)) | [\_sorting\_ops.py:330](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L330) | Set membership; cost = (n+m)*ceil(log2(n+m)). |
| `intersect1d` | 1.0000 | 7.7863 | high | (n+m) * ceil(log2(n+m)) | [\_sorting\_ops.py:344](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L344) | Set intersection; cost = (n+m)*ceil(log2(n+m)). |
| `setdiff1d` | 1.0000 | 3.8751 | medium | (n+m) * ceil(log2(n+m)) | [\_sorting\_ops.py:376](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L376) | Set difference; cost = (n+m)*ceil(log2(n+m)). |
| `setxor1d` | 1.0000 | 7.7863 | high | (n+m) * ceil(log2(n+m)) | [\_sorting\_ops.py:393](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L393) | Symmetric set difference; cost = (n+m)*ceil(log2(n+m)). |
| `union1d` | 1.0000 | 4.2686 | medium | (n+m) * ceil(log2(n+m)) | [\_sorting\_ops.py:361](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L361) | Set union; cost = (n+m)*ceil(log2(n+m)). |
| `unique_all` | 1.0000 | 0.3720 | low | n * ceil(log2(n)) | [\_sorting\_ops.py:243](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L243) | Sort-based unique; cost = n*ceil(log2(n)). |
| `unique_counts` | 1.0000 | 4.0363 | medium | n * ceil(log2(n)) | [\_sorting\_ops.py:256](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L256) | Sort-based unique; cost = n*ceil(log2(n)). |
| `unique_inverse` | 1.0000 | 4.8736 | high | n * ceil(log2(n)) | [\_sorting\_ops.py:272](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L272) | Sort-based unique; cost = n*ceil(log2(n)). |
| `unique_values` | 1.0000 | 4.0363 | medium | n * ceil(log2(n)) | [\_sorting\_ops.py:288](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L288) | Sort-based unique; cost = n*ceil(log2(n)). |

### FFT (18 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `fft.fft` | 1.0000 | 0.8404 | medium | 5*n*ceil(log2(n)) | [\_transforms.py:173](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L173) | 1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.ifft` | 1.0000 | 1.3274 | high | 5*n*ceil(log2(n)) | [\_transforms.py:189](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L189) | Inverse 1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.rfft` | 1.0000 | 0.8288 | medium | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:205](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L205) | 1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.irfft` | 1.0000 | 0.9466 | high | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:221](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L221) | Inverse 1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.fft2` | 1.0000 | 0.7183 | medium | 5*n*ceil(log2(n)) | [\_transforms.py:242](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L242) | 2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.ifft2` | 1.0000 | 0.7693 | high | 5*n*ceil(log2(n)) | [\_transforms.py:265](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L265) | Inverse 2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.rfft2` | 1.0000 | 0.7013 | medium | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:288](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L288) | 2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.irfft2` | 1.0000 | 0.8267 | high | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:314](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L314) | Inverse 2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.fftn` | 1.0000 | 0.7183 | medium | 5*n*ceil(log2(n)) | [\_transforms.py:339](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L339) | N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.ifftn` | 1.0000 | 0.7693 | high | 5*n*ceil(log2(n)) | [\_transforms.py:363](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L363) | Inverse N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.rfftn` | 1.0000 | 0.7013 | medium | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:387](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L387) | N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.irfftn` | 1.0000 | 0.8267 | high | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:423](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L423) | Inverse N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.hfft` | 1.0000 | 2.2743 | high | 5*n*ceil(log2(n)) | [\_transforms.py:443](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L443) | FFT of Hermitian-symmetric signal. Cost: 5*n_out*ceil(log2(n_out)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.ihfft` | 1.0000 | 0.4244 | medium | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:462](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/fft/_transforms.py#L462) | Inverse FFT of Hermitian signal. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.fftfreq` | 0.0000 | 0.0000 |  |  |  |  |
| `fft.fftshift` | 0.0000 | 0.0000 |  |  |  |  |
| `fft.ifftshift` | 0.0000 | 0.0000 |  |  |  |  |
| `fft.rfftfreq` | 0.0000 | 0.0000 |  |  |  |  |

### Linalg (14 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `linalg.cholesky` | 4.0000 | 0.5350 | high | n^3 | [\_decompositions.py:45](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_decompositions.py#L45) | Cholesky decomposition. Cost: $n^3$. |
| `linalg.qr` | 4.0000 | 2.7316 | high | m*n*min(m,n) | [\_decompositions.py:83](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_decompositions.py#L83) | QR decomposition. Cost: $m \cdot n \cdot \min(m,n)$. |
| `linalg.eig` | 4.0000 | 14.5964 | medium | n^3 | [\_decompositions.py:119](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_decompositions.py#L119) | Eigendecomposition. Cost: $n^3$. |
| `linalg.eigh` | 4.0000 | 4.6253 | high | n^3 | [\_decompositions.py:155](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_decompositions.py#L155) | Symmetric eigendecomposition. Cost: $n^3$. |
| `linalg.eigvals` | 4.0000 | 7.0728 | medium | n^3 | [\_decompositions.py:192](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_decompositions.py#L192) | Eigenvalues only. Cost: $n^3$. |
| `linalg.eigvalsh` | 4.0000 | 1.5505 | high | n^3 | [\_decompositions.py:228](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_decompositions.py#L228) | Symmetric eigenvalues. Cost: $n^3$. |
| `linalg.svd` | 4.0000 | 8.9457 | medium | m*n*min(m,n) | [\_svd.py:67](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_svd.py#L67) | Singular value decomposition; cost ~ O(min(m,n)*m*n). |
| `linalg.svdvals` | 4.0000 | 2.8094 | high | m*n*min(m,n) | [\_decompositions.py:274](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_decompositions.py#L274) | Singular values only. Cost: m*n*min(m,n) (Golub-Reinsch). |
| `linalg.solve` | 4.0000 | 0.6690 | medium | n^3 | [\_solvers.py:50](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_solvers.py#L50) | Solve Ax=b. Cost: $n^3$. |
| `linalg.inv` | 4.0000 | 2.0003 | medium | n^3 | [\_solvers.py:92](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_solvers.py#L92) | Matrix inverse. Cost: $n^3$ (LU + solve). |
| `linalg.lstsq` | 4.0000 | 2.9676 | high | m*n*min(m,n) | [\_solvers.py:138](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_solvers.py#L138) | Least squares. Cost: m*n*min(m,n) (LAPACK gelsd/SVD). |
| `linalg.pinv` | 4.0000 | 10.9467 | medium | m*n*min(m,n) | [\_solvers.py:178](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_solvers.py#L178) | Pseudoinverse. Cost: m*n*min(m,n) (via SVD). |
| `linalg.det` | 4.0000 | 0.6670 | medium | n^3 | [\_properties.py:84](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_properties.py#L84) | Determinant. Cost: $n^3$. |
| `linalg.slogdet` | 4.0000 | 0.6670 | medium | n^3 | [\_properties.py:123](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_properties.py#L123) | Sign + log determinant. Cost: $n^3$. |

### Linalg Delegates (17 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `linalg.cond` | 4.0000 | 2.9318 | high | m*n*min(m,n) | [\_properties.py:339](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_properties.py#L339) | Condition number. Cost: m*n*min(m,n) (via SVD). |
| `linalg.matrix_rank` | 4.0000 | 2.9318 | high | m*n*min(m,n) | [\_properties.py:383](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_properties.py#L383) | Matrix rank. Cost: m*n*min(m,n) (via SVD). |
| `linalg.tensorinv` | 4.0000 | 2.2304 | high | n^3 (delegates to inv) | [\_solvers.py:269](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_solvers.py#L269) | Weight=4 (same as linalg.inv). Reshapes to 2D then calls inv. Cost n^3 in formula. |
| `linalg.tensorsolve` | 4.0000 | 0.9282 | high | n^3 (delegates to solve) | [\_solvers.py:224](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_solvers.py#L224) | Weight=4 (same as linalg.solve). Reshapes to 2D then calls solve. Cost n^3 in formula. |
| `linalg.cross` | 1.0000 | 1.7413 | high | 6*n |  | Delegates to `me.cross` which charges `numel(output)` FLOPs. |
| `linalg.matmul` | 1.0000 | 2.0009 | high | MNK |  | Delegates to `me.matmul` which charges `m*k*n` FLOPs (FMA=1). |
| `linalg.matrix_norm` | 1.0000 | 1.1225 | high | 2*numel (fro/L1/Linf, FMA=2) or 4*m*n*min(m,n) (ord=2/nuc) | [\_properties.py:289](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_properties.py#L289) | Weight=1 (baked into cost function). Elementwise norms cost 2*numel (FMA=2). SVD-based norms (ord=2, nuc) cost 4*m*n*min(m,n) — the 4x is baked in for SVD consistency. |
| `linalg.matrix_power` | 1.0000 | 2.0030 | high | (ceil(log2(k))+popcount(k)-1)*n^3 | [\_compound.py:116](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_compound.py#L116) | Matrix power. Cost: $(\lfloor\log_2 k\rfloor + \text{popcount}(k) - 1) \cdot n^3$ (exponentiation by squaring). |
| `linalg.multi_dot` | 1.0000 | 1.0029 | high | 2 * sum of chain MNK costs (FMA=2) | [\_compound.py:68](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_compound.py#L68) | Chain matmul. Cost: 2 * sum of optimal chain matmul costs (CLRS §15.2) (FMA=2). |
| `linalg.norm` | 1.0000 | 2.2412 | high | 2*numel (vector/fro/L1/Linf, FMA=2) or 4*m*n*min(m,n) (ord=2/nuc) | [\_properties.py:187](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_properties.py#L187) | Weight=1 (baked into cost function). Elementwise norms cost 2*numel (FMA=2). SVD-based norms (ord=2, nuc) cost 4*m*n*min(m,n) — the 4x is baked in for SVD consistency. |
| `linalg.outer` | 1.0000 | 1.0001 | high | M*N |  | Delegates to `me.outer` which charges `m*n` FLOPs. |
| `linalg.tensordot` | 1.0000 | 2.0001 | high | product of free * contracted dims |  | Delegates to `me.tensordot` which charges FLOPs based on contraction. |
| `linalg.trace` | 1.0000 | 0.7872 | high | min(m,n) | [\_properties.py:45](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_properties.py#L45) | Blacklisted per reviewer — datetime ops not in scope. |
| `linalg.vecdot` | 1.0000 | 2.4841 | medium | batch*K |  | Delegates to `me.vecdot` which charges `2*n` FLOPs. |
| `linalg.vector_norm` | 1.0000 | 2.2412 | high | 2*numel (FMA=2) | [\_properties.py:238](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/_properties.py#L238) | Vector norm. Cost: 2*numel (FMA=2 — one multiply + accumulate per element). |
| `linalg.diagonal` | 0.0000 | 0.0000 |  |  |  |  |
| `linalg.matrix_transpose` | 0.0000 | 0.0000 |  |  |  |  |

### Contractions (9 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `dot` | 1.0000 | 2.0012 | high | MNK | [\_pointwise.py:619](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L619) | Dot product; cost = M*K*N (FMA=1). |
| `matmul` | 1.0000 | 2.0012 | high | MNK | [\_pointwise.py:655](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L655) | Matrix multiplication; cost = M*K*N (FMA=1). |
| `inner` | 1.0000 | 2.6010 | medium | N (a.size) | [\_pointwise.py:681](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L681) | Inner product; cost = N (FMA=1). |
| `vdot` | 1.0000 | 2.6010 | medium | N (a.size) | [\_pointwise.py:739](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L739) | Dot product with conjugation; cost = N (FMA=1). |
| `vecdot` | 1.0000 | 2.6019 | medium | batch * K (output_size * contracted_axis) | [\_pointwise.py:485](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L485) | Vector dot product along last axis. |
| `outer` | 1.0000 | 1.0002 | high | M*N | [\_pointwise.py:697](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L697) | Outer product of two vectors; cost = M*N. |
| `tensordot` | 1.0000 | 2.0001 | high | product of free * contracted dims | [\_\_init\_\_.py:74](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/__init__.py#L74) | Tensor dot product along specified axes. |
| `kron` | 1.0000 | 1.0002 | high | numel(output) | [\_pointwise.py:755](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L755) | Kronecker product; cost proportional to output size. |
| `einsum` | 1.0000 | 2.0012 | high | product of index dims (FMA=1) | [\_einsum.py:139](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_einsum.py#L139) | Generalized Einstein summation. |

### Polynomial (10 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `roots` | 16.0000 | 10.2924 | high | degree^3 | [\_polynomial.py:217](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_polynomial.py#L217) | Return roots of polynomial with given coefficients. Cost: $n^3$ (companion matrix eig, simplified). |
| `polyval` | 1.0000 | 2.0214 | high | 2 * n * degree (FMA=2) | [\_polynomial.py:78](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_polynomial.py#L78) | Evaluate polynomial at given points. Cost: $2m \cdot \text{deg}$ (Horner's method, FMA=2). |
| `polyfit` | 1.0000 | 1.1977 | high | 2 * n * (degree+1)^2 | [\_polynomial.py:187](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_polynomial.py#L187) | Least squares polynomial fit. Cost: 2 * m * (deg+1)^2 FLOPs. |
| `polyadd` | 1.0000 | 10.8861 | high | degree + 1 | [\_polynomial.py:96](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_polynomial.py#L96) | Add two polynomials. Cost: max(n1, n2) FLOPs. |
| `polysub` | 1.0000 | 10.8861 | high | degree + 1 | [\_polynomial.py:113](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_polynomial.py#L113) | Difference (subtraction) of two polynomials. Cost: max(n1, n2) FLOPs. |
| `polymul` | 1.0000 | 2.0976 | high | (degree+1)^2 | [\_polynomial.py:158](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_polynomial.py#L158) | Multiply polynomials. Cost: n1 * n2 FLOPs. |
| `polydiv` | 1.0000 | 0.1393 | high | (degree+1)^2 | [\_polynomial.py:174](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_polynomial.py#L174) | Divide one polynomial by another. Cost: n1 * n2 FLOPs. |
| `polyder` | 1.0000 | 11.7564 | high | degree + 1 | [\_polynomial.py:127](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_polynomial.py#L127) | Differentiate polynomial. Cost: n FLOPs. |
| `polyint` | 1.0000 | 10.7960 | high | degree + 1 | [\_polynomial.py:140](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_polynomial.py#L140) | Integrate polynomial. Cost: n FLOPs. |
| `poly` | 1.0000 | 2.1195 | high | degree^2 | [\_polynomial.py:204](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_polynomial.py#L204) | Polynomial from roots. Cost: $n^2$ FLOPs. |

### Random (49 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `random.standard_normal` | 16.0000 | 22.3069 | high | numel(output) | [\_\_init\_\_.py:119](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L119) | Sampling; cost = numel(output). |
| `random.standard_exponential` | 16.0000 | 27.0629 | high | numel(output) | [\_\_init\_\_.py:121](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L121) | Sampling; cost = numel(output). |
| `random.standard_cauchy` | 16.0000 | 45.6145 | high | numel(output) | [\_\_init\_\_.py:133](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L133) | Sampling; cost = numel(output). |
| `random.standard_gamma` | 16.0000 | 27.0629 | high | numel(output) | [\_\_init\_\_.py:135](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L135) | Sampling; cost = numel(output). |
| `random.standard_t` | 16.0000 | 71.1389 | high | numel(output) | [\_\_init\_\_.py:134](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L134) | Sampling; cost = numel(output). |
| `random.poisson` | 16.0000 | 43.9992 | high | numel(output) | [\_\_init\_\_.py:124](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L124) | Sampling; cost = numel(output). |
| `random.binomial` | 16.0000 | 28.9996 | high | numel(output) | [\_\_init\_\_.py:125](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L125) | Sampling; cost = numel(output). |
| `random.beta` | 16.0000 | 88.5899 | high | numel(output) | [\_\_init\_\_.py:151](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L151) | Sampling; cost = numel(output). |
| `random.chisquare` | 16.0000 | 29.0629 | high | numel(output) | [\_\_init\_\_.py:145](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L145) | Sampling; cost = numel(output). |
| `random.dirichlet` | 16.0000 | 120.8423 | high | numel(output) | [\_\_init\_\_.py:157](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L157) | Sampling; cost = numel(output). |
| `random.exponential` | 16.0000 | 28.0629 | high | numel(output) | [\_\_init\_\_.py:123](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L123) | Sampling; cost = numel(output). |
| `random.f` | 16.0000 | 93.4150 | high | numel(output) | [\_\_init\_\_.py:150](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L150) | Sampling; cost = numel(output). |
| `random.gamma` | 16.0000 | 44.5281 | high | numel(output) | [\_\_init\_\_.py:152](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L152) | Sampling; cost = numel(output). |
| `random.geometric` | 16.0000 | 6.0000 | high | numel(output) | [\_\_init\_\_.py:126](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L126) | Sampling; cost = numel(output). |
| `random.gumbel` | 16.0000 | 51.8584 | high | numel(output) | [\_\_init\_\_.py:138](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L138) | Sampling; cost = numel(output). |
| `random.hypergeometric` | 16.0000 | 573.5947 | high | numel(output) | [\_\_init\_\_.py:127](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L127) | Sampling; cost = numel(output). |
| `random.laplace` | 16.0000 | 29.5617 | high | numel(output) | [\_\_init\_\_.py:139](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L139) | Sampling; cost = numel(output). |
| `random.logistic` | 16.0000 | 29.5397 | high | numel(output) | [\_\_init\_\_.py:140](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L140) | Sampling; cost = numel(output). |
| `random.lognormal` | 16.0000 | 44.3069 | high | numel(output) | [\_\_init\_\_.py:141](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L141) | Sampling; cost = numel(output). |
| `random.logseries` | 16.0000 | 43.5238 | high | numel(output) | [\_\_init\_\_.py:129](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L129) | Sampling; cost = numel(output). |
| `random.multinomial` | 16.0000 | 136.9985 | high | numel(output) | [\_\_init\_\_.py:153](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L153) | Sampling; cost = numel(output). |
| `random.multivariate_normal` | 16.0000 | 433.0721 | high | numel(output) | [\_\_init\_\_.py:155](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L155) | Sampling; cost = numel(output). |
| `random.negative_binomial` | 16.0000 | 141.8094 | high | numel(output) | [\_\_init\_\_.py:128](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L128) | Sampling; cost = numel(output). |
| `random.noncentral_chisquare` | 16.0000 | 150.1467 | high | numel(output) | [\_\_init\_\_.py:147](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L147) | Sampling; cost = numel(output). |
| `random.noncentral_f` | 16.0000 | 120.8930 | high | numel(output) | [\_\_init\_\_.py:149](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L149) | Sampling; cost = numel(output). |
| `random.normal` | 16.0000 | 24.3069 | high | numel(output) | [\_\_init\_\_.py:117](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L117) | Sampling; cost = numel(output). |
| `random.pareto` | 16.0000 | 49.0629 | high | numel(output) | [\_\_init\_\_.py:131](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L131) | Sampling; cost = numel(output). |
| `random.power` | 16.0000 | 110.0629 | high | numel(output) | [\_\_init\_\_.py:130](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L130) | Sampling; cost = numel(output). |
| `random.randn` | 16.0000 | 22.3069 | high | numel(output) | [\_\_init\_\_.py:110](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L110) | Sampling; cost = numel(output). |
| `random.rayleigh` | 16.0000 | 38.0709 | high | numel(output) | [\_\_init\_\_.py:132](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L132) | Sampling; cost = numel(output). |
| `random.triangular` | 16.0000 | 11.0000 | high | numel(output) | [\_\_init\_\_.py:144](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L144) | Sampling; cost = numel(output). |
| `random.vonmises` | 16.0000 | 104.4415 | high | numel(output) | [\_\_init\_\_.py:142](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L142) | Sampling; cost = numel(output). |
| `random.wald` | 16.0000 | 39.9707 | high | numel(output) | [\_\_init\_\_.py:143](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L143) | Sampling; cost = numel(output). |
| `random.weibull` | 16.0000 | 89.0629 | high | numel(output) | [\_\_init\_\_.py:136](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L136) | Sampling; cost = numel(output). |
| `random.zipf` | 16.0000 | 229.7840 | high | numel(output) | [\_\_init\_\_.py:137](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L137) | Sampling; cost = numel(output). |
| `random.uniform` | 1.0000 | 5.0001 | high | numel(output) | [\_\_init\_\_.py:118](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L118) | Sampling; cost = numel(output). |
| `random.permutation` | 1.0000 | 0.0001 | high | numel(output) | [\_\_init\_\_.py:198](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L198) | Shuffle; cost = n*ceil(log2(n)). |
| `random.shuffle` | 1.0000 | 0.2001 | high | numel(output) | [\_\_init\_\_.py:213](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L213) | Shuffle; cost = n*ceil(log2(n)). |
| `random.choice` | 1.0000 | 0.0001 | high | numel(output) | [\_\_init\_\_.py:237](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L237) | Sampling; cost = numel(output) if replace, n*ceil(log2(n)) if not. |
| `random.rand` | 1.0000 | 3.0001 | high | numel(output) | [\_\_init\_\_.py:109](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L109) | Sampling; cost = numel(output). |
| `random.randint` | 1.0000 | 0.0001 | high | numel(output) | [\_\_init\_\_.py:158](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L158) | Sampling; cost = numel(output). |
| `random.random` | 1.0000 | 3.0001 | high | numel(output) | [\_\_init\_\_.py:179](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L179) | Sampling; cost = numel(output). |
| `random.random_sample` | 1.0000 | 3.0001 | high | numel(output) | [\_\_init\_\_.py:180](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/random/__init__.py#L180) | Sampling; cost = numel(output). |
| `random.bytes` | 1.0000 | 0.8610 |  | numel(output) |  | EC2 timing = 0.86. Fast PRNG byte generation — cheaper than add. |
| `random.random_integers` | 1.0000 | 3.5105 |  | numel(output) |  | EC2 timing = 3.51. Deprecated randint wrapper. |
| `random.default_rng` | 0.0000 | 0.0000 |  |  |  |  |
| `random.get_state` | 0.0000 | 0.0000 |  |  |  |  |
| `random.seed` | 0.0000 | 0.0000 |  |  |  |  |
| `random.set_state` | 0.0000 | 0.0000 |  |  |  |  |

### Stats (24 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `stats.norm.pdf` | 16.0000 | 27.0000 | high | numel(input) |  | Raw alpha=27.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.norm.cdf` | 16.0000 | 47.5991 | low | numel(input) |  | Raw alpha=47.8993, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.norm.ppf` | 16.0000 | 83.0527 | high | numel(input) |  | Raw alpha=83.3529, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.expon.pdf` | 16.0000 | 25.0000 | high | numel(input) |  | Raw alpha=25.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.expon.cdf` | 16.0000 | 25.0000 | high | numel(input) |  | Raw alpha=25.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.expon.ppf` | 16.0000 | 43.0000 | high | numel(input) |  | Raw alpha=43.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.cauchy.cdf` | 16.0000 | 51.0000 | high | numel(input) |  | Raw alpha=51.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.cauchy.ppf` | 16.0000 | 64.0000 | high | numel(input) |  | Raw alpha=64.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.logistic.pdf` | 16.0000 | 28.0000 | high | numel(input) |  | Raw alpha=28.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.logistic.cdf` | 16.0000 | 26.0000 | high | numel(input) |  | Raw alpha=26.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.logistic.ppf` | 16.0000 | 35.0000 | high | numel(input) |  | Raw alpha=35.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.laplace.pdf` | 16.0000 | 25.0000 | high | numel(input) |  | Raw alpha=25.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.laplace.cdf` | 16.0000 | 49.0000 | high | numel(input) |  | Raw alpha=49.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.laplace.ppf` | 16.0000 | 71.0000 | high | numel(input) |  | Raw alpha=71.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.lognorm.pdf` | 16.0000 | 62.0000 | high | numel(input) |  | Raw alpha=62.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.lognorm.cdf` | 16.0000 | 69.9835 | medium | numel(input) |  | Raw alpha=70.2837, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.lognorm.ppf` | 16.0000 | 106.0527 | high | numel(input) |  | Raw alpha=106.3529, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.truncnorm.pdf` | 16.0000 | 28.0000 | high | numel(input) |  | Raw alpha=28.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.truncnorm.cdf` | 16.0000 | 50.5992 | low | numel(input) |  | Raw alpha=50.8994, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.truncnorm.ppf` | 16.0000 | 82.5206 | high | numel(input) |  | Raw alpha=82.8208, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.cauchy.pdf` | 4.0000 | 6.0000 | high | numel(input) |  | Raw alpha=6.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.uniform.pdf` | 1.0000 | 0.0000 | low | numel(input) |  | Raw alpha=0.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.uniform.cdf` | 1.0000 | 4.0000 | high | numel(input) |  | Raw alpha=4.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |
| `stats.uniform.ppf` | 1.0000 | 2.0000 | high | numel(input) |  | Raw alpha=2.3002, setup overhead=0.3002 subtracted. FP instruction count per element. |

### Misc (152 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `fromiter` | 16.0000 | 20.3777 |  | numel(output) |  | EC2 timing = 20.38. Python iterator overhead per element. |
| `apply_along_axis` | 4.0000 | 1.9490 |  | numel(output) |  | EC2 timing = 1.95. Note: cost formula (result.size) underestimates — actual work is input.size. |
| `argwhere` | 4.0000 | 1.9631 |  | numel(input) |  | EC2 timing ratio vs add = 1.9631. |
| `bmat` | 4.0000 | 3.0046 |  | numel(output) |  | EC2 timing = 3.00. Block matrix assembly from nested list. |
| `choose` | 4.0000 | 8.0516 |  | numel(output) |  | EC2 timing ratio vs add = 8.0516. |
| `compress` | 4.0000 | 2.4640 |  | numel(input) |  | EC2 timing ratio vs add = 2.4640. |
| `diag` | 4.0000 | 0.3780 |  | numel(output) when 1D->2D, min(m,n) when 2D->1D |  | EC2 timing = 0.38 (1D->2D). Formula fixed to numel(output) for construction. |
| `extract` | 4.0000 | 2.4549 |  | numel(input) |  | EC2 timing ratio vs add = 2.4549. |
| `fill_diagonal` | 4.0000 | 6.2633 |  | min(m,n) |  | EC2 timing = 6.26 per diagonal element. Strided cache misses in large matrices. |
| `insert` | 4.0000 | 1.6005 |  | numel(values) |  | EC2 timing ratio vs add = 1.6005. |
| `mask_indices` | 4.0000 | 4.5645 |  | numel(output) |  | EC2 timing ratio vs add = 4.5645. |
| `piecewise` | 4.0000 | 13.7055 |  | numel(input) |  | EC2 timing ratio vs add = 13.7055. |
| `place` | 4.0000 | 3.7818 |  | numel(input) |  | EC2 timing ratio vs add = 3.7818. |
| `putmask` | 4.0000 | 3.0306 |  | numel(input) |  | EC2 timing ratio vs add = 3.0306. |
| `select` | 4.0000 | 7.9267 |  | numel(input) |  | EC2 timing ratio vs add = 7.9267. |
| `take` | 4.0000 | 3.8564 |  | numel(output) |  | EC2 timing ratio vs add = 3.8564. |
| `trim_zeros` | 4.0000 | 2.3378 |  | num trimmed |  | EC2 timing ratio vs add = 2.3378. |
| `where` | 4.0000 | 3.5686 |  | numel(input) |  | EC2 timing ratio vs add = 3.5686. |
| `allclose` | 1.0000 | 3.7001 | high | n | [\_counting\_ops.py:45](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_counting_ops.py#L45) | Element-wise tolerance check; cost = numel(a). |
| `array_equal` | 1.0000 | 0.6001 | low | n | [\_counting\_ops.py:60](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_counting_ops.py#L60) | Element-wise equality; cost = numel(a). |
| `array_equiv` | 1.0000 | 0.6001 | low | n | [\_counting\_ops.py:82](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_counting_ops.py#L82) | Element-wise equivalence; cost = numel(a). |
| `clip` | 1.0000 | 2.0000 | high | numel(output) | [\_pointwise.py:505](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L505) | Clip array to [a_min, a_max] element-wise. |
| `diff` | 1.0000 | 1.3001 | medium | n | [\_pointwise.py:791](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L791) | n-th discrete difference along axis. |
| `ediff1d` | 1.0000 | 1.3001 | medium | n | [\_pointwise.py:821](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L821) | Differences between consecutive elements. |
| `gradient` | 1.0000 | 2.3001 | high | n | [\_pointwise.py:807](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L807) | Gradient using central differences. |
| `unwrap` | 1.0000 | 6.7693 | medium | n | [\_unwrap.py:40](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_unwrap.py#L40) | Phase unwrap. Cost: $\text{numel}(\text{input})$ (diff + conditional adjustment). |
| `convolve` | 1.0000 | 2.0003 | high | n * k | [\_pointwise.py:841](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L841) | 1-D discrete convolution. |
| `correlate` | 1.0000 | 2.0003 | high | n * k | [\_pointwise.py:861](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L861) | 1-D cross-correlation. |
| `corrcoef` | 1.0000 | 1.0014 | high | 2 * f^2 * s | [\_pointwise.py:898](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L898) | Pearson correlation coefficients. |
| `cov` | 1.0000 | 1.0012 | high | 2 * f^2 * s | [\_pointwise.py:913](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L913) | Covariance matrix. |
| `cross` | 1.0000 | 1.8002 | medium | 6 * n | [\_\_init\_\_.py:72](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/linalg/__init__.py#L72) | Cross product of two 3-D vectors. |
| `histogram` | 1.0000 | 0.7574 | high | n * ceil(log2(bins)) | [\_counting\_ops.py:106](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_counting_ops.py#L106) | Binning; cost = n*ceil(log2(bins)). |
| `histogram2d` | 1.0000 | 0.3289 | high | n * 2 * ceil(log2(bins)) | [\_counting\_ops.py:148](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_counting_ops.py#L148) | 2D binning; cost = n*(ceil(log2(bx))+ceil(log2(by))). |
| `histogramdd` | 1.0000 | 0.3834 | high | n * ndim * ceil(log2(bins)) | [\_counting\_ops.py:189](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_counting_ops.py#L189) | ND binning; cost = n*sum(ceil(log2(b_i))). |
| `histogram_bin_edges` | 1.0000 | 2.3021 | high | n | [\_counting\_ops.py:207](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_counting_ops.py#L207) | Bin edge computation; cost = numel(a). |
| `digitize` | 1.0000 | 0.0429 | low | n * ceil(log2(bins)) | [\_sorting\_ops.py:197](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_sorting_ops.py#L197) | Bin search; cost = n*ceil(log2(bins)). |
| `bincount` | 1.0000 | 0.0001 | high | n | [\_counting\_ops.py:221](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_counting_ops.py#L221) | Integer counting; cost = numel(x). |
| `interp` | 1.0000 | 0.2364 | high | n * ceil(log2(xp)) | [\_pointwise.py:955](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L955) | 1-D linear interpolation. |
| `trace` | 1.0000 | 0.7872 | high | min(m, n) | [\_counting\_ops.py:26](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_counting_ops.py#L26) | Re-benchmarked with np.ones setup to avoid random-gen overhead. Trace is a sum of diagonal — mode=ufunc_reduction. |
| `trapezoid` | 1.0000 | 4.3001 | high | n | [\_pointwise.py:926](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L926) | Integrate using the trapezoidal rule. |
| `logspace` | 1.0000 | 75.0001 | high | n | [\_counting\_ops.py:236](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_counting_ops.py#L236) | Log-spaced generation; cost = num. |
| `geomspace` | 1.0000 | 76.0001 | high | n | [\_counting\_ops.py:246](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_counting_ops.py#L246) | Geometric-spaced generation; cost = num. |
| `vander` | 1.0000 | 0.9939 | high | n * (degree - 1) | [\_counting\_ops.py:260](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_counting_ops.py#L260) | Vandermonde matrix; cost = len(x)*(N-1). |
| `isnan` | 1.0000 | 0.3375 | high | numel(output) |  | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.3375 |
| `isinf` | 1.0000 | 0.3493 | high | numel(output) |  | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.3493 |
| `isfinite` | 1.0000 | 0.3454 | high | numel(output) |  | Timing-based (fp_arith_inst_retired blind to this op). Ratio vs add=0.3454 |
| `append` | 1.0000 | 0.9206 |  | numel(values) |  | EC2 timing ratio vs add = 0.9206. |
| `apply_over_axes` | 1.0000 | 0.1407 |  | numel(output) |  | EC2 timing = 0.14. Efficient numpy reduction path. |
| `arange` | 1.0000 | 0.6141 |  | numel(output) |  | EC2 timing ratio vs add = 0.6141. |
| `array` | 1.0000 | 0.9035 |  | numel(input) |  | Timing ratio vs add = 0.9035. Benchmark: 10,000,000 elements. |
| `asarray_chkfinite` | 1.0000 | 0.2892 |  | numel(input) |  | EC2 timing ratio vs add = 0.2892. |
| `block` | 1.0000 | 0.0167 |  | numel(output) |  | Formerly-free op. Timing ratio vs add = 0.0167. |
| `concat` | 1.0000 | 0.9209 |  | numel(output) |  | EC2 timing = 0.92. Memcpy (alias for concatenate). |
| `concatenate` | 1.0000 | 0.9106 |  | numel(output) |  | EC2 timing ratio vs add = 0.9106. |
| `copyto` | 1.0000 | 1.1734 |  | numel(output) |  | EC2 timing ratio vs add = 1.1734. |
| `delete` | 1.0000 | 1.1182 |  | num deleted |  | EC2 timing ratio vs add = 1.1182. |
| `diagflat` | 1.0000 | 0.2349 |  | numel(output) |  | EC2 timing = 0.23. Formula fixed from len(v) to numel(output). |
| `diagonal` | 1.0000 | 0.5792 |  | numel(input) |  | EC2 timing ratio vs add = 0.5792. |
| `dstack` | 1.0000 | 1.1037 |  | numel(output) |  | EC2 timing ratio vs add = 1.1037. |
| `einsum_path` | 1.0000 | 1.0000 |  | 1 |  | Weight=1 by design. Path planning only — intentionally charges 1 FLOP for budget tracking. |
| `flatnonzero` | 1.0000 | 0.8717 |  | numel(input) |  | EC2 timing ratio vs add = 0.8717. |
| `fromfunction` | 1.0000 | 0.7226 |  | numel(output) |  | EC2 timing ratio vs add = 0.7226. |
| `full` | 1.0000 | 0.5706 |  | numel(output) |  | EC2 timing ratio vs add = 0.5706. |
| `full_like` | 1.0000 | 0.5616 |  | numel(output) |  | EC2 timing ratio vs add = 0.5616. |
| `indices` | 1.0000 | 0.1694 |  | numel(output) |  | EC2 timing ratio vs add = 0.1694. |
| `ix_` | 1.0000 | 1.2920 |  | numel(output) |  | EC2 timing ratio vs add = 1.2920. |
| `linspace` | 1.0000 | 1.1761 |  | numel(output) |  | EC2 timing ratio vs add = 1.1761. |
| `meshgrid` | 1.0000 | 0.1446 |  | numel(output) |  | EC2 timing ratio vs add = 0.1446. |
| `nonzero` | 1.0000 | 0.8322 |  | numel(input) |  | EC2 timing ratio vs add = 0.8322. |
| `packbits` | 1.0000 | 0.3987 |  | numel(input) |  | EC2 timing ratio vs add = 0.3987. |
| `pad` | 1.0000 | 0.9146 |  | numel(output) |  | EC2 timing ratio vs add = 0.9146. |
| `put` | 1.0000 | 0.9866 |  | numel(input) |  | EC2 timing ratio vs add = 0.9866. |
| `put_along_axis` | 1.0000 | 0.0310 |  | numel(input) |  | Formerly-free op. Timing ratio vs add = 0.0310. |
| `repeat` | 1.0000 | 0.5985 |  | numel(output) |  | EC2 timing ratio vs add = 0.5985. |
| `resize` | 1.0000 | 0.6234 |  | numel(output) |  | EC2 timing ratio vs add = 0.6234. |
| `roll` | 1.0000 | 0.9077 |  | numel(output) |  | EC2 timing ratio vs add = 0.9077. |
| `stack` | 1.0000 | 0.9017 |  | numel(output) |  | EC2 timing ratio vs add = 0.9017. |
| `take_along_axis` | 1.0000 | 0.4811 |  | numel(output) |  | Formerly-free op. Timing ratio vs add = 0.4811. |
| `tile` | 1.0000 | 0.6244 |  | numel(output) |  | EC2 timing ratio vs add = 0.6244. |
| `unpackbits` | 1.0000 | 0.0385 |  | numel(input) |  | EC2 timing ratio vs add = 0.0385. |
| `unstack` | 1.0000 | 0.0007 |  | numel(input) |  | EC2 timing = 0.0007. Returns views — should be free. |
| `vstack` | 1.0000 | 0.9082 |  | numel(output) |  | EC2 timing ratio vs add = 0.9082. |
| `array_split` | 0.0000 | 0.0000 |  |  |  |  |
| `asarray` | 0.0000 | 0.0000 |  |  |  |  |
| `astype` | 0.0000 | 0.0000 |  |  |  |  |
| `atleast_1d` | 0.0000 | 0.0000 |  |  |  |  |
| `atleast_2d` | 0.0000 | 0.0000 |  |  |  |  |
| `atleast_3d` | 0.0000 | 0.0000 |  |  |  |  |
| `broadcast_arrays` | 0.0000 | 0.0000 |  |  |  |  |
| `broadcast_shapes` | 0.0000 | 0.0000 |  |  |  |  |
| `broadcast_to` | 0.0000 | 0.0000 |  |  |  |  |
| `can_cast` | 0.0000 | 0.0000 |  |  |  |  |
| `column_stack` | 0.0000 | 0.0000 |  |  |  |  |
| `common_type` | 0.0000 | 0.0000 |  |  |  |  |
| `copy` | 0.0000 | 0.0000 |  |  |  |  |
| `diag_indices` | 0.0000 | 0.0000 |  |  |  |  |
| `diag_indices_from` | 0.0000 | 0.0000 |  |  |  |  |
| `dsplit` | 0.0000 | 0.0000 |  |  |  |  |
| `empty` | 0.0000 | 0.0000 |  |  |  |  |
| `empty_like` | 0.0000 | 0.0000 |  |  |  |  |
| `expand_dims` | 0.0000 | 0.0000 |  |  |  |  |
| `eye` | 0.0000 | 0.0000 |  |  |  |  |
| `flip` | 0.0000 | 0.0000 |  |  |  |  |
| `fliplr` | 0.0000 | 0.0000 |  |  |  |  |
| `flipud` | 0.0000 | 0.0000 |  |  |  |  |
| `from_dlpack` | 0.0000 | 0.0000 |  |  |  |  |
| `frombuffer` | 0.0000 | 0.0000 |  |  |  |  |
| `hsplit` | 0.0000 | 0.0000 |  |  |  |  |
| `hstack` | 0.0000 | 0.0000 |  |  |  |  |
| `identity` | 0.0000 | 0.0000 |  |  |  |  |
| `isdtype` | 0.0000 | 0.0000 |  |  |  |  |
| `isfortran` | 0.0000 | 0.0000 |  |  |  |  |
| `isscalar` | 0.0000 | 0.0000 |  |  |  |  |
| `issubdtype` | 0.0000 | 0.0000 |  |  |  |  |
| `iterable` | 0.0000 | 0.0000 |  |  |  |  |
| `matrix_transpose` | 0.0000 | 0.0000 |  |  |  |  |
| `may_share_memory` | 0.0000 | 0.0000 |  |  |  |  |
| `min_scalar_type` | 0.0000 | 0.0000 |  |  |  |  |
| `mintypecode` | 0.0000 | 0.0000 |  |  |  |  |
| `moveaxis` | 0.0000 | 0.0000 |  |  |  |  |
| `ndim` | 0.0000 | 0.0000 |  |  |  |  |
| `ones` | 0.0000 | 0.0000 |  |  |  |  |
| `ones_like` | 0.0000 | 0.0000 |  |  |  |  |
| `permute_dims` | 0.0000 | 0.0000 |  |  |  |  |
| `promote_types` | 0.0000 | 0.0000 |  |  |  |  |
| `ravel` | 0.0000 | 0.0000 |  |  |  |  |
| `ravel_multi_index` | 0.0000 | 0.0000 |  |  |  |  |
| `require` | 0.0000 | 0.0000 |  |  |  |  |
| `reshape` | 0.0000 | 0.0000 |  |  |  |  |
| `result_type` | 0.0000 | 0.0000 |  |  |  |  |
| `rollaxis` | 0.0000 | 0.0000 |  |  |  |  |
| `rot90` | 0.0000 | 0.0000 |  |  |  |  |
| `row_stack` | 0.0000 | 0.0000 |  |  |  |  |
| `shape` | 0.0000 | 0.0000 |  |  |  |  |
| `shares_memory` | 0.0000 | 0.0000 |  |  |  |  |
| `size` | 0.0000 | 0.0000 |  |  |  |  |
| `split` | 0.0000 | 0.0000 |  |  |  |  |
| `squeeze` | 0.0000 | 0.0000 |  |  |  |  |
| `swapaxes` | 0.0000 | 0.0000 |  |  |  |  |
| `transpose` | 0.0000 | 0.0000 |  |  |  |  |
| `tri` | 0.0000 | 0.0000 |  |  |  |  |
| `tril` | 0.0000 | 0.0000 |  |  |  |  |
| `tril_indices` | 0.0000 | 0.0000 |  |  |  |  |
| `tril_indices_from` | 0.0000 | 0.0000 |  |  |  |  |
| `triu` | 0.0000 | 0.0000 |  |  |  |  |
| `triu_indices` | 0.0000 | 0.0000 |  |  |  |  |
| `triu_indices_from` | 0.0000 | 0.0000 |  |  |  |  |
| `typename` | 0.0000 | 0.0000 |  |  |  |  |
| `unravel_index` | 0.0000 | 0.0000 |  |  |  |  |
| `vsplit` | 0.0000 | 0.0000 |  |  |  |  |
| `zeros` | 0.0000 | 0.0000 |  |  |  |  |
| `zeros_like` | 0.0000 | 0.0000 |  |  |  |  |

### Window (5 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `blackman` | 16.0000 | 24.2347 | high | 3*n | [\_window.py:65](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_window.py#L65) | Blackman window. Cost: 3*n (three cosine terms per sample). |
| `kaiser` | 16.0000 | 37.4439 | high | 3*n | [\_window.py:155](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_window.py#L155) | Kaiser window. Cost: 3*n (Bessel function eval per sample). |
| `hamming` | 8.0000 | 34.3767 | high | n | [\_window.py:95](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_window.py#L95) | Hamming window. Cost: n (one cosine per sample). |
| `hanning` | 8.0000 | 34.3767 | high | n | [\_window.py:125](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_window.py#L125) | Hanning window. Cost: n (one cosine per sample). |
| `bartlett` | 1.0000 | 6.0001 | high | n | [\_window.py:35](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_window.py#L35) | Bartlett window. Cost: n (one linear eval per sample). |

### Bitwise (13 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `gcd` | 16.0000 | 99.0872 | high | n | [\_pointwise.py:445](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L445) | Element-wise greatest common divisor. |
| `lcm` | 16.0000 | 104.1010 | high | n | [\_pointwise.py:450](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L450) | Element-wise least common multiple. |
| `bitwise_not` | 1.0000 | 8.2180 | high | n | [\_pointwise.py:306](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L306) | Element-wise bitwise NOT. |
| `bitwise_invert` | 1.0000 | 8.2113 | high | n | [\_pointwise.py:305](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L305) | Element-wise bitwise invert (alias for bitwise_not). |
| `bitwise_count` | 1.0000 | 15.3460 | high | n | [\_pointwise.py:304](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L304) | Count set bits element-wise (popcount). |
| `invert` | 1.0000 | 8.1990 | high | n | [\_pointwise.py:319](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L319) | Bitwise NOT element-wise. |
| `bitwise_and` | 1.0000 | 11.5166 | high | n | [\_pointwise.py:433](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L433) | Element-wise bitwise AND. |
| `bitwise_or` | 1.0000 | 11.5722 | high | n | [\_pointwise.py:435](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L435) | Element-wise bitwise OR. |
| `bitwise_xor` | 1.0000 | 11.5398 | high | n | [\_pointwise.py:437](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L437) | Element-wise bitwise XOR. |
| `bitwise_left_shift` | 1.0000 | 12.8120 | high | n | [\_pointwise.py:434](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L434) | Element-wise left bit shift. |
| `bitwise_right_shift` | 1.0000 | 18.7846 | high | n | [\_pointwise.py:436](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L436) | Element-wise right bit shift. |
| `left_shift` | 1.0000 | 12.7592 | high | n | [\_pointwise.py:452](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L452) | Element-wise left bit shift (legacy name). |
| `right_shift` | 1.0000 | 18.7921 | high | n | [\_pointwise.py:464](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L464) | Element-wise right bit shift (legacy name). |

### Complex (11 operations)

| Op | Active Weight | Empirical Weight | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `angle` | 16.0000 | 53.7818 | high | numel(output) | [\_pointwise.py:267](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L267) | Return angle of complex argument element-wise. |
| `conj` | 1.0000 | 0.7818 | high | numel(output) | [\_pointwise.py:308](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L308) | Complex conjugate element-wise. |
| `conjugate` | 1.0000 | 0.7818 | high | numel(output) | [\_pointwise.py:309](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L309) | Complex conjugate element-wise. |
| `imag` | 1.0000 | 0.7818 | high | numel(output) | [\_pointwise.py:318](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L318) | Return imaginary part of complex array. |
| `real` | 1.0000 | 0.7818 | high | numel(output) | [\_pointwise.py:333](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L333) | Return real part of complex array. |
| `real_if_close` | 1.0000 | 0.7818 | low | numel(output) | [\_pointwise.py:334](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L334) | Return real array if imaginary part is negligible. |
| `iscomplex` | 1.0000 | 0.7818 | high | numel(output) | [\_pointwise.py:320](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L320) | Test if element is complex element-wise. |
| `isreal` | 1.0000 | 0.7818 | high | numel(output) | [\_pointwise.py:325](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L325) | Test if element is real (imag == 0) element-wise. |
| `sort_complex` | 1.0000 | 0.7830 | high | n * ceil(log2(n)) | [\_pointwise.py:376](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L376) | Sort complex array. Cost: $n \cdot \lceil\log_2 n\rceil$. |
| `iscomplexobj` | 1.0000 | 18.3055 | high | numel(output) | [\_pointwise.py:321](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L321) | Return True if input is a complex type or array. |
| `isrealobj` | 1.0000 | 18.3070 | high | numel(output) | [\_pointwise.py:326](https://github.com/AIcrowd/flopscope/blob/main/src/flopscope/_pointwise.py#L326) | Return True if x is a not complex type or array. |

## Summary by category

| Category | Count | Avg Weight | Min | Max |
|:---------|------:|-----------:|----:|----:|
| Pointwise Unary | 47 | 8.02 | 1.0000 | 16.0000 |
| Pointwise Binary | 34 | 5.41 | 1.0000 | 16.0000 |
| Reductions | 35 | 1.11 | 1.0000 | 2.0000 |
| Sorting | 17 | 1.00 | 1.0000 | 1.0000 |
| FFT | 18 | 0.78 | 0.0000 | 1.0000 |
| Linalg | 14 | 4.00 | 4.0000 | 4.0000 |
| Linalg Delegates | 17 | 1.59 | 0.0000 | 4.0000 |
| Contractions | 9 | 1.00 | 1.0000 | 1.0000 |
| Polynomial | 10 | 2.50 | 1.0000 | 16.0000 |
| Random | 49 | 11.63 | 0.0000 | 16.0000 |
| Stats | 24 | 13.62 | 1.0000 | 16.0000 |
| Misc | 152 | 0.97 | 0.0000 | 16.0000 |
| Window | 5 | 9.80 | 1.0000 | 16.0000 |
| Bitwise | 13 | 3.31 | 1.0000 | 16.0000 |
| Complex | 11 | 2.36 | 1.0000 | 16.0000 |

**Total benchmarked operations:** 455

## Validation

Every operation is measured in both **perf mode** (hardware counters) and
**timing mode** (wall-clock nanoseconds).

### Correlation statistics

*scipy not available — correlation stats skipped*

### Maximum divergence

| Field | Value |
|:------|:------|
| Operation | `imag` |
| Perf weight | 1.0 |
| Timing weight | 0.0 |
| Ratio | inf |

### Interpreting divergence

The moderate correlation values and large max divergence for BLAS operations are
**expected**. Perf mode counts FP instructions regardless of execution time,
while timing mode measures wall-clock time including memory bandwidth and cache
effects. BLAS operations achieve near-peak FLOP throughput, so their per-instruction
timing is much lower than for scalar pointwise operations. For pointwise ops
(which dominate the count), the two modes agree well in relative ordering.

**Correlation caveats:**
The Pearson and Spearman values span all operations, including BLAS/linalg
ops where timing and perf divergence is structurally expected. For the
subset of pointwise operations, both correlations are substantially higher.

## Known limitations

### BLAS vectorization effects

Operations backed by optimized BLAS routines (`matmul`, `dot`, contraction ops)
show weights near 1.0 because flopscope's analytical FLOP count and the perf
instruction counter both treat each FMA as 2 retired ops.

### Random number generators

RNG weights vary dramatically (0.0001 to 367) because the analytical formula
(`numel(output)`) captures only the output size, not the internal algorithmic
complexity. Complex distributions like `hypergeometric` involve rejection
sampling loops that execute many FP instructions per output element.

## Related pages

- [How to calibrate weights](../how-to/calibrate-weights.md)
- [FLOP counting model](../concepts/flop-counting-model.md)
- [Operation audit](operation-audit.md)
- [Agent cheat sheet](for-agents.md)
