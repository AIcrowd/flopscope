# Truncated-normal numerical cost bound

This is an analytical **upper bound for numerical work**, not a hardware calibration or a literal count of NumPy memory operations. It applies to `src/flopscope/stats/_truncnorm_kernels.py`, with four fixed Newton steps, eight fixed quadrature nodes, and import-time scalar thresholds.

Conventions come from `docs/reference/cost-model.md`: FMA = 2; arithmetic, square root, comparison and conditional selection = 1; exp/expm1/log/log1p/logaddexp = 16; a fixed square = one multiplication; each output assignment/fill = 1. A zero/uninitialized allocation and indexing/view mechanics are not numerical operations. Input coercion and Python dispatch are excluded, as in the stats composite convention. All comparison masks are counted even when a branch is empty. Each `any` reduction is bounded by one operation per input element. This is deliberately conservative; mutually exclusive arithmetic branches use their maximum, and eager expressions (including both `where` values) are both counted.

All constants below are per element entering the named helper, before the public float64 dtype multiplier. Write costs in common paths may conservatively cover incompatible endpoint cases simultaneously.

## Helper bounds

| Helper | Derivation | Bound |
|---|---|---:|
| `_erf`, restricted to abs(x) < 1.25 | masks/reductions/sign/fill 19 + max(first-region arithmetic/write 23, second-region arithmetic/write 29) | 48 |
| `_erf`, unrestricted | same common 19 + max tail arithmetic/write 57 | 76 |
| `_ndtri`, unrestricted | masks, endpoint writes, reductions 25 + max rational seed 40 + Newton correction 104 | 169 |
| `_narrow` | two finite predicates, two logical conjunctions, subtraction, two absolutes, two maxima, multiplication, comparison | 11 |
| `_local_integral` | 41 multiplications + one negation + eight subtractions + eight exp at 16 + seven reduction additions | 185 |
| `log_ndtr`, nonpositive input | common masks/normalization/writes 20 + max(moderate branch 68, first rational tail 60, second rational tail 56) | 88 |
| `log_ndtr`, arbitrary input | nonpositive bound 88 + optional exp/negation/log1p/write 34 | 122 |
| `log_mass` | common predicates/fills 26 + max(narrow 207, reflected same-tail 218, central 228) | 254 |
| `ndtri_log_lower` | common 10 + max(middle seed 186, tail seed 24) + 4 Newton steps of 146 | 780 |

`log_mass` only calls `log_ndtr` on nonpositive arguments: its same-tail branch reflects positive bounds and its central branch uses `a < 0` and `-b < 0`. The restricted erf bound applies inside `log_ndtr`'s moderate branch because it explicitly filters abs(x/sqrt(2)) < 1.25. The general inverse's Newton step conservatively uses the unrestricted `log_ndtr` bound, even though ordinary iterates are nonpositive.

The inverse-normal Newton sub-bound 104 is: CDF 79 (= division 1 + unrestricted erf 76 + add/multiply 2), PDF 19, safety comparison 1, eager subtract/divide/select 3, final subtraction/write 2. The log-domain Newton sub-bound 146 is: log CDF 122, derivative arithmetic/exp 20, correction subtract/divide/subtract/write 4.

## Public numerical kernels

| Method | Common | Narrow branch | Regular branch | Bound |
|---|---:|---:|---:|---:|
| PDF | 39 | 211 | 276 | **315** |
| CDF | 44 | 374 | 800 | **844** |
| PPF | 49 | 1037 | 1343 | **1392** |

PDF's regular branch is one log mass (254), square/multiply/two subtractions (4), exp (16), scale division (1), and write (1). Its narrow branch uses a 185-cost integral, 21-cost exponential density factor, width/offset subtractions (2), two divisions and write.

CDF's regular branch is three interval log masses (762), two subtractions, and 36 for the final comparison, both eager exp/expm1 values, negation, selection and write. Its narrow branch is two integrals (370), two offset subtractions, division and write.

PPF's regular branch is log(q)/log1p(-q) (33), four log CDFs (488), two logaddexp combinations including their additions (36), side comparison (1), the one selected inverse branch including optional sign and write (782), two full-subset inversions of the right-side mask (2), and output write (1). Its narrow branch has width and initial-point arithmetic (2), one integral (185), four steps of 212, then final offset addition/write (2). A narrow Newton step has integral (185), multiply/subtract (2), exponential derivative (21), division/subtraction (2), and clip (2).

The common costs cover input-domain predicates (12), method masks and endpoint writes, and applicable normalization/clipping/affine operations. Empty inputs retain the existing wrapper minimum of one billed element. The total charges are therefore 315, 844, and 1392 times `max(numel(broadcast(x,a,b,loc,scale)),1)`, subsequently multiplied by the configured float64 dtype rate and stats weight. With packaged defaults those factors are 2 and 1.

## Numerical scope

The implementation retains normal tail probabilities in logarithmic form and
uses a factored local integral for nearly coincident bounds. Regression checks
cover positive and negative 9- and 40-sigma intervals, semi-infinite intervals,
central intervals, and adjacent representable bounds. The log-CDF helper is
checked on a grid through absolute standardized arguments of 100. Narrow-interval
checks use independent adaptive quadrature because subtracting two rounded CDF
values is not an adequate reference there.

This is not an arbitrary-finite-double accuracy guarantee. At extremely large
standardized bounds, squaring and subtracting log probabilities can still lose
precision or overflow; an interval near 1e10 can still produce a NaN quantile.
For very narrow intervals, the quantile's representable spacing can itself
prevent exact inversion. Correct domain handling returns NaN for invalid
probabilities, nonpositive scales and invalid bounds. The numerical cost stays
fixed across these cases, so input values cannot select a cheaper billed path.
