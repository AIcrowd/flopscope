"""Registry of all public numpy 2.x callables with FLOP-counting categories.

Categories
----------
counted_unary           scalar math on each element, cost = numel(output)
counted_binary          element-wise binary op, cost = numel(output)
counted_reduction       reduce array, cost = numel(input)
counted_custom          bespoke cost formulas
counted_random_method   method on Generator/RandomState, counted via cost_formula
free                    zero FLOP cost (allocation, indexing, shape ops, etc.)
free_random_method      method on Generator/RandomState, no FLOP cost (state, spawn, etc.)
blacklisted             intentionally unsupported
"""

from __future__ import annotations

REGISTRY_META: dict = {
    "numpy_version": "2.4.4",
    "numpy_supported": ">=2.0.0,<2.5.0",
    "last_updated": "2026-04-17",
}

# ---------------------------------------------------------------------------
# Full registry — every entry has:  category, module, notes
# ---------------------------------------------------------------------------
REGISTRY: dict[str, dict] = {
    # ------------------------------------------------------------------
    # counted_unary — implemented in _pointwise.py
    # ------------------------------------------------------------------
    "abs": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise absolute value; alias for absolute.",
    },
    "absolute": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise absolute value.",
    },
    "negative": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise negation.",
    },
    "positive": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise unary plus (copy with sign preserved).",
    },
    "exp": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise e^x.",
    },
    "exp2": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise 2^x.",
    },
    "expm1": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise e^x - 1 (accurate near zero).",
    },
    "log": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise natural logarithm.",
    },
    "log2": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise base-2 logarithm.",
    },
    "log10": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise base-10 logarithm.",
    },
    "log1p": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise log(1+x) (accurate near zero).",
    },
    "sqrt": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise square root.",
    },
    "cbrt": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise cube root.",
    },
    "square": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise x^2.",
    },
    "reciprocal": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise 1/x.",
    },
    "sin": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise sine.",
    },
    "cos": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise cosine.",
    },
    "tan": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise tangent.",
    },
    "arcsin": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise inverse sine.",
    },
    "arccos": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise inverse cosine.",
    },
    "arctan": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise inverse tangent.",
    },
    "sinh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise hyperbolic sine.",
    },
    "cosh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise hyperbolic cosine.",
    },
    "tanh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise hyperbolic tangent.",
    },
    "arcsinh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise inverse hyperbolic sine.",
    },
    "arccosh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise inverse hyperbolic cosine.",
    },
    "arctanh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise inverse hyperbolic tangent.",
    },
    # asin/acos/atan/asinh/acosh/atanh — NumPy 2.x aliases
    "asin": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for arcsin (NumPy 2.x).",
    },
    "acos": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for arccos (NumPy 2.x).",
    },
    "atan": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for arctan (NumPy 2.x).",
    },
    "asinh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for arcsinh (NumPy 2.x).",
    },
    "acosh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for arccosh (NumPy 2.x).",
    },
    "atanh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for arctanh (NumPy 2.x).",
    },
    "degrees": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Convert radians to degrees element-wise.",
    },
    "radians": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Convert degrees to radians element-wise.",
    },
    "deg2rad": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for radians.",
    },
    "rad2deg": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for degrees.",
    },
    "sign": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise sign function.",
    },
    "signbit": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Returns True for elements with negative sign bit.",
    },
    "fabs": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise absolute value (always float).",
    },
    "ceil": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise ceiling.",
    },
    "floor": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise floor.",
    },
    "rint": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Round to nearest integer element-wise.",
    },
    "trunc": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Truncate toward zero element-wise.",
    },
    "fix": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Round toward zero element-wise (alias for trunc).",
    },
    "round": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Round to given number of decimals element-wise.",
    },
    "around": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for round.",
    },
    "spacing": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return ULP spacing for each element.",
    },
    "logical_not": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise logical NOT.",
    },
    "bitwise_not": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise bitwise NOT.",
    },
    "bitwise_invert": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise bitwise invert (alias for bitwise_not).",
    },
    "bitwise_count": {
        "category": "counted_unary",
        "module": "numpy",
        "min_numpy": "2.1",
        "notes": "Count set bits element-wise (popcount).",
    },
    "invert": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Bitwise NOT element-wise.",
    },
    "conj": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Complex conjugate element-wise.",
    },
    "conjugate": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Complex conjugate element-wise.",
    },
    "angle": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return angle of complex argument element-wise.",
    },
    "real": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return real part of complex array.",
    },
    "imag": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return imaginary part of complex array.",
    },
    "sinc": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Normalized sinc function element-wise.",
    },
    "i0": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Modified Bessel function of order 0, element-wise.",
    },
    "nan_to_num": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Replace NaN/inf with finite numbers element-wise.",
    },
    "modf": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return fractional and integral parts element-wise.",
    },
    "frexp": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Decompose x into mantissa and exponent element-wise.",
    },
    "isclose": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise approximate equality test.",
    },
    "isnat": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Blacklisted per reviewer — datetime ops not in scope.",
    },
    "isneginf": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Test for negative infinity element-wise.",
    },
    "isposinf": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Test for positive infinity element-wise.",
    },
    "isreal": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Test if element is real (imag == 0) element-wise.",
    },
    "iscomplex": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Test if element is complex element-wise.",
    },
    "iscomplexobj": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return True if input is a complex type or array.",
    },
    "isrealobj": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return True if x is a not complex type or array.",
    },
    "real_if_close": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return real array if imaginary part is negligible.",
    },
    "sort_complex": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Sort complex array. Cost: $n \\cdot \\lceil\\log_2 n\\rceil$.",
    },
    # ------------------------------------------------------------------
    # counted_binary — implemented in _pointwise.py
    # ------------------------------------------------------------------
    "add": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise addition.",
    },
    "subtract": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise subtraction.",
    },
    "multiply": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise multiplication.",
    },
    "divide": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise true division.",
    },
    "true_divide": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise true division (explicit).",
    },
    "floor_divide": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise floor division.",
    },
    "power": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise exponentiation x**y.",
    },
    "pow": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Alias for power (NumPy 2.x).",
    },
    "float_power": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise exponentiation in float64.",
    },
    "mod": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise modulo.",
    },
    "remainder": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise remainder (same as mod).",
    },
    "fmod": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise C-style fmod (remainder toward zero).",
    },
    "maximum": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise maximum (propagates NaN).",
    },
    "minimum": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise minimum (propagates NaN).",
    },
    "fmax": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise maximum ignoring NaN.",
    },
    "fmin": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise minimum ignoring NaN.",
    },
    "logaddexp": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "log(exp(x1) + exp(x2)) element-wise.",
    },
    "logaddexp2": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "log2(2**x1 + 2**x2) element-wise.",
    },
    "arctan2": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise arctan(y/x) considering quadrant.",
    },
    "atan2": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Alias for arctan2 (NumPy 2.x).",
    },
    "hypot": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise Euclidean norm sqrt(x1^2 + x2^2).",
    },
    "copysign": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Copy sign of x2 to magnitude of x1 element-wise.",
    },
    "nextafter": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Return next float after x1 toward x2 element-wise.",
    },
    "ldexp": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Return x1 * 2**x2 element-wise.",
    },
    "heaviside": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Heaviside step function element-wise.",
    },
    "greater": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise x1 > x2.",
    },
    "greater_equal": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise x1 >= x2.",
    },
    "less": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise x1 < x2.",
    },
    "less_equal": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise x1 <= x2.",
    },
    "equal": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise x1 == x2.",
    },
    "not_equal": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise x1 != x2.",
    },
    "logical_and": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise logical AND.",
    },
    "logical_or": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise logical OR.",
    },
    "logical_xor": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise logical XOR.",
    },
    "bitwise_and": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise bitwise AND.",
    },
    "bitwise_or": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise bitwise OR.",
    },
    "bitwise_xor": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise bitwise XOR.",
    },
    "bitwise_left_shift": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise left bit shift.",
    },
    "bitwise_right_shift": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise right bit shift.",
    },
    "left_shift": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise left bit shift (legacy name).",
    },
    "right_shift": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise right bit shift (legacy name).",
    },
    "gcd": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise greatest common divisor.",
    },
    "lcm": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise least common multiple.",
    },
    "divmod": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise (quotient, remainder) tuple.",
    },
    "vecdot": {
        "category": "counted_binary",
        "module": "numpy",
        "min_numpy": "2.1",
        "notes": "Vector dot product along last axis.",
    },
    "matvec": {
        "category": "counted_binary",
        "module": "numpy",
        "min_numpy": "2.2",
        "notes": "Matrix-vector product. Cost = output_size * contracted_axis.",
    },
    "vecmat": {
        "category": "counted_binary",
        "module": "numpy",
        "min_numpy": "2.2",
        "notes": "Vector-matrix product. Cost = output_size * contracted_axis.",
    },
    # ------------------------------------------------------------------
    # counted_reduction — implemented in _pointwise.py
    # ------------------------------------------------------------------
    "sum": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Sum of array elements.",
    },
    "prod": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Product of array elements.",
    },
    "max": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Maximum value of array.",
    },
    "min": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Minimum value of array.",
    },
    "amax": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Maximum value of array (alias for max/numpy.amax).",
    },
    "amin": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Minimum value of array (alias for min/numpy.amin).",
    },
    "mean": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Arithmetic mean of array elements.",
    },
    "std": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Standard deviation; cost_multiplier=2 (two passes).",
    },
    "var": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Variance; cost_multiplier=2 (two passes).",
    },
    "argmax": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Index of maximum value.",
    },
    "argmin": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Index of minimum value.",
    },
    "cumsum": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Cumulative sum of array elements.",
    },
    "cumprod": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Cumulative product of array elements.",
    },
    "any": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Test whether any array element is true.",
    },
    "all": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Test whether all array elements are true.",
    },
    "count_nonzero": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Count non-zero elements.",
    },
    "nansum": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Sum ignoring NaNs.",
    },
    "nanprod": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Product ignoring NaNs.",
    },
    "nanmax": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Maximum ignoring NaNs.",
    },
    "nanmin": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Minimum ignoring NaNs.",
    },
    "nanmean": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Mean ignoring NaNs.",
    },
    "nanstd": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Standard deviation ignoring NaNs.",
    },
    "nanvar": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Variance ignoring NaNs.",
    },
    "nanargmax": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Index of maximum ignoring NaNs.",
    },
    "nanargmin": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Index of minimum ignoring NaNs.",
    },
    "nancumsum": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Cumulative sum ignoring NaNs.",
    },
    "nancumprod": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Cumulative product ignoring NaNs.",
    },
    "median": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Median of array elements (sorts internally).",
    },
    "nanmedian": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Median ignoring NaNs.",
    },
    "percentile": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "q-th percentile of array elements.",
    },
    "nanpercentile": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "q-th percentile ignoring NaNs.",
    },
    "quantile": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "q-th quantile of array elements.",
    },
    "nanquantile": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "q-th quantile ignoring NaNs.",
    },
    "ptp": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Peak-to-peak (max - min) range of array.",
    },
    "average": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Weighted average of array elements.",
    },
    "cumulative_sum": {
        "category": "counted_reduction",
        "module": "numpy",
        "min_numpy": "2.1",
        "notes": "Cumulative sum (NumPy 2.x array API).",
    },
    "cumulative_prod": {
        "category": "counted_reduction",
        "module": "numpy",
        "min_numpy": "2.1",
        "notes": "Cumulative product (NumPy 2.x array API).",
    },
    # ------------------------------------------------------------------
    # counted_custom — implemented in _pointwise.py / _einsum.py
    # ------------------------------------------------------------------
    "dot": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Dot product; cost = M*K*N (weight-calibrated).",
    },
    "matmul": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Matrix multiplication; cost = M*K*N (weight-calibrated).",
    },
    "einsum": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Generalized Einstein summation.",
    },
    "einsum_path": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Optimize einsum contraction path (no numeric output).",
    },
    "clip": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Clip array to [a_min, a_max] element-wise.",
    },
    "inner": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Inner product; cost = N (weight-calibrated).",
    },
    "outer": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Outer product of two vectors; cost = M*N.",
    },
    "tensordot": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Tensor dot product along specified axes.",
    },
    "vdot": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Dot product with conjugation; cost = N (weight-calibrated).",
    },
    "kron": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Kronecker product; cost proportional to output size.",
    },
    "cross": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Cross product of two 3-D vectors.",
    },
    "diff": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "n-th discrete difference along axis.",
    },
    "gradient": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Gradient using central differences.",
    },
    "ediff1d": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Differences between consecutive elements.",
    },
    "convolve": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "1-D discrete convolution.",
    },
    "correlate": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "1-D cross-correlation.",
    },
    "corrcoef": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Pearson correlation coefficients.",
    },
    "cov": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Covariance matrix.",
    },
    "trapezoid": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Integrate using the trapezoidal rule.",
    },
    "trapz": {
        "category": "counted_custom",
        "module": "numpy",
        "max_numpy": "2.4",
        "notes": "Alias for trapezoid (deprecated). Removed in numpy 2.4; use `trapezoid` instead.",
    },
    "interp": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "1-D linear interpolation.",
    },
    # ------------------------------------------------------------------
    # linalg — svd + decompositions implemented; rest blacklisted
    # ------------------------------------------------------------------
    "linalg.svd": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Singular value decomposition; cost ~ O(min(m,n)*m*n).",
    },
    "linalg.cholesky": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Cholesky decomposition. Cost: $n^3$.",
    },
    "linalg.cond": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Condition number. Cost: m*n*min(m,n) (via SVD).",
    },
    "linalg.cross": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Delegates to `fnp.cross` which charges `numel(output)` FLOPs.",
    },
    "linalg.det": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Determinant. Cost: $n^3$.",
    },
    "linalg.diagonal": {
        "category": "free",
        "module": "numpy.linalg",
        "notes": "View of diagonal — delegates to flopscope.diagonal. Cost: 0 FLOPs.",
    },
    "linalg.eig": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Eigendecomposition. Cost: $n^3$.",
    },
    "linalg.eigh": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Symmetric eigendecomposition. Cost: $n^3$.",
    },
    "linalg.eigvals": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Eigenvalues only. Cost: $n^3$.",
    },
    "linalg.eigvalsh": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Symmetric eigenvalues. Cost: $n^3$.",
    },
    "linalg.inv": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Matrix inverse. Cost: $n^3$ (LU + solve).",
    },
    "linalg.lstsq": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Least squares. Cost: m*n*min(m,n) (LAPACK gelsd/SVD).",
    },
    "linalg.matmul": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Delegates to `fnp.matmul` which charges `m*k*n` FLOPs (weight-calibrated).",
    },
    "linalg.matrix_norm": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Matrix norm. Cost depends on ord: 2*numel for Frobenius, m*n*min(m,n) for ord=2.",
    },
    "linalg.matrix_power": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Matrix power. Cost: $(\\lfloor\\log_2 k\\rfloor + \\text{popcount}(k) - 1) \\cdot n^3$ (exponentiation by squaring).",
    },
    "linalg.matrix_rank": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Matrix rank. Cost: m*n*min(m,n) (via SVD).",
    },
    "linalg.matrix_transpose": {
        "category": "free",
        "module": "numpy.linalg",
        "notes": "Transpose view — delegates to flopscope.matrix_transpose. Cost: 0 FLOPs.",
    },
    "linalg.multi_dot": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Chain matmul. Cost: sum of optimal chain matmul costs (CLRS §15.2).",
    },
    "linalg.norm": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Norm. Cost depends on ord: numel for L1/inf, 2*numel for Frobenius, m*n*min(m,n) for ord=2.",
    },
    "linalg.outer": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Delegates to `fnp.outer` which charges `m*n` FLOPs.",
    },
    "linalg.pinv": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Pseudoinverse. Cost: m*n*min(m,n) (via SVD).",
    },
    "linalg.qr": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "QR decomposition. Cost: $m \\cdot n \\cdot \\min(m,n)$.",
    },
    "linalg.slogdet": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Sign + log determinant. Cost: $n^3$.",
    },
    "linalg.solve": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Solve Ax=b. Cost: $n^3$.",
    },
    "linalg.svdvals": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Singular values only. Cost: m*n*min(m,n) (Golub-Reinsch).",
    },
    "linalg.tensordot": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Delegates to `fnp.tensordot` which charges FLOPs based on contraction.",
    },
    "linalg.tensorinv": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Tensor inverse. Cost: $n^3$ after reshape (delegates to inv).",
    },
    "linalg.tensorsolve": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Tensor solve. Cost: $n^3$ after reshape (delegates to solve).",
    },
    "linalg.trace": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Matrix trace. Cost: n (sum of diagonal elements).",
    },
    "linalg.vecdot": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Delegates to `fnp.vecdot` which charges `2*n` FLOPs.",
    },
    "linalg.vector_norm": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Vector norm. Cost: numel (or 2*numel for general p-norm).",
    },
    # ------------------------------------------------------------------
    # fft — counted_custom (14 transforms) + free (4 utility ops)
    # ------------------------------------------------------------------
    "fft.fft": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.fft2": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.fftn": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.fftfreq": {
        "category": "free",
        "module": "numpy.fft",
        "notes": "FFT sample frequencies. No arithmetic; returns index array.",
    },
    "fft.fftshift": {
        "category": "free",
        "module": "numpy.fft",
        "notes": "Shift zero-frequency component to center. No arithmetic; index reordering only.",
    },
    "fft.hfft": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "FFT of Hermitian-symmetric signal. Cost: 5*n_out*ceil(log2(n_out)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.ifft": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse 1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.ifft2": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse 2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.ifftn": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.ifftshift": {
        "category": "free",
        "module": "numpy.fft",
        "notes": "Inverse of fftshift. No arithmetic; index reordering only.",
    },
    "fft.ihfft": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse FFT of Hermitian signal. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.irfft": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse 1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.irfft2": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse 2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.irfftn": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.rfft": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.rfft2": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.rfftfreq": {
        "category": "free",
        "module": "numpy.fft",
        "notes": "Real FFT sample frequencies. No arithmetic; returns index array.",
    },
    "fft.rfftn": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    # ------------------------------------------------------------------
    # free — implemented in _free_ops.py
    # ------------------------------------------------------------------
    "array": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Create array from data. Cost: numel(input).",
    },
    "zeros": {
        "category": "free",
        "module": "numpy",
        "notes": "Create zero-filled array.",
    },
    "ones": {
        "category": "free",
        "module": "numpy",
        "notes": "Create one-filled array.",
    },
    "full": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Create array filled with scalar value. Cost: num copied.",
    },
    "eye": {
        "category": "free",
        "module": "numpy",
        "notes": "Create identity matrix.",
    },
    "identity": {
        "category": "free",
        "module": "numpy",
        "notes": "Create square identity matrix.",
    },
    "diag": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Extract diagonal or construct diagonal array. Cost: len(diagonal).",
    },
    "arange": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Return evenly spaced values in given interval. Cost: numel(output).",
    },
    "linspace": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Return evenly spaced numbers over interval. Cost: numel(output).",
    },
    "zeros_like": {
        "category": "free",
        "module": "numpy",
        "notes": "Array of zeros with same shape/type as input.",
    },
    "ones_like": {
        "category": "free",
        "module": "numpy",
        "notes": "Array of ones with same shape/type as input.",
    },
    "full_like": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Array filled with scalar, same shape/type as input. Cost: numel(output).",
    },
    "empty": {
        "category": "free",
        "module": "numpy",
        "notes": "Uninitialized array allocation.",
    },
    "empty_like": {
        "category": "free",
        "module": "numpy",
        "notes": "Uninitialized array with same shape/type as input.",
    },
    "reshape": {
        "category": "free",
        "module": "numpy",
        "notes": "Reshape array without copying.",
    },
    "transpose": {
        "category": "free",
        "module": "numpy",
        "notes": "Permute array dimensions.",
    },
    "swapaxes": {
        "category": "free",
        "module": "numpy",
        "notes": "Interchange two axes of an array.",
    },
    "moveaxis": {
        "category": "free",
        "module": "numpy",
        "notes": "Move axes to new positions.",
    },
    "concatenate": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Join arrays along axis. Cost: numel(output).",
    },
    "stack": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Join arrays along new axis. Cost: numel(output).",
    },
    "vstack": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Stack arrays vertically. Cost: numel(output).",
    },
    "hstack": {
        "category": "free",
        "module": "numpy",
        "notes": "Stack arrays horizontally.",
    },
    "split": {
        "category": "free",
        "module": "numpy",
        "notes": "Split array into sub-arrays. Cost: numel(output).",
    },
    "hsplit": {
        "category": "free",
        "module": "numpy",
        "notes": "Split array into columns.",
    },
    "vsplit": {
        "category": "free",
        "module": "numpy",
        "notes": "Split array into rows. Cost: numel(output).",
    },
    "squeeze": {
        "category": "free",
        "module": "numpy",
        "notes": "Remove size-1 dimensions.",
    },
    "expand_dims": {
        "category": "free",
        "module": "numpy",
        "notes": "Insert new size-1 axis.",
    },
    "ravel": {
        "category": "free",
        "module": "numpy",
        "notes": "Return contiguous flattened array. Cost: numel(input).",
    },
    "copy": {
        "category": "free",
        "module": "numpy",
        "notes": "Return array copy.",
    },
    "where": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Select elements based on condition. Cost: numel(input).",
    },
    "tile": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Repeat array by tiling. Cost: numel(output).",
    },
    "repeat": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Repeat elements of an array. Cost: numel(output).",
    },
    "flip": {
        "category": "free",
        "module": "numpy",
        "notes": "Reverse order of elements along axis.",
    },
    "roll": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Roll array elements along axis. Cost: numel(output).",
    },
    "sort": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Comparison sort; cost = n*ceil(log2(n)) per slice.",
    },
    "argsort": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Indirect sort; cost = n*ceil(log2(n)) per slice.",
    },
    "searchsorted": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Binary search; cost = m*ceil(log2(n)).",
    },
    "unique": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Sort-based unique; cost = n*ceil(log2(n)).",
    },
    "pad": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Pad array. Cost: numel(output).",
    },
    "triu": {
        "category": "free",
        "module": "numpy",
        "notes": "Upper triangle of array.",
    },
    "tril": {
        "category": "free",
        "module": "numpy",
        "notes": "Lower triangle of array.",
    },
    "diagonal": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Return specified diagonals. Cost: numel(input).",
    },
    "trace": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Diagonal sum; cost = min(n,m).",
    },
    "broadcast_to": {
        "category": "free",
        "module": "numpy",
        "notes": "Broadcast array to new shape. Cost: numel(output).",
    },
    "meshgrid": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Coordinate matrices from coordinate vectors. Cost: numel(output).",
    },
    "astype": {
        "category": "free",
        "module": "numpy",
        "notes": "Cast array to specified type.",
    },
    "asarray": {
        "category": "free",
        "module": "numpy",
        "notes": "Convert input to array. Cost: numel(input).",
    },
    "isnan": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Test for NaN element-wise. Cost: numel(input).",
    },
    "isinf": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Test for infinity element-wise. Cost: numel(input).",
    },
    "isfinite": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Test for finite values element-wise. Cost: numel(input).",
    },
    "allclose": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Element-wise tolerance check; cost = numel(a).",
    },
    # Additional free ops
    "rot90": {
        "category": "free",
        "module": "numpy",
        "notes": "Rotate array 90 degrees.",
    },
    "fliplr": {
        "category": "free",
        "module": "numpy",
        "notes": "Flip array left-right.",
    },
    "flipud": {
        "category": "free",
        "module": "numpy",
        "notes": "Flip array up-down.",
    },
    "atleast_1d": {
        "category": "free",
        "module": "numpy",
        "notes": "View inputs as arrays with at least one dimension.",
    },
    "atleast_2d": {
        "category": "free",
        "module": "numpy",
        "notes": "View inputs as arrays with at least two dimensions.",
    },
    "atleast_3d": {
        "category": "free",
        "module": "numpy",
        "notes": "View inputs as arrays with at least three dimensions.",
    },
    "column_stack": {
        "category": "free",
        "module": "numpy",
        "notes": "Stack 1-D arrays as columns into 2-D array.",
    },
    "dstack": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Stack arrays depth-wise (along third axis). Cost: numel(output).",
    },
    "row_stack": {
        "category": "free",
        "module": "numpy",
        "notes": "Stack arrays vertically (alias for vstack).",
    },
    "flatnonzero": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Return indices of non-zero elements in flattened array. Cost: numel(input).",
    },
    "nonzero": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Return indices of non-zero elements. Cost: numel(input).",
    },
    "argwhere": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Find indices of non-zero elements. Cost: numel(input).",
    },
    "isin": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Set membership; cost = (n+m)*ceil(log2(n+m)).",
    },
    "in1d": {
        "category": "counted_custom",
        "module": "numpy",
        "max_numpy": "2.4",
        "notes": "Set membership; cost = (n+m)*ceil(log2(n+m)). Removed in numpy 2.4; use `isin` instead.",
    },
    "select": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Return array from list of choices based on conditions. Cost: numel(input).",
    },
    "extract": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Return elements satisfying condition. Cost: numel(input).",
    },
    "place": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Change elements satisfying condition. Cost: numel(input).",
    },
    "put": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Replace elements at given flat indices. Cost: numel(input).",
    },
    "put_along_axis": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Put values into destination array using indices. Cost: numel(input).",
    },
    "putmask": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Change elements of array based on condition and input values. Cost: numel(input).",
    },
    "take": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Take elements from array along axis. Cost: numel(output).",
    },
    "take_along_axis": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Take values from input array by matching 1-D index. Cost: numel(output).",
    },
    "choose": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Construct array from index array and choices. Cost: numel(output).",
    },
    "compress": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Return selected slices along axis. Cost: numel(input).",
    },
    "array_equal": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Element-wise equality; cost = numel(a).",
    },
    "array_equiv": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Element-wise equivalence; cost = numel(a).",
    },
    "shape": {
        "category": "free",
        "module": "numpy",
        "notes": "Return shape of array.",
    },
    "size": {
        "category": "free",
        "module": "numpy",
        "notes": "Return number of elements in array.",
    },
    "ndim": {
        "category": "free",
        "module": "numpy",
        "notes": "Return number of dimensions of array.",
    },
    "dsplit": {
        "category": "free",
        "module": "numpy",
        "notes": "Split array into multiple sub-arrays depth-wise. Cost: numel(output).",
    },
    "array_split": {
        "category": "free",
        "module": "numpy",
        "notes": "Split array into sub-arrays (possibly unequal). Cost: numel(output).",
    },
    "trim_zeros": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Trim leading/trailing zeros from 1-D array. Cost: num trimmed.",
    },
    "resize": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Return new array with given shape by repeating. Cost: numel(output).",
    },
    "broadcast_shapes": {
        "category": "free",
        "module": "numpy",
        "notes": "Compute broadcast shape from input shapes.",
    },
    "broadcast_arrays": {
        "category": "free",
        "module": "numpy",
        "notes": "Broadcast arrays against each other. Cost: numel(output).",
    },
    "result_type": {
        "category": "free",
        "module": "numpy",
        "notes": "Return type that results from applying NumPy type promotion.",
    },
    "can_cast": {
        "category": "free",
        "module": "numpy",
        "notes": "Returns True if cast is safe.",
    },
    "common_type": {
        "category": "free",
        "module": "numpy",
        "notes": "Return scalar type common to all input arrays.",
    },
    "min_scalar_type": {
        "category": "free",
        "module": "numpy",
        "notes": "Return the minimum scalar type for a value.",
    },
    "promote_types": {
        "category": "free",
        "module": "numpy",
        "notes": "Return smallest type to which both types may be safely cast.",
    },
    "shares_memory": {
        "category": "free",
        "module": "numpy",
        "notes": "Determine if two arrays share memory.",
    },
    "may_share_memory": {
        "category": "free",
        "module": "numpy",
        "notes": "Determine if two arrays might share memory.",
    },
    "packbits": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Pack elements of array into bits. Cost: numel(input).",
    },
    "unpackbits": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Unpack elements of array into bits. Cost: numel(input).",
    },
    "fromfunction": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Construct array by executing function over each coordinate. Cost: numel(output).",
    },
    "fromiter": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Create array from an iterable. Cost: numel(output).",
    },
    "frombuffer": {
        "category": "free",
        "module": "numpy",
        "notes": "Interpret buffer as 1-D array. Cost: numel(output).",
    },
    "fromstring": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Create 1-D array from string data. Cost: numel(output).",
    },
    "fromfile": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Construct array from binary/text file. Cost: numel(output).",
    },
    "fromregex": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Construct array from text file using regex. Cost: numel(output).",
    },
    "from_dlpack": {
        "category": "free",
        "module": "numpy",
        "notes": "Create ndarray from DLPack object (zero-copy). Cost: numel(output).",
    },
    "block": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Assemble ndarray from nested list of blocks. Cost: numel(output).",
    },
    "bmat": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Build matrix from nested list of matrices. Cost: numel(output).",
    },
    "lexsort": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Multi-key sort; cost = k*n*ceil(log2(n)).",
    },
    "partition": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Quickselect; cost = n per slice.",
    },
    "argpartition": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Indirect partition; cost = n per slice.",
    },
    "union1d": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Set union; cost = (n+m)*ceil(log2(n+m)).",
    },
    "intersect1d": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Set intersection; cost = (n+m)*ceil(log2(n+m)).",
    },
    "setdiff1d": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Set difference; cost = (n+m)*ceil(log2(n+m)).",
    },
    "setxor1d": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Symmetric set difference; cost = (n+m)*ceil(log2(n+m)).",
    },
    "histogram": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Binning; cost = n*ceil(log2(bins)).",
    },
    "histogram2d": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "2D binning; cost = n*(ceil(log2(bx))+ceil(log2(by))).",
    },
    "histogramdd": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "ND binning; cost = n*sum(ceil(log2(b_i))).",
    },
    "histogram_bin_edges": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Bin edge computation; cost = numel(a).",
    },
    "bincount": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Integer counting; cost = numel(x).",
    },
    "digitize": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Bin search; cost = n*ceil(log2(bins)).",
    },
    "unravel_index": {
        "category": "free",
        "module": "numpy",
        "notes": "Convert flat index to multi-dimensional index.",
    },
    "ravel_multi_index": {
        "category": "free",
        "module": "numpy",
        "notes": "Convert multi-dimensional index to flat index.",
    },
    "indices": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Return array representing indices of a grid. Cost: numel(output).",
    },
    "diag_indices": {
        "category": "free",
        "module": "numpy",
        "notes": "Return indices to access main diagonal of n-D array.",
    },
    "diag_indices_from": {
        "category": "free",
        "module": "numpy",
        "notes": "Return indices to access main diagonal of given array.",
    },
    "diagflat": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Create diagonal array from flattened input. Cost: len(v).",
    },
    "mask_indices": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Return indices of mask for n x n array. Cost: numel(output).",
    },
    "tril_indices": {
        "category": "free",
        "module": "numpy",
        "notes": "Return lower-triangle indices for n x n array.",
    },
    "tril_indices_from": {
        "category": "free",
        "module": "numpy",
        "notes": "Return lower-triangle indices for given array.",
    },
    "triu_indices": {
        "category": "free",
        "module": "numpy",
        "notes": "Return upper-triangle indices for n x n array.",
    },
    "triu_indices_from": {
        "category": "free",
        "module": "numpy",
        "notes": "Return upper-triangle indices for given array.",
    },
    "fill_diagonal": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Fill main diagonal of given array. Cost: min(m,n).",
    },
    "tri": {
        "category": "free",
        "module": "numpy",
        "notes": "Array with ones at and below given diagonal.",
    },
    "geomspace": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Geometric-spaced generation; cost = num.",
    },
    "logspace": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Log-spaced generation; cost = num.",
    },
    "concat": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Join arrays along axis (NumPy 2.x array API alias for concatenate). Cost: numel(output).",
    },
    "vander": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Vandermonde matrix; cost = len(x)*(N-1).",
    },
    "ix_": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Construct open mesh from multiple sequences. Cost: numel(output).",
    },
    "rollaxis": {
        "category": "free",
        "module": "numpy",
        "notes": "Roll specified axis backwards. Cost: numel(output).",
    },
    "permute_dims": {
        "category": "free",
        "module": "numpy",
        "notes": "Permute dimensions (NumPy 2.x array API).",
    },
    "matrix_transpose": {
        "category": "free",
        "module": "numpy",
        "notes": "Transpose last two dimensions (NumPy 2.x array API).",
    },
    "unstack": {
        "category": "counted_custom",
        "module": "numpy",
        "min_numpy": "2.1",
        "notes": "Unstack array along axis into tuple of arrays (NumPy 2.x). Cost: numel(output).",
    },
    "delete": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Return array with sub-arrays deleted along axis. Cost: num deleted.",
    },
    "insert": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Insert values along axis before given indices. Cost: numel(values).",
    },
    "append": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Append values to end of array. Cost: numel(values).",
    },
    "copyto": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Copy values from src to dst array. Cost: num copied.",
    },
    "unique_all": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Sort-based unique; cost = n*ceil(log2(n)).",
    },
    "unique_counts": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Sort-based unique; cost = n*ceil(log2(n)).",
    },
    "unique_inverse": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Sort-based unique; cost = n*ceil(log2(n)).",
    },
    "unique_values": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Sort-based unique; cost = n*ceil(log2(n)).",
    },
    "asarray_chkfinite": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Convert to array, raising if NaN or inf. Cost: numel(input).",
    },
    "require": {
        "category": "free",
        "module": "numpy",
        "notes": "Return array that satisfies requirements.",
    },
    "issubdtype": {
        "category": "free",
        "module": "numpy",
        "notes": "Return True if first argument is lower in type hierarchy.",
    },
    "isdtype": {
        "category": "free",
        "module": "numpy",
        "notes": "Return True if array or dtype is of specified kind (NumPy 2.x).",
    },
    "isscalar": {
        "category": "free",
        "module": "numpy",
        "notes": "Return True if input is a scalar.",
    },
    "isfortran": {
        "category": "free",
        "module": "numpy",
        "notes": "Return True if array is Fortran contiguous.",
    },
    "iterable": {
        "category": "free",
        "module": "numpy",
        "notes": "Return True if object is iterable.",
    },
    "typename": {
        "category": "free",
        "module": "numpy",
        "notes": "Return description of given data type code.",
    },
    "mintypecode": {
        "category": "free",
        "module": "numpy",
        "notes": "Return minimum data type character that can satisfy all given types.",
    },
    "base_repr": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return string representation of number in given base. Cost: numel(input).",
    },
    "binary_repr": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return binary string representation of the input number. Cost: numel(input).",
    },
    # ------------------------------------------------------------------
    # random — passthrough, category=free
    # ------------------------------------------------------------------
    "random.beta": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.binomial": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.bytes": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.chisquare": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.choice": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output) if replace, n*ceil(log2(n)) if not.",
    },
    "random.default_rng": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Construct a new Generator with default BitGenerator.",
    },
    "random.dirichlet": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.exponential": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.f": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.gamma": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.geometric": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.get_state": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Return tuple representing internal state of generator.",
    },
    "random.gumbel": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.hypergeometric": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.laplace": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.logistic": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.lognormal": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.logseries": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.multinomial": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.multivariate_normal": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.negative_binomial": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.noncentral_chisquare": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.noncentral_f": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.normal": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.pareto": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.permutation": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Shuffle; cost = n*ceil(log2(n)).",
    },
    "random.poisson": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.power": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.rand": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.randint": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.randn": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.random": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.random_integers": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.random_sample": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.ranf": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.rayleigh": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.sample": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.seed": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Seed random number generator.",
    },
    "random.set_state": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Set internal state of generator.",
    },
    "random.shuffle": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Shuffle; cost = n*ceil(log2(n)).",
    },
    "random.standard_cauchy": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.standard_exponential": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.standard_gamma": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.standard_normal": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.standard_t": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.triangular": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.uniform": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.vonmises": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.wald": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.weibull": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    "random.zipf": {
        "category": "counted_custom",
        "module": "numpy.random",
        "notes": "Sampling; cost = numel(output).",
    },
    # ------------------------------------------------------------------
    # random.Generator.* — counted method-level entries (issue #18)
    # category=counted_random_method or free_random_method
    # ------------------------------------------------------------------
    "random.Generator.beta": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Beta distribution; cost = numel(output).",
    },
    "random.Generator.binomial": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Binomial distribution; cost = numel(output).",
    },
    "random.Generator.bytes": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "length",
        "notes": "Raw bytes; cost = length argument.",
    },
    "random.Generator.chisquare": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Chi-square distribution; cost = numel(output).",
    },
    "random.Generator.choice": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "choice_cost",
        "notes": "numel(output) if replace else sort_cost(n).",
    },
    "random.Generator.dirichlet": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Dirichlet distribution; cost = numel(output).",
    },
    "random.Generator.exponential": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Exponential distribution; cost = numel(output).",
    },
    "random.Generator.f": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "F distribution; cost = numel(output).",
    },
    "random.Generator.gamma": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Gamma distribution; cost = numel(output).",
    },
    "random.Generator.geometric": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Geometric distribution; cost = numel(output).",
    },
    "random.Generator.gumbel": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Gumbel distribution; cost = numel(output).",
    },
    "random.Generator.hypergeometric": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Hypergeometric distribution; cost = numel(output).",
    },
    "random.Generator.integers": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Modern Generator integers (replaces legacy randint).",
    },
    "random.Generator.laplace": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Laplace distribution; cost = numel(output).",
    },
    "random.Generator.logistic": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Logistic distribution; cost = numel(output).",
    },
    "random.Generator.lognormal": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Log-normal distribution; cost = numel(output).",
    },
    "random.Generator.logseries": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Log-series distribution; cost = numel(output).",
    },
    "random.Generator.multinomial": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Multinomial distribution; cost = numel(output).",
    },
    "random.Generator.multivariate_hypergeometric": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Multivariate hypergeometric; cost = numel(output).",
    },
    "random.Generator.multivariate_normal": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Multivariate normal; cost = numel(output).",
    },
    "random.Generator.negative_binomial": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Negative binomial distribution; cost = numel(output).",
    },
    "random.Generator.noncentral_chisquare": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Noncentral chi-square; cost = numel(output).",
    },
    "random.Generator.noncentral_f": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Noncentral F; cost = numel(output).",
    },
    "random.Generator.normal": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Normal distribution; cost = numel(output).",
    },
    "random.Generator.pareto": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Pareto distribution; cost = numel(output).",
    },
    "random.Generator.permutation": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "shape[axis]",
        "notes": "Random permutation; cost = shape[axis] (Fisher-Yates draws).",
    },
    "random.Generator.permuted": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(input)",
        "notes": "Permute along axis; cost from input array size.",
    },
    "random.Generator.poisson": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Poisson distribution; cost = numel(output).",
    },
    "random.Generator.power": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Power distribution; cost = numel(output).",
    },
    "random.Generator.random": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Uniform [0, 1) — modern Generator equivalent of rand/random_sample.",
    },
    "random.Generator.rayleigh": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Rayleigh distribution; cost = numel(output).",
    },
    "random.Generator.shuffle": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "shape[axis]",
        "notes": "In-place shuffle; cost = shape[axis] (Fisher-Yates draws).",
    },
    "random.Generator.standard_cauchy": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Standard Cauchy distribution; cost = numel(output).",
    },
    "random.Generator.standard_exponential": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Standard exponential; cost = numel(output).",
    },
    "random.Generator.standard_gamma": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Standard gamma; cost = numel(output).",
    },
    "random.Generator.standard_normal": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Standard normal; cost = numel(output).",
    },
    "random.Generator.standard_t": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Standard Student-t; cost = numel(output).",
    },
    "random.Generator.triangular": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Triangular distribution; cost = numel(output).",
    },
    "random.Generator.uniform": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Uniform distribution; cost = numel(output).",
    },
    "random.Generator.vonmises": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Von Mises distribution; cost = numel(output).",
    },
    "random.Generator.wald": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Wald (inverse Gaussian) distribution; cost = numel(output).",
    },
    "random.Generator.weibull": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Weibull distribution; cost = numel(output).",
    },
    "random.Generator.zipf": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Zipf distribution; cost = numel(output).",
    },
    # Free (passthrough; counting happens at the sampler-method level)
    "random.Generator.bit_generator": {
        "category": "free_random_method",
        "module": "numpy.random",
        "notes": "Underlying BitGenerator; attribute access, no math.",
    },
    "random.Generator.spawn": {
        "category": "free_random_method",
        "module": "numpy.random",
        "notes": "Returns child Generators; subclass override wraps them as _CountedGenerator.",
    },
    # ------------------------------------------------------------------
    # random.RandomState.* — counted method-level entries (issue #18)
    # ------------------------------------------------------------------
    "random.RandomState.beta": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy beta sampler; cost = numel(output).",
    },
    "random.RandomState.binomial": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy binomial sampler; cost = numel(output).",
    },
    "random.RandomState.bytes": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "length",
        "notes": "Legacy bytes sampler; cost = length argument.",
    },
    "random.RandomState.chisquare": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy chi-square sampler; cost = numel(output).",
    },
    "random.RandomState.choice": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "choice_cost",
        "notes": "Legacy choice sampler; numel(output) if replace else sort_cost(n).",
    },
    "random.RandomState.dirichlet": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy Dirichlet sampler; cost = numel(output).",
    },
    "random.RandomState.exponential": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy exponential sampler; cost = numel(output).",
    },
    "random.RandomState.f": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy F sampler; cost = numel(output).",
    },
    "random.RandomState.gamma": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy gamma sampler; cost = numel(output).",
    },
    "random.RandomState.geometric": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy geometric sampler; cost = numel(output).",
    },
    "random.RandomState.gumbel": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy Gumbel sampler; cost = numel(output).",
    },
    "random.RandomState.hypergeometric": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy hypergeometric sampler; cost = numel(output).",
    },
    "random.RandomState.laplace": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy Laplace sampler; cost = numel(output).",
    },
    "random.RandomState.logistic": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy logistic sampler; cost = numel(output).",
    },
    "random.RandomState.lognormal": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy log-normal sampler; cost = numel(output).",
    },
    "random.RandomState.logseries": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy log-series sampler; cost = numel(output).",
    },
    "random.RandomState.multinomial": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy multinomial sampler; cost = numel(output).",
    },
    "random.RandomState.multivariate_normal": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy multivariate normal; cost = numel(output).",
    },
    "random.RandomState.negative_binomial": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy negative binomial sampler; cost = numel(output).",
    },
    "random.RandomState.noncentral_chisquare": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy noncentral chi-square; cost = numel(output).",
    },
    "random.RandomState.noncentral_f": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy noncentral F; cost = numel(output).",
    },
    "random.RandomState.normal": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy normal sampler; cost = numel(output).",
    },
    "random.RandomState.pareto": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy Pareto sampler; cost = numel(output).",
    },
    "random.RandomState.permutation": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "shape[axis]",
        "notes": "Legacy permutation; cost = shape[axis] (Fisher-Yates draws). RandomState has no axis kwarg; defaults to 0.",
    },
    "random.RandomState.poisson": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy Poisson sampler; cost = numel(output).",
    },
    "random.RandomState.power": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy power sampler; cost = numel(output).",
    },
    "random.RandomState.rand": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy uniform [0,1); cost = numel(output).",
    },
    "random.RandomState.randint": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy integer sampler; cost = numel(output).",
    },
    "random.RandomState.randn": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy standard normal alias; cost = numel(output).",
    },
    "random.RandomState.random": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy uniform [0,1) alias for random_sample.",
    },
    "random.RandomState.random_integers": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy deprecated integer sampler; cost = numel(output).",
    },
    "random.RandomState.random_sample": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy uniform [0,1); cost = numel(output).",
    },
    "random.RandomState.rayleigh": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy Rayleigh sampler; cost = numel(output).",
    },
    "random.RandomState.shuffle": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "shape[axis]",
        "notes": "Legacy in-place shuffle; cost = shape[axis] (Fisher-Yates draws). RandomState has no axis kwarg; defaults to 0.",
    },
    "random.RandomState.standard_cauchy": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy standard Cauchy; cost = numel(output).",
    },
    "random.RandomState.standard_exponential": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy standard exponential; cost = numel(output).",
    },
    "random.RandomState.standard_gamma": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy standard gamma; cost = numel(output).",
    },
    "random.RandomState.standard_normal": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy standard normal; cost = numel(output).",
    },
    "random.RandomState.standard_t": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy standard Student-t; cost = numel(output).",
    },
    "random.RandomState.tomaxint": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy max-int sampler; cost = numel(output).",
    },
    "random.RandomState.triangular": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy triangular distribution; cost = numel(output).",
    },
    "random.RandomState.uniform": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy uniform sampler; cost = numel(output).",
    },
    "random.RandomState.vonmises": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy Von Mises sampler; cost = numel(output).",
    },
    "random.RandomState.wald": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy Wald sampler; cost = numel(output).",
    },
    "random.RandomState.weibull": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy Weibull sampler; cost = numel(output).",
    },
    "random.RandomState.zipf": {
        "category": "counted_random_method",
        "module": "numpy.random",
        "cost_formula": "numel(output)",
        "notes": "Legacy Zipf sampler; cost = numel(output).",
    },
    # Free RandomState methods
    "random.RandomState.get_state": {
        "category": "free_random_method",
        "module": "numpy.random",
        "notes": "State accessor; no math.",
    },
    "random.RandomState.seed": {
        "category": "free_random_method",
        "module": "numpy.random",
        "notes": "Seed setter; no math.",
    },
    "random.RandomState.set_state": {
        "category": "free_random_method",
        "module": "numpy.random",
        "notes": "State setter; no math.",
    },
    # ------------------------------------------------------------------
    # stats distributions (pdf/cdf/ppf)
    # ------------------------------------------------------------------
    "stats.norm.pdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Normal PDF; cost = numel(input).",
    },
    "stats.norm.cdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Normal CDF; cost = numel(input).",
    },
    "stats.norm.ppf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Normal PPF (inverse CDF); cost = numel(input).",
    },
    "stats.uniform.pdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Uniform PDF; cost = numel(input).",
    },
    "stats.uniform.cdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Uniform CDF; cost = numel(input).",
    },
    "stats.uniform.ppf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Uniform PPF; cost = numel(input).",
    },
    "stats.expon.pdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Exponential PDF; cost = numel(input).",
    },
    "stats.expon.cdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Exponential CDF; cost = numel(input).",
    },
    "stats.expon.ppf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Exponential PPF; cost = numel(input).",
    },
    "stats.cauchy.pdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Cauchy PDF; cost = numel(input).",
    },
    "stats.cauchy.cdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Cauchy CDF; cost = numel(input).",
    },
    "stats.cauchy.ppf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Cauchy PPF; cost = numel(input).",
    },
    "stats.logistic.pdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Logistic PDF; cost = numel(input).",
    },
    "stats.logistic.cdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Logistic CDF; cost = numel(input).",
    },
    "stats.logistic.ppf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Logistic PPF; cost = numel(input).",
    },
    "stats.laplace.pdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Laplace PDF; cost = numel(input).",
    },
    "stats.laplace.cdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Laplace CDF; cost = numel(input).",
    },
    "stats.laplace.ppf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Laplace PPF; cost = numel(input).",
    },
    "stats.lognorm.pdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Log-normal PDF; cost = numel(input).",
    },
    "stats.lognorm.cdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Log-normal CDF; cost = numel(input).",
    },
    "stats.lognorm.ppf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Log-normal PPF; cost = numel(input).",
    },
    "stats.truncnorm.pdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Truncated normal PDF; cost = numel(input).",
    },
    "stats.truncnorm.cdf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Truncated normal CDF; cost = numel(input).",
    },
    "stats.truncnorm.ppf": {
        "category": "counted_custom",
        "module": "flopscope.stats",
        "notes": "Truncated normal PPF; cost = numel(input).",
    },
    # ------------------------------------------------------------------
    # blacklisted — poly functions
    # ------------------------------------------------------------------
    "poly": {
        "category": "counted_custom",
        "module": "flopscope._polynomial",
        "notes": "Polynomial from roots. Cost: $n^2$ FLOPs.",
    },
    "roots": {
        "category": "counted_custom",
        "module": "flopscope._polynomial",
        "notes": "Return roots of polynomial with given coefficients. Cost: $n^3$ (companion matrix eig, simplified).",
    },
    "polyadd": {
        "category": "counted_custom",
        "module": "flopscope._polynomial",
        "notes": "Add two polynomials. Cost: max(n1, n2) FLOPs.",
    },
    "polyder": {
        "category": "counted_custom",
        "module": "flopscope._polynomial",
        "notes": "Differentiate polynomial. Cost: n FLOPs.",
    },
    "polydiv": {
        "category": "counted_custom",
        "module": "flopscope._polynomial",
        "notes": "Divide one polynomial by another. Cost: n1 * n2 FLOPs.",
    },
    "polyfit": {
        "category": "counted_custom",
        "module": "flopscope._polynomial",
        "notes": "Least squares polynomial fit. Cost: 2 * m * (deg+1)^2 FLOPs.",
    },
    "polyint": {
        "category": "counted_custom",
        "module": "flopscope._polynomial",
        "notes": "Integrate polynomial. Cost: n FLOPs.",
    },
    "polymul": {
        "category": "counted_custom",
        "module": "flopscope._polynomial",
        "notes": "Multiply polynomials. Cost: n1 * n2 FLOPs.",
    },
    "polysub": {
        "category": "counted_custom",
        "module": "flopscope._polynomial",
        "notes": "Difference (subtraction) of two polynomials. Cost: max(n1, n2) FLOPs.",
    },
    "polyval": {
        "category": "counted_custom",
        "module": "flopscope._polynomial",
        "notes": "Evaluate polynomial at given points. Cost: $2 \\cdot m \\cdot \\text{deg}$ (Horner's method, FMA=2).",
    },
    # counted_custom — window functions
    "bartlett": {
        "category": "counted_custom",
        "module": "flopscope._window",
        "notes": "Bartlett window. Cost: n (one linear eval per sample).",
    },
    "blackman": {
        "category": "counted_custom",
        "module": "flopscope._window",
        "notes": "Blackman window. Cost: 3*n (three cosine terms per sample).",
    },
    "hamming": {
        "category": "counted_custom",
        "module": "flopscope._window",
        "notes": "Hamming window. Cost: n (one cosine per sample).",
    },
    "hanning": {
        "category": "counted_custom",
        "module": "flopscope._window",
        "notes": "Hanning window. Cost: n (one cosine per sample).",
    },
    "kaiser": {
        "category": "counted_custom",
        "module": "flopscope._window",
        "notes": "Kaiser window. Cost: 3*n (Bessel function eval per sample).",
    },
    # blacklisted — IO
    "genfromtxt": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Load data from text file with missing values. Not supported.",
    },
    "loadtxt": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Load data from text file. Not supported.",
    },
    "load": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Load arrays from .npy/.npz files. Not supported.",
    },
    "save": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Save array to .npy file. Not supported.",
    },
    "savetxt": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Save array to text file. Not supported.",
    },
    "savez": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Save multiple arrays to .npz file. Not supported.",
    },
    "savez_compressed": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Save multiple arrays to compressed .npz file. Not supported.",
    },
    # blacklisted — config / runtime
    "show_config": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Show NumPy build configuration. Not supported.",
    },
    "show_runtime": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Show runtime info. Not supported.",
    },
    "get_include": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return directory containing NumPy C header files. Not supported.",
    },
    "getbufsize": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return size of buffer used in ufuncs. Not supported.",
    },
    "setbufsize": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Set size of buffer used in ufuncs. Not supported.",
    },
    "geterr": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Get current way of handling floating-point errors. Not supported.",
    },
    "seterr": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Set how floating-point errors are handled. Not supported.",
    },
    "geterrcall": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return current callback function for floating-point errors. Not supported.",
    },
    "seterrcall": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Set callback function for floating-point errors. Not supported.",
    },
    # blacklisted — advanced/meta
    "asmatrix": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Interpret input as matrix (deprecated). Not supported.",
    },
    "nested_iters": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Create nested iterators for multi-index broadcasting. Not supported.",
    },
    "frompyfunc": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Take arbitrary Python function and return NumPy ufunc. Not supported.",
    },
    "piecewise": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Piecewise function. Cost: numel(input).",
    },
    "apply_along_axis": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Apply function along axis. Cost: numel(output). Inner function costs tracked separately.",
    },
    "apply_over_axes": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Apply function over multiple axes. Cost: numel(output).",
    },
    # blacklisted — datetime
    "datetime_as_string": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Convert datetime array to string representation. Not supported.",
    },
    "datetime_data": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Get information about the step size of datetime dtype. Not supported.",
    },
    "busday_count": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Count valid days between begindate and enddate. Not supported.",
    },
    "busday_offset": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Apply offset to dates subject to valid day rules. Not supported.",
    },
    "is_busday": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Calculates which of given dates are valid days. Not supported.",
    },
    # blacklisted — print/string formatting
    "array2string": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return string representation of array. Not supported.",
    },
    "array_repr": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return string representation of array. Not supported.",
    },
    "array_str": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return string representation of data in array. Not supported.",
    },
    "format_float_positional": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Format floating point scalar as decimal string. Not supported.",
    },
    "format_float_scientific": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Format floating point scalar as scientific notation. Not supported.",
    },
    "unwrap": {
        "category": "counted_custom",
        "module": "flopscope._unwrap",
        "notes": "Phase unwrap. Cost: $\\text{numel}(\\text{input})$ (diff + conditional adjustment).",
    },
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def make_module_getattr(module_prefix: str, module_label: str):
    """Create a __getattr__ that consults the registry."""

    def __getattr__(name: str):
        qualified = f"{module_prefix}{name}" if module_prefix else name
        if qualified in REGISTRY:
            entry = REGISTRY[qualified]
            cat = entry["category"]
            notes = entry.get("notes", "")
            if cat == "blacklisted":
                raise AttributeError(
                    f"{module_label} does not support '{name}' (blacklisted). {notes}"
                )
            if cat == "unclassified":
                raise AttributeError(
                    f"{module_label} has not yet classified '{name}'. "
                    f"Please report this at https://github.com/AIcrowd/flopscope/issues"
                )
            raise AttributeError(
                f"'{name}' is registered but not yet implemented in {module_label}. {notes}"
            )
        raise AttributeError(
            f"{module_label} does not provide '{name}'. "
            f"See https://github.com/AIcrowd/flopscope for supported operations."
        )

    return __getattr__
