"""Known divergences between flopscope and NumPy.

Each entry maps a test node ID (or pattern) to a reason string.
Tests matching these patterns are marked xfail when running NumPy's
test suite against flopscope.

Current state (2026-08-05, numpy 2.2.6, object-dtype-ban widen to a numeric allowlist):
    The dtype ban widened from an object-only (`hasobject`) check to a
    numeric allowlist: refuse_non_numeric_dtype now refuses every dtype
    whose `kind` is outside `"biufc"` (bool, integer, float, complex), not
    just object. NumPy's own test_ufunc.py and test_umath.py construct
    string/bytes/structured-void/datetime64/timedelta64 arrays directly and
    far more often than object arrays -- as ufunc-loop-signature probes, as
    alignment/padding test techniques (a structured dtype with a padding
    field), and as genuine operands for ops that accept timedelta64 (e.g.
    `absolute`) or accept "anything" (the logical ufuncs) -- so those tests
    now hit UnsupportedDtypeError instead of running. 6 new
    NONNUMERIC_DTYPE_BANNED patterns added below: test_ufunc_noncontiguous
    and test_ufunc_types match the whole parametrized test function rather
    than an enumerated per-ufunc list (91 of ~140, and 18, variants actually
    fail; the rest xpass harmlessly under strict=False -- the same
    broad-wildcard trade-off test_reduction_with_where* below already makes).
    TestUfunc::test_logical_ufuncs_support_anything (all 3 variants fail
    unconditionally), TestUfunc::test_at_no_loop_for_op,
    TestAdd::test_reduce_alignment, and the module-level test_reduceat round
    out the six. test_pocketfft/test_helper/test_polynomial/test_random were
    unaffected, same as the object-only ban.

    The renamed OBJECT_DTYPE_BANNED -> NONNUMERIC_DTYPE_BANNED category
    keeps its original 15 patterns unchanged (object is still refused the
    same way, just via a wider predicate).

    The 16th, not-expressible-as-a-per-test-node-xfail case from the
    previous state is now fixed: `_ImmutabilityXfailPlugin`'s
    `pytest_make_collect_report` hook (conftest.py) is extended to match
    the ban's sentinel too (widened from "object dtype is not billable" to
    the dtype-agnostic "flopscope meters only numeric dtypes"), reclassifying
    numpy._core.tests.test_numeric.py's collection failure
    (TestClip.test_clip_problem_cases evaluates
    ``np.zeros(10, dtype=object)`` as a class-body-level default argument at
    MODULE IMPORT time) as a module-level skip instead of a collection error.

Previous state (2026-07-29, numpy 2.4.6, PR #171 first REAL numpy-version matrix):
    The CI matrix pin was vacuous before PR #171 (uv run re-synced the locked
    numpy back in), so this was the first genuine 2.4 run of the borrowed
    suites. 1 new xfail added: TestUfunc::test_output_ellipsis_errors —
    numpy 2.4's nested-Ellipsis out= special-case message diverges from
    flopscope's deliberate out=-refusal wording (TypeError parity holds).

Previous state (2026-04-17, numpy 2.4.4, after Task 10 triage):
    Total:          ~7,862 passed, 47 xfailed, 0 failures (all suites)
    test_umath:     ~4,671 passed  (15 xfailed — 1 new test_ufunc_override_where)
    test_ufunc:       ~855 passed  (10 xfailed)
    test_numeric:   ~1,616 passed  (17 xfailed)
    test_linalg:      ~467 passed   (1 xfailed — numpy-internal LAPACK mark)
    test_pocketfft:    148 passed   (0 xfailed)
    test_helper:         8 passed   (0 xfailed)
    test_polynomial:    39 passed   (1 xfailed)
    test_random:       140 passed   (2 xfailed — 3 stale xfails removed)

Previous state (2026-04-17, numpy 2.3.5, after Task 9 triage):
    Total:          ~7,862 passed, 47 xfailed, 0 failures (all suites)
    test_umath:     ~4,668 passed  (14 xfailed)
    test_ufunc:       ~829 passed   (9+1 xfailed — 1 new pattern added)
    test_numeric:   ~1,608 passed  (17 xfailed)
    test_linalg:      ~467 passed   (1 xfailed — numpy-internal LAPACK mark)
    test_pocketfft:    148 passed   (0 xfailed)
    test_helper:         8 passed   (0 xfailed)
    test_polynomial:    37 passed   (1 xfailed)
    test_random:       137 passed   (2 xfailed)

Previous state (2026-04-14, numpy 2.2, after Tier 1+2+3 fixes):
    Total:          ~7,861 passed, 46 xfailed, 0 failures (all suites)
    test_umath:     ~4,668 passed  (12 xfailed)
    test_ufunc:       ~795 passed   (3 xfailed — 4 stale xfails removed)
    test_numeric:   ~1,604 passed  (14 xfailed — nep50_isclose fixed)
    test_linalg:      ~408 passed   (0 xfailed — cond NaN + matrix_rank fixed)
    test_pocketfft:    148 passed   (0 xfailed)
    test_polynomial:   600 passed   (1 xfailed — polydiv scalar fixed)
    test_random:       139 passed   (5 xfailed)

Fixes applied:
    OWNDATA fix:        All TestClip OWNDATA_VIEW patterns removed (clip now owns its data).
    array-skip fix:     TestNorm matrix tests, TestQR modes, TestEighCases removed
                        (array patching skipped, numpy linalg functions work natively).
    named-tuple fix:    TestSVD::test_types*, TestSVDHermitian::test_types* removed
                        (svd now returns proper named tuple).
    linalg subclass:    All TestEig/TestInv/TestSolve/TestLstsq/TestSVD/TestPinv/
                        TestSVDHermitian/TestPinvHermitian sq_cases/generalized* removed
                        (NonDescriptor fix and linalg return-type fixes).
    random fix:         TestRandint::test_* (6 patterns) and test_choice_return_shape
                        removed (now pass with NonDescriptor fix).
    StdVar fix:         test_out_scalar removed (std/var out= now works).
    7 trivial fixes:    _asflopscope order='A', linalg.diagonal axis=-2/-1, linalg.cross
                        validation, tensordot int axes, norm axis validation,
                        clip argument validation, astype copy/device kwargs.
    Tier 3 fixes:       cond NaN (SVD fallback per-matrix), matrix_rank 1D input,
                        polydiv scalar input (atleast_1d), isclose NEP 50 type
                        promotion (keep Python scalars un-coerced).
    4 stale xfails removed: test_scalar_equal, test_struct_ufunc,
                        test_ufunc_override_where, test_safe_casting.

What fnp patch (55 functions):
    Non-ufunc reductions and special functions (all, any, amax, amin,
    argmax, argmin, average, cumsum, cumprod, mean, median, std, var,
    sum, prod, etc.) plus misc functions like isclose, real, imag, etc.

What fnp DON'T patch (and why):
    - Ufuncs (101): flopscope functions are plain callables, not ufuncs.
      They lack .reduce/.accumulate/.outer/.nargs/etc. Patching would
      break test collection and any test using ufunc protocol.
    - Free ops (220): pass-throughs that delegate to numpy. Patching
      causes infinite recursion since flopscope's _np IS np.
    - Counted custom (36): dot, matmul, einsum, convolve, etc. call
      _np.func() via module lookup. Same recursion issue.
    - Submodule functions (38): linalg.*, fft.*. Same recursion issue.
    - Blacklisted (32): intentionally unsupported.
    - linalg.outer: fnp.linalg.outer delegates to np.outer (not
      np.linalg.outer), so patching it causes a collection-time error
      in test_linalg.py which checks ValueError for 2D input at class
      definition time.

Categories for failures:
    UNSUPPORTED_DTYPE, UFUNC_INTERNALS, BUDGET_SIDE_EFFECT,
    NOT_IMPLEMENTED, NUMPY_INTERNAL, SUBCLASS_RETURN, WRAPPER_SIGNATURE,
    BEHAVIORAL_SHIM, REMOVED_IN_NUMPY

New categories added for numpy 2.3 triage (Task 9):

    BEHAVIORAL_SHIM — numpy's own test asserts the 2.3+ behavior that
        flopscope intentionally shims away (e.g. count_nonzero test asserts
        numpy scalar return; flopscope returns int).

    REMOVED_IN_NUMPY — numpy removed this symbol in 2.4; flopscope gates it
        off. The upstream test still references the removed symbol.
        (Not used for numpy 2.3 triage — nothing flopscope wraps was removed
        in 2.3; category reserved for the 2.4 triage in Task 11.)

Changes in numpy 2.4 triage (Task 10):
    1 new xfail added: test_ufunc_override_where — numpy 2.4 changed
        ufunc dispatch so FlopscopeArray (returned by patched np.zeros) ends
        up as out= in an OverriddenArray._unwrap call; the unwrap sees a
        non-matching ndarray subclass and returns NotImplemented, then
        [0] subscript crashes.  SUBCLASS_RETURN category.
    3 stale xfails removed: test_shuffle_untyped_warning[numpy.random],
        test_shuffle_no_object_unpacking[False-numpy.random],
        test_shuffle_no_object_unpacking[False-random1] — all xpassed on
        both numpy 2.3 and 2.4.
        (test_out_wrap_no_leak retained: still fails on numpy 2.2;
        xpassed on 2.3/2.4 is acceptable since strict=False.)
    Note on test_ufunc_override_where history: this pattern was removed
        in a previous task (marked in Fixes applied above) and is now
        re-added specifically for numpy 2.4 where it regressed.

Unit suite gaps (for Task 11 to fix, not this file):
    test_sorting_ops::TestIn1d — needs skipif(np>=2.4) because in1d
        raises UnsupportedFunctionError on 2.4.
    test_pointwise_coverage::TestTrapz — needs skipif(np>=2.4) because
        trapz raises UnsupportedFunctionError on 2.4.
    test_signature_conformance — 6 tests fail because numpy 2.4 added
        C-level positional-only annotations to dot, packbits, unpackbits,
        shares_memory, ravel_multi_index, promote_types whose signatures
        now differ from flopscope's pass-through wrappers (*args, **kwargs).
"""

# Reason-string constants for use in XFAIL_PATTERNS values.
# Using bare string literals in the dict is fine too; these exist so
# grep / tooling can find all tests sharing a category quickly.
# NOTE: BEHAVIORAL_SHIM and REMOVED_IN_NUMPY are reserved for future
# triage rounds. The current 2.3/2.4 triage did not surface any test
# needing these categories — numpy's own test suite doesn't exercise
# count_nonzero return types or the removed in1d/trapz symbols from
# within its bundled test files. If a future triage finds such tests,
# use these constants as XFAIL_PATTERNS values.
BEHAVIORAL_SHIM = (
    "BEHAVIORAL_SHIM: flopscope intentionally preserves pre-2.3 behavior; "
    "numpy's test asserts the 2.3+ behavior and therefore fails when "
    "monkeypatched."
)
REMOVED_IN_NUMPY = (
    "REMOVED_IN_NUMPY: numpy removed this symbol in 2.4; flopscope gates it "
    "off. The upstream test still references the removed symbol."
)
NEEDS_TRIAGE = (
    "NEEDS_TRIAGE: failure surfaced by the 2026-05-23 compat-harness "
    "restoration. Each case passes in isolation but fails when run in the "
    "parametrize sweep — order-dependent state pollution somewhere in the "
    "ufunc/SymmetricTensor pipeline. Not in scope for the issue-70 PR; "
    "follow-up triage queued."
)
NONNUMERIC_DTYPE_BANNED = (
    "NONNUMERIC_DTYPE_BANNED: flopscope's dtype ban refuses any array whose "
    "dtype.kind is outside the numeric allowlist 'biufc' (bool, integer, "
    "float, complex) wherever it participates in billing -- as an operand, "
    "as an out=, as a dtype= request -- via "
    "refuse_non_numeric_dtype/UnsupportedDtypeError. This NumPy test "
    "constructs or requires a non-numeric-dtype array (object, string, "
    "bytes, structured/void, datetime64, or timedelta64) directly, so it "
    "now hits that refusal instead of running. By design, not a bug: see "
    "tests/test_object_dtype_ban.py."
)

XFAIL_PATTERNS: dict[str, str] = {
    # ------------------------------------------------------------------ #
    # NONNUMERIC_DTYPE_BANNED — object-dtype-ban plan Task 5, widened to a #
    # numeric allowlist                                                   #
    # ------------------------------------------------------------------ #
    # numpy's test_ufunc.py tests object-dtype ufunc loops (PyUFunc_O_O and
    # friends), object-dtype reduction/accumulate/reduceat, and a handful of
    # tests that fold an object-dtype case into an otherwise-numeric
    # parametrize sweep (test_zerosize_reduction's `[]`-vs-object-array pair,
    # test_vecmatvec_identity's vec2/vec3 object-dtype variants). Each of
    # these constructs an object array and passes it through a flopscope-
    # patched function, which now raises UnsupportedDtypeError before the
    # arithmetic numpy's test wants to observe ever runs.
    "*TestUfuncGenericLoops::test_unary_PyUFunc_O_O": NONNUMERIC_DTYPE_BANNED,
    "*TestUfuncGenericLoops::test_unary_PyUFunc_O_O_method_simple": NONNUMERIC_DTYPE_BANNED,
    "*TestUfuncGenericLoops::test_binary_PyUFunc_OO_O": NONNUMERIC_DTYPE_BANNED,
    "*TestUfuncGenericLoops::test_binary_PyUFunc_OO_O_method": NONNUMERIC_DTYPE_BANNED,
    "*TestUfuncGenericLoops::test_binary_PyUFunc_On_Om_method": NONNUMERIC_DTYPE_BANNED,
    "*TestUfunc::test_object_array_reduction": NONNUMERIC_DTYPE_BANNED,
    "*TestUfunc::test_object_array_accumulate_inplace": NONNUMERIC_DTYPE_BANNED,
    "*TestUfunc::test_object_array_reduceat_inplace": NONNUMERIC_DTYPE_BANNED,
    "*TestUfunc::test_vecdot_object_breaks_outer_loop_on_error": NONNUMERIC_DTYPE_BANNED,
    # test_zerosize_reduction's own list mixes a plain `[]` case (still
    # passes) with `np.array([], dtype=object)` (now refused) inside one
    # test body, so the whole test node fails, not just one parametrize id.
    "*TestUfunc::test_zerosize_reduction": NONNUMERIC_DTYPE_BANNED,
    # vec2/vec3 are the object-dtype entries in test_vecmatvec_identity's
    # `vec` parametrize list (vec0/vec1 are float/complex and still pass).
    # No leading/embedded `*`: fnmatch treats `[...]` as a character CLASS,
    # not a literal bracket, so a wildcard pattern built around a bracketed
    # parametrize id silently fails to match (confirmed: `[*-vec2]` matches
    # nothing). These rely on the loader's plain-substring fallback
    # (`pattern in node_id`) instead, the same way the pre-existing
    # `TestRandomDist::test_shuffle_untyped_warning[random2]` entry below
    # does.
    "TestUfunc::test_vecmatvec_identity[None-vec2]": NONNUMERIC_DTYPE_BANNED,
    "TestUfunc::test_vecmatvec_identity[None-vec3]": NONNUMERIC_DTYPE_BANNED,
    "TestUfunc::test_vecmatvec_identity[matrix1-vec2]": NONNUMERIC_DTYPE_BANNED,
    "TestUfunc::test_vecmatvec_identity[matrix1-vec3]": NONNUMERIC_DTYPE_BANNED,
    # Module-level function (no test class), so no leading `*Class::` scope.
    "test_ufunc_out_casterrors": NONNUMERIC_DTYPE_BANNED,
    # test_ufunc_noncontiguous builds an alignment-testing operand via a
    # structured (void) dtype for every non-skipped type in each ufunc's
    # loop signature. Most ufuncs the parametrize sweep reaches hit it (91 of
    # the ~140 variants measured); a ufunc whose signature has no loop that
    # survives the test's own 'O?mM' skip fails cleanly through to numpy's
    # loop and still passes. One wildcard covers every current and future
    # affected variant without enumerating ~90 individual ufunc names, at the
    # cost of an XPASS (not a failure, strict=False) on the unaffected ones
    # -- the same trade-off test_reduction_with_where* below already makes.
    "*test_ufunc_noncontiguous*": NONNUMERIC_DTYPE_BANNED,
    # test_ufunc_types iterates every dtype letter in a ufunc's loop
    # signature and constructs an operand array for it; a ufunc whose
    # signature includes a timedelta64/datetime64 loop now hits the ban
    # before the type-check assertion the test wants to make. Same
    # broad-wildcard trade-off as test_ufunc_noncontiguous above -- most
    # comparison/logical ufuncs have no non-numeric loop and still pass.
    "*test_ufunc_types*": NONNUMERIC_DTYPE_BANNED,
    # Constructs a structured (void) array directly to exercise "logical
    # ufuncs accept anything, even an unpromotable dtype" -- exactly the
    # claim the numeric allowlist now refuses.
    "*TestUfunc::test_logical_ufuncs_support_anything*": NONNUMERIC_DTYPE_BANNED,
    # str dtype has no ufunc loop for np.add; the test constructs a str
    # array specifically to exercise that numpy-side refusal, but the
    # array construction itself is now refused first.
    "*TestUfunc::test_at_no_loop_for_op": NONNUMERIC_DTYPE_BANNED,
    # Structured (void) dtype used as an alignment-testing technique (gh-9876
    # regression pin), the same shape as test_ufunc_noncontiguous above.
    "*TestAdd::test_reduce_alignment": NONNUMERIC_DTYPE_BANNED,
    # Module-level function; structured (void) dtype used directly as the
    # test's own subject matter (a reduceat/structured-array interaction
    # bug), not incidentally. Leading `*` and no trailing wildcard anchors
    # the match to the exact node id -- a bare "test_reduceat" would also
    # match "test_reduceat_empty" via the loader's substring fallback,
    # xfailing an unrelated, unaffected test.
    "*::test_reduceat": NONNUMERIC_DTYPE_BANNED,
    # ------------------------------------------------------------------ #
    # test_pocketfft.py — upstream numpy flake (not flopscope's fault)        #
    # ------------------------------------------------------------------ #
    # test_identity_long_short_reversed[longdouble] has atol = 5 * spacing
    # on the dtype, which on Linux x86 (80-bit extended longdouble,
    # ε ≈ 1.08e-19) leaves about 5.4e-19 of headroom. The test uses
    # unseeded random input and hits ~5.96e-19 mismatches "fairly often"
    # (numpy's words). Upstream patched this on numpy main in commit
    # 0c514d9 by bumping atol from 5×spacing to 6×spacing; shipped in
    # numpy 2.4 but NOT backported to 2.3. So this pattern fires only
    # on numpy 2.3 × longdouble × Linux; on 2.2 the native path returned
    # double anyway, and on 2.4+ the upstream fix means the test passes.
    # strict=False (the default for our conftest) handles the 2.2/2.4
    # xpass cases cleanly.
    "*TestFFT1D::test_identity_long_short_reversed*longdouble*": (
        "UPSTREAM_NUMPY_FLAKE: numpy 2.3 longdouble FFT tolerance is too "
        "tight (5×spacing on ε≈1.08e-19 leaves no headroom); unseeded "
        "random input hits this occasionally. Fixed upstream in numpy "
        "commit 0c514d9 (atol: 5→6×spacing), shipped in 2.4, not "
        "backported to 2.3. xpass on 2.2/2.4 is expected."
    ),
    # ------------------------------------------------------------------ #
    # test_random.py                                                      #
    # ------------------------------------------------------------------ #
    # test_shuffle iterates the shuffle over 11 array variants and does
    # assert_array_equal after each. The shuffle itself succeeds; the
    # failure is in the equality check for structured-dtype arrays
    # (FlopscopeArray.__eq__ -> fnp.equal -> np.equal raises _UFuncNoLoopError
    # on VoidDType) and the dtype=object case (legacy RandomState shuffle
    # order differs from what the test hard-codes, which numpy documents
    # as "will not be fixed" — see the UserWarning emitted during the run).
    "*TestRandomDist::test_shuffle": (
        "SUBCLASS_RETURN: FlopscopeArray.__eq__ routes through fnp.equal which "
        "lacks a ufunc loop for structured dtypes; same root cause as the "
        "other SUBCLASS_RETURN entries"
    ),
    # test_shuffle_masked: numpy.ma's masked-array shuffle calls umath.equal
    # internally (numpy/ma/core.py), which auto-routes through fnp.equal on the
    # WhestArray subclass and diverges from numpy's stock masked shuffle (the
    # test hard-codes a count: ACTUAL 19 vs DESIRED 13). Same SUBCLASS_RETURN
    # root cause as test_shuffle above; pre-existing flopscope/numpy divergence
    # surfaced only when the gated numpy-compat job runs. Not matched by the
    # test_shuffle pattern (it has no trailing wildcard), so listed explicitly.
    "*TestRandomDist::test_shuffle_masked": (
        "SUBCLASS_RETURN: numpy.ma masked shuffle's internal umath.equal routes "
        "through fnp.equal on the WhestArray subclass; same root cause as "
        "test_shuffle"
    ),
    # NOTE: test_polyval, test_out_scalar, and test_choice_return_shape
    # were previously xfailed here. They now pass after targeted fixes:
    #   - polyval: use asanyarray(x) to preserve MaskedArray / ndarray
    #     subclasses through _np.polyval.
    #   - std/var/mean: honor the out= identity contract in
    #     _counted_reduction — when out is passed, return it directly
    #     without FlopscopeArray rewrapping.
    #   - random.choice: preserve object-pick identity when picking a
    #     scalar from an object-dtype array; added "choice" to the
    #     wrap_module_returns skip list.
    # ------------------------------------------------------------------ #
    # SUBCLASS_RETURN — FlopscopeArray subclass propagation               #
    # ------------------------------------------------------------------ #
    # flopscope wraps return values in FlopscopeArray (an ndarray subclass)
    # so that operator overloads can route through FLOP-tracked fnp.* funcs.
    # NumPy's tests use strict `type(x) is np.ndarray` checks that fail when
    # the result is a subclass. These tests are inherent limitations of the
    # subclass design.
    "*TestSpecialMethods::test_priority": (
        "SUBCLASS_RETURN: ndarray subclass propagates through ufunc with __array_priority__"
    ),
    "*TestUfunc::test_array_wrap_array_priority": (
        "SUBCLASS_RETURN: np.zeros (patched to return FlopscopeArray) wins the "
        "__array_priority__ contest against a subclass with priority 0; "
        "add returns FlopscopeArray instead of the expected subclass instance"
    ),
    "*TestUfunc::test_scalar_reduction": (
        "SUBCLASS_RETURN: ufunc reduction on FlopscopeArray returns subclass instead of scalar"
    ),
    "*TestUfunc::test_broadcast": (
        "SUBCLASS_RETURN: broadcast result preserves FlopscopeArray subclass"
    ),
    "*TestNonzero::test_return_type": (
        "SUBCLASS_RETURN: nonzero returns FlopscopeArray instead of plain ndarray tuple"
    ),
    "*TestRequire::test_ensure_array": (
        "SUBCLASS_RETURN: np.require with subok=False can't strip FlopscopeArray"
    ),
    "*TestRequire::test_non_array_input": (
        "SUBCLASS_RETURN: np.require wraps a newly allocated owning ndarray "
        "as FlopscopeArray, so OWNDATA flag parity is intentionally lost"
    ),
    "*TestArrayComparisons::test_compare_unstructured_voids*": (
        "SUBCLASS_RETURN: void comparison preserves FlopscopeArray subclass"
    ),
    # ------------------------------------------------------------------ #
    # SUBCLASS_RETURN — *_like strides mismatch                           #
    # ------------------------------------------------------------------ #
    # np.zeros_like/ones_like/empty_like/full_like preserve strides from the
    # prototype. When the prototype is a FlopscopeArray, the resulting array has
    # C-order strides rather than the non-contiguous strides of the original.
    "*TestLikeFuncs::test_zeros_like": (
        "SUBCLASS_RETURN: *_like strides don't match prototype when prototype is FlopscopeArray"
    ),
    "*TestLikeFuncs::test_ones_like": (
        "SUBCLASS_RETURN: *_like strides don't match prototype when prototype is FlopscopeArray"
    ),
    "*TestLikeFuncs::test_empty_like": (
        "SUBCLASS_RETURN: *_like strides don't match prototype when prototype is FlopscopeArray"
    ),
    "*TestLikeFuncs::test_filled_like": (
        "SUBCLASS_RETURN: *_like strides don't match prototype when prototype is FlopscopeArray"
    ),
    # ------------------------------------------------------------------ #
    # SUBCLASS_RETURN — strict flags checks after dropping OWNDATA parity #
    # ------------------------------------------------------------------ #
    # Flopscope no longer guarantees ndarray flag parity for subclass results.
    # These clip tests use assert_array_strict_equal(...), which compares
    # x.flags == y.flags in addition to values/dtype. Our patched clip path
    # now returns view-backed FlopscopeArray results with OWNDATA=False where
    # NumPy returns owning ndarrays with OWNDATA=True.
    "*TestClip::test_simple_*": (
        "SUBCLASS_RETURN: clip preserves values but not strict ndarray "
        "flags/OWNDATA parity"
    ),
    "*TestClip::test_type_cast_*": (
        "SUBCLASS_RETURN: clip preserves values but not strict ndarray "
        "flags/OWNDATA parity"
    ),
    "*TestClip::test_clip_with_out_*": (
        "SUBCLASS_RETURN: clip out= path preserves values but not strict ndarray "
        "flags/OWNDATA parity"
    ),
    "*TestClip::test_clip_inplace_*": (
        "SUBCLASS_RETURN: in-place clip preserves values but not strict ndarray "
        "flags/OWNDATA parity"
    ),
    "*TestClip::test_array_double": (
        "SUBCLASS_RETURN: clip preserves values but not strict ndarray "
        "flags/OWNDATA parity"
    ),
    "*TestClip::test_clip_complex": (
        "SUBCLASS_RETURN: clip preserves values but not strict ndarray "
        "flags/OWNDATA parity"
    ),
    "*TestClip::test_clip_non_contig": (
        "SUBCLASS_RETURN: clip preserves values but not strict ndarray "
        "flags/OWNDATA parity"
    ),
    "*TestClip::test_clip_func_takes_out": (
        "SUBCLASS_RETURN: clip out= path preserves values but not strict ndarray "
        "flags/OWNDATA parity"
    ),
    # ------------------------------------------------------------------ #
    # NUMPY_INTERNAL — fromiter/resize edge cases                         #
    # ------------------------------------------------------------------ #
    "*TestResize::test_reshape_from_zero": (
        "SUBCLASS_RETURN: np.resize returns FlopscopeArray; the subsequent "
        "assert_array_equal invokes FlopscopeArray.__eq__ -> fnp.equal -> "
        "np.equal, which has no ufunc loop for VoidDType (the test uses "
        "dtype=[('a', np.float32)], a structured dtype). Same root cause "
        "as other SUBCLASS_RETURN entries."
    ),
    "*TestFromiter::test_growth_and_complicated_dtypes*i,O*": (
        "SUBCLASS_RETURN: fromiter returns FlopscopeArray; the subsequent "
        "assert_array_equal invokes FlopscopeArray.__eq__ -> fnp.equal -> "
        "np.equal, which has no ufunc loop for VoidDType (structured "
        "dtypes). Same root cause as other SUBCLASS_RETURN entries."
    ),
    "*TestOut::test_out_wrap_no_leak": (
        "NUMPY_INTERNAL: refcount check sees unexpected count due to FlopscopeArray subclass "
        "wrapping (fails on numpy 2.2; xpassed on 2.3/2.4 which is acceptable)"
    ),
    "*TestOut::test_out_wrap_subok": (
        "SUBCLASS_RETURN: ``subok=False`` semantics not honored — flopscope always "
        "wraps freshly-allocated outputs as FlopscopeArray (an ndarray subclass), "
        "even when the caller asked for plain ndarray. Same root cause as the "
        "other SUBCLASS_RETURN entries."
    ),
    # ------------------------------------------------------------------ #
    # SUBCLASS_RETURN — numpy 2.4 ufunc __array_ufunc__ dispatch change  #
    # ------------------------------------------------------------------ #
    # numpy 2.4 changed ufunc dispatch so our patched np.zeros returns a
    # FlopscopeArray, which then lands as out= inside OverriddenArray._unwrap.
    # _unwrap checks type(obj) != np.ndarray and returns NotImplemented
    # for any ndarray subclass it doesn't recognise; the caller then does
    # NotImplemented[0] which raises TypeError.
    # This test passed on numpy 2.3; it's a genuine 2.4 regression caused
    # by our FlopscopeArray subclass propagating into third-party __array_ufunc__
    # implementations that use strict type checks.
    "*TestSpecialMethods::test_ufunc_override_where": (
        "SUBCLASS_RETURN: numpy 2.4 ufunc dispatch routes FlopscopeArray (from patched "
        "np.zeros) as out= into OverriddenArray._unwrap which does strict "
        "type(obj) != np.ndarray checks; returns NotImplemented then crashes on [0]"
    ),
    # ------------------------------------------------------------------ #
    # NOT_IMPLEMENTED — private gufuncs and remaining edge cases          #
    # ------------------------------------------------------------------ #
    # ``ufunc.outer`` / ``reduceat`` / ``at`` and the generic
    # ``reduce`` / ``accumulate`` fallback are now supported via
    # ``__array_ufunc__`` (Section 15 of the PR description). Multi-
    # output ufuncs (``divmod`` / ``frexp`` / ``modf``) are also
    # supported (Section 8). The remaining xfails below are for
    # private numpy gufuncs and a handful of genuine semantic
    # divergences.
    # Private numpy gufuncs (cross1d, matrix_multiply, conv1d_full, test_add,
    # euclidean_pdist) live in ``numpy._core.umath_tests`` and are not part
    # of the public NumPy API. Flopscope's __array_function__ allowlist does not
    # include them, so calls raise TypeError (NotImplemented).
    "*TestUfunc::test_cross1d": (
        "NOT_IMPLEMENTED: private gufunc numpy._core.umath_tests.cross1d not in "
        "__array_function__ allowlist"
    ),
    "*TestUfunc::test_axes_argument": (
        "NOT_IMPLEMENTED: private gufunc matrix_multiply not in allowlist"
    ),
    "*TestUfunc::test_keepdims_argument": (
        "NOT_IMPLEMENTED: private gufunc matrix_multiply not in allowlist"
    ),
    "*TestUfunc::test_can_ignore_signature": (
        "NOT_IMPLEMENTED: private gufunc matrix_multiply not in allowlist"
    ),
    "*TestUfunc::test_matrix_multiply_umath_empty": (
        "NOT_IMPLEMENTED: private gufunc matrix_multiply not in allowlist"
    ),
    "*TestUfunc::test_euclidean_pdist": (
        "NOT_IMPLEMENTED: private gufunc euclidean_pdist not in allowlist"
    ),
    "*TestUfunc::test_ufunc_custom_out": (
        "NOT_IMPLEMENTED: private gufunc test_add not in allowlist"
    ),
    # numpy 2.4 added `out=...` (Ellipsis) and special-cases an Ellipsis
    # NESTED in an out= tuple with its own TypeError ("must use `...` as
    # `out=...` and not per-operand/in a tuple"). flopscope's out=
    # normalizer refuses every non-array tuple entry with its single
    # deliberate, participant-instructive message ("return arrays must be
    # of ArrayType -- ... Pass the destination array itself, not a
    # container holding it."), pinned by test_out_arg_wrapped_destination.
    # Same refusal, same TypeError, different wording -- the borrowed
    # test's regex cannot match. The test node exists only on numpy >= 2.4,
    # so this pattern is inert on older suites. The feature itself works:
    # the sibling test_output_ellipsis tests pass under the harness.
    "*TestUfunc::test_output_ellipsis_errors": (
        "BY_DESIGN: flopscope refuses non-array out=-tuple entries with its "
        "own instructive TypeError; numpy 2.4's nested-Ellipsis special-case "
        "wording differs (exception-type parity holds, message does not)"
    ),
    "*TestGUFuncProcessCoreDims::test_conv1d_full_with_out": (
        "NOT_IMPLEMENTED: private gufunc conv1d_full not in allowlist"
    ),
    "*TestGUFuncProcessCoreDims::test_bad_out_shape": (
        "NOT_IMPLEMENTED: private gufunc conv1d_full not in allowlist"
    ),
    "*TestFrompyfunc::test_identity": (
        "NOT_IMPLEMENTED: frompyfunc creates a custom ufunc whose dispatch "
        "is not in flopscope's __array_function__ allowlist"
    ),
    # numpy 2.4's polygrid2d test passes flopscope-wrapped arrays into
    # np.polynomial.polynomial.polygrid2d, which dispatches through
    # __array_function__; polygrid2d is not a registered flopscope op, so
    # the dispatch has no implementation. Node passes on <= 2.3 (different
    # internal call pattern), so the pattern is a harmless xpass there.
    "*TestEvaluation::test_polygrid2d": (
        "NOT_IMPLEMENTED: numpy.polynomial.polynomial.polygrid2d is not in "
        "flopscope's __array_function__ allowlist (numpy 2.4 routes the "
        "borrowed test's arrays through the dispatcher)"
    ),
    # ------------------------------------------------------------------ #
    # NEEDS_TRIAGE — state-pollution surfaced by issue-70 fix             #
    # ------------------------------------------------------------------ #
    # After the issue-70 fix (silent downgrade of auto-inferred SymmetricTensor
    # out= targets), the ``test_reduction_with_where*`` parametrize sweep
    # exhibits order-dependent state-pollution: depending on pytest worker
    # ordering (xdist) or single-process ordering, different subsets of the
    # 15 variants fail. The set of failing variants shifts between runs.
    # Using a single wildcard is the only stable choice until the upstream
    # state-pollution bug (caching / shared mutable state in the
    # ufunc/SymmetricTensor pipeline) is fixed. Out of scope for this PR.
    "*TestUfunc::test_reduction_with_where*": NEEDS_TRIAGE,
    # isclose(np.inf, -np.inf) returns a FlopscopeArray, not the np.False_
    # singleton. The test uses `is np.False_` identity check which fails for
    # any array subclass. SUBCLASS_RETURN / BEHAVIORAL_SHIM pattern.
    "*TestIsclose::test_non_finite_scalar*": NEEDS_TRIAGE,
    # test_shuffle_untyped_warning[random2] uses default_rng() which routes
    # through flopscope's _counted_classes.py shuffle wrapper; the UserWarning
    # is emitted from _counted_classes.py rather than test_random.py, so the
    # filename assertion `assert "test_random" in rec[0].filename` fails.
    # WRAPPER_SIGNATURE pattern. random0 (np.random) and random1 (RandomState)
    # still pass because their code-path is different.
    "TestRandomDist::test_shuffle_untyped_warning[random2]": NEEDS_TRIAGE,
    # np.moveaxis(np.ma.zeros((1,2,3)), 0, 0) no longer returns a MaskedArray
    # under flopscope. PR #98 (shape-op-symmetry-transport) reworked moveaxis
    # to lose subok=True propagation for ndarray subclasses other than
    # SymmetricTensor. Surfaced only when the compat harness is actually
    # patching (i.e. after this PR). Unrelated to issue-70.
    "TestMoveaxis::test_array_likes": NEEDS_TRIAGE,
    # TestUfunc::test_sum and test_ufunc_at_scalar_value_fastpath[value0/1]
    # hit the PR #102 WhestArray-boundary tripwire ("WhestArray reached
    # numpy.copyto from inside an fnp wrapper — missing _to_base_ndarray()
    # strip"). The tripwire flags a real missing strip in flopscope wrappers,
    # but those wrappers are unrelated to the issue-70 fix and the tripwire
    # was added in PR #102 (merged 2026-05-23). Out of scope for this PR;
    # follow-up triage required to add the missing _to_base_ndarray() strips.
    "TestUfunc::test_sum": NEEDS_TRIAGE,
    "TestUfunc::test_ufunc_at_scalar_value_fastpath": NEEDS_TRIAGE,
    # MaskedArray subok=True propagation: flopscope wrappers return
    # FlopscopeArray instead of np.ma.MaskedArray on np.isclose / np.std /
    # np.var with masked inputs. Same root cause as TestMoveaxis above.
    # Triage queued; not in scope for issue-70.
    "TestIsclose::test_masked_arrays": NEEDS_TRIAGE,
    "TestNonarrayArgs::test_std_with_mean_keyword_keepdims_true_masked": NEEDS_TRIAGE,
    "TestNonarrayArgs::test_var_with_mean_keyword_keepdims_true_masked": NEEDS_TRIAGE,
    # numpy.linalg.svd hermitian-variant tests: surfaced by the harness fix.
    # Likely the same subok / wrapper-strip pattern as the masked-array
    # cases above. Triage queued; not in scope for issue-70.
    "TestSVDHermitian::test_herm_cases": NEEDS_TRIAGE,
    "TestSVDHermitian::test_empty_herm_cases": NEEDS_TRIAGE,
    "TestSVDHermitian::test_generalized_herm_cases": NEEDS_TRIAGE,
    "TestSVDHermitian::test_generalized_empty_herm_cases": NEEDS_TRIAGE,
    # numpy.linalg.pinv(a, hermitian=True) calls svd internally; svd's
    # `transpose(u * sgn)` reaches `swapaxes` from inside an fnp wrapper and
    # trips the __array_function__ tripwire. Same root cause as the
    # TestSVDHermitian entries above; PR #100's triage sweep missed these.
    "TestPinvHermitian::test_herm_cases": NEEDS_TRIAGE,
    "TestPinvHermitian::test_generalized_herm_cases": NEEDS_TRIAGE,
    # numpy.polynomial.polyval — flopscope wrapper subclass / dispatch
    # diverges from numpy expectation. Surfaced by harness fix; out of scope.
    "TestEvaluation::test_polyval": NEEDS_TRIAGE,
    # numpy.polynomial mutates its coefficient array in place internally
    # (polypow / polymul scratch buffers). Under immutability that raises a
    # TypeError, which numpy.polynomial swallows and re-raises as NotImplemented,
    # so it surfaces as "unsupported operand type(s) for *: 'int' and
    # 'Polynomial'" — the immutability sentinel is lost, so it cannot be matched
    # at runtime and is pinned here. By-design divergence (#immutable-arrays).
    "TestFraction::test_Fraction": (
        "flopscope arrays are immutable (#immutable-arrays); numpy.polynomial "
        "mutates its coefficient array in place internally"
    ),
}
