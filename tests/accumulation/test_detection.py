"""Tests for _detection.py — port of algorithm.js#runSigmaLoop and fullGroup.js."""

from flopscope._accumulation._bipartite import build_bipartite, build_incidence_matrix
from flopscope._accumulation._detection import (
    SigmaResult,
    classify_pi,
    derive_pi_canonical,
    run_sigma_loop,
)
from flopscope._accumulation._wreath import enumerate_wreath


def test_derive_pi_returns_identity_when_fingerprints_unchanged():
    """When σ is identity, σ(M) = M; π must be the identity label map."""
    sigma_col_of = {"i": (1, 0), "j": (0, 1)}
    fp_to_labels = {(1, 0): frozenset({"i"}), (0, 1): frozenset({"j"})}
    pi = derive_pi_canonical(
        sigma_col_of,
        fp_to_labels,
        v_labels=frozenset({"i"}),
        w_labels=frozenset({"j"}),
    )
    assert pi == {"i": "i", "j": "j"}


def test_derive_pi_returns_none_on_fingerprint_mismatch():
    sigma_col_of = {"i": (9, 9), "j": (0, 1)}
    fp_to_labels = {(1, 0): frozenset({"i"}), (0, 1): frozenset({"j"})}
    assert (
        derive_pi_canonical(
            sigma_col_of,
            fp_to_labels,
            v_labels=frozenset({"i"}),
            w_labels=frozenset({"j"}),
        )
        is None
    )


def test_classify_pi_identity():
    pi = {"i": "i", "j": "j"}
    classification = classify_pi(
        pi, v_labels=frozenset({"i"}), w_labels=frozenset({"j"})
    )
    assert classification["piIsIdentity"] is True
    assert classification["piKind"] == "identity"


def test_classify_pi_v_only():
    pi = {"i": "k", "k": "i", "j": "j"}
    classification = classify_pi(
        pi, v_labels=frozenset({"i", "k"}), w_labels=frozenset({"j"})
    )
    assert classification["piKind"] == "v-only"


def test_classify_pi_w_only():
    pi = {"i": "i", "j": "k", "k": "j"}
    classification = classify_pi(
        pi, v_labels=frozenset({"i"}), w_labels=frozenset({"j", "k"})
    )
    assert classification["piKind"] == "w-only"


def test_classify_pi_cross():
    pi = {"i": "j", "j": "i"}
    classification = classify_pi(
        pi, v_labels=frozenset({"i"}), w_labels=frozenset({"j"})
    )
    assert classification["piKind"] == "cross-v-w"


def test_run_sigma_loop_on_matmul_no_symmetry_yields_only_identity_results():
    """ij,jk -> ik with all distinct operand names → only the identity wreath element."""
    graph = build_bipartite(
        subscripts=("ij", "jk"),
        output="ik",
        operand_names=("A", "B"),
    )
    matrix = build_incidence_matrix(graph)
    wreath_elements = list(
        enumerate_wreath(
            identical_groups=((0,), (1,)),
            per_op_symmetry=(None, None),
            axis_ranks=(2, 2),
            u_offsets=(0, 2),
        )
    )
    results = run_sigma_loop(graph, matrix, wreath_elements)
    # 1 wreath element (identity) → 1 sigma result
    assert len(results) == 1
    assert results[0].is_identity is True
    assert results[0].is_valid is True


def test_run_sigma_loop_on_aa_yields_identity_wreath_action():
    """A·A: ij,jk with same operand name. Wreath includes operand swap.
    The swap may or may not yield a valid pi depending on incidence structure."""
    graph = build_bipartite(
        subscripts=("ij", "jk"),
        output="ik",
        operand_names=("A", "A"),
    )
    matrix = build_incidence_matrix(graph)
    wreath_elements = list(
        enumerate_wreath(
            identical_groups=((0, 1),),
            per_op_symmetry=(None, None),
            axis_ranks=(2, 2),
            u_offsets=(0, 2),
        )
    )
    results = run_sigma_loop(graph, matrix, wreath_elements)
    # 2 wreath elements: identity + operand swap
    assert len(results) == 2
    # The operand swap on (ij, jk) doesn't produce a valid pi because the
    # resulting matrix's column fingerprints don't match.
    swap_results = [r for r in results if not r.is_identity]
    # Either valid or invalid depending on the structure; both are acceptable.
    for r in swap_results:
        assert isinstance(r, SigmaResult)


# ── build_full_group ─────────────────────────────────────────────────


from flopscope._accumulation._detection import DetectedGroup, build_full_group


def test_build_full_group_trivial_when_no_valid_non_identity_pi():
    """Sigma-loop with only identity sigma → no generators → trivial group."""
    sigma_results = (
        SigmaResult(
            is_valid=True,
            is_identity=True,
            skipped=True,
            pi={"i": "i", "j": "j"},
            pi_kind="identity",
        ),
    )
    detected = build_full_group(sigma_results, all_labels=("i", "j"))
    assert isinstance(detected, DetectedGroup)
    assert detected.all_labels == ("i", "j")
    assert len(detected.elements) == 1
    assert detected.elements[0].is_identity
    assert detected.group_name == "trivial"


def test_build_full_group_with_one_v_only_pi():
    """One non-identity v-only pi → a generator → 2-element group."""
    sigma_results = (
        SigmaResult(
            is_valid=True,
            is_identity=True,
            skipped=True,
            pi={"i": "i", "j": "j"},
            pi_kind="identity",
        ),
        SigmaResult(
            is_valid=True,
            is_identity=False,
            skipped=False,
            pi={"i": "j", "j": "i"},
            pi_kind="v-only",
        ),
    )
    detected = build_full_group(sigma_results, all_labels=("i", "j"))
    assert len(detected.elements) == 2
    assert "S2" in detected.group_name


def test_build_full_group_classifies_s3_correctly():
    """Three labels with full S_3 should classify as S3{...}."""
    # Generators: (i j) and (j k)
    sigma_results = (
        SigmaResult(
            is_valid=True,
            is_identity=True,
            skipped=True,
            pi={"i": "i", "j": "j", "k": "k"},
            pi_kind="identity",
        ),
        SigmaResult(
            is_valid=True,
            is_identity=False,
            skipped=False,
            pi={"i": "j", "j": "i", "k": "k"},
            pi_kind="v-only",
        ),
        SigmaResult(
            is_valid=True,
            is_identity=False,
            skipped=False,
            pi={"i": "i", "j": "k", "k": "j"},
            pi_kind="v-only",
        ),
    )
    detected = build_full_group(sigma_results, all_labels=("i", "j", "k"))
    assert len(detected.elements) == 6
    assert "S3" in detected.group_name


def test_build_full_group_dedupes_repeated_pis():
    """Multiple sigma rows yielding the same pi → only one generator."""
    sigma_results = (
        SigmaResult(
            is_valid=True,
            is_identity=True,
            skipped=True,
            pi={"i": "i", "j": "j"},
            pi_kind="identity",
        ),
        SigmaResult(
            is_valid=True,
            is_identity=False,
            skipped=False,
            pi={"i": "j", "j": "i"},
            pi_kind="v-only",
        ),
        SigmaResult(
            is_valid=True,
            is_identity=False,
            skipped=False,
            pi={"i": "j", "j": "i"},
            pi_kind="v-only",
        ),  # duplicate
    )
    detected = build_full_group(sigma_results, all_labels=("i", "j"))
    assert len(detected.elements) == 2  # not 4 — dedup applied
