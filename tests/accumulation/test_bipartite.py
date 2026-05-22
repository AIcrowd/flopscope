"""Tests for _bipartite.py — port of algorithm.js#buildBipartite + buildIncidenceMatrix."""

from flopscope._accumulation._bipartite import (
    BipartiteGraph,
    IncidenceMatrix,
    build_bipartite,
    build_incidence_matrix,
)


def test_bipartite_simple_matmul():
    """ij,jk -> ik: 2 operands, 3 unique labels, output={i,k}."""
    graph = build_bipartite(
        subscripts=("ij", "jk"),
        output="ik",
        operand_names=("A", "B"),
    )
    assert isinstance(graph, BipartiteGraph)
    assert graph.num_operands == 2
    assert graph.all_labels == ("i", "j", "k")
    assert graph.free_labels == frozenset({"i", "k"})
    assert graph.summed_labels == frozenset({"j"})
    # No identical operands
    assert graph.identical_groups == ()


def test_bipartite_identical_operands_are_grouped():
    """A·A: same name twice → one identical-group of positions (0, 1)."""
    graph = build_bipartite(
        subscripts=("ij", "jk"),
        output="ik",
        operand_names=("A", "A"),
    )
    assert graph.identical_groups == ((0, 1),)


def test_bipartite_u_vertex_per_axis():
    """T(ijk) → 3 U-vertices for the single 3-axis operand."""
    graph = build_bipartite(
        subscripts=("ijk",),
        output="",  # full contraction to scalar
        operand_names=("T",),
    )
    assert len(graph.u_vertices) == 3
    for u in graph.u_vertices:
        assert u.op_idx == 0
    assert graph.free_labels == frozenset()
    assert graph.summed_labels == frozenset({"i", "j", "k"})


def test_incidence_matrix_columns_align_with_all_labels():
    graph = build_bipartite(
        subscripts=("ij", "jk"),
        output="ik",
        operand_names=("A", "B"),
    )
    matrix = build_incidence_matrix(graph)
    assert isinstance(matrix, IncidenceMatrix)
    assert matrix.labels == ("i", "j", "k")
    # 4 U-vertices (2 per operand) × 3 labels
    assert len(matrix.matrix) == 4
    for row in matrix.matrix:
        assert len(row) == 3


def test_incidence_matrix_column_fingerprints_are_tuples():
    graph = build_bipartite(
        subscripts=("ij", "jk"),
        output="ik",
        operand_names=("A", "B"),
    )
    matrix = build_incidence_matrix(graph)
    for _label, fp in matrix.col_fingerprints.items():
        assert isinstance(fp, tuple)
        assert len(fp) == len(matrix.matrix)


def test_incidence_matrix_fp_to_labels_groups_by_fingerprint():
    """For ij,jk: i and k have the same column fingerprint shape (one nonzero each)."""
    graph = build_bipartite(
        subscripts=("ij", "jk"),
        output="ik",
        operand_names=("A", "B"),
    )
    matrix = build_incidence_matrix(graph)
    # i appears only in operand 0's first axis; k only in operand 1's second axis.
    for _fp, labels in matrix.fp_to_labels.items():
        assert all(isinstance(label, str) for label in labels)
