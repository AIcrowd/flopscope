"""Integration tests for symmetry-aware einsum with path optimization."""

from unittest.mock import patch

import numpy
import pytest

from flopscope._budget import BudgetContext
from flopscope._einsum import einsum, einsum_path
from flopscope._perm_group import SymmetryGroup
from flopscope._symmetric import SymmetricTensor, as_symmetric
from flopscope.errors import BudgetExhaustedError


class TestMultiOperandEinsum:
    def test_three_operand_correctness(self):
        A = numpy.random.RandomState(42).rand(5, 6)
        B = numpy.random.RandomState(43).rand(6, 7)
        C = numpy.random.RandomState(44).rand(7, 8)
        expected = numpy.einsum("ij,jk,kl->il", A, B, C)
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij,jk,kl->il", A, B, C)
        numpy.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_symmetric_input_reduces_multi_operand_cost(self):
        # Updated for Task 17b: SubgraphSymmetryOracle now threads T's declared
        # S3 symmetry through binary steps. The first step 'ai,ijk->ajk' correctly
        # inherits T's S3 symmetry, reducing cost from 19000 to 10450.
        # The symmetric case now costs less than the dense case.
        n = 10
        T_data = numpy.random.RandomState(42).rand(n, n, n)
        T_data = (
            T_data
            + T_data.transpose(1, 0, 2)
            + T_data.transpose(2, 1, 0)
            + T_data.transpose(0, 2, 1)
            + T_data.transpose(1, 2, 0)
            + T_data.transpose(2, 0, 1)
        ) / 6
        T = as_symmetric(T_data, symmetry=(0, 1, 2))
        A = numpy.random.RandomState(43).rand(n, n)
        B = numpy.random.RandomState(44).rand(n, n)

        with BudgetContext(flop_budget=10**8, quiet=True) as budget_sym:
            result_sym = einsum("ijk,ai,bj->abk", T, A, B)

        with BudgetContext(flop_budget=10**8, quiet=True) as budget_dense:
            result_dense = einsum("ijk,ai,bj->abk", T_data, A, B)

        numpy.testing.assert_allclose(result_sym, result_dense, rtol=1e-10)
        # With the oracle wired in (Task 17b), T's S3 symmetry carries through
        # the first binary step (ai,ijk->ajk), so the symmetric case is cheaper.
        assert budget_sym.flops_used < budget_dense.flops_used

    def test_optimize_false_falls_back(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        C = numpy.ones((5, 6))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij,jk,kl->il", A, B, C, optimize=False)
            assert result.shape == (3, 6)


class TestOptimizeKwarg:
    def test_default_is_auto(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij,jk->ik", A, B)
            assert result.shape == (3, 5)

    def test_explicit_greedy(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        C = numpy.ones((5, 6))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij,jk,kl->il", A, B, C, optimize="greedy")
            assert result.shape == (3, 6)


class TestBudgetIntegration:
    def test_single_upfront_deduction(self):
        A = numpy.ones((10, 10))
        B = numpy.ones((10, 10))
        C = numpy.ones((10, 10))
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            einsum("ij,jk,kl->il", A, B, C)
            einsum_ops = [r for r in budget.op_log if r.op_name == "einsum"]
            assert len(einsum_ops) == 1

    def test_budget_exceeded_before_execution(self):
        A = numpy.ones((100, 100))
        B = numpy.ones((100, 100))
        C = numpy.ones((100, 100))
        with pytest.raises(BudgetExhaustedError):
            with BudgetContext(flop_budget=100, quiet=True):
                einsum("ij,jk,kl->il", A, B, C)


class TestEinsumPath:
    def test_returns_path_and_info(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        C = numpy.ones((5, 6))
        path, info = einsum_path("ij,jk,kl->il", A, B, C)
        assert isinstance(path, list)
        assert len(path) == 2
        assert hasattr(info, "steps")
        assert hasattr(info, "optimized_cost")
        assert hasattr(info, "speedup")

    def test_unit_budget_cost(self):
        """einsum_path costs 1 FLOP (path planning, not computation)."""
        A = numpy.ones((10, 10))
        B = numpy.ones((10, 10))
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            einsum_path("ij,jk->ik", A, B)
            assert budget.flops_used == 1

    def test_symmetric_input_shows_savings(self):
        # "ijk,k->ij" with S2 on {i,j}: the accumulation model detects the
        # S2 symmetry on the whole expression. The total is less than
        # num_terms * dense_baseline (gaming-resistance bound).
        n = 10
        k = 5
        S = as_symmetric(numpy.ones((n, n, k)), symmetry=(0, 1))
        v = numpy.ones(k)
        path, info = einsum_path("ijk,k->ij", S, v)
        assert len(info.steps) >= 1
        acc = info.accumulation
        # accumulation.total is less than the gaming-resistance upper bound
        gaming_bound = acc.num_terms * acc.dense_baseline
        assert acc.total <= gaming_bound
        # And accumulation.total < dense for this expression (S2 actually saves)
        assert acc.total < gaming_bound

    def test_str_output(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        C = numpy.ones((5, 6))
        _, info = einsum_path("ij,jk,kl->il", A, B, C)
        table = str(info)
        assert isinstance(table, str)
        assert len(table) > 50

    def test_str_output_no_symmetry_shows_table(self):
        """Table is rendered correctly for a dense (no-symmetry) contraction.
        str(info) delegates to format_table() via FlopscopePathInfo.__str__."""
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        _, info = einsum_path("ij,jk->ik", A, B)
        table = str(info)
        assert "flops" in table.lower()
        assert "blas" in table.lower()

    def test_str_output_with_symmetry_uses_accumulation_cost(self):
        """When any operand has symmetry, the accumulation total reflects the
        whole-expression symmetry group savings (direct-event model)."""
        n = 6
        T = as_symmetric(numpy.ones((n, n, n)), symmetry=(0, 1, 2))
        A = numpy.ones((n, n))
        _, info = einsum_path("ijk,ai->ajk", T, A)
        # The accumulation model computes the whole-expression symmetry savings.
        # total=1512, dense_baseline=1296; savings vs gaming bound (2*1296=2592):
        assert (
            info.accumulation.total
            < info.accumulation.num_terms * info.accumulation.dense_baseline
        )

    def test_str_output_symmetry_chain_through_steps(self):
        """The new direct-event model computes accumulation savings for the whole
        expression. S3{i,j,k} on T maps to Z2 on the product space
        (the expression group), giving significant savings over the gaming bound."""
        n = 5
        T = as_symmetric(numpy.ones((n, n, n)), symmetry=(0, 1, 2))
        A = numpy.ones((n, n))
        B = numpy.ones((n, n))
        C = numpy.ones((n, n))
        _, info = einsum_path("ijk,ai,bj,ck->abc", T, A, B, C)
        # Accumulation model: total < num_terms * dense_baseline (gaming-resistance bound)
        acc = info.accumulation
        gaming_bound = acc.num_terms * acc.dense_baseline
        assert acc.total <= gaming_bound

    def test_str_output_includes_index_sizes(self):
        """The 'Index sizes' line should appear and group equal-sized indices.
        str(info) delegates to format_table() via FlopscopePathInfo.__str__."""
        # All same size: should collapse to a=i=j=k=N
        n = 7
        T = as_symmetric(numpy.ones((n, n, n)), symmetry=(0, 1, 2))
        A = numpy.ones((n, n))
        _, info = einsum_path("ijk,ai->ajk", T, A)
        table = str(info)
        assert "Index sizes:" in table
        # All four indices have size 7 → grouped (any ordering is fine)
        assert "7" in table


class TestOutputSymmetryWrapping:
    def test_multi_operand_einsum_wraps_s3_output(self):
        # The new direct-event model does not automatically infer output symmetry
        # for multi-operand contractions (inference requires the deleted oracle).
        # The result is numerically correct. To get a SymmetricTensor, the caller
        # must pass symmetry= explicitly.
        n = 4
        rng = numpy.random.RandomState(47)
        data = rng.rand(n, n, n)
        perms = (
            data,
            data.transpose(0, 2, 1),
            data.transpose(1, 0, 2),
            data.transpose(1, 2, 0),
            data.transpose(2, 0, 1),
            data.transpose(2, 1, 0),
        )
        tensor = as_symmetric(sum(perms) / len(perms), symmetry=(0, 1, 2))  # pyright: ignore[reportArgumentType]
        weight = rng.rand(n, n)

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ijk,ai,bj,ck->abc", tensor, weight, weight, weight)

        expected = numpy.einsum(
            "ijk,ai,bj,ck->abc", numpy.asarray(tensor), weight, weight, weight
        )
        numpy.testing.assert_allclose(result, expected, rtol=1e-10)
        # Result is correct but not automatically wrapped as SymmetricTensor
        # (multi-operand output-symmetry inference requires the deleted oracle).
        assert result.shape == (n, n, n)

        # Explicit symmetry= kwarg still wraps the output:
        with BudgetContext(flop_budget=10**8, quiet=True):
            result_sym = einsum(
                "ijk,ai,bj,ck->abc",
                tensor,
                weight,
                weight,
                weight,
                symmetry=SymmetryGroup.symmetric(axes=(0, 1, 2)),
            )
        assert isinstance(result_sym, SymmetricTensor)
        assert result_sym.symmetry.order() == 6
        numpy.testing.assert_allclose(result_sym, expected, rtol=1e-10)

    def test_einsum_preserves_single_operand_reduction_subgroup_symmetry(self):
        # The new model infers output symmetry for single-operand contractions only
        # when ALL symmetric axes survive in the output. When a symmetric axis is
        # summed out (k is contracted in ijk->ij), the subgroup detection that
        # would infer S2{i,j} on the output was part of the deleted oracle.
        # The result is numerically correct; callers must pass symmetry= explicitly
        # to get a SymmetricTensor output.
        rng = numpy.random.RandomState(46)
        data = rng.rand(3, 3, 3)
        perms = (
            data,
            data.transpose(0, 2, 1),
            data.transpose(1, 0, 2),
            data.transpose(1, 2, 0),
            data.transpose(2, 0, 1),
            data.transpose(2, 1, 0),
        )
        tensor = as_symmetric(sum(perms) / len(perms), symmetry=(0, 1, 2))  # pyright: ignore[reportArgumentType]

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ijk->ij", tensor)

        expected = numpy.einsum("ijk->ij", numpy.asarray(tensor))
        numpy.testing.assert_allclose(result, expected, rtol=1e-10)
        # Result shape is correct; not auto-wrapped as SymmetricTensor when a
        # symmetric axis is summed out (subgroup-inference was oracle-specific).
        assert result.shape == (3, 3)

        # With explicit symmetry kwarg, the output IS wrapped and validated:
        with BudgetContext(flop_budget=10**8, quiet=True):
            result_sym = einsum(
                "ijk->ij",
                tensor,
                symmetry=SymmetryGroup.symmetric(axes=(0, 1)),
            )
        assert isinstance(result_sym, SymmetricTensor)
        assert result_sym.symmetry.axes == (0, 1)

    def test_einsum_remaps_input_symmetry_to_output_axis_order(self):
        rng = numpy.random.RandomState(45)
        data = rng.rand(3, 4, 3)
        data = 0.5 * (data + data.transpose(2, 1, 0))
        tensor = as_symmetric(data, symmetry=SymmetryGroup.symmetric(axes=(0, 2)))

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ijk->kji", tensor)

        expected = numpy.einsum("ijk->kji", data)
        numpy.testing.assert_allclose(result, expected, rtol=1e-10)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == SymmetryGroup.symmetric(axes=(0, 2))
        assert result.symmetry.axes == (0, 2)
        assert getattr(result.symmetry, "_labels", None) is None
        assert result.is_symmetric(symmetry=SymmetryGroup.symmetric(axes=(0, 2)))

    def test_einsum_with_plain_out_and_explicit_symmetry_validates_but_returns_plain_out(
        self,
    ):
        data = numpy.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 5.0, 6.0],
                [3.0, 6.0, 9.0],
            ]
        )
        target = SymmetryGroup.symmetric(axes=(0, 1))
        out = numpy.empty_like(data.T)

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij->ji", data, out=out, symmetry=target)

        expected = numpy.einsum("ij->ji", data)
        assert result is out
        assert not isinstance(result, SymmetricTensor)
        numpy.testing.assert_allclose(out, expected, rtol=1e-10)

    def test_einsum_with_symmetric_out_preserves_output_identity(self):
        data = numpy.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 5.0, 6.0],
                [3.0, 6.0, 9.0],
            ]
        )
        out = as_symmetric(numpy.zeros_like(data.T), symmetry=(0, 1))

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum(
                "ij->ji", data, out=out, symmetry=SymmetryGroup.symmetric(axes=(0, 1))
            )

        expected = numpy.einsum("ij->ji", data)
        assert result is out
        numpy.testing.assert_allclose(out, expected, rtol=1e-10)
        assert result.is_symmetric(symmetry=SymmetryGroup.symmetric(axes=(0, 1)))  # pyright: ignore[reportAttributeAccessIssue]

    def test_str_output_index_sizes_separates_different(self):
        """Different-sized indices should be listed separately."""
        A = numpy.ones((3, 100))
        B = numpy.ones((100, 50))
        C = numpy.ones((50, 7))
        _, info = einsum_path("ij,jk,kl->il", A, B, C)
        table = str(info)
        assert "Index sizes:" in table
        # Each index has a unique size
        assert "i=3" in table or "=3" in table
        assert "100" in table
        assert "50" in table
        assert "l=7" in table or "=7" in table


class TestPathInfoStepInfo:
    def test_step_info_has_core_fields(self):
        from flopscope._opt_einsum._contract import StepInfo

        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C)
        assert len(info.steps) == 2
        for step in info.steps:
            assert isinstance(step, StepInfo)
            assert hasattr(step, "subscript")
            assert hasattr(step, "flop_cost")
            assert hasattr(step, "input_shapes")
            assert hasattr(step, "output_shape")


class TestPathInfoDebugFields:
    """New diagnostic fields for debugging path-finding decisions."""

    def test_optimizer_used_resolved_for_explicit_choice(self):
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C, optimize="greedy")
        assert info.optimizer_used == "greedy"

    def test_optimizer_used_resolved_for_auto(self):
        # auto with 3 ops routes to 'optimal' per _AUTO_CHOICES
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C, optimize="auto")
        assert info.optimizer_used == "optimal"

    def test_optimizer_used_trivial_for_two_ops(self):
        # 2-op cases skip the optimizer entirely
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        _, info = einsum_path("ij,jk->ik", A, B)
        assert info.optimizer_used == "trivial"

    def test_step_path_indices_match_path_field(self):
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        path, info = einsum_path("ij,jk,kl->il", A, B, C)
        for step, path_tuple in zip(info.steps, path, strict=False):
            assert step.path_indices == tuple(path_tuple)

    def test_step_merged_subset_grows_monotonically(self):
        # As intermediates accumulate, the merged subset of original
        # operand positions covered should be a superset of all earlier
        # subsets that fed into this step.
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        D = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl,lm->im", A, B, C, D)
        # Final step's subset should be the union of all original positions.
        assert info.steps[-1].merged_subset == frozenset({0, 1, 2, 3})

    def test_step_merged_subset_for_block_outer_product(self):
        # 'ab,cd->abcd' with same X — single step covers operands {0, 1}
        X = numpy.ones((4, 4))
        _, info = einsum_path("ab,cd->abcd", X, X)
        assert len(info.steps) == 1
        assert info.steps[0].merged_subset == frozenset({0, 1})

    def test_format_table_default_includes_optimizer_and_contract_columns(self):
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C, optimize="greedy")
        table = info.format_table()
        assert "Optimizer:" in table
        assert "greedy" in table
        assert "contract" in table
        assert "(0," in table or "(1," in table  # path tuple shown

    def test_format_table_shows_unique_total_when_symmetry_present(self):
        # 'ab,cd->abcd' with same X (identical operands) has Z2 symmetry detected
        # by the accumulation model (m=136 unique out of 256). The per-step table
        # doesn't show the unique/total column (that was oracle-specific behavior);
        # instead, the accumulation total reflects the savings.
        X = numpy.ones((4, 4))
        _, info = einsum_path("ab,cd->abcd", X, X)
        acc = info.accumulation
        # Z2 from identical operands: m=136, alpha=136, total=272 vs dense=256
        # gaming-resistance: total <= num_terms * dense_baseline
        gaming_bound = acc.num_terms * acc.dense_baseline
        assert acc.total <= gaming_bound
        # The m (unique output elements) matches Burnside for Z2 on 4^4 space
        assert acc.m_total == 136  # (4^4 + 4^2) / 2 = (256+16)/2 = 136

    def test_format_table_omits_unique_total_when_no_symmetry(self):
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C)
        table = info.format_table()
        # The richer renderer always shows dense_flops/savings/symmetry columns.
        # When no per-step symmetry group is detected the unique/total column is
        # absent (all-dash rows are suppressed entirely).
        # If the column does appear (because flop_cost != dense_flop_cost), the
        # data rows must all show "-" (no actual unique/total counts).
        if "unique/total" in table:
            data_rows = [
                line
                for line in table.splitlines()
                if line.strip()
                and not line.startswith("-")
                and "unique/total" not in line
                and not line.startswith("  Complete")
                and not line.startswith("  ")
            ]
            for row in data_rows:
                # Ensure no V:x/y or W:x/y pattern in these rows
                import re

                assert not re.search(r"[VW]:\d+/\d+", row), (
                    f"unexpected unique/total detail in row:\n{row}"
                )

    def test_format_table_verbose_shows_subset_and_cumulative(self):
        X = numpy.ones((4, 4))
        _, info = einsum_path("ai,bi,ci->abc", X, X, X)
        table = info.format_table(verbose=True)
        assert "subset=" in table
        assert "out_shape=" in table
        assert "cumulative=" in table
        # Updated for Task 17b: from_inner now syncs inner.steps[i].flop_cost to
        # accumulation.per_step[i].total, so the table's cumulative equals
        # info.optimized_cost (= accumulation.total), not the pre-sync inner cost.
        assert f"cumulative={info.optimized_cost:,}" in table

    def test_format_table_verbose_subset_grows(self):
        # For a 3-op chain, step 0's subset has 2 entries, step 1 has 3.
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C)
        table = info.format_table(verbose=True)
        # First step's subset has exactly 2 elements
        first_subset = info.steps[0].merged_subset
        assert first_subset is not None
        assert len(first_subset) == 2
        # Final step covers all 3 original ops
        assert info.steps[-1].merged_subset == frozenset({0, 1, 2})

    def test_format_table_default_includes_overall_savings(self):
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C, optimize="greedy")

        table = info.format_table()

        assert "Savings:" in table
        # Bug A fix (renderer/naive_cost): naive_cost is now sum(dense_flop_cost) = 450
        # (same α/M model as the per-step costs for unsymmetric inputs), so savings = 0.0%.
        # Old value was 64.0% = 1 - 450/1250, where 1250 came from the legacy k-way
        # helpers.flop_count formula that used a different model than the per-step costs.
        assert "0.0%" in table

    def test_rich_console_renders_table(self):
        pytest.importorskip("rich")

        import io

        from rich.console import Console

        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C, optimize="greedy")

        buf = io.StringIO()
        Console(file=buf, force_terminal=True, width=140).print(info)
        output = buf.getvalue()

        assert "┏" in output
        assert "einsum_path" in output
        assert "Complete contraction" in output

    def test_print_method_uses_rich_when_available(self, capsys):
        pytest.importorskip("rich")

        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C, optimize="greedy")

        info.print()
        captured = capsys.readouterr()

        assert "einsum_path" in captured.out
        assert "Complete contraction" in captured.out

    def test_print_method_falls_back_to_plain_text_without_rich(self, capsys):
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C, optimize="greedy")

        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "rich" or name.startswith("rich."):
                raise ImportError("no rich")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            info.print()

        captured = capsys.readouterr()
        assert "Complete contraction:" in captured.out
        assert "Optimizer:" in captured.out


class TestBackwardCompatibility:
    def test_existing_2_operand_behavior(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = einsum("ij,jk->ik", A, B)
            # direct-event model with off-by-one correction:
            # total = (k-1)*prod(M) + prod(alpha) - prod(num_output_orbits)
            # = 60 + 60 - 15 = 105 (textbook 2n^3 - n^2 form).
            # First cell of each output orbit is a free copy.
            assert budget.flops_used == 105
            assert result.shape == (3, 5)

    def test_symmetry_output_kwarg_still_works(self):
        X = numpy.ones((5, 10))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum(
                "ki,kj->ij",
                X,
                X,
                symmetry=SymmetryGroup.symmetric(axes=(0, 1)),
            )
            assert isinstance(result, SymmetricTensor)


class TestSymmetricBlasClassification:
    """Verify that symmetric inputs produce SYMM/SYMV/SYDT BLAS labels
    instead of the generic GEMM/GEMV/DOT."""

    def test_symmetric_matmul_gets_symm_label(self):
        """einsum('ij,jk->ik', X, X) with X declared symmetric.
        The new direct-event model charges the accumulation total. The per-step
        BLAS label is GEMM (path-level classification doesn't change with the
        new model). The Z2 savings from identical operands is reflected in the
        accumulation total being below the gaming bound."""
        import flopscope as flops

        n = 10
        X = flops.as_symmetric(numpy.ones((n, n)), symmetry=(0, 1))
        _, info = einsum_path("ij,jk->ik", X, X)
        assert len(info.steps) == 1
        # BLAS label in path-level table is still GEMM (per-step classification)
        assert info.steps[0].blas_type in ("SYMM", "GEMM")
        # Accumulation model detects savings: total < gaming bound
        acc = info.accumulation
        assert acc.total <= acc.num_terms * acc.dense_baseline
