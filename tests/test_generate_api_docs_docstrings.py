from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "generate_api_docs.py"

spec = importlib.util.spec_from_file_location(
    "generate_api_docs_docstrings", MODULE_PATH
)
assert spec is not None
assert spec.loader is not None
generate_api_docs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = generate_api_docs
spec.loader.exec_module(generate_api_docs)


def test_workflow_helpers_require_examples():
    required = {
        "flopscope.BudgetContext",
        "flopscope.budget",
        "flopscope.budget_live",
        "flopscope.namespace",
        "flopscope.as_symmetric",
        "flopscope.symmetrize",
        "flopscope.accounting.einsum_cost",
        "flopscope.accounting.pointwise_cost",
    }
    rules = generate_api_docs.public_doc_contract_rules()
    assert required.issubset(set(rules["require_examples_for"]))


def test_budgeting_helpers_are_in_the_required_examples_set():
    required = {
        "flopscope.BudgetContext",
        "flopscope.budget",
        "flopscope.budget_live",
        "flopscope.budget_summary",
        "flopscope.budget_summary_dict",
        "flopscope.budget_reset",
        "flopscope.namespace",
        "flopscope.configure",
        "flopscope.numpy.clear_einsum_cache",
        "flopscope.numpy.einsum_cache_info",
    }
    rules = generate_api_docs.public_doc_contract_rules()
    assert required.issubset(set(rules["require_examples_for"]))


def test_required_public_callables_must_have_parameters_and_returns():
    rules = generate_api_docs.public_doc_contract_rules()
    assert rules["require_parameters_and_returns_for_kind"] == {"function", "method"}


def test_stale_aliases_are_rejected_in_public_doc_examples():
    lines = ["import flopscope as we", "we.einsum('ij,j->i', W, x)"]
    problems = generate_api_docs.find_public_doc_contract_violations(
        import_path="flopscope.numpy.einsum",
        kind="function",
        summary="einsum summary",
        sections={"Parameters": ["x"], "Returns": ["value"], "Examples": lines},
    )
    assert any("stale alias" in problem for problem in problems)


def test_bmat_page_documents_the_flopscope_return_type_not_numpys():
    """The generator reads NumPy's docstring, so overrides must be applied here.

    ``build_structured_doc`` prefers the upstream docstring for numpy ops, so
    replacing the ``Returns`` section in the flopscope wrapper's own
    ``__doc__`` is invisible to the website. ``bmat`` returns a
    ``FlopscopeArray`` where ``numpy.bmat`` returns a ``numpy.matrix``, and
    without this the published page states NumPy's type -- the one thing on
    that page that is false.
    """
    _name, parsed, _example, _sig, _html = generate_api_docs.build_structured_doc(
        "bmat", "numpy"
    )
    assert [field.type for field in parsed.returns] == ["FlopscopeArray"]
    body = " ".join(line for field in parsed.returns for line in field.body)
    assert "Returns a matrix object" not in body, (
        "the bmat page still carries NumPy's Returns body, which promises a "
        "numpy.matrix the wrapper does not return"
    )


def test_ops_without_an_override_keep_numpys_returns_section():
    """The override is opt-in; every other page must be unaffected."""
    _name, parsed, _example, _sig, _html = generate_api_docs.build_structured_doc(
        "block", "numpy"
    )
    assert parsed.returns, "block lost its upstream Returns section"
    assert all(field.type != "FlopscopeArray" for field in parsed.returns)
