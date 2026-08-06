"""Regression coverage for the client-side budget public API."""


def test_budget_api_is_in_the_public_export_list():
    import flopscope as flops

    for name in ("BudgetSnapshot", "budget_reset", "current_budget"):
        assert name in flops.__all__
