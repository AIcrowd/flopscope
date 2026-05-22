"""Tests for new budget settings: partition_budget, dimino_budget."""

import pytest

from flopscope._config import get_setting, set_setting


def test_partition_budget_default_is_100k():
    assert get_setting("partition_budget") == 100_000


def test_dimino_budget_default_is_500k():
    assert get_setting("dimino_budget") == 500_000


def test_partition_budget_can_be_overridden():
    original = get_setting("partition_budget")
    try:
        set_setting("partition_budget", 50_000)
        assert get_setting("partition_budget") == 50_000
    finally:
        set_setting("partition_budget", original)


def test_dimino_budget_can_be_overridden():
    original = get_setting("dimino_budget")
    try:
        set_setting("dimino_budget", 1_000_000)
        assert get_setting("dimino_budget") == 1_000_000
    finally:
        set_setting("dimino_budget", original)


def test_partition_budget_rejects_negative():
    original = get_setting("partition_budget")
    try:
        with pytest.raises((ValueError, TypeError)):
            set_setting("partition_budget", -1)
    finally:
        set_setting("partition_budget", original)


def test_dimino_budget_rejects_negative():
    original = get_setting("dimino_budget")
    try:
        with pytest.raises((ValueError, TypeError)):
            set_setting("dimino_budget", -1)
    finally:
        set_setting("dimino_budget", original)


def test_partition_budget_zero_is_valid():
    """budget=0 forces fallback for any non-trivial component — useful escape hatch."""
    original = get_setting("partition_budget")
    try:
        set_setting("partition_budget", 0)
        assert get_setting("partition_budget") == 0
    finally:
        set_setting("partition_budget", original)
