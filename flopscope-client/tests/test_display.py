"""Tests for mapping-driven budget displays."""

from __future__ import annotations

import builtins
import sys
import types
from unittest.mock import MagicMock

import flopscope._dispatch as dispatch_module
import pytest

import flopscope._display as display_module
from tests.test_authoritative_budget_summary import (
    _canonical_summary,
    _display_totals,
)


@pytest.fixture(autouse=True)
def _reset_dispatch_accounting():
    dispatch_module.reset_dispatch()
    yield
    dispatch_module.reset_dispatch()


@pytest.fixture
def fake_rich(monkeypatch):
    class FakeText:
        def __init__(self, value, *, style=None):
            self.value = value
            self.style = style

    class FakeTable:
        instances = []

        def __init__(self, *args, **kwargs):
            self.title = kwargs.get("title")
            self.columns = []
            self.rows = []
            self.sections = 0
            self.__class__.instances.append(self)

        def add_column(self, label, **kwargs):
            self.columns.append(label)

        def add_row(self, *cells):
            self.rows.append(cells)

        def add_section(self):
            self.sections += 1

    class FakeGroup:
        def __init__(self, *renderables):
            self.renderables = renderables

    class FakePanel:
        def __init__(self, renderable, **kwargs):
            self.renderable = renderable
            self.title = kwargs.get("title")
            self.border_style = kwargs.get("border_style")

    class FakeLive:
        instances = []

        def __init__(self, renderable, *, refresh_per_second):
            self.renderables = [renderable]
            self.refresh_per_second = refresh_per_second
            self.entered = False
            self.exited = False
            self.exit_args = None
            self.__class__.instances.append(self)

        def __enter__(self):
            self.entered = True
            return self

        def update(self, renderable):
            self.renderables.append(renderable)

        def __exit__(self, *args):
            self.exited = True
            self.exit_args = args

    rich = types.ModuleType("rich")
    rich.__path__ = []
    modules = {
        "rich": rich,
        "rich.console": types.ModuleType("rich.console"),
        "rich.live": types.ModuleType("rich.live"),
        "rich.panel": types.ModuleType("rich.panel"),
        "rich.table": types.ModuleType("rich.table"),
        "rich.text": types.ModuleType("rich.text"),
    }
    modules["rich.console"].Group = FakeGroup
    modules["rich.live"].Live = FakeLive
    modules["rich.panel"].Panel = FakePanel
    modules["rich.table"].Table = FakeTable
    modules["rich.text"].Text = FakeText
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return types.SimpleNamespace(
        Group=FakeGroup,
        Live=FakeLive,
        Panel=FakePanel,
        Table=FakeTable,
    )


def test_budget_summary_fetches_once(monkeypatch, capsys) -> None:
    request = MagicMock(return_value=(_canonical_summary(), _display_totals()))
    monkeypatch.setattr(display_module, "_request_budget_summary", request)
    monkeypatch.setattr(display_module, "_render_data", lambda *a, **k: "rendered")

    display_module.budget_summary(by_namespace=True)

    assert capsys.readouterr().out == "rendered\n"
    request.assert_called_once_with(scope="session", by_namespace=True)


def test_render_budget_summary_fetches_once(monkeypatch) -> None:
    request = MagicMock(return_value=(_canonical_summary(), _display_totals()))
    render = MagicMock(return_value="rendered")
    monkeypatch.setattr(display_module, "_request_budget_summary", request)
    monkeypatch.setattr(display_module, "_render_data", render)

    assert display_module.render_budget_summary(by_namespace=False) == "rendered"

    request.assert_called_once_with(scope="session", by_namespace=False)
    render.assert_called_once()


def test_data_renderers_never_fetch(monkeypatch) -> None:
    pytest.importorskip("rich")
    request = MagicMock(side_effect=AssertionError("unexpected RPC"))
    monkeypatch.setattr(display_module, "_request_budget_summary", request)
    data = _canonical_summary(by_namespace=True)
    totals = _display_totals()

    display_module._plain_text_summary_from_data(data, by_namespace=True)
    display_module._rich_summary_from_data(
        data, display_totals=totals, by_namespace=True
    )

    request.assert_not_called()


def test_fake_rich_renderer_builds_canonical_tables_without_fetch(
    monkeypatch, fake_rich
) -> None:
    request = MagicMock(side_effect=AssertionError("unexpected RPC"))
    monkeypatch.setattr(display_module, "_request_budget_summary", request)

    result = display_module._render_data(
        _canonical_summary(by_namespace=True),
        display_totals=_display_totals(),
        by_namespace=True,
    )

    assert isinstance(result, fake_rich.Panel)
    assert isinstance(result.renderable, fake_rich.Group)
    totals, namespaces, operations = result.renderable.renderables
    assert [totals.title, namespaces.title, operations.title] == [
        None,
        "By namespace",
        "By operation",
    ]
    assert [row[0] for row in totals.rows] == [
        "Budget",
        "Used",
        "Remaining",
        "Total Wall Time",
        "Flopscope Backend",
        "Flopscope Overhead",
        "Residual Wall Time",
    ]
    assert namespaces.columns == [
        "Namespace",
        "FLOPs",
        "%",
        "Calls",
        "Backend",
        "Overhead",
    ]
    assert operations.columns == [
        "Operation",
        "FLOPs",
        "%",
        "Backend",
        "Overhead",
        "Calls",
    ]
    request.assert_not_called()


def test_complete_display_totals_trusts_server_metadata() -> None:
    data = _canonical_summary(flops_used=7)
    data["flop_budget"] = 10**15
    data["flops_remaining"] = 10**15 - 7

    totals = display_module._complete_display_totals(
        {
            "has_explicit_budget": True,
            "budget": 40,
            "used": 30,
            "client_context_compute_ns": None,
        },
        data,
    )

    assert totals == {
        "has_explicit_budget": True,
        "budget": 40,
        "used": 30,
        "remaining": 10,
        "color": "yellow",
    }


def test_public_summary_render_and_print_are_dispatch_overhead(
    monkeypatch,
) -> None:
    clock = {"ns": 0}
    dispatch_module.reset_dispatch()
    monkeypatch.setattr(dispatch_module, "_now_ns", lambda: clock["ns"])
    monkeypatch.setattr(
        display_module,
        "_request_budget_summary",
        lambda **_: (_canonical_summary(), _display_totals()),
    )

    def timed_render(*args, **kwargs):
        clock["ns"] += 200_000_000
        return "rendered"

    def timed_print(*args, **kwargs):
        clock["ns"] += 100_000_000

    monkeypatch.setattr(display_module, "_render_data", timed_render)
    monkeypatch.setattr(builtins, "print", timed_print)

    display_module.budget_summary()

    assert dispatch_module.total_dispatch_ns() == 300_000_000


def test_mapping_to_plain_text_is_dispatch_overhead(monkeypatch) -> None:
    clock = {"ns": 0}
    monkeypatch.setattr(dispatch_module, "_now_ns", lambda: clock["ns"])

    def timed_format(*args, **kwargs):
        clock["ns"] += 200_000_000
        return "rendered"

    monkeypatch.setattr(display_module, "_format_budget_summary_text", timed_format)

    assert (
        display_module._plain_text_summary_from_data(
            _canonical_summary(), by_namespace=False
        )
        == "rendered"
    )
    assert dispatch_module.total_dispatch_ns() == 200_000_000


def test_budget_live_accepts_namespace_flag_and_each_refresh_fetches_once(
    monkeypatch,
) -> None:
    pytest.importorskip("rich")
    request = MagicMock(return_value=(_canonical_summary(), _display_totals()))
    monkeypatch.setattr(display_module, "_request_budget_summary", request)

    live = display_module.budget_live(by_namespace=True)
    with live:
        pass

    assert request.call_count == 2
    assert all(
        call.kwargs == {"scope": "session", "by_namespace": True}
        for call in request.call_args_list
    )


def test_fake_rich_live_fetches_twice_and_propagates_namespace(
    monkeypatch, fake_rich
) -> None:
    request = MagicMock(return_value=(_canonical_summary(), _display_totals()))
    monkeypatch.setattr(display_module, "_request_budget_summary", request)
    monkeypatch.setattr(display_module, "_render_data", lambda *a, **k: "rendered")

    with display_module.budget_live(by_namespace=True):
        pass

    delegated = fake_rich.Live.instances[-1]
    assert delegated.entered is True
    assert delegated.exited is True
    assert delegated.renderables == ["rendered", "rendered"]
    assert request.call_count == 2
    assert all(
        call.kwargs == {"scope": "session", "by_namespace": True}
        for call in request.call_args_list
    )


def test_fake_rich_live_exits_when_final_refresh_fails(monkeypatch, fake_rich) -> None:
    request = MagicMock(
        side_effect=[
            (_canonical_summary(), _display_totals()),
            RuntimeError("refresh failed"),
        ]
    )
    monkeypatch.setattr(display_module, "_request_budget_summary", request)
    monkeypatch.setattr(display_module, "_render_data", lambda *a, **k: "rendered")
    live = display_module.budget_live(by_namespace=True)
    live.__enter__()
    delegated = fake_rich.Live.instances[-1]

    with pytest.raises(RuntimeError, match="refresh failed"):
        live.__exit__(None, None, None)

    assert delegated.exited is True
    assert delegated.exit_args == (None, None, None)
    assert request.call_count == 2


def test_budget_live_rich_factory_time_is_dispatch_overhead(
    monkeypatch, fake_rich
) -> None:
    clock = {"ns": 0}
    monkeypatch.setattr(dispatch_module, "_now_ns", lambda: clock["ns"])
    real_import = builtins.__import__

    def timed_import(name, *args, **kwargs):
        if name == "rich.live":
            clock["ns"] += 100_000_000
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", timed_import)

    display_module.budget_live()

    assert dispatch_module.total_dispatch_ns() == 100_000_000


def test_budget_live_plain_factory_time_is_dispatch_overhead(monkeypatch) -> None:
    clock = {"ns": 0}
    monkeypatch.setattr(dispatch_module, "_now_ns", lambda: clock["ns"])
    real_import = builtins.__import__

    def timed_import(name, *args, **kwargs):
        if name == "rich.live":
            clock["ns"] += 100_000_000
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", timed_import)

    display_module.budget_live()

    assert dispatch_module.total_dispatch_ns() == 100_000_000


def test_budget_live_plain_fallback_fetches_once_on_exit(monkeypatch, capsys) -> None:
    request = MagicMock(return_value=(_canonical_summary(), _display_totals()))
    monkeypatch.setattr(display_module, "_request_budget_summary", request)
    monkeypatch.setattr(display_module, "_render_data", lambda *a, **k: "rendered")
    real_import = builtins.__import__

    def import_without_rich_live(name, *args, **kwargs):
        if name == "rich.live":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_rich_live)

    with display_module.budget_live(by_namespace=False):
        pass

    assert capsys.readouterr().out == "rendered\n"
    request.assert_called_once_with(scope="session", by_namespace=False)


def test_budget_live_enter_and_exit_work_are_dispatch_overhead(
    monkeypatch,
) -> None:
    rich_live = pytest.importorskip("rich.live")
    clock = {"ns": 0}
    monkeypatch.setattr(dispatch_module, "_now_ns", lambda: clock["ns"])

    def advance():
        clock["ns"] += 100_000_000

    def timed_fetch(_by_namespace):
        advance()
        return "rendered"

    class TimedLive:
        def __init__(self, *args, **kwargs):
            advance()

        def __enter__(self):
            advance()
            return self

        def update(self, *args, **kwargs):
            advance()

        def __exit__(self, *args):
            advance()

    monkeypatch.setattr(display_module, "_fetch_and_render", timed_fetch)
    monkeypatch.setattr(rich_live, "Live", TimedLive)

    with display_module.budget_live():
        pass

    assert dispatch_module.total_dispatch_ns() == 600_000_000
