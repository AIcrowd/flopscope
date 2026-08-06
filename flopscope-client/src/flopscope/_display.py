"""Budget display rendering with Rich (optional) and plain-text fallback."""

from __future__ import annotations


def _request_budget_summary(*, scope: str, by_namespace: bool) -> tuple[dict, dict]:
    """Lazily delegate so isolated formatter loading needs no client budget module."""
    from flopscope._budget import _request_budget_summary as request

    return request(scope=scope, by_namespace=by_namespace)


def _display_dispatch_span():
    """Lazily load the client-only dispatch accounting helper."""
    from flopscope._dispatch import dispatch_span

    return dispatch_span()


def _format_flops(n: int) -> str:
    """Format a FLOP count with thousands separators."""
    return f"{n:,}"


def _pct(used: int, total: int) -> str:
    """Return a percentage string."""
    if total == 0:
        return "0.0%"
    return f"{100 * used / total:.1f}%"


def _usage_color(used: int, total: int) -> str:
    """Return a Rich color name based on usage percentage."""
    if total == 0:
        return "white"
    ratio = used / total
    if ratio < 0.5:
        return "green"
    if ratio < 0.8:
        return "yellow"
    return "red"


def _namespace_label(namespace: str | None) -> str:
    return namespace if namespace is not None else "(unlabeled)"


def _call_label(calls: int) -> str:
    return f"{calls} call{'s' if calls != 1 else ''}"


def _sorted_namespace_rows(
    by_namespace: dict[str | None, dict],
) -> list[tuple[str | None, dict]]:
    return sorted(
        by_namespace.items(),
        key=lambda item: (
            -item[1]["flops_used"],
            _namespace_label(item[0]),
        ),
    )


def _format_budget_summary_text(
    data: dict,
    *,
    by_namespace: bool = False,
    header: str = "flopscope FLOP Budget Summary",
) -> str:
    if data["flops_used"] == 0 and not data.get("operations"):
        return "No budget data recorded yet."

    lines = [
        header,
        "=" * len(header),
        f"  Total budget:    {_format_flops(data['flop_budget']):>20}",
        f"  Used:            {_format_flops(data['flops_used']):>20}  ({_pct(data['flops_used'], data['flop_budget'])})",
        f"  Remaining:       {_format_flops(data['flops_remaining']):>20}  ({_pct(data['flops_remaining'], data['flop_budget'])})",
    ]
    if by_namespace and data.get("by_namespace"):
        lines += ["", "  By namespace:"]
        for namespace, bucket in _sorted_namespace_rows(data["by_namespace"]):
            lines.append(
                f"    {_namespace_label(namespace):<24} "
                f"{_format_flops(bucket['flops_used']):>12}  "
                f"({_pct(bucket['flops_used'], data['flops_used']):>6})  "
                f"[{_call_label(bucket['calls'])}]  "
                f"Backend {bucket['flopscope_backend_time_s']:.3f}s  "
                f"Overhead {bucket['flopscope_overhead_time_s']:.3f}s"
            )

    operations = data.get("operations", {})
    if operations:
        lines += ["", "  By operation:"]
        for op_name, op_info in sorted(
            operations.items(), key=lambda item: -item[1]["flop_cost"]
        ):
            lines.append(
                f"    {op_name:<20} "
                f"{_format_flops(op_info['flop_cost']):>12}  "
                f"({_pct(op_info['flop_cost'], data['flops_used']):>6})  "
                f"[{_call_label(op_info['calls'])}]"
            )

    wall_time = data.get("wall_time_s")
    backend_time = data.get("flopscope_backend_time_s", 0.0)
    overhead_time = data.get("flopscope_overhead_time_s", 0.0)
    residual_time = data.get("residual_wall_time_s")
    if wall_time is not None and residual_time is not None:
        lines += [
            "",
            f"  Total Wall Time:     {wall_time:.3f}s",
            f"  Flopscope Backend:   {backend_time:.3f}s  ({_pct(backend_time, wall_time)})",
            f"  Flopscope Overhead:  {overhead_time:.3f}s  ({_pct(overhead_time, wall_time)})",
            f"  Residual Wall Time:  {residual_time:.3f}s  ({_pct(residual_time, wall_time)})",
        ]

    op_backend_times = {
        op_name: op_info["flopscope_backend_time_s"]
        for op_name, op_info in operations.items()
        if op_info.get("flopscope_backend_time_s", 0.0) > 0
    }
    if backend_time > 0 and op_backend_times:
        lines += ["", "  By operation (time):"]
        for op_name, op_backend_time in sorted(
            op_backend_times.items(), key=lambda item: -item[1]
        ):
            lines.append(
                f"    {op_name:<20} {op_backend_time:.3f}s  "
                f"({_pct(op_backend_time, backend_time):>6})  "
                f"[{_call_label(operations[op_name]['calls'])}]"
            )
    return "\n".join(lines)


def _plain_text_summary_from_data(data: dict, *, by_namespace: bool) -> str:
    with _display_dispatch_span():
        return _format_budget_summary_text(data, by_namespace=by_namespace)


def _complete_display_totals(metadata: dict, data: dict) -> dict:
    budget = metadata["budget"]
    used = metadata["used"]
    explicit = metadata["has_explicit_budget"]
    return {
        "has_explicit_budget": explicit,
        "budget": budget,
        "used": used,
        "remaining": budget - used if explicit else data["flops_remaining"],
        "color": _usage_color(used, budget) if explicit else "green",
    }


def _rich_totals_table(data: dict, totals: dict):
    from rich.table import Table
    from rich.text import Text

    table = Table(show_header=False, expand=True, padding=(0, 1), box=None)
    table.add_column("label", style="bold")
    table.add_column("value", justify="right")
    if totals["has_explicit_budget"]:
        table.add_row("Budget", _format_flops(totals["budget"]))
    used_text = (
        f"{_format_flops(totals['used'])}  ({_pct(totals['used'], totals['budget'])})"
        if totals["has_explicit_budget"]
        else _format_flops(totals["used"])
    )
    table.add_row("Used", Text(used_text, style=totals["color"]))
    if totals["has_explicit_budget"]:
        table.add_row(
            "Remaining",
            f"{_format_flops(totals['remaining'])}  "
            f"({_pct(totals['remaining'], totals['budget'])})",
        )

    wall_time = data.get("wall_time_s")
    backend_time = data.get("flopscope_backend_time_s", 0.0)
    overhead_time = data.get("flopscope_overhead_time_s", 0.0)
    residual_time = data.get("residual_wall_time_s")
    if wall_time is not None and residual_time is not None:
        table.add_section()
        table.add_row("Total Wall Time", f"{wall_time:.3f}s")
        table.add_row(
            "Flopscope Backend",
            Text(
                f"{backend_time:.3f}s  ({_pct(backend_time, wall_time)})",
                style="dim",
            ),
        )
        table.add_row(
            "Flopscope Overhead",
            Text(
                f"{overhead_time:.3f}s  ({_pct(overhead_time, wall_time)})",
                style="dim",
            ),
        )
        table.add_row(
            "Residual Wall Time",
            Text(
                f"{residual_time:.3f}s  ({_pct(residual_time, wall_time)})",
                style="dim",
            ),
        )
    return table


def _rich_attribution_table(
    by_namespace: dict[str | None, dict], total_flops_used: int
):
    from rich.table import Table

    table = Table(
        title="By namespace",
        show_header=True,
        header_style="bold",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Namespace")
    table.add_column("FLOPs", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Calls", justify="right")
    table.add_column("Backend", justify="right")
    table.add_column("Overhead", justify="right")
    for namespace, bucket in _sorted_namespace_rows(by_namespace):
        table.add_row(
            _namespace_label(namespace),
            _format_flops(bucket["flops_used"]),
            _pct(bucket["flops_used"], total_flops_used),
            str(bucket["calls"]),
            f"{bucket['flopscope_backend_time_s']:.3f}s",
            f"{bucket['flopscope_overhead_time_s']:.3f}s",
        )
    return table


def _rich_operations_table(operations: dict[str, dict], total_used: int):
    from rich.table import Table

    table = Table(
        title="By operation",
        show_header=True,
        header_style="bold",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Operation", style="dim")
    table.add_column("FLOPs", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Backend", justify="right", style="dim")
    table.add_column("Overhead", justify="right", style="dim")
    table.add_column("Calls", justify="right", style="dim")
    for op_name, op_info in sorted(
        operations.items(), key=lambda item: -item[1]["flop_cost"]
    ):
        backend = op_info.get("flopscope_backend_time_s", 0.0)
        overhead = op_info.get("flopscope_overhead_time_s", 0.0)
        table.add_row(
            op_name,
            _format_flops(op_info["flop_cost"]),
            _pct(op_info["flop_cost"], total_used),
            f"{backend:.3f}s" if backend > 0 else "",
            f"{overhead:.3f}s" if overhead > 0 else "",
            _call_label(op_info["calls"]),
        )
    return table


def _rich_summary_from_data(
    data: dict,
    *,
    display_totals: dict,
    by_namespace: bool,
):
    from rich.console import Group
    from rich.panel import Panel

    with _display_dispatch_span():
        if data["flops_used"] == 0 and not data.get("operations"):
            return Panel("No budget data recorded yet.", title="flopscope Budget")
        if "color" not in display_totals:
            display_totals = _complete_display_totals(display_totals, data)
        renderables = [_rich_totals_table(data, display_totals)]
        if by_namespace and data.get("by_namespace"):
            renderables.append(
                _rich_attribution_table(data["by_namespace"], data["flops_used"])
            )
        if data.get("operations"):
            renderables.append(
                _rich_operations_table(data["operations"], data["flops_used"])
            )
        return Panel(
            Group(*renderables),
            title="[bold cyan]flopscope FLOP Budget Summary[/bold cyan]",
            border_style="cyan",
        )


def _render_data(
    data: dict,
    *,
    display_totals: dict,
    by_namespace: bool,
):
    with _display_dispatch_span():
        try:
            import rich  # noqa: F401
        except ImportError:
            return _plain_text_summary_from_data(data, by_namespace=by_namespace)
        return _rich_summary_from_data(
            data,
            display_totals=_complete_display_totals(display_totals, data),
            by_namespace=by_namespace,
        )


def render_budget_summary(by_namespace: bool = False):
    """Return a Rich renderable if Rich is installed, otherwise plain text."""
    with _display_dispatch_span():
        data, totals = _request_budget_summary(
            scope="session", by_namespace=by_namespace
        )
        return _render_data(
            data,
            display_totals=totals,
            by_namespace=by_namespace,
        )


def _fetch_and_render(by_namespace: bool):
    with _display_dispatch_span():
        data, totals = _request_budget_summary(
            scope="session", by_namespace=by_namespace
        )
        return _render_data(
            data,
            display_totals=totals,
            by_namespace=by_namespace,
        )


def budget_live(by_namespace: bool = False):
    """Return a live-updating budget display context manager."""
    with _display_dispatch_span():
        try:
            from rich.live import Live

            class _RichBudgetLive:
                def __init__(self, include_namespaces: bool):
                    self._by_namespace = include_namespaces
                    self._live = None

                def __enter__(self):
                    with _display_dispatch_span():
                        self._live = Live(
                            _fetch_and_render(self._by_namespace),
                            refresh_per_second=2,
                        )
                        self._live.__enter__()
                    return self

                def __exit__(self, *args):
                    if self._live is not None:
                        with _display_dispatch_span():
                            try:
                                self._live.update(_fetch_and_render(self._by_namespace))
                            finally:
                                self._live.__exit__(*args)
                    return None

            return _RichBudgetLive(by_namespace)
        except ImportError:

            class _PlainTextLive:
                def __init__(self, include_namespaces: bool):
                    self._by_namespace = include_namespaces

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    with _display_dispatch_span():
                        print(_fetch_and_render(self._by_namespace))
                    return None

            return _PlainTextLive(by_namespace)


def budget_summary(by_namespace: bool = False):
    """Print or return the session-wide budget summary."""
    with _display_dispatch_span():
        result = render_budget_summary(by_namespace=by_namespace)
        try:
            _ = get_ipython  # type: ignore[name-defined]  # noqa: F821
            return result
        except NameError:
            if isinstance(result, str):
                print(result)
            else:
                from rich.console import Console

                Console().print(result)
            return None
