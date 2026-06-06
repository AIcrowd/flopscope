"""Budget display rendering with Rich (optional) and plain-text fallback."""

from __future__ import annotations

from flopscope._budget import budget_summary_dict


def _format_flops(n: int) -> str:
    """Format a FLOP count with thousands separators."""
    return f"{n:,}"


def _pct(used: int, total: int) -> str:
    """Return a percentage string."""
    if total == 0:
        return "0.0%"
    return f"{100 * used / total:.1f}%"


def _format_seconds(s: float) -> str:
    """Format a wall-time value (seconds) for the budget summary."""
    return f"{s:.3f}s"


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


def _plain_text_summary() -> str:
    """Render a plain-text budget summary across all namespaces."""
    data = budget_summary_dict(by_namespace=True)
    if data["flops_used"] == 0 and not data.get("by_namespace"):
        return "No budget data recorded yet."

    lines = [
        "flopscope FLOP Budget Summary",
        "=" * 50,
        f"  Total budget:    {_format_flops(data['flop_budget']):>20}",
        f"  Used:            {_format_flops(data['flops_used']):>20}  ({_pct(data['flops_used'], data['flop_budget'])})",
        f"  Remaining:       {_format_flops(data['flops_remaining']):>20}  ({_pct(data['flops_remaining'], data['flop_budget'])})",
    ]

    wall = data.get("wall_time_s") or 0.0
    if wall > 0:
        lines.append(f"  Wall time:       {_format_seconds(wall):>20}")
        lines.append(
            f"    backend  (compute):          {_format_seconds(data.get('flopscope_backend_time_s') or 0.0):>10}"
        )
        lines.append(
            f"    overhead (flopscope):        {_format_seconds(data.get('flopscope_overhead_time_s') or 0.0):>10}"
        )
        lines.append(
            f"    residual (your code, billed):{_format_seconds(data.get('residual_wall_time_s') or 0.0):>10}"
        )

    by_ns = data.get("by_namespace", {})
    if by_ns:
        lines.append("")
        for ns, ns_data in by_ns.items():
            label = ns if ns is not None else "(default)"
            lines.append(f"  [{label}]")
            lines.append(f"    Budget:  {_format_flops(ns_data['flop_budget']):>16}")
            lines.append(
                f"    Used:    {_format_flops(ns_data['flops_used']):>16}  ({_pct(ns_data['flops_used'], ns_data['flop_budget'])})"
            )
            ops = ns_data.get("operations", {})
            if ops:
                lines.append("    Operations:")
                for op_name, op_info in sorted(
                    ops.items(), key=lambda x: -x[1]["flop_cost"]
                ):
                    call_word = "call" if op_info["calls"] == 1 else "calls"
                    op_pct = _pct(op_info["flop_cost"], ns_data["flops_used"])
                    lines.append(
                        f"      {op_name:<20} {_format_flops(op_info['flop_cost']):>12}  ({op_pct:>6})  [{op_info['calls']} {call_word}]"
                    )
            lines.append("")

    ops = data.get("operations", {})
    if ops:
        lines.append("  All operations (session total):")
        for op_name, op_info in sorted(ops.items(), key=lambda x: -x[1]["flop_cost"]):
            call_word = "call" if op_info["calls"] == 1 else "calls"
            op_pct = _pct(op_info["flop_cost"], data["flops_used"])
            lines.append(
                f"    {op_name:<20} {_format_flops(op_info['flop_cost']):>12}  ({op_pct:>6})  [{op_info['calls']} {call_word}]"
            )

    return "\n".join(lines)


_GLOBAL_DEFAULT_BUDGET = int(1e15)


def _is_global_default_ns(ns, ns_data: dict) -> bool:
    """Check if a namespace record is the implicit global default."""
    return ns is None and ns_data.get("flop_budget", 0) >= _GLOBAL_DEFAULT_BUDGET


def _rich_namespace_table(label: str, ns_data: dict, color: str):
    """Build a Rich Table for a single namespace."""
    from rich.table import Table
    from rich.text import Text

    table = Table(
        show_header=True,
        header_style="bold",
        title_style="bold",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Operation", style="dim")
    table.add_column("FLOPs", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Calls", justify="right", style="dim")

    ops = ns_data.get("operations", {})
    for op_name, op_info in sorted(ops.items(), key=lambda x: -x[1]["flop_cost"]):
        op_pct = _pct(op_info["flop_cost"], ns_data["flops_used"])
        call_word = "call" if op_info["calls"] == 1 else "calls"
        table.add_row(
            op_name,
            _format_flops(op_info["flop_cost"]),
            op_pct,
            f"{op_info['calls']} {call_word}",
        )

    table.add_section()
    remaining = ns_data["flop_budget"] - ns_data["flops_used"]
    table.add_row(
        Text("Used", style="bold"),
        Text(_format_flops(ns_data["flops_used"]), style=f"bold {color}"),
        Text(_pct(ns_data["flops_used"], ns_data["flop_budget"]), style=color),
        "",
    )
    table.add_row(
        Text("Remaining", style="bold"),
        Text(_format_flops(remaining), style="bold"),
        "",
        "",
    )

    return table


def _rich_summary():
    """Render a Rich-formatted budget summary with nested namespace panels."""
    from rich.columns import Columns
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    data = budget_summary_dict(by_namespace=True)
    if data["flops_used"] == 0 and not data.get("by_namespace"):
        return Panel("No budget data recorded yet.", title="flopscope Budget")

    by_ns = data.get("by_namespace", {})

    # Compute display totals excluding the global default's inflated budget
    explicit_budget = 0
    explicit_used = 0
    for ns, ns_data in by_ns.items():
        if _is_global_default_ns(ns, ns_data):
            # Only count the used FLOPs, not the huge budget
            explicit_used += ns_data["flops_used"]
        else:
            explicit_budget += ns_data["flop_budget"]
            explicit_used += ns_data["flops_used"]

    has_explicit = explicit_budget > 0
    display_budget = explicit_budget if has_explicit else data["flop_budget"]
    display_used = explicit_used
    display_remaining = (
        display_budget - display_used if has_explicit else data["flops_remaining"]
    )
    color = _usage_color(display_used, display_budget) if has_explicit else "green"

    # Session totals bar
    totals = Table(show_header=False, expand=True, padding=(0, 1), box=None)
    totals.add_column("label", style="bold")
    totals.add_column("value", justify="right")
    if has_explicit:
        totals.add_row("Budget", _format_flops(display_budget))
    totals.add_row(
        "Used",
        Text(
            f"{_format_flops(display_used)}  ({_pct(display_used, display_budget) if has_explicit else ''})",
            style=color,
        ),
    )
    if has_explicit:
        totals.add_row(
            "Remaining",
            f"{_format_flops(display_remaining)}  ({_pct(display_remaining, display_budget)})",
        )

    wall = data.get("wall_time_s") or 0.0
    if wall > 0:
        totals.add_row("Wall time", _format_seconds(wall))
        totals.add_row(
            "  backend (compute)",
            _format_seconds(data.get("flopscope_backend_time_s") or 0.0),
        )
        totals.add_row(
            "  overhead (flopscope)",
            _format_seconds(data.get("flopscope_overhead_time_s") or 0.0),
        )
        totals.add_row(
            "  residual (your code, billed)",
            _format_seconds(data.get("residual_wall_time_s") or 0.0),
        )

    renderables = [totals]

    # Per-namespace panels
    if by_ns:
        ns_panels = []
        for ns, ns_data in by_ns.items():
            is_default = _is_global_default_ns(ns, ns_data)
            label = ns if ns is not None else "(unscoped)"
            ns_color = _usage_color(ns_data["flops_used"], ns_data["flop_budget"])
            ns_table = _rich_namespace_table(label, ns_data, ns_color)
            if is_default:
                subtitle = "[dim]implicit global budget[/dim]"
            else:
                subtitle = f"[dim]Budget: {_format_flops(ns_data['flop_budget'])}[/dim]"
            ns_panel = Panel(
                ns_table,
                title=f"[bold]{label}[/bold]",
                subtitle=subtitle,
                border_style=ns_color,
                expand=True,
            )
            ns_panels.append(ns_panel)

        if len(ns_panels) <= 3:
            renderables.append(Columns(ns_panels, equal=True, expand=True))
        else:
            for p in ns_panels:
                renderables.append(p)

    return Panel(
        Group(*renderables),
        title="[bold cyan]flopscope FLOP Budget Summary[/bold cyan]",
        border_style="cyan",
    )


def render_budget_summary():
    """Return a Rich renderable if Rich is installed, otherwise plain text."""
    try:
        import rich  # noqa: F401

        return _rich_summary()
    except ImportError:
        return _plain_text_summary()


class _PlainTextLive:
    """Fallback live display that prints summary on exit."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print(_plain_text_summary())
        return None


def budget_live():
    """Return a live-updating budget display context manager."""
    try:
        from rich.live import Live

        class _RichBudgetLive:
            def __init__(self):
                self._live = None

            def __enter__(self):
                self._live = Live(_rich_summary(), refresh_per_second=2)
                self._live.__enter__()
                return self

            def __exit__(self, *args):
                if self._live is not None:
                    self._live.update(_rich_summary())
                    self._live.__exit__(*args)
                return None

        return _RichBudgetLive()
    except ImportError:
        return _PlainTextLive()


def budget_summary():
    """Print or return the session-wide budget summary."""
    result = render_budget_summary()
    try:
        _ = get_ipython  # noqa: F821
        return result
    except NameError:
        if isinstance(result, str):
            print(result)
        else:
            from rich.console import Console

            Console().print(result)
        return None
