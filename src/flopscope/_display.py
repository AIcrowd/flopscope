"""Budget display rendering with Rich (optional) and plain-text fallback."""

from __future__ import annotations

from flopscope._budget import _snapshot_records, budget_summary_dict


def _format_flops(n: int) -> str:
    """Format a FLOP count with thousands separators."""
    return f"{n:,}"


def _pct(used: int | float, total: int | float) -> str:
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


def _namespace_label(ns: str | None) -> str:
    return ns if ns is not None else "(unlabeled)"


def _call_label(calls: int) -> str:
    return f"{calls} call{'s' if calls != 1 else ''}"


def _sorted_namespace_rows(
    by_namespace: dict[str | None, dict],
) -> list[tuple[str | None, dict]]:
    return sorted(
        by_namespace.items(),
        key=lambda item: (-item[1]["flops_used"], _namespace_label(item[0])),
    )


def _format_budget_summary_text(
    data: dict,
    *,
    by_namespace: bool = False,
    header: str = "flopscope FLOP Budget Summary",
) -> str:
    """Render structured budget data as plain text."""
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
                f"    {_namespace_label(namespace):<24} {_format_flops(bucket['flops_used']):>12}  ({_pct(bucket['flops_used'], data['flops_used']):>6})  [{_call_label(bucket['calls'])}]  Backend {bucket['flopscope_backend_time_s']:.3f}s  Overhead {bucket['flopscope_overhead_time_s']:.3f}s"
            )

    ops = data.get("operations", {})
    if ops:
        lines += ["", "  By operation:"]
        for op_name, op_info in sorted(
            ops.items(), key=lambda item: -item[1]["flop_cost"]
        ):
            lines.append(
                f"    {op_name:<20} {_format_flops(op_info['flop_cost']):>12}  ({_pct(op_info['flop_cost'], data['flops_used']):>6})  [{_call_label(op_info['calls'])}]"
            )

    wall_time = data.get("wall_time_s")
    backend_time = data.get("flopscope_backend_time_s", 0.0)
    overhead_time = data.get("flopscope_overhead_time_s", 0.0)
    residual_wall_time_s = data.get("residual_wall_time_s")
    if wall_time is not None and residual_wall_time_s is not None:
        lines += [
            "",
            f"  Total Wall Time:     {wall_time:.3f}s",
            f"  Flopscope Backend:   {backend_time:.3f}s  ({_pct(backend_time, wall_time)})",
            f"  Flopscope Overhead:  {overhead_time:.3f}s  ({_pct(overhead_time, wall_time)})",
            f"  Residual Wall Time:  {residual_wall_time_s:.3f}s  ({_pct(residual_wall_time_s, wall_time)})",
        ]

    op_backend_times = {
        op_name: op_info["flopscope_backend_time_s"]
        for op_name, op_info in ops.items()
        if op_info.get("flopscope_backend_time_s", 0.0) > 0
    }
    if backend_time > 0 and op_backend_times:
        lines += ["", "  By operation (time):"]
        for op_name, op_backend_time in sorted(
            op_backend_times.items(), key=lambda item: -item[1]
        ):
            lines.append(
                f"    {op_name:<20} {op_backend_time:.3f}s  ({_pct(op_backend_time, backend_time):>6})  [{_call_label(ops[op_name]['calls'])}]"
            )

    return "\n".join(lines)


def _plain_text_summary(by_namespace: bool = False) -> str:
    """Render a plain-text budget summary."""
    return _format_budget_summary_text(
        budget_summary_dict(by_namespace=by_namespace),
        by_namespace=by_namespace,
    )


_GLOBAL_DEFAULT_BUDGET = int(1e15)


def _is_global_default_ns(ns, ns_data: dict) -> bool:
    """Check if a namespace record is the implicit global default."""
    return ns is None and ns_data.get("flop_budget", 0) >= _GLOBAL_DEFAULT_BUDGET


def _display_totals(data: dict | None = None) -> dict:
    """Compute user-facing totals, hiding the implicit global default budget."""
    if data is None:
        data = budget_summary_dict()

    explicit_budget = 0
    explicit_used = 0
    for record in _snapshot_records():
        if record.namespace is None and record.flop_budget >= _GLOBAL_DEFAULT_BUDGET:
            explicit_used += record.flops_used
        else:
            explicit_budget += record.flop_budget
            explicit_used += record.flops_used

    has_explicit_budget = explicit_budget > 0
    if has_explicit_budget:
        return {
            "has_explicit_budget": True,
            "budget": explicit_budget,
            "used": explicit_used,
            "remaining": explicit_budget - explicit_used,
            "color": _usage_color(explicit_used, explicit_budget),
        }

    return {
        "has_explicit_budget": False,
        "budget": data["flop_budget"],
        "used": data["flops_used"],
        "remaining": data["flops_remaining"],
        "color": "green",
    }


def _rich_namespace_table(label: str, ns_data: dict, color: str):
    """Build a Rich Table for a single namespace bucket."""
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
    table.add_column("Backend", justify="right", style="dim")
    table.add_column("Overhead", justify="right", style="dim")
    table.add_column("Calls", justify="right", style="dim")

    ops = ns_data.get("operations", {})
    total = ns_data.get("flop_budget", ns_data.get("flops_used", 0))
    for op_name, op_info in sorted(ops.items(), key=lambda item: -item[1]["flop_cost"]):
        backend = op_info.get("flopscope_backend_time_s", 0.0)
        overhead = op_info.get("flopscope_overhead_time_s", 0.0)
        table.add_row(
            op_name,
            _format_flops(op_info["flop_cost"]),
            _pct(op_info["flop_cost"], total),
            f"{backend:.3f}s" if backend > 0 else "",
            f"{overhead:.3f}s" if overhead > 0 else "",
            _call_label(op_info["calls"]),
        )

    table.add_section()
    table.add_row(
        Text("Used", style="bold"),
        Text(_format_flops(ns_data.get("flops_used", 0)), style=f"bold {color}"),
        Text(_pct(ns_data.get("flops_used", 0), total), style=color),
        "",
        "",
        "",
    )
    if "flop_budget" in ns_data:
        remaining = ns_data["flop_budget"] - ns_data.get("flops_used", 0)
        table.add_row(
            Text("Remaining", style="bold"),
            Text(_format_flops(remaining), style="bold"),
            "",
            "",
            "",
            "",
        )

    return table


def _rich_totals_table(data: dict):
    from rich.table import Table
    from rich.text import Text

    table = Table(show_header=False, expand=True, padding=(0, 1), box=None)
    table.add_column("label", style="bold")
    table.add_column("value", justify="right")

    totals = _display_totals(data)
    if totals["has_explicit_budget"]:
        table.add_row("Budget", _format_flops(totals["budget"]))
    table.add_row(
        "Used",
        Text(
            (
                f"{_format_flops(totals['used'])}  ({_pct(totals['used'], totals['budget'])})"
                if totals["has_explicit_budget"]
                else _format_flops(totals["used"])
            ),
            style=totals["color"],
        ),
    )
    if totals["has_explicit_budget"]:
        table.add_row(
            "Remaining",
            f"{_format_flops(totals['remaining'])}  ({_pct(totals['remaining'], totals['budget'])})",
        )

    wall_time = data.get("wall_time_s")
    backend_time = data.get("flopscope_backend_time_s", 0.0)
    overhead_time = data.get("flopscope_overhead_time_s", 0.0)
    residual_wall_time_s = data.get("residual_wall_time_s")
    if wall_time is not None and residual_wall_time_s is not None:
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
                f"{residual_wall_time_s:.3f}s  ({_pct(residual_wall_time_s, wall_time)})",
                style="dim",
            ),
        )

    return table


def _rich_attribution_table(by_ns: dict[str | None, dict], total_flops_used: int):
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

    for namespace, bucket in _sorted_namespace_rows(by_ns):
        table.add_row(
            _namespace_label(namespace),
            _format_flops(bucket["flops_used"]),
            _pct(bucket["flops_used"], total_flops_used),
            str(bucket["calls"]),
            f"{bucket['flopscope_backend_time_s']:.3f}s",
            f"{bucket['flopscope_overhead_time_s']:.3f}s",
        )

    return table


def _rich_operations_table(ops: dict[str, dict], total_flops_used: int):
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

    for op_name, op_info in sorted(ops.items(), key=lambda item: -item[1]["flop_cost"]):
        backend = op_info.get("flopscope_backend_time_s", 0.0)
        overhead = op_info.get("flopscope_overhead_time_s", 0.0)
        table.add_row(
            op_name,
            _format_flops(op_info["flop_cost"]),
            _pct(op_info["flop_cost"], total_flops_used),
            f"{backend:.3f}s" if backend > 0 else "",
            f"{overhead:.3f}s" if overhead > 0 else "",
            _call_label(op_info["calls"]),
        )

    return table


def _rich_summary(by_namespace: bool = False):
    """Render a Rich-formatted budget summary."""
    from rich.console import Group
    from rich.panel import Panel

    data = budget_summary_dict(by_namespace=by_namespace)
    if data["flops_used"] == 0 and not data.get("operations"):
        return Panel("No budget data recorded yet.", title="flopscope Budget")

    renderables = [_rich_totals_table(data)]
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


def render_budget_summary(by_namespace: bool = False):
    """Return a Rich renderable if Rich is installed, otherwise plain text."""
    try:
        import rich  # noqa: F401

        return _rich_summary(by_namespace=by_namespace)
    except ImportError:
        return _plain_text_summary(by_namespace=by_namespace)


class _PlainTextLive:
    """Fallback live display that prints summary on exit."""

    def __init__(self, by_namespace: bool = False):
        self._by_namespace = by_namespace

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print(_plain_text_summary(by_namespace=self._by_namespace))
        return None


def budget_live(by_namespace: bool = False):
    """Return a live-updating budget summary context manager.

    Parameters
    ----------
    by_namespace : bool, optional
        If ``True``, include the namespace breakdown in the live display.
        Default ``False``.

    Returns
    -------
    object
        A context manager that refreshes the session-wide budget summary while
        the ``with`` block is active. When Rich is installed this is a
        live-rendered display; otherwise it falls back to a plain-text summary
        printed when the block exits.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>> with flops.budget_live():
    ...     with flops.BudgetContext(flop_budget=100):
    ...         _ = fnp.add(fnp.array([1.0]), fnp.array([2.0]))
    """
    try:
        from rich.live import Live

        class _RichBudgetLive:
            def __init__(self, by_namespace: bool):
                self._by_namespace = by_namespace
                self._live = None

            def __enter__(self):
                self._live = Live(
                    _rich_summary(by_namespace=self._by_namespace),
                    refresh_per_second=2,
                )
                self._live.__enter__()
                return self

            def __exit__(self, *args):
                if self._live is not None:
                    self._live.update(_rich_summary(by_namespace=self._by_namespace))
                    self._live.__exit__(*args)
                return None

        return _RichBudgetLive(by_namespace)
    except ImportError:
        return _PlainTextLive(by_namespace=by_namespace)


def budget_summary(by_namespace: bool = False):
    """Render the session-wide budget summary.

    Parameters
    ----------
    by_namespace : bool, optional
        If ``True``, include the namespace breakdown in the rendered summary.
        Default ``False``.

    Returns
    -------
    object or None
        In notebook-style environments, returns the Rich renderable or plain
        text summary object. In a standard terminal, prints the summary and
        returns ``None``.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>> with flops.BudgetContext(flop_budget=100):
    ...     _ = fnp.add(fnp.array([1.0]), fnp.array([2.0]))
    >>> flops.budget_summary()
    """
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
