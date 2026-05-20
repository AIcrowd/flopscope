"""Contains PathInfo, StepInfo, and build_path_info (stripped from opt_einsum contract.py).

Excluded: contract, _core_contract, ContractExpression, _einsum, _tensordot,
_transpose, backends/sharing imports, _filter_einsum_defaults,
format_const_einsum_str, shape_only.

The local contract_path() body was removed in Task 7+8; upstream opt_einsum is
used directly via __init__.py's wrapper.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from functools import cached_property
from hashlib import sha1
from typing import Any

from flopscope._perm_group import SymmetryGroup

from . import _helpers as helpers
from ._hsluv import rgb_distance_hex, rich_label_palette

__all__ = [
    "build_path_info",
    "PathInfo",
    "StepInfo",
]

_RICH_SYMMETRY_STYLES = {
    "S": "bold bright_cyan",
    "C": "bold bright_magenta",
    "D": "bold bright_yellow",
    "W": "bold bright_green",
}


@dataclass
class StepInfo:
    """Per-step diagnostics for a contraction path."""

    subscript: str
    """Einsum subscript for this step, e.g. ``"ijk,ai->ajk"``."""

    flop_cost: int
    """FLOP cost (FMA = 1 op)."""

    input_shapes: list[tuple[int, ...]]
    """Shapes of the input operands for this step."""

    output_shape: tuple[int, ...]
    """Shape of the output operand for this step."""

    input_groups: list = field(default_factory=list)
    """SymmetryGroup for each input in this step."""

    output_group: object | None = None
    """SymmetryGroup of the output, or None."""

    dense_flop_cost: int = 0
    """FLOP cost without symmetry (FMA = 1 op)."""

    symmetry_savings: float = 0.0
    """Fraction saved: ``1 - (flop_cost / dense_flop_cost)``. Zero when no symmetry."""

    inner_group: object | None = None
    """SymmetryGroup among the contracted (summed) labels, or None."""

    inner_applied: bool = False
    """Whether inner (W-side) symmetry was actually applied at this step."""

    blas_type: str | bool = False
    """BLAS classification for this step (e.g. 'GEMM', 'SYMM', False)."""

    path_indices: tuple[int, ...] = ()
    """The SSA-id contraction tuple for this step (the entry from
    ``PathInfo.path[i]``). Useful for cross-referencing the table with
    the raw path field."""

    merged_subset: frozenset[int] | None = None
    """Subset of *original* operand positions that this step's output
    intermediate covers. For step 0 contracting two original operands i
    and j, this is ``frozenset({i, j})``. For later steps it's the union
    of the subsets of all SSA inputs being contracted."""

    @property
    def flop_count(self) -> int:
        """Alias for ``flop_cost`` (adapter compatibility)."""
        return self.flop_cost


@dataclass
class PathInfo:
    """Information about a contraction path."""

    path: list[tuple[int, ...]]
    """The optimized contraction path (list of index-tuples)."""

    steps: list[StepInfo]
    """Per-step diagnostics."""

    naive_cost: int
    """Naive (single-step) FLOP cost (FMA = 1 op)."""

    optimized_cost: int
    """Sum of per-step costs (FMA = 1 op)."""

    largest_intermediate: int
    """Number of elements in the largest intermediate tensor."""

    speedup: float
    """``naive_cost / optimized_cost``."""

    input_subscripts: str = ""
    """Comma-separated input subscripts, e.g. ``"ij,jk,kl"``."""

    output_subscript: str = ""
    """Output subscript, e.g. ``"il"``."""

    size_dict: dict[str, int] = field(default_factory=dict)
    """Mapping from index label to dimension size."""

    optimizer_used: str = ""
    """Name of the path-finding function actually invoked. For ``optimize='auto'``
    or ``'auto-hq'`` this resolves to the underlying inner choice
    (e.g. ``'optimal'``, ``'branch_2'``, ``'dynamic_programming'``,
    ``'random_greedy_128'``) so users can tell which algorithm produced
    the path. For explicit choices it matches the requested name. Empty
    string for the trivial num_ops <= 2 case where no optimizer runs."""

    # Legacy fields for backward-compat with opt_einsum tests
    contraction_list: list = field(default_factory=list)
    scale_list: list[int] = field(default_factory=list)
    size_list: list[int] = field(default_factory=list)
    _oe_naive_cost: int = 0
    _oe_opt_cost: int = 0

    @property
    def opt_cost(self) -> Decimal:
        """Legacy: opt_einsum-style cost (FMA = 1 op)."""
        return Decimal(self._oe_opt_cost)

    @property
    def eq(self) -> str:
        return f"{self.input_subscripts}->{self.output_subscript}"

    @staticmethod
    def _preferred_label_style_index(label: str, total_slots: int) -> int | None:
        """Return the preferred stable palette slot for a label."""
        if not label or not label[0].isalpha():
            return None
        return int.from_bytes(sha1(label.encode("utf-8")).digest()[:2], "big") % (
            total_slots
        )

    @cached_property
    def _label_styles(self) -> dict[str, str]:
        """Assign non-colliding styles for the active labels in this expression."""
        labels = list(dict.fromkeys(ch for ch in self.eq if ch.isalpha()))
        slot_count = max(64, len(labels))
        label_styles = tuple(
            f"bold {color}" for color in rich_label_palette(slot_count)
        )
        used_slots: set[int] = set()
        styles: dict[str, str] = {}
        total_slots = len(label_styles)

        for label in labels:
            preferred = self._preferred_label_style_index(label, total_slots)
            if preferred is None:
                styles[label] = "bold"
                continue

            if not used_slots:
                used_slots.add(preferred)
                styles[label] = label_styles[preferred]
                continue

            best_slot: int | None = None
            best_score: tuple[float, int] | None = None
            for slot, style in enumerate(label_styles):
                if slot in used_slots:
                    continue

                color = style.rsplit(" ", 1)[-1]
                min_distance = min(
                    rgb_distance_hex(color, label_styles[used_slot].rsplit(" ", 1)[-1])
                    for used_slot in used_slots
                )
                circular_preference_distance = min(
                    (slot - preferred) % total_slots,
                    (preferred - slot) % total_slots,
                )
                score = (min_distance, -circular_preference_distance)
                if best_score is None or score > best_score:
                    best_score = score
                    best_slot = slot

            assert best_slot is not None
            used_slots.add(best_slot)
            styles[label] = label_styles[best_slot]

        return styles

    def _label_style(self, label: str) -> str:
        """Return the resolved style for a label within this expression."""
        return self._label_styles.get(label, "bold")

    def _style_text_charwise(self, text: str):
        from rich.text import Text

        result = Text()
        for ch in text:
            if ch.isalpha():
                result.append(ch, style=self._label_style(ch))
            elif ch in ",->[]{}()<>✓×:":
                result.append(ch, style="dim")
            else:
                result.append(ch)
        return result

    def _rich_symmetry_token_text(self, token: str):
        from rich.text import Text

        if token == "-":
            return Text("-", style="dim")
        if token in {"×", "→"}:
            return Text(token, style="dim")
        if token.startswith("PermGroup⟨"):
            return self._style_text_charwise(token)

        result = Text()
        if token.startswith("W"):
            sym_style = _RICH_SYMMETRY_STYLES["W"]
            result.append("W", style=sym_style)
            if token.startswith("W✓"):
                result.append("✓", style=sym_style)
            if ":" in token:
                result.append(":", style=sym_style)
            remainder = token.split(":", 1)[1].lstrip() if ":" in token else token[1:]
            if remainder:
                result.append(" ", style="dim")
                result.append_text(self._rich_symmetry_token_text(remainder))
            return result

        if token[0] in _RICH_SYMMETRY_STYLES and token[1:].split("{", 1)[0].isdigit():
            prefix = token[0]
            digits = []
            i = 1
            while i < len(token) and token[i].isdigit():
                digits.append(token[i])
                i += 1
            result.append(prefix, style=_RICH_SYMMETRY_STYLES[prefix])
            result.append("".join(digits), style=_RICH_SYMMETRY_STYLES[prefix])
            if i < len(token) and token[i] == "{":
                result.append("{", style="dim")
                i += 1
                while i < len(token) and token[i] != "}":
                    ch = token[i]
                    if ch.isalpha():
                        result.append(ch, style=self._label_style(ch))
                    elif ch == ",":
                        result.append(ch, style="dim")
                    else:
                        result.append(ch)
                    i += 1
                if i < len(token) and token[i] == "}":
                    result.append("}", style="dim")
                return result

        return self._style_text_charwise(token)

    def _rich_step_sym_text(self, step: StepInfo):
        from rich.text import Text

        in_parts = [self._fmt_sym(s) for s in step.input_groups]
        out_part = self._fmt_sym(step.output_group)  # type: ignore[arg-type]
        w_part = self._fmt_sym(step.inner_group)  # type: ignore[arg-type]
        if all(p == "-" for p in in_parts) and out_part == "-" and w_part == "-":
            return Text("-", style="dim")

        result = Text()
        for idx, part in enumerate(in_parts):
            if idx:
                result.append(" × ", style="dim")
            result.append_text(self._rich_symmetry_token_text(part))
        result.append(" → ", style="dim")
        result.append_text(self._rich_symmetry_token_text(out_part))
        if w_part != "-":
            result.append("  [", style="dim")
            result.append(
                "W✓" if step.inner_applied else "W", style=_RICH_SYMMETRY_STYLES["W"]
            )
            result.append(": ", style="dim")
            result.append_text(self._rich_symmetry_token_text(w_part))
            result.append("]", style="dim")
        return result

    def _rich_eq_text(self):
        """Render the full einsum expression with global label styling."""
        from rich.text import Text

        result = Text()
        prefix = "Complete contraction: "
        result.append(prefix, style="bold")
        result.append_text(self._style_text_charwise(self.eq))
        return result

    def _rich_subscript_text(self, subscript: str):
        """Render a subscript or step expression with global label styling."""
        return self._style_text_charwise(subscript)

    def _rich_index_sizes_text(self):
        """Render the index-size summary with label styling."""
        return self._style_text_charwise(self._fmt_index_sizes())

    def _fmt_overall_savings(self) -> str:
        """Format total optimized-vs-dense savings for the whole contraction."""
        if self.naive_cost <= 0:
            return "0.0%"
        return f"{1 - (self.optimized_cost / self.naive_cost):.1%}"

    def _rich_metric_pill(
        self,
        label: str,
        value: str | Any,
        *,
        highlight: bool = False,
        value_style: str | None = None,
        border_style: str | None = None,
    ):
        from rich import box
        from rich.panel import Panel
        from rich.text import Text

        resolved_value_style = value_style or ("bold cyan" if highlight else "bold")
        resolved_border_style = border_style or ("cyan" if highlight else "dim")
        body = Text()
        body.append(label, style="bold")
        body.append(": ", style="dim")
        if not value:
            body.append("-", style=resolved_value_style)
        elif isinstance(value, Text):
            if highlight:
                value = value.copy()
                value.stylize(resolved_value_style)
            body.append_text(value)
        else:
            body.append(str(value), style=resolved_value_style)
        return Panel.fit(
            body,
            box=box.ROUNDED,
            padding=(0, 1),
            border_style=resolved_border_style,
        )

    def _rich_summary_strip(self):
        from rich.columns import Columns

        pills = []
        pills.append(self._rich_metric_pill("Naive cost", f"{self.naive_cost:,}"))
        pills.append(
            self._rich_metric_pill(
                "Optimized cost", f"{self.optimized_cost:,}", highlight=True
            )
        )
        speedup_style = "bold green" if self.speedup > 1 else "bold"
        speedup_border = "green" if self.speedup > 1 else "dim"
        pills.append(
            self._rich_metric_pill(
                "Speedup",
                f"{self.speedup:.3f}x",
                value_style=speedup_style,
                border_style=speedup_border,
            )
        )
        savings_style = (
            "bold green" if self.optimized_cost < self.naive_cost else "bold"
        )
        savings_border = "green" if self.optimized_cost < self.naive_cost else "dim"
        pills.append(
            self._rich_metric_pill(
                "Savings",
                self._fmt_overall_savings(),
                value_style=savings_style,
                border_style=savings_border,
            )
        )
        pills.append(
            self._rich_metric_pill(
                "Largest intermediate", f"{self.largest_intermediate:,} elements"
            )
        )
        if self.size_dict:
            pills.append(
                self._rich_metric_pill("Index sizes", self._rich_index_sizes_text())
            )
        if self.optimizer_used:
            pills.append(self._rich_metric_pill("Optimizer", self.optimizer_used))
        return Columns(pills, expand=True, equal=False, padding=(0, 1))

    def _rich_verbose_detail_text(
        self, step: StepInfo, cumulative: int, *, step_index: int | None = None
    ):
        from rich.text import Text

        shape = (
            "(" + ",".join(str(d) for d in step.output_shape) + ")"
            if step.output_shape
            else "()"
        )
        result = Text()
        if step_index is not None:
            result.append(f"step {step_index}: ", style="dim")
        result.append("subset=", style="dim")
        result.append(self._fmt_subset(step.merged_subset), style="bold")
        result.append("\n")
        result.append("out_shape=", style="dim")
        result.append(shape, style="bold")
        result.append("\n")
        result.append("cumulative=", style="dim")
        result.append(f"{cumulative:,}", style="bold cyan")
        # NEW: per-step M / α / −O from attached accumulation.
        acc_step = getattr(step, "_acc_step", None)
        if acc_step is not None:
            m_value = acc_step.m_total
            alpha_value = acc_step.alpha or 0
            o_value = (
                acc_step.per_component[0].num_output_orbits
                if acc_step.per_component
                else 0
            )
            result.append("\n")
            result.append("M=", style="dim")
            result.append(str(m_value), style="bold")
            result.append("  α=", style="dim")
            result.append(str(alpha_value), style="bold")
            result.append("  −O=", style="dim")
            result.append(str(o_value), style="bold")
        return result

    def _fmt_index_sizes(self) -> str:
        """Format index sizes compactly. Groups indices with the same size."""
        if not self.size_dict:
            return ""
        from collections import defaultdict

        by_size: dict[int, list[str]] = defaultdict(list)
        for idx, sz in self.size_dict.items():
            by_size[sz].append(idx)
        parts = []
        for sz, idxs in sorted(by_size.items(), key=lambda kv: (-len(kv[1]), -kv[0])):
            idxs_sorted = sorted(idxs)
            parts.append(f"{'='.join(idxs_sorted)}={sz}")
        return ", ".join(parts)

    @staticmethod
    def _fmt_contract(step: StepInfo) -> str:
        """Format the path-supplied contraction tuple, e.g. '(0, 1)'."""
        if not step.path_indices:
            return "-"
        if len(step.path_indices) == 2:
            return f"({step.path_indices[0]}, {step.path_indices[1]})"
        return "(" + ",".join(str(p) for p in step.path_indices) + ")"

    @staticmethod
    def _try_named_group(k: int, order: int) -> str | None:
        """Return the named prefix (e.g. 'S3') if recognised, else None."""
        if order == 1:
            return None
        from math import factorial

        if order == factorial(k):
            return f"S{k}"
        if order == k:
            return f"C{k}"
        if order == 2 * k and k >= 3:
            return f"D{k}"
        return None

    @staticmethod
    def _fmt_generators(group: SymmetryGroup, labels: tuple) -> str:
        """Format generators in cycle notation with labels."""
        parts = []
        for gen in group.generators:
            if gen.is_identity:
                continue
            cycles = gen.cyclic_form
            if not cycles:
                continue
            perm_str = "".join(
                "(" + " ".join(labels[i] for i in cycle) + ")" for cycle in cycles
            )
            parts.append(perm_str)
        return ", ".join(parts) if parts else "e"

    def _fmt_sym(self, group: SymmetryGroup | None) -> str:
        """Format a SymmetryGroup for display."""
        if group is None:
            return "-"
        labels = group._labels or tuple(str(i) for i in range(group.degree))
        k = group.degree
        order = group.order()

        name = self._try_named_group(k, order)
        if name is not None:
            return f"{name}{{{','.join(labels)}}}"

        orbits = [orb for orb in group.orbits() if len(orb) >= 2]
        if not orbits:
            return "-"

        if len(orbits) == 1:
            orbit = orbits[0]
            moved_labels = tuple(labels[i] for i in sorted(orbit))
            mk = len(moved_labels)
            name = self._try_named_group(mk, order)
            if name is not None:
                return f"{name}{{{','.join(moved_labels)}}}"

        gen_str = self._fmt_generators(group, labels)
        return f"PermGroup⟨{gen_str}⟩"

    def _fmt_step_regime(self, step) -> str:
        """Return the regime name for a step, or '-' when unknown.

        FlopscopePathInfo.__str__ patches `_regime` per step from
        accumulation.per_step before calling format_table.
        """
        return getattr(step, "_regime", "-")

    def _fmt_step_sym(self, step: StepInfo) -> str:
        """Format inputs→output symmetry transformation for one step."""
        in_parts = [self._fmt_sym(s) for s in step.input_groups]
        out_part = self._fmt_sym(step.output_group)  # type: ignore[arg-type]
        w_part = self._fmt_sym(step.inner_group)  # type: ignore[arg-type]
        if all(p == "-" for p in in_parts) and out_part == "-" and w_part == "-":
            return ""
        result = f"{' × '.join(in_parts)} → {out_part}"
        if w_part != "-":
            w_prefix = "W✓" if step.inner_applied else "W"
            result += f"  [{w_prefix}: {w_part}]"
        return result

    def _fmt_unique_dense(self, step: StepInfo) -> str:
        """Show output and inner unique/dense element counts."""
        from math import prod

        def _unique_elements(
            indices: frozenset[str],
            size_dict: dict[str, int],
            perm_group: SymmetryGroup | None,
        ) -> int:
            """Count unique elements for a set of subscript indices under symmetry."""
            if not indices:
                return 1
            if perm_group is not None:
                labels = perm_group._labels or tuple(
                    sorted(indices)[: perm_group.degree]
                )
                pg_size_dict: dict[int, int] = {}
                accounted: set[str] = set()
                for i, lbl in enumerate(labels):
                    pg_size_dict[i] = size_dict[lbl]
                    accounted.add(lbl)
                count = perm_group.burnside_unique_count(pg_size_dict)
                for idx in indices:
                    if idx not in accounted:
                        count *= size_dict[idx]
                return count
            return prod(size_dict[i] for i in indices)

        if step.flop_cost == step.dense_flop_cost:
            return "-"

        parts: list[str] = []

        if step.output_group is not None and step.output_shape:
            out_str = step.subscript.split("->")[1] if "->" in step.subscript else ""
            out_total = prod(step.output_shape)
            out_unique = _unique_elements(
                frozenset(out_str),
                self.size_dict,
                perm_group=step.output_group,  # type: ignore[arg-type]
            )
            if out_unique != out_total:
                parts.append(f"V:{out_unique:,}/{out_total:,}")

        if step.inner_applied and step.inner_group is not None:
            lhs = (
                step.subscript.split("->")[0]
                if "->" in step.subscript
                else step.subscript
            )
            out_str = step.subscript.split("->")[1] if "->" in step.subscript else ""
            contracted = frozenset(lhs.replace(",", "")) - frozenset(out_str)
            if contracted:
                inner_total = prod(self.size_dict[c] for c in contracted)
                inner_unique = _unique_elements(
                    contracted,
                    self.size_dict,
                    perm_group=step.inner_group,  # type: ignore[arg-type]
                )
                if inner_unique != inner_total:
                    parts.append(f"W:{inner_unique:,}/{inner_total:,}")

        return " ".join(parts) if parts else "-"

    @staticmethod
    def _fmt_subset(s: frozenset[int] | None) -> str:
        if s is None:
            return "-"
        if not s:
            return "{}"
        return "{" + ",".join(str(i) for i in sorted(s)) + "}"

    def _header_lines(self) -> list[str]:
        sizes_line = self._fmt_index_sizes()
        header_lines = [
            f"  Complete contraction:  {self.eq}",
            f"      Naive cost (flopscope):  {self.naive_cost:,}",
            f"  Optimized cost (flopscope):  {self.optimized_cost:,}",
            f"                     Speedup:  {self.speedup:.3f}x",
            f"                     Savings:  {self._fmt_overall_savings()}",
            f"       Largest intermediate:  {self.largest_intermediate:,} elements",
        ]
        if sizes_line:
            header_lines.append(f"                Index sizes:  {sizes_line}")
        if self.optimizer_used:
            header_lines.append(f"                  Optimizer:  {self.optimizer_used}")
        return header_lines

    def _rich_step_table(self, verbose: bool = False):
        from rich import box
        from rich.table import Table

        any_unique = any(
            s.dense_flop_cost > 0 and s.flop_cost != s.dense_flop_cost
            for s in self.steps
        )
        any_regime = any(hasattr(s, "_regime") for s in self.steps)

        contract_width = max(
            len("contract"),
            max((len(self._fmt_contract(step)) for step in self.steps), default=0),
        )
        subscript_width = min(
            24,
            max(
                len("subscript"),
                max((len(step.subscript) for step in self.steps), default=0),
            ),
        )
        flops_width = max(
            len("flops"),
            max((len(f"{step.flop_cost:,}") for step in self.steps), default=0),
        )
        dense_width = max(
            len("dense_flops"),
            max((len(f"{step.dense_flop_cost:,}") for step in self.steps), default=0),
        )
        savings_width = max(
            len("savings"),
            max(
                (len(f"{step.symmetry_savings:0.1%}") for step in self.steps), default=0
            ),
        )
        blas_width = max(
            len("blas"),
            max(
                (
                    len(str(step.blas_type) if step.blas_type else "-")
                    for step in self.steps
                ),
                default=0,
            ),
        )
        regime_width = None
        if any_regime:
            regime_width = max(
                len("regime"),
                max(
                    (len(getattr(step, "_regime", "-")) for step in self.steps),
                    default=len("regime"),
                ),
            )
        unique_width = None
        if any_unique:
            unique_width = max(
                len("unique/total"),
                max(
                    (len(self._fmt_unique_dense(step)) for step in self.steps),
                    default=0,
                ),
            )

        table = Table(
            show_header=True,
            header_style="bold",
            expand=True,
            box=box.HEAVY,
            pad_edge=False,
            padding=(0, 1),
            collapse_padding=True,
        )
        table.add_column("step", justify="right", no_wrap=True, width=len("step"))
        table.add_column("contract", justify="left", no_wrap=True, width=contract_width)
        table.add_column("subscript", overflow="fold", width=subscript_width)
        if any_regime:
            table.add_column("regime", justify="left", no_wrap=True, width=regime_width)
        table.add_column("flops", justify="right", no_wrap=True, width=flops_width)
        table.add_column(
            "dense_flops", justify="right", no_wrap=True, width=dense_width
        )
        table.add_column("savings", justify="right", no_wrap=True, width=savings_width)
        table.add_column("blas", no_wrap=True, width=blas_width)
        if any_unique:
            table.add_column("unique/total", no_wrap=True, width=unique_width)
        table.add_column(
            "symmetry (inputs → output)",
            overflow="fold",
            min_width=len("symmetry (inputs → output)"),
            ratio=1,
        )

        cumulative = 0
        for i, step in enumerate(self.steps):
            row = [
                str(i),
                self._fmt_contract(step),
                self._rich_subscript_text(step.subscript),
            ]
            if any_regime:
                row.append(getattr(step, "_regime", "-"))
            row += [
                f"{step.flop_cost:,}",
                f"{step.dense_flop_cost:,}",
                f"{step.symmetry_savings:>7.1%}",
                str(step.blas_type) if step.blas_type else "-",
            ]
            if any_unique:
                row.append(self._fmt_unique_dense(step))
            row.append(self._rich_step_sym_text(step) or "-")
            table.add_row(*row)
            if verbose:
                cumulative += step.flop_cost
                detail_row = [""] * len(table.columns)
                detail_row[2] = self._rich_verbose_detail_text(step, cumulative)  # type: ignore[call-overload, assignment]
                detail_row[-1] = self._rich_verbose_detail_text(  # type: ignore[call-overload, assignment]
                    step, cumulative, step_index=i
                )
                table.add_row(*detail_row)

        return table

    def _rich_renderable(self, verbose: bool = False):
        from rich.console import Group
        from rich.panel import Panel

        expr = self._rich_eq_text()
        summary = self._rich_summary_strip()
        table = self._rich_step_table(verbose=verbose)
        return Panel(
            Group(expr, summary, table),
            title="[bold cyan]einsum_path[/bold cyan]",
            border_style="cyan",
        )

    def format_table(self, verbose: bool = False) -> str:
        """Render the path info as a printable table.

        Parameters
        ----------
        verbose : bool, optional
            When True, emit an additional indented details row under each
            step showing the operand subset covered by the intermediate,
            its output shape, the unique-vs-dense element counts that the
            symmetry savings derive from, and the cumulative cost so far.
            Useful for debugging why a particular step's savings are what
            they are. Default False.
        """
        sym_strs = [self._fmt_step_sym(s) for s in self.steps]
        max_sym_width = max((len(s) for s in sym_strs), default=0)
        header_lines = self._header_lines()

        # Common columns: step, contract, subscript, regime, flops, dense_flops, savings, blas
        # Plus: symmetry (when any step has symmetry) and unique/dense (when any
        # step has reduced cost).
        any_unique = any(
            s.dense_flop_cost > 0 and s.flop_cost != s.dense_flop_cost
            for s in self.steps
        )

        contract_strs = [self._fmt_contract(s) for s in self.steps]
        contract_col_width = max(
            len("contract"), max((len(c) for c in contract_strs), default=0)
        )
        unique_col_width = max(
            len("unique/total"),
            max((len(self._fmt_unique_dense(s)) for s in self.steps), default=0),
        )
        regime_strs = [self._fmt_step_regime(s) for s in self.steps]
        regime_col_width = max(
            len("regime"), max((len(r) for r in regime_strs), default=0)
        )

        # Build the header line
        cols = [
            f"{'step':>4}",
            f"{'contract':<{contract_col_width}}",
            f"{'subscript':<30}",
            f"{'regime':<{regime_col_width}}",
            f"{'flops':>14}",
            f"{'dense_flops':>14}",
            f"{'savings':>8}",
            f"{'blas':<8}",
        ]
        if any_unique:
            cols.append(f"{'unique/total':<{unique_col_width}}")
        sym_col_width = min(max(max_sym_width, len("symmetry (inputs → output)")), 60)
        cols.append(f"{'symmetry (inputs → output)':<{sym_col_width}}")

        header_row = "  ".join(cols)
        width = max(len(header_row), 84)
        lines = header_lines + ["-" * width, header_row, "-" * width]

        cumulative = 0
        for i, step in enumerate(self.steps):
            blas_label = str(step.blas_type) if step.blas_type else "-"
            row_parts = [
                f"{i:>4}",
                f"{contract_strs[i]:<{contract_col_width}}",
                f"{step.subscript:<30}",
                f"{regime_strs[i]:<{regime_col_width}}",
                f"{step.flop_cost:>14,}",
                f"{step.dense_flop_cost:>14,}",
                f"{step.symmetry_savings:>7.1%}",
                f"{blas_label:<8}",
            ]
            if any_unique:
                row_parts.append(f"{self._fmt_unique_dense(step):<{unique_col_width}}")
            sym_str = sym_strs[i] or "-"
            if len(sym_str) > sym_col_width:
                sym_str = sym_str[: sym_col_width - 1] + "…"
            row_parts.append(f"{sym_str:<{sym_col_width}}")
            lines.append("  ".join(row_parts))

            cumulative += step.flop_cost
            if verbose:
                # Indented details row: subset, out_shape, cumulative cost.
                # Aligned under the subscript column for visual clarity.
                subset_str = self._fmt_subset(step.merged_subset)
                shape_str = (
                    "(" + ",".join(str(d) for d in step.output_shape) + ")"
                    if step.output_shape
                    else "()"
                )
                detail_parts = [
                    f"subset={subset_str}",
                    f"out_shape={shape_str}",
                    f"cumulative={cumulative:,}",
                ]
                # NEW: per-step M / α / −O from attached accumulation.
                acc_step = getattr(step, "_acc_step", None)
                if acc_step is not None:
                    m_value = acc_step.m_total
                    alpha_value = acc_step.alpha or 0
                    o_value = (
                        acc_step.per_component[0].num_output_orbits
                        if acc_step.per_component
                        else 0
                    )
                    detail_parts.append(f"M={m_value} α={alpha_value} −O={o_value}")
                lines.append("        " + "  ".join(detail_parts))

        return "\n".join(lines)

    def __rich__(self):
        return self._rich_renderable()

    def print(self, verbose: bool = False) -> None:
        """Print using Rich when available, otherwise plain text.

        Notes
        -----
        Builtin ``print(info)`` still goes through ``__str__`` and remains
        the plain-text fallback. This convenience method chooses the Rich
        renderer whenever Rich is importable, including the Rich verbose
        layout when ``verbose=True``.
        """
        import builtins

        try:
            from rich.console import Console
        except ImportError:
            builtins.print(self.format_table(verbose=verbose))
            return None

        if verbose:
            Console().print(self._rich_renderable(verbose=True))
        else:
            Console().print(self)
        return None

    def __str__(self) -> str:
        return self.format_table(verbose=False)

    def __repr__(self) -> str:
        return self.__str__()


# ── per-step cost helper ───────────────────────────────────────────


def symmetric_flop_count(
    idx_contract,
    inner,
    num_terms,
    size_dict,
    *,
    input_subscripts=None,
    output_subscript=None,
    input_shapes=None,
):
    """Per-step symmetry-aware cost. Delegates to compute_accumulation_cost
    on the binary sub-expression so path-walker per-step costs match
    accumulation per-step costs by construction.

    Legacy fallback: when subscripts/shapes are not provided (older callers),
    falls back to the dense direct-event count from helpers.flop_count.
    """
    if input_subscripts is None or input_shapes is None:
        return helpers.flop_count(
            idx_contraction=idx_contract,
            inner=inner,
            num_terms=num_terms,
            size_dictionary=size_dict,
        )

    from flopscope._accumulation._cache import get_accumulation_cost_cached
    from flopscope._config import get_setting

    canonical = ",".join(input_subscripts) + "->" + (output_subscript or "")
    partition_budget = int(get_setting("partition_budget"))  # type: ignore[arg-type]
    cost = get_accumulation_cost_cached(
        canonical_subscripts=canonical,
        input_parts=tuple(input_subscripts),
        output_subscript=output_subscript or "",
        shapes=tuple(tuple(s) for s in input_shapes),
        sym_fingerprint=tuple(None for _ in input_subscripts),
        identity_pattern=None,
        partition_budget=partition_budget,
    )
    return cost.total


# ── build_path_info adapter (Task 5) ───────────────────────────────


def build_path_info(
    upstream_path,
    upstream_info,
    *,
    size_dict,
    optimizer_used: str = "",
    per_op_symmetries=None,
    identity_pattern=None,
):
    """Adapt upstream opt_einsum's PathInfo to flopscope's PathInfo.

    Per-step ``flop_cost`` is recomputed using flopscope's
    ``_helpers.flop_count`` (FMA = 1 by default; configurable via the
    ``fma_cost`` setting). ``naive_cost`` and ``optimized_cost`` are also
    recomputed from the per-step costs.

    Parameters
    ----------
    upstream_path : list[tuple[int, ...]]
        The contraction path returned by opt_einsum.contract_path.
    upstream_info : opt_einsum.contract.PathInfo
        Upstream's PathInfo with contraction_list, naive_cost, etc.
    size_dict : dict[str, int]
        Label -> dimension size mapping.
    optimizer_used : str, optional
        Name of the optimizer that produced ``upstream_path``. Propagated
        into the returned PathInfo for display. Defaults to ``''``.
    per_op_symmetries : sequence of SymmetryGroup or None, optional
        Per-operand declared symmetries (parallel to operands). When provided,
        a SubgraphSymmetryOracle is built and queried per step to populate
        ``input_groups``, ``output_group``, and ``inner_group`` on each
        StepInfo. Defaults to ``None`` (all dense, no symmetry).

    Returns
    -------
    PathInfo
        flopscope's PathInfo with FMA-aware per-step costs.
    """
    from math import prod

    # Walk the contraction list. Each entry has the shape:
    #   (idx_contract: tuple[int,...], idx_removed: frozenset[str],
    #    einsum_str: str, remaining: tuple[str,...] | None, do_blas: bool|str)
    steps_out: list[StepInfo] = []
    largest_intermediate = 0

    # Reconstruct merged_subset tracking from the path itself.
    # upstream_path[i] gives the original (pre-sort) indices for step i.
    _first_remaining = (
        upstream_info.contraction_list[0][3]
        if upstream_info.contraction_list
        and upstream_info.contraction_list[0][3] is not None
        else None
    )
    num_ops = (
        (len(_first_remaining) + 1)
        if _first_remaining is not None
        else (len(list(upstream_path)) + 1)
    )

    # ssa_to_subset tracks which original operands each SSA id covers.
    ssa_to_subset: dict[int, frozenset[int]] = {
        k: frozenset({k}) for k in range(num_ops)
    }
    ssa_ids: list[int] = list(range(num_ops))
    next_ssa = num_ops

    # Bug B fix: build a SubgraphSymmetryOracle from the declared operand
    # symmetries so per-step input_groups / output_group / inner_group are
    # populated correctly instead of always being empty / None.
    # The oracle is built once here and queried per step below.
    oracle = None
    if per_op_symmetries is not None:
        orig_input_parts = getattr(upstream_info, "input_subscripts", None)
        orig_output = getattr(upstream_info, "output_subscript", "")
        if orig_input_parts is not None:
            _orig_parts_list = orig_input_parts.split(",")
            if len(_orig_parts_list) == num_ops:
                try:
                    import numpy as _np

                    from flopscope._opt_einsum._subgraph_symmetry import (
                        SubgraphSymmetryOracle,
                    )

                    # Build dummy operands with the right shapes, then alias
                    # positions in the same identity-group to share object
                    # identity.  The oracle's Source-B (identical-operand
                    # swap) and Source-C (coordinated relabel) π-generators
                    # rely on object identity (``_dummy_ops[i] is _dummy_ops[j]``);
                    # without aliasing, residual symmetries that come from
                    # identical operands (e.g. A @ A → S₂ on output) would
                    # silently be omitted from the per-step annotation.
                    _dummy_ops: list = [
                        _np.empty(tuple(size_dict[c] for c in part))
                        for part in _orig_parts_list
                    ]
                    if identity_pattern is not None:
                        for _group in identity_pattern:
                            _canonical = _dummy_ops[_group[0]]
                            for _pos in _group[1:]:
                                _dummy_ops[_pos] = _canonical

                    def _sym_to_group_list(sym: Any, subscript: str) -> list | None:
                        """Convert per_op_symmetry entry → oracle per_op_groups list.

                        Source A generators in ``_collect_pi_permutations`` require
                        ``group._labels`` to be populated (the function short-circuits
                        on ``group._labels is None``).  User-supplied groups created
                        via ``SymmetryGroup.symmetric(axes=...)`` have ``axes`` set
                        but ``_labels`` empty, so we synthesize labels here from the
                        operand's subscript at the symmetry's axis positions.

                        We build a fresh ``SymmetryGroup`` so the user's object
                        stays untouched (the oracle is per-call and these clones
                        live only for its duration).
                        """
                        if sym is None:
                            return None
                        if not isinstance(sym, SymmetryGroup):
                            return None
                        if sym._labels is not None:
                            return [sym]
                        if sym.axes is None:
                            return [sym]
                        try:
                            labels = tuple(subscript[ax] for ax in sym.axes)
                        except (IndexError, TypeError):
                            return None
                        clone = SymmetryGroup(*sym.generators, axes=sym.axes)
                        clone._labels = labels
                        return [clone]

                    oracle = SubgraphSymmetryOracle(
                        operands=_dummy_ops,
                        subscript_parts=_orig_parts_list,
                        per_op_groups=[
                            _sym_to_group_list(s, part)
                            for s, part in zip(
                                per_op_symmetries, _orig_parts_list, strict=False
                            )
                        ],
                        output_chars=orig_output,
                    )
                except Exception:
                    oracle = None

    # SSA current-subsets list for oracle queries — parallel to ssa_ids list.
    # Each entry is the frozenset of original operand indices covered by the
    # corresponding current operand.  Starts as singletons.
    current_oracle_subsets: list[frozenset[int]] = [
        frozenset({k}) for k in range(num_ops)
    ]

    for step_idx, entry in enumerate(upstream_info.contraction_list):
        idx_removed = entry[1]  # frozenset of label chars removed (inner product)
        einsum_str = entry[2]  # e.g. "jk,ij->ik"
        do_blas = entry[4]  # BLAS classification string or False

        # The original path indices for this step (pre-sort, from upstream_path).
        original_path_tuple: tuple[int, ...] = tuple(upstream_path[step_idx])

        if "->" in einsum_str:
            lhs, rhs = einsum_str.split("->", 1)
        else:
            lhs, rhs = einsum_str, ""

        lhs_parts = lhs.split(",")
        num_terms = len(lhs_parts)

        # Reconstruct idx_contraction (set of all labels touched) from lhs
        idx_contraction: frozenset[str] = frozenset(
            c for part in lhs_parts for c in part
        )

        inner = bool(idx_removed)

        input_shapes_for_step: list[tuple[int, ...]] = [
            tuple(size_dict[c] for c in part) for part in lhs_parts
        ]
        output_shape_for_step: tuple[int, ...] = tuple(size_dict[c] for c in rhs)

        cost = symmetric_flop_count(
            idx_contraction,
            inner,
            num_terms,
            size_dict,
            input_subscripts=lhs_parts,
            output_subscript=rhs,
            input_shapes=input_shapes_for_step,
        )

        # Dense cost: what this step would cost without any symmetry reduction.
        step_dense_flop_cost = helpers.flop_count(
            idx_contraction,
            inner,
            num_terms,
            size_dict,
            input_subscripts=lhs_parts,
            output_subscript=rhs,
            input_shapes=input_shapes_for_step,
        )
        # Fraction of dense cost saved by symmetry (0.0 when no symmetry or
        # when the accumulation model costs more than the dense baseline due to
        # FMA vs. flop_count differences on this branch).
        step_symmetry_savings = (
            max(0.0, 1.0 - cost / step_dense_flop_cost)
            if step_dense_flop_cost > 0
            else 0.0
        )

        if output_shape_for_step:
            largest_intermediate = max(
                largest_intermediate, prod(output_shape_for_step)
            )

        # Reconstruct merged_subset by tracking which original operands each
        # SSA id covers. The path gives us the positions to contract.
        contract_positions = tuple(sorted(original_path_tuple, reverse=True))
        new_merged_subset: frozenset[int] = frozenset()
        for ci in contract_positions:
            if ci < len(ssa_ids):
                new_merged_subset = new_merged_subset | ssa_to_subset[ssa_ids[ci]]
        for ci in contract_positions:
            if ci < len(ssa_ids):
                ssa_ids.pop(ci)
        ssa_to_subset[next_ssa] = new_merged_subset
        ssa_ids.append(next_ssa)
        next_ssa += 1

        # Bug B fix: query the oracle for per-step symmetry groups.
        # The oracle uses merged_subsets of the step's input operands to derive
        # the V-side (output_group) and W-side (inner_group) symmetries.
        # For the input_groups list we use the per-input subset groups.
        step_input_groups: list = []
        step_output_group: object | None = None
        step_inner_group: object | None = None
        if oracle is not None:
            try:
                # Gather which oracle-tracked subsets map to each lhs input.
                # opt_einsum pops positions highest-to-lowest (contract_positions
                # is sorted descending), so the lhs subscript order matches
                # the descending-position order of the path entry.
                step_input_subsets = [
                    current_oracle_subsets[pos]
                    for pos in sorted(original_path_tuple, reverse=True)
                    if pos < len(current_oracle_subsets)
                ]
                for inp_subset in step_input_subsets:
                    ss = oracle.sym(inp_subset)
                    step_input_groups.append(ss.output)  # V-side group for this input
                # For output_group and inner_group, query the merged subset.
                merged_ss = oracle.sym(new_merged_subset)
                step_output_group = merged_ss.output
                step_inner_group = merged_ss.inner
            except Exception:
                step_input_groups = []
                step_output_group = None
                step_inner_group = None

        # Update current_oracle_subsets to mirror the SSA merge above.
        # Guard against out-of-range indices the same way the ssa_ids loop does.
        _oracle_contract_positions = tuple(
            pos
            for pos in sorted(original_path_tuple, reverse=True)
            if pos < len(current_oracle_subsets)
        )
        if _oracle_contract_positions:
            merged_oracle_subset: frozenset[int] = frozenset().union(
                *(current_oracle_subsets[pos] for pos in _oracle_contract_positions)
            )
            for pos in _oracle_contract_positions:
                current_oracle_subsets.pop(pos)
            current_oracle_subsets.append(merged_oracle_subset)
        else:
            current_oracle_subsets.append(frozenset())

        steps_out.append(
            StepInfo(
                subscript=einsum_str,
                flop_cost=cost,
                input_shapes=input_shapes_for_step,
                output_shape=output_shape_for_step,
                blas_type=do_blas,
                path_indices=original_path_tuple,
                merged_subset=new_merged_subset,
                # Diagnostic fields: dense baseline and symmetry savings.
                dense_flop_cost=step_dense_flop_cost,
                symmetry_savings=step_symmetry_savings,
                # Bug B fix: symmetry groups from oracle (empty list / None when
                # per_op_symmetries was not provided or oracle build failed).
                input_groups=step_input_groups,
                output_group=step_output_group,
                inner_group=step_inner_group,
            )
        )

    optimized_cost = sum(s.flop_cost for s in steps_out)

    # Bug A fix: naive_cost uses the same α/M model as the per-step dense_flop_cost
    # (helpers.flop_count, no symmetry), so header "Savings" and per-step "savings"
    # columns are computed from the same model.  The old approach used
    # helpers.flop_count over ALL labels as if they were contracted in one shot,
    # which can be a very different number than the sum of per-step dense costs.
    naive_cost = sum(s.dense_flop_cost for s in steps_out)

    speedup = (naive_cost / optimized_cost) if optimized_cost > 0 else 1.0

    return PathInfo(
        path=list(upstream_path),
        steps=steps_out,
        naive_cost=naive_cost,
        optimized_cost=optimized_cost,
        largest_intermediate=largest_intermediate,
        speedup=speedup,
        input_subscripts=getattr(upstream_info, "input_subscripts", ""),
        output_subscript=getattr(upstream_info, "output_subscript", ""),
        size_dict=dict(size_dict),
        optimizer_used=optimizer_used,
        contraction_list=list(upstream_info.contraction_list),
        scale_list=list(getattr(upstream_info, "scale_list", [])),
        size_list=list(getattr(upstream_info, "size_list", [])),
        _oe_naive_cost=naive_cost,
        _oe_opt_cost=optimized_cost,
    )
