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

from . import _helpers as helpers
from ._hsluv import rgb_distance_hex, rich_label_palette

__all__ = [
    "build_path_info",
    "PathInfo",
    "StepInfo",
]


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
        table.add_column("flops", justify="right", no_wrap=True, width=flops_width)
        table.add_column("blas", no_wrap=True, width=blas_width)

        cumulative = 0
        for i, step in enumerate(self.steps):
            row = [
                str(i),
                self._fmt_contract(step),
                self._rich_subscript_text(step.subscript),
                f"{step.flop_cost:,}",
                str(step.blas_type) if step.blas_type else "-",
            ]
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
            its output shape, and the cumulative cost so far.
            Useful for debugging why a particular step's cost is what
            it is. Default False.
        """
        header_lines = self._header_lines()

        contract_strs = [self._fmt_contract(s) for s in self.steps]
        contract_col_width = max(
            len("contract"), max((len(c) for c in contract_strs), default=0)
        )

        cols = [
            f"{'step':>4}",
            f"{'contract':<{contract_col_width}}",
            f"{'subscript':<30}",
            f"{'flops':>14}",
            f"{'blas':<8}",
        ]

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
                f"{step.flop_cost:>14,}",
                f"{blas_label:<8}",
            ]
            lines.append("  ".join(row_parts))

            cumulative += step.flop_cost
            if verbose:
                # Indented details row: subset, out_shape, cumulative cost.
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
    partition_budget = int(get_setting("partition_budget"))
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
    upstream_path, upstream_info, *, size_dict, optimizer_used: str = ""
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

        steps_out.append(
            StepInfo(
                subscript=einsum_str,
                flop_cost=cost,
                input_shapes=input_shapes_for_step,
                output_shape=output_shape_for_step,
                blas_type=do_blas,
                path_indices=original_path_tuple,
                merged_subset=new_merged_subset,
            )
        )

    optimized_cost = sum(s.flop_cost for s in steps_out)

    # Recompute naive_cost: single-step contraction over all labels.
    all_labels: frozenset[str] = frozenset(size_dict.keys())
    n_ops = (
        len(upstream_info.contraction_list[0][3]) + 1
        if upstream_info.contraction_list
        and upstream_info.contraction_list[0][3] is not None
        else 2
    )
    try:
        naive_cost = helpers.flop_count(
            idx_contraction=all_labels,
            inner=True,
            num_terms=n_ops,
            size_dictionary=size_dict,
        )
    except Exception:
        naive_cost = int(upstream_info.naive_cost)

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
