"""Public FLOP cost query API.

All exported helpers return weighted FLOP costs by default, using the active
operation multipliers from ``flopscope._weights``. The underlying analytical
formulas remain available to the runtime in the private modules.
"""

from __future__ import annotations

import inspect
from functools import wraps

from flopscope._flops import (
    analytical_pointwise_cost as _analytical_pointwise_cost,
)
from flopscope._flops import (
    analytical_reduction_cost as _analytical_reduction_cost,
)
from flopscope._flops import (
    einsum_cost as _analytical_einsum_cost,
)
from flopscope._flops import (
    svd_cost as _analytical_svd_cost,
)
from flopscope._perm_group import SymmetryGroup
from flopscope._polynomial import (
    poly_cost as _analytical_poly_cost,
)
from flopscope._polynomial import (
    polyadd_cost as _analytical_polyadd_cost,
)
from flopscope._polynomial import (
    polyder_cost as _analytical_polyder_cost,
)
from flopscope._polynomial import (
    polydiv_cost as _analytical_polydiv_cost,
)
from flopscope._polynomial import (
    polyfit_cost as _analytical_polyfit_cost,
)
from flopscope._polynomial import (
    polyint_cost as _analytical_polyint_cost,
)
from flopscope._polynomial import (
    polymul_cost as _analytical_polymul_cost,
)
from flopscope._polynomial import (
    polysub_cost as _analytical_polysub_cost,
)
from flopscope._polynomial import (
    polyval_cost as _analytical_polyval_cost,
)
from flopscope._polynomial import (
    roots_cost as _analytical_roots_cost,
)
from flopscope._unwrap import unwrap_cost as _analytical_unwrap_cost
from flopscope._weights import get_weight
from flopscope._window import (
    bartlett_cost as _analytical_bartlett_cost,
)
from flopscope._window import (
    blackman_cost as _analytical_blackman_cost,
)
from flopscope._window import (
    hamming_cost as _analytical_hamming_cost,
)
from flopscope._window import (
    hanning_cost as _analytical_hanning_cost,
)
from flopscope._window import (
    kaiser_cost as _analytical_kaiser_cost,
)
from flopscope.numpy.fft._transforms import (
    fft_cost as _analytical_fft_cost,
)
from flopscope.numpy.fft._transforms import (
    fftn_cost as _analytical_fftn_cost,
)
from flopscope.numpy.fft._transforms import (
    hfft_cost as _analytical_hfft_cost,
)
from flopscope.numpy.fft._transforms import (
    rfft_cost as _analytical_rfft_cost,
)
from flopscope.numpy.fft._transforms import (
    rfftn_cost as _analytical_rfftn_cost,
)
from flopscope.numpy.linalg._compound import (
    matrix_power_cost as _analytical_matrix_power_cost,
)
from flopscope.numpy.linalg._compound import (
    multi_dot_cost as _analytical_multi_dot_cost,
)
from flopscope.numpy.linalg._decompositions import (
    cholesky_cost as _analytical_cholesky_cost,
)
from flopscope.numpy.linalg._decompositions import (
    eig_cost as _analytical_eig_cost,
)
from flopscope.numpy.linalg._decompositions import (
    eigh_cost as _analytical_eigh_cost,
)
from flopscope.numpy.linalg._decompositions import (
    eigvals_cost as _analytical_eigvals_cost,
)
from flopscope.numpy.linalg._decompositions import (
    eigvalsh_cost as _analytical_eigvalsh_cost,
)
from flopscope.numpy.linalg._decompositions import (
    qr_cost as _analytical_qr_cost,
)
from flopscope.numpy.linalg._decompositions import (
    svdvals_cost as _analytical_svdvals_cost,
)
from flopscope.numpy.linalg._properties import (
    cond_cost as _analytical_cond_cost,
)
from flopscope.numpy.linalg._properties import (
    det_cost as _analytical_det_cost,
)
from flopscope.numpy.linalg._properties import (
    matrix_norm_cost as _analytical_matrix_norm_cost,
)
from flopscope.numpy.linalg._properties import (
    matrix_rank_cost as _analytical_matrix_rank_cost,
)
from flopscope.numpy.linalg._properties import (
    norm_cost as _analytical_norm_cost,
)
from flopscope.numpy.linalg._properties import (
    slogdet_cost as _analytical_slogdet_cost,
)
from flopscope.numpy.linalg._properties import (
    trace_cost as _analytical_trace_cost,
)
from flopscope.numpy.linalg._properties import (
    vector_norm_cost as _analytical_vector_norm_cost,
)
from flopscope.numpy.linalg._solvers import (
    inv_cost as _analytical_inv_cost,
)
from flopscope.numpy.linalg._solvers import (
    lstsq_cost as _analytical_lstsq_cost,
)
from flopscope.numpy.linalg._solvers import (
    pinv_cost as _analytical_pinv_cost,
)
from flopscope.numpy.linalg._solvers import (
    solve_cost as _analytical_solve_cost,
)
from flopscope.numpy.linalg._solvers import (
    tensorinv_cost as _analytical_tensorinv_cost,
)
from flopscope.numpy.linalg._solvers import (
    tensorsolve_cost as _analytical_tensorsolve_cost,
)


def _weight_cost(
    op_name: str,
    analytical_cost: int,
    *,
    dtype_rate: float = 1.0,
    complex_factor: float = 1.0,
) -> int:
    """Convert an analytical FLOP count into the public weighted estimate.

    Mirrors ``BudgetContext.deduct()`` exactly: same factors, same
    multiplication order, floored once via ``int(...)``, so public estimates
    and runtime accounting stay in sync.
    """
    return int(analytical_cost * dtype_rate * complex_factor * get_weight(op_name))


def _make_weighted_cost(op_name: str, analytical_fn):
    example = _EXAMPLE_DOCS.get(op_name)

    @wraps(analytical_fn)
    def wrapper(*args, **kwargs):
        return _weight_cost(op_name, analytical_fn(*args, **kwargs))

    wrapper.__doc__ = _build_weighted_cost_docstring(
        op_name,
        analytical_fn,
        example=example,
    )
    return wrapper


_EXAMPLE_DOCS = {
    "einsum": """\
>>> import flopscope as flops
>>> cost = flops.accounting.einsum_cost(
...     "ij,jk->ik",
...     [(8, 16), (16, 4)],
... )
>>> isinstance(cost, int)
True
""",
    "linalg.svd": """\
>>> import flopscope as flops
>>> cost = flops.accounting.svd_cost(128, 64)
>>> cost_topk = flops.accounting.svd_cost(128, 64, k=8)  # min(4*128*64*8, economy)
>>> isinstance(cost, int)
True
""",
}


_PARAMETER_DESCRIPTIONS = {
    "axis": "Axis or axes forwarded to the analytical cost formula.",
    "input_shape": "Shape of the reduction input.",
    "k": "Target rank or number of singular components to estimate.",
    "m": "Number of rows in the input matrix.",
    "n": "Number of columns in the input matrix.",
    "operand_symmetries": "Optional symmetry metadata for each einsum operand.",
    "shape": "Shape of the array passed to the analytical cost formula.",
    "shapes": "Operand shapes in the same order as the einsum operands.",
    "subscripts": "Einstein summation expression that defines the contraction.",
    "symmetry_info": (
        "Optional symmetry metadata used to count only analytically distinct elements."
    ),
}


def _build_weighted_cost_docstring(
    op_name: str,
    analytical_fn,
    *,
    example: str | None = None,
) -> str:
    summary, description, notes = _parse_analytical_doc(analytical_fn.__doc__ or "")
    signature = inspect.signature(analytical_fn)

    parts = [
        _make_weighted_summary(summary, op_name),
    ]
    if description:
        parts.extend(["", description])
    parts.extend(
        [
            "",
            "Parameters",
            "----------",
            _render_parameters(signature, op_name),
            "",
            "Returns",
            "-------",
            "int",
            "    Weighted public cost estimate, floored to match runtime accounting.",
            "",
            "Notes",
            "-----",
            (
                "This helper multiplies the analytical FLOP count by the active "
                "weight from ``flopscope._weights`` and then applies ``int(...)`` "
                "so public estimates match budget deductions."
            ),
        ]
    )
    if notes:
        parts.extend(["", notes])
    if example:
        parts.extend(["", "Examples", "--------", example.rstrip()])
    return "\n".join(parts)


def _make_weighted_summary(summary: str, op_name: str) -> str:
    if summary:
        return summary.replace("FLOP cost", "Weighted FLOP cost", 1)
    return f"Weighted FLOP cost estimate for ``{op_name}``."


def _parse_analytical_doc(doc: str) -> tuple[str, str, str]:
    lines = inspect.cleandoc(doc).splitlines()
    if not lines:
        return "", "", ""

    summary = ""
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index < len(lines):
        summary = lines[index].strip()
        index += 1

    while index < len(lines) and not lines[index].strip():
        index += 1

    description_lines: list[str] = []
    notes_lines: list[str] = []
    current_section: str | None = None

    while index < len(lines):
        if _is_numpydoc_header(lines, index):
            current_section = lines[index].strip()
            index += 2
            continue

        line = lines[index]
        if current_section == "Notes":
            notes_lines.append(line)
        elif current_section is None:
            description_lines.append(line)
        index += 1

    description = "\n".join(description_lines).strip()
    notes = "\n".join(notes_lines).strip()
    return summary, description, notes


def _is_numpydoc_header(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    title = lines[index].strip()
    underline = lines[index + 1].strip()
    return bool(title) and underline == "-" * len(title)


def _render_parameters(signature: inspect.Signature, op_name: str) -> str:
    rendered: list[str] = []
    for parameter in signature.parameters.values():
        annotation = _format_annotation(parameter.annotation)
        header = parameter.name
        if annotation:
            header = f"{header} : {annotation}"
        if parameter.default is not inspect.Signature.empty:
            header = f"{header}, optional"

        description = _PARAMETER_DESCRIPTIONS.get(
            parameter.name,
            f"Argument forwarded to the analytical ``{op_name}`` cost formula.",
        )
        if parameter.default is not inspect.Signature.empty:
            description = f"{description} Defaults to ``{parameter.default!r}``."
        rendered.append(f"{header}\n    {description}")
    return "\n\n".join(rendered)


def _format_annotation(annotation) -> str:
    if annotation is inspect.Signature.empty:
        return "object"
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", repr(annotation))


def pointwise_cost(
    op_name: str,
    *,
    shape: tuple[int, ...],
    symmetry: SymmetryGroup | None = None,
) -> int:
    """Weighted FLOP cost of a pointwise operation.

    Parameters
    ----------
    op_name : str
        Operation name used for weight lookup, e.g. ``"exp"`` or ``"add"``.
    shape : tuple of int
        Output shape of the pointwise operation.
    symmetry : SymmetryGroup or None, optional
        If provided, only unique elements are counted analytically before the
        operation weight is applied.

    Returns
    -------
    int
        Weighted public cost estimate, floored to match runtime accounting.

    Notes
    -----
    This helper multiplies the analytical pointwise count by the active weight
    for ``op_name`` and then floors with ``int(...)`` to match runtime budget
    accounting.

    Examples
    --------
    >>> import flopscope as flops
    >>> cost = flops.accounting.pointwise_cost("exp", shape=(2, 3))
    >>> isinstance(cost, int)
    True
    """
    if not isinstance(op_name, str):
        raise TypeError("pointwise_cost() requires op_name as the first argument")
    return _weight_cost(
        op_name,
        _analytical_pointwise_cost(shape, symmetry=symmetry),
    )


def reduction_cost(
    op_name: str,
    *,
    input_shape: tuple[int, ...],
    axis: int | None = None,
    symmetry: SymmetryGroup | None = None,
) -> int:
    """Weighted FLOP cost of a reduction operation.

    Parameters
    ----------
    op_name : str
        Operation name used for weight lookup, e.g. ``"sum"`` or ``"max"``.
    input_shape : tuple of int
        Shape of the reduction input.
    axis : int or None, optional
        Reduction axis. Accepted for API consistency with the analytical helper.
    symmetry : SymmetryGroup or None, optional
        If provided, only unique elements are counted analytically before the
        operation weight is applied.

    Returns
    -------
    int
        Weighted public cost estimate, floored to match runtime accounting.

    Notes
    -----
    This helper multiplies the analytical reduction count by the active weight
    for ``op_name`` and then floors with ``int(...)`` to match runtime budget
    accounting.

    Examples
    --------
    >>> import flopscope as flops
    >>> cost = flops.accounting.reduction_cost(
    ...     "sum",
    ...     input_shape=(4, 5),
    ...     axis=1,
    ... )
    >>> isinstance(cost, int)
    True
    """
    if not isinstance(op_name, str):
        raise TypeError("reduction_cost() requires op_name as the first argument")
    return _weight_cost(
        op_name,
        _analytical_reduction_cost(
            input_shape,
            axis=axis,
            symmetry=symmetry,
        ),
    )


einsum_cost = _make_weighted_cost("einsum", _analytical_einsum_cost)
svd_cost = _make_weighted_cost("linalg.svd", _analytical_svd_cost)

polyval_cost = _make_weighted_cost("polyval", _analytical_polyval_cost)
polyadd_cost = _make_weighted_cost("polyadd", _analytical_polyadd_cost)
polysub_cost = _make_weighted_cost("polysub", _analytical_polysub_cost)
polymul_cost = _make_weighted_cost("polymul", _analytical_polymul_cost)
polydiv_cost = _make_weighted_cost("polydiv", _analytical_polydiv_cost)
polyfit_cost = _make_weighted_cost("polyfit", _analytical_polyfit_cost)
poly_cost = _make_weighted_cost("poly", _analytical_poly_cost)
roots_cost = _make_weighted_cost("roots", _analytical_roots_cost)
polyder_cost = _make_weighted_cost("polyder", _analytical_polyder_cost)
polyint_cost = _make_weighted_cost("polyint", _analytical_polyint_cost)

unwrap_cost = _make_weighted_cost("unwrap", _analytical_unwrap_cost)

bartlett_cost = _make_weighted_cost("bartlett", _analytical_bartlett_cost)
blackman_cost = _make_weighted_cost("blackman", _analytical_blackman_cost)
hamming_cost = _make_weighted_cost("hamming", _analytical_hamming_cost)
hanning_cost = _make_weighted_cost("hanning", _analytical_hanning_cost)
kaiser_cost = _make_weighted_cost("kaiser", _analytical_kaiser_cost)

fft_cost = _make_weighted_cost("fft.fft", _analytical_fft_cost)
rfft_cost = _make_weighted_cost("fft.rfft", _analytical_rfft_cost)
fftn_cost = _make_weighted_cost("fft.fftn", _analytical_fftn_cost)
rfftn_cost = _make_weighted_cost("fft.rfftn", _analytical_rfftn_cost)
hfft_cost = _make_weighted_cost("fft.hfft", _analytical_hfft_cost)

multi_dot_cost = _make_weighted_cost("linalg.multi_dot", _analytical_multi_dot_cost)
matrix_power_cost = _make_weighted_cost(
    "linalg.matrix_power",
    _analytical_matrix_power_cost,
)

cholesky_cost = _make_weighted_cost("linalg.cholesky", _analytical_cholesky_cost)
qr_cost = _make_weighted_cost("linalg.qr", _analytical_qr_cost)
eig_cost = _make_weighted_cost("linalg.eig", _analytical_eig_cost)
eigh_cost = _make_weighted_cost("linalg.eigh", _analytical_eigh_cost)
eigvals_cost = _make_weighted_cost("linalg.eigvals", _analytical_eigvals_cost)
eigvalsh_cost = _make_weighted_cost("linalg.eigvalsh", _analytical_eigvalsh_cost)
svdvals_cost = _make_weighted_cost("linalg.svdvals", _analytical_svdvals_cost)

trace_cost = _make_weighted_cost("linalg.trace", _analytical_trace_cost)
det_cost = _make_weighted_cost("linalg.det", _analytical_det_cost)
slogdet_cost = _make_weighted_cost("linalg.slogdet", _analytical_slogdet_cost)
norm_cost = _make_weighted_cost("linalg.norm", _analytical_norm_cost)
vector_norm_cost = _make_weighted_cost(
    "linalg.vector_norm",
    _analytical_vector_norm_cost,
)
matrix_norm_cost = _make_weighted_cost(
    "linalg.matrix_norm",
    _analytical_matrix_norm_cost,
)
cond_cost = _make_weighted_cost("linalg.cond", _analytical_cond_cost)
matrix_rank_cost = _make_weighted_cost(
    "linalg.matrix_rank",
    _analytical_matrix_rank_cost,
)

solve_cost = _make_weighted_cost("linalg.solve", _analytical_solve_cost)
inv_cost = _make_weighted_cost("linalg.inv", _analytical_inv_cost)
lstsq_cost = _make_weighted_cost("linalg.lstsq", _analytical_lstsq_cost)
pinv_cost = _make_weighted_cost("linalg.pinv", _analytical_pinv_cost)
tensorsolve_cost = _make_weighted_cost(
    "linalg.tensorsolve",
    _analytical_tensorsolve_cost,
)
tensorinv_cost = _make_weighted_cost("linalg.tensorinv", _analytical_tensorinv_cost)

__all__ = [
    # Existing
    "einsum_cost",
    "pointwise_cost",
    "reduction_cost",
    "svd_cost",
    # Linalg
    "cholesky_cost",
    "qr_cost",
    "eig_cost",
    "eigh_cost",
    "eigvals_cost",
    "eigvalsh_cost",
    "svdvals_cost",
    "solve_cost",
    "inv_cost",
    "lstsq_cost",
    "pinv_cost",
    "tensorsolve_cost",
    "tensorinv_cost",
    "trace_cost",
    "det_cost",
    "slogdet_cost",
    "norm_cost",
    "vector_norm_cost",
    "matrix_norm_cost",
    "cond_cost",
    "matrix_rank_cost",
    "multi_dot_cost",
    "matrix_power_cost",
    # FFT
    "fft_cost",
    "rfft_cost",
    "fftn_cost",
    "rfftn_cost",
    "hfft_cost",
    # Polynomial
    "polyval_cost",
    "polyadd_cost",
    "polysub_cost",
    "polymul_cost",
    "polydiv_cost",
    "polyfit_cost",
    "poly_cost",
    "roots_cost",
    "polyder_cost",
    "polyint_cost",
    # Window
    "bartlett_cost",
    "blackman_cost",
    "hamming_cost",
    "hanning_cost",
    "kaiser_cost",
    # Other
    "unwrap_cost",
]
