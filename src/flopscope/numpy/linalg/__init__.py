"""Linear algebra submodule for flopscope."""

from flopscope._registry import make_module_getattr as _make_module_getattr
from flopscope.numpy.linalg._aliases import (  # noqa: F401
    cross,
    diagonal,
    matmul,
    matrix_transpose,
    outer,
    tensordot,
    vecdot,
)
from flopscope.numpy.linalg._compound import (  # noqa: F401
    matrix_power,
    matrix_power_cost,
    multi_dot,
    multi_dot_cost,
)
from flopscope.numpy.linalg._decompositions import (  # noqa: F401
    cholesky,
    cholesky_cost,
    eig,
    eig_cost,
    eigh,
    eigh_cost,
    eigvals,
    eigvals_cost,
    eigvalsh,
    eigvalsh_cost,
    qr,
    qr_cost,
    svdvals,
    svdvals_cost,
)
from flopscope.numpy.linalg._properties import (  # noqa: F401
    cond,
    cond_cost,
    det,
    det_cost,
    matrix_norm,
    matrix_norm_cost,
    matrix_rank,
    matrix_rank_cost,
    norm,
    norm_cost,
    slogdet,
    slogdet_cost,
    trace,
    trace_cost,
    vector_norm,
    vector_norm_cost,
)
from flopscope.numpy.linalg._solvers import (  # noqa: F401
    inv,
    inv_cost,
    lstsq,
    lstsq_cost,
    pinv,
    pinv_cost,
    solve,
    solve_cost,
    tensorinv,
    tensorinv_cost,
    tensorsolve,
    tensorsolve_cost,
)
from flopscope.numpy.linalg._svd import svd  # noqa: F401

__all__ = [
    "svd",
    "matmul",
    "cross",
    "outer",
    "tensordot",
    "vecdot",
    "diagonal",
    "matrix_transpose",
    "cholesky",
    "qr",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "svdvals",
    "solve",
    "inv",
    "lstsq",
    "pinv",
    "tensorsolve",
    "tensorinv",
    "trace",
    "det",
    "slogdet",
    "norm",
    "vector_norm",
    "matrix_norm",
    "cond",
    "matrix_rank",
    "multi_dot",
    "matrix_power",
]

__getattr__ = _make_module_getattr(
    module_prefix="linalg.", module_label="flopscope.numpy.linalg"
)

import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

# ``check_module=False`` because every name above is re-exported from a
# submodule, so the default ``__module__`` filter would skip all of them.
#
# ``multi_dot`` is skipped because it is the one op here that takes ``out=``
# and returns that buffer by identity. Wrapping the return view-casts a plain
# ndarray destination, so ``multi_dot([m, n], out=dest) is dest`` goes True ->
# False -- measured, not assumed. See
# ``test_multi_dot_out_hands_back_the_callers_buffer``.
_wrap_module_returns(
    _sys.modules[__name__], check_module=False, skip_names={"multi_dot"}
)
