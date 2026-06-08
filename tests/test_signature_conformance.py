"""Test that flopscope public signatures match numpy plus explicit flopscope extensions."""

import inspect

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp

# Flopscope-only keyword extensions that numpy doesn't have
ALLOWED_EXTRA_PARAMS = {
    "einsum": {"symmetry", "subscripts"},
    "linalg.svd": {"k"},
    "linalg.svdvals": {"k"},
}

# Functions where flopscope intentionally differs (with reason)
SKIP_FUNCTIONS = {
    "einsum_path",
    # File I/O deliberately omits numpy's allow_pickle/mmap_mode/fix_imports/
    # encoding params: flopscope never unpickles (that is the security boundary),
    # so these signatures intentionally diverge from numpy's.
    "load",
    "save",
    "savez",
    "savez_compressed",
}

_NUMPY_GE_2_4 = tuple(int(x) for x in np.__version__.split(".")[:2]) >= (2, 4)
if _NUMPY_GE_2_4:
    SKIP_FUNCTIONS |= {
        "dot",
        "packbits",
        "unpackbits",
        "shares_memory",
        "ravel_multi_index",
        "promote_types",
    }


def _iter_flopscope_functions():
    for name in sorted(dir(fnp)):
        if name.startswith("_"):
            continue
        we_fn = getattr(fnp, name, None)
        np_fn = getattr(np, name, None)
        if not callable(we_fn) or not callable(np_fn):
            continue
        if isinstance(np_fn, np.ufunc):
            continue
        yield name, we_fn, np_fn

    for submod in ["linalg", "fft", "random"]:
        np_sub = getattr(np, submod, None)
        we_sub = getattr(fnp, submod, None)
        if np_sub is None or we_sub is None:
            continue
        for name in sorted(dir(we_sub)):
            if name.startswith("_"):
                continue
            we_fn = getattr(we_sub, name, None)
            np_fn = getattr(np_sub, name, None)
            if not callable(we_fn) or not callable(np_fn):
                continue
            if isinstance(np_fn, np.ufunc):
                continue
            yield f"{submod}.{name}", we_fn, np_fn


def _get_param_names(sig):
    return {
        name
        for name, p in sig.parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


@pytest.mark.parametrize(
    "dotted_name,we_fn,np_fn",
    [
        pytest.param(name, we_fn, np_fn, id=name)
        for name, we_fn, np_fn in _iter_flopscope_functions()
        if name not in SKIP_FUNCTIONS
    ],
)
def test_signature_matches_numpy(dotted_name, we_fn, np_fn):
    try:
        np_sig = inspect.signature(np_fn)
    except (ValueError, TypeError):
        pytest.skip(f"Cannot inspect numpy.{dotted_name}")
    try:
        we_sig = inspect.signature(we_fn)
    except (ValueError, TypeError):
        pytest.skip(f"Cannot inspect flopscope.{dotted_name}")

    np_params = _get_param_names(np_sig)
    we_params = _get_param_names(we_sig)
    allowed_extra = ALLOWED_EXTRA_PARAMS.get(dotted_name, set())

    missing = np_params - we_params
    unexpected_extra = (we_params - np_params) - allowed_extra

    assert not missing, (
        f"flopscope.{dotted_name} is missing numpy params: {missing}\n"
        f"  numpy:  {np_sig}\n"
        f"  flopscope:  {we_sig}"
    )
    assert not unexpected_extra, (
        f"flopscope.{dotted_name} has unexpected extra params: {unexpected_extra}\n"
        f"  (add to ALLOWED_EXTRA_PARAMS if intentional)\n"
        f"  numpy:  {np_sig}\n"
        f"  flopscope:  {we_sig}"
    )


def test_no_legacy_public_symmetry_names():
    for legacy_name in ("PermutationGroup", "Permutation", "Cycle", "SymmetryInfo"):
        assert not hasattr(fnp, legacy_name)


def test_einsum_signature_only_uses_symmetry_keyword():
    params = inspect.signature(fnp.einsum).parameters
    assert "symmetry" in params
    assert "symmetric_axes" not in params


def test_symmetrize_and_random_symmetric_use_new_keyword():
    assert "symmetry" in inspect.signature(flops.symmetrize).parameters
    assert "symmetry" in inspect.signature(fnp.random.symmetric).parameters
    assert "symmetric_axes" not in inspect.signature(flops.symmetrize).parameters
    assert "symmetric_axes" not in inspect.signature(fnp.random.symmetric).parameters


def test_public_flop_apis_use_symmetry_keyword():
    for fn in (flops.accounting.pointwise_cost, flops.accounting.reduction_cost):
        params = inspect.signature(fn).parameters
        assert "symmetry" in params
        assert "symmetric_axes" not in params

    einsum_params = inspect.signature(flops.accounting.einsum_cost).parameters
    assert "operand_symmetries" in einsum_params
    assert "symmetric_axes" not in einsum_params
