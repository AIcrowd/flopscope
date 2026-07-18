"""Regression tests for the data-movement free-tier cost-model change.

See .aicrowd/superpowers/specs/2026-06-15-data-movement-free-tier-design.md.
"""

import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._symmetric import SymmetricTensor
from flopscope._weights import get_weight, load_weights, reset_weights

# Ops that must bill 0 FLOPs under production weights (data movement / select).
FREE_DATA_MOVEMENT_OPS = [
    # tril/triu/diag/diagflat left the free tier too -- the triangular/diagonal
    # family now bills the values it writes (kept-triangle count / v.shape[0] /
    # numel(v)) at weight 1.0, see
    # tests/test_triage_price_pins.py::test_diag_family_bills_written_values_only,
    # test_triu_batch_leading_dims_multiply.
    # take/take_along_axis/choose (gather x4), put/place/putmask/put_along_axis/
    # fill_diagonal/extract/compress (scatter x1) left the free tier -- see
    # tests/test_triage_price_pins.py::test_gather_tier_bills_4x_output,
    # test_scatter_ops_bill_elements_touched, test_extract_and_compress_bill_scan_plus_gather.
    # concatenate/concat/stack/vstack/hstack/dstack/column_stack/row_stack/
    # block/bmat/tile/repeat/roll/resize/delete/insert/append/fromiter/full/
    # full_like/meshgrid left the free tier too -- the array-assembly and
    # replication family now bills numel(output) at weight 1.0, see
    # tests/test_triage_price_pins.py::test_creation_and_copy_family_bills_output,
    # test_creation_and_copy_family_remaining_ops_bill_output.
    # select and 3-arg where left the free tier too -- the select class now
    # bills its scan/select work at weight 1.0/4.0, see
    # tests/test_triage_price_pins.py::test_select_and_piecewise_bill_per_condition,
    # test_where_three_arg_bills_4x_broadcast_output.
    "unstack",
    "ix_",
]


@pytest.fixture
def production_weights(monkeypatch):
    """Load the packaged production weight table for this test only.

    conftest's autouse fixture resets to unit weights around every test.
    """
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    load_weights()
    yield


@pytest.mark.parametrize("op", FREE_DATA_MOVEMENT_OPS)
def test_data_movement_op_is_weight_zero(production_weights, op):
    assert get_weight(op) == 0.0, f"{op} should be free (weight 0.0)"


def test_row_stack_bills_same_as_vstack_under_production_rates(production_weights):
    """row_stack is a bare `return vstack(tup)` alias — pure data movement.

    It has no deduct() of its own (billing happens under `vstack`), so its
    own weight in default_weights.json is inert: the array-assembly and
    replication family (including row_stack, for consistency) now carries
    weight 1.0, but row_stack's billed cost is always exactly vstack's,
    by construction, regardless of that key's value.
    """
    a = fnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as ctx:
        fnp.row_stack([a, a])
        row_stack_cost = ctx.flops_used
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as ctx:
        fnp.vstack([a, a])
        vstack_cost = ctx.flops_used
    # a is float64 (2,3): numel(output)=12 * dtype_rate 2.0 * weight 1.0 = 24.
    assert row_stack_cost == 24
    assert row_stack_cost == vstack_cost


def _mk1d():
    return fnp.asarray([float(i) for i in range(100)])


def _mk2d():
    return fnp.asarray([float(i) for i in range(100)]).reshape(10, 10)


# (build, call): build runs BEFORE n0 so ONLY the op under test is measured.
# (After migration, building inputs via reshape would itself add a record, so
# shaped inputs are constructed outside the measured region.)
# reshape/copy/fft.fftshift/fft.ifftshift left this dict in Task 4: they now
# bill numel(input)/numel(output) (weight 1.0), so they no longer belong among
# the "free view op, 0 FLOPs" cases below -- see
# tests/test_triage_price_pins.py::test_fft_shifts_bill_their_copy,
# test_conditional_view_copies_bill_numel for their exact-value pins.
VIEW_OPS_126 = {
    "transpose": (_mk2d, lambda a: fnp.transpose(a)),
    "swapaxes": (_mk2d, lambda a: fnp.swapaxes(a, 0, 1)),
    "moveaxis": (_mk2d, lambda a: fnp.moveaxis(a, 0, 1)),
    "squeeze": (lambda: _mk1d().reshape(1, 100), lambda a: fnp.squeeze(a)),
    "expand_dims": (_mk1d, lambda a: fnp.expand_dims(a, 0)),
    "flip": (_mk1d, lambda a: fnp.flip(a)),
    "fliplr": (_mk2d, lambda a: fnp.fliplr(a)),
    "flipud": (_mk2d, lambda a: fnp.flipud(a)),
    "rot90": (_mk2d, lambda a: fnp.rot90(a)),
    "atleast_1d": (_mk1d, lambda a: fnp.atleast_1d(a)),
    "atleast_2d": (_mk1d, lambda a: fnp.atleast_2d(a)),
    "atleast_3d": (_mk1d, lambda a: fnp.atleast_3d(a)),
    "hsplit": (_mk2d, lambda a: fnp.hsplit(a, 2)),
}


@pytest.mark.parametrize("name", sorted(VIEW_OPS_126))
def test_view_op_is_time_accounted(name):
    """#126: free view ops route through deduct -> >=1 op-log record, all 0 FLOPs."""
    build, call = VIEW_OPS_126[name]
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as ctx:
        a = build()
        n0 = len(ctx.op_log)
        call(a)
        new = ctx.op_log[n0:]
    assert len(new) >= 1, f"{name}: no op-log record (still bypasses deduct)"
    assert all(r.flop_cost == 0 for r in new), f"{name}: free op billed nonzero FLOPs"


# ones/eye/identity left this dict in Task 4: they now bill numel(output) /
# diagonal length (weight 1.0) -- see
# tests/test_triage_price_pins.py::test_writing_creation_bills_output_zeros_stay_free.
INIT_OPS_126 = {
    "zeros": lambda: fnp.zeros((10, 10)),
    "empty": lambda: fnp.empty((10, 10)),
    "tri": lambda: fnp.tri(10),
}


@pytest.mark.parametrize("name", sorted(INIT_OPS_126))
def test_init_op_is_time_accounted(name):
    """#126: constant-init ops route through deduct -> one op-log record, 0 FLOPs."""
    call = INIT_OPS_126[name]
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as ctx:
        n0 = len(ctx.op_log)
        call()
        new = ctx.op_log[n0:]
    assert len(new) == 1, f"{name}: expected 1 op-log record, got {len(new)}"
    assert new[0].op_name == name
    assert new[0].flop_cost == 0


@pytest.mark.parametrize("name", ["zeros_like", "empty_like"])
def test_init_like_op_is_time_accounted(name):
    # ones_like left this list in Task 4 -- it now bills numel(output).
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as ctx:
        a = fnp.asarray([float(i) for i in range(100)])
        n0 = len(ctx.op_log)
        getattr(fnp, name)(a)
        new = ctx.op_log[n0:]
    assert len(new) == 1, f"{name}: expected 1 op-log record, got {len(new)}"
    assert new[0].flop_cost == 0


def test_empty_and_tri_are_not_falsely_symmetric():
    """empty/empty_like/tri are NOT constant fills, so must not infer symmetry.

    A triangular (`tri`) or uninitialized (`empty`) square array tagged S_n would
    let a symmetry-aware op undercount. Only genuine constant fills (zeros/ones)
    and structural constructors (eye/identity) carry symmetry.
    """
    assert not isinstance(fnp.empty((3, 3)), SymmetricTensor)
    assert not isinstance(fnp.empty_like(fnp.zeros((2, 2))), SymmetricTensor)
    assert not isinstance(fnp.tri(3), SymmetricTensor)
    # Contrast: genuine constant fills still infer symmetry (unchanged).
    assert isinstance(fnp.zeros((3, 3)), SymmetricTensor)
    assert isinstance(fnp.ones((3, 3)), SymmetricTensor)


def test_where_one_arg_is_charged_like_nonzero():
    """1-arg where IS nonzero -> deducts under "nonzero" (alias parity),
    charged numel (unit weights)."""
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as ctx:
        mask = fnp.asarray([True, False, True, False] * 25)  # 100 elems, prebuilt
        n0 = len(ctx.op_log)
        fnp.where(mask)
        new = ctx.op_log[n0:]
    assert len(new) == 1 and new[0].op_name == "nonzero"
    assert new[0].flop_cost == 100  # numel, unit weight


def test_where_three_arg_bills_broadcast_output():
    """3-arg where selects by a given mask but still scans and writes every
    output element -> charged numel(broadcast output) (unit weights)."""
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as ctx:
        mask = fnp.asarray([True, False] * 50)
        x = fnp.asarray([1.0] * 100)
        y = fnp.asarray([0.0] * 100)
        n0 = len(ctx.op_log)
        fnp.where(mask, x, y)
        new = ctx.op_log[n0:]
    assert len(new) == 1 and new[0].op_name == "where"
    assert new[0].flop_cost == 100  # numel(broadcast), unit weight


def test_where_predicate_still_charged(production_weights):
    """where(a > 0.5): the comparison is charged; the 1-arg form deducts
    under "nonzero" at nonzero's weight, distinct from 3-arg where's own."""

    def call():
        a = fnp.asarray([i / 100 for i in range(100)])
        return fnp.where(a > 0.5)

    with flops.BudgetContext(flop_budget=10**9, quiet=True) as ctx:
        n0 = len(ctx.op_log)
        call()
        names = [r.op_name for r in ctx.op_log[n0:]]
    assert "greater" in names  # predicate charged
    assert "nonzero" in names  # 1-arg where deducts under nonzero (alias parity)
    assert (
        get_weight("nonzero") == 1.0
    )  # 1-arg where charged at nonzero's weight 1.0, not 3-arg where's 4.0


def test_nonzero_method_matches_function():
    """a.nonzero() must charge the same as fnp.nonzero(a) (numel)."""
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as ctx:
        a = fnp.asarray([float(i - 50) for i in range(100)])
        n0 = len(ctx.op_log)
        a.nonzero()
        method_records = ctx.op_log[n0:]
    assert len(method_records) == 1, "a.nonzero() produced no op-log record"
    assert method_records[0].op_name == "nonzero"
    assert method_records[0].flop_cost == 100  # numel, unit weight


# ---------------------------------------------------------------------------
# Task 6: value-changing astype is charged; lossless cast stays free
# ---------------------------------------------------------------------------


def _flop_cost(call):
    with flops.BudgetContext(flop_budget=10**12, quiet=True) as ctx:
        n0 = len(ctx.op_log)
        call()
        new = ctx.op_log[n0:]
    return sum(r.flop_cost for r in new), [r.op_name for r in new]


@pytest.mark.parametrize(
    "dtype, changes_values",
    [
        (bool, True),  # !=0 test
        ("int64", True),  # float->int truncation
        ("float32", True),  # narrowing (round)
        ("float64", False),  # width cast (lossless) - stays free
    ],
)
def test_astype_function_charges_value_changing_casts(dtype, changes_values):
    a = fnp.asarray([float(i) - 50 for i in range(100)])  # float64 source
    cost, names = _flop_cost(lambda: fnp.astype(a, dtype))
    assert "astype" in names, "astype must produce an op-log record"
    assert cost == (100 if changes_values else 0)


@pytest.mark.parametrize(
    "dtype, changes_values",
    [(bool, True), ("int64", True), ("float64", False)],
)
def test_astype_method_charges_value_changing_casts(dtype, changes_values):
    a = fnp.asarray([float(i) - 50 for i in range(100)])
    cost, names = _flop_cost(lambda: a.astype(dtype))
    assert "astype" in names
    assert cost == (100 if changes_values else 0)


def test_astype_method_honors_casting_kwarg():
    """a.astype(dt, casting='safe') must raise on an unsafe cast (numpy parity)."""
    a = fnp.asarray([1.0, 2.0, 3.0])  # float64
    with pytest.raises(TypeError):
        a.astype("float32", casting="safe")  # f64->f32 is unsafe


# ---------------------------------------------------------------------------
# Task 7: per-op docstring labels must match actual billing
# ---------------------------------------------------------------------------
import pathlib
import re

import flopscope._array_ops as _array_ops_mod


def test_free_labels_match_actual_weight():
    """Every op labeled "free"/"0 FLOPs" in _array_ops.py must truly bill 0
    (weight 0 under production weights). Flags charged ops mislabeled "free"."""
    load_weights()
    try:
        src = pathlib.Path(_array_ops_mod.__file__).read_text()
        pattern = re.compile(
            r'attach_docstring\(\s*(\w+)\s*,[^,]+,\s*"free"\s*,\s*"([^"]*)"\s*\)'
        )
        mislabeled = [
            (fn, get_weight(fn), cost)
            for fn, cost in pattern.findall(src)
            if get_weight(fn) != 0.0
        ]
        assert not mislabeled, (
            'ops labeled "free" but weight != 0 — relabel to "counted_custom" '
            f"with the real cost: {mislabeled}"
        )
    finally:
        reset_weights()


def test_counted_custom_labels_match_actual_weight():
    """Every op labeled "counted_custom" in _array_ops.py must have weight != 0
    (free ops mislabeled as counted would falsely charge budget). Reverse of
    test_free_labels_match_actual_weight."""
    load_weights()
    try:
        src = pathlib.Path(_array_ops_mod.__file__).read_text()
        pattern = re.compile(
            r'attach_docstring\(\s*(\w+)\s*,[^,]+,\s*"counted_custom"\s*,\s*"([^"]*)"\s*\)'
        )
        mislabeled = [
            (fn, get_weight(fn), cost)
            for fn, cost in pattern.findall(src)
            if get_weight(fn) == 0.0
        ]
        assert not mislabeled, (
            'ops labeled "counted_custom" but weight == 0.0 — relabel to "free"/"0 FLOPs": '
            f"{mislabeled}"
        )
    finally:
        reset_weights()


def test_ops_json_weight_equals_billed_weight():
    """The published ops.json weight must equal what get_weight() actually bills.

    Locks the single-source-of-truth invariant: ops.json is generated from
    default_weights.json, the same table _weights.py loads. Sampled across tiers.
    """
    import json
    import pathlib

    from flopscope._weights import get_weight, load_weights, reset_weights

    ops_path = (
        pathlib.Path(__file__).resolve().parents[1] / "website" / "public" / "ops.json"
    )
    ops = {
        o["name"]: o["weight"] for o in json.loads(ops_path.read_text())["operations"]
    }
    load_weights()
    try:
        for op in [
            "take",
            "hstack",
            "where",
            "astype",
            "add",
            "exp",
            "sort",
            "linalg.inv",
            "geomspace",
        ]:
            if op in ops:
                assert ops[op] == get_weight(op), (
                    f"{op}: ops.json={ops[op]} but get_weight={get_weight(op)}"
                )
    finally:
        reset_weights()
