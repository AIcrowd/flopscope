"""Billing, NumPy parity, and timing coverage for ``fnp.ix_``."""

import re
import time

import numpy as np
import pytest

import flopscope as flops
import flopscope._array_ops as array_ops
import flopscope.numpy as fnp


def billed(*args):
    """Run ``fnp.ix_`` under the unit weights restored by the autouse fixture."""
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as budget:
        result = fnp.ix_(*args)
    return result, budget.flops_used


def assert_rejected_without_charge(arg):
    backend_calls = 0

    def unexpected_ix(*args, **kwargs):
        nonlocal backend_calls
        backend_calls += 1
        raise AssertionError("NumPy ix_ backend executed")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(array_ops._np, "ix_", unexpected_ix)
        with flops.BudgetContext(flop_budget=10**9, quiet=True) as budget:
            with pytest.raises(
                TypeError, match="ix_: ndarray subclasses are not supported"
            ):
                fnp.ix_(arg)
    assert budget.flops_used == 0
    assert budget.op_log == []
    assert backend_calls == 0


def test_ix_public_docstring_exposes_cost_and_input_boundary():
    expected = """FLOP Cost
---------
sum(numel(outputs)) + sum(numel(Boolean inputs)) FLOPs

Input Support
-------------
Plain NumPy ndarray, FlopscopeArray, and non-ndarray array-like inputs are supported.
Foreign NumPy ndarray subclasses, including MaskedArray and memmap, raise TypeError."""

    assert fnp.ix_.__doc__ is not None
    assert expected in fnp.ix_.__doc__


@pytest.mark.parametrize("true_count", [0, 1, 4, 8])
def test_ix_bills_full_boolean_mask_scan_independent_of_popcount(true_count):
    mask = np.arange(8) < true_count

    result, cost = billed(mask)

    assert result[0].size == true_count
    assert cost - result[0].size == mask.size


def test_ix_bills_outputs_and_every_boolean_input_scan():
    rows = np.array([True, False, True, False])
    columns = np.array([1, 3, 5], dtype=np.int32)
    depths = np.array([True, False, True, False, True])

    result, cost = billed(rows, columns, depths)

    assert [array.size for array in result] == [2, 3, 3]
    assert cost == 8 + 9


def test_ix_integer_only_inputs_pay_output_construction():
    rows = np.array([1, 4], dtype=np.int32)
    columns = np.array([2, 5, 8], dtype=np.int32)

    result, cost = billed(rows, columns)

    assert [array.size for array in result] == [2, 3]
    assert cost == 5


@pytest.mark.parametrize(
    "args",
    [
        ([0, 2], [1, 3]),
        (np.array([True, False, True]), np.array([1, 4], dtype=np.int32)),
        ([], np.array([], dtype=bool)),
    ],
)
def test_ix_matches_numpy_value_shape_and_dtype(args):
    expected = np.ix_(*args)

    actual, _ = billed(*args)

    assert len(actual) == len(expected)
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(actual_array), expected_array)
        assert actual_array.shape == expected_array.shape
        assert actual_array.dtype == expected_array.dtype


def test_ix_resolves_a_non_ndarray_operand_once():
    class StatefulBooleanMask:
        calls = 0

        def __array__(self, dtype=None, copy=None):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("operand was resolved more than once")
            return np.array([True, False, True], dtype=dtype)

    mask = StatefulBooleanMask()

    result, cost = billed(mask)

    np.testing.assert_array_equal(np.asarray(result[0]), np.array([0, 2]))
    assert mask.calls == 1
    assert cost == 5


def test_ix_rejects_a_resolved_subclass_before_reading_metadata(monkeypatch):
    class HostileResolvedArray(np.ndarray):
        size_reads = 0

        @property
        def size(self):
            type(self).size_reads += 1
            return 1

    operand = object()
    resolved = np.array([0], dtype=np.int32).view(HostileResolvedArray)

    def hostile_asarray(value):
        assert value is operand
        return resolved

    monkeypatch.setattr(array_ops._np, "asarray", hostile_asarray)

    assert_rejected_without_charge(operand)

    assert HostileResolvedArray.size_reads == 0


def test_ix_rejects_stateful_dtype_subclass_before_execution():
    class StatefulDtypeMask(np.ndarray):
        dtype_reads = 0

        @property
        def dtype(self):
            self.dtype_reads += 1
            return np.dtype(bool if self.dtype_reads == 1 else np.int8)

    mask = np.array([True, False, True]).view(StatefulDtypeMask)

    assert_rejected_without_charge(mask)

    assert mask.dtype_reads == 0


def test_ix_rejects_lying_size_subclass_before_execution():
    class LyingSizeMask(np.ndarray):
        size_reads = 0

        @property
        def size(self):
            self.size_reads += 1
            return 0

    mask = np.ones(1000, dtype=bool).view(LyingSizeMask)

    assert_rejected_without_charge(mask)

    assert mask.size_reads == 0


def test_ix_rejects_lying_dtype_subclass_before_execution():
    class LyingFloatIndex(np.ndarray):
        dtype_reads = 0

        @property
        def dtype(self):
            self.dtype_reads += 1
            return np.dtype(np.int8)

    index = np.array([1.0, 2.0]).view(LyingFloatIndex)

    assert_rejected_without_charge(index)

    assert index.dtype_reads == 0


def test_ix_rejects_dynamic_dtype_override_before_execution():
    class DynamicBooleanIndex(np.ndarray):
        dtype_reads = 0

        def __getattribute__(self, name):
            if name == "dtype":
                type(self).dtype_reads += 1
                return np.dtype(bool)
            return super().__getattribute__(name)

    index = np.array([1, 0, 2], dtype=np.int32).view(DynamicBooleanIndex)

    assert_rejected_without_charge(index)

    assert DynamicBooleanIndex.dtype_reads == 0


def test_ix_rejects_metaclass_spoofed_dtype_override_before_execution():
    class SpoofedClassLookup(type):
        def __getattribute__(cls, name):
            if name == "__getattribute__":
                return np.ndarray.__getattribute__
            return super().__getattribute__(name)

    class MetaclassSpoofedIndex(np.ndarray, metaclass=SpoofedClassLookup):
        dtype_reads = 0

        def __getattribute__(self, name):
            if name == "dtype":
                type(self).dtype_reads += 1
                return np.dtype(bool)
            return super().__getattribute__(name)

    index = np.array([1, 0, 2], dtype=np.int32).view(MetaclassSpoofedIndex)

    assert_rejected_without_charge(index)

    assert MetaclassSpoofedIndex.dtype_reads == 0


def test_ix_rejects_metaclass_mro_property_without_calling_it():
    class SpoofedMroMeta(type):
        mro_reads = 0

        @property
        def __mro__(cls):  # pyright: ignore[reportIncompatibleMethodOverride]
            SpoofedMroMeta.mro_reads += 1
            return (cls, array_ops.FlopscopeArray, np.ndarray, object)

    class SpoofedMroIndex(np.ndarray, metaclass=SpoofedMroMeta):
        pass

    index = np.arange(3).view(SpoofedMroIndex)

    assert_rejected_without_charge(index)

    assert SpoofedMroMeta.mro_reads == 0


@pytest.mark.parametrize("dtype", [object, np.dtype("U1")], ids=["object", "string"])
def test_ix_nonnumeric_subclass_boundary_precedes_dtype_preflight(dtype):
    class NonnumericIndex(np.ndarray):
        dtype_reads = 0

        @property
        def dtype(self):
            type(self).dtype_reads += 1
            return np.ndarray.dtype.__get__(self)

    index = np.array(["x"], dtype=dtype).view(NonnumericIndex)

    assert_rejected_without_charge(index)

    assert NonnumericIndex.dtype_reads == 0


def test_ix_rejects_ndim_hook_before_it_can_install_a_boolean_scan():
    class DelayedBooleanIndex(np.ndarray):
        ndim_reads = 0
        nonzero_calls = 0

        @property
        def ndim(self):
            type(self).ndim_reads += 1
            type(self).dtype = property(  # pyright: ignore[reportAttributeAccessIssue]
                lambda _: np.dtype(bool)
            )
            return 1

        def nonzero(self):
            type(self).nonzero_calls += 1
            return super().nonzero()

    index = np.zeros(100_000, dtype=np.int64).view(DelayedBooleanIndex)

    assert_rejected_without_charge(index)

    assert DelayedBooleanIndex.ndim_reads == 0
    assert DelayedBooleanIndex.nonzero_calls == 0


def test_ix_rejects_dtype_hook_before_it_can_expand_the_scanned_buffer():
    class RetypingBooleanIndex(np.ndarray):
        dtype_reads = 0

        @property
        def dtype(self):
            type(self).dtype_reads += 1
            np.ndarray.dtype.__set__(  # pyright: ignore[reportAttributeAccessIssue]
                self, np.dtype(bool)
            )
            return np.ndarray.dtype.__get__(self)

    element_count = 16
    index = np.zeros(element_count, dtype=np.int64).view(RetypingBooleanIndex)

    assert_rejected_without_charge(index)

    assert RetypingBooleanIndex.dtype_reads == 0
    assert np.asarray(index).dtype == np.int64
    assert np.asarray(index).size == element_count


@pytest.mark.parametrize(
    "hook_name", ["__array_function__", "__array_finalize__", "nonzero", "reshape"]
)
def test_ix_rejects_overridden_execution_hook_before_calling_it(hook_name):
    class UnsafeIndex(np.ndarray):
        hook_calls = 0

    index = np.array([0, 1, 2], dtype=np.int32).view(UnsafeIndex)

    def hostile_hook(self, *args, **kwargs):
        type(self).hook_calls += 1
        raise AssertionError(f"{hook_name} executed")

    setattr(UnsafeIndex, hook_name, hostile_hook)

    assert_rejected_without_charge(index)

    assert UnsafeIndex.hook_calls == 0


def test_ix_rejects_trivial_integer_subclass():
    class HonestIntegerIndex(np.ndarray):
        pass

    index = np.array([0, 2, 4], dtype=np.int32).view(HonestIntegerIndex)

    assert_rejected_without_charge(index)


def test_ix_rejects_masked_boolean_array():
    mask = np.ma.array([True, False, True], mask=[True, False, False])

    assert_rejected_without_charge(mask)


def test_ix_rejects_masked_integer_array():
    index = np.ma.array([0, 1, 2], mask=[False, True, False])

    assert_rejected_without_charge(index)


def test_ix_rejects_masked_array_subclass():
    class HonestMaskedIndex(np.ma.MaskedArray):
        pass

    index = np.ma.array([0, 1, 2], mask=[False, True, False]).view(HonestMaskedIndex)

    assert_rejected_without_charge(index)


def test_ix_rejects_memmap(tmp_path):
    path = tmp_path / "index.dat"
    index = np.memmap(path, dtype=np.int32, mode="w+", shape=(3,))
    index[:] = [0, 2, 4]

    assert_rejected_without_charge(index)


def test_ix_strips_flopscope_subclass_without_calling_view_override():
    class HostileFlopscopeIndex(array_ops.FlopscopeArray):
        view_calls = 0

        def view(self, *args, **kwargs):
            type(self).view_calls += 1
            raise AssertionError("overrideable view executed")

    index = np.array([True, False, True]).view(HostileFlopscopeIndex)

    actual, cost = billed(index)

    np.testing.assert_array_equal(np.asarray(actual[0]), np.array([0, 2]))
    assert cost == 5
    assert HostileFlopscopeIndex.view_calls == 0


def test_ix_rejects_subclass_spoofing_flopscope_class_without_calling_hook():
    class SpoofedFlopscopeIndex(np.ndarray):
        class_reads = 0

        @property
        def __class__(self):  # pyright: ignore[reportIncompatibleMethodOverride]
            type(self).class_reads += 1
            return array_ops.FlopscopeArray

    index = np.arange(3).view(SpoofedFlopscopeIndex)

    assert_rejected_without_charge(index)

    assert SpoofedFlopscopeIndex.class_reads == 0


def test_ix_rejects_masked_array_instance_execution_hook():
    index = np.ma.array([True, False, True])
    nonzero_calls = 0

    def hostile_nonzero():
        nonlocal nonzero_calls
        nonzero_calls += 1
        return (np.array([], dtype=np.intp),)

    index.nonzero = hostile_nonzero

    assert_rejected_without_charge(index)

    assert nonzero_calls == 0


def test_ix_rejects_masked_array_filled_shadow_before_scan():
    index = np.ma.array([True, False, True])
    filled_calls = 0

    def hostile_filled(fill_value):
        nonlocal filled_calls
        filled_calls += 1
        return np.zeros(100_000, dtype=bool)

    index.filled = hostile_filled

    assert_rejected_without_charge(index)

    assert filled_calls == 0


def test_ix_bills_a_lying_backend_output_at_its_real_dtype(monkeypatch):
    class LyingOutput(np.ndarray):
        dtype_reads = 0

        @property
        def dtype(self):
            self.dtype_reads += 1
            return np.dtype(np.int8)

    original_ix = array_ops._np.ix_
    output = np.array([1.0, 2.0]).view(LyingOutput)

    def ix_with_lying_output(*args, **kwargs):
        original_ix(*args, **kwargs)
        return (output,)

    monkeypatch.setattr(array_ops._np, "ix_", ix_with_lying_output)
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as budget:
        result = fnp.ix_(np.array([1, 2], dtype=np.int32))

    np.testing.assert_array_equal(np.asarray(result[0]), np.array([1.0, 2.0]))
    assert budget.op_log[-1].resolved_dtype == "float64"
    assert output.dtype_reads == 0


def test_ix_numpy_failure_does_not_charge_or_record():
    invalid = np.array([[0, 1], [2, 3]], dtype=np.int32)
    with pytest.raises(ValueError) as numpy_error:
        np.ix_(invalid)  # pyright: ignore[reportCallIssue, reportArgumentType]

    with flops.BudgetContext(flop_budget=10**9, quiet=True) as budget:
        with pytest.raises(
            type(numpy_error.value), match=re.escape(str(numpy_error.value))
        ):
            fnp.ix_(invalid)  # pyright: ignore[reportCallIssue]

    assert budget.flops_used == 0
    assert budget.op_log == []


def test_ix_attributes_numpy_backend_time(monkeypatch):
    original_ix = array_ops._np.ix_

    def delayed_ix(*args, **kwargs):
        time.sleep(0.03)
        return original_ix(*args, **kwargs)

    monkeypatch.setattr(array_ops._np, "ix_", delayed_ix)

    with flops.BudgetContext(flop_budget=10**9, quiet=True) as budget:
        fnp.ix_(np.array([0, 2], dtype=np.int32))

    record = budget.op_log[-1]
    assert record.op_name == "ix_"
    assert record.flopscope_backend_duration_s is not None
    assert record.flopscope_backend_duration_s >= 0.02
    assert record.flopscope_overhead_duration_s is not None
    assert record.flopscope_overhead_duration_s < 0.02


def test_ix_attributes_boundary_preflight_time_to_flopscope_overhead(monkeypatch):
    original_classify = array_ops._ix_argument_is_flopscope_array
    classification_calls = 0

    def delayed_first_classification(arg):
        nonlocal classification_calls
        classification_calls += 1
        if classification_calls == 1:
            time.sleep(0.04)
        return original_classify(arg)

    monkeypatch.setattr(
        array_ops, "_ix_argument_is_flopscope_array", delayed_first_classification
    )

    with flops.BudgetContext(flop_budget=10**9, quiet=True) as budget:
        fnp.ix_(np.array([0, 2], dtype=np.int32))

    record = budget.op_log[-1]
    assert classification_calls == 2
    assert record.flopscope_overhead_duration_s is not None
    assert record.flopscope_overhead_duration_s >= 0.03
    assert budget.flopscope_overhead_time_s >= 0.03
    assert budget.residual_wall_time_s is not None
    assert budget.residual_wall_time_s < 0.02
