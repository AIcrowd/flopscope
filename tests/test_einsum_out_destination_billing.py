"""``out=`` on ``fnp.einsum``: the destination write is a copy, and copies are metered.

Every other contraction with a destination hands ``out=`` to numpy, which
writes the caller's buffer directly as part of the kernel -- one write per
output element, already paid for inside the contraction's own cost. That is
why ``matmul(a, b, out=d)`` bills exactly what ``matmul(a, b)`` bills, and why
the cost model calls such a destination price-neutral: it is the buffer numpy
would have allocated anyway.

``einsum`` is the one contraction that does not forward ``out=``. numpy writes
the result into a buffer of its own, and the wrapper then copies that buffer
into the caller's destination by hand. That copy is a second, full,
materialising pass over the data -- ``dest.size`` elements written
sequentially -- and it is not a buffer numpy would have allocated anyway. It
is an extra one, on top of the one already paid for.

The cost model has one rule for that, and it is not a special case for
contractions (``docs/reference/cost-model.md``, "every byte written is
metered", decision-procedure step 2):

    Any op that writes a new buffer is charged at least 1 per element
    written, whether the values it writes are computed, copied, replicated,
    or a repeated constant.

    ... a participant who can move arbitrary amounts of data for free can
    launder real compute through a materializing copy chain.

The copy bills zero. So ``einsum(subs, *ops, out=dest)`` is a materialising
copy of arbitrary size at no charge, reachable in one call: the identity
contraction ``einsum("ij->ij", src, out=dest)`` computes nothing, copies
``src.size`` elements into ``dest`` correctly, and bills 0, where the same
write spelled ``fnp.copyto(dest, src)`` bills ``src.size``. The transposing
form ``einsum("ij->ji", src, out=dest)`` materialises a full reorder on the
same terms.

Without a destination the identity form is correct and free: numpy returns a
view, nothing is written, and weight 0 is exactly right (decision-procedure
step 1). It is the destination that turns the view into a write, so it is the
destination that has to be priced.

The pins here are exact-equality FLOP counts, not timings: the price of the
same work must not depend on which of two spellings the caller picked. The
last test covers the other half of the same defect -- the bucket the copy's
wall lands in -- and states it as a mechanism (an op record exists and carries
backend time) rather than as one wall clock beating another, because the
timing-inequality form of that assertion already flakes elsewhere in this
suite under parallel load.
"""

import numpy as np
import pytest

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights

N = 32


def _billed(fn):
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn()
        return b.flops_used


def _square(dtype, n=N, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a: np.ndarray = rng.standard_normal((n, n))
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        a = a + 1j * rng.standard_normal((n, n))
    return np.asarray(a, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_einsum_destination_costs_what_the_same_copy_costs(dtype):
    """``einsum(out=d)`` == ``einsum()`` + ``copyto(d, result)``, exactly.

    The two spellings run the same numpy calls in the same order and produce
    the same bytes in the same buffer. Charging less for the shorter one
    prices a call by how it was written rather than by what it did.
    """
    load_weights()
    a, b = _square(dtype), _square(dtype, seed=1)

    def bare():
        fnp.einsum("ij,jk->ik", a, b)

    def with_dest():
        fnp.einsum("ij,jk->ik", a, b, out=np.empty((N, N), dtype=dtype))

    def spelled_out():
        r = fnp.einsum("ij,jk->ik", a, b)
        fnp.copyto(np.empty((N, N), dtype=dtype), r, casting="unsafe")

    assert _billed(with_dest) == _billed(spelled_out)
    assert _billed(with_dest) - _billed(bare) == _billed(spelled_out) - _billed(bare)
    assert _billed(with_dest) > _billed(bare)  # the extra pass is not free


@pytest.mark.parametrize("subs", ["ij->ij", "ij->ji"])
def test_identity_einsum_with_destination_is_not_a_free_copy(subs):
    """A contraction that computes nothing still writes ``dest.size`` elements.

    ``einsum("ij->ij", src, out=dest)`` is a memcpy and ``einsum("ij->ji", src,
    out=dest)`` is a materialised transpose. Both are correct, both move every
    element, and a caller who can spell either for nothing has an unmetered
    copy channel of arbitrary width.
    """
    load_weights()
    src = _square(np.float64)
    expected = np.einsum(subs, src)

    dest = np.zeros((N, N))
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        returned = fnp.einsum(subs, src, out=dest)
        billed = b.flops_used

    # the write really happened -- cost assertions are blind to an unwritten
    # destination, so content is asserted separately (see test_matmul_out).
    assert np.array_equal(np.asarray(dest), expected)
    assert returned is dest

    assert billed == _billed(lambda: fnp.copyto(np.zeros((N, N)), src))


def test_identity_einsum_without_destination_stays_free():
    """The other direction of the same rule: no destination, no write, no charge.

    numpy returns a view here, so nothing is materialised and weight 0 is
    correct. Pricing the destination must not leak into the viewing form.
    """
    load_weights()
    src = _square(np.float64)
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        r = fnp.einsum("ij->ij", src)
        assert np.may_share_memory(np.asarray(r), src)
        assert b.flops_used == 0


def test_wider_destination_still_bills_the_wider_rate():
    """The destination charge rides on top of the existing rate doctrine.

    ``store_billing_dtypes`` already folds the destination into the rate
    (#156). Pricing the write must not disturb that: a float64 destination on
    a float32 contraction stays strictly dearer than a float32 one.
    """
    load_weights()
    a, b = _square(np.float32), _square(np.float32, seed=1)
    narrow = _billed(
        lambda: fnp.einsum("ij,jk->ik", a, b, out=np.empty((N, N), np.float32))
    )
    wide = _billed(
        lambda: fnp.einsum("ij,jk->ik", a, b, out=np.empty((N, N), np.float64))
    )
    assert wide > narrow


def test_destination_write_is_recorded_and_reaches_backend():
    """The other half of the same defect: which bucket the copy's wall lands in.

    ``_call_numpy`` exists so that every numpy call inside a counted-op
    wrapper reports its duration as backend time; the docstring says every
    such call MUST go through it. This copy did not, and being outside the
    contraction's ``deduct`` block did not send it to residual either -- the
    enclosing ``@_counted_wrapper`` bills its whole non-backend remainder to
    ``flopscope_overhead_time_s``. Both totals are subtracted when
    ``residual_wall_time_s`` is formed, so the write showed up in no meter the
    caller reads.

    Asserted as a mechanism rather than as a race between two wall clocks: an
    op record has to exist for the write, priced at the elements written, and
    its backend duration has to be non-zero, which happens only if the copy
    went through ``_call_numpy`` while that op's timer was live. Restoring the
    bare ``_np.copyto`` -- inside the block or outside it -- leaves the record
    at zero backend, and dropping the charge leaves no record at all. The
    contraction here is an identity, so numpy returns a view in microseconds
    and the 32 MiB copy is milliseconds: the reading being checked against
    zero is three orders of magnitude away from it.
    """
    load_weights()
    src = np.ones((2048, 2048), dtype=np.float64)  # 32 MiB
    dest = np.zeros((2048, 2048), dtype=np.float64)
    f.budget_reset()
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fnp.einsum("ij->ij", src, out=dest)

    writes = [r for r in b.op_log if r.op_name == "copyto"]
    assert len(writes) == 1, [r.op_name for r in b.op_log]
    # ``flop_cost`` on the record is the rate-adjusted charge, so it is
    # compared against what ``fnp.copyto`` charges for this very write rather
    # than against a hand-multiplied ``dest.size``.
    assert writes[0].flop_cost == _billed(
        lambda: fnp.copyto(np.zeros((2048, 2048)), src, casting="unsafe")
    )
    backend_s = writes[0].flopscope_backend_duration_s
    assert backend_s is not None and backend_s > 0, writes[0]
