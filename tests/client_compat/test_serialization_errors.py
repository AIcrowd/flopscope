"""Operator/op dispatch must raise a clear RemoteSerializationError naming the
offending type, not a raw msgpack 'can not serialize' TypeError.

The function-dispatch proxy already does this; the operator path (_dispatch_op)
leaked the raw error (this is how the RemoteScalar bug surfaced cryptically).
"""

from __future__ import annotations

import pytest

import flopscope as fnp
from flopscope.errors import RemoteSerializationError


def test_operator_dispatch_unserializable_arg_is_clear():
    a = fnp.array([1.0, 2.0, 3.0])
    with pytest.raises(RemoteSerializationError) as ei:
        a + (lambda x: x)  # a function is not serializable
    assert "function" in str(ei.value)
