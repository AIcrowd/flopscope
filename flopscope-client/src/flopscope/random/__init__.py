"""flopscope.numpy.random — random number generation submodule proxy."""

from __future__ import annotations

from flopscope._connection import get_connection
from flopscope._dispatch import timed_dispatch
from flopscope._getattr import make_module_getattr
from flopscope._protocol import encode_request
from flopscope._registry import iter_proxyable
from flopscope._remote_array import (
    RemoteArray,
    RemoteScalar,
    _encode_arg,
    _result_from_response,
)


def _make_random_proxy(op_name: str):
    """Create a proxy function for ``random.<op_name>``."""
    qualified = f"random.{op_name}"

    def proxy(*args, **kwargs):
        conn = get_connection()
        encoded_args = [_encode_arg(a) for a in args]
        encoded_kwargs = {k: _encode_arg(v) for k, v in kwargs.items()}
        resp = conn.send_recv(
            encode_request(qualified, args=encoded_args, kwargs=encoded_kwargs)
        )
        return _result_from_response(resp)

    proxy.__name__ = op_name
    proxy.__qualname__ = f"random.{op_name}"
    return timed_dispatch(proxy)


# Auto-generate proxies for all non-blacklisted random.* functions
_random_ops = iter_proxyable(prefix="random.")
for _qname in _random_ops:
    _short = _qname[len("random.") :]
    locals()[_short] = _make_random_proxy(_short)

# Cleanup loop vars from module namespace
del _qname, _short, _random_ops

# Fall-through for unknown / blacklisted names
__getattr__ = make_module_getattr("random.", "flopscope.numpy.random")
