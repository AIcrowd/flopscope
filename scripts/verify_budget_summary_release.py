#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import os
import select
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import msgpack
import zmq
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

TOKEN_TIMEOUT_S = 10.0
CLIENT_TIMEOUT_S = 15.0

BASE_KEYS = {
    "flop_budget",
    "flops_used",
    "flops_remaining",
    "operations",
    "wall_time_s",
    "flopscope_backend_time_s",
    "flopscope_overhead_time_s",
    "residual_wall_time_s",
}
OP_KEYS = {
    "flop_cost",
    "calls",
    "flopscope_backend_time_s",
    "flopscope_overhead_time_s",
}
NS_KEYS = {
    "flops_used",
    "calls",
    "flopscope_backend_time_s",
    "flopscope_overhead_time_s",
    "operations",
}


def require(condition: bool, message: object) -> None:
    if not condition:
        raise AssertionError(message)


def require_exact_dependency(
    raw_requirements: list[str],
    *,
    distribution: str,
    version: str,
    required_extra: str | None = None,
) -> None:
    expected_name = canonicalize_name(distribution)
    expected_marker = None if required_extra is None else f'extra == "{required_extra}"'
    parsed: list[Requirement] = []
    for raw in raw_requirements:
        try:
            parsed.append(Requirement(raw))
        except InvalidRequirement as exc:
            raise AssertionError(f"invalid Requires-Dist entry: {raw!r}") from exc

    for requirement in parsed:
        if canonicalize_name(requirement.name) != expected_name:
            continue
        if requirement.extras or requirement.url is not None:
            continue
        if str(requirement.specifier) != f"=={version}":
            continue
        marker = None if requirement.marker is None else str(requirement.marker)
        if marker == expected_marker:
            return

    marker_suffix = (
        " without a marker"
        if required_extra is None
        else f' with marker extra == "{required_extra}"'
    )
    raise AssertionError(
        f"Requires-Dist must pin exactly {distribution}=={version}{marker_suffix}: "
        f"{raw_requirements}"
    )


def request(url: str, payload: dict) -> dict:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDTIMEO, 5_000)
    sock.setsockopt(zmq.RCVTIMEO, 5_000)
    try:
        sock.connect(url)
        sock.send(msgpack.packb(payload, use_bin_type=True))
        response = msgpack.unpackb(sock.recv(), raw=False, strict_map_key=False)
        require(isinstance(response, dict), f"expected mapping response: {response!r}")
        return response
    finally:
        sock.close()
        ctx.term()


def _stderr(proc: subprocess.Popen) -> str:
    if proc.stderr is None:
        return ""
    raw = proc.stderr.read()
    if isinstance(raw, bytes):
        return raw.decode(errors="replace")
    return str(raw)


def read_control_token(
    proc: subprocess.Popen, read_fd: int, *, timeout_s: float = TOKEN_TIMEOUT_S
) -> str:
    deadline = time.monotonic() + timeout_s
    raw_token = bytearray()
    while b"\n" not in raw_token:
        returncode = proc.poll()
        if returncode is not None:
            raise AssertionError(
                f"server exited while providing control token ({returncode}): "
                f"{_stderr(proc)}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                f"timed out waiting for control token after {timeout_s:g}s"
            )
        readable, _, _ = select.select([read_fd], [], [], min(remaining, 0.05))
        if not readable:
            continue
        chunk = os.read(read_fd, 66 - len(raw_token))
        if not chunk:
            raise AssertionError("server closed control-token pipe before newline")
        raw_token.extend(chunk)
        if len(raw_token) > 65:
            raise AssertionError("server control token exceeded 64 bytes")

    token_bytes, newline, trailing = bytes(raw_token).partition(b"\n")
    require(newline == b"\n" and not trailing, "malformed control-token response")
    try:
        token = token_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise AssertionError("control token was not ASCII") from exc
    require(len(token) == 64, f"expected 64-character control token, got {token!r}")
    require(
        all(character in "0123456789abcdef" for character in token),
        f"control token was not lowercase hexadecimal: {token!r}",
    )
    return token


@contextlib.contextmanager
def server(
    python: str, expected_version: str, *, tokenized: bool
) -> Iterator[tuple[str, str | None]]:
    temp = tempfile.TemporaryDirectory(prefix="whest-release-server-")
    url = f"ipc://{Path(temp.name) / 'flopscope.sock'}"
    read_fd = write_fd = None
    proc: subprocess.Popen | None = None
    command = [python, "-m", "flopscope_server", "--url", url, "--timeout", "30"]
    popen_kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE}
    try:
        if tokenized:
            read_fd, write_fd = os.pipe()
            command.extend(["--token-fd", str(write_fd)])
            popen_kwargs["pass_fds"] = (write_fd,)
        try:
            proc = subprocess.Popen(command, **popen_kwargs)
        finally:
            if write_fd is not None:
                os.close(write_fd)
                write_fd = None

        token = None
        if read_fd is not None:
            try:
                token = read_control_token(proc, read_fd)
            finally:
                os.close(read_fd)
                read_fd = None
        deadline = time.monotonic() + 10
        while True:
            if proc.poll() is not None:
                raise AssertionError(f"server exited during startup: {_stderr(proc)}")
            try:
                hello = request(
                    url,
                    {
                        "op": "hello",
                        "args": None,
                        "kwargs": {"client_version": expected_version},
                    },
                )
                break
            except (zmq.ZMQError, AssertionError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        require(hello.get("status") == "ok", hello)
        require(hello.get("server_version") == expected_version, hello)
        capabilities = hello.get("capabilities")
        if not isinstance(capabilities, list):
            raise AssertionError(hello)
        require("authoritative_budget_summary_v1" in capabilities, hello)
        yield url, token
    finally:
        try:
            if read_fd is not None:
                os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                else:
                    proc.wait(timeout=5)
        finally:
            temp.cleanup()


CLIENT_PROGRAM = r"""
import msgpack
import sys
import flopscope as f
from flopscope._connection import get_connection

mode = sys.argv[1]
if mode == "global":
    a = f.array([1.0, 2.0, 3.0, 4.0])
    b = f.array([5.0, 6.0, 7.0, 8.0])
    f.add(a, b)
    output = f.budget_summary_dict(by_namespace=True)
elif mode == "summary":
    output = f.budget_summary_dict(by_namespace=True)
elif mode == "context":
    with f.BudgetContext(flop_budget=1000, namespace="context", quiet=True) as ctx:
        a = f.array([1.0, 2.0, 3.0, 4.0])
        f.add(a, a)
        live = ctx.summary_dict(by_namespace=True)
    global_summary = f.budget_summary_dict(by_namespace=True)

    def fail_send_recv(*args, **kwargs):
        raise AssertionError("closed context summary unexpectedly used RPC")

    get_connection().send_recv = fail_send_recv
    closed = ctx.summary_dict(by_namespace=True)
    damaged = ctx.summary_dict(by_namespace=True)
    damaged["operations"].clear()
    damaged["by_namespace"]["context"]["operations"].clear()
    reread = ctx.summary_dict(by_namespace=True)
    output = {"live": live, "closed": closed, "reread": reread,
              "global": global_summary,
              "properties": {
                  "wall_time_s": ctx.wall_time_s,
                  "flopscope_backend_time_s": ctx.flopscope_backend_time_s,
                  "flopscope_overhead_time_s": ctx.flopscope_overhead_time_s,
                  "residual_wall_time_s": ctx.residual_wall_time_s,
              }}
else:
    raise AssertionError(mode)
sys.stdout.buffer.write(msgpack.packb(output, use_bin_type=True))
"""


def client(python: str, url: str, mode: str) -> dict:
    env = os.environ.copy()
    env["FLOPSCOPE_SERVER_URL"] = url
    env.pop("PYTHONPATH", None)
    try:
        run = subprocess.run(
            [python, "-c", CLIENT_PROGRAM, mode],
            check=False,
            capture_output=True,
            env=env,
            cwd=tempfile.gettempdir(),
            timeout=CLIENT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"client timed out after {CLIENT_TIMEOUT_S:g}s in {mode!r} mode"
        ) from exc
    if run.returncode:
        raise AssertionError(
            f"client failed ({run.returncode}): {run.stderr.decode(errors='replace')}"
        )
    response = msgpack.unpackb(run.stdout, raw=False, strict_map_key=False)
    require(isinstance(response, dict), f"client returned non-mapping: {response!r}")
    return response


def assert_timing(summary: dict) -> None:
    for key in (
        "wall_time_s",
        "flopscope_backend_time_s",
        "flopscope_overhead_time_s",
        "residual_wall_time_s",
    ):
        require(summary[key] >= 0, (key, summary))
    total = (
        summary["flopscope_backend_time_s"]
        + summary["flopscope_overhead_time_s"]
        + summary["residual_wall_time_s"]
    )
    require(
        abs(summary["wall_time_s"] - total) <= max(1e-6, summary["wall_time_s"] * 1e-5),
        summary,
    )


def assert_summary(
    summary: dict,
    *,
    budget: int,
    used: int,
    calls: int,
    namespaces: dict[str, tuple[int, int]],
) -> None:
    require(set(summary) == BASE_KEYS | {"by_namespace"}, summary)
    require(summary["flop_budget"] == budget, summary)
    require(summary["flops_used"] == used, summary)
    require(summary["flops_remaining"] == budget - used, summary)
    require(set(summary["operations"]) == {"add"}, summary)
    require(set(summary["operations"]["add"]) == OP_KEYS, summary)
    require(summary["operations"]["add"]["flop_cost"] == used, summary)
    require(summary["operations"]["add"]["calls"] == calls, summary)
    require(set(summary["by_namespace"]) == set(namespaces), summary)
    for name, (namespace_used, namespace_calls) in namespaces.items():
        bucket = summary["by_namespace"][name]
        require(set(bucket) == NS_KEYS, bucket)
        require(bucket["flops_used"] == namespace_used, bucket)
        require(bucket["calls"] == namespace_calls, bucket)
        require(set(bucket["operations"]) == {"add"}, bucket)
        operation = bucket["operations"]["add"]
        require(set(operation) == OP_KEYS, operation)
        require(operation["flop_cost"] == namespace_used, operation)
        require(operation["calls"] == namespace_calls, operation)
    assert_timing(summary)


def resource_accounting(summary: dict) -> dict:
    return {
        "flop_budget": summary["flop_budget"],
        "flops_used": summary["flops_used"],
        "flops_remaining": summary["flops_remaining"],
        "operations": {
            name: {"flop_cost": values["flop_cost"], "calls": values["calls"]}
            for name, values in summary["operations"].items()
        },
        "by_namespace": {
            name: {
                "flops_used": values["flops_used"],
                "calls": values["calls"],
                "operations": {
                    operation: {
                        "flop_cost": details["flop_cost"],
                        "calls": details["calls"],
                    }
                    for operation, details in values["operations"].items()
                },
            }
            for name, values in summary["by_namespace"].items()
        },
    }


def control(url: str, token: str, op: str, **kwargs: object) -> dict:
    payload = {"op": op, "args": None, "kwargs": {**kwargs, "control_token": token}}
    response = request(url, payload)
    require(response.get("status") == "ok", response)
    return response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-python", required=True)
    parser.add_argument("--client-python", required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    with server(args.server_python, args.expected_version, tokenized=True) as (
        url,
        token,
    ):
        if token is None:
            raise AssertionError("tokenized server did not provide a token")
        control_token = token
        control(url, control_token, "budget_open", flop_budget=1000, namespace="first")
        first = client(args.client_python, url, "global")
        assert_summary(
            first, budget=1000, used=8, calls=1, namespaces={"first": (8, 1)}
        )
        accounting_before = resource_accounting(first)
        unauthorized = request(
            url,
            {
                "op": "budget_summary_reset",
                "args": None,
                "kwargs": {"control_token": "wrong-token"},
            },
        )
        require(unauthorized.get("status") == "error", unauthorized)
        require(
            unauthorized.get("error_type") == "UnauthorizedControlError", unauthorized
        )
        after_unauthorized = client(args.client_python, url, "summary")
        assert_summary(
            after_unauthorized,
            budget=1000,
            used=8,
            calls=1,
            namespaces={"first": (8, 1)},
        )
        require(
            resource_accounting(after_unauthorized) == accounting_before,
            (accounting_before, resource_accounting(after_unauthorized)),
        )
        control(url, control_token, "budget_close")

        control(url, control_token, "budget_open", flop_budget=1000, namespace="second")
        second = client(args.client_python, url, "global")
        assert_summary(
            second,
            budget=2000,
            used=16,
            calls=2,
            namespaces={"first": (8, 1), "second": (8, 1)},
        )
        control(url, control_token, "budget_close")
        closed_session = client(args.client_python, url, "summary")
        assert_summary(
            closed_session,
            budget=2000,
            used=16,
            calls=2,
            namespaces={"first": (8, 1), "second": (8, 1)},
        )
        control(url, control_token, "budget_summary_reset")
        empty = client(args.client_python, url, "summary")
        require(set(empty) == BASE_KEYS | {"by_namespace"}, empty)
        require(
            empty["flop_budget"]
            == empty["flops_used"]
            == empty["flops_remaining"]
            == 0,
            empty,
        )
        require(empty["operations"] == {}, empty)
        require(empty["by_namespace"] == {}, empty)

    with server(args.server_python, args.expected_version, tokenized=False) as (url, _):
        context = client(args.client_python, url, "context")
        for key in ("live", "closed", "reread", "global"):
            assert_summary(
                context[key],
                budget=1000,
                used=8,
                calls=1,
                namespaces={"context": (8, 1)},
            )
        require(
            context["reread"]["operations"] == context["closed"]["operations"],
            context,
        )
        require(
            context["reread"]["by_namespace"] == context["closed"]["by_namespace"],
            context,
        )
        for key, value in context["properties"].items():
            require(value == context["closed"][key], (key, context))
        require(context["properties"]["flopscope_overhead_time_s"] > 0, context)

    print(f"budget-summary release contract passed for {args.expected_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
