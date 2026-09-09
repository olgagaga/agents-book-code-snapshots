"""Exercise 1: Per-tool timeout.

The chapter's `_wait_for_tools` blocks until every submitted future is done,
no matter how long that takes. A misbehaving tool can therefore freeze the
entire turn — the model is waiting on the future, the loop is waiting on the
model, and the user is staring at a dead terminal.

The fix is a per-future timeout, enforced from the *wait* side rather than
from inside the worker. CPython cannot preemptively kill a running thread,
so the simplest version (this exercise) accepts that timed-out tools keep
running in the background; we just stop *waiting* for them and feed the
model a synthetic `[timed out after Ns]` observation in place of the real
return value. Exercise 2 takes the next step: a cooperative cancel signal
that the tool itself can honour.

Two functions change shape:

  - `_run_tool(tc, timeout=None)` accepts a timeout parameter for symmetry,
    but the simple version does not enforce it inside the worker.
  - `_wait_for_tools(futures, get_injections, timeout=None)` now returns
    `(overrides, pending_injections)`. `overrides` maps `id(future)` to a
    synthetic observation string for any future that timed out; the caller
    uses the override when present and `future.result()` otherwise.

The patch to `agent/loop.py` is straightforward — see the call-site change
in `_apply_to_agent_step` at the bottom of this file. Then pick a sensible
default timeout per tool, or accept it as a `Tool` dataclass field.
"""

import concurrent.futures
import time
from typing import Callable

_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="ex1")


def _heartbeat(get_injections: Callable[[], list[str]] | None) -> list[dict]:
    if get_injections is None:
        return []
    return [
        {"role": "user", "content": f"[mid-turn user note] {line}"}
        for line in get_injections()
    ]


def _run_tool(tc, timeout: float | None = None) -> str:
    """Same body as the chapter's `_run_tool`, with a `timeout` parameter
    accepted but unused. Exercise 2 wires it through to a cooperative
    cancel event so the tool can honour it from inside."""
    # In a real port, this calls `find_tool(tc.name)` and `tool.run(tc.args)`.
    # The demo at the bottom of the file substitutes a callable directly.
    return tc["fn"](tc["args"])


def _wait_for_tools(
    futures: list[tuple[concurrent.futures.Future, dict]],
    get_injections: Callable[[], list[str]] | None,
    timeout: float | None = None,
) -> tuple[dict[int, str], list[dict]]:
    """Block until every future is either done or has been marked as timed
    out. Returns `(overrides, pending_injections)` where `overrides[id(fut)]`
    is the synthetic observation for a timed-out future."""
    started = time.monotonic()
    overrides: dict[int, str] = {}
    pending: list[dict] = []

    def remaining() -> list[tuple[concurrent.futures.Future, dict]]:
        return [(f, tc) for f, tc in futures
                if not f.done() and id(f) not in overrides]

    while remaining():
        pending.extend(_heartbeat(get_injections))
        if timeout is not None:
            elapsed = time.monotonic() - started
            if elapsed > timeout:
                for f, _ in remaining():
                    f.cancel()  # succeeds only if the worker has not started
                    overrides[id(f)] = f"[timed out after {elapsed:.1f}s]"
                break
        time.sleep(0.1)
    pending.extend(_heartbeat(get_injections))
    return overrides, pending


def _apply_to_agent_step() -> str:
    """Sketch of how the call site in `agent_step` changes. The for-loop
    that collects observations becomes a small branch:

        overrides, pending_injections = _wait_for_tools(
            futures, get_injections, timeout=10.0,
        )
        for future, tc in futures:
            if id(future) in overrides:
                observation = overrides[id(future)]
            else:
                observation = _truncate_observation(future.result())
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": observation,
            })
        messages.extend(pending_injections)

    The timeout itself should live somewhere honest — either as a constant
    near `MAX_ITERATIONS`, a per-`Tool` field, or a parameter on the chat
    entry point. Picking 10s as a global default is enough to make the
    exercise demonstrable without inventing new policy.
    """
    return ""


if __name__ == "__main__":
    # Demo: one fast tool, one tool that exceeds the timeout.
    def fast_tool(args: dict) -> str:
        time.sleep(0.2)
        return f"ok in {args['seconds']}s"

    def slow_tool(args: dict) -> str:
        time.sleep(args["seconds"])
        return f"slept for {args['seconds']}s"

    calls = [
        {"id": "tc_fast", "fn": fast_tool, "args": {"seconds": 0.2}},
        {"id": "tc_slow", "fn": slow_tool, "args": {"seconds": 5.0}},
    ]
    futures = [(_POOL.submit(_run_tool, tc), tc) for tc in calls]

    overrides, pending = _wait_for_tools(futures, get_injections=None, timeout=1.0)

    print("results:")
    for fut, tc in futures:
        if id(fut) in overrides:
            obs = overrides[id(fut)]
        else:
            obs = fut.result()
        print(f"  {tc['id']}: {obs}")
    print(f"pending injections: {pending}")

    assert overrides, "the slow tool should have timed out"
    assert any("timed out after" in v for v in overrides.values())
    print("\nok — slow tool produced a synthetic observation; fast tool returned normally")
