"""Exercise 4: A virtual `wait_for` tool.

The chapter's "virtual tool-call pattern" section described a tool with
a schema but no `run` function — the harness intercepts the call by name
and acts on it directly. This exercise builds the smallest useful
instance: `wait_for(seconds, reason)`. The model can express "remind me
in N seconds about R" as a structured tool call instead of free text the
harness has to parse.

The shape of the call:

  - The model issues `wait_for(seconds=10, reason="check on the build")`.
  - `_run_tool` recognises the name, calls the harness handler instead
    of looking the tool up in the registry, and returns immediately with
    the observation `"wait registered: 10s for 'check on the build'"`.
    The model sees that observation in the very next iteration and
    typically replies with a brief acknowledgement.
  - The harness records `(wake_at, reason)` in a module-level list.
  - On every heartbeat tick, the harness scans the pending-wait list for
    any whose `wake_at` has passed. Each expired wait is consumed and
    injected as a synthetic user note: `[wait_for elapsed: <reason>]`.
    The model sees that note on its next iteration and decides what to
    do about it.

Two things to notice. The first is that there is no `run` function — the
tool's behaviour is entirely the harness's responsibility. The second is
that the *registration* and *firing* happen at different times, on
different threads (the tool-call thread vs. the main thread's heartbeat).
The pending-wait list is the rendezvous point, and a `threading.Lock`
keeps the two from stepping on each other.

Apply to `agent/`:

  - Add the `_PENDING_WAITS` list and `_handle_wait_for` to `agent/loop.py`.
  - Patch `_run_tool` to special-case `tc.name == "wait_for"` before the
    registry lookup. (A more general design routes any name that has no
    `run` function through a `_VIRTUAL_TOOLS` dispatch table.)
  - Patch `_heartbeat` to consume expired waits before draining user
    lines, so the model sees the wait-elapsed notes alongside any
    mid-turn user input from the same tick.
  - Register `wait_for` in `agent/tools/registry.py` with `run=None`
    (the dispatch in `_run_tool` short-circuits before `run` is called).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class _PendingWait:
    wake_at: float
    reason: str


_PENDING_WAITS: list[_PendingWait] = []
_PENDING_LOCK = threading.Lock()


def _handle_wait_for(args: dict) -> str:
    """The harness side of the virtual tool. Registers the wait and
    returns immediately with a structured-looking observation."""
    seconds = float(args["seconds"])
    reason = str(args["reason"])
    wake_at = time.monotonic() + seconds
    with _PENDING_LOCK:
        _PENDING_WAITS.append(_PendingWait(wake_at=wake_at, reason=reason))
    return f"wait registered: {seconds:.0f}s for {reason!r}"


def _drain_expired_waits(now: float | None = None) -> list[str]:
    """Pop every wait whose wake_at has passed and return their reasons."""
    if now is None:
        now = time.monotonic()
    expired: list[str] = []
    with _PENDING_LOCK:
        keep: list[_PendingWait] = []
        for w in _PENDING_WAITS:
            if w.wake_at <= now:
                expired.append(w.reason)
            else:
                keep.append(w)
        _PENDING_WAITS[:] = keep
    return expired


def _run_tool(tc) -> str:
    """The chapter's `_run_tool`, with one branch added for `wait_for`."""
    if tc["name"] == "wait_for":
        return _handle_wait_for(tc["args"])
    # The real port falls through to `find_tool` + `tool.run` here.
    raise NotImplementedError(f"demo only handles 'wait_for'; got {tc['name']!r}")


def _heartbeat(drain_user_lines: Callable[[], list[str]] | None) -> list[dict]:
    """Heartbeat with one new branch: expired waits become synthetic
    user notes before the user's own lines are drained."""
    msgs: list[dict] = []
    for reason in _drain_expired_waits():
        msgs.append({"role": "user", "content": f"[wait_for elapsed: {reason}]"})
    if drain_user_lines is not None:
        for line in drain_user_lines():
            msgs.append({"role": "user", "content": f"[mid-turn user note] {line}"})
    return msgs


if __name__ == "__main__":
    # Pretend the model just issued `wait_for(seconds=1, reason="check build")`.
    tc = {"id": "tc_1", "name": "wait_for", "args": {"seconds": 1, "reason": "check build"}}
    observation = _run_tool(tc)
    print(f"observation (the tool result the model sees):\n  {observation!r}")
    assert "wait registered" in observation

    # First heartbeat tick — wait has not elapsed yet.
    out = _heartbeat(drain_user_lines=lambda: [])
    print(f"heartbeat at t+0:    {out}")
    assert out == [], "no waits should have elapsed yet"

    time.sleep(1.1)

    # Second tick after the wait should fire.
    out = _heartbeat(drain_user_lines=lambda: ["something the user typed meanwhile"])
    print(f"heartbeat at t+1.1s: {out}")
    expected_kinds = {"[wait_for elapsed:", "[mid-turn user note]"}
    seen_kinds = {m["content"].split(" ")[0] for m in out}
    assert "[wait_for" in " ".join(seen_kinds) or any(
        m["content"].startswith("[wait_for elapsed:") for m in out
    )
    assert any(m["content"].startswith("[mid-turn user note]") for m in out)
    print("\nok — wait_for registered, elapsed on schedule, fired alongside user input")
