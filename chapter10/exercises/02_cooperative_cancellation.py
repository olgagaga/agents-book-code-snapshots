"""Exercise 2: Cooperative cancellation.

Exercise 1 stopped *waiting* on a timed-out tool but let the worker keep
running in the background. That is fine when the tool is short-running
anyway; it is wasteful when the tool is a thirty-second sleep the user
wants to abort.

The fix is cooperative: the tool accepts a `threading.Event` and polls it
periodically. The heartbeat watches the user's injections for a `cancel:`
prefix and sets the event when it sees one. A well-behaved tool sees the
event within its polling interval and returns early; an ill-behaved tool
ignores it and runs to completion, no worse off than Exercise 1.

What changes:

  - `Tool` definitions can now accept an optional `cancel_event` argument.
    `_run_tool` creates a fresh event per tool call, registers it in a
    module-level list so the heartbeat can find it, passes it to the tool,
    and removes it from the list when the call returns.
  - `_sleep_for` replaces its `time.sleep(seconds)` with
    `cancel_event.wait(timeout=seconds)` — which returns True when the
    event fires and False when the timeout elapses.
  - The heartbeat learns one new branch: if a drained line starts with
    `cancel:`, set every active cancel event before emitting the line as
    a mid-turn user note.

Apply to `agent/`:

  - Add the `_ACTIVE_CANCEL_EVENTS` list and the `_set_all_cancel_events`
    helper to `agent/loop.py`.
  - Patch `_run_tool` to create and register the event, then pass it via
    `args["_cancel_event"]` (kept as a private key so tools that ignore
    it are unaffected).
  - Patch `_heartbeat` to scan drained lines for a `cancel:` prefix and
    call `_set_all_cancel_events()` before wrapping the line.
  - Patch `_sleep_for` in `agent/tools/tools.py` to read
    `args.get("_cancel_event")` and use `.wait()` when present.

The `_cancel_event` key is intentionally underscore-prefixed: it never
appears in the tool's JSON schema and the model never produces it. It
travels through `args` only because that is the path the existing
`Tool.run` callable already accepts.
"""

import threading
import time
from typing import Callable

_ACTIVE_CANCEL_EVENTS: list[threading.Event] = []


def _set_all_cancel_events() -> None:
    """Signal cancellation to every currently-running tool."""
    for event in list(_ACTIVE_CANCEL_EVENTS):
        event.set()


def _heartbeat(
    drain: Callable[[], list[str]] | None,
) -> list[dict]:
    """Same shape as the chapter helper, plus the `cancel:` branch."""
    if drain is None:
        return []
    out: list[dict] = []
    for line in drain():
        if line.lower().startswith("cancel:"):
            _set_all_cancel_events()
        out.append({"role": "user", "content": f"[mid-turn user note] {line}"})
    return out


def _sleep_for(args: dict) -> str:
    """Cooperative sleep: returns early if `_cancel_event` is set."""
    seconds = float(args["seconds"])
    event = args.get("_cancel_event")
    if event is None:
        time.sleep(seconds)
        return f"slept for {seconds:.1f}s"
    started = time.monotonic()
    if event.wait(timeout=seconds):
        elapsed = time.monotonic() - started
        return f"sleep interrupted after {elapsed:.1f}s of {seconds:.1f}s"
    return f"slept for {seconds:.1f}s"


def _run_tool(tc) -> str:
    """Wraps the chapter's `_run_tool` with cancel-event plumbing."""
    cancel_event = threading.Event()
    _ACTIVE_CANCEL_EVENTS.append(cancel_event)
    try:
        # In the real port, `tool = find_tool(tc.name)` then
        # `tool.run({**tc.args, "_cancel_event": cancel_event})`. The demo
        # below substitutes a callable directly so this file is runnable
        # without the rest of the agent.
        fn = tc["fn"]
        try:
            return fn({**tc["args"], "_cancel_event": cancel_event})
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"
    finally:
        _ACTIVE_CANCEL_EVENTS.remove(cancel_event)


if __name__ == "__main__":
    import concurrent.futures

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="ex2")

    # The "user" types this line ~0.3s into a 5-second sleep.
    queued_lines: list[str] = []

    def drain() -> list[str]:
        lines, queued_lines[:] = list(queued_lines), []
        return lines

    def schedule_user_input(delay: float, text: str) -> None:
        def fire() -> None:
            time.sleep(delay)
            queued_lines.append(text)
        threading.Thread(target=fire, daemon=True).start()

    tc = {"id": "tc_1", "fn": _sleep_for, "args": {"seconds": 5.0}}
    future = pool.submit(_run_tool, tc)

    schedule_user_input(0.3, "cancel: skip this please")

    started = time.monotonic()
    pending: list[dict] = []
    while not future.done():
        pending.extend(_heartbeat(drain))
        time.sleep(0.05)
    pending.extend(_heartbeat(drain))
    elapsed = time.monotonic() - started

    result = future.result()
    print(f"tool returned: {result!r}")
    print(f"wall time:    {elapsed:.2f}s (target: well under 5s)")
    print(f"pending injections: {pending}")

    assert "interrupted" in result, f"expected an interrupted sleep, got {result!r}"
    assert elapsed < 1.0, f"cancel should have landed within ~100ms, took {elapsed:.2f}s"
    print("\nok — sleep_for honoured the cancel event within the heartbeat tick")
