"""Exercise 3: Progress callbacks.

A long-running tool with no feedback is the failure mode this chapter
opened with — the user can't tell whether the agent is stuck or working.
Exercise 1 stopped *waiting* past a deadline; exercise 2 let the user
cancel. This exercise gives the user something to watch while it runs.

The mechanism is small. Tools optionally accept a callback that the
harness binds at registration time; the tool calls it whenever it has
something to report. The harness prints each report on the terminal,
overwriting the previous line with `\\r` so the output stays one line tall.

What changes:

  - `Tool` gains an optional `on_progress: Callable[[str], None] | None`
    field — see `Tool` near the bottom of this file. The default is
    `None`, which makes the change backwards-compatible: tools that
    ignore progress need no edits.
  - `_run_tool` looks up `tool.on_progress` (the *default* registered
    on the tool) and, if present, forwards a stream-aware callback to
    the tool via `args["_on_progress"]`. The terminal writer lives at
    the harness layer so tools stay output-agnostic.
  - `_sleep_for_with_progress` (a small reimplementation of `_sleep_for`)
    calls `on_progress(f"slept {n}s of {total}s")` every second.

Apply to `agent/`:

  - Edit `agent/tools/base.py` to add the `on_progress` field with a
    `None` default. Update `agent/tools/registry.py`: for `sleep_for`,
    pass `on_progress=_print_progress` (defined wherever the registry's
    side-effect dependencies live — probably a new `agent/tools/output.py`).
  - Edit `agent/loop.py`'s `_run_tool` to forward the callback via
    `args["_on_progress"]`. The change is one branch.
  - Edit `_sleep_for` in `agent/tools/tools.py` to read
    `args.get("_on_progress")` and call it every second.

The `\\r` trick relies on the cursor being at the start of the line. If a
streaming model token has just printed, the cursor is after the last
token; the harness should print a leading `\\n` the first time the
progress callback fires, or save the cursor position and restore it.
The demo below uses the simpler always-with-newline approach.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    """Same shape as `agent/tools/base.py`, plus `on_progress`."""

    name: str
    description: str
    schema: dict
    run: Callable[[dict], str]
    on_progress: Callable[[str], None] | None = None


def _print_progress_inplace(line: str) -> None:
    """Write a progress line, overwriting the previous one."""
    sys.stdout.write("\r" + line + " " * 8)  # trailing spaces clear leftover chars
    sys.stdout.flush()


def _sleep_for_with_progress(args: dict) -> str:
    """`_sleep_for` variant that emits one progress line per second.

    Reads `_on_progress` from `args` rather than as a positional kwarg, so
    the existing `Tool.run` signature (`Callable[[dict], str]`) is preserved.
    """
    total = float(args["seconds"])
    on_progress = args.get("_on_progress")
    elapsed = 0.0
    while elapsed < total:
        step = min(1.0, total - elapsed)
        time.sleep(step)
        elapsed += step
        if on_progress is not None:
            on_progress(f"slept {elapsed:.0f}s of {total:.0f}s")
    return f"slept for {total:.1f}s"


def _run_tool(tool: Tool, args: dict) -> str:
    """Forwards the tool's `on_progress` into `args` if one is registered."""
    enriched = dict(args)
    if tool.on_progress is not None:
        enriched["_on_progress"] = tool.on_progress
    try:
        return tool.run(enriched)
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    sleep_for = Tool(
        name="sleep_for",
        description="Sleep for the given number of seconds.",
        schema={
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": ["seconds"],
        },
        run=_sleep_for_with_progress,
        on_progress=_print_progress_inplace,
    )

    print("running sleep_for(seconds=3) — watch the line update in place:")
    print()  # blank line so \r overwrites cleanly
    result = _run_tool(sleep_for, {"seconds": 3})
    print()  # newline after the in-place updates
    print(f"final observation: {result!r}")

    assert result == "slept for 3.0s"
    print("\nok — progress callbacks fired during the sleep; final observation unchanged")
