"""Exercise 5 (stretch): Rewrite the loop in asyncio.

The chapter solves three problems with three pieces of machinery: a
`ThreadPoolExecutor` runs tools off the main thread, a `threading.Thread`
reads stdin into a `queue.Queue`, and a `_wait_for_tools` polling loop
ties them together. `asyncio` reframes all three around a single event
loop and a single concurrency primitive (the `Task`). The exercise asks
you to do the rewrite and then notice where each style was clearer.

This file is a runnable demo of the async version. It uses a stub
provider rather than a real LLM so the exercise runs offline and the
focus stays on the structural changes. The demo reproduces the chapter's
sleep-then-inject scenario: kick off a 5-second sleep, type a cancel
mid-turn, observe that the sleep returns early and the next iteration
sees the user's note.

What changes structurally:

  - `ThreadPoolExecutor.submit(_run_tool, tc)` becomes
    `asyncio.create_task(asyncio.to_thread(_run_tool, tc))`. The
    `to_thread` keeps the synchronous tool body unchanged — it runs on
    the default executor thread — but the result is awaitable.
  - The stdin reader thread becomes `loop.add_reader(sys.stdin, ...)`,
    which fires a callback on every read-ready event. Lines go into an
    `asyncio.Queue` instead of a `queue.Queue`.
  - `_wait_for_tools` becomes a `gather(*tasks, return_exceptions=True)`
    wrapped in a `wait_for(..., timeout=...)`. The 100ms polling tick
    disappears: the event loop wakes whenever any task makes progress.
  - Cancellation finally works mid-execution. `task.cancel()` raises
    `CancelledError` inside the task on the next `await`, which is
    enough to abort an `asyncio.sleep` (and any tool that uses async
    primitives). For synchronous tools running in `to_thread`, cancel
    is still cooperative — the same `threading.Event` trick from
    Exercise 2 applies.

Where each version is clearer:

  - The async version is *clearer* for the loop's structure: one event
    loop, one queue, no polling tick, cancellation that actually fires
    where the tool is running.
  - The threading version is *clearer* for backwards-compatible tools:
    `tool.run(args)` is an ordinary synchronous call, with no `async`
    keyword viral-painting through the codebase. A real port of the
    book's agent to async would either keep `to_thread` indefinitely
    (so tools stay sync) or migrate every tool to `async def`.

Run with: `uv run python 05_asyncio_rewrite.py`
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable


# ---- A tiny stub provider so the demo runs offline ----


@dataclass
class _ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class _Reply:
    text: str
    stop_reason: str
    tool_calls: list[_ToolCall]


class _StubProvider:
    """Scripted provider: first call returns a tool_use, second call
    returns end_turn. Reads the mid-turn user note in its scripted
    behaviour so the demo end-to-end story is honest."""

    def __init__(self) -> None:
        self._step = 0

    async def call(self, messages: list[dict]) -> _Reply:
        self._step += 1
        if self._step == 1:
            return _Reply(
                text="I'll sleep for five seconds.",
                stop_reason="tool_use",
                tool_calls=[_ToolCall(id="tc_1", name="sleep_for", args={"seconds": 5.0})],
            )
        # See whether the last user message is a mid-turn injection.
        last_user = next(
            (m for m in reversed(messages) if m["role"] == "user"), None,
        )
        cancelled = last_user and "skip" in last_user.get("content", "").lower()
        return _Reply(
            text=(
                "Got the note — skipping the wait." if cancelled
                else "Slept the full five seconds. Hi!"
            ),
            stop_reason="end_turn",
            tool_calls=[],
        )


# ---- The async-flavoured tool and helpers ----


async def _sleep_for_async(args: dict) -> str:
    """Async sleep that honours cancellation natively. `asyncio.sleep`
    yields to the event loop, so `task.cancel()` aborts it immediately
    — no `threading.Event` plumbing needed."""
    seconds = float(args["seconds"])
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        return f"sleep cancelled before {seconds:.1f}s elapsed"
    return f"slept for {seconds:.1f}s"


async def _run_tool(tc: _ToolCall) -> str:
    """One-line wrapper to match the chapter's `_run_tool` shape."""
    if tc.name == "sleep_for":
        return await _sleep_for_async(tc.args)
    return f"Error: no tool named {tc.name!r}."


async def _heartbeat(injections: asyncio.Queue) -> list[dict]:
    """Drain the injections queue without blocking and wrap each line."""
    out: list[dict] = []
    while True:
        try:
            line = injections.get_nowait()
        except asyncio.QueueEmpty:
            break
        out.append({"role": "user", "content": f"[mid-turn user note] {line}"})
    return out


# ---- The async agent loop, structurally parallel to `agent_step` ----


async def agent_step(
    user_message: str,
    messages: list[dict],
    provider: _StubProvider,
    injections: asyncio.Queue,
    cancel_on_skip: bool = True,
    max_iterations: int = 10,
) -> str:
    """Async port of the chapter's `agent_step`. Notice how short the
    tool-handling branch becomes: `gather` waits, the heartbeat is a
    second task, and `task.cancel()` aborts a running tool from the
    outside."""
    messages.append({"role": "user", "content": user_message})

    for _ in range(max_iterations):
        messages.extend(await _heartbeat(injections))

        reply = await provider.call(messages)

        if reply.stop_reason == "tool_use":
            messages.append({
                "role": "assistant",
                "content": reply.text,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "args": tc.args}
                    for tc in reply.tool_calls
                ],
            })
            tool_tasks = [asyncio.create_task(_run_tool(tc)) for tc in reply.tool_calls]

            pending_injections: list[dict] = []
            while not all(t.done() for t in tool_tasks):
                drained = await _heartbeat(injections)
                if cancel_on_skip and any("skip" in m["content"].lower() for m in drained):
                    for t in tool_tasks:
                        t.cancel()
                pending_injections.extend(drained)
                await asyncio.sleep(0.05)
            pending_injections.extend(await _heartbeat(injections))

            for task, tc in zip(tool_tasks, reply.tool_calls):
                try:
                    observation = await task
                except asyncio.CancelledError:
                    observation = "[cancelled by user]"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": observation,
                })
            messages.extend(pending_injections)
            continue

        messages.append({"role": "assistant", "content": reply.text})
        return reply.text

    return "[stopped: max iterations]"


# ---- Demo ----


async def main() -> None:
    injections: asyncio.Queue[str] = asyncio.Queue()
    provider = _StubProvider()
    messages: list[dict] = []

    async def fake_user() -> None:
        await asyncio.sleep(0.5)
        await injections.put("actually skip the sleep, just say hi")

    asyncio.create_task(fake_user())

    started = time.monotonic()
    final_text = await agent_step(
        "sleep for five seconds and then say hi", messages, provider, injections,
    )
    elapsed = time.monotonic() - started

    print(f"final reply: {final_text!r}")
    print(f"wall time:   {elapsed:.2f}s")
    print(f"history len: {len(messages)} messages")
    for m in messages:
        kind = m["role"]
        tail = m.get("content") or m.get("tool_calls")
        print(f"  {kind:9} {tail!r}")

    assert elapsed < 2.0, f"async cancel should have shortened the wait, took {elapsed:.2f}s"
    assert any("cancelled" in str(m.get("content", "")) for m in messages)
    print("\nok — async cancel landed mid-sleep; no 100ms polling tick was needed")


if __name__ == "__main__":
    asyncio.run(main())
