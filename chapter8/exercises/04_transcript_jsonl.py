"""Exercise 4: Persist a transcript.

`chat()` currently discards the value `agent_step` returns. The user
still sees the reply on screen because the streaming callback prints
each delta as it arrives, but nothing on disk records the turn for
later inspection. Capturing the return value and writing a single JSON
line per turn turns the chat into a queryable transcript.

Each turn writes one record with three fields:

  - `user`: the user's prompt
  - `assistant`: the final assistant text
  - `iterations`: the number of for-loop passes `agent_step` actually
    used before returning

The iteration count is the diagnostic signal the exercise highlights.
A turn with `iterations=1` answered without calling any tools (the
model went straight to text). A turn with `iterations=5` did a
multi-step exploration. A turn at or near `MAX_ITERATIONS` is the loop
running out of budget — usually the model is stuck on the same broken
call (see Exercise 3 for an early-bail fix).

The cleanest way to expose the iteration count without changing
`agent_step`'s return type is to return a small dataclass, or — as
shown below — return a `(text, iterations)` tuple. This file
demonstrates the wiring with a scripted provider so the demo runs
without any API call.
"""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


MAX_ITERATIONS = 10


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    run: Callable[[dict], str]


def agent_step_with_count(
    user_message: str,
    messages: list[dict],
    provider,
    system: str,
    tools: list[Tool],
    on_text_delta: Callable[[str], None] | None = None,
) -> tuple[str, int]:
    """Same shape as `agent_step` from the chapter, plus the iteration count."""
    messages.append({"role": "user", "content": user_message})
    tools_by_name = {t.name: t for t in tools}

    for i in range(1, MAX_ITERATIONS + 1):
        reply: Reply = provider.call(
            messages, system=system, tools=tools, on_text_delta=on_text_delta,
        )
        if not reply.tool_calls:
            messages.append({"role": "assistant", "content": reply.text})
            return reply.text, i

        messages.append({
            "role": "assistant",
            "content": reply.text,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "args": tc.args}
                for tc in reply.tool_calls
            ],
        })
        for tc in reply.tool_calls:
            tool = tools_by_name.get(tc.name)
            if tool is None:
                observation = f"Error: no tool named {tc.name!r}."
            else:
                try:
                    observation = tool.run(tc.args)
                except Exception as exc:
                    observation = f"Error: {type(exc).__name__}: {exc}"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": observation,
            })

    return (
        "I exceeded the maximum number of steps without producing a final answer.",
        MAX_ITERATIONS,
    )


def append_transcript(path: Path, user: str, assistant: str, iterations: int) -> None:
    record = {"user": user, "assistant": assistant, "iterations": iterations}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class _ScriptedProvider:
    """A fake provider that returns the next pre-written `Reply` on each call."""

    def __init__(self, replies: list[Reply]):
        self._replies = list(replies)

    def call(self, messages, system="", tools=(), on_text_delta=None) -> Reply:
        return self._replies.pop(0)


def _now(args: dict) -> str:
    return "2026-05-13T15:40:22Z"


if __name__ == "__main__":
    tmpdir = Path(tempfile.mkdtemp())
    transcript = tmpdir / "transcript.jsonl"

    now_tool = Tool(name="now", description="", schema={"type": "object"}, run=_now)

    # Turn 1: the model answers without calling a tool (1 iteration).
    provider = _ScriptedProvider([Reply(text="hello!", tool_calls=[])])
    text, iters = agent_step_with_count("hi", [], provider, "", [now_tool])
    append_transcript(transcript, "hi", text, iters)

    # Turn 2: one tool call, then the final answer (2 iterations).
    provider = _ScriptedProvider([
        Reply(text="", tool_calls=[ToolCall(id="a", name="now", args={})]),
        Reply(text="It is 2026-05-13T15:40:22Z.", tool_calls=[]),
    ])
    text, iters = agent_step_with_count(
        "what time is it?", [], provider, "", [now_tool],
    )
    append_transcript(transcript, "what time is it?", text, iters)

    # Turn 3: the model loops on broken calls until MAX_ITERATIONS (struggle).
    looping_replies = [
        Reply(text="", tool_calls=[ToolCall(id=f"x{i}", name="bogus", args={})])
        for i in range(MAX_ITERATIONS)
    ]
    provider = _ScriptedProvider(looping_replies)
    text, iters = agent_step_with_count(
        "do the bogus thing", [], provider, "", [now_tool],
    )
    append_transcript(transcript, "do the bogus thing", text, iters)

    records = [json.loads(L) for L in transcript.read_text().splitlines()]
    assert records[0]["iterations"] == 1
    assert records[1]["iterations"] == 2
    assert records[2]["iterations"] == MAX_ITERATIONS

    print(f"transcript: {transcript}\n")
    for r in records:
        marker = "  "
        if r["iterations"] == 1:
            marker = "* "  # trivial
        elif r["iterations"] >= MAX_ITERATIONS:
            marker = "! "  # likely stuck
        print(f"{marker}iter={r['iterations']:>2}  {r['user']!r}  ->  {r['assistant']!r}")
