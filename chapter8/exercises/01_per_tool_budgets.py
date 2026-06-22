"""Exercise 1: Per-tool observation budgets.

The chapter uses a single `MAX_OBSERVATION_CHARS = 4000` for every tool.
That is a reasonable default but wrong in detail — `now` returns ~30 chars
and a future `read_file` can return 200 KB. A shared budget wastes context
on the small results or risks truncating them too aggressively, depending
on which side you tune for.

The fix is to move the budget onto the `Tool` dataclass itself. Each tool
gets a `max_observation_chars` field with a sensible default; the loop
reads that field instead of the module-level constant when truncating
each result.

This file rebuilds the relevant pieces of `loop.py`, `tools/base.py`, and
`tools/registry.py` in a single module so the behavior is testable
without a real API call. To apply the patch to the real agent:

  - add `max_observation_chars: int = 4000` to `Tool` in `tools/base.py`,
  - pass per-tool budgets in the `tools/registry.py` entries,
  - in `agent/loop.py` replace `_truncate_observation(observation)` with
    a call that reads `tool.max_observation_chars`, defaulting to
    `MAX_OBSERVATION_CHARS` when the tool is unknown (the "no tool named
    X" error path still needs *some* budget).
"""

from dataclasses import dataclass
from typing import Callable


MAX_OBSERVATION_CHARS = 4000


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    run: Callable[[dict], str]
    max_observation_chars: int = MAX_OBSERVATION_CHARS


def _truncate_observation(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 60
    omitted = len(text) - head - tail
    return (
        f"{text[:head]}\n\n"
        f"[...truncated: {omitted} characters omitted from the middle...]\n\n"
        f"{text[-tail:]}"
    )


def truncate_for_tool(tool: Tool | None, observation: str) -> str:
    budget = tool.max_observation_chars if tool is not None else MAX_OBSERVATION_CHARS
    return _truncate_observation(observation, budget)


def _now(args: dict) -> str:
    return "2026-05-13T15:40:22Z"


def _wordcount(args: dict) -> str:
    return str(len(args["text"].split()))


def _lorem(args: dict) -> str:
    return "lorem ipsum " * 5000  # ~60_000 chars


TOOLS: list[Tool] = [
    Tool(
        name="now",
        description="Return the current UTC time.",
        schema={"type": "object", "properties": {}},
        run=_now,
        max_observation_chars=100,
    ),
    Tool(
        name="wordcount",
        description="Count words in the given text.",
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
        run=_wordcount,
        max_observation_chars=50,
    ),
    Tool(
        name="lorem",
        description="Return placeholder text.",
        schema={"type": "object", "properties": {}},
        run=_lorem,
        max_observation_chars=8000,
    ),
]


def find_tool(name: str) -> Tool | None:
    return next((t for t in TOOLS if t.name == name), None)


if __name__ == "__main__":
    now = find_tool("now")
    raw = now.run({})
    assert truncate_for_tool(now, raw) == raw, "short results pass through"
    print(f"now:       {len(raw):>5} chars -> no truncation")

    wc = find_tool("wordcount")
    raw = wc.run({"text": "the quick brown fox jumps over the lazy dog"})
    assert truncate_for_tool(wc, raw) == "9"
    print(f"wordcount: {len(raw):>5} chars -> no truncation")

    lorem = find_tool("lorem")
    raw = lorem.run({})
    truncated = truncate_for_tool(lorem, raw)
    assert len(truncated) < len(raw)
    assert "[...truncated:" in truncated
    print(f"lorem:     {len(raw):>5} chars -> {len(truncated)} chars (middle-snipped)")
    print(f"  head: {truncated[:60]!r}")
    print(f"  tail: {truncated[-60:]!r}")

    unknown_tool_observation = "Error: no tool named 'bogus'."
    out = truncate_for_tool(None, unknown_tool_observation)
    assert out == unknown_tool_observation
    print(f"unknown:   {len(unknown_tool_observation):>5} chars -> default budget path")
