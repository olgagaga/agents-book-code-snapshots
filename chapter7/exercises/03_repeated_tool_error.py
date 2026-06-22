"""Exercise 3: Bail out on repeated tool-call failures.

The chapter's loop returns `Error: <type>: <message>` as the observation
when a tool raises. The model usually reads that, apologizes, and tries
something else. Occasionally it loops: same tool, same arguments, same
exception, three or four times in a row, until `MAX_ITERATIONS` runs out.

This patch terminates the loop early when the same `(tool_name, args)`
combination raises *twice in a row* with no successful call between them.
"Same arguments" is checked by canonicalizing each `args` dict to JSON
(sorted keys, no whitespace), which is robust to dict ordering and nested
structures.

The deliberate part is what counts as "the same call." We compare *what the
model asked for* — name and arguments — not the failure shape. The model
controls those; if it varies either, that counts as trying something else
and the counter resets. Comparing on the exception type instead would let
the model retry the same broken call as long as it produced a different
error each time, which is a worse loop to be stuck in.

Apply this patch to `agent_step` in `agent/loop.py`. Below is a focused
reproduction that swaps the provider for a hand-built fake so we can
exercise the bail-out path without spending tokens.
"""

import json

from dataclasses import dataclass, field
from typing import Callable


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


MAX_ITERATIONS = 10


def _canon(args: dict) -> str:
    return json.dumps(args, sort_keys=True, separators=(",", ":"))


def agent_step(
    user_message: str,
    history: list[dict],
    provider,
    system: str,
    tools: list[Tool],
    on_text_delta: Callable[[str], None] | None = None,
) -> str:
    messages = list(history)
    messages.append({"role": "user", "content": user_message})
    tools_by_name = {t.name: t for t in tools}

    last_failed: tuple[str, str] | None = None

    for _ in range(MAX_ITERATIONS):
        reply: Reply = provider.call(
            messages, system=system, tools=tools, on_text_delta=on_text_delta,
        )

        if not reply.tool_calls:
            return reply.text

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
            signature = (tc.name, _canon(tc.args))
            failed_this_call = False

            if tool is None:
                observation = f"Error: no tool named {tc.name!r}."
                failed_this_call = True
            else:
                try:
                    observation = tool.run(tc.args)
                except Exception as exc:
                    observation = f"Error: {type(exc).__name__}: {exc}"
                    failed_this_call = True

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": observation,
            })

            if failed_this_call and last_failed == signature:
                return (
                    f"Stopping: tool {tc.name!r} failed twice in a row with the "
                    f"same arguments ({tc.args!r}). Last error: {observation}"
                )
            last_failed = signature if failed_this_call else None

    return "I exceeded the maximum number of steps without producing a final answer."


class _ScriptedProvider:
    """A fake provider that returns the next pre-written `Reply` on each call."""

    def __init__(self, replies: list[Reply]):
        self._replies = list(replies)

    def call(self, messages, system="", tools=(), on_text_delta=None) -> Reply:
        return self._replies.pop(0)


def _always_fails(args: dict) -> str:
    raise RuntimeError("nope")


if __name__ == "__main__":
    broken = Tool(
        name="broken", description="raises", schema={"type": "object"}, run=_always_fails,
    )

    provider = _ScriptedProvider([
        Reply(text="", tool_calls=[ToolCall(id="a", name="broken", args={"x": 1})]),
        Reply(text="", tool_calls=[ToolCall(id="b", name="broken", args={"x": 1})]),
        Reply(text="should never reach here", tool_calls=[]),
    ])
    out = agent_step("trigger it", [], provider, "", [broken])
    assert "failed twice in a row" in out, out
    assert "broken" in out
    print("bail-out path:", out)

    provider = _ScriptedProvider([
        Reply(text="", tool_calls=[ToolCall(id="a", name="broken", args={"x": 1})]),
        Reply(text="", tool_calls=[ToolCall(id="b", name="broken", args={"x": 2})]),
        Reply(text="recovered", tool_calls=[]),
    ])
    out = agent_step("varied args", [], provider, "", [broken])
    assert out == "recovered", out
    print("variation resets counter:", out)
