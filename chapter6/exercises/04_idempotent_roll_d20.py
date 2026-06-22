"""Exercise 4: Idempotent tools and re-runs — `roll_d20`.

Replace `now` with a `roll_d20` tool that returns a random integer 1–20.
Ask the model to roll three times and report the sum. Watch the trace.

The interesting property is that `roll_d20` is *non-deterministic*: each
call returns a fresh number, even with the same arguments. If the loop is
re-run from a checkpoint after a crash (production reference, point two),
the second run will see a different roll. That is fine for a dice game;
catastrophic for "send Slack message" or "transfer money". It is why
nanobot's `_emit_checkpoint` records the *result* in session state —
production agents replay results, not tool calls, on recovery.

Run with `ANTHROPIC_API_KEY` set. The trace should show three iterations
calling `roll_d20`, then a final iteration with the sum.
"""

import json
import os
import random
from typing import Iterator

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-opus-4-6"

MAX_ITERATIONS = 10


def _roll_d20(args: dict) -> str:
    return str(random.randint(1, 20))


TOOLS = {"roll_d20": _roll_d20}


TOOL_INSTRUCTIONS = """
You have access to tools. To call a tool, reply with exactly one line of JSON:

    {"tool": "<name>", "args": {<arguments>}}

Available tools:

- `roll_d20` — roll a 20-sided die and return an integer 1-20. No arguments.

When you call a tool, do not write anything else. Wait for the tool result, then continue.
When you have nothing left to do, reply with a normal text answer to the user. No JSON.
""".strip()


def _stream(messages: list[dict], system: str = "") -> Iterator[str]:
    with client.messages.stream(
        model=MODEL, max_tokens=1024, system=system, messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def _parse_tool_call(text: str) -> dict | None:
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "tool" not in parsed:
        return None
    return parsed


def agent_step(user_message: str, history: list[dict], system: str) -> str:
    messages = list(history)
    messages.append({"role": "user", "content": user_message})

    for i in range(MAX_ITERATIONS):
        reply = "".join(_stream(messages, system=system))
        print(f"[iter {i}] reply={reply[:120]!r}")
        call = _parse_tool_call(reply)

        if call is None:
            return reply

        tool_name = call["tool"]
        tool_args = call.get("args") or {}
        tool = TOOLS.get(tool_name)
        if tool is None:
            observation = f"Error: no tool named {tool_name!r}."
        else:
            try:
                observation = tool(tool_args)
            except Exception as exc:
                observation = f"Error: {type(exc).__name__}: {exc}"

        messages.append({"role": "assistant", "content": reply})
        messages.append({
            "role": "user",
            "content": f"<tool_result tool={tool_name}>\n{observation}\n</tool_result>",
        })

    return "I exceeded the maximum number of steps without producing a final answer."


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env")

    reply = agent_step(
        "Roll a d20 three times, then tell me the sum and the individual results.",
        history=[],
        system=TOOL_INSTRUCTIONS,
    )
    print(f"\nfinal: {reply}")
