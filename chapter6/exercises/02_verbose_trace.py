"""Exercise 2: Verbose iteration trace.

Promote the debug `print` we added at the end of *The loop itself* into a
proper `verbose` argument. When `verbose=True`, log a one-line summary of
each iteration: the iteration number, which tool was called with what
args, and the observation truncated to 80 characters. When `verbose=False`,
the function prints nothing at all.

This is a small but useful upgrade. The chapter's `print` shows only the
model's reply, which leaves the *tool result* invisible. The verbose trace
shows both sides of each iteration, so a multi-step turn becomes a
readable log.

The file inlines the relevant parts of `loop.py` so it can run standalone.
Set `ANTHROPIC_API_KEY` in `.env` and run it.
"""

import json
import os
from datetime import datetime, timezone
from typing import Iterator

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-opus-4-6"

MAX_ITERATIONS = 10


def _now(args: dict) -> str:
    return datetime.now(timezone.utc).isoformat()


TOOLS = {"now": _now}


TOOL_INSTRUCTIONS = """
You have access to tools. To call a tool, reply with exactly one line of JSON:

    {"tool": "<name>", "args": {<arguments>}}

Available tools:

- `now` — return the current UTC time. No arguments.

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


def _truncate(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def agent_step(
    user_message: str,
    history: list[dict],
    system: str,
    verbose: bool = False,
) -> str:
    messages = list(history)
    messages.append({"role": "user", "content": user_message})

    for i in range(MAX_ITERATIONS):
        reply = "".join(_stream(messages, system=system))
        call = _parse_tool_call(reply)

        if call is None:
            if verbose:
                print(f"[iter {i}] final ({len(reply)} chars): {_truncate(reply)}")
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

        if verbose:
            print(
                f"[iter {i}] tool={tool_name} args={tool_args} "
                f"-> {_truncate(observation)}"
            )

        messages.append({"role": "assistant", "content": reply})
        messages.append({
            "role": "user",
            "content": f"<tool_result tool={tool_name}>\n{observation}\n</tool_result>",
        })

    return "I exceeded the maximum number of steps without producing a final answer."


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env")

    print("--- verbose=True ---")
    reply = agent_step(
        "what is the current UTC time, to the second?",
        history=[],
        system=TOOL_INSTRUCTIONS,
        verbose=True,
    )
    print(f"\nfinal: {reply}\n")

    print("--- verbose=False (no trace lines should appear) ---")
    reply = agent_step(
        "what is the capital of Spain?",
        history=[],
        system=TOOL_INSTRUCTIONS,
        verbose=False,
    )
    print(f"final: {reply}")
