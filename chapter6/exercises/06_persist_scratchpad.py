"""Exercise 6 (stretch): Persist the intermediate scratchpad across turns.

The chapter throws the intermediate `messages` list away when `agent_step`
returns. This version returns *every* message it appended — the assistant's
tool-call replies, the synthetic tool-result user messages, the lot — and
the caller decides whether to fold them into long-term history.

Two short sessions, each two turns:

  Session A: "what is the current UTC time?"  then  "and what about now,
             ten seconds later?"

  Session B: "what is the current UTC time?"  then  "tell me a joke."

Print `len(history)` after each turn. A's second turn might benefit from
seeing the previous tool result (the model knows it just used `now` and
can ask itself "what changed?"); B's second turn does not. The persisted
scratchpad pays off only when the *next* turn shares the agent's prior
state; otherwise it is dead weight that grows the prompt.

Production note: nanobot's `prepare_session`
(`nanobot/nanobot/agent/autocompact.py`) does this trade-off
automatically. It keeps recent scratchpad turns full, summarises older
ones, and prunes the rest, so the cost stays bounded as a session grows.
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


def agent_step(
    user_message: str, history: list[dict], system: str,
) -> tuple[str, list[dict]]:
    """Return (final_reply, full_turn_messages).

    `full_turn_messages` is everything appended this turn: the user message,
    every intermediate assistant tool-call reply, every synthetic tool-result
    user message, and the final assistant reply. Caller decides what to keep.
    """
    messages = list(history)
    messages.append({"role": "user", "content": user_message})
    boundary = len(history)

    for _ in range(MAX_ITERATIONS):
        reply = "".join(_stream(messages, system=system))
        call = _parse_tool_call(reply)

        if call is None:
            messages.append({"role": "assistant", "content": reply})
            return reply, messages[boundary:]

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

    final = "I exceeded the maximum number of steps without producing a final answer."
    messages.append({"role": "assistant", "content": final})
    return final, messages[boundary:]


def run_session(label: str, prompts: list[str]) -> None:
    print(f"--- Session {label} ---")
    history: list[dict] = []
    for prompt in prompts:
        reply, turn = agent_step(prompt, history, system=TOOL_INSTRUCTIONS)
        history.extend(turn)
        print(f"q: {prompt}")
        print(f"a: {reply}")
        print(f"   history len after this turn: {len(history)}")
        print()


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env")

    run_session("A", [
        "what is the current UTC time?",
        "and what about now, ten seconds later?",
    ])
    run_session("B", [
        "what is the current UTC time?",
        "tell me a joke.",
    ])
