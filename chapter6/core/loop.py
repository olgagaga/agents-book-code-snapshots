from datetime import datetime, timezone
from context import build_context
import json
from providers.base import Provider

MAX_ITERATIONS = 10


def _now(args: dict) -> str:
    return datetime.now(timezone.utc).isoformat()


TOOLS = {
    "now": _now,
}


TOOL_INSTRUCTIONS = """
You have access to tools. To call a tool, reply with exactly one line of JSON:

    {"tool": "<name>", "args": {<arguments>}}

Available tools:

- `now` — return the current UTC time. No arguments.

When you call a tool, do not write anything else. Wait for the tool result, then continue.
When you have nothing left to do, reply with a normal text answer to the user. No JSON.
""".strip()


def build_agent_system() -> str:
    return f"{build_context()}\n\n{TOOL_INSTRUCTIONS}"


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
    user_message: str,
    history: list[dict],
    provider: Provider,
    system: str,
) -> str:
    messages = list(history)
    messages.append({"role": "user", "content": user_message})

    for i in range(MAX_ITERATIONS):
        reply = "".join(provider.stream(messages, system=system))
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