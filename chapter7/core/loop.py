from context import build_context
from providers.base import Provider, Reply
from typing import Callable
from tools.registry import TOOLS, find_tool

MAX_ITERATIONS = 10


def agent_step(
    user_message: str,
    history: list[dict],
    provider: Provider,
    system: str,
    on_text_delta: Callable[[str], None] | None = None,
) -> str:
    messages = list(history)
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_ITERATIONS):
        reply: Reply = provider.call(
            messages, system=system, tools=TOOLS, on_text_delta=on_text_delta,
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
            tool = find_tool(tc.name)
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

    return "I exceeded the maximum number of steps without producing a final answer."
