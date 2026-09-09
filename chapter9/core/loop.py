from context import build_context
from providers.base import Provider, Reply
from typing import Callable
from tools.registry import TOOLS, find_tool

MAX_ITERATIONS = 10
MAX_OBSERVATION_CHARS = 4000

_STOP_REASON_NOTES = {
    "max_tokens": "[response truncated by the model's output limit]",
    "refusal": "[model declined to answer this request]",
}

MAX_ITERATIONS_NOTE = (
    f"[stopped after {MAX_ITERATIONS} iterations without a final answer]"
)

INTERRUPT_NOTE = "[interrupted]"

def _truncate_observation(text: str, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
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


def _backfill_orphan_tool_results(messages: list[dict]) -> None:
    last_assistant_with_tools = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            last_assistant_with_tools = i
            break
    if last_assistant_with_tools is None:
        return
    expected_ids = [tc["id"] for tc in messages[last_assistant_with_tools]["tool_calls"]]
    satisfied_ids = {
        m["tool_call_id"]
        for m in messages[last_assistant_with_tools + 1:]
        if m.get("role") == "tool"
    }
    for tc_id in expected_ids:
        if tc_id in satisfied_ids:
            continue
        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "name": "<interrupted>",
            "content": INTERRUPT_NOTE,
        })

def agent_step(
    user_message: str,
    messages: list[dict],
    provider: Provider,
    system: str,
    on_text_delta: Callable[[str], None] | None = None,
) -> str:
    messages.append({"role": "user", "content": user_message})

    try:
        for _ in range(MAX_ITERATIONS):
            reply: Reply = provider.call(
                messages, system=system, tools=TOOLS, on_text_delta=on_text_delta,
            )

            if reply.stop_reason == "tool_use":
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
                    observation = _truncate_observation(observation)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": observation,
                    })
                continue

            final_text = reply.text
            note = _STOP_REASON_NOTES.get(reply.stop_reason)
            if note:
                final_text = f"{final_text}\n\n{note}" if final_text else note
            elif reply.stop_reason != "end_turn":
                final_text = f"{final_text}\n\n[stopped: {reply.stop_reason}]"

            messages.append({"role": "assistant", "content": final_text})
            return final_text

        messages.append({"role": "assistant", "content": MAX_ITERATIONS_NOTE})
        return MAX_ITERATIONS_NOTE
    except KeyboardInterrupt:
        _backfill_orphan_tool_results(messages)
        messages.append({"role": "assistant", "content": INTERRUPT_NOTE})
        print()
        return INTERRUPT_NOTE