"""Exercise 1: Tolerate malformed tool-call JSON.

The chapter's `AnthropicProvider.call` parses tool-call arguments with
`json.loads(raw)` once the stream finishes. Production Claude models almost
never produce malformed JSON in a tool-call channel — the API enforces the
schema before the bytes leave the server — but smaller or self-hosted models
do, and a single `JSONDecodeError` will crash the whole `agent_step` instead
of giving the model a chance to retry.

The fix is to catch the parse failure per tool call: set the offending
`ToolCall.args` to `{}` and append a synthetic note to `text_parts` so the
loop's next iteration sees a hint of what went wrong. The model reads its
own prior text as part of the conversation; an inline mention of the bad
call is enough for it to apologize and retry. The same patch applies to
`OpenAIProvider.call`, which has the exact same parse step.

This file is a focused study version: it inlines just the parsing tail of
both providers wrapped behind a small helper so we can unit-test the
recovery path without a real API. Apply the actual two-line diff to
`agent/providers/anthropic_provider.py` and `agent/providers/openai_compatible_provider.py`.
"""

import json

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def finalize_anthropic(
    text_parts: list[str],
    tc_by_index: dict[int, ToolCall],
    partial_args: dict[int, str],
) -> Reply:
    """The patched tail of `AnthropicProvider.call`.

    Replaces:

        for i, raw in partial_args.items():
            tc_by_index[i].args = json.loads(raw) if raw else {}

    with a per-call try/except that degrades to empty args plus a note in
    `text_parts` for the model to read on the next iteration.
    """
    for i, raw in partial_args.items():
        tc = tc_by_index[i]
        if not raw:
            tc.args = {}
            continue
        try:
            tc.args = json.loads(raw)
        except json.JSONDecodeError:
            tc.args = {}
            text_parts.append(
                f"\n[invalid JSON arguments for tool {tc.name!r}; retry with valid JSON]"
            )
    return Reply(text="".join(text_parts), tool_calls=list(tc_by_index.values()))


def finalize_openai(
    text_parts: list[str], partial: dict[int, dict],
) -> Reply:
    """The patched tail of `OpenAIProvider.call`.

    The OpenAI version parses inside the list comprehension that builds
    `tool_calls`; the recovery shape is the same, just unrolled into a loop
    so the per-call try/except has somewhere to live.
    """
    tool_calls: list[ToolCall] = []
    for p in partial.values():
        raw = p["args"] or ""
        if not raw:
            args = {}
        else:
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {}
                text_parts.append(
                    f"\n[invalid JSON arguments for tool {p['name']!r}; retry with valid JSON]"
                )
        tool_calls.append(ToolCall(id=p["id"], name=p["name"], args=args))
    return Reply(text="".join(text_parts), tool_calls=tool_calls)


if __name__ == "__main__":
    text = ["I'll look that up."]
    tc_by_index = {0: ToolCall(id="t1", name="wordcount", args={})}
    partial_args = {0: '{"text": "hello, world}'}  # missing closing quote
    reply = finalize_anthropic(text, tc_by_index, partial_args)
    assert reply.tool_calls[0].args == {}
    assert "invalid JSON" in reply.text
    assert "wordcount" in reply.text
    print("anthropic recovery:", reply.text)

    text = ["Counting words now."]
    partial = {0: {"id": "call_1", "name": "wordcount", "args": "{not json"}}
    reply = finalize_openai(text, partial)
    assert reply.tool_calls[0].args == {}
    assert "invalid JSON" in reply.text
    print("openai recovery:   ", reply.text)

    text = []
    tc_by_index = {0: ToolCall(id="t1", name="now", args={})}
    partial_args = {0: ""}
    reply = finalize_anthropic(text, tc_by_index, partial_args)
    assert reply.tool_calls[0].args == {}
    assert reply.text == ""
    print("empty-args case still works.")
