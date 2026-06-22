"""Exercise 5: Streaming tool-call spinner.

`on_text_delta` lets `chat()` print the model's text character by character.
Tool calls are silent — between text bursts the user sees an unmoving
cursor while the loop runs whatever the model asked for. We add a second
lifecycle hook, `on_tool_call_start: Callable[[str], None]`, that fires the
moment a tool-use block starts streaming. `chat()` wires it to a printer:
`[calling now()]` between text bursts is enough movement to reassure a
human that work is happening.

Where the hook fires:

- Anthropic: at the `content_block_start` event where `block.type == "tool_use"`.
  The block's `name` is already known at that moment; arguments stream in
  later as `input_json_delta` events.
- OpenAI: the first chunk in which a given `tc_delta.index` carries a
  non-empty `function.name`. The name can arrive in any of the early
  chunks, not necessarily the first one for that index, so we track a
  per-index "announced" flag and fire once.

This file shows the two patched `call` methods next to a small fake-stream
demo so the wiring is testable without an API key.
"""

import json
from dataclasses import dataclass, field
from typing import Callable, Iterator


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


# -- Anthropic-style patched call ------------------------------------------

def anthropic_call(
    event_stream: Iterator,
    on_text_delta: Callable[[str], None] | None = None,
    on_tool_call_start: Callable[[str], None] | None = None,  # <-- new
) -> Reply:
    """The streaming tail of `AnthropicProvider.call`, parameterised on the
    event iterator so we can drive it with a hand-built fake stream."""
    text_parts: list[str] = []
    tc_by_index: dict[int, ToolCall] = {}
    partial_args: dict[int, str] = {}

    for event in event_stream:
        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                tc_by_index[event.index] = ToolCall(id=block.id, name=block.name, args={})
                partial_args[event.index] = ""
                if on_tool_call_start:                          # <-- fire here
                    on_tool_call_start(block.name)
        elif event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                text_parts.append(delta.text)
                if on_text_delta:
                    on_text_delta(delta.text)
            elif delta.type == "input_json_delta":
                partial_args[event.index] += delta.partial_json

    for i, raw in partial_args.items():
        tc_by_index[i].args = json.loads(raw) if raw else {}
    return Reply(text="".join(text_parts), tool_calls=list(tc_by_index.values()))


# -- OpenAI-style patched call ---------------------------------------------

def openai_call(
    chunks: Iterator,
    on_text_delta: Callable[[str], None] | None = None,
    on_tool_call_start: Callable[[str], None] | None = None,  # <-- new
) -> Reply:
    text_parts: list[str] = []
    partial: dict[int, dict] = {}
    announced: set[int] = set()                                # <-- per-index flag

    for chunk in chunks:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            text_parts.append(delta.content)
            if on_text_delta:
                on_text_delta(delta.content)
        for tc_delta in (delta.tool_calls or []):
            slot = partial.setdefault(tc_delta.index, {"id": "", "name": "", "args": ""})
            if tc_delta.id:
                slot["id"] = tc_delta.id
            fn = tc_delta.function
            if fn and fn.name:
                slot["name"] = fn.name
                if tc_delta.index not in announced:             # <-- fire once
                    announced.add(tc_delta.index)
                    if on_tool_call_start:
                        on_tool_call_start(fn.name)
            if fn and fn.arguments:
                slot["args"] += fn.arguments

    tool_calls = [
        ToolCall(id=p["id"], name=p["name"], args=json.loads(p["args"] or "{}"))
        for p in partial.values()
    ]
    return Reply(text="".join(text_parts), tool_calls=tool_calls)


# -- Tiny fakes for the demo -----------------------------------------------

class _Event:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _anthropic_fake_stream():
    yield _Event(type="content_block_delta", index=0,
                 delta=_Event(type="text_delta", text="Let me check. "))
    yield _Event(type="content_block_start", index=1,
                 content_block=_Event(type="tool_use", id="t1", name="now"))
    yield _Event(type="content_block_delta", index=1,
                 delta=_Event(type="input_json_delta", partial_json=""))
    yield _Event(type="content_block_start", index=2,
                 content_block=_Event(type="tool_use", id="t2", name="wordcount"))
    yield _Event(type="content_block_delta", index=2,
                 delta=_Event(type="input_json_delta",
                              partial_json='{"text":"the quick"}'))


class _Chunk:
    def __init__(self, choices):
        self.choices = choices


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _ToolCallDelta:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = _Event(name=name, arguments=arguments)


def _openai_fake_stream():
    yield _Chunk([_Choice(_Delta(content="Let me check. "))])
    yield _Chunk([_Choice(_Delta(tool_calls=[_ToolCallDelta(0, id="call_1", name="now")]))])
    yield _Chunk([_Choice(_Delta(tool_calls=[_ToolCallDelta(0, arguments="{}")]))])
    yield _Chunk([_Choice(_Delta(tool_calls=[_ToolCallDelta(1, id="call_2", name="wordcount")]))])
    yield _Chunk([_Choice(_Delta(tool_calls=[_ToolCallDelta(1, arguments='{"text":"the quick"}')]))])


if __name__ == "__main__":
    text_buf: list[str] = []
    starts: list[str] = []
    reply = anthropic_call(
        _anthropic_fake_stream(),
        on_text_delta=text_buf.append,
        on_tool_call_start=lambda name: starts.append(f"[calling {name}()]"),
    )
    assert "".join(text_buf) == "Let me check. "
    assert starts == ["[calling now()]", "[calling wordcount()]"], starts
    assert [(tc.name, tc.args) for tc in reply.tool_calls] == [
        ("now", {}), ("wordcount", {"text": "the quick"}),
    ]
    print("anthropic spinner trace:", starts)

    text_buf.clear()
    starts.clear()
    reply = openai_call(
        _openai_fake_stream(),
        on_text_delta=text_buf.append,
        on_tool_call_start=lambda name: starts.append(f"[calling {name}()]"),
    )
    assert "".join(text_buf) == "Let me check. "
    assert starts == ["[calling now()]", "[calling wordcount()]"], starts
    assert [(tc.name, tc.args) for tc in reply.tool_calls] == [
        ("now", {}), ("wordcount", {"text": "the quick"}),
    ]
    print("openai spinner trace:   ", starts)

    print("\nwiring from chat():")
    print('    reply = agent_step(')
    print('        user_input, messages, provider, system,')
    print('        on_text_delta=lambda d: print(d, end="", flush=True),')
    print('        on_tool_call_start=lambda n: print(f"\\n[calling {n}()]", flush=True),')
    print('    )')
