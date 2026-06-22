"""Exercise 4: `tool_choice`.

Both Anthropic and OpenAI let the caller force or forbid tool use via a
`tool_choice` parameter [1][2]. We add an optional `tool_choice` argument
to `Provider.call`, pick Anthropic's shape as the canonical one (it's a
small dict that survives a one-line translation), and use the new control
to write a one-shot helper that forces a `now` call and returns the result
string directly.

Canonical shape (Anthropic-style):

    {"type": "auto"}                     -> model decides (default)
    {"type": "any"}                      -> must call SOME tool
    {"type": "tool", "name": "now"}      -> must call THIS specific tool

OpenAI's equivalents:

    "auto"
    "required"
    {"type": "function", "function": {"name": "now"}}

The translation is six lines inside `OpenAIProvider`. Apply the matching
patches to `agent/providers/{base,anthropic_provider,openai_compatible_provider}.py`.

[1] https://platform.claude.com/docs/en/build-with-claude/tool-use
[2] https://platform.openai.com/docs/guides/function-calling
"""

import os
from datetime import datetime, timezone
from typing import Callable

import dotenv

# Import from the chapter's snapshot — the real `agent/` would import the
# same modules via the parent package, but for a runnable exercise we point
# straight at the snapshot's providers.
import sys
import pathlib

_CORE = pathlib.Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(_CORE))

import anthropic
from openai import OpenAI

from providers.base import Provider, Reply, ToolCall
from tools.base import Tool


def _to_openai_tool_choice(tc: dict | None) -> str | dict | None:
    """Translate an Anthropic-shape tool_choice into OpenAI's shape."""
    if tc is None:
        return None
    if tc["type"] == "auto":
        return "auto"
    if tc["type"] == "any":
        return "required"
    if tc["type"] == "tool":
        return {"type": "function", "function": {"name": tc["name"]}}
    raise ValueError(f"unknown tool_choice shape: {tc!r}")


class AnthropicProviderWithChoice(Provider):
    """`AnthropicProvider` from the chapter, plus a `tool_choice` argument.

    Only the lines that changed from the chapter version are annotated.
    """

    def __init__(self, model: str = "claude-opus-4-6", max_tokens: int = 16000):
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()

    @staticmethod
    def _to_anthropic_tools(tools: list[Tool]) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.schema}
                for t in tools]

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
        tool_choice: dict | None = None,  # <-- new
    ) -> Reply:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(list(tools))
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice  # <-- passed through as-is

        text_parts: list[str] = []
        tc_by_index: dict[int, ToolCall] = {}
        partial_args: dict[int, str] = {}
        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tc_by_index[event.index] = ToolCall(
                            id=block.id, name=block.name, args={},
                        )
                        partial_args[event.index] = ""
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_parts.append(delta.text)
                        if on_text_delta:
                            on_text_delta(delta.text)
                    elif delta.type == "input_json_delta":
                        partial_args[event.index] += delta.partial_json
        import json
        for i, raw in partial_args.items():
            tc_by_index[i].args = json.loads(raw) if raw else {}
        return Reply(text="".join(text_parts), tool_calls=list(tc_by_index.values()))


class OpenAIProviderWithChoice(Provider):
    def __init__(
        self,
        model: str = "gpt-5",
        max_tokens: int = 16000,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _to_openai_tools(tools: list[Tool]) -> list[dict]:
        return [{"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.schema}}
                for t in tools]

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
        tool_choice: dict | None = None,  # <-- new
    ) -> Reply:
        oai_messages: list[dict] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)
        kwargs: dict = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": oai_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = self._to_openai_tools(list(tools))
        translated = _to_openai_tool_choice(tool_choice)
        if translated is not None:
            kwargs["tool_choice"] = translated  # <-- translated, not passed verbatim

        import json
        text_parts: list[str] = []
        partial: dict[int, dict] = {}
        for chunk in self.client.chat.completions.create(**kwargs):
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
                if fn and fn.arguments:
                    slot["args"] += fn.arguments
        tool_calls = [
            ToolCall(id=p["id"], name=p["name"], args=json.loads(p["args"] or "{}"))
            for p in partial.values()
        ]
        return Reply(text="".join(text_parts), tool_calls=tool_calls)


def _now(args: dict) -> str:
    return datetime.now(timezone.utc).isoformat()


NOW_TOOL = Tool(
    name="now",
    description="Return the current UTC time as an ISO 8601 string.",
    schema={"type": "object", "properties": {}, "required": []},
    run=_now,
)


def what_time_is_it(provider) -> str:
    """One-shot helper: force a `now` call and run the tool locally.

    Skips the agent loop entirely. We force the model to emit exactly the
    tool call we want, then run the tool ourselves and return its result.
    No second model round-trip; no risk of the model deciding to chat.
    """
    reply = provider.call(
        messages=[{"role": "user", "content": "what time is it?"}],
        tools=[NOW_TOOL],
        tool_choice={"type": "tool", "name": "now"},
    )
    assert reply.tool_calls, "tool_choice should have forced a call"
    return NOW_TOOL.run(reply.tool_calls[0].args)


if __name__ == "__main__":
    dotenv.load_dotenv()

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("anthropic:", what_time_is_it(AnthropicProviderWithChoice()))
    else:
        print("ANTHROPIC_API_KEY not set; skipping anthropic check.")

    if os.environ.get("OPENAI_API_KEY"):
        print("openai:   ", what_time_is_it(OpenAIProviderWithChoice()))
    else:
        print("OPENAI_API_KEY not set; skipping openai check.")
