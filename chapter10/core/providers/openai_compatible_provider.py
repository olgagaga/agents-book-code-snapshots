import json
from typing import Callable

from openai import OpenAI

from providers.base import Provider, Reply, ToolCall
from tools.base import Tool

_OPENAI_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}


class OpenAIProvider(Provider):
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
        return [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            },
        } for t in tools]

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
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

        text_parts: list[str] = []
        partial: dict[int, dict] = {}
        finish_reason_raw: str | None = None

        for chunk in self.client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason_raw = choice.finish_reason
            delta = choice.delta
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
        return Reply(text="".join(text_parts), 
                     tool_calls=tool_calls,
                     stop_reason=_OPENAI_STOP.get(finish_reason_raw or "", finish_reason_raw or "end_turn"),)