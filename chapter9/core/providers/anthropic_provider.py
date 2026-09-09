from typing import Callable

import json
import anthropic

from providers.base import Provider, Reply, ToolCall
from tools.base import Tool

_ANTHROPIC_STOP = {
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "tool_use": "tool_use",
    "refusal": "refusal",
}


class AnthropicProvider(Provider):
    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_tokens: int = 16000,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key)
    
    @staticmethod
    def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    })
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                out.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                }]})
        return out

    @staticmethod
    def _to_anthropic_tools(tools: list[Tool]) -> list[dict]:
        return [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.schema,
        } for t in tools]
    
    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._to_anthropic_messages(messages),
        }

        if system:
            kwargs["system"] = system
            kwargs["cache_control"] = {"type": "ephemeral"}
        
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(list(tools))
        
        text_parts: list[str] = []
        tc_by_index: dict[int, ToolCall] = {}
        partial_args: dict[int, str] = {}
        stop_reason_raw: str | None = None 

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
                elif event.type == "message_delta":                       
                    if event.delta.stop_reason:                           
                        stop_reason_raw = event.delta.stop_reason         

             
        for i, raw in partial_args.items():
            tc_by_index[i].args = json.loads(raw) if raw else {}

        return Reply(text="".join(text_parts), 
                     tool_calls=list(tc_by_index.values()),
                     stop_reason=_ANTHROPIC_STOP.get(stop_reason_raw or "", stop_reason_raw or "end_turn"),)