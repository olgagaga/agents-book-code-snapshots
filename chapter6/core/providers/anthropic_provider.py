from typing import Iterator

import anthropic

from providers.base import Provider


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

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            cache_control={"type": "ephemeral"},
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text