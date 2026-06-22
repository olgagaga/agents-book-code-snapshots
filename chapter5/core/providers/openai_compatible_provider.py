from typing import Iterator

from openai import OpenAI

from providers.base import Provider


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

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        oai_messages: list[dict] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        stream = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=oai_messages,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content