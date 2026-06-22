"""Exercise 1: Native Gemini provider.

A `GeminiProvider` that uses Google's native `google-genai` SDK instead of
going through the OpenAI-compatible endpoint. The shape is more different
than OpenAI's: `contents` instead of `messages`, `system_instruction`
inside a config object, "model" instead of "assistant" for prior replies,
and a separate streaming method.

Install: `uv add google-genai`. Set `GEMINI_API_KEY` in `.env`.

Whether you would rather have this or use Gemini through `OpenAIProvider`
depends on whether you need any Gemini-specific feature — tuned models,
file-based grounding, Vertex AI. For the common chat path, the OpenAI
compat endpoint is the practical default.
"""

import os
from abc import ABC, abstractmethod
from typing import Iterator

import dotenv
from google import genai
from google.genai import types

dotenv.load_dotenv()


class Provider(ABC):
    @abstractmethod
    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        ...


class GeminiProvider(Provider):
    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        max_output_tokens: int = 16000,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        contents: list[types.Content] = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=m["content"])])
            )

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=self.max_output_tokens,
        )

        for chunk in self.client.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        ):
            if chunk.text:
                yield chunk.text


if __name__ == "__main__":
    provider = GeminiProvider()
    messages = [{"role": "user", "content": "Who made you, in one short sentence?"}]
    for text in provider.stream(messages, system="Be concise."):
        print(text, end="", flush=True)
    print()
