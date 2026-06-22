"""Exercise 2: Smarter FallbackProvider.

The chapter version catches every `Exception` and falls back. In
production you want to distinguish *retryable* errors (rate limits, 5xx,
connection issues) from *non-retryable* ones (auth, 4xx bad-request) —
the second class will fail on the next provider too, and trying it just
adds latency to a guaranteed failure.

This version catches the specific SDK exceptions, falls back only on
retryable errors, re-raises everything else immediately, and adds
exponential backoff between attempts.

Test by setting `ANTHROPIC_API_KEY=bogus` (auth → re-raised immediately,
no fallback) versus a real key (works on first try).
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Iterator

import anthropic
import dotenv
import httpx
import openai

dotenv.load_dotenv()


class Provider(ABC):
    @abstractmethod
    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        ...


class AnthropicProvider(Provider):
    def __init__(self, model: str = "claude-opus-4-6", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text


class OpenAIProvider(Provider):
    def __init__(
        self,
        model: str = "gpt-5",
        max_tokens: int = 1024,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

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


RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    anthropic.RateLimitError,
    openai.RateLimitError,
    httpx.ConnectError,
    httpx.ReadTimeout,
)


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, RETRYABLE_EXCEPTIONS):
        return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and status >= 500


class SmartFallbackProvider(Provider):
    def __init__(self, providers: list[Provider], initial_backoff_s: float = 0.5):
        if not providers:
            raise ValueError("SmartFallbackProvider needs at least one provider")
        self.providers = providers
        self.initial_backoff_s = initial_backoff_s

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        last_error: Exception | None = None
        backoff = self.initial_backoff_s
        for i, provider in enumerate(self.providers):
            try:
                yielded_anything = False
                for text in provider.stream(messages, system=system):
                    yielded_anything = True
                    yield text
                return
            except Exception as e:
                if yielded_anything:
                    raise
                if not _is_retryable(e):
                    raise
                last_error = e
                if i + 1 < len(self.providers):
                    time.sleep(backoff)
                    backoff *= 2
        raise RuntimeError(f"All providers failed; last error: {last_error!r}")


if __name__ == "__main__":
    chain: list[Provider] = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        chain.append(AnthropicProvider())
    if os.environ.get("OPENAI_API_KEY"):
        chain.append(OpenAIProvider(model="gpt-5"))
    if not chain:
        raise RuntimeError("Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY in .env")

    provider = SmartFallbackProvider(chain)
    messages = [{"role": "user", "content": "Who made you, in one short sentence?"}]
    for text in provider.stream(messages, system="Be concise."):
        print(text, end="", flush=True)
    print()
