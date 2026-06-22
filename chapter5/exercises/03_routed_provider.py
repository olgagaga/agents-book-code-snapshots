"""Exercise 3: Cost-routed provider.

A `RoutedProvider` that picks a cheaper backend for short messages and a
stronger one for longer messages. The heuristic is intentionally trivial:
total content length under 2,000 characters → cheap; over → strong. This
is the routing pattern Chapter 19 will revisit when subagents pick their
own backend.
"""

import os
from abc import ABC, abstractmethod
from typing import Iterator

import anthropic
import dotenv

dotenv.load_dotenv()


class Provider(ABC):
    @abstractmethod
    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        ...


class AnthropicProvider(Provider):
    def __init__(self, model: str, max_tokens: int = 1024):
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


class RoutedProvider(Provider):
    def __init__(
        self,
        cheap: Provider,
        strong: Provider,
        threshold_chars: int = 2000,
    ):
        self.cheap = cheap
        self.strong = strong
        self.threshold_chars = threshold_chars

    def _pick(self, messages: list[dict], system: str) -> Provider:
        size = len(system) + sum(len(m["content"]) for m in messages)
        chosen = self.cheap if size < self.threshold_chars else self.strong
        print(f"[routed] size={size} -> {type(chosen).__name__}/{chosen.model}")
        return chosen

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        yield from self._pick(messages, system).stream(messages, system=system)


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env")

    provider = RoutedProvider(
        cheap=AnthropicProvider("claude-haiku-4-5"),
        strong=AnthropicProvider("claude-opus-4-6"),
    )

    short = [{"role": "user", "content": "Say hi."}]
    print("--- short ---")
    for text in provider.stream(short, system="Be concise."):
        print(text, end="", flush=True)
    print("\n")

    long_user = "Summarize the following passage. " + ("lorem ipsum " * 500)
    long_msg = [{"role": "user", "content": long_user}]
    print("--- long ---")
    for text in provider.stream(long_msg, system="Be concise."):
        print(text, end="", flush=True)
    print()
