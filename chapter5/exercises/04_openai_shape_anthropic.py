"""Exercise 4 (stretch): caller-shape-as-OpenAI.

In litellm, the *caller* always speaks OpenAI's request shape — a flat
`messages` list with a leading `{"role": "system", ...}` entry. Each
provider's `transform_request` then translates outward to its native
shape.

This file is the same idea applied to our `AnthropicProvider`: instead of
`(messages, system)`, it accepts a single `messages` list and pulls the
system message out internally.

Discussion (one paragraph, as the chapter asks).

Caller-shape-as-OpenAI is easier to scale. If every provider implements a
single `transform_request` from one canonical shape to its native shape,
adding a new provider is one file with one method. The price is a leaky
abstraction at the *caller* boundary: the agent has to know that "system"
goes in the messages list, that some models accept multi-turn system
messages and others do not, and so on. Caller-shape-as-our-`Provider`
(the `(messages, system)` split we used in the chapter) is easier to
*teach*: each parameter has one obvious meaning, the call site reads
left-to-right, and provider-specific knobs stay inside the provider. So
the project tradeoff is real — the OpenAI shape wins at scale, the split
shape wins at clarity. Litellm picked the first; this book picked the
second on purpose.
"""

import os
from abc import ABC, abstractmethod
from typing import Iterator

import anthropic
import dotenv

dotenv.load_dotenv()


class Provider(ABC):
    @abstractmethod
    def stream(self, messages: list[dict]) -> Iterator[str]:
        ...


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Pull a leading system message out of an OpenAI-shape list.

    Returns `(system_text, remaining_messages)`. If the first message is
    not a system message, returns `("", messages)` unchanged.
    """
    if messages and messages[0].get("role") == "system":
        return messages[0]["content"], messages[1:]
    return "", messages


class AnthropicProvider(Provider):
    def __init__(self, model: str = "claude-opus-4-6", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()

    def stream(self, messages: list[dict]) -> Iterator[str]:
        system, rest = _split_system(messages)
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=rest,
        ) as stream:
            for text in stream.text_stream:
                yield text


def chat_demo(provider: Provider) -> None:
    """A minimal chat() that speaks OpenAI shape end-to-end."""
    messages: list[dict] = [
        {"role": "system", "content": "You are concise."},
    ]
    print("chat (OpenAI-shape) — empty line to exit\n")
    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            break
        messages.append({"role": "user", "content": user_input})

        print("\nassistant: ", end="", flush=True)
        chunks: list[str] = []
        for text in provider.stream(messages):
            print(text, end="", flush=True)
            chunks.append(text)
        print("\n")
        messages.append({"role": "assistant", "content": "".join(chunks)})


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env")
    chat_demo(AnthropicProvider())
