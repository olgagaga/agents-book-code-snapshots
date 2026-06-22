"""Exercise 6 (stretch): Cancel a stream.

Wrap the streaming loop in `try / except KeyboardInterrupt`. When the
user hits Ctrl-C mid-reply, break out cleanly, append the partial reply
to `messages`, and return to the input prompt. The conversation
continues from a half-finished assistant turn.

The model is fine receiving a partial reply followed by a new user
message — it sees the assistant abandoned mid-sentence and writes the
next turn as a fresh response. Chapter 9 builds the full interrupt
story (stop reasons, mid-tool cancellation, etc.).

Try it: ask the agent to write you a long essay, then hit Ctrl-C after
a sentence or two and ask a follow-up question.
"""

import os
from typing import Iterator

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"


def llm(messages: list[dict]) -> Iterator[str]:
    with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def chat() -> None:
    messages: list[dict] = []
    print("chat — Ctrl-D or empty line to exit. Ctrl-C cancels a reply.\n")
    while True:
        try:
            user_input = input("you: ").strip()
        except EOFError:
            print()
            break
        if not user_input:
            break

        messages.append({"role": "user", "content": user_input})
        print("\nassistant: ", end="", flush=True)
        chunks: list[str] = []
        try:
            for text in llm(messages):
                print(text, end="", flush=True)
                chunks.append(text)
        except KeyboardInterrupt:
            print(" [interrupted]", flush=True)
        print("\n")

        partial = "".join(chunks).strip() or "[interrupted]"
        messages.append({"role": "assistant", "content": partial})


if __name__ == "__main__":
    chat()
