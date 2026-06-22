"""Exercise 4: Watch the cache.

Add `cache_control={"type": "ephemeral"}` to the stream and print
`cache_creation_input_tokens` and `cache_read_input_tokens` after every
turn. With a small system prompt (well below the model's per-block
minimum), both numbers will stay zero — the cache silently does nothing.
That is the expected baseline; Exercise 5 grows the prefix until the
cache kicks in.
"""

import os

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"

SMALL_SYSTEM = "You are a terse assistant. Answer in one sentence."

QUESTIONS = [
    "What is a hash table?",
    "What is a binary search tree?",
    "What is a linked list?",
    "What is a heap?",
    "What is a trie?",
]


def turn(messages: list[dict], question: str) -> None:
    messages.append({"role": "user", "content": question})
    print(f"q: {question}")
    print("a: ", end="", flush=True)
    chunks: list[str] = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=200,
        system=SMALL_SYSTEM,
        cache_control={"type": "ephemeral"},
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            chunks.append(text)
        print()
        final = stream.get_final_message()

    messages.append({"role": "assistant", "content": "".join(chunks)})

    u = final.usage
    print(
        f"   input={u.input_tokens}  "
        f"cache_creation={getattr(u, 'cache_creation_input_tokens', 0)}  "
        f"cache_read={getattr(u, 'cache_read_input_tokens', 0)}"
    )
    print()


if __name__ == "__main__":
    messages: list[dict] = []
    for q in QUESTIONS:
        turn(messages, q)
