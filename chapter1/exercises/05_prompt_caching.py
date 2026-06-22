"""Exercise 5 (stretch): prompt caching.

Mark a long, stable prefix with `cache_control: {"type": "ephemeral"}`
and the provider will cache it. Subsequent calls within ~5 minutes that
hit the same prefix are billed at roughly 1/10th the input rate.

What to watch in the output:
- First call:  usage.cache_creation_input_tokens > 0  (paid full price + a small write fee)
- Second call: usage.cache_read_input_tokens > 0      (paid the discount)

Note: Anthropic only caches blocks above a model-dependent token floor
(~1024 tokens for Sonnet/Opus, ~2048 for Haiku). The fake doc below is
padded so the system prompt comfortably clears the threshold.
"""

import os

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"

LONG_DOC = (
    "You are a research assistant. Below is the project handbook. "
    "Treat it as authoritative.\n\n"
    + ("This is a fake handbook entry. " * 400)
)

SYSTEM = [
    {
        "type": "text",
        "text": LONG_DOC,
        "cache_control": {"type": "ephemeral"},
    }
]


def ask(question: str):
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    print(f"q: {question}")
    print(f"a: {response.content[0].text}")
    print(f"   usage: {response.usage}")
    print()


if __name__ == "__main__":
    ask("In one short sentence, what is this handbook about?")
    ask("Repeat the answer in different words.")
