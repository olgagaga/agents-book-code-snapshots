"""Exercise 3: Write a robust text extractor.

response.content is a list. Indexing [0] is fine until thinking blocks
or tool-use blocks show up — then it silently returns the wrong block.
This helper walks the whole list and joins every text block.

Compare with `_parse_response` in
nanobot/nanobot/providers/anthropic_provider.py — same shape, with
tool_use and thinking routed onto separate fields instead of dropped.
"""

import os

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def extract_text(response) -> str:
    """Concatenate every text block in the response, ignoring others."""
    return "".join(
        block.text for block in response.content if block.type == "text"
    )


def call(prompt: str, **extra) -> object:
    return client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        **extra,
    )


if __name__ == "__main__":
    plain = call("Say hello in five words.")
    print("plain reply:", repr(extract_text(plain)))

    thinking = call(
        "Think briefly before answering: what is 17 * 23?",
        thinking={"type": "adaptive"},
    )
    print("with thinking:", repr(extract_text(thinking)))
    print(f"  ({len(thinking.content)} blocks total, "
          f"types: {[b.type for b in thinking.content]})")
