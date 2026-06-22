"""Exercise 9 (stretch): Cache audit.

Compute a per-turn cache hit ratio:

    hit_ratio = cache_read / (cache_read + cache_creation + input_tokens)

Run a 10-turn conversation against a padded system prompt and watch the
ratio climb toward 1.0 once the cache is warm.

About `_apply_cache_control`: nanobot puts the cache marker on the
*next-to-last* user message rather than the last one. The reason is
that the *last* user message is the one that just changed — caching
above it lets the *next* turn (whose new last-user message is appended
on top) reuse the cached prefix all the way through to the previous
assistant reply. Marking the truly-last message instead would invalidate
the cache on every turn.
"""

import os

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"

HANDBOOK = (
    "The Westwind Project is a long-running internal initiative whose "
    "stated goal is to consolidate the engineering team's onboarding "
    "documentation into a single authoritative handbook. "
) * 200

SYSTEM = [
    {
        "type": "text",
        "text": (
            "You are a project assistant. Below is the project handbook. "
            "Treat it as authoritative.\n\n"
            + HANDBOOK
        ),
        "cache_control": {"type": "ephemeral"},
    }
]


def hit_ratio(usage) -> float:
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    fresh = usage.input_tokens
    total = cache_read + cache_creation + fresh
    if total == 0:
        return 0.0
    return cache_read / total


def turn(messages: list[dict], question: str, turn_idx: int) -> None:
    messages.append({"role": "user", "content": question})
    chunks: list[str] = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=120,
        system=SYSTEM,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
        final = stream.get_final_message()

    messages.append({"role": "assistant", "content": "".join(chunks)})
    u = final.usage
    print(
        f"turn {turn_idx:2d}  "
        f"input={u.input_tokens:4d}  "
        f"cache_creation={getattr(u, 'cache_creation_input_tokens', 0):5d}  "
        f"cache_read={getattr(u, 'cache_read_input_tokens', 0):5d}  "
        f"hit_ratio={hit_ratio(u):.3f}"
    )


if __name__ == "__main__":
    messages: list[dict] = []
    for i in range(1, 11):
        turn(messages, f"Question {i}: paraphrase the project goal slightly differently.", i)
