"""Exercise 5 (stretch): nanobot's prompt-caching strategy.

Nanobot's `_apply_cache_control` (in
`nanobot/nanobot/providers/anthropic_provider.py`, around line 379)
places markers in three places: the tail of the system prompt,
`messages[-2]`, and indexed tool entries.

This simplified version hits the first two: it attaches
`cache_control={"type": "ephemeral"}` to the system message *and* to
`messages[-2]` (skipping the second marker for short conversations).

Run a 6-turn conversation against a padded system prompt (the Westwind
handbook trick from Chapter 4 Exercise 5 pushes it above the 4,096-token
floor) and watch `cache_read_input_tokens` climb from turn 3 onward.

Why `messages[-2]` and not `messages[-1]`? On turn N, `messages[-1]` is
the user message we just appended — its content is *new* every turn, so a
marker on it would never match a previous prefix. `messages[-2]` is the
*assistant* reply from turn N-1 — it is fixed by the time turn N starts,
which is exactly what makes its prefix cacheable. Try moving the marker
to `messages[-1]` to confirm: the read counts collapse to zero.
"""

import os
from typing import Iterator

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-opus-4-6"

HANDBOOK = (
    "The Westwind Project is a long-running internal initiative whose "
    "stated goal is to consolidate the engineering team's onboarding "
    "documentation into a single authoritative handbook. "
) * 200


def _system_with_cache(text: str) -> list[dict]:
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _messages_with_cache(messages: list[dict]) -> list[dict]:
    """Mark messages[-2] as a cache breakpoint, if there are at least three."""
    if len(messages) < 3:
        return messages
    out = [dict(m) for m in messages]
    target = out[-2]
    target["content"] = [
        {
            "type": "text",
            "text": target["content"],
            "cache_control": {"type": "ephemeral"},
        }
    ]
    return out


def stream_turn(
    messages: list[dict], system_text: str, question: str
) -> None:
    messages.append({"role": "user", "content": question})
    print(f"q: {question}")
    print("a: ", end="", flush=True)

    chunks: list[str] = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=120,
        system=_system_with_cache(system_text),
        messages=_messages_with_cache(messages),
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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env")

    system_text = (
        "You are a project assistant. Below is the project handbook. "
        "Treat it as authoritative.\n\n" + HANDBOOK
    )
    questions = [
        "In one sentence, what is the Westwind Project?",
        "Why was the handbook created?",
        "Summarize the goal in five words.",
        "Who is the intended audience?",
        "Rephrase the goal as a question.",
        "Now phrase it as a tagline.",
    ]
    msgs: list[dict] = []
    for q in questions:
        stream_turn(msgs, system_text, q)
