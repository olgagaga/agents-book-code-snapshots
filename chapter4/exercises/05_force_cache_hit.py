"""Exercise 5: Force a cache hit.

Pad the system prompt above the model's minimum cacheable size (4,096
tokens for Opus 4.7) and re-run the multi-turn conversation. The first
turn should report a non-zero `cache_creation_input_tokens`; every turn
within the next five minutes should report `cache_read_input_tokens`
close to that number.

The padding is a fake handbook — repeated lorem-ipsum-style content. In a
real agent, the same role is played by long persona files, accumulated
memory, and tool catalogs.
"""

import os

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"

# A 30-word sentence repeated until comfortably above the 4,096-token floor.
# ~6,000 tokens at the rule of thumb of 4 chars / token.
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

QUESTIONS = [
    "In one sentence, what is the Westwind Project?",
    "Why was the handbook created?",
    "Summarize the goal in five words.",
    "Who is the intended audience?",
    "Rephrase the goal as a question.",
]


def turn(messages: list[dict], question: str) -> None:
    messages.append({"role": "user", "content": question})
    print(f"q: {question}")
    print("a: ", end="", flush=True)
    chunks: list[str] = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=200,
        system=SYSTEM,
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
