"""Exercise 2: Watch the system tokens.

Run the same question against a small system prompt and a large one.
Print `usage.input_tokens` after every reply and notice that the input
cost scales with the size of the system prompt — every turn, forever.

The projection at the end is a back-of-envelope estimate of what a
100-turn conversation would cost in input tokens with each prompt size.
This is the wastefulness Chapter 4 fixes with prompt caching.
"""

import os

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"

SMALL_PROMPT = "You are a concise, direct assistant.\n"

LARGE_PROMPT = (
    "You are a concise, direct assistant.\n\n"
    + ("Background: this is a fake handbook entry. " * 400)
)


def ask(system: str, prompt: str) -> int:
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.content[0].text.strip())
    print(f"   usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}")
    return response.usage.input_tokens


if __name__ == "__main__":
    question = "In one sentence, what is a list comprehension?"

    print(f"=== small system prompt ({len(SMALL_PROMPT)} chars) ===")
    small_in = ask(SMALL_PROMPT, question)
    print()
    print(f"=== large system prompt ({len(LARGE_PROMPT)} chars) ===")
    large_in = ask(LARGE_PROMPT, question)

    delta = large_in - small_in
    print()
    print("--- projection over 100 turns ---")
    print(f"extra input tokens per turn:   {delta}")
    print(f"extra input tokens (100 turns): {delta * 100:,}")
    print("Chapter 4 makes this ~10x cheaper with prompt caching.")
