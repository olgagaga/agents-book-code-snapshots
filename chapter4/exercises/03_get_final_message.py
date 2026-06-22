"""Exercise 3: Get the final message after streaming.

Streaming gives us text deltas. The metadata — `usage`, `stop_reason`,
the assembled `content` list — is still available via
`stream.get_final_message()` once the stream has been consumed.

This script calls `client.messages.stream(...)` directly (no generator
wrapper) so we can reach `get_final_message()` without redesigning `llm`'s
return type. Chapters 7 and 9 take stop reasons seriously; for now, just
confirm `end_turn` for a normal reply.
"""

import os

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"


def one_turn(prompt: str) -> None:
    print(f"q: {prompt}")
    print("a: ", end="", flush=True)
    with client.messages.stream(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()
        final = stream.get_final_message()

    print(f"   stop_reason: {final.stop_reason}")
    print(f"   usage:       {final.usage}")
    print()


if __name__ == "__main__":
    one_turn("In one sentence, what is a hash table?")
