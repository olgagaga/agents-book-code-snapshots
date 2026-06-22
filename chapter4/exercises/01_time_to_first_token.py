"""Exercise 1: Time to first token vs. total time.

Wrap the streaming `llm` in timing instrumentation. Compare time-to-first-token
to total wall-clock, then call the non-streaming version and confirm total
wall-clock is roughly the same. The gap between streaming TTFB and blocking
TTFB is the perceived-latency improvement streaming buys you.
"""

import os
import time
from typing import Iterator

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"
PROMPT = "Explain the Krebs cycle in around 200 words."


def llm_stream(prompt: str) -> Iterator[str]:
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def llm_blocking(prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def time_streaming() -> None:
    start = time.monotonic()
    first_token_at: float | None = None
    for _ in llm_stream(PROMPT):
        if first_token_at is None:
            first_token_at = time.monotonic()
    end = time.monotonic()
    assert first_token_at is not None
    print(
        f"streaming: ttfb={first_token_at - start:.2f}s  "
        f"total={end - start:.2f}s"
    )


def time_blocking() -> None:
    start = time.monotonic()
    _ = llm_blocking(PROMPT)
    end = time.monotonic()
    # For blocking, ttfb == total — there is no "first token" before the end.
    print(f"blocking:  ttfb={end - start:.2f}s  total={end - start:.2f}s")


if __name__ == "__main__":
    time_streaming()
    time_blocking()
