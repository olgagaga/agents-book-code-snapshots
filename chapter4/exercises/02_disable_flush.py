"""Exercise 2: Disable flush=True.

Run the streaming loop without `flush=True` on the `print` call and watch
Python's stdout buffering hide the stream until the buffer fills or the
program exits. Useful thing to have seen once.

Tip: pipe the output through `cat` (`uv run 02_disable_flush.py | cat`) to
make the buffering even more aggressive — Python switches to fully buffered
mode when stdout is a pipe rather than a TTY, which means *nothing* will
appear until the entire response has finished.
"""

import os
from typing import Iterator

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"
PROMPT = "Tell me a 6-sentence story about a stubborn teapot."


def llm(prompt: str) -> Iterator[str]:
    with client.messages.stream(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


if __name__ == "__main__":
    print("--- with flush=True (correct) ---")
    for text in llm(PROMPT):
        print(text, end="", flush=True)
    print("\n")

    print("--- without flush=True (buffered) ---")
    for text in llm(PROMPT):
        print(text, end="")
    print("\n")
