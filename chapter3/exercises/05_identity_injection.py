"""Exercise 5 (stretch): Identity injection.

Prepend a small block of runtime metadata — date, OS, username — to the
system prompt. The agent can now answer "what's today?" or "where am I
running you?" without being told.

This is the smallest example of *dynamic context*: parts of the prompt
that change between calls. Compare to `_get_identity` in
nanobot/nanobot/agent/context.py, which adds the workspace path,
Python version, and channel identifier in the same shape.

Chapter 15 picks up the dynamic-prompt theme in earnest.
"""

import os
import pathlib
import platform
from datetime import date

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"

TEMPLATES_DIR = pathlib.Path(__file__).resolve().parent.parent / "core" / "templates"


def build_identity_block() -> str:
    return (
        "# Runtime\n\n"
        f"- Date: {date.today().isoformat()}\n"
        f"- OS: {platform.system()} ({platform.machine()})\n"
        f"- User: {os.getenv('USER') or os.getenv('USERNAME') or 'unknown'}\n"
    )


def build_context() -> str:
    parts = [build_identity_block()]
    if TEMPLATES_DIR.exists():
        for md_file in sorted(TEMPLATES_DIR.glob("*.md")):
            parts.append(md_file.read_text())
    return "\n\n".join(parts)


def ask(system: str, prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    system = build_context()
    print("--- identity block ---")
    print(build_identity_block())
    print("--- probes ---")
    for probe in ("What is today's date?", "Where am I running you?"):
        print(f"q: {probe}")
        print(f"a: {ask(system, probe)}")
        print()
