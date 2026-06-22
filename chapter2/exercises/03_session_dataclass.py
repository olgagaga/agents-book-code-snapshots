"""Exercise 3: Wrap the messages list in a Session dataclass.

Compare with `Session` in nanobot/nanobot/session/manager.py. Things this
toy version is missing, in rough order of when each becomes load-bearing:

- `metadata: dict`             — per-session scratch space (workspace, channel
                                  state, etc.). Used from Chapter 3 onward.
- `last_consolidated: int`     — index marking how much of `messages` has
                                  already been summarized into long-term memory.
                                  Part 4.
- `get_history(...)`           — return the tail of `messages` bounded by a
                                  message count and/or token budget, with
                                  guards against starting on an assistant turn
                                  or an orphan tool result. Chapters 7 and 15.
- `enforce_file_cap(...)`      — keep on-disk size bounded by archiving old
                                  prefixes. Chapters 16-17.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@dataclass
class Session:
    """A single conversation's state — the production version of `messages: list[dict]`."""

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.updated_at = datetime.now()


def llm(messages: list[dict]) -> str:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text


def chat() -> None:
    session = Session(key="cli:default")
    print("chat — Ctrl-D or empty line to exit\n")
    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            break
        session.add_message("user", user_input)
        reply = llm(session.messages)
        session.add_message("assistant", reply)
        print(f"assistant: {reply}\n")


if __name__ == "__main__":
    chat()
