"""Exercise 4 (stretch): Hot reload.

Re-read the templates on every turn instead of once at startup. Now you
can edit `persona.md` between messages and watch the model's behavior
change mid-conversation.

Useful for iterating on a persona; risky in production because the
transcript no longer tells you which version of the prompt was active
when. Most production agents do not hot-reload — understand why before
you ship it.

To see the effect: start the chat, ask `who are you?`, then in another
window edit `persona.md` (e.g., add "Speak only in pirate dialect."),
save, ask `who are you?` again. The voice should change.
"""

import os
import pathlib

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

TEMPLATES_DIR = pathlib.Path(__file__).resolve().parent.parent / "core" / "templates"


def build_context() -> str:
    if not TEMPLATES_DIR.exists():
        return ""
    parts = [p.read_text() for p in sorted(TEMPLATES_DIR.glob("*.md"))]
    return "\n\n".join(parts)


def llm(messages: list[dict], system: str = "") -> str:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return response.content[0].text


def chat() -> None:
    messages: list[dict] = []
    print("chat (hot-reload) — Ctrl-D or empty line to exit\n")
    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            break
        system = build_context()  # rebuilt on every turn
        messages.append({"role": "user", "content": user_input})
        reply = llm(messages, system=system)
        messages.append({"role": "assistant", "content": reply})
        print(f"assistant: {reply}\n")


if __name__ == "__main__":
    chat()
