"""Exercise 4 (stretch): slash commands.

/reset  — drop all messages and start over.
/undo   — pop the last user/assistant pair.
/print  — show the current messages list.

Slash commands run locally; they do not produce an LLM turn.
"""

import os
import json

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def llm(messages: list[dict]) -> str:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text


def handle_command(cmd: str, messages: list[dict]) -> bool:
    """Handle a slash command. Return True if the command was recognized."""
    if cmd == "/reset":
        messages.clear()
        print("(messages cleared)\n")
        return True
    if cmd == "/undo":
        # Pop trailing assistant turn, then the user turn it answered.
        for _ in range(2):
            if messages:
                messages.pop()
        print(f"(undid one turn — {len(messages)} messages remain)\n")
        return True
    if cmd == "/print":
        print(json.dumps(messages, indent=2, ensure_ascii=False))
        print()
        return True
    return False


def chat() -> None:
    messages: list[dict] = []
    print("chat — Ctrl-D or empty line to exit. Commands: /reset /undo /print\n")
    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            break
        if user_input.startswith("/"):
            if not handle_command(user_input, messages):
                print(f"(unknown command: {user_input})\n")
            continue
        messages.append({"role": "user", "content": user_input})
        reply = llm(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"assistant: {reply}\n")


if __name__ == "__main__":
    chat()
