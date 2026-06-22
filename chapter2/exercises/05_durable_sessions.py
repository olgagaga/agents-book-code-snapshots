"""Exercise 5 (stretch): durable sessions.

/save <file>  — write the messages list to disk.
/load <file>  — replace the in-memory messages with the file contents.

The save uses an atomic write: we serialize to a `.tmp` sibling first,
fsync optionally, then `os.replace()` it onto the real path. `os.replace`
is atomic on every supported OS, so a process kill mid-save can never
leave the user looking at a half-written file.

Compare with `SessionManager.save` in
nanobot/nanobot/session/manager.py — same pattern. Nanobot additionally
stores one JSON object per line (JSONL) instead of one big array; that
shape lets a corrupt write lose at most the last line, and lets you
`tail -f` a live session to watch it grow. For chapter 2 we keep it
simple and use a plain JSON file.
"""

import os
import json
from pathlib import Path

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


def save_messages(messages: list[dict], path: Path) -> None:
    """Atomically write `messages` to `path` as JSON."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_messages(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def handle_command(cmd: str, messages: list[dict]) -> bool:
    parts = cmd.split(maxsplit=1)
    name = parts[0]
    arg = parts[1] if len(parts) > 1 else None

    if name == "/save":
        if not arg:
            print("(usage: /save <filename>)\n")
            return True
        save_messages(messages, Path(arg))
        print(f"(saved {len(messages)} messages to {arg})\n")
        return True
    if name == "/load":
        if not arg:
            print("(usage: /load <filename>)\n")
            return True
        loaded = load_messages(Path(arg))
        messages.clear()
        messages.extend(loaded)
        print(f"(loaded {len(messages)} messages from {arg})\n")
        return True
    return False


def chat() -> None:
    messages: list[dict] = []
    print("chat — Ctrl-D or empty line to exit. Commands: /save <file> /load <file>\n")
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
