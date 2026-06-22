"""Exercise 1: Watch the list grow.

Print the running message count at the top of every turn so you can see
the list grow by exactly two entries per exchange — one user, one assistant.
"""

import os

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


def chat() -> None:
    messages: list[dict] = []
    print("chat — Ctrl-D or empty line to exit\n")
    while True:
        print(f"--- {len(messages)} messages so far ---")
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            break
        messages.append({"role": "user", "content": user_input})
        reply = llm(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"assistant: {reply}\n")


if __name__ == "__main__":
    chat()
