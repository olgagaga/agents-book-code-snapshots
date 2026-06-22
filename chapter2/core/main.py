import os

import anthropic
import dotenv

dotenv.load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def llm(messages: list[dict]) -> str:
    """Send a list of messages to the model and return the reply text."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text


def chat() -> None:
    """Run an interactive chat loop, accumulating turns in a single messages list."""
    messages: list[dict] = []
    print("chat — Ctrl-D or empty line to exit\n")
    while True:
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
