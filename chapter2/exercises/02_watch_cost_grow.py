"""Exercise 2: Watch the cost grow.

llm() now returns (text, usage). After each turn we print input_tokens.
Run the chat for 8+ turns and the input_tokens series should grow roughly
linearly with the turn number — sum it up across N turns and you get the
quadratic total cost shape from the chapter.
"""

import os

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def llm(messages: list[dict]) -> tuple[str, object]:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text, response.usage


def chat() -> None:
    messages: list[dict] = []
    print("chat — Ctrl-D or empty line to exit\n")
    turn = 0
    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            break
        turn += 1
        messages.append({"role": "user", "content": user_input})
        reply, usage = llm(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"assistant: {reply}")
        print(f"  [turn {turn}: input_tokens={usage.input_tokens} "
              f"output_tokens={usage.output_tokens}]\n")


if __name__ == "__main__":
    chat()
