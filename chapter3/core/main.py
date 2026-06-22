import anthropic
import os
import dotenv
from context import build_context

dotenv.load_dotenv()  # Load environment variables from .env file

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def llm(messages: list[dict], system: str = "") -> str:
    """Send a list of messages to the model and return the reply text."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return response.content[0].text


def chat() -> None:
    """Run an interactive chat loop, accumulating turns in a single messages list."""
    
    system = build_context() # build context
    
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
        reply = llm(messages, system=system) # pass context for every call
        messages.append({"role": "assistant", "content": reply})

        print(f"assistant: {reply}\n")


if __name__ == "__main__":
    chat()