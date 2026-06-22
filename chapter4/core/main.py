import anthropic
import os
import dotenv
from context import build_context
from typing import Iterator

dotenv.load_dotenv()  # Load environment variables from .env file

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def llm(messages: list[dict], system: str = "") -> Iterator[str]:
    """Stream the model's reply, yielding text deltas as they arrive."""
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system,
        cache_control={"type": "ephemeral"},
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text
    


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
        
        print("\nassistant: ", end="", flush=True)
        chunks: list[str] = []
        for text in llm(messages, system=system):
            print(text, end="", flush=True)
            chunks.append(text)
        print("\n")

        messages.append({"role": "assistant", "content": "".join(chunks)})




if __name__ == "__main__":
    chat()