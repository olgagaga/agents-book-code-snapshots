import os

import dotenv

from loop import agent_step
from context import build_context
from providers.anthropic_provider import AnthropicProvider
from providers.base import Provider
from providers.fallback_provider import FallbackProvider
from providers.openai_compatible_provider import OpenAIProvider

dotenv.load_dotenv()

chain: list[Provider] = []
if os.environ.get("ANTHROPIC_API_KEY"):
    chain.append(AnthropicProvider())
if os.environ.get("OPENAI_API_KEY"):
    chain.append(OpenAIProvider(model="gpt-5"))
if os.environ.get("GEMINI_API_KEY"):
    chain.append(OpenAIProvider(
        model="gemini-2.5-flash-lite",
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ))
DEFAULT_PROVIDER = FallbackProvider(chain)



def chat(provider: Provider | None = None) -> None:
    """Run an interactive chat loop, dispatching each user turn to the agent loop."""
    if provider is None:
        provider = DEFAULT_PROVIDER

    system = build_context()
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

        print("\nassistant: ", end="", flush=True)
        reply = agent_step(
            user_input, messages, provider, system,
            on_text_delta=lambda delta: print(delta, end="", flush=True),
        )
        print("\n")


if __name__ == "__main__":
    chat()
