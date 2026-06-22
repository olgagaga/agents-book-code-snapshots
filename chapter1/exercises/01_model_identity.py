"""Exercise 1: Model identity.

Ask three Claude models the same question — "what model are you?" — and
compare the answers. The lesson: a model is not a reliable narrator about
itself. Treat self-reports as a starting hypothesis, not a fact.
"""

import os

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

PROMPT = "Without searching the web, what model are you?"
MODELS = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"]


def ask(model: str, prompt: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    for model in MODELS:
        print(f"=== {model} ===")
        print(ask(model, PROMPT))
        print()
