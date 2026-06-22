"""Exercise 4 (stretch): count tokens before sending.

`messages.count_tokens` returns the input-token count for a candidate
prompt without actually sending it. Wrap it in a small helper that
multiplies by a hardcoded per-million-token rate to get a dollar
estimate. Pricing changes; treat the rates here as illustrative only.
"""

import os

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# Illustrative rates in USD per million input tokens. Update from the
# provider's pricing page; these are a snapshot, not a contract.
INPUT_RATE_PER_MTOK = {
    "claude-opus-4-6": 15.0,
    "claude-sonnet-4-6": 3.0,
    "claude-haiku-4-5": 1.0,
}


def cost_estimate(messages: list[dict], model: str) -> float:
    """Estimated USD cost of the input portion of one call."""
    n = client.messages.count_tokens(model=model, messages=messages).input_tokens
    rate = INPUT_RATE_PER_MTOK.get(model, 5.0)
    return n * rate / 1_000_000


if __name__ == "__main__":
    short = [{"role": "user", "content": "Say hello in five words."}]
    long_msg = [{"role": "user", "content": "Summarize the following: " + "lorem ipsum " * 1000}]

    for model in INPUT_RATE_PER_MTOK:
        print(f"{model}:")
        print(f"  short prompt: ${cost_estimate(short, model):.6f}")
        print(f"  long prompt:  ${cost_estimate(long_msg, model):.6f}")
