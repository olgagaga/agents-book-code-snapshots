"""Exercise 2: Inspect the response object.

With adaptive thinking enabled, response.content is no longer a single
text block. Print it raw to see what is actually there, and watch the
token bill: thinking tokens count toward output usage.
"""

import os

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=2048,
    thinking={"type": "adaptive"},
    messages=[{"role": "user", "content": "Think briefly before answering: what is 17 * 23?"}],
)

print("--- response.content ---")
for i, block in enumerate(response.content):
    print(f"[{i}] type={block.type!r}")
    print(block)
    print()

print("--- response.stop_reason ---")
print(response.stop_reason)

print("\n--- response.usage ---")
print(response.usage)
