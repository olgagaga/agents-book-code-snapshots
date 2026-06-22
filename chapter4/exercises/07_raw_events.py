"""Exercise 7 (stretch): Raw events.

Replace `stream.text_stream` with iteration over `stream` directly.
Filter for the events we care about and reproduce `text_stream`'s
output, plus a `stop_reason` print at the end.

This is the path Chapter 7 takes for tool calls — `text_stream` filters
to text deltas, but tool-call deltas live in their own
`content_block_delta` events with `delta.type == "input_json_delta"`.
Once you can iterate raw events, you can dispatch on whatever block type
matters to you.
"""

import os

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"
PROMPT = "In two sentences, explain why streaming reduces perceived latency."


def stream_with_raw_events(prompt: str) -> None:
    with client.messages.stream(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            if (
                event.type == "content_block_delta"
                and event.delta.type == "text_delta"
            ):
                print(event.delta.text, end="", flush=True)
            elif event.type == "message_delta":
                print(f"\n[stop_reason: {event.delta.stop_reason}]")
            elif event.type == "message_stop":
                print("[message_stop]")


if __name__ == "__main__":
    stream_with_raw_events(PROMPT)
