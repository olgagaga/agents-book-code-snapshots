"""Exercise 8 (stretch): Stream without the SDK.

Drop one layer below the Anthropic SDK and call the API yourself with
`httpx`. Iterate raw lines from the response, parse the SSE protocol by
hand (event/data prefixes, blank-line terminators), and reproduce the
text-only output of `stream.text_stream`.

This is what `client.messages.stream(...)` is doing for you on every
call. Once you have written it, "just use the SDK" becomes the
considered default rather than an act of faith.

Run with: `uv run --with httpx 08_stream_without_sdk.py`
"""

import json
import os

import dotenv
import httpx

dotenv.load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-opus-4-6"
PROMPT = "Name three rivers on three different continents. One short sentence each."


def stream_raw(prompt: str) -> None:
    payload = {
        "model": MODEL,
        "max_tokens": 256,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": API_KEY or "",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    with httpx.stream("POST", URL, json=payload, headers=headers, timeout=60.0) as r:
        r.raise_for_status()
        # SSE: lines come as `event: <name>` / `data: <json>` / blank
        # We only need `data:` for our purposes — the JSON payload carries
        # the type field too.
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload_str = line.removeprefix("data:").strip()
            event = json.loads(payload_str)
            if (
                event.get("type") == "content_block_delta"
                and event["delta"].get("type") == "text_delta"
            ):
                print(event["delta"]["text"], end="", flush=True)
            elif event.get("type") == "message_delta":
                stop = event["delta"].get("stop_reason")
                if stop:
                    print(f"\n[stop_reason: {stop}]")
        print()


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY not set")
    stream_raw(PROMPT)
