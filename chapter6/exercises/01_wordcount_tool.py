"""Exercise 1: A second tool — wordcount.

Add `wordcount` to `TOOLS` and update `TOOL_INSTRUCTIONS` so the model
knows about it. The whole abstraction's payoff is that this requires *no*
changes to `agent_step`: the registry handles dispatch, the parser handles
the shape, and the model picks the right tool entirely from the prompt.

The point of the exercise is to convince yourself that adding a tool is
one new entry in `TOOLS` plus one new bullet in `TOOL_INSTRUCTIONS`. Wire
both into `loop.py` and run the agent (`uv run main.py`):

    you: how many words are in 'the quick brown fox jumps over the lazy dog'?
    you: what is the time?

The first should pick `wordcount`; the second should pick `now`. Both
prompts succeed without `agent_step` inspecting the tool name.
"""


def _wordcount(args: dict) -> str:
    text = args.get("text", "")
    return str(len(text.split()))


TOOLS = {
    "wordcount": _wordcount,
}


TOOL_INSTRUCTIONS = """
You have access to tools. To call a tool, reply with exactly one line of JSON:

    {"tool": "<name>", "args": {<arguments>}}

Available tools:

- `now` — return the current UTC time. No arguments.
- `wordcount` — count whitespace-separated words. Args: {"text": "<the text>"}.

When you call a tool, do not write anything else. Wait for the tool result, then continue.
When you have nothing left to do, reply with a normal text answer to the user. No JSON.
""".strip()


if __name__ == "__main__":
    # Unit-test the tool itself; the integration test is at the agent level.
    assert _wordcount({"text": "the quick brown fox"}) == "4"
    assert _wordcount({"text": "one"}) == "1"
    assert _wordcount({"text": ""}) == "0"
    assert _wordcount({"text": "  spaces   between   words  "}) == "3"
    print("wordcount unit tests passed.")
