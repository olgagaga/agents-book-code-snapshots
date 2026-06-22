"""Exercise 3: Stricter parser with a `TOOL:` sentinel.

The chapter parser silently treats malformed JSON as a final answer —
convenient when the model improvises, dangerous when it produces a
malformed tool call that you genuinely meant to be a tool call. This
version requires a literal `TOOL:` token on its own line as a tool-call
sentinel; everything else is a final answer.

To wire this into the agent, replace `_parse_tool_call` in `loop.py` and
swap the format paragraph in `TOOL_INSTRUCTIONS` for the version below —
the model has to know to emit the sentinel.

Trade-off: false positives (treating a malformed tool call as text) versus
false negatives (treating prose with a stray `{` as a tool call). The
sentinel cuts both, at the cost of one extra line in every tool call. In
practice that is the right trade for production: the chapter's parser is
biased toward graceful degradation; this one is biased toward clarity.
"""

import json


TOOL_INSTRUCTIONS = """
You have access to tools. To call a tool, output exactly two lines:

    TOOL:
    {"tool": "<name>", "args": {<arguments>}}

The first line must be the literal token `TOOL:` and nothing else.
The second line must be JSON of the shape above.

Available tools:

- `now` — return the current UTC time. No arguments.

When you call a tool, do not write anything else (no prose, no markdown).
When you have nothing left to do, reply with a normal text answer. No `TOOL:` line.
""".strip()


def _parse_tool_call(text: str) -> dict | None:
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return None
    if lines[0].strip() != "TOOL:":
        return None
    body = "\n".join(lines[1:]).strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "tool" not in parsed:
        return None
    return parsed


if __name__ == "__main__":
    # Tool calls
    assert _parse_tool_call('TOOL:\n{"tool": "now", "args": {}}') == {
        "tool": "now",
        "args": {},
    }
    assert _parse_tool_call(
        'TOOL:\n{"tool": "wordcount", "args": {"text": "x"}}'
    ) == {"tool": "wordcount", "args": {"text": "x"}}

    # Final answers (no TOOL: sentinel)
    assert _parse_tool_call("Madrid.") is None
    # Stray `{` in prose no longer trips the parser:
    assert _parse_tool_call('The answer is {"x": 1}.') is None
    # Bare JSON without sentinel is now treated as text, not a tool call:
    assert _parse_tool_call('{"tool": "now", "args": {}}') is None

    # Malformed JSON after sentinel — treated as final answer (the model
    # tried to call a tool but botched it; better to surface the prose
    # than to crash on parse failure).
    assert _parse_tool_call("TOOL:\n{not valid json") is None

    # Extra leading whitespace on the sentinel line is tolerated:
    assert _parse_tool_call('  TOOL:  \n{"tool": "now", "args": {}}') == {
        "tool": "now",
        "args": {},
    }

    print("All strict-parser unit tests passed.")
