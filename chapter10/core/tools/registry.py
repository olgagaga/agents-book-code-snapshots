from tools.tools import _now, _wordcount, _sleep_for
from tools.base import Tool


TOOLS: list[Tool] = [
    Tool(
        name="now",
        description="Return the current UTC time as an ISO 8601 string.",
        schema={"type": "object", "properties": {}, "required": []},
        run=_now,
    ),
    Tool(
        name="wordcount",
        description="Count whitespace-separated words in the given text.",
        schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        run=_wordcount,
    ),
    Tool(
        name="sleep_for",
        description="Sleep for the given number of seconds (useful for simulating long-running work).",
        schema={
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": ["seconds"],
        },
        run=_sleep_for,
    ),
]


def find_tool(name: str) -> Tool | None:
    return next((t for t in TOOLS if t.name == name), None)