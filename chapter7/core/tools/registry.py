from tools.tools import _now, _wordcount
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
]


def find_tool(name: str) -> Tool | None:
    return next((t for t in TOOLS if t.name == name), None)
