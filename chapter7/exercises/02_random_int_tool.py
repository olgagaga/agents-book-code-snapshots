"""Exercise 2: A third tool — `random_int`.

Add `random_int` to `TOOLS`: takes `{"low": int, "high": int}` and returns a
random integer in `[low, high]` inclusive. Encode the `low <= high` invariant
in the schema using JSON Schema's `minimum`/`maximum` keywords so the model
gets a hint about the relationship before it tries to call the tool.

The trick is that JSON Schema's `minimum`/`maximum` constrain a property
against a *literal* value, not against another property. There is no
portable way to write "high must be >= low" in plain JSON Schema; the best
we can do at the schema layer is bound each side with a sane literal range
(say, -1_000_000 to 1_000_000) and validate the relationship inside the
tool itself. The model reads the schema's description for the cross-field
hint, and we keep the runtime check as a backstop.

Apply this by adding the `_random_int` callable to `agent/tools/tools.py`
and the `Tool` entry to `TOOLS` in `agent/tools/registry.py`. Then run the
agent:

    uv run main.py

    you: roll three random integers between 1 and 100 and tell me their sum

A capable model will emit three `random_int` tool calls in one iteration;
our sync loop runs them back-to-back and the next iteration returns the sum.
"""

import random

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    run: Callable[[dict], str]


def _random_int(args: dict) -> str:
    low = int(args["low"])
    high = int(args["high"])
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    return str(random.randint(low, high))


RANDOM_INT_TOOL = Tool(
    name="random_int",
    description=(
        "Return a uniformly random integer in [low, high] inclusive. "
        "The caller must pass low <= high; the tool will reject the call otherwise."
    ),
    schema={
        "type": "object",
        "properties": {
            "low": {
                "type": "integer",
                "minimum": -1_000_000,
                "maximum": 1_000_000,
                "description": "Inclusive lower bound. Must be <= high.",
            },
            "high": {
                "type": "integer",
                "minimum": -1_000_000,
                "maximum": 1_000_000,
                "description": "Inclusive upper bound. Must be >= low.",
            },
        },
        "required": ["low", "high"],
    },
    run=_random_int,
)


if __name__ == "__main__":
    random.seed(0)
    for _ in range(100):
        value = int(_random_int({"low": 1, "high": 100}))
        assert 1 <= value <= 100, value
    assert _random_int({"low": 5, "high": 5}) == "5"

    try:
        _random_int({"low": 10, "high": 1})
    except ValueError as exc:
        print(f"caught expected ValueError: {exc}")
    else:
        raise AssertionError("low > high should have raised")

    print("random_int passes inclusive bounds and same-value edge case.")
    print("sample roll [1,100]:", _random_int({"low": 1, "high": 100}))
