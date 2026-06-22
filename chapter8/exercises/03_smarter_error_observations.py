"""Exercise 3: Smarter error observations.

The chapter's error formatting is fine but coarse. Two small upgrades
shorten the retry loop for the model in the common cases:

  1. For unknown tool names, suggest the closest valid one using
     `difflib.get_close_matches`. The model frequently spells a tool
     name slightly off (`now()` instead of `now`, `read_files` instead
     of `read_file`); a "did you mean…" hint shortens the retry.
  2. For raised exceptions, include the exception class and its
     message in the same form Python prints when it crashes. The
     standard-library helper `traceback.format_exception_only` returns
     exactly that — usually one line, occasionally two for syntax
     errors. Joining it preserves the chapter's one-line shape while
     pulling in slightly richer signal.

Both changes live entirely inside the error-formatting branch of
`agent_step` in `agent/loop.py`:

    if tool is None:
        observation = format_unknown_tool_error(tc.name, known_tool_names)
    else:
        try:
            observation = tool.run(tc.args)
        except Exception as exc:
            observation = format_exception_observation(exc)

where `known_tool_names = [t.name for t in TOOLS]`.
"""

import difflib
import traceback


def format_unknown_tool_error(name: str, known: list[str]) -> str:
    suggestions = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    if suggestions:
        return f"Error: no tool named {name!r}. Did you mean {suggestions[0]!r}?"
    return f"Error: no tool named {name!r}."


def format_exception_observation(exc: BaseException) -> str:
    formatted = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return f"Error: {formatted}"


def _run_broken_tool(args: dict) -> str:
    raise RuntimeError("something went wrong inside the tool")


if __name__ == "__main__":
    known = ["now", "wordcount", "read_file"]

    out = format_unknown_tool_error("read_files", known)
    assert "Did you mean 'read_file'?" in out, out
    print("close match:   ", out)

    out = format_unknown_tool_error("now()", known)
    assert "Did you mean 'now'?" in out, out
    print("close match:   ", out)

    out = format_unknown_tool_error("definitely_not_a_tool", known)
    assert "Did you mean" not in out, out
    print("no match:      ", out)

    try:
        _run_broken_tool({})
    except Exception as exc:
        out = format_exception_observation(exc)
        assert "RuntimeError" in out and "something went wrong" in out, out
        print("raised exc:    ", out)

    try:
        d = {}
        d["missing_key"]
    except KeyError as exc:
        out = format_exception_observation(exc)
        assert "KeyError" in out and "missing_key" in out, out
        print("key error:     ", out)
