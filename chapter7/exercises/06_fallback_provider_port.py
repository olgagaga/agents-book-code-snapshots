"""Exercise 6 (stretch): Port `FallbackProvider` to the new shape.

Chapter 5's `FallbackProvider.stream` yielded text chunks; Chapter 7's
abstract `Provider.call` returns a `Reply` and streams text through an
`on_text_delta` callback along the way. The port is small but has one
subtle piece: the mid-stream guard.

The guarantee Chapter 5 established was that once a provider has streamed
*any* text to the caller, a later exception in that same call must
propagate rather than silently retry against the next provider — otherwise
the user would see a partial answer, then a second, contradictory full
answer pasted after it. In Chapter 5 the guard was a boolean flag set
inside the yielding loop. Here the streaming is invisible to us (it
happens inside the inner provider's `call`), so we wrap the user's
`on_text_delta` in a closure that flips a `streamed` flag before
forwarding the delta. The outer `try/except` then reads the flag to decide
whether to re-raise or move on.

To verify: run the agent with a bogus `ANTHROPIC_API_KEY` set. Anthropic
auth fails before any byte is streamed, so `FallbackProvider` moves on to
`OpenAIProvider` silently and the user sees a normal answer:

    ANTHROPIC_API_KEY=bogus uv run main.py

Below is the ported provider plus deterministic tests using scripted fake
providers — no API calls required.
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    run: Callable[[dict], str]


class FallbackProvider:
    """Try each provider in order; fall back on exceptions, unless a previous
    provider has already begun streaming text — in which case re-raise so the
    user does not see two answers stitched together."""

    def __init__(self, providers: list):
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self.providers = providers

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
        last_error: Exception | None = None
        for provider in self.providers:
            streamed = False

            def wrapped(text: str) -> None:
                nonlocal streamed
                streamed = True
                if on_text_delta is not None:
                    on_text_delta(text)

            try:
                return provider.call(
                    messages, system=system, tools=tools, on_text_delta=wrapped,
                )
            except Exception as exc:
                if streamed:
                    raise
                last_error = exc
                continue
        raise RuntimeError(f"All providers failed; last error: {last_error!r}")


# -- Scripted fakes for deterministic tests --------------------------------

class _FailsImmediately:
    """Raises before producing any output — auth failure, network error, etc."""

    def __init__(self, exc: Exception):
        self.exc = exc

    def call(self, messages, system="", tools=(), on_text_delta=None):
        raise self.exc


class _FailsMidStream:
    """Streams some text, then raises."""

    def __init__(self, prefix: str, exc: Exception):
        self.prefix = prefix
        self.exc = exc

    def call(self, messages, system="", tools=(), on_text_delta=None):
        if on_text_delta:
            on_text_delta(self.prefix)
        raise self.exc


class _Succeeds:
    """Streams its text through the callback and returns a matching Reply."""

    def __init__(self, text: str):
        self.text = text

    def call(self, messages, system="", tools=(), on_text_delta=None):
        for ch in self.text:
            if on_text_delta:
                on_text_delta(ch)
        return Reply(text=self.text)


if __name__ == "__main__":
    seen: list[str] = []

    fp = FallbackProvider([_FailsImmediately(RuntimeError("auth")), _Succeeds("hello")])
    reply = fp.call([], on_text_delta=seen.append)
    assert reply.text == "hello"
    assert "".join(seen) == "hello"
    print("fallback after pre-stream failure:", reply.text)

    seen.clear()
    fp = FallbackProvider([_FailsMidStream("Once upon ", RuntimeError("rate limit")),
                           _Succeeds("a happy ending")])
    try:
        fp.call([], on_text_delta=seen.append)
    except RuntimeError as exc:
        assert "rate limit" in str(exc), exc
        assert "".join(seen) == "Once upon "
        print("mid-stream failure re-raised; user saw:", repr("".join(seen)))
    else:
        raise AssertionError("mid-stream failure should have re-raised")

    seen.clear()
    fp = FallbackProvider([
        _FailsImmediately(RuntimeError("first failure")),
        _FailsImmediately(RuntimeError("second failure")),
    ])
    try:
        fp.call([], on_text_delta=seen.append)
    except RuntimeError as exc:
        assert "second failure" in str(exc), exc
        assert seen == []
        print("all providers failed, last error surfaced:", exc)
    else:
        raise AssertionError("all-fail should have raised")
