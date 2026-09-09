"""Exercise 1: Log the stop reason.

The chapter's `agent_step` has five distinct exit points — a clean
`end_turn`, a `max_tokens` cut-off, a `refusal`, the iteration cap, the
KeyboardInterrupt handler — plus a sixth that lives one level up, in
`chat()`'s catch for provider/SDK failures. Each one is meaningful on its
own; together they are a surprisingly good summary of how a session
spent its turns.

The exercise asks you to log each one as the loop exits. The
implementation choice is small but worth getting right: rather than
sprinkling `print(...)` statements at every exit, factor a single
`log_exit_reason` helper and call it from exactly the places that exit
the loop. That keeps the vocabulary in one spot and makes it trivial to
swap `print` for a real logger later.

To apply the patch to the real agent:

  - Add the constants and the `log_exit_reason` helper at the top of
    `agent/loop.py`.
  - At each terminal exit in `agent_step` — the fall-through after the
    `tool_use` branch, the post-loop `MAX_ITERATIONS_NOTE` return, and
    the `KeyboardInterrupt` handler — call `log_exit_reason(...)` with
    the constant for that exit.
  - In `agent/main.py`'s `except Exception` block, call
    `log_exit_reason(EXIT_PROVIDER_ERROR)`.

This file rebuilds the dispatch in a self-contained form so you can run
a fake five-turn session without an API key and see the mix of reasons.
"""

from dataclasses import dataclass, field


EXIT_END_TURN = "end_turn"
EXIT_MAX_TOKENS = "max_tokens"
EXIT_REFUSAL = "refusal"
EXIT_TOOL_USE_CAPPED = "tool_use_capped"
EXIT_INTERRUPTED = "interrupted"
EXIT_PROVIDER_ERROR = "provider_error"


EXIT_LOG: list[str] = []


def log_exit_reason(reason: str) -> None:
    EXIT_LOG.append(reason)
    print(f"[exit: {reason}]")


@dataclass
class FakeReply:
    text: str = ""
    stop_reason: str = "end_turn"
    tool_calls: list = field(default_factory=list)


@dataclass
class FakeProvider:
    """Replays a scripted sequence of FakeReply objects, one per `call`."""
    script: list[FakeReply]
    _i: int = 0

    def call(self, *_, **__) -> FakeReply:
        reply = self.script[self._i]
        self._i += 1
        return reply


def agent_step(provider: FakeProvider, max_iterations: int = 10) -> str:
    try:
        for _ in range(max_iterations):
            reply = provider.call()
            if reply.stop_reason == "tool_use":
                continue
            reason = reply.stop_reason if reply.stop_reason in {
                EXIT_END_TURN, EXIT_MAX_TOKENS, EXIT_REFUSAL,
            } else reply.stop_reason
            log_exit_reason(reason)
            return reply.text
        log_exit_reason(EXIT_TOOL_USE_CAPPED)
        return "[capped]"
    except KeyboardInterrupt:
        log_exit_reason(EXIT_INTERRUPTED)
        return "[interrupted]"


def chat_turn(provider: FakeProvider) -> str:
    try:
        return agent_step(provider)
    except Exception:
        log_exit_reason(EXIT_PROVIDER_ERROR)
        return "[provider error]"


class _BoomProvider:
    def call(self, *_, **__):
        raise RuntimeError("all providers failed")


class _InterruptProvider:
    def call(self, *_, **__):
        raise KeyboardInterrupt


if __name__ == "__main__":
    sessions = [
        ("turn 1 (clean)",
         FakeProvider([FakeReply(text="hi", stop_reason="end_turn")])),
        ("turn 2 (tool then end)",
         FakeProvider([
             FakeReply(stop_reason="tool_use", tool_calls=[object()]),
             FakeReply(text="done", stop_reason="end_turn"),
         ])),
        ("turn 3 (max_tokens cut-off)",
         FakeProvider([FakeReply(text="partial", stop_reason="max_tokens")])),
        ("turn 4 (cap)",
         FakeProvider([FakeReply(stop_reason="tool_use", tool_calls=[object()])] * 10)),
        ("turn 5 (interrupted)",
         _InterruptProvider()),
        ("turn 6 (provider blew up)",
         _BoomProvider()),
    ]

    for label, provider in sessions:
        print(f"-- {label} --")
        chat_turn(provider)

    print()
    print("exit-reason mix across the session:")
    from collections import Counter
    for reason, count in Counter(EXIT_LOG).most_common():
        print(f"  {reason:<20} {count}")

    expected = [
        EXIT_END_TURN, EXIT_END_TURN, EXIT_MAX_TOKENS,
        EXIT_TOOL_USE_CAPPED, EXIT_INTERRUPTED, EXIT_PROVIDER_ERROR,
    ]
    assert EXIT_LOG == expected, EXIT_LOG
