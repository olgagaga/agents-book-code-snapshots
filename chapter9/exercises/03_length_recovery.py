"""Exercise 3: Length recovery.

The chapter handles `stop_reason == "max_tokens"` by appending a
`[response truncated...]` marker to the partial text and returning. That
is honest but defeatist — if the model was cut off mid-sentence, the
clean recovery is to let it keep going from where it stopped.

Nanobot's pattern: when a reply comes back with `finish_reason ==
"length"` and non-blank text, append the partial assistant text *plus*
a synthetic user message (`"Please continue from where you left off."`)
and let the loop iterate. The model picks up the thread. A counter
caps the number of consecutive recoveries so a model that keeps
tripping the limit on every turn cannot loop forever.

Two details matter:

  - The counter must be *consecutive*. A `tool_use` or a clean
    `end_turn` between two `max_tokens` replies should reset it,
    because the agent is making real progress between cut-offs.
  - The partial text goes in *before* the synthetic user message,
    in that order. The model needs to see what it already said so
    the continuation flows naturally.

To apply the patch to the real agent: in `agent/loop.py`, add a
`length_recoveries = 0` counter at the top of the `try:` block in
`agent_step`, and replace the fall-through branch with the dispatch
below.

Set `max_tokens=200` in the provider and ask for a 1,000-word essay
to see the recovery cycle in action.
"""

from dataclasses import dataclass, field

MAX_LENGTH_RECOVERIES = 3
LENGTH_RECOVERY_PROMPT = "Please continue from where you left off."


@dataclass
class FakeReply:
    text: str = ""
    stop_reason: str = "end_turn"
    tool_calls: list = field(default_factory=list)


@dataclass
class ScriptedProvider:
    script: list[FakeReply]
    _i: int = 0

    def call(self, *_, **__) -> FakeReply:
        reply = self.script[self._i]
        self._i += 1
        return reply


def agent_step(provider: ScriptedProvider, messages: list[dict], max_iterations: int = 10) -> str:
    length_recoveries = 0
    for _ in range(max_iterations):
        reply = provider.call()

        if reply.stop_reason == "max_tokens" and reply.text and length_recoveries < MAX_LENGTH_RECOVERIES:
            messages.append({"role": "assistant", "content": reply.text})
            messages.append({"role": "user", "content": LENGTH_RECOVERY_PROMPT})
            length_recoveries += 1
            continue

        # Any non-recovery exit resets the counter for the *next* turn.
        # (Inside one `agent_step` call we are about to return, so this
        # reset only matters when the helper is re-entered.)
        length_recoveries = 0

        final_text = reply.text
        if reply.stop_reason == "max_tokens":
            final_text = f"{final_text}\n\n[response truncated after {MAX_LENGTH_RECOVERIES} recovery attempts]"
        elif reply.stop_reason not in {"end_turn", "tool_use"}:
            final_text = f"{final_text}\n\n[stopped: {reply.stop_reason}]"

        messages.append({"role": "assistant", "content": final_text})
        return final_text

    return "[capped]"


if __name__ == "__main__":
    # Case 1: two truncations, then a clean finish — model recovers.
    messages: list[dict] = [{"role": "user", "content": "Write 1000 words."}]
    provider = ScriptedProvider([
        FakeReply(text="The quick brown fox jumps over the lazy dog. " * 5, stop_reason="max_tokens"),
        FakeReply(text="It continues at speed across the meadow, " * 5, stop_reason="max_tokens"),
        FakeReply(text="and eventually finds rest beneath the apple tree.", stop_reason="end_turn"),
    ])
    final = agent_step(provider, messages)
    recovery_prompts = [m for m in messages if m["content"] == LENGTH_RECOVERY_PROMPT]
    assert len(recovery_prompts) == 2, recovery_prompts
    assert final.endswith("beneath the apple tree.")
    assert "truncated after" not in final
    print(f"case 1 (recovers): {len(messages)} messages, final ends '{final[-30:]}'")

    # Case 2: model keeps tripping the limit — counter caps it.
    messages = [{"role": "user", "content": "Write 10000 words."}]
    provider = ScriptedProvider([
        FakeReply(text="round %d. " % i + "lorem " * 30, stop_reason="max_tokens")
        for i in range(MAX_LENGTH_RECOVERIES + 1)
    ])
    final = agent_step(provider, messages)
    recovery_prompts = [m for m in messages if m["content"] == LENGTH_RECOVERY_PROMPT]
    assert len(recovery_prompts) == MAX_LENGTH_RECOVERIES, recovery_prompts
    assert "truncated after 3 recovery attempts" in final
    print(f"case 2 (capped):   recoveries={len(recovery_prompts)}, final tail '{final[-40:]}'")

    # Case 3: empty partial text — no recovery, just surface the truncation.
    messages = [{"role": "user", "content": "..."}]
    provider = ScriptedProvider([
        FakeReply(text="", stop_reason="max_tokens"),
    ])
    final = agent_step(provider, messages)
    recovery_prompts = [m for m in messages if m["content"] == LENGTH_RECOVERY_PROMPT]
    assert len(recovery_prompts) == 0
    assert "truncated" in final
    print(f"case 3 (blank):    recoveries={len(recovery_prompts)}, final '{final}'")

    # Case 4: clean end_turn unchanged — no recovery, no marker.
    messages = [{"role": "user", "content": "say hi"}]
    provider = ScriptedProvider([FakeReply(text="hello", stop_reason="end_turn")])
    final = agent_step(provider, messages)
    assert final == "hello"
    print(f"case 4 (clean):    final '{final}'")
