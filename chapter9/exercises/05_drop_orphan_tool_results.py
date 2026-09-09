"""Exercise 5 (stretch): Drop orphan tool calls instead of backfilling.

The chapter's KeyboardInterrupt handler calls
`_backfill_orphan_tool_results`, which appends a synthetic `tool`
message for every dangling assistant `tool_call` so the next provider
request sees the `(assistant tool_calls, tool result, ...)` invariant
satisfied. The alternative is to *drop* the unfinished bits: remove
the trailing assistant message's `tool_calls` field (and any partial
`tool` results that follow it) so the invariant is satisfied by
absence rather than by synthesis.

Both choices leave the history valid for the next turn. The trade-off:

  - Backfill (chapter): preserves the audit trail of what the model
    was about to do. The transcript shows "I tried to call X" with an
    `[interrupted]` placeholder beside it. The model on the next turn
    sees that it issued a call and got an interrupted result; it can
    decide to retry, ask the user, or move on.
  - Drop (this exercise): shorter, cleaner history with no synthetic
    rows. The model has no idea it ever issued the call. Suitable
    for cases where the user just wants to start fresh and would
    rather not see the half-finished attempt at all.

A reasonable rule of thumb: backfill when interruption is an
interactive nicety (Ctrl-C in a chat); drop when it's a recovery from
an unexpected failure (a checkpoint reload, a crashed subagent) where
the half-finished state would only confuse the model.

To apply the patch to the real agent: replace
`_backfill_orphan_tool_results` in `agent/loop.py` with the
`_drop_orphan_tool_results` below and call it instead from the
`except KeyboardInterrupt` block.
"""


def _drop_orphan_tool_results(messages: list[dict]) -> None:
    """Drop the trailing assistant tool_calls and any partial tool results.

    Walks the history backward to the most recent assistant message that
    carried tool_calls. If every expected `tool_call_id` has a matching
    `tool` result after it, leaves the history alone. Otherwise, drops
    the trailing tool messages that *do* belong to this assistant
    message (the partial results from the interrupted run) and strips
    `tool_calls` from the assistant message itself.

    The assistant message itself is preserved — only its `tool_calls`
    field is removed — because the model's text reply (often empty for
    a pure tool-call turn, but sometimes a sentence of preamble) is
    still a real thing the model said.
    """
    last_assistant_with_tools = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            last_assistant_with_tools = i
            break
    if last_assistant_with_tools is None:
        return

    expected_ids = {tc["id"] for tc in messages[last_assistant_with_tools]["tool_calls"]}
    satisfied_ids = {
        m["tool_call_id"]
        for m in messages[last_assistant_with_tools + 1:]
        if m.get("role") == "tool"
    }
    if expected_ids <= satisfied_ids:
        return

    # Drop the partial tool results that belong to this assistant message.
    tail_kept: list[dict] = []
    for m in messages[last_assistant_with_tools + 1:]:
        if m.get("role") == "tool" and m.get("tool_call_id") in expected_ids:
            continue
        tail_kept.append(m)

    messages[last_assistant_with_tools + 1:] = tail_kept
    # Strip tool_calls off the assistant message; keep its text content.
    assistant = messages[last_assistant_with_tools]
    if "tool_calls" in assistant:
        del assistant["tool_calls"]


if __name__ == "__main__":
    # Case 1: assistant issued two tool calls, neither ran.
    messages = [
        {"role": "user", "content": "weather and time?"},
        {"role": "assistant", "content": "Looking it up.", "tool_calls": [
            {"id": "tc_1", "name": "get_weather", "args": {}},
            {"id": "tc_2", "name": "get_time", "args": {}},
        ]},
    ]
    _drop_orphan_tool_results(messages)
    assert messages[-1] == {"role": "assistant", "content": "Looking it up."}, messages[-1]
    assert "tool_calls" not in messages[-1]
    print("case 1 (no results yet):  trimmed to", len(messages), "messages, assistant text kept")

    # Case 2: one of two tool calls completed before interrupt.
    messages = [
        {"role": "user", "content": "weather and time?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc_1", "name": "get_weather", "args": {}},
            {"id": "tc_2", "name": "get_time", "args": {}},
        ]},
        {"role": "tool", "tool_call_id": "tc_1", "name": "get_weather", "content": "75 and sunny"},
    ]
    _drop_orphan_tool_results(messages)
    assert "tool_calls" not in messages[1]
    assert not any(m.get("role") == "tool" for m in messages)
    print("case 2 (one result done): trimmed to", len(messages), "messages, partial tc_1 dropped")

    # Case 3: all tool calls satisfied — no-op.
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "tc_1", "name": "get_weather", "args": {}}]},
        {"role": "tool", "tool_call_id": "tc_1", "name": "get_weather", "content": "75 and sunny"},
    ]
    before = [dict(m) for m in messages]
    _drop_orphan_tool_results(messages)
    assert messages == before, "should be a no-op when nothing is orphaned"
    print("case 3 (already clean):   no-op,", len(messages), "messages")

    # Case 4: no assistant-with-tools in history — no-op.
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hi"}]
    before = [dict(m) for m in messages]
    _drop_orphan_tool_results(messages)
    assert messages == before
    print("case 4 (no tool calls):   no-op,", len(messages), "messages")
