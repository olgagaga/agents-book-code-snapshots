"""Exercise 5 (stretch): Orphan tool results.

`nanobot/nanobot/agent/runner.py` (line 912) defines
`_drop_orphan_tool_results`, which removes any `tool_result` content block
whose matching `tool_use` is no longer in the message list. Anthropic's
API rejects requests that contain a `tool_result` without a paired
`tool_use`, so this cleanup is mandatory in production.

How does an orphan get created in the first place? The two common ways:

1. **History trimming.** A previous `tool_use` is dropped from the message
   list — for example, `_snip_history` cuts the front of the conversation
   to fit the context window. Any `tool_result` further down that
   referenced the now-gone `tool_use` is suddenly an orphan. The fix is
   to either drop the orphan too or reinsert a placeholder `tool_use`.

2. **Selective continuation.** A user resumes a session but rewinds to
   before some assistant turn, replaying with a different message. The
   `tool_use` from the discarded branch is gone, but the `tool_result`
   that used to follow it might still live in some persisted state.

Two-sentence summary: an orphan tool result is a `tool_result` block whose
paired `tool_use` is no longer in the message list, usually because some
upstream message-list edit dropped the `tool_use` while leaving the
`tool_result` behind. It must be removed before the next model call,
because providers (Anthropic in particular) reject the message list as
malformed otherwise.

Below is a small hand-crafted example showing the orphan and a minimal
version of the cleanup, so you can see the shape concretely.
"""


# Imagine the conversation looked like this after some history trimming
# removed the original assistant turn that issued the tool_use with id="abc":
ORPHANED_MESSAGES: list[dict] = [
    {"role": "user", "content": "What is the time?"},
    # The assistant message that issued tool_use id="abc" used to live
    # here, but `_snip_history` dropped it as too old.
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "abc",
                "content": "2026-05-08T13:30:45+00:00",
            }
        ],
    },
    {"role": "user", "content": "And the weather?"},
]


def drop_orphan_tool_results(messages: list[dict]) -> list[dict]:
    """Minimal version of nanobot's `_drop_orphan_tool_results`."""
    seen_tool_use_ids: set[str] = set()
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_use":
                    seen_tool_use_ids.add(block.get("id", ""))

    cleaned: list[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            kept = [
                b
                for b in content
                if b.get("type") != "tool_result"
                or b.get("tool_use_id", "") in seen_tool_use_ids
            ]
            if not kept:
                continue
            cleaned.append({**m, "content": kept})
        else:
            cleaned.append(m)
    return cleaned


if __name__ == "__main__":
    cleaned = drop_orphan_tool_results(ORPHANED_MESSAGES)

    # The orphan tool_result message is gone; the two plain user turns remain.
    assert len(cleaned) == 2
    assert all(
        not (
            isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        )
        for m in cleaned
    )

    print(f"Original messages: {len(ORPHANED_MESSAGES)}")
    print(f"After cleanup:     {len(cleaned)}")
    print("Orphan dropped successfully.")
