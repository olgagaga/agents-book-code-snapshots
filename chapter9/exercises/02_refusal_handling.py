"""Exercise 2: Refusal handling.

The chapter treats `refusal` like every other non-`end_turn` stop reason:
it appends the partial text plus a `[model declined to answer this
request]` marker to `messages`, and returns the same string. That works
for the user but is sub-optimal for the *next turn's prompt* — the model
sees its own refusal in its own conversation history and, in practice,
tends to "remember" it and re-refuse a rephrased version of the same
question.

The fix is to split what the user sees from what the model sees:

  - The user still sees the refusal note in their terminal.
  - `messages` carries only the model's own refusal text (or, if the
    text is empty, a neutral single-line entry like `"[no answer]"` —
    we still need *something* in the assistant slot for the turn
    to be well-formed).

That way the model's view of the conversation is just "user asked X,
assistant replied Y" — Y being whatever it actually said — with no
meta-commentary about a refusal that would bias the next response.

To apply the patch to the real agent, replace the existing
non-`tool_use` branch in `agent/loop.py`'s `agent_step` with a
three-case dispatch: `end_turn` (current behavior), `refusal` (this
exercise), and a default for everything else (current behavior).
"""

from dataclasses import dataclass, field


REFUSAL_NOTE_FOR_USER = "[model declined to answer this request]"
EMPTY_REFUSAL_PLACEHOLDER = "[no answer]"


@dataclass
class FakeReply:
    text: str = ""
    stop_reason: str = "end_turn"
    tool_calls: list = field(default_factory=list)


def handle_terminal_reply(
    reply: FakeReply,
    messages: list[dict],
    on_user_visible: callable,
) -> str:
    """Apply the chapter's terminal-stop logic, with refusal split out."""
    if reply.stop_reason == "refusal":
        history_text = reply.text or EMPTY_REFUSAL_PLACEHOLDER
        messages.append({"role": "assistant", "content": history_text})
        user_text = f"{reply.text}\n\n{REFUSAL_NOTE_FOR_USER}" if reply.text else REFUSAL_NOTE_FOR_USER
        on_user_visible(user_text)
        return user_text

    final_text = reply.text
    if reply.stop_reason == "max_tokens":
        note = "[response truncated by the model's output limit]"
        final_text = f"{final_text}\n\n{note}" if final_text else note
    elif reply.stop_reason != "end_turn":
        final_text = f"{final_text}\n\n[stopped: {reply.stop_reason}]"
    messages.append({"role": "assistant", "content": final_text})
    on_user_visible(final_text)
    return final_text


if __name__ == "__main__":
    # Case 1: refusal with model text — user sees note, history hides note.
    messages: list[dict] = [{"role": "user", "content": "do something disallowed"}]
    user_screen: list[str] = []
    out = handle_terminal_reply(
        FakeReply(text="I can't help with that.", stop_reason="refusal"),
        messages,
        on_user_visible=user_screen.append,
    )
    assert REFUSAL_NOTE_FOR_USER in out
    assert REFUSAL_NOTE_FOR_USER in user_screen[-1]
    assert messages[-1] == {"role": "assistant", "content": "I can't help with that."}
    print(f"refusal+text  user-sees:    {user_screen[-1]!r}")
    print(f"              history-has:  {messages[-1]['content']!r}")
    print()

    # Case 2: refusal with empty model text — placeholder lands in history.
    messages = [{"role": "user", "content": "do something disallowed"}]
    user_screen = []
    handle_terminal_reply(
        FakeReply(text="", stop_reason="refusal"),
        messages,
        on_user_visible=user_screen.append,
    )
    assert user_screen[-1] == REFUSAL_NOTE_FOR_USER
    assert messages[-1]["content"] == EMPTY_REFUSAL_PLACEHOLDER
    print(f"refusal+empty user-sees:    {user_screen[-1]!r}")
    print(f"              history-has:  {messages[-1]['content']!r}")
    print()

    # Case 3: clean end_turn unchanged — text appears in both places.
    messages = [{"role": "user", "content": "what's 2+2?"}]
    user_screen = []
    handle_terminal_reply(
        FakeReply(text="4", stop_reason="end_turn"),
        messages,
        on_user_visible=user_screen.append,
    )
    assert user_screen[-1] == "4"
    assert messages[-1]["content"] == "4"
    print(f"end_turn      user-sees:    {user_screen[-1]!r}")
    print(f"              history-has:  {messages[-1]['content']!r}")
    print()

    # Case 4: max_tokens unchanged — note appears in both places.
    messages = [{"role": "user", "content": "long answer please"}]
    user_screen = []
    handle_terminal_reply(
        FakeReply(text="partial...", stop_reason="max_tokens"),
        messages,
        on_user_visible=user_screen.append,
    )
    assert "truncated" in user_screen[-1]
    assert "truncated" in messages[-1]["content"]
    print(f"max_tokens    user-sees:    {user_screen[-1]!r}")
    print(f"              history-has:  {messages[-1]['content']!r}")
