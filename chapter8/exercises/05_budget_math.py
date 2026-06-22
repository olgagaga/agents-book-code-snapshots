"""Exercise 5 (stretch): Run the budget math.

This script plots how token cost grows turn-by-turn before and after
the Chapter 8 continuation change. The shape of the curve tells you
when a long session is going to pressure the context window.

The analysis works on any transcript that records, per turn, the
*messages list as it stood at the end of that turn* — not just the
user prompt and final reply. The chapter's pre-Chapter-8 loop never
exposed that scratchpad to the caller, which is precisely why this
chapter rewrites `agent_step` to mutate the messages list in place.

Two cumulative-cost curves are useful to plot:

  - *Without continuation* (the Chapter 7 model): every turn pays for
    only `user_prompt + final_assistant_reply`, growing linearly.
  - *With continuation* (the Chapter 8 model): every turn pays for the
    full scratchpad of every prior turn — every intermediate assistant
    message and every tool result rides along.

For a session whose tool calls are evenly distributed, the
continuation curve is roughly *quadratic* in turn number: each turn's
prompt is a prefix of the next turn's prompt, and the sum of prefix
lengths grows like n^2/2. For a session whose tool calls cluster in a
single bursty turn, the curve is closer to linear with one large step.

This file ships synthetic transcript data so the demo runs without
real tokens or an API key. Replace `_SYNTHETIC_SESSION` with a parsed
transcript from your own runs (see Exercise 4) when you want the
plot for real.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Crude 4-chars-per-token estimate; swap for tiktoken or the
    Anthropic counter when you want precise numbers."""
    return max(1, len(text) // 4)


def message_tokens(message: dict) -> int:
    content = message.get("content") or ""
    base = estimate_tokens(content) if isinstance(content, str) else 0
    for tc in message.get("tool_calls") or []:
        base += estimate_tokens(repr(tc))
    return base + 4  # rough per-message overhead


def cost_without_continuation(turns: list[dict]) -> list[int]:
    cumulative = 0
    out: list[int] = []
    for turn in turns:
        cumulative += estimate_tokens(turn["user"]) + estimate_tokens(turn["assistant"])
        out.append(cumulative)
    return out


def cost_with_continuation(turns: list[dict]) -> list[int]:
    scratchpad: list[dict] = []
    out: list[int] = []
    for turn in turns:
        scratchpad.append({"role": "user", "content": turn["user"]})
        for m in turn["scratchpad"]:
            scratchpad.append(m)
        scratchpad.append({"role": "assistant", "content": turn["assistant"]})
        out.append(sum(message_tokens(m) for m in scratchpad))
    return out


def render_curve(label: str, values: list[int], width: int = 40) -> str:
    peak = max(values) if values else 1
    lines = [f"{label}  (peak {peak} tokens)"]
    for i, v in enumerate(values, start=1):
        bar = "#" * int(width * v / peak)
        lines.append(f"  turn {i:>2}  {v:>5}  {bar}")
    return "\n".join(lines)


_SCRATCHPAD_TOOL_TURN = [
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "name": "read_file", "args": {"path": "config.yaml"}},
    ]},
    {"role": "tool", "tool_call_id": "a", "name": "read_file",
     "content": "x" * 2000},  # a chunky 2_000-char tool result
]

_SCRATCHPAD_TRIVIAL_TURN: list[dict] = []  # no tool calls

_SYNTHETIC_SESSION = [
    {"user": "hi",                      "assistant": "hello!",                 "scratchpad": _SCRATCHPAD_TRIVIAL_TURN},
    {"user": "read config.yaml",        "assistant": "The config defines ...", "scratchpad": _SCRATCHPAD_TOOL_TURN},
    {"user": "and the database key?",   "assistant": "It is set to ...",       "scratchpad": _SCRATCHPAD_TOOL_TURN},
    {"user": "explain the auth block",  "assistant": "Auth uses OAuth2 ...",   "scratchpad": _SCRATCHPAD_TOOL_TURN},
    {"user": "any TODOs in there?",     "assistant": "Three: ...",             "scratchpad": _SCRATCHPAD_TOOL_TURN},
    {"user": "what's the timeout?",     "assistant": "30 seconds.",            "scratchpad": _SCRATCHPAD_TOOL_TURN},
    {"user": "summarize so far",        "assistant": "We've covered ...",      "scratchpad": _SCRATCHPAD_TRIVIAL_TURN},
    {"user": "now the deploy script",   "assistant": "The script runs ...",    "scratchpad": _SCRATCHPAD_TOOL_TURN},
    {"user": "is it idempotent?",       "assistant": "Mostly; ...",            "scratchpad": _SCRATCHPAD_TOOL_TURN},
    {"user": "thanks",                  "assistant": "anytime!",               "scratchpad": _SCRATCHPAD_TRIVIAL_TURN},
]


if __name__ == "__main__":
    no_cont = cost_without_continuation(_SYNTHETIC_SESSION)
    with_cont = cost_with_continuation(_SYNTHETIC_SESSION)

    print(render_curve("WITHOUT continuation (Chapter 7)", no_cont))
    print()
    print(render_curve("WITH continuation    (Chapter 8)", with_cont))
    print()

    ratio_first = with_cont[0] / max(1, no_cont[0])
    ratio_last = with_cont[-1] / max(1, no_cont[-1])
    print(f"continuation cost multiplier: turn 1 = {ratio_first:.1f}x, turn 10 = {ratio_last:.1f}x")
    print(
        "Note the bend after turn 2: every chatty tool-call turn adds ~500 tokens of\n"
        "scratchpad to *every subsequent turn's* prompt, so the curve goes super-linear\n"
        "until the trivial turns (7, 10) flatten it briefly."
    )
