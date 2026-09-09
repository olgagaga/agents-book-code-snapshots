"""Exercise 6 (stretch): Read the injection production code.

Open `nanobot/nanobot/agent/runner.py` and read:

  - `_MAX_INJECTIONS_PER_TURN`            (line 41)
  - `_MAX_INJECTION_CYCLES`               (line 42)
  - `_append_injected_messages`           (line 121)
  - `_merge_message_content`              (line 104)
  - `_try_drain_injections`               (line 142)
  - `_drain_injections`                   (line 186)
  - The call sites inside `AgentRunner.run` around lines 355, 377, 457,
    481, 498, and 546 — six different drain points across one turn.

Our `get_injections` callback already mirrors the *shape* of the
production design (a function the loop calls to ask "anything new?").
The exercise asks you to evaluate the three extra details nanobot layers
on top, decide which to adopt now and which to defer, and write up the
reasoning. The notes below are a model paragraph; yours will be shorter
or longer depending on which way your own agent is heading.
"""

NOTES = """
Three details I would adopt — and three I would defer — given where the
book's agent is today.

ADOPT NOW:

(c) `_MAX_INJECTION_CYCLES` (=5). This is the cheapest defence with the
    highest payoff. Without it, an adversarial or unlucky injection
    source (a websocket producer that fires every time the loop drains;
    a buggy notification daemon that re-emits the same alert) keeps the
    loop alive indefinitely. We already have the same defensive instinct
    in `MAX_ITERATIONS` from Chapter 6; this is the same fuse, scoped
    to the injection side. Add a `cycles` counter to `agent_step`,
    increment it in `_heartbeat` when injections are produced, refuse
    to extend the turn once it crosses the cap. Three lines of code,
    no policy decisions to weigh. Adopt today.

DEFER UNTIL NEEDED:

(b) `_append_injected_messages` adjacent-user merging. The reason
    nanobot needs this is structural: both Anthropic and OpenAI reject
    conversations with two consecutive user messages. Our toy is safe
    today because the heartbeat fires *after* assistant turns (between
    iterations, after tool batches) — the message before an injection
    is always assistant or tool. The moment we add a second injection
    site that fires after a user message — say, a separate "system
    event" injection that the heartbeat drains right after a `cancel:`
    prefix sets the cancel flag — we will need this merging. Add it
    when the second injection site arrives, not now: the merge logic
    has subtle edge cases (string content vs. content-block list) that
    are easier to get right against a concrete second use case than
    against a hypothetical one.

(a) Signature-based `limit=` invocation. The production callback
    optionally accepts a `limit` keyword via `inspect.signature` — this
    is how nanobot lets the producer opt into back-pressure. It is a
    small piece of cleverness that pays off only when the producer has
    a real choice to make about how many items to surface. Our
    `_drain_user_lines` always drains everything; there is no choice
    to expose. Adopt this the day we wire a second injection source
    whose producer cares about back-pressure (a chat channel that can
    surface 1000 unread messages from before the agent woke up).

A NON-OBVIOUS FOURTH:

I would also adopt `_MAX_INJECTIONS_PER_TURN` (=3) sooner rather than
later, even though the exercise did not call it out. The cycle cap
protects against infinite injection bursts across turns; the per-turn
cap protects against a single producer flooding one turn with so many
"[mid-turn user note]" lines that they overwhelm whatever the user's
actual question was. With our current stdin source the risk is low —
the user types one or two lines, not fifty — but our heartbeat happily
emits one message per drained line with no cap. A four-line edit in
`_heartbeat` (cap the loop to N, log a warning when truncated) gets us
the production safety net at no design cost.

WHAT NOT TO ADOPT YET:

The six call sites inside `AgentRunner.run` are not the point. Each
exists because some specific failure mode (provider error, length
recovery, max-iterations termination) was found to leak ungranted
injections in production. They are sensible defensive additions for a
loop that runs in a long-lived service, not a guide for what every
agent should do on day one. Our `_heartbeat` fires at the two natural
moments — top of iteration, during tool wait — and that is enough until
we add a real third moment that needs a third drain point.
"""


if __name__ == "__main__":
    print(NOTES.strip())
