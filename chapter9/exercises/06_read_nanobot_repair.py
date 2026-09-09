"""Exercise 6 (stretch): Read the production repair pass.

Open `nanobot/nanobot/agent/runner.py` and read:

  - `_BACKFILL_CONTENT` (line 50)
  - `_drop_orphan_tool_results` (line 912)
  - `_backfill_missing_tool_results` (line 938)
  - The call sites around line 252 in `AgentRunner.run`

This file is a written deliverable, not running code. The exercise
asks you to compare nanobot's `_BACKFILL_CONTENT` to our
`INTERRUPT_NOTE` and pick one wording to adopt in your own agent.
The model paragraph below picks nanobot's wording for a reason that
becomes clear once you look at *where* the repair pass runs and who
the audience is.
"""

NOTES = """
The two strings:

  ours    = "[interrupted]"
  nanobot = "[Tool result unavailable — call was interrupted or lost]"

The difference is not stylistic. They tell the model something
different about what happened, and the difference matters in different
deployment shapes.

Where each runs determines the audience:

  - Our `_backfill_orphan_tool_results` runs only inside the
    `except KeyboardInterrupt:` block in `agent_step`. The cause is
    always the same: the user pressed Ctrl-C. The placeholder is
    accurate, terse, and a single word the model can act on.

  - Nanobot's `_backfill_missing_tool_results` runs on *every*
    iteration of the loop, before every model call. The orphan can
    come from many sources: a crashed subagent, a partially saved
    checkpoint that was reloaded, a context-governance trim that
    snipped one half of a pair, a network failure mid-tool, or yes,
    an interrupt. "Interrupted" would be wrong in most of those
    cases; "unavailable — call was interrupted or lost" admits the
    full set of possibilities and tells the model not to over-read
    the gap.

What the model does with each:

  - Reading `[interrupted]`, a well-behaved model will treat the
    call as cancelled by the user and usually wait for the user's
    next instruction rather than retry. That's the right behavior
    for an interactive REPL.

  - Reading `[Tool result unavailable — call was interrupted or lost]`,
    a well-behaved model treats the call as failed-and-unknown and
    is more likely to retry, ask for clarification, or pick a
    different approach. That's the right behavior for a long-running
    autonomous agent where the model is the one driving recovery.

The wording also reflects the breadth of the invariant guarantee.
Our toy only protects against the one source of orphans we control
(the interrupt path); nanobot's runs as a defence in depth before
every model request, so the wording has to accommodate orphans whose
provenance we genuinely do not know.

PICK ONE: nanobot's wording, if you plan to call the repair pass on
every iteration like nanobot does. If you keep the repair pass scoped
to the interrupt handler — which is the right call for a toy or a
single-user chat client — keep `[interrupted]` because it tells the
model exactly what happened. The wording should match the scope of
the guarantee, not the other way around.

A small follow-up worth doing: rename the helper. Our
`_backfill_orphan_tool_results` is misleadingly named once you put
the wording-broader-than-the-cause question; if the function only
runs on Ctrl-C, the helper should probably be called
`_backfill_interrupted_tool_results`, and the placeholder string
narrowed to match. Keep helper name and placeholder wording in sync;
either both should describe a specific cause (interrupt) or both
should describe the generic effect (the call's result is missing).
"""


if __name__ == "__main__":
    print(NOTES.strip())
