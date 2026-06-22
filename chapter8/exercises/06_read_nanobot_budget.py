"""Exercise 6 (stretch): Read `_apply_tool_result_budget` and `_microcompact`.

Open `nanobot/nanobot/agent/runner.py` and read:

  - `_apply_tool_result_budget` (around line 1004)
  - `_microcompact` (around line 979)
  - `_normalize_tool_result` (around line 883)
  - `truncate_text` in `nanobot/nanobot/utils/helpers.py` (line 135)
  - `maybe_persist_tool_result` (referenced from `_normalize_tool_result`)

This file is a written deliverable, not running code — the exercise
asks you to pick *one* of nanobot's four design decisions and write a
paragraph on whether you would adopt it in your own agent. Below is a
model paragraph on each of the four, picking the disk-persistence
choice as the most interesting trade-off. The right answer for your
agent depends on what your tools do.
"""

NOTES = """
Nanobot makes four decisions that differ from this chapter's toy:

  1. Truncation strategy: tail-snip, not middle-snip.
     `truncate_text` keeps `text[:max_chars]` and appends "... (truncated)".
     The bet is that for nanobot's tool mix (`read_file`, `exec`, `grep`,
     `web_fetch`), the head of the output is more diagnostic than the
     tail — file headers, command invocations, page titles — and if the
     model needs the missing bytes it can re-issue a more specific call
     (read_file with a line range, exec with a different grep filter).

  2. Budget location: per-AgentRunSpec, not per-tool.
     `AgentRunSpec.max_tool_result_chars` (default 16_000) is a single
     budget shared by every tool the agent runs, configurable per agent
     rather than per tool. The trade-off is operational simplicity at
     the cost of the small/large asymmetry Exercise 1 fixes.

  3. Full results persisted to disk.
     Before truncating, `_normalize_tool_result` calls
     `maybe_persist_tool_result(workspace, session_key, tool_call_id,
     result, max_chars=...)`. That writer stashes the full bytes on disk
     under a stable id; the truncated string that lands in the prompt
     is purely for context-window hygiene, while the truth is preserved
     elsewhere and can be re-fetched by a separate tool.

  4. Microcompact: a second pass that replaces *older* entries.
     `_microcompact` keeps the most recent 10 results from a curated set
     of chatty tools (`read_file`, `exec`, `grep`, `glob`, `web_search`,
     `web_fetch`, `list_dir`) and rewrites older entries as one-line
     placeholders like `[read_file result omitted from context]`. The
     bet is that the model overwhelmingly reasons about its most recent
     observations, so older chatty-tool results are cheap to page out.

PICK ONE: disk persistence (decision 3).

I would adopt this for any agent whose tools can legitimately return more
content than the context window can hold — anything that touches a file
system, a database query, a web page, or a long shell output. The reason
is that the design separates two concerns the chapter's toy conflates:
the prompt's *context budget* (what the model sees on the next request)
and the session's *factual record* (what was actually observed). Once
those are separated, every other choice gets easier:

  - The truncation strategy stops carrying so much weight. Middle-snip
    versus tail-snip becomes a tactical question about prompt clarity,
    not a permanent information-loss decision, because the full result
    is one tool call away.
  - You can give the model a `fetch_full_result(tool_call_id)` tool
    that returns the persisted bytes for an earlier call. The model
    learns to reach for it when its truncated view of an earlier
    observation no longer suffices — the kind of self-directed context
    retrieval that consolidation (Chapter 16) tries to approximate
    automatically.
  - Debugging gets dramatically easier. A truncated prompt is a lossy
    record of what the agent saw; a sidecar file is not. When a
    production agent makes a confusing decision, you can replay the
    exact bytes it had access to.

The cost is operational: a workspace directory, a cleanup policy, an
extra failure mode (the disk write can fail; nanobot's `_normalize_tool_result`
catches the exception and falls back to the raw result). For a toy
agent or a single-process chat client, that is too much machinery —
keep middle-snip and a fixed budget. For anything that runs longer than
a few turns or touches non-trivial tools, the disk-backed shape is
worth the operational cost.

Other natural picks:

  - If your tools are mostly small and uniform: stick with the toy's
    single budget; nanobot's per-spec value buys little.
  - If you have one or two tools that return a lot and the rest are
    tiny: Exercise 1's per-tool budgets are the simplest fix and
    nanobot's microcompact is overkill.
  - If your sessions are long and tool-heavy: microcompact alone, on
    top of the chapter's truncation, gets you most of the way to
    nanobot's behavior without the disk dependency.
"""


if __name__ == "__main__":
    print(NOTES.strip())
