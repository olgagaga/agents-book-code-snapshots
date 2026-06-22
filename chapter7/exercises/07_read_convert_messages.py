"""Exercise 7 (stretch): Read `_convert_messages` and friends.

Open `nanobot/nanobot/providers/anthropic_provider.py` and read:

  - `_convert_messages` (line 121)
  - `_assistant_blocks` (line 178)
  - `_tool_result_block` (line 163)
  - `_merge_consecutive` (line 266)
  - `_has_tool_use` (line 251)
  - `_convert_user_content` / `_convert_image_block` (lines 213, 234)

This file is a written deliverable, not running code — the exercise asks
for a list of things production code does that our toy `_to_anthropic_messages`
does not, with a one-sentence explanation per item. Below are four; each
points at a line in the file and names the real-world failure it guards
against. There are more — the exercise asks for at least three.
"""

NOTES = """
1. Image-block translation (`_convert_image_block`, lines 234-249).

   The production translator converts OpenAI's `image_url` block — either a
   `data:image/...;base64,...` URI or a plain HTTPS URL — into Anthropic's
   `{"type": "image", "source": {...}}` shape. Without this step, any
   multimodal `user` message coming in through OpenAI shape would reach
   Anthropic verbatim and fail validation with "unknown content block
   type", because Anthropic does not recognise `image_url`.

2. Consecutive same-role merging (`_merge_consecutive`, lines 286-299).

   Anthropic's `/messages` endpoint requires strict role alternation:
   two `user` turns in a row, or two `assistant` turns in a row, both
   return 400. Our chapter translator emits one synthetic `user` turn per
   tool result, so two parallel tool results (the chapter's two-tool query
   is a small example) produce two consecutive user turns and a 400 from
   the API. The production version collapses them into a single user turn
   whose `content` list contains both `tool_result` blocks.

3. Trailing-assistant strip and conversation-opener injection
   (`_merge_consecutive` rules 2 and 3, lines 301-328).

   Anthropic refuses to extend an assistant turn (it does not support
   assistant-message prefill), and it refuses to start a conversation with
   an `assistant` turn. Either condition can arise organically — history
   truncation drops the leading user message; a session resumes from a
   checkpoint that ends mid-reasoning — and the toy translator would
   forward both shapes to a 400. The production code strips trailing
   assistants and, if the first surviving turn is an assistant, injects a
   synthetic `(conversation continued)` user message to keep the request
   well-formed.

4. The `_has_tool_use` guard on those recovery paths (lines 251-264, used
   at lines 313 and 326).

   The two recovery paths in rule 2 / rule 3 can either reroute an
   assistant turn into a user turn or prepend a user opener before one.
   Both are unsafe if the assistant turn carries a `tool_use` block:
   Anthropic forbids `tool_use` inside `user` turns, and prepending a
   user message before a `tool_use`-bearing assistant would orphan the
   `tool_use`/`tool_result` pair that follows it. `_has_tool_use` lets
   the recovery code recognise this case and bail out instead of turning
   one validation error into a stranger one.

A few more worth noticing on a second pass:

  - `_convert_user_content` returns the literal string `"(empty)"`
    instead of an empty content list (line 217, 232). Anthropic rejects
    empty content arrays; passing `(empty)` is a load-bearing string
    that costs one token and avoids a 400.

  - `_assistant_blocks` returns `[{"type": "text", "text": ""}]` instead
    of `[]` when no blocks were produced (line 211). Same shape, same
    reason as above, on the assistant side.

  - `_assistant_blocks` first appends any `thinking_blocks` from the
    incoming message (lines 183-189). Anthropic's extended-thinking
    feature requires the prior `thinking` content to round-trip back
    into the next request, otherwise the model loses the chain.

  - `_convert_messages` routes `role="tool"` into the *previous* user
    message's content list when one exists (lines 138-145), rather than
    always starting a new user turn. Combined with `_merge_consecutive`
    this keeps the wire payload compact and avoids the parallel-tool
    consecutive-user-turn case in (2).
"""


if __name__ == "__main__":
    print(NOTES.strip())
