"""Exercise 4: Friendly provider errors.

The chapter's `chat()` catch prints `[error: ConnectionError: Connection
reset by peer (54)]` — the class name and the SDK's raw message. That
is useful for a developer staring at the terminal; it is hostile to a
user who only wants to know whether to retry.

The fix is a small mapping from the most common SDK exception classes
to a single sentence the user can act on. Three classes cover the bulk
of what an interactive REPL hits:

  - `RateLimitError`         -> "rate limited; try again in a moment"
  - `APIConnectionError`     -> "network issue; check your connection"
  - `AuthenticationError`    -> "the provider rejected your API key"

Anything else falls back to the chapter's existing `[error: Class:
message]`, so unfamiliar failures still surface fully — we don't want
to swallow a real bug behind a friendly sentence.

The lookup is by *class name string*, not the class itself, because
each SDK ships its own version of these classes — `anthropic.RateLimitError`
and `openai.RateLimitError` are distinct types but the user message is
the same. Matching on `type(exc).__name__` keeps the helper provider-
agnostic and avoids forcing `chat()` to import both SDKs just for the
isinstance check.

To apply the patch to the real agent: replace the `print(f"[error:
...]")` line inside `chat()`'s `except Exception` block in
`agent/main.py` with `print(friendly_error_message(exc))`.
"""

_FRIENDLY_MESSAGES = {
    "RateLimitError": "rate limited; try again in a moment",
    "APIConnectionError": "network issue; check your connection",
    "AuthenticationError": "the provider rejected your API key",
}


def friendly_error_message(exc: BaseException) -> str:
    name = type(exc).__name__
    friendly = _FRIENDLY_MESSAGES.get(name)
    if friendly is not None:
        return f"[error: {friendly}]"
    return f"[error: {name}: {exc}]"


# Stand-in exception classes that mirror the ones the SDKs raise. Matching
# is by class name string, so these work without importing anthropic/openai.
class RateLimitError(Exception): pass
class APIConnectionError(Exception): pass
class AuthenticationError(Exception): pass


if __name__ == "__main__":
    out = friendly_error_message(RateLimitError("429 too many requests"))
    assert out == "[error: rate limited; try again in a moment]", out
    print(out)

    out = friendly_error_message(APIConnectionError("ECONNRESET"))
    assert out == "[error: network issue; check your connection]", out
    print(out)

    out = friendly_error_message(AuthenticationError("invalid_api_key"))
    assert out == "[error: the provider rejected your API key]", out
    print(out)

    # Unknown exception falls back to the chapter's raw form.
    out = friendly_error_message(RuntimeError("All providers failed; last error: ..."))
    assert out == "[error: RuntimeError: All providers failed; last error: ...]", out
    print(out)

    # A KeyError's str() is the repr of the key — make sure the fallback
    # doesn't break on it.
    try:
        {}["missing"]
    except KeyError as exc:
        out = friendly_error_message(exc)
        assert out.startswith("[error: KeyError:"), out
        print(out)
