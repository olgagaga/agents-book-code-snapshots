"""Exercise 2: Token-aware truncation.

The chapter's `_truncate_observation` counts characters. That is a fine
first approximation, but the model's real cost is measured in *tokens*,
and a token spans a variable number of characters depending on content.
English prose averages roughly four characters per token; a payload of
JSON, base64, or hex packs many fewer. A character budget therefore
either over- or under-truncates depending on what the tool returned.

Each major SDK exposes a counter:

  - OpenAI: `tiktoken` (offline, deterministic per encoding).
  - Anthropic: the API's `messages.count_tokens` endpoint
    (`client.messages.count_tokens(...).input_tokens`). It is *online*,
    which makes it unsuitable for inner-loop work; in practice you cache
    the result per (message, model) or fall back to a local tokenizer
    such as tiktoken's `cl100k_base` as a close-enough approximation
    for Claude-class models.

The rewrite below takes a `max_tokens` budget and an encoder. It does
*token-level* slicing: encode the whole text, take a `head_n`/`tail_n`
window of tokens, decode each window back to a string, and glue them
with the same marker the character version used. The marker itself
costs a small number of tokens — we subtract 30 from the available
budget to leave room for it, mirroring the character version's
`max_chars - head - tail - 60` arithmetic.

The demo uses tiktoken if available; otherwise it falls back to a crude
4-chars-per-token approximation so the file still runs. Either way you
can see the difference between character-based and token-based cuts on
the same input.
"""

from __future__ import annotations


def _truncate_observation_chars(text: str, max_chars: int) -> str:
    """The chapter's character-based version, for comparison."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 60
    omitted = len(text) - head - tail
    return (
        f"{text[:head]}\n\n"
        f"[...truncated: {omitted} characters omitted from the middle...]\n\n"
        f"{text[-tail:]}"
    )


def _truncate_observation_tokens(text: str, max_tokens: int, encoder) -> str:
    """Token-aware middle-snip using any encoder with `.encode` and `.decode`."""
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    head_n = max_tokens // 2
    tail_n = max_tokens - head_n - 30  # leave room for the marker itself
    omitted = len(tokens) - head_n - tail_n
    head = encoder.decode(tokens[:head_n])
    tail = encoder.decode(tokens[-tail_n:])
    return (
        f"{head}\n\n"
        f"[...truncated: {omitted} tokens omitted from the middle...]\n\n"
        f"{tail}"
    )


class _CharsPerTokenEncoder:
    """Fallback encoder when tiktoken is not installed.

    Treats every 4-char span as a "token" and slices the string directly.
    Wildly inaccurate for symbolic content; good enough for the demo to
    show that token-aware cuts make different decisions than char-aware
    ones.
    """

    CHARS_PER_TOKEN = 4

    def encode(self, text: str) -> list[int]:
        n = self.CHARS_PER_TOKEN
        return [i for i in range(0, len(text), n)]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._slice(t) for t in tokens)

    _src: str = ""

    def bind(self, text: str) -> "_CharsPerTokenEncoder":
        self._src = text
        return self

    def _slice(self, start: int) -> str:
        return self._src[start : start + self.CHARS_PER_TOKEN]


def get_encoder(provider: str):
    """Return whichever counter matches the active provider.

    For OpenAI-compatible providers, tiktoken's `cl100k_base` is the
    standard offline tokenizer. For Anthropic, the API's
    `messages.count_tokens` endpoint is authoritative but requires a
    network round-trip — too slow for inner-loop truncation work, so in
    practice you cache its output or fall back to tiktoken as a
    close-enough offline stand-in (Claude tokenization is similar enough
    that the cuts come out almost the same).
    """
    try:
        import tiktoken
    except ImportError:
        return None
    if provider == "openai":
        return tiktoken.get_encoding("cl100k_base")
    if provider == "anthropic":
        # In production: cache `client.messages.count_tokens(...).input_tokens`
        # results per-(message, model). The offline fallback below is fine for
        # truncation decisions but not for billing-accurate counts.
        return tiktoken.get_encoding("cl100k_base")
    raise ValueError(f"unknown provider: {provider!r}")


if __name__ == "__main__":
    json_blob = (
        '{"items": ['
        + ", ".join(f'{{"id": {i}, "value": "{"x" * 60}"}}' for i in range(80))
        + "]}"
    )
    assert len(json_blob) > 5000, f"sample needs to be >5000 chars, got {len(json_blob)}"

    encoder = get_encoder("openai")
    if encoder is None:
        encoder_path = "fallback 4-chars-per-token"
        encoder = _CharsPerTokenEncoder().bind(json_blob)
    else:
        encoder_path = "tiktoken cl100k_base"

    by_chars = _truncate_observation_chars(json_blob, max_chars=1200)
    by_tokens = _truncate_observation_tokens(json_blob, max_tokens=300, encoder=encoder)

    print(f"using: {encoder_path}")
    print(f"input:      {len(json_blob):>5} chars, {len(encoder.encode(json_blob)):>4} tokens")
    print(f"by chars:   {len(by_chars):>5} chars, {len(encoder.encode(by_chars)):>4} tokens")
    print(f"by tokens:  {len(by_tokens):>5} chars, {len(encoder.encode(by_tokens)):>4} tokens")
    print()
    print(f"by-chars head:  {by_chars[:80]!r}")
    print(f"by-tokens head: {by_tokens[:80]!r}")
    print()
    print(f"by-chars tail:  {by_chars[-80:]!r}")
    print(f"by-tokens tail: {by_tokens[-80:]!r}")
