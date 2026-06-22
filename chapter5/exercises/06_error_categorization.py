"""Exercise 6 (stretch): nanobot's error categorization.

Nanobot's `openai_compat_provider.py` distinguishes rate-limit errors
from context-window-exceeded errors from generic transient errors. This
file lifts that idea into a small enum and a `_categorize` helper, then
wires the result into a `FallbackProvider`:

- `CONTEXT_WINDOW` — re-raise immediately. The next provider has the same
  context limit; falling back will fail the same way.
- `RATELIMIT` and `TRANSIENT` — fall back.
- `PERMANENT` (auth, 4xx bad request) — re-raise immediately.

The `_categorize` heuristic inspects exception type, status code, and the
error message text — context-window errors typically surface as 400s
whose message contains "context length", "maximum context", or similar.
"""

import enum
import os
from abc import ABC, abstractmethod
from typing import Iterator

import dotenv
import httpx
import openai

dotenv.load_dotenv()


class Provider(ABC):
    @abstractmethod
    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        ...


class ErrorCategory(enum.Enum):
    RATELIMIT = "ratelimit"
    CONTEXT_WINDOW = "context_window"
    TRANSIENT = "transient"
    PERMANENT = "permanent"


_CONTEXT_WINDOW_HINTS = (
    "context length",
    "context window",
    "maximum context",
    "too many tokens",
    "context_length_exceeded",
)


def _categorize(error: Exception) -> ErrorCategory:
    if isinstance(error, openai.RateLimitError):
        return ErrorCategory.RATELIMIT
    if isinstance(error, (httpx.ConnectError, httpx.ReadTimeout)):
        return ErrorCategory.TRANSIENT

    msg = str(error).lower()
    if any(hint in msg for hint in _CONTEXT_WINDOW_HINTS):
        return ErrorCategory.CONTEXT_WINDOW

    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        if status == 400 and any(hint in msg for hint in _CONTEXT_WINDOW_HINTS):
            return ErrorCategory.CONTEXT_WINDOW
        if status == 429:
            return ErrorCategory.RATELIMIT
        if status >= 500:
            return ErrorCategory.TRANSIENT
        return ErrorCategory.PERMANENT

    return ErrorCategory.TRANSIENT


class OpenAIProvider(Provider):
    def __init__(
        self,
        model: str = "gpt-5",
        max_tokens: int = 1024,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        oai_messages: list[dict] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)
        stream = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=oai_messages,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class CategorizedFallbackProvider(Provider):
    def __init__(self, providers: list[Provider]):
        if not providers:
            raise ValueError("CategorizedFallbackProvider needs at least one provider")
        self.providers = providers

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                yielded_anything = False
                for text in provider.stream(messages, system=system):
                    yielded_anything = True
                    yield text
                return
            except Exception as e:
                if yielded_anything:
                    raise
                category = _categorize(e)
                print(f"[fallback] {type(provider).__name__} -> {category.value}: {e!r}")
                if category in (ErrorCategory.PERMANENT, ErrorCategory.CONTEXT_WINDOW):
                    raise
                last_error = e
        raise RuntimeError(f"All providers failed; last error: {last_error!r}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in .env")

    # Smoke test the categorizer on synthetic errors.
    cases: list[tuple[str, Exception, ErrorCategory]] = [
        (
            "rate limit",
            openai.RateLimitError(
                "rate limit", response=httpx.Response(429), body=None
            ),
            ErrorCategory.RATELIMIT,
        ),
        (
            "connect error",
            httpx.ConnectError("could not connect"),
            ErrorCategory.TRANSIENT,
        ),
        (
            "context window message",
            ValueError("This model's maximum context length is 200000 tokens"),
            ErrorCategory.CONTEXT_WINDOW,
        ),
    ]
    for label, err, expected in cases:
        actual = _categorize(err)
        ok = "OK" if actual == expected else "FAIL"
        print(f"{ok}  {label}: expected {expected.value}, got {actual.value}")

    # End-to-end smoke test against the real OpenAI provider.
    provider = CategorizedFallbackProvider([OpenAIProvider(model="gpt-5")])
    msgs = [{"role": "user", "content": "Say hi in five words."}]
    for text in provider.stream(msgs, system="Be concise."):
        print(text, end="", flush=True)
    print()
