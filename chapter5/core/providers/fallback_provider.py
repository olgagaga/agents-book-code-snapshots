from typing import Iterator

from providers.base import Provider


class FallbackProvider(Provider):
    def __init__(self, providers: list[Provider]):
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
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
                last_error = e
                continue
        raise RuntimeError(
            f"All providers failed; last error: {last_error!r}"
        )
