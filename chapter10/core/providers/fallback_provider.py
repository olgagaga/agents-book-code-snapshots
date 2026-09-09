from typing import Callable

from tools.base import Tool
from providers.base import Provider, Reply


class FallbackProvider(Provider):
    def __init__(self, providers: list[Provider]):
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self.providers = providers

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
        last_error: Exception | None = None
        for provider in self.providers:
            streamed = False

            def wrapped_on_text_delta(text: str) -> None:
                nonlocal streamed
                streamed = True
                if on_text_delta is not None:
                    on_text_delta(text)

            try:
                return provider.call(
                    messages,
                    system=system,
                    tools=tools,
                    on_text_delta=wrapped_on_text_delta,
                )
            except Exception as e:
                if streamed:
                    raise
                last_error = e
                continue
        raise RuntimeError(
            f"All providers failed; last error: {last_error!r}"
        )
