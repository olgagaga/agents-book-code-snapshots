from abc import ABC, abstractmethod
from typing import Iterator


class Provider(ABC):
    """A streaming LLM backend."""

    @abstractmethod
    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        """Stream the model's reply, yielding text deltas as they arrive."""
        ...