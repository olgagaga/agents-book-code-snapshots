from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from tools.base import Tool

@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class Provider(ABC):
    """A streaming LLM backend with tool call support."""

    @abstractmethod
    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
        """Run one model turn and return the final reply.

        Streams text deltas through ``on_text_delta`` as they arrive.
        ``Reply.text`` carries the same text concatenated.
        """
        ...
