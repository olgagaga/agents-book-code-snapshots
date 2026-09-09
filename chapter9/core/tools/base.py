from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    run: Callable[[dict], str]