from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ChatRole(StrEnum):
    Assistant = "assistant"
    Developer = "developer"
    System = "system"
    Tool = "tool"
    User = "user"


@dataclass
class ChatMessage:
    message: str
    role: ChatRole


def get_model_path(path: str | Path) -> str:
    if isinstance(path, Path):
        return str(path.expanduser())
    else:
        return path
