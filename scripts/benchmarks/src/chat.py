from dataclasses import dataclass
from enum import StrEnum


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
