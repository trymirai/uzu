from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass(frozen=True)
class File:
    name: str
    url: str

    @classmethod
    def from_dict(cls, data: dict) -> "File":
        return cls(name=data["name"], url=data["url"])


@dataclass(frozen=True)
class Model:
    id: str
    toolchain_version: str
    repod_id: str
    name: str
    files: List[File]

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        return cls(
            id=data["id"],
            toolchain_version=data["toolchainVersion"],
            repod_id=data["repoId"],
            name=data["name"],
            files=[File.from_dict(file) for file in data["files"]],
        )


@dataclass(frozen=True)
class Registry:
    models: List[Model]

    @classmethod
    def from_dict(cls, data: dict) -> "Registry":
        return cls(models=[Model.from_dict(model) for model in data["models"]])


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class BenchmarkTask:
    identifier: str
    repo_id: str
    number_of_runs: int
    tokens_limit: int
    messages: List[Message]
    greedy: bool
