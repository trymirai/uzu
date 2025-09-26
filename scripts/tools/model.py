from dataclasses import dataclass
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
