from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import cattrs


@dataclass(frozen=True)
class _Hash:
    method: str
    value: str


@dataclass(frozen=True)
class _Repository:
    identifier: str


@dataclass(frozen=True)
class _ReferenceFile:
    name: str
    url: str
    size: int
    hashes: list[_Hash]


@dataclass(frozen=True)
class _Reference:
    type: str
    toolchain_version: str
    files: list[_ReferenceFile]
    source_repository: _Repository | None = None
    repository: _Repository | None = None


@dataclass(frozen=True)
class _Accessibility:
    type: str
    reference: _Reference | None = None


@dataclass(frozen=True)
class _MiraiModel:
    id: str
    accessibility: _Accessibility


_CONVERTER = cattrs.Converter()


@dataclass(frozen=True)
class Device:
    os_name: str
    cpu_name: str | None
    memory_total: int | None = None


@dataclass(frozen=True)
class File:
    name: str
    url: str
    crc32c: str


@dataclass(frozen=True)
class Model:
    id: str
    toolchain_version: str
    repo_id: str
    name: str
    files: list[File]


@dataclass(frozen=True)
class Registry:
    models: list[Model]


class Role(StrEnum):
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
    messages: list[Message]
    greedy: bool


def parse_registry(payload: dict[str, Any]) -> Registry:
    raw_models = _CONVERTER.structure(payload["models"], list[_MiraiModel])
    models: list[Model] = []
    for raw in raw_models:
        if raw.accessibility.type != "local":
            continue
        reference = raw.accessibility.reference
        if reference is None or reference.type != "mirai":
            continue
        repository = reference.source_repository or reference.repository
        if repository is None:
            continue
        repo_id = repository.identifier
        files = [
            File(name=raw_file.name, url=raw_file.url, crc32c=_find_crc32c(raw_file)) for raw_file in reference.files
        ]
        models.append(
            Model(
                id=raw.id,
                toolchain_version=reference.toolchain_version,
                repo_id=repo_id,
                name=repo_id.split("/")[-1],
                files=files,
            ),
        )
    return Registry(models=models)


def _find_crc32c(raw_file: _ReferenceFile) -> str:
    crc32c = next((h.value for h in raw_file.hashes if h.method == "crc32c"), None)
    if crc32c is None:
        raise ValueError(f"File {raw_file.name!r} is missing a crc32c hash")
    return crc32c
