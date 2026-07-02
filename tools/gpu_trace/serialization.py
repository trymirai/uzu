from __future__ import annotations

import json
from enum import Enum
from typing import Any

import cattrs

from .models import (
    CounterType,
    PerformanceState,
    ShaderType,
    TraceExport,
)


def _make_converter() -> cattrs.Converter:
    converter = cattrs.Converter()

    def enum_unstructure(val: Enum) -> str:
        return val.value

    def enum_structure(val: str, cls: type[Enum]) -> Enum:
        return cls(val)

    for enum_cls in (CounterType, PerformanceState, ShaderType):
        converter.register_unstructure_hook(enum_cls, enum_unstructure)
        converter.register_structure_hook(enum_cls, enum_structure)

    def tuple_unstructure(val: tuple[Any, ...]) -> list[Any]:
        return [converter.unstructure(item) for item in val]

    converter.register_unstructure_hook(tuple, tuple_unstructure)

    return converter


_converter = _make_converter()


def to_dict(
    export: TraceExport,
    include_counter_samples: bool = False,
) -> dict[str, Any]:
    data = _converter.unstructure(export)
    if not include_counter_samples:
        data.pop("counter_samples", None)
    return data


def to_json(
    export: TraceExport,
    indent: int | None = 2,
    include_counter_samples: bool = False,
) -> str:
    return json.dumps(to_dict(export, include_counter_samples), indent=indent)


def from_dict(data: dict[str, Any]) -> TraceExport:
    return _converter.structure(data, TraceExport)


def from_json(json_str: str) -> TraceExport:
    return from_dict(json.loads(json_str))
