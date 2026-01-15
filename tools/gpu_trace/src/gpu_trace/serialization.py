from __future__ import annotations

import json
from enum import Enum
from typing import Any

import cattrs
from cattrs.strategies import configure_tagged_union

from .models import (
    CounterType,
    GpuCounterDefinition,
    GpuPerformanceStateInterval,
    GpuUtilization,
    KernelDispatch,
    KernelStats,
    PerformanceState,
    ShaderInfo,
    ShaderType,
    TraceExport,
    TraceMetadata,
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


def to_dict(export: TraceExport) -> dict[str, Any]:
    return _converter.unstructure(export)


def to_json(export: TraceExport, indent: int | None = 2) -> str:
    return json.dumps(to_dict(export), indent=indent)


def from_dict(data: dict[str, Any]) -> TraceExport:
    return _converter.structure(data, TraceExport)


def from_json(json_str: str) -> TraceExport:
    return from_dict(json.loads(json_str))
