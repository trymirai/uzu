"""Compare model declarations without building the Rust, C++, or Zig engines.

Python's unbounded integers, float precision, and existing ChatRole enum are
normalized to their native counterparts. Rust/C++ numeric widths are also
compared directly. This checks declarations, not the full behavior of serde,
Glaze, or Pydantic.
"""

import inspect
import re
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union, get_args, get_origin

import bench
import pytest
from pydantic import BaseModel, JsonValue

ROOT = Path(__file__).resolve().parents[1]

CHAT_ROLES = {"assistant", "developer", "system", "tool", "user"}

CONTAINERS = {
    "Option": "optional",
    "std::optional": "optional",
    "Vec": "list",
    "std::vector": "list",
}

NATIVE_TYPES = {
    "String": "string",
    "std::string": "string",
    "i32": "int32",
    "int32_t": "int32",
    "usize": "size",
    "size_t": "size",
    "u64": "uint64",
    "uint64_t": "uint64",
    "f32": "float32",
    "float": "float32",
    "f64": "float64",
    "double": "float64",
    "[]constu8": "string",
    "Value": "json",
    "glz::generic_u64": "json",
    "std.json.Value": "json",
}

OMIT_WHEN_NONE = {
    "ChatMessage.tool_calls": [],
    "ChatMessage.tool_call_id": "call-1",
    "BenchRequest.tools": [],
    "BenchRequest.tool_choice": "auto",
}

PYTHON_MODELS = {
    name: model
    for name, model in vars(bench).items()
    if inspect.isclass(model) and issubclass(model, BaseModel) and model.__module__ == bench.__name__
}


class Default(Enum):
    REQUIRED = "required"


@dataclass(frozen=True)
class Field:
    type: str
    default: object


def native_type(annotation: str, model_names: set[str]) -> str:
    annotation = re.sub(r"\s+", "", annotation)
    if annotation.startswith("?"):
        return f"optional<{native_type(annotation[1:], model_names)}>"
    if annotation.startswith("[]const") and annotation != "[]constu8":
        return f"list<{native_type(annotation[len('[]const') :], model_names)}>"
    if match := re.fullmatch(r"([\w:]+)<(.+)>", annotation):
        container, inner = match.groups()
        assert container in CONTAINERS, f"Unsupported native container: {annotation}"
        return f"{CONTAINERS[container]}<{native_type(inner, model_names)}>"

    if annotation in model_names:
        return annotation

    assert annotation in NATIVE_TYPES, f"Unsupported native type: {annotation}"
    return NATIVE_TYPES[annotation]


def read_native_models(path: Path, language: str) -> dict[str, dict[str, Field]]:
    source = path.read_text()
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
    if language == "rust":
        source = re.sub(r"#\[derive\([^)]*\)\]", "", source)
        source = re.sub(r"\buse\s+[^;]+;", "", source)
        struct_pattern = r"\bpub\s+struct\s+(\w+)\s*\{([^{}]*)\}"
        field_pattern = r"\s*pub\s+(?P<name>\w+)\s*:\s*(?P<type>[\w\s:<>]+)\s*,"
    elif language == "zig":
        source = re.sub(r'const std = @import\("std"\);', "", source)
        struct_pattern = r"\bpub\s+const\s+(\w+)\s*=\s*struct\s*\{([^{}]*)\}\s*;"
        field_pattern = r"\s*(?P<name>\w+)\s*:\s*(?P<type>[\w\s.:?\[\]]+?)(?:\s*=\s*(?P<default>[^,]+))?\s*,"
    else:
        source = re.sub(r"^\s*#(?:include|ifndef|define|endif)[^\n]*", "", source, flags=re.MULTILINE)
        struct_pattern = r"\bstruct\s+(\w+)\s*\{([^{}]*)\}\s*;"
        field_pattern = r"\s*(?P<type>[\w\s:<>]+?)\s+(?P<name>\w+)(?:\s*=\s*(?P<default>[^;]+))?\s*;"

    structs = list(re.finditer(struct_pattern, source))
    # Fail on unrecognized syntax, including serialization attributes/metadata,
    # rather than overlooking a field or a change to its JSON representation.
    remainder = re.sub(struct_pattern, "", source).strip()
    assert not remainder, f"{path.relative_to(ROOT)}: unsupported model syntax: {remainder}"

    names = {struct[1] for struct in structs}
    assert len(names) == len(structs), f"{path}: duplicate model declarations"

    models = {}
    for struct in structs:
        name, body = struct.groups()
        fields = {}
        position = 0
        while body[position:].strip():
            match = re.compile(field_pattern).match(body, position)
            assert match, f"{path.relative_to(ROOT)}: {name}: unsupported field syntax: {body[position:]}"
            field_name = match["name"]
            field_type = native_type(match["type"], names)
            default = None if field_type.startswith("optional<") else Default.REQUIRED
            if language == "zig" and default is None:
                assert match["default"] is not None, f"{name}.{field_name}: optional Zig fields must default to null"
            if language in ("cpp", "zig") and match["default"] is not None:
                initializer = match["default"].strip()
                expected = "null" if language == "zig" else "std::nullopt"
                assert initializer == expected and default is None, (
                    f"{path.relative_to(ROOT)}: {name}.{field_name}: unexpected default {initializer}"
                )
            assert field_name not in fields, f"{path}: {name}.{field_name}: duplicate field"
            fields[field_name] = Field(field_type, default)
            position = match.end()

        models[name] = fields

    return models


def python_type(annotation: object, field_path: str) -> str:
    if annotation is JsonValue:
        return "json"
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (types.UnionType, Union):
        assert len(args) == 2 and type(None) in args, f"{field_path}: unsupported union {annotation}"
        inner = next(arg for arg in args if arg is not type(None))
        return f"optional<{python_type(inner, field_path)}>"

    if origin is list:
        return f"list<{python_type(args[0], field_path)}>"

    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation.__name__

    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        assert field_path == "ChatMessage.role" and issubclass(annotation, str), (
            f"{field_path}: enum constraint has no Rust/C++ counterpart"
        )
        assert {member.value for member in annotation} == CHAT_ROLES, "ChatMessage.role: enum constraints changed"
        return "string"

    primitives: dict[object, str] = {str: "string", int: "integer", float: "number"}
    assert annotation in primitives, f"{field_path}: unsupported Python type {annotation}"
    return primitives[annotation]


def read_python_models() -> dict[str, dict[str, Field]]:
    models = {}
    omissions = set()
    for name, model in PYTHON_MODELS.items():
        assert not model.model_config, f"{name}: review model configuration for parity"
        assert not model.model_computed_fields, f"{name}: computed fields have no Rust/C++ counterpart"
        fields = {}
        for field_name, field in model.model_fields.items():
            path = f"{name}.{field_name}"
            assert not field.metadata, f"{path}: constraints have no Rust/C++ counterpart: {field.metadata}"
            assert field.alias is field.validation_alias is field.serialization_alias is None, (
                f"{path}: alias has no Rust/C++ counterpart"
            )
            assert not field.exclude, f"{path}: excluded field has no Rust/C++ counterpart"
            assert field.default_factory is None, f"{path}: default factory has no Rust/C++ counterpart"
            if field.exclude_if is not None:
                assert path in OMIT_WHEN_NONE, f"{path}: conditional omission has no Rust/C++ counterpart"
                assert field.exclude_if(None) is True and field.exclude_if(OMIT_WHEN_NONE[path]) is False, (
                    f"{path}: omission behavior changed"
                )
                omissions.add(path)
            fields[field_name] = Field(
                python_type(field.annotation, path), Default.REQUIRED if field.is_required() else field.default
            )
        models[name] = fields

    assert omissions == OMIT_WHEN_NONE.keys(), "Python's known null-omission differences changed"
    return models


def portable_fields(fields: dict[str, Field]) -> dict[str, Field]:
    """Python numbers do not express native integer ranges or float precision."""
    numbers = {"int32": "integer", "uint64": "integer", "size": "integer", "float32": "number", "float64": "number"}
    return {
        name: Field(
            re.sub(r"\b(?:int32|uint64|size|float32|float64)\b", lambda m: numbers[m[0]], field.type), field.default
        )
        for name, field in fields.items()
    }


@pytest.fixture(scope="module")
def models() -> dict[str, dict[str, dict[str, Field]]]:
    return {
        "python": read_python_models(),
        "rust": read_native_models(ROOT / "engine-uzu/src/bench.rs", "rust"),
        "cpp": read_native_models(ROOT / "common-cpp/src/bench.hpp", "cpp"),
        "zig": read_native_models(ROOT / "engine-mlxserve/src/bench.zig", "zig"),
    }


def test_model_names_match(models):
    assert {"BenchRequest", "BenchResponse", "ChatMessage", "BenchSampling"} <= models["python"].keys()
    assert models["rust"].keys() == models["cpp"].keys() == models["python"].keys() == models["zig"].keys()


def test_native_models_match(models):
    """Also catch Rust/C++ signedness and numeric-width differences."""
    assert models["rust"] == models["cpp"], "engine-uzu/src/bench.rs and common-cpp/src/bench.hpp differ"
    assert models["rust"] == models["zig"], "engine-uzu/src/bench.rs and engine-mlxserve/src/bench.zig differ"


def test_python_models_match_rust(models):
    assert models["python"] == {name: portable_fields(fields) for name, fields in models["rust"].items()}, (
        "common-py/src/bench.py and engine-uzu/src/bench.rs fields, types, or defaults differ"
    )


def test_python_models_match_cpp(models):
    assert models["python"] == {name: portable_fields(fields) for name, fields in models["cpp"].items()}, (
        "common-py/src/bench.py and common-cpp/src/bench.hpp fields, types, or defaults differ"
    )
