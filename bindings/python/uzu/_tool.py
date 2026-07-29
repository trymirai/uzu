from __future__ import annotations

import asyncio
import collections.abc
import functools
import inspect
import json
import types
import typing
from typing import Annotated, Any, Literal, overload

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model

_EMPTY = inspect.Parameter.empty
_NONE_TYPE = type(None)


class UzuToolFunction[**P, R]:
    """A Python function and its Uzu tool definition."""

    def __init__(
        self,
        function: collections.abc.Callable[P, R],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if not callable(function):
            raise TypeError("tool must be callable")

        inferred_name = getattr(function, "__name__", None)
        if name is None:
            if not inferred_name or inferred_name == "<lambda>":
                raise TypeError("a tool name is required for lambdas and unnamed callables")
            name = inferred_name

        self._function = function
        self.name = name
        self.description = description if description is not None else inspect.getdoc(function) or ""
        self._signature = inspect.signature(function)

        try:
            annotations = typing.get_type_hints(function, include_extras=True)
        except (NameError, TypeError) as error:
            raise TypeError(f"unable to resolve annotations for tool {name!r}: {error}") from error

        fields: dict[str, tuple[object, object]] = {}
        self._argument_fields: dict[str, str] = {}
        for parameter in self._signature.parameters.values():
            if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL}:
                raise TypeError(f"tool {name!r} cannot use positional-only or variadic positional parameters")
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                raise TypeError(f"tool {name!r} cannot use variadic keyword parameters")

            annotation = annotations.get(parameter.name, parameter.annotation)
            if annotation is _EMPTY:
                raise TypeError(f"parameter {parameter.name!r} of tool {name!r} must have a type annotation")

            field_metadata = (
                Field(alias=parameter.name, description=annotation_description)
                if (annotation_description := _annotation_description(annotation))
                else Field(alias=parameter.name)
            )
            annotation = Annotated[annotation, field_metadata]
            default = ... if parameter.default is _EMPTY else parameter.default
            internal_name = f"uzu_argument_{len(fields)}"
            while hasattr(BaseModel, internal_name):
                internal_name += "_"
            fields[internal_name] = (annotation, default)
            self._argument_fields[parameter.name] = internal_name

        self._arguments_model: type[BaseModel] | None = None
        self.parameters_schema: dict[str, object] | None = None
        if fields:
            self._arguments_model = create_model(
                f"{name}_arguments",
                __config__=ConfigDict(extra="forbid"),
                **fields,
            )
            self.parameters_schema = self._arguments_model.model_json_schema(mode="validation")
            _apply_annotated_descriptions(
                self._arguments_model,
                self.parameters_schema,
                self.parameters_schema,
                mode="validation",
                visited=set(),
            )

        return_annotation = annotations.get("return", self._signature.return_annotation)
        self._return_adapter = TypeAdapter(Any if return_annotation is _EMPTY else return_annotation)
        self.return_schema: dict[str, object] | None = None
        if return_annotation is not _EMPTY and return_annotation is not None and return_annotation is not _NONE_TYPE:
            self.return_schema = self._return_adapter.json_schema(mode="serialization")
            _apply_annotated_descriptions(
                return_annotation,
                self.return_schema,
                self.return_schema,
                mode="serialization",
                visited=set(),
            )

        functools.update_wrapper(self, function, updated=())

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        return self._function(*args, **kwargs)

    def _native_definition(self) -> object:
        # Imported lazily because the extension imports this module while initializing.
        from . import ToolFunction, Value

        parameters = None if self.parameters_schema is None else Value(json.dumps(self.parameters_schema))
        return_definition = None if self.return_schema is None else Value(json.dumps(self.return_schema))
        return ToolFunction(
            name=self.name,
            description=self.description,
            parameters=parameters,
            return_definition=return_definition,
        )

    def _invoke_json(self, arguments_json: str) -> str | collections.abc.Awaitable[str]:
        if self._arguments_model is None:
            arguments = json.loads(arguments_json)
            if arguments not in ({}, None):
                raise TypeError(f"tool {self.name!r} does not accept arguments")
            kwargs: dict[str, object] = {}
        else:
            arguments = self._arguments_model.model_validate_json(arguments_json)
            kwargs = {
                parameter_name: getattr(arguments, internal_name)
                for parameter_name, internal_name in self._argument_fields.items()
            }

        result = self._function(**kwargs)
        if inspect.isawaitable(result):

            async def finish() -> str:
                return self._encode_result(await result)

            return finish()
        return self._encode_result(result)

    async def _invoke_json_on_loop(self, arguments_json: str) -> str:
        result = self._invoke_json(arguments_json)
        return await result if inspect.isawaitable(result) else result

    def _new_json_invocation(self, arguments_json: str) -> _ToolInvocation:
        return _ToolInvocation(self, arguments_json)

    def _encode_result(self, result: object) -> str:
        validated = self._return_adapter.validate_python(result)
        return self._return_adapter.dump_json(validated).decode()


class _ToolInvocation:
    def __init__(
        self,
        tool: UzuToolFunction,
        arguments_json: str,
    ) -> None:
        self._tool = tool
        self._arguments_json = arguments_json
        self._cancelled = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[object] | None = None

    async def run(self) -> str:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        if self._cancelled:
            raise asyncio.CancelledError
        return await self._tool._invoke_json_on_loop(self._arguments_json)  # noqa: SLF001

    def cancel(self) -> None:
        self._cancelled = True
        if self._loop is not None and self._task is not None:
            self._loop.call_soon_threadsafe(self._task.cancel)


@overload
def uzu_tool_function[**P, R](
    function: collections.abc.Callable[P, R],
    *,
    name: str | None = None,
    description: str | None = None,
) -> UzuToolFunction[P, R]: ...


@overload
def uzu_tool_function[**P, R](
    function: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> collections.abc.Callable[[collections.abc.Callable[P, R]], UzuToolFunction[P, R]]: ...


def uzu_tool_function[**P, R](
    function: collections.abc.Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> UzuToolFunction[P, R] | collections.abc.Callable[[collections.abc.Callable[P, R]], UzuToolFunction[P, R]]:
    """Create an Uzu tool from an annotated Python function."""

    def decorate(target: collections.abc.Callable[P, R]) -> UzuToolFunction[P, R]:
        return UzuToolFunction(target, name=name, description=description)

    return decorate if function is None else decorate(function)


def _annotation_description(annotation: object) -> str | None:
    if typing.get_origin(annotation) is not typing.Annotated:
        return None

    description: str | None = None
    for metadata in typing.get_args(annotation)[1:]:
        candidate = metadata if isinstance(metadata, str) else getattr(metadata, "description", None)
        if candidate:
            description = str(candidate)
    return description


def _apply_annotated_descriptions(  # noqa: PLR0911
    annotation: object,
    schema: dict[str, object],
    root_schema: dict[str, object],
    *,
    mode: Literal["validation", "serialization"],
    visited: set[tuple[object, int]],
) -> None:
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)

    if origin is typing.Annotated:
        if description := _annotation_description(annotation):
            schema["description"] = description
        _apply_annotated_descriptions(
            arguments[0],
            schema,
            root_schema,
            mode=mode,
            visited=visited,
        )
        return

    if origin in {typing.Union, types.UnionType}:
        branches = schema.get("anyOf")
        if isinstance(branches, list):
            for member, branch in zip(arguments, branches, strict=False):
                if isinstance(branch, dict):
                    _apply_annotated_descriptions(
                        member,
                        branch,
                        root_schema,
                        mode=mode,
                        visited=visited,
                    )
        return

    if origin in {list, set, frozenset, collections.abc.Sequence}:
        items = schema.get("items")
        if arguments and isinstance(items, dict):
            _apply_annotated_descriptions(
                arguments[0],
                items,
                root_schema,
                mode=mode,
                visited=visited,
            )
        return

    if origin is tuple:
        items = schema.get("items")
        if isinstance(items, dict) and arguments:
            _apply_annotated_descriptions(
                arguments[0],
                items,
                root_schema,
                mode=mode,
                visited=visited,
            )
        prefix_items = schema.get("prefixItems")
        if isinstance(prefix_items, list):
            for member, item in zip(arguments, prefix_items, strict=False):
                if isinstance(item, dict):
                    _apply_annotated_descriptions(
                        member,
                        item,
                        root_schema,
                        mode=mode,
                        visited=visited,
                    )
        return

    if origin in {dict, collections.abc.Mapping}:
        values = schema.get("additionalProperties")
        if len(arguments) == 2 and isinstance(values, dict):
            _apply_annotated_descriptions(
                arguments[1],
                values,
                root_schema,
                mode=mode,
                visited=visited,
            )
        return

    if not inspect.isclass(annotation) or not issubclass(annotation, BaseModel):
        return

    model_schema = _resolve_local_reference(schema, root_schema)
    marker = (annotation, id(model_schema))
    if marker in visited:
        return
    visited.add(marker)

    properties = model_schema.get("properties")
    if not isinstance(properties, dict):
        return

    try:
        annotations = typing.get_type_hints(annotation, include_extras=True)
    except (NameError, TypeError):
        annotations = {name: field.annotation for name, field in annotation.model_fields.items()}

    for name, field in annotation.model_fields.items():
        field_annotation = annotations.get(name, field.annotation)
        alias = field.serialization_alias if mode == "serialization" else field.validation_alias
        field_schema = properties.get(alias if isinstance(alias, str) else name)
        if isinstance(field_schema, dict):
            _apply_annotated_descriptions(
                field_annotation,
                field_schema,
                root_schema,
                mode=mode,
                visited=visited,
            )


def _resolve_local_reference(
    schema: dict[str, object],
    root_schema: dict[str, object],
) -> dict[str, object]:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return schema

    target: object = root_schema
    for raw_token in reference[2:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or token not in target:
            return schema
        target = target[token]
    return target if isinstance(target, dict) else schema
