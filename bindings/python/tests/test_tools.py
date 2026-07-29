# ruff: noqa: SLF001

import asyncio
import contextvars
import json
import threading
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from uzu import ChatSession, UzuToolFunction, uzu_tool_function

_TOOL_CONTEXT = contextvars.ContextVar[str]("tool_context")


class Coordinate(BaseModel):
    """A geographic coordinate."""

    latitude: Annotated[float, "Latitude in decimal degrees."]
    longitude: float = Field(description="Longitude in decimal degrees.")


class ForecastRequest(BaseModel):
    """A weather forecast request."""

    location: Annotated[Coordinate, "Location to forecast."]
    days: int = Field(description="Number of days ahead.")


class Node(BaseModel):
    """A recursive tree node."""

    value: Annotated[int, "Value stored in this node."]
    children: list["Node"] = Field(default_factory=list)


class Weather(BaseModel):
    """The current weather."""

    temperature: float = Field(description="Temperature in degrees Celsius.")
    summary: Annotated[str | None, "Optional human-readable summary."]


class AliasedResult(BaseModel):
    value: int = Field(serialization_alias="answer")


@uzu_tool_function
def get_forecast(
    request: Annotated[ForecastRequest, "Request supplied to the forecast service."],
    note: Annotated[str | None, "Required nullable note."],
    unit: Annotated[Literal["celsius", "fahrenheit"], "Temperature unit."] = "celsius",
) -> Weather:
    """Get the weather forecast for a location."""
    assert isinstance(request, ForecastRequest)
    assert isinstance(request.location, Coordinate)
    return Weather(
        temperature=request.location.latitude + request.location.longitude,
        summary=None if note is None else f"{note} ({unit})",
    )


@uzu_tool_function(name="sum_values", description="Add all node values.")
async def add_node_values(root: Node) -> int:
    await asyncio.sleep(0)
    return root.value + sum(child.value for child in root.children)


@uzu_tool_function
def current_runtime_state() -> dict[str, int | str]:
    return {
        "context": _TOOL_CONTEXT.get(),
        "loop": id(asyncio.get_running_loop()),
        "thread": threading.get_ident(),
    }


@uzu_tool_function
def reserved_parameter_names(
    model_config: Annotated[int, "Configuration number."],
    model_dump: Annotated[str, "Dump label."],
) -> dict[str, int | str]:
    return {
        "model_config": model_config,
        "model_dump": model_dump,
    }


@uzu_tool_function
def unhashable_return_metadata() -> Annotated[int, {"tag": "x"}]:
    return 42


@uzu_tool_function
def get_aliased_result() -> AliasedResult:
    return AliasedResult(value=1)


def _resolve_reference(
    root: dict[str, object],
    schema: dict[str, object],
) -> dict[str, object]:
    reference = schema.get("$ref")
    if not isinstance(reference, str):
        return schema

    target: object = root
    for raw_token in reference.removeprefix("#/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        assert isinstance(target, dict)
        target = target[token]
    assert isinstance(target, dict)
    return target


def test_decorator_builds_schema_from_function_and_pydantic_metadata() -> None:
    assert isinstance(get_forecast, UzuToolFunction)
    assert get_forecast.name == "get_forecast"
    assert get_forecast.description == "Get the weather forecast for a location."

    parameters = get_forecast.parameters_schema
    assert parameters is not None
    assert parameters["required"] == ["request", "note"]

    properties = parameters["properties"]
    request_property = properties["request"]
    assert request_property["description"] == "Request supplied to the forecast service."
    request = _resolve_reference(parameters, request_property)
    location_property = request["properties"]["location"]
    assert location_property["description"] == "Location to forecast."
    location = _resolve_reference(parameters, location_property)
    assert location["properties"]["latitude"]["description"] == "Latitude in decimal degrees."
    assert location["properties"]["longitude"]["description"] == "Longitude in decimal degrees."
    assert request["properties"]["days"]["description"] == "Number of days ahead."

    assert properties["note"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert properties["note"]["description"] == "Required nullable note."
    assert properties["unit"]["description"] == "Temperature unit."
    assert properties["unit"]["default"] == "celsius"


def test_pydantic_return_schema_uses_docstrings_and_field_metadata() -> None:
    schema = get_forecast.return_schema
    assert schema is not None
    assert schema["description"] == "The current weather."
    assert schema["properties"]["temperature"]["description"] == "Temperature in degrees Celsius."
    assert schema["properties"]["summary"]["description"] == "Optional human-readable summary."
    assert schema["required"] == ["temperature", "summary"]


def test_sync_tool_constructs_and_serializes_pydantic_models() -> None:
    result = get_forecast._invoke_json(
        json.dumps(
            {
                "request": {
                    "location": {
                        "latitude": 51.5,
                        "longitude": -0.1,
                    },
                    "days": 2,
                },
                "note": None,
            }
        )
    )

    assert isinstance(result, str)
    assert json.loads(result) == {
        "temperature": 51.4,
        "summary": None,
    }


def test_return_serialization_uses_schema_aliases() -> None:
    schema = get_aliased_result.return_schema
    assert schema is not None
    assert set(schema["properties"]) == {"answer"}
    assert schema["required"] == ["answer"]

    result = get_aliased_result._invoke_json("{}")
    assert isinstance(result, str)
    assert json.loads(result) == {"answer": 1}


def test_required_nullable_parameter_cannot_be_omitted() -> None:
    with pytest.raises(ValidationError, match="note"):
        get_forecast._invoke_json(
            json.dumps(
                {
                    "request": {
                        "location": {
                            "latitude": 51.5,
                            "longitude": -0.1,
                        },
                        "days": 2,
                    }
                }
            )
        )


def test_async_tool_constructs_recursive_pydantic_model() -> None:
    async def run() -> None:
        result = await add_node_values._invoke_json_on_loop(
            json.dumps(
                {
                    "root": {
                        "value": 1,
                        "children": [
                            {
                                "value": 2,
                                "children": [],
                            },
                            {
                                "value": 3,
                            },
                        ],
                    }
                }
            )
        )
        assert json.loads(result) == 6

    asyncio.run(run())

    parameters = add_node_values.parameters_schema
    assert parameters is not None
    root_property = parameters["properties"]["root"]
    root = _resolve_reference(parameters, root_property)
    assert root["properties"]["children"]["items"]["$ref"] == "#/$defs/Node"


def test_sync_tool_trampoline_runs_on_python_loop_and_context() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        token = _TOOL_CONTEXT.set("reply-context")
        try:
            worker_thread, invocation = await asyncio.to_thread(
                lambda: (
                    threading.get_ident(),
                    current_runtime_state._invoke_json_on_loop("{}"),
                )
            )
            assert worker_thread != loop_thread
            assert json.loads(await invocation) == {
                "context": "reply-context",
                "loop": id(loop),
                "thread": loop_thread,
            }
        finally:
            _TOOL_CONTEXT.reset(token)

    asyncio.run(run())


def test_parameters_can_use_reserved_base_model_names() -> None:
    parameters = reserved_parameter_names.parameters_schema
    assert parameters is not None
    assert parameters["required"] == ["model_config", "model_dump"]
    assert set(parameters["properties"]) == {"model_config", "model_dump"}
    assert parameters["properties"]["model_config"]["description"] == "Configuration number."
    assert parameters["properties"]["model_dump"]["description"] == "Dump label."

    result = reserved_parameter_names._invoke_json('{"model_config":7,"model_dump":"ready"}')
    assert isinstance(result, str)
    assert json.loads(result) == {
        "model_config": 7,
        "model_dump": "ready",
    }


def test_return_annotation_can_have_unhashable_metadata() -> None:
    assert unhashable_return_metadata.return_schema is not None
    assert unhashable_return_metadata.return_schema["type"] == "integer"
    assert unhashable_return_metadata._invoke_json("{}") == "42"


def test_wrapper_metadata_does_not_replace_tool_state() -> None:
    def target(value: int) -> int:
        return value

    target.__dict__.update(
        {
            "_function": lambda value: value + 100,
            "description": "Wrong description.",
            "name": "wrong_name",
            "parameters_schema": {"wrong": True},
        }
    )
    tool = uzu_tool_function(target, name="configured_name", description="Configured description.")

    assert tool.name == "configured_name"
    assert tool.description == "Configured description."
    assert tool.parameters_schema is not None
    assert set(tool.parameters_schema["properties"]) == {"value"}
    assert tool._invoke_json('{"value":5}') == "5"


def test_cancelling_invocation_cancels_async_python_tool() -> None:
    async def run() -> None:
        started = asyncio.Event()
        cleaned_up = asyncio.Event()
        side_effect = False

        @uzu_tool_function
        async def waiting_tool() -> None:
            nonlocal side_effect
            started.set()
            try:
                await asyncio.Event().wait()
                side_effect = True
            finally:
                cleaned_up.set()

        invocation = waiting_tool._new_json_invocation("{}")
        task = asyncio.create_task(invocation.run())
        await started.wait()
        await asyncio.to_thread(invocation.cancel)

        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cleaned_up.wait(), timeout=1)
        assert not side_effect

    asyncio.run(run())


def test_configured_decorator_overrides_name_and_description() -> None:
    assert add_node_values.name == "sum_values"
    assert add_node_values.description == "Add all node values."


def test_chat_session_exposes_tool_registration_methods() -> None:
    assert callable(ChatSession.add_tool)
    assert callable(ChatSession.add_tools)


def test_native_definition_matches_decorator_metadata() -> None:
    definition = get_forecast._native_definition()
    assert definition.name == get_forecast.name
    assert definition.description == get_forecast.description
    assert json.loads(definition.parameters.json) == get_forecast.parameters_schema
    assert json.loads(definition.return_definition.json) == get_forecast.return_schema
