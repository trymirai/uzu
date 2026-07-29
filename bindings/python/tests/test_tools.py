# ruff: noqa: SLF001

import asyncio
import json
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from uzu import ChatSession, UzuToolFunction, uzu_tool_function


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
        result = add_node_values._invoke_json(
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
        assert not isinstance(result, str)
        assert json.loads(await result) == 6

    asyncio.run(run())

    parameters = add_node_values.parameters_schema
    assert parameters is not None
    root_property = parameters["properties"]["root"]
    root = _resolve_reference(parameters, root_property)
    assert root["properties"]["children"]["items"]["$ref"] == "#/$defs/Node"


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
