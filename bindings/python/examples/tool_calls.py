import asyncio
from typing import Annotated

from pydantic import BaseModel

from uzu import (
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    Engine,
    EngineConfig,
    SamplingMethod,
    SamplingPolicy,
    uzu_tool_function,
)


class Coordinate(BaseModel):
    """A geographic coordinate.

    Attributes:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
    """

    latitude: float
    longitude: Annotated[float, "Longitude in decimal degrees."]


@uzu_tool_function(name="get_location", description="Return the current location in coordinates")
def get_current_location() -> Coordinate:
    return Coordinate(latitude=51.5074, longitude=-0.1278)


@uzu_tool_function
def get_current_temperature(
    latitude: float,
    longitude: Annotated[float, "Longitude in decimal degrees."],
) -> float:
    """Return the temperature at the provided coordinates.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: This is overridden by the Annotated description.
    """
    _ = latitude, longitude
    return 25.0


async def main() -> None:
    engine = await Engine.create(EngineConfig.create())
    model = await engine.model("mlx-community/Qwen3.5-9B-MLX-8bit")
    if model is None:
        raise RuntimeError("Model not found")

    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    session = await engine.chat(model, ChatConfig.create())
    await session.add_tool(get_current_location)
    await session.add_tool(get_current_temperature)

    messages = [
        ChatMessage.system().with_text("You are a helpful assistant"),
        ChatMessage.user().with_text("What temperature is it now at my location?"),
    ]
    config = ChatReplyConfig.create().with_sampling_policy(SamplingPolicy.Custom(method=SamplingMethod.Greedy()))
    replies = await session.reply(messages, config)
    if replies:
        message = replies[-1].message
        print(f"Reasoning: {message.reasoning or ''}")
        print(f"Text: {message.text or ''}")


if __name__ == "__main__":
    asyncio.run(main())
