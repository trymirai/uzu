import asyncio

from uzu import ClassificationMessage, Engine, EngineConfig


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("trymirai/chat-moderation-router")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"\rDownload progress: {update.progress:.2%}", end="", flush=True)
    print()

    messages = [ClassificationMessage.user("Hi")]

    session = await engine.classification(model)
    output = await session.classify(messages)
    print(f"Output: {output.probabilities.values}")


if __name__ == "__main__":
    asyncio.run(main())
